import argparse
import duckdb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import os
import time
from sklearn.preprocessing import StandardScaler


##############################################################################
# Model Definitions (copied from your original script)
##############################################################################

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob, num_layers=1):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size, hidden_size,
            num_layers=num_layers,
            dropout=dropout_prob if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.ln(out)
        out = self.dropout(out)
        out = self.fc(out)   
        out = self.activation(out)
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            dropout=dropout_prob if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)  
        out = self.ln(out)
        out = self.dropout(out)
        out = self.fc(out) 
        out = self.activation(out)
        return out


##############################################################################
# Helper Functions
##############################################################################

def fetch_rolling_data(db_path, table_name, window_size, param_index=None):
    """
    Fetch data for a specific parameter_index with rolling window aggregation
    If param_index is None, returns a list of available parameter indices
    """
    print(f"[INFO] Connecting to DuckDB at {db_path}")
    start_time = time.time()

    con = duckdb.connect(db_path, read_only=True)
    con.execute("PRAGMA threads=8;")

    if param_index is None:
        # Just fetch available parameter indices and their global indices
        query = f"SELECT DISTINCT parameter_index, global_index FROM {table_name} ORDER BY parameter_index"
        result = con.execute(query).fetchall()
        con.close()
        return [(r[0], r[1]) for r in result]

    # When we have a specific parameter index, fetch all data for it
    param_where_clause = f"WHERE parameter_index = {param_index}"

    distinct_sims_subquery = f"""
        SELECT DISTINCT parameter_index, simulation_index, global_index
        FROM {table_name}
        {param_where_clause}
    """

    random_sims_subquery = distinct_sims_subquery

    cte_subquery = f"""
        SELECT
            t.parameter_index,
            t.simulation_index,
            t.global_index,
            t.timesteps,
            CASE WHEN t.n_age_0_1825 = 0 THEN NULL
                 ELSE CAST(t.n_detect_lm_0_1825 AS DOUBLE) / t.n_age_0_1825
            END AS raw_prevalence,
            t.eir,
            t.dn0_use,
            t.dn0_future,
            t.Q0,
            t.phi_bednets,
            t.seasonal,
            t.routine,
            t.itn_use,
            t.irs_use,
            t.itn_future,
            t.irs_future,
            t.lsm
        FROM {table_name} t
        JOIN ({random_sims_subquery}) rs
        USING (parameter_index, simulation_index)
    """

    preceding = window_size - 1
    last_6_years_day = 6 * 365

    final_query = f"""
        WITH cte AS (
            {cte_subquery}
        )
        SELECT
            parameter_index,
            simulation_index,
            global_index,
            ROW_NUMBER() OVER (
                PARTITION BY parameter_index, simulation_index
                ORDER BY timesteps
            ) AS timesteps,
            AVG(raw_prevalence) OVER (
                PARTITION BY parameter_index, simulation_index
                ORDER BY timesteps
                ROWS BETWEEN {preceding} PRECEDING AND CURRENT ROW
            ) AS prevalence,

            eir,
            dn0_use,
            dn0_future,
            Q0,
            phi_bednets,
            seasonal,
            routine,
            itn_use,
            irs_use,
            itn_future,
            irs_future,
            lsm
        FROM cte
        WHERE cte.timesteps >= {last_6_years_day}
          AND (cte.timesteps % {window_size}) = 0
        ORDER BY parameter_index, simulation_index, timesteps
    """

    df = con.execute(final_query).df()
    con.close()

    print(f"[INFO] Data fetched in {time.time() - start_time:.2f} seconds.")
    return df


def predict_full_sequence(model, full_ts, device):
    """Predict a full time series sequence"""
    model.eval()
    with torch.no_grad(), autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
        x_torch = torch.tensor(full_ts, dtype=torch.float32).unsqueeze(1).to(device)
        pred = model(x_torch).squeeze(-1).squeeze(-1).cpu().numpy()
    return pred


def load_model(model_path, input_size, hidden_size, output_size=1, dropout_prob=0.1, num_layers=1, model_type="gru"):
    """Load a trained model from checkpoint"""
    if model_type.lower() == "gru":
        model = GRUModel(input_size, hidden_size, output_size, dropout_prob, num_layers)
    else:
        model = LSTMModel(input_size, hidden_size, output_size, dropout_prob, num_layers)
    
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[INFO] Loaded {model_type.upper()} model from {model_path}")
    return model


def run_counterfactual(param_name, param_value, static_vals, static_covars, static_scaler, 
                       sim_df, t, use_cyclical_time, gru_model, lstm_model, device):
    """Run model with a counterfactual parameter value"""
    
    # Create a copy of the static values
    cf_static_vals = static_vals.copy()
    
    # Get the index of the parameter to modify
    param_idx = static_covars.index(param_name)
    
    # Store original value for reporting
    original_value = sim_df.iloc[0][param_name]
    
    # Create a copy of the original static values (before scaling)
    original_static_vals = sim_df.iloc[0][static_covars].values.astype(np.float32)
    
    # Modify the value at the specified index
    modified_static_vals = original_static_vals.copy()
    modified_static_vals[param_idx] = param_value
    
    # Normalize the modified static values
    cf_static_vals = static_scaler.transform(modified_static_vals.reshape(1, -1)).flatten()
    
    # Prepare inputs for model
    T = len(sim_df)
    
    if use_cyclical_time:
        day_of_year = t % 365.0
        sin_t = np.sin(2 * np.pi * day_of_year / 365.0)
        cos_t = np.cos(2 * np.pi * day_of_year / 365.0)
        X_full = np.zeros((T, 2 + len(static_covars)), dtype=np.float32)
        for j in range(T):
            X_full[j, 0] = sin_t[j]
            X_full[j, 1] = cos_t[j]
            X_full[j, 2:] = cf_static_vals
    else:
        # Normalize timesteps
        t_min, t_max = np.min(t), np.max(t)
        t_norm = (t - t_min) / (t_max - t_min) if t_max > t_min else t
        
        X_full = np.zeros((T, 1 + len(static_covars)), dtype=np.float32)
        for j in range(T):
            X_full[j, 0] = t_norm[j]
            X_full[j, 1:] = cf_static_vals
    
    # Get predictions
    y_gru = predict_full_sequence(gru_model, X_full, device)
    y_lstm = predict_full_sequence(lstm_model, X_full, device)
    
    return y_gru, y_lstm, original_value


##############################################################################
# Main Function
##############################################################################

def main():
    parser = argparse.ArgumentParser(description="Run trained emulator on a specific parameter")
    parser.add_argument("--db-path", default="/home/cosmo/net/malaria/Cosmo/segMINT/db3.duckdb", 
                       help="Path to DuckDB database file")
    parser.add_argument("--table-name", default="simulation_results", 
                       help="Table name inside DuckDB")
    parser.add_argument("--window-size", default=14, type=int, 
                       help="Window size for rolling average in days")
    parser.add_argument("--param-index", type=int, 
                       help="Specific parameter index to analyze. If not provided, a random one will be selected.")
    parser.add_argument("--models-dir", default="./resultsFINAL", 
                       help="Directory with trained models")
    parser.add_argument("--output-dir", default="./emulator_predictions", 
                       help="Directory to save prediction plots")
    parser.add_argument("--device", default=None, 
                       help="Choose device: 'cuda' or 'cpu'. If None, auto-detect.")
    parser.add_argument("--tight", action="store_true", 
                       help="If set, plot will use tight y-axis scaling focused on data range. If not set, y-axis will be fixed from 0 to 1.")
    parser.add_argument("--counterfactual", nargs='+', metavar=('PARAM_NAME', 'VALUES...'),
                       help="Run counterfactual analysis with specified parameter values. Format: PARAM_NAME VALUE1 VALUE2 ...")
    args = parser.parse_args()
    
    # Process counterfactual arguments
    cf_param_name = None
    cf_param_values = []
    
    if args.counterfactual and len(args.counterfactual) >= 2:
        cf_param_name = args.counterfactual[0]
        cf_param_values = [float(val) for val in args.counterfactual[1:]]
        print(f"[INFO] Counterfactual analysis: Varying {cf_param_name} with values {cf_param_values}")
    
    # Load the original training args to ensure we use the same model architecture
    import json
    try:
        with open(os.path.join(args.models_dir, "args.json"), 'r') as f:
            training_args = json.load(f)
            print("[INFO] Loaded original training parameters from args.json")
    except FileNotFoundError:
        print("[WARNING] Could not find args.json in models directory. Using default values.")
        # Set defaults based on your provided args.json
        training_args = {
            "hidden_size": 256,
            "num_layers": 4,
            "dropout": 0.05,
            "use_cyclical_time": True
        }

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device for model inference
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")
    
    # Get all parameter indices if not specified
    if args.param_index is None:
        all_params = fetch_rolling_data(args.db_path, args.table_name, args.window_size)
        if not all_params:
            print("[ERROR] No parameters found in the database!")
            return
        
        # Print some available parameters and their global indices
        print("[INFO] Available parameters (showing first 10):")
        for i, (param_idx, global_idx) in enumerate(all_params[:10]):
            print(f"  - Parameter Index: {param_idx}, Global Index: {global_idx} (RDS: simulation_results_{global_idx}.rds)")
        
        # Select a random parameter
        param_index, _ = random.choice(all_params)
        print(f"[INFO] Randomly selected parameter index: {param_index}")
    else:
        param_index = args.param_index
        print(f"[INFO] Using specified parameter index: {param_index}")
    
    # Fetch all simulations for the selected parameter
    df = fetch_rolling_data(args.db_path, args.table_name, args.window_size, param_index)
    
    if df.empty:
        print(f"[ERROR] No data found for parameter index {param_index}")
        return
    
    # Extract the global index for this parameter (should be consistent across all simulations)
    global_indices = df['global_index'].unique()
    if len(global_indices) > 1:
        print(f"[WARNING] Multiple global indices found for parameter {param_index}: {global_indices}")
        print(f"[INFO] Using the first global index: {global_indices[0]}")
        global_index = global_indices[0]
    else:
        global_index = global_indices[0]
    
    # Display the parameter and global indices
    print(f"\n[INFO] Parameter Information:")
    print(f"  - Parameter Index: {param_index}")
    print(f"  - Global Index: {global_index}")
    print(f"  - Corresponding RDS file: simulation_results_{global_index}.rds")
    
    print(f"[INFO] Fetched {len(df)} rows of data for parameter {param_index} (Global Index: {global_index})")
    
    # Group by simulation index
    sim_groups = df.groupby("simulation_index")
    num_sims = len(sim_groups)
    print(f"[INFO] Parameter {param_index} has {num_sims} simulations")
    
    # Define static covariates (same as in original script)
    static_covars = [
        "eir", "dn0_use", "dn0_future", "Q0", "phi_bednets",
        "seasonal", "routine", "itn_use", "irs_use",
        "itn_future", "irs_future", "lsm"
    ]
    
    # Validate counterfactual parameter
    if cf_param_name and cf_param_name not in static_covars:
        print(f"[ERROR] Counterfactual parameter '{cf_param_name}' not found in available parameters: {static_covars}")
        return
    
    # Print the parameter values from the first simulation (they should be the same across all sims for this parameter)
    first_sim_idx = list(sim_groups.groups.keys())[0]
    first_sim_df = sim_groups.get_group(first_sim_idx)
    param_values = first_sim_df.iloc[0][static_covars]
    
    print("\n[INFO] Input Parameter Values:")
    for param_name, param_value in param_values.items():
        print(f"  - {param_name}: {param_value}")
        
        # Print counterfactual values for the selected parameter
        if cf_param_name and param_name == cf_param_name:
            print(f"    * Counterfactual values: {cf_param_values}")
    
    # Load feature scaler
    scaler_path = os.path.join(args.models_dir, "static_scaler.pkl")
    static_scaler = pd.read_pickle(scaler_path)
    print(f"[INFO] Loaded feature scaler from {scaler_path}")
    
    # Load GRU and LSTM models
    gru_model_path = os.path.join(args.models_dir, "gru_best.pt")
    lstm_model_path = os.path.join(args.models_dir, "lstm_best.pt")
    
    # Use the same cyclical time setting as during training
    use_cyclical_time = training_args.get("use_cyclical_time", True)
    
    # Determine input size based on time encoding
    if use_cyclical_time:
        input_size = 2 + len(static_covars)
    else:
        input_size = 1 + len(static_covars)
    
    # Use the same model architecture as during training
    hidden_size = training_args.get("hidden_size", 256)
    num_layers = training_args.get("num_layers", 4)
    dropout = training_args.get("dropout", 0.05)
    
    print(f"[INFO] Using model architecture: hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout}")
    print(f"[INFO] Using cyclical time encoding: {use_cyclical_time}")
    
    # Load models
    gru_model = load_model(
        gru_model_path, 
        input_size=input_size,
        hidden_size=hidden_size,
        dropout_prob=dropout,
        num_layers=num_layers,
        model_type="gru"
    )
    
    lstm_model = load_model(
        lstm_model_path, 
        input_size=input_size,
        hidden_size=hidden_size,
        dropout_prob=dropout,
        num_layers=num_layers,
        model_type="lstm"
    )
    
    gru_model.to(device)
    lstm_model.to(device)
    
    # Set models to evaluation mode
    gru_model.eval()
    lstm_model.eval()
    
    # Create a figure for all simulations
    fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
    
    # Store all predictions for saving to CSV
    all_predictions = []
    
    # For counterfactual analysis, store predictions
    cf_predictions = []
    
    # Temporary variable to track if we've added the "Actual" label to the legend
    actual_label_added = False
    
    # Run predictions on each simulation
    for sim_idx, sim_df in sim_groups:
        sim_df = sim_df.sort_values("timesteps")
        
        # Get global index
        global_idx = sim_df['global_index'].iloc[0]
        
        # Get actual prevalence data
        t = sim_df["timesteps"].values.astype(np.float32)
        y_true = sim_df["prevalence"].values.astype(np.float32)
        
        # Plot actual data - all simulations as dashed black lines
        label = "Actual" if not actual_label_added else None
        ax.plot(t, y_true, color="black", alpha=0.5, linewidth=1, 
                linestyle="--", label=label)
        actual_label_added = True
        
        # Prepare inputs for model
        static_vals = sim_df.iloc[0][static_covars].values.astype(np.float32)
        # Normalize static values
        static_vals = static_scaler.transform(static_vals.reshape(1, -1)).flatten()
        
        T = len(sim_df)
        
        if use_cyclical_time:
            day_of_year = t % 365.0
            sin_t = np.sin(2 * np.pi * day_of_year / 365.0)
            cos_t = np.cos(2 * np.pi * day_of_year / 365.0)
            X_full = np.zeros((T, 2 + len(static_covars)), dtype=np.float32)
            for j in range(T):
                X_full[j, 0] = sin_t[j]
                X_full[j, 1] = cos_t[j]
                X_full[j, 2:] = static_vals
        else:
            # Normalize timesteps
            t_min, t_max = np.min(t), np.max(t)
            t_norm = (t - t_min) / (t_max - t_min) if t_max > t_min else t
            
            X_full = np.zeros((T, 1 + len(static_covars)), dtype=np.float32)
            for j in range(T):
                X_full[j, 0] = t_norm[j]
                X_full[j, 1:] = static_vals
        
        # Get predictions
        y_gru = predict_full_sequence(gru_model, X_full, device)
        y_lstm = predict_full_sequence(lstm_model, X_full, device)
        
        # Only plot model predictions for the first simulation (to avoid clutter)
        if sim_idx == list(sim_groups.groups.keys())[0]:
            ax.plot(t, y_gru, label="GRU", color="red", linewidth=2)
            ax.plot(t, y_lstm, label="LSTM", color="blue", linewidth=2)
            
            # Run counterfactual analysis for the first simulation only
            if cf_param_name and cf_param_values:
                num_cf = len(cf_param_values)
                
                # Create color maps for GRU and LSTM with varying darkness
                gru_cmap = plt.cm.Reds(np.linspace(0.3, 0.8, num_cf))
                lstm_cmap = plt.cm.Blues(np.linspace(0.3, 0.8, num_cf))
                
                # Run the counterfactual analysis for each value
                for idx, cf_val in enumerate(cf_param_values):
                    cf_gru, cf_lstm, orig_val = run_counterfactual(
                        cf_param_name, cf_val, static_vals, static_covars, static_scaler,
                        sim_df, t, use_cyclical_time, gru_model, lstm_model, device
                    )
                    
                    # Get colors for this counterfactual
                    gru_color = gru_cmap[idx]
                    lstm_color = lstm_cmap[idx]
                    
                    # Plot counterfactual predictions
                    ax.plot(t, cf_gru, color=gru_color, linestyle='--', linewidth=1.5,
                           label=f"GRU {cf_param_name}={cf_val}")
                    ax.plot(t, cf_lstm, color=lstm_color, linestyle='--', linewidth=1.5,
                           label=f"LSTM {cf_param_name}={cf_val}")
                    
                    # Store counterfactual predictions for CSV
                    for j in range(len(t)):
                        cf_predictions.append({
                            'parameter_index': param_index,
                            'simulation_index': sim_idx,
                            'global_index': global_idx,
                            'timestep': t[j],
                            'counterfactual_param': cf_param_name,
                            'counterfactual_value': cf_val,
                            'original_value': orig_val,
                            'gru_prediction': cf_gru[j],
                            'lstm_prediction': cf_lstm[j]
                        })
        
        # Store predictions for CSV
        for j in range(len(t)):
            all_predictions.append({
                'parameter_index': param_index,
                'simulation_index': sim_idx,
                'global_index': global_idx,
                'timestep': t[j],
                'true_prevalence': y_true[j],
                'gru_prediction': y_gru[j],
                'lstm_prediction': y_lstm[j]
            })
    
    # Finalize plot
    title = f"Parameter Index = {param_index} | Global Index = {global_index}\n({num_sims} Simulations)"
    if cf_param_name and cf_param_values:
        title += f"\nCounterfactual Analysis: {cf_param_name} varied from baseline {param_values[cf_param_name]}"
    ax.set_title(title)
    ax.set_xlabel("Time Step (Days)")
    ax.set_ylabel("Prevalence")
    
    # Set y-axis limits based on the --tight flag
    if not args.tight:
        ax.set_ylim([0, 1])
        print("[INFO] Using full prevalence scale (0-1) for plot")
    else:
        print("[INFO] Using tight y-axis scaling based on data range")
    
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(alpha=0.3)
    
    # Save plot
    scaling_suffix = "tight" if args.tight else "full"
    plot_filename = f"parameter_{param_index}_global_{global_index}_predictions_{scaling_suffix}"
    if cf_param_name:
        plot_filename += f"_cf_{cf_param_name}"
    plot_path = os.path.join(args.output_dir, f"{plot_filename}.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"[INFO] Saved prediction plot to {plot_path}")
    
    # Save predictions to CSV
    csv_path = os.path.join(args.output_dir, f"parameter_{param_index}_global_{global_index}_predictions.csv")
    pd.DataFrame(all_predictions).to_csv(csv_path, index=False)
    print(f"[INFO] Saved prediction data to {csv_path}")
    
    # Save counterfactual predictions to CSV if applicable
    if cf_param_name and cf_param_values:
        cf_csv_path = os.path.join(args.output_dir, f"parameter_{param_index}_global_{global_index}_cf_{cf_param_name}_predictions.csv")
        pd.DataFrame(cf_predictions).to_csv(cf_csv_path, index=False)
        print(f"[INFO] Saved counterfactual prediction data to {cf_csv_path}")
    
    print("\n[INFO] Summary:")
    print(f"  - Parameter Index: {param_index}")
    print(f"  - Global Index: {global_index}")
    print(f"  - RDS file: simulation_results_{global_index}.rds")
    print(f"  - Number of simulations: {num_sims}")
    print(f"  - Plot y-axis scaling: {'Tight (data range)' if args.tight else 'Full (0-1)'}")
    
    if cf_param_name and cf_param_values:
        print(f"  - Counterfactual analysis: {cf_param_name} varied with values {cf_param_values}")
        print(f"    Original value: {param_values[cf_param_name]}")
    
    print("[INFO] Done!")


if __name__ == "__main__":
    main()