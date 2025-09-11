# #!/usr/bin/env python3
# """
# Standalone script to visualize test set predictions for time series models.
# Optimized version with batch data fetching for speed.
# """

# import argparse
# import json
# import logging
# import math
# import os
# import pickle
# from typing import Dict, List, Tuple, Optional

# import duckdb
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from tqdm import tqdm

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)


# class GRUModel(nn.Module):
#     """GRU Model for time series prediction."""
    
#     def __init__(self, input_size: int, hidden_size: int, output_size: int, 
#                  dropout_prob: float, num_layers: int = 1, predictor: str = "prevalence"):
#         """Initialize GRU model."""
#         super(GRUModel, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.predictor = predictor
#         self.gru = nn.GRU(
#             input_size, hidden_size,
#             num_layers=num_layers,
#             dropout=dropout_prob if num_layers > 1 else 0.0
#         )
#         self.fc = nn.Linear(hidden_size, output_size)
#         self.ln = nn.LayerNorm(hidden_size)
#         self.dropout = nn.Dropout(dropout_prob)
        
#         # Different activation functions based on predictor type
#         if predictor == "prevalence":
#             self.activation = nn.Sigmoid()  # Bounded between 0 and 1
#         else:  # cases
#             self.activation = nn.Softplus()  # Non-negative but unbounded above

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass through the model."""
#         out, _ = self.gru(x)
#         out = self.ln(out)
#         out = self.dropout(out)
#         out = self.fc(out)   
#         out = self.activation(out)
#         return out


# class LSTMModel(nn.Module):
#     """LSTM Model for time series prediction."""
    
#     def __init__(self, input_size: int, hidden_size: int, output_size: int, 
#                  dropout_prob: float, num_layers: int = 1, predictor: str = "prevalence"):
#         """Initialize LSTM model."""
#         super(LSTMModel, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.predictor = predictor
#         self.lstm = nn.LSTM(
#             input_size, hidden_size,
#             num_layers=num_layers,
#             dropout=dropout_prob if num_layers > 1 else 0.0
#         )
#         self.fc = nn.Linear(hidden_size, output_size)
#         self.ln = nn.LayerNorm(hidden_size)
#         self.dropout = nn.Dropout(dropout_prob)
        
#         # Different activation functions based on predictor type
#         if predictor == "prevalence":
#             self.activation = nn.Sigmoid()  # Bounded between 0 and 1
#         else:  # cases
#             self.activation = nn.Softplus()  # Non-negative but unbounded above

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass through the model."""
#         out, _ = self.lstm(x)  
#         out = self.ln(out)
#         out = self.dropout(out)
#         out = self.fc(out) 
#         out = self.activation(out)
#         return out


# class TestVisualizer:
#     """Class to visualize test set predictions."""
    
#     def __init__(self, config_path: str, split_file: str, 
#                  gru_model_path: str, lstm_model_path: str, 
#                  scaler_path: str, output_pdf: str, max_params: int = None,
#                  num_threads: int = 16):
#         """Initialize the visualizer with paths to required files."""
        
#         # Load configuration
#         with open(config_path, 'r') as f:
#             self.config = json.load(f)
            
#         # Load split information
#         self.split_df = pd.read_csv(split_file)
#         self.test_df = self.split_df[self.split_df['split'] == 'test'].copy()
        
#         # Load scaler
#         with open(scaler_path, 'rb') as f:
#             self.static_scaler = pickle.load(f)
            
#         # Paths
#         self.gru_model_path = gru_model_path
#         self.lstm_model_path = lstm_model_path
#         self.output_pdf = output_pdf
#         self.max_params = max_params
#         self.num_threads = num_threads
        
#         # Extract key config parameters
#         self.db_path = self.config['db_path']
#         self.table_name = self.config['table_name']
#         self.window_size = self.config['window_size']
#         self.predictor = self.config['predictor']
#         self.use_cyclical_time = self.config['use_cyclical_time']
#         self.device = torch.device(self.config['device'])
        
#         # Static covariates
#         self.static_covars = [
#             "eir", "dn0_use", "dn0_future", "Q0", "phi_bednets",
#             "seasonal", "routine", "itn_use", "irs_use",
#             "itn_future", "irs_future", "lsm"
#         ]
        
#         # Determine input size
#         if self.use_cyclical_time:
#             self.input_size = 2 + len(self.static_covars)  # sin, cos, static features
#         else:
#             self.input_size = 1 + len(self.static_covars)  # time, static features
            
#         logger.info(f"Loaded configuration for {self.predictor} prediction")
#         logger.info(f"Found {len(self.test_df)} test simulations")
        
#     def fetch_all_test_data(self) -> Dict[Tuple[int, int], pd.DataFrame]:
#         """Fetch all test simulation data in a single batch query."""
        
#         logger.info("Starting batch data fetch for all test simulations...")
        
#         # Get all test parameter-simulation pairs
#         test_pairs = [(row['parameter_index'], row['simulation_index']) 
#                       for _, row in self.test_df.iterrows()]
        
#         # Limit if max_params is specified
#         if self.max_params:
#             unique_params = self.test_df['parameter_index'].unique()[:self.max_params]
#             test_pairs = [(p, s) for p, s in test_pairs if p in unique_params]
        
#         logger.info(f"Fetching data for {len(test_pairs)} simulation pairs...")
        
#         # Connect to database
#         con = duckdb.connect(self.db_path, read_only=True)
#         con.execute(f"PRAGMA memory_limit='32GB';")
#         con.execute(f"PRAGMA threads={self.num_threads};")
        
#         # Build WHERE clause for all pairs
#         where_conditions = " OR ".join([
#             f"(parameter_index = {p} AND simulation_index = {s})"
#             for p, s in test_pairs
#         ])
        
#         last_6_years_day = 6 * 365
        
#         # Build query based on predictor type
#         if self.predictor == "prevalence":
#             query = f"""
#                 WITH raw_data AS (
#                     SELECT
#                         parameter_index,
#                         simulation_index,
#                         global_index,
#                         timesteps,
#                         CASE WHEN n_age_0_1825 = 0 THEN NULL
#                              ELSE CAST(n_detect_lm_0_1825 AS DOUBLE) / n_age_0_1825
#                         END AS raw_prevalence,
#                         eir, dn0_use, dn0_future, Q0, phi_bednets,
#                         seasonal, routine, itn_use, irs_use,
#                         itn_future, irs_future, lsm
#                     FROM {self.table_name}
#                     WHERE ({where_conditions})
#                       AND timesteps >= {last_6_years_day}
#                 ),
#                 windowed_data AS (
#                     SELECT
#                         parameter_index,
#                         simulation_index,
#                         global_index,
#                         timesteps,
#                         AVG(raw_prevalence) OVER (
#                             PARTITION BY parameter_index, simulation_index
#                             ORDER BY timesteps
#                             ROWS BETWEEN {self.window_size - 1} PRECEDING AND CURRENT ROW
#                         ) AS prevalence,
#                         eir, dn0_use, dn0_future, Q0, phi_bednets,
#                         seasonal, routine, itn_use, irs_use,
#                         itn_future, irs_future, lsm
#                     FROM raw_data
#                     WHERE (timesteps % {self.window_size}) = 0
#                 )
#                 SELECT
#                     parameter_index,
#                     simulation_index,
#                     global_index,
#                     ROW_NUMBER() OVER (
#                         PARTITION BY parameter_index, simulation_index 
#                         ORDER BY timesteps
#                     ) AS timesteps,
#                     prevalence,
#                     eir, dn0_use, dn0_future, Q0, phi_bednets,
#                     seasonal, routine, itn_use, irs_use,
#                     itn_future, irs_future, lsm
#                 FROM windowed_data
#                 ORDER BY parameter_index, simulation_index, timesteps
#             """
#         else:  # cases
#             query = f"""
#                 WITH raw_data AS (
#                     SELECT
#                         parameter_index,
#                         simulation_index,
#                         global_index,
#                         timesteps,
#                         n_inc_clinical_0_36500,
#                         n_age_0_36500,
#                         eir, dn0_use, dn0_future, Q0, phi_bednets,
#                         seasonal, routine, itn_use, irs_use,
#                         itn_future, irs_future, lsm
#                     FROM {self.table_name}
#                     WHERE ({where_conditions})
#                       AND timesteps >= {last_6_years_day}
#                 ),
#                 timestep_groups AS (
#                     SELECT
#                         parameter_index,
#                         simulation_index,
#                         global_index,
#                         FLOOR((timesteps - {last_6_years_day}) / {self.window_size}) AS group_id,
#                         1000.0 * SUM(n_inc_clinical_0_36500) / AVG(n_age_0_36500) AS cases,
#                         MAX(eir) AS eir,
#                         MAX(dn0_use) AS dn0_use,
#                         MAX(dn0_future) AS dn0_future,
#                         MAX(Q0) AS Q0,
#                         MAX(phi_bednets) AS phi_bednets,
#                         MAX(seasonal) AS seasonal,
#                         MAX(routine) AS routine,
#                         MAX(itn_use) AS itn_use,
#                         MAX(irs_use) AS irs_use,
#                         MAX(itn_future) AS itn_future,
#                         MAX(irs_future) AS irs_future,
#                         MAX(lsm) AS lsm
#                     FROM raw_data
#                     GROUP BY parameter_index, simulation_index, global_index, group_id
#                 )
#                 SELECT
#                     parameter_index,
#                     simulation_index,
#                     global_index,
#                     ROW_NUMBER() OVER (
#                         PARTITION BY parameter_index, simulation_index 
#                         ORDER BY group_id
#                     ) AS timesteps,
#                     cases,
#                     eir, dn0_use, dn0_future, Q0, phi_bednets,
#                     seasonal, routine, itn_use, irs_use,
#                     itn_future, irs_future, lsm
#                 FROM timestep_groups
#                 ORDER BY parameter_index, simulation_index, group_id
#             """
        
#         # Execute query
#         logger.info("Executing batch query...")
#         all_data_df = con.execute(query).df()
#         con.close()
        
#         logger.info(f"Fetched {len(all_data_df)} total rows")
        
#         # Group by parameter and simulation indices
#         data_dict = {}
#         for (param_idx, sim_idx), group_df in all_data_df.groupby(['parameter_index', 'simulation_index']):
#             data_dict[(param_idx, sim_idx)] = group_df.reset_index(drop=True)
        
#         logger.info(f"Organized data for {len(data_dict)} simulation pairs")
        
#         return data_dict
    
#     def load_models(self) -> Tuple[nn.Module, nn.Module]:
#         """Load the GRU and LSTM models from saved checkpoints."""
        
#         # Load GRU model checkpoint to get hyperparameters
#         gru_checkpoint = torch.load(self.gru_model_path, map_location=self.device)
#         gru_state_dict = gru_checkpoint['model_state_dict']
        
#         # Extract GRU model dimensions from state dict
#         gru_hidden_size = gru_state_dict['gru.weight_ih_l0'].shape[0] // 3  # GRU has 3x hidden_size
#         gru_num_layers = sum(1 for key in gru_state_dict.keys() if 'weight_ih_l' in key and 'gru' in key)
        
#         # Initialize GRU model
#         gru_model = GRUModel(
#             input_size=self.input_size,
#             hidden_size=gru_hidden_size,
#             output_size=1,
#             dropout_prob=0.0,  # Set to 0 for evaluation
#             num_layers=gru_num_layers,
#             predictor=self.predictor
#         )
#         gru_model.load_state_dict(gru_state_dict)
#         gru_model.to(self.device)
#         gru_model.eval()
        
#         # Load LSTM model checkpoint
#         lstm_checkpoint = torch.load(self.lstm_model_path, map_location=self.device)
#         lstm_state_dict = lstm_checkpoint['model_state_dict']
        
#         # Extract LSTM model dimensions
#         lstm_hidden_size = lstm_state_dict['lstm.weight_ih_l0'].shape[0] // 4  # LSTM has 4x hidden_size
#         lstm_num_layers = sum(1 for key in lstm_state_dict.keys() if 'weight_ih_l' in key and 'lstm' in key)
        
#         # Initialize LSTM model
#         lstm_model = LSTMModel(
#             input_size=self.input_size,
#             hidden_size=lstm_hidden_size,
#             output_size=1,
#             dropout_prob=0.0,  # Set to 0 for evaluation
#             num_layers=lstm_num_layers,
#             predictor=self.predictor
#         )
#         lstm_model.load_state_dict(lstm_state_dict)
#         lstm_model.to(self.device)
#         lstm_model.eval()
        
#         logger.info(f"Loaded GRU model: {gru_num_layers} layers, hidden size {gru_hidden_size}")
#         logger.info(f"Loaded LSTM model: {lstm_num_layers} layers, hidden size {lstm_hidden_size}")
        
#         return gru_model, lstm_model
    
#     def prepare_input_features(self, df: pd.DataFrame) -> np.ndarray:
#         """Prepare input features for model prediction."""
        
#         t = df["timesteps"].values.astype(np.float32)
#         static_vals = df.iloc[0][self.static_covars].values.astype(np.float32)
#         static_vals = self.static_scaler.transform(static_vals.reshape(1, -1)).flatten()
        
#         T = len(df)
        
#         if self.use_cyclical_time:
#             if self.predictor == "cases":
#                 day_of_year = (t * self.window_size) % 365.0
#             else:
#                 day_of_year = t % 365.0
                
#             sin_t = np.sin(2 * np.pi * day_of_year / 365.0)
#             cos_t = np.cos(2 * np.pi * day_of_year / 365.0)
            
#             X = np.zeros((T, 2 + len(self.static_covars)), dtype=np.float32)
#             X[:, 0] = sin_t
#             X[:, 1] = cos_t
#             X[:, 2:] = np.tile(static_vals, (T, 1))
#         else:
#             # Normalize timesteps
#             t_min, t_max = np.min(t), np.max(t)
#             t_norm = (t - t_min) / (t_max - t_min) if t_max > t_min else t
            
#             X = np.zeros((T, 1 + len(self.static_covars)), dtype=np.float32)
#             X[:, 0] = t_norm
#             X[:, 1:] = np.tile(static_vals, (T, 1))
            
#         return X
    
#     def predict_with_model(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
#         """Make predictions with a model."""
        
#         with torch.no_grad():
#             x_torch = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
#             pred = model(x_torch).squeeze(-1).squeeze(-1).cpu().numpy()
#         return pred
    
#     def create_plots(self):
#         """Create PDF with plots for test simulations."""
        
#         # Load models
#         gru_model, lstm_model = self.load_models()
        
#         # Fetch all data upfront
#         logger.info("Fetching all test data in batch...")
#         all_data = self.fetch_all_test_data()
        
#         # Get unique parameters in test set
#         test_params = self.test_df['parameter_index'].unique()
        
#         # Limit number of parameters if specified
#         if self.max_params:
#             test_params = test_params[:self.max_params]
            
#         logger.info(f"Creating plots for {len(test_params)} parameters")
        
#         # Create PDF
#         with PdfPages(self.output_pdf) as pdf:
            
#             for param_idx in tqdm(test_params, desc="Creating plots"):
#                 # Get all test simulations for this parameter
#                 param_sims = self.test_df[self.test_df['parameter_index'] == param_idx]
                
#                 if len(param_sims) == 0:
#                     continue
                
#                 # Collect all predictions for this parameter
#                 all_y_true = []
#                 all_gru_preds = []
#                 all_lstm_preds = []
#                 param_values = None
                
#                 for _, row in param_sims.iterrows():
#                     sim_idx = row['simulation_index']
                    
#                     # Look up pre-fetched data
#                     if (param_idx, sim_idx) not in all_data:
#                         logger.warning(f"No data for param {param_idx}, sim {sim_idx}")
#                         continue
                    
#                     df = all_data[(param_idx, sim_idx)]
                    
#                     if len(df) == 0:
#                         logger.warning(f"Empty data for param {param_idx}, sim {sim_idx}")
#                         continue
                    
#                     # Store parameter values from first simulation
#                     if param_values is None:
#                         param_values = df.iloc[0][self.static_covars].to_dict()
                    
#                     # Get true values
#                     target_col = "prevalence" if self.predictor == "prevalence" else "cases"
#                     y_true = df[target_col].values
                    
#                     # Prepare features for prediction
#                     X = self.prepare_input_features(df)
                    
#                     # Get predictions from both models
#                     gru_pred = self.predict_with_model(gru_model, X)
#                     lstm_pred = self.predict_with_model(lstm_model, X)
                    
#                     all_y_true.append(y_true)
#                     all_gru_preds.append(gru_pred)
#                     all_lstm_preds.append(lstm_pred)
                
#                 if len(all_y_true) == 0:
#                     continue
                
#                 # Create single plot for this parameter
#                 fig, ax = plt.subplots(figsize=(12, 8))
                
#                 # Convert timesteps to years
#                 # Data represents years 6-12 with timesteps every 14 days
#                 # We want to plot years 8-12 (the last 4 years), shown as years 0-4
#                 timesteps = np.arange(len(all_y_true[0]))
#                 days = timesteps * self.window_size  # Convert to days
#                 years_raw = days / 365.0  # Convert to years (starting at year 6)
                
#                 # We want to show years 8-12 as years 0-4
#                 # So we filter for years >= 2 (which is year 8 in absolute terms)
#                 # and shift by 2 to make it start at 0
#                 year_2_idx = np.where(years_raw >= 2)[0][0] if any(years_raw >= 2) else 0
                
#                 # Filter data to show only years 2-6 (which become 0-4 after shifting)
#                 years = years_raw[year_2_idx:] - 2  # Shift to start at 0
                
#                 # Plot individual simulations in gray
#                 for i, (y_true, gru_pred, lstm_pred) in enumerate(zip(all_y_true, all_gru_preds, all_lstm_preds)):
#                     y_true_filtered = y_true[year_2_idx:]
#                     gru_pred_filtered = gru_pred[year_2_idx:]
#                     lstm_pred_filtered = lstm_pred[year_2_idx:]
                    
#                     label_suffix = ' (individual)' if i == 0 else ''
#                     ax.plot(years, y_true_filtered, color='gray', alpha=0.3, linewidth=0.8, 
#                            label=f'True{label_suffix}' if i == 0 else '')
#                     ax.plot(years, gru_pred_filtered, color='lightcoral', alpha=0.3, linewidth=0.8,
#                            label=f'GRU{label_suffix}' if i == 0 else '')
#                     ax.plot(years, lstm_pred_filtered, color='lightblue', alpha=0.3, linewidth=0.8,
#                            label=f'LSTM{label_suffix}' if i == 0 else '')
                
#                 # Calculate and plot averages in bold colors
#                 avg_y_true = np.mean([y[year_2_idx:] for y in all_y_true], axis=0)
#                 avg_gru_pred = np.mean([p[year_2_idx:] for p in all_gru_preds], axis=0)
#                 avg_lstm_pred = np.mean([p[year_2_idx:] for p in all_lstm_preds], axis=0)
                
#                 ax.plot(years, avg_y_true, 'k-', label='True (average)', linewidth=2.5)
#                 ax.plot(years, avg_gru_pred, 'r-', label='GRU', linewidth=2.5, alpha=0.9)
#                 ax.plot(years, avg_lstm_pred, 'b-', label='LSTM', linewidth=2.5, alpha=0.9)
                
#                 # Add vertical dashed line at year 1
#                 ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Intervention Start')
                
#                 # Set y-axis limits for prevalence
#                 if self.predictor == "prevalence":
#                     ax.set_ylim(0, 1)
                
#                 ax.set_xlabel("Years", fontsize=12)
#                 ax.set_ylabel("Prevalence" if self.predictor == "prevalence" else "Cases per 1000", fontsize=12)
#                 ax.set_xlim(0, 4)
                
#                 # Create parameter text for title
#                 if param_values:
#                     param_text_lines = []
#                     # Group parameters for better display
#                     param_groups = [
#                         ['eir', 'Q0', 'phi_bednets'],
#                         ['dn0_use', 'dn0_future', 'seasonal', 'routine'],
#                         ['itn_use', 'irs_use', 'itn_future', 'irs_future', 'lsm']
#                     ]
                    
#                     for group in param_groups:
#                         line_parts = []
#                         for key in group:
#                             if key in param_values:
#                                 val = param_values[key]
#                                 # Format based on value type
#                                 if isinstance(val, (int, np.integer)) or (isinstance(val, (float, np.floating)) and val.is_integer()):
#                                     formatted_val = f"{int(val)}"
#                                 else:
#                                     formatted_val = f"{val:.3f}"
#                                 line_parts.append(f"{key}={formatted_val}")
#                         if line_parts:
#                             param_text_lines.append("  ".join(line_parts))
                    
#                     param_text = "\n".join(param_text_lines)
                    
#                     # Main title with parameter index
#                     ax.set_title(f'Parameter {param_idx} - {len(param_sims)} Simulations ({self.predictor.capitalize()})\n\n{param_text}', 
#                                fontsize=12, fontweight='bold', pad=20)
#                 else:
#                     ax.set_title(f'Parameter {param_idx} - {len(param_sims)} Simulations ({self.predictor.capitalize()})', 
#                                fontsize=12, fontweight='bold')
                
#                 ax.legend(loc='best', fontsize=10)
#                 ax.grid(True, alpha=0.3)
                
#                 plt.tight_layout()
                
#                 # Save to PDF
#                 pdf.savefig(fig, bbox_inches='tight')
#                 plt.close(fig)
                
#         logger.info(f"PDF saved to {self.output_pdf}")


# def main():
#     """Main function to run visualization."""
    
#     parser = argparse.ArgumentParser(description="Visualize test set predictions for time series models")
    
#     parser.add_argument("--config", required=True, 
#                         help="Path to args.json configuration file")
#     parser.add_argument("--split-file", required=True,
#                         help="Path to train_val_test_split.csv file")
#     parser.add_argument("--gru-model", required=True,
#                         help="Path to GRU model checkpoint (gru_best.pt)")
#     parser.add_argument("--lstm-model", required=True,
#                         help="Path to LSTM model checkpoint (lstm_best.pt)")
#     parser.add_argument("--scaler", required=True,
#                         help="Path to static scaler pickle file")
#     parser.add_argument("--output-pdf", default="test_predictions.pdf",
#                         help="Output PDF filename")
#     parser.add_argument("--max-params", type=int, default=None,
#                         help="Maximum number of parameters to plot")
#     parser.add_argument("--num-threads", type=int, default=16,
#                         help="Number of threads for DuckDB parallel query execution (default: 16)")
    
#     args = parser.parse_args()
    
#     # Create visualizer and generate plots
#     visualizer = TestVisualizer(
#         config_path=args.config,
#         split_file=args.split_file,
#         gru_model_path=args.gru_model,
#         lstm_model_path=args.lstm_model,
#         scaler_path=args.scaler,
#         output_pdf=args.output_pdf,
#         max_params=args.max_params,
#         num_threads=args.num_threads
#     )
    
#     visualizer.create_plots()


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""
Test set visualizer (updated for new training setup)

What changed vs your previous viz.py
------------------------------------
1) **Matches your new training setup**: models output in the *transformed* space (identity activation).
   We now invert predictions at plot-time:
      - prevalence: sigmoid
      - cases: expm1
2) **Works for prevalence *or* cases** using the `predictor` field in your args.json.
3) **DB queries now carry `abs_timesteps`** (absolute day index) and compute windowed aggregates
   the same way your training script does (ratio-of-sums for prevalence; denom-weighted rate for cases).
4) **Feature prep now gates future covars** (dn0_future, itn_future, irs_future, lsm, routine)
   before day 9*365 to mirror training.
5) **Early-window filters**:
   - Prevalence: use `--prev-cutoff` (default 0.02) as before
   - Cases: new `--cases-cutoff` (default 0.1 per 1000 per day)

Usage examples (DB mode)
------------------------
python viz_updated_cases_or_prevalence.py \
  --config /path/to/args.json \
  --gru-model /path/to/gru_final.pt \
  --lstm-model /path/to/lstm_final.pt \
  --scaler /path/to/static_scaler.pkl \
  --output-pdf out.pdf \
  --param-start 4096 --param-count 257

Legacy split mode still works with --split-file (but DB mode is preferred).
"""

import argparse
import json
import logging
import pickle
from typing import Dict, List, Tuple, Optional

import duckdb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------
# Inverse transforms (match training)
# ---------------------------

def inverse_transform_np(y: np.ndarray, predictor: str) -> np.ndarray:
    if predictor == "prevalence":
        return 1.0 / (1.0 + np.exp(-y))  # sigmoid
    else:  # cases
        return np.expm1(y)

# ---------------------------
# Models (no output activation; identity like training)
# ---------------------------

class GRUModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 dropout_prob: float, num_layers: int = 1, predictor: str = "prevalence"):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.predictor = predictor
        self.gru = nn.GRU(
            input_size, hidden_size,
            num_layers=num_layers,
            dropout=dropout_prob if num_layers > 1 else 0.0,
            batch_first=False
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        out = self.ln(out)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.activation(out)
        return out

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 dropout_prob: float, num_layers: int = 1, predictor: str = "prevalence"):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.predictor = predictor
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            dropout=dropout_prob if num_layers > 1 else 0.0,
            batch_first=False
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.ln(out)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.activation(out)
        return out

# ---------------------------
# Visualizer
# ---------------------------

class TestVisualizer:
    def __init__(self, config_path: str,
                 gru_model_path: str, lstm_model_path: str,
                 scaler_path: str, output_pdf: str,
                 split_file: Optional[str] = None,
                 max_params: Optional[int] = None,
                 num_threads: int = 16,
                 selected_params_from_split: Optional[List[int]] = None,
                 prev_cutoff: float = 0.02,
                 cases_cutoff: float = 0.1,
                 db_param_indices: Optional[List[int]] = None):
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Early fields for DB
        self.db_path = self.config['db_path']
        self.table_name = self.config['table_name']
        self.num_threads = num_threads

        # Optional split file (legacy)
        self.split_df = None
        if split_file:
            self.split_df = pd.read_csv(split_file)

        # Selection mode
        self.db_param_indices = sorted(set(db_param_indices)) if db_param_indices else None
        self.selected_params = sorted(set(selected_params_from_split)) if selected_params_from_split else None

        if self.db_param_indices is None:
            if self.split_df is None:
                raise ValueError(
                    "No parameter indices provided and no --split-file given. "
                    "Provide --param-indices / --param-start/--param-end/--param-count to use the DB directly, "
                    "or pass --split-file to use the legacy split flow."
                )
            self.test_df = self.split_df[self.split_df['split'] == 'test'].copy()
            if self.selected_params:
                before = len(self.test_df)
                self.test_df = self.test_df[self.test_df['parameter_index'].isin(self.selected_params)].copy()
                after = len(self.test_df)
                missing = sorted(set(self.selected_params) - set(self.test_df['parameter_index'].unique()))
                logger.info(f"Filtering to {len(self.selected_params)} specified parameter_index values → kept {after} test rows (from {before}).")
                if missing:
                    logger.warning(f"{len(missing)} specified parameter_index not present in test split: {missing}")
        else:
            self.test_df = self._build_pairs_from_db(self.db_param_indices)

        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.static_scaler = pickle.load(f)

        self.gru_model_path = gru_model_path
        self.lstm_model_path = lstm_model_path
        self.output_pdf = output_pdf

        self.max_params = None if self.db_param_indices is not None else max_params

        self.prev_cutoff = float(prev_cutoff)
        self.cases_cutoff = float(cases_cutoff)  # per 1000 per day

        # Other config
        self.window_size = self.config['window_size']
        self.predictor = self.config['predictor']  # 'prevalence' or 'cases'
        self.use_cyclical_time = self.config['use_cyclical_time']
        self.device = torch.device(self.config['device'])

        # Static covariates and gating set
        self.static_covars = [
            "eir", "dn0_use", "dn0_future", "Q0", "phi_bednets",
            "seasonal", "routine", "itn_use", "irs_use",
            "itn_future", "irs_future", "lsm"
        ]
        self.after9_covars = {"dn0_future", "itn_future", "irs_future", "lsm", "routine"}
        self.intervention_day = 9 * 365

        # Determine network input size
        self.input_size = (2 if self.use_cyclical_time else 1) + len(self.static_covars)

        logger.info(f"Loaded configuration for {self.predictor} prediction")
        logger.info(f"Found {len(self.test_df)} selected simulation rows")
        logger.info("Selection mode: %s", "DB" if self.db_param_indices is not None else "split CSV")

    # --- Helpers ---

    def _build_pairs_from_db(self, param_indices: List[int]) -> pd.DataFrame:
        if not param_indices:
            raise ValueError("No parameter indices provided for DB-based selection.")
        con = duckdb.connect(self.db_path, read_only=True)
        con.execute("PRAGMA memory_limit='32GB';")
        con.execute(f"PRAGMA threads={self.num_threads};")
        params_sorted = sorted(set(int(p) for p in param_indices))
        logger.info(f"Querying DB for {len(params_sorted)} parameter_index values…")
        vals = ",".join(f"({p})" for p in params_sorted)
        q = f"""
            WITH param_list(parameter_index) AS (VALUES {vals})
            SELECT DISTINCT t.parameter_index, t.simulation_index, t.global_index
            FROM {self.table_name} AS t
            INNER JOIN param_list p ON t.parameter_index = p.parameter_index
            ORDER BY t.parameter_index, t.simulation_index
        """
        df = con.execute(q).df()
        con.close()
        if df.empty:
            raise ValueError("No rows found in DB for the requested parameter indices.")
        df['split'] = 'db'
        return df[['parameter_index', 'simulation_index', 'global_index', 'split']]

    # --- Data fetch ---

    def fetch_all_test_data(self) -> Dict[Tuple[int, int], pd.DataFrame]:
        logger.info("Starting batch data fetch for the selected simulations…")
        if len(self.test_df) == 0:
            logger.warning("No simulations to fetch. Skipping data fetch.")
            return {}

        pairs = [(int(row['parameter_index']), int(row['simulation_index'])) for _, row in self.test_df.iterrows()]
        if self.db_param_indices is None and self.max_params:
            allowed = set(self.test_df['parameter_index'].unique()[:self.max_params])
            pairs = [(p, s) for p, s in pairs if p in allowed]
        logger.info(f"Fetching data for {len(pairs)} simulation pairs…")

        con = duckdb.connect(self.db_path, read_only=True)
        con.execute("PRAGMA memory_limit='32GB';")
        con.execute(f"PRAGMA threads={self.num_threads};")

        where_conditions = " OR ".join([f"(parameter_index = {p} AND simulation_index = {s})" for p, s in pairs])
        if not where_conditions:
            logger.warning("No (parameter_index, simulation_index) pairs after limiting. Skipping.")
            con.close(); return {}

        last_6_years_day = 6 * 365
        W = self.window_size

        if self.predictor == "prevalence":
            query = f"""
                WITH raw AS (
                    SELECT
                        parameter_index, simulation_index, global_index,
                        timesteps AS abs_timesteps,
                        CAST(n_detect_lm_0_1825 AS DOUBLE) AS n_detect,
                        CAST(n_age_0_1825        AS DOUBLE) AS n_age,
                        eir, dn0_use, dn0_future, Q0, phi_bednets,
                        seasonal, routine, itn_use, irs_use,
                        itn_future, irs_future, lsm
                    FROM {self.table_name}
                    WHERE ({where_conditions}) AND timesteps >= {last_6_years_day}
                ),
                groups AS (
                    SELECT
                        parameter_index, simulation_index, global_index,
                        FLOOR((abs_timesteps - {last_6_years_day}) / {W}) AS gid,
                        SUM(n_detect) / NULLIF(SUM(n_age), 0) AS prevalence,
                        MIN(abs_timesteps) AS abs_timesteps,
                        MAX(eir) AS eir,
                        MAX(dn0_use) AS dn0_use,
                        MAX(dn0_future) AS dn0_future,
                        MAX(Q0) AS Q0,
                        MAX(phi_bednets) AS phi_bednets,
                        MAX(seasonal) AS seasonal,
                        MAX(routine) AS routine,
                        MAX(itn_use) AS itn_use,
                        MAX(irs_use) AS irs_use,
                        MAX(itn_future) AS itn_future,
                        MAX(irs_future) AS irs_future,
                        MAX(lsm) AS lsm
                    FROM raw
                    GROUP BY 1,2,3,4
                )
                SELECT
                    parameter_index, simulation_index, global_index,
                    ROW_NUMBER() OVER (PARTITION BY parameter_index, simulation_index ORDER BY gid) AS timesteps,
                    abs_timesteps,
                    prevalence,
                    eir, dn0_use, dn0_future, Q0, phi_bednets,
                    seasonal, routine, itn_use, irs_use,
                    itn_future, irs_future, lsm
                FROM groups
                ORDER BY parameter_index, simulation_index, gid
            """
        else:  # cases per 1000 per day
            query = f"""
                WITH raw AS (
                    SELECT
                        parameter_index, simulation_index, global_index,
                        timesteps AS abs_timesteps,
                        CAST(n_inc_clinical_0_36500 AS DOUBLE) AS inc,
                        CAST(n_age_0_36500          AS DOUBLE) AS pop,
                        eir, dn0_use, dn0_future, Q0, phi_bednets,
                        seasonal, routine, itn_use, irs_use,
                        itn_future, irs_future, lsm
                    FROM {self.table_name}
                    WHERE ({where_conditions}) AND timesteps >= {last_6_years_day}
                ),
                groups AS (
                    SELECT
                        parameter_index, simulation_index, global_index,
                        FLOOR((abs_timesteps - {last_6_years_day}) / {W}) AS gid,
                        1000.0 * SUM(inc) / NULLIF(SUM(pop),0) AS cases,
                        SUM(inc) AS cases_count,
                        SUM(pop) AS exposure_pd,
                        MIN(abs_timesteps) AS abs_timesteps,
                        MAX(eir) AS eir,
                        MAX(dn0_use) AS dn0_use,
                        MAX(dn0_future) AS dn0_future,
                        MAX(Q0) AS Q0,
                        MAX(phi_bednets) AS phi_bednets,
                        MAX(seasonal) AS seasonal,
                        MAX(routine) AS routine,
                        MAX(itn_use) AS itn_use,
                        MAX(irs_use) AS irs_use,
                        MAX(itn_future) AS itn_future,
                        MAX(irs_future) AS irs_future,
                        MAX(lsm) AS lsm
                    FROM raw
                    GROUP BY 1,2,3,4
                )
                SELECT
                    parameter_index, simulation_index, global_index,
                    ROW_NUMBER() OVER (PARTITION BY parameter_index, simulation_index ORDER BY gid) AS timesteps,
                    abs_timesteps,
                    cases, cases_count, exposure_pd,
                    eir, dn0_use, dn0_future, Q0, phi_bednets,
                    seasonal, routine, itn_use, irs_use,
                    itn_future, irs_future, lsm
                FROM groups
                ORDER BY parameter_index, simulation_index, gid
            """

        logger.info("Executing batch query…")
        all_data_df = con.execute(query).df()
        con.close()
        logger.info(f"Fetched {len(all_data_df)} total rows")

        data_dict = {}
        for (p, s), g in all_data_df.groupby(['parameter_index', 'simulation_index']):
            data_dict[(int(p), int(s))] = g.reset_index(drop=True)
        logger.info(f"Organized data for {len(data_dict)} simulation pairs")
        return data_dict

    # --- Models ---

    def load_models(self) -> Tuple[nn.Module, nn.Module]:
        # GRU
        gru_checkpoint = torch.load(self.gru_model_path, map_location=self.device)
        gru_state = gru_checkpoint['model_state_dict']
        gru_hidden = gru_state['gru.weight_ih_l0'].shape[0] // 3
        gru_layers = sum(1 for k in gru_state.keys() if k.startswith('gru.weight_ih_l'))
        gru_model = GRUModel(self.input_size, gru_hidden, 1, dropout_prob=0.0, num_layers=gru_layers, predictor=self.predictor)
        gru_model.load_state_dict(gru_state)
        gru_model.to(self.device).eval()

        # LSTM
        lstm_checkpoint = torch.load(self.lstm_model_path, map_location=self.device)
        lstm_state = lstm_checkpoint['model_state_dict']
        lstm_hidden = lstm_state['lstm.weight_ih_l0'].shape[0] // 4
        lstm_layers = sum(1 for k in lstm_state.keys() if k.startswith('lstm.weight_ih_l'))
        lstm_model = LSTMModel(self.input_size, lstm_hidden, 1, dropout_prob=0.0, num_layers=lstm_layers, predictor=self.predictor)
        lstm_model.load_state_dict(lstm_state)
        lstm_model.to(self.device).eval()

        logger.info(f"Loaded GRU: {gru_layers} layers, hidden {gru_hidden}; LSTM: {lstm_layers} layers, hidden {lstm_hidden}")
        return gru_model, lstm_model

    # --- Feature prep & prediction (mirror training) ---

    def prepare_input_features(self, df: pd.DataFrame) -> np.ndarray:
        T = len(df)
        abs_t = df["abs_timesteps"].values.astype(np.float32)
        rel_t = df["timesteps"].values.astype(np.float32)

        base_static = df.iloc[0][self.static_covars].values.astype(np.float32)
        raw_matrix = np.tile(base_static, (T, 1))
        # Gate future-only covariates before intervention day
        post_mask = (abs_t >= self.intervention_day)
        for cov in self.after9_covars:
            j = self.static_covars.index(cov)
            raw_matrix[~post_mask, j] = 0.0
        # Scale per timestep with train-fitted scaler
        scaled = self.static_scaler.transform(raw_matrix)

        if self.use_cyclical_time:
            day_of_year = abs_t % 365.0
            sin_t = np.sin(2 * np.pi * day_of_year / 365.0)
            cos_t = np.cos(2 * np.pi * day_of_year / 365.0)
            X = np.zeros((T, 2 + len(self.static_covars)), dtype=np.float32)
            X[:, 0] = sin_t
            X[:, 1] = cos_t
            X[:, 2:] = scaled
        else:
            t_min, t_max = np.min(rel_t), np.max(rel_t)
            t_norm = (rel_t - t_min) / (t_max - t_min) if t_max > t_min else rel_t
            X = np.zeros((T, 1 + len(self.static_covars)), dtype=np.float32)
            X[:, 0] = t_norm
            X[:, 1:] = scaled
        return X

    def predict_with_model(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)  # [T, 1, F]
            pred_tr = model(x_t).squeeze(-1).squeeze(-1).cpu().numpy()               # transformed space
        return inverse_transform_np(pred_tr, self.predictor)

    # --- Plotting ---

    def create_plots(self):
        gru_model, lstm_model = self.load_models()
        logger.info("Fetching all data…")
        all_data = self.fetch_all_test_data()

        test_params = self.test_df['parameter_index'].unique()
        if self.db_param_indices is None and self.max_params:
            test_params = test_params[:self.max_params]
        logger.info(f"Creating plots for {len(test_params)} parameters")
        if len(test_params) == 0:
            logger.warning("No parameters to plot. Exiting.")
            return

        target_col = "prevalence" if self.predictor == "prevalence" else "cases"

        with PdfPages(self.output_pdf) as pdf:
            for param_idx in tqdm(test_params, desc="Creating plots"):
                param_sims = self.test_df[self.test_df['parameter_index'] == param_idx]
                if len(param_sims) == 0:
                    continue

                all_y_true, all_gru_preds, all_lstm_preds = [], [], []
                param_values = None

                for _, row in param_sims.iterrows():
                    sim_idx = int(row['simulation_index'])
                    key = (int(param_idx), sim_idx)
                    if key not in all_data:
                        logger.warning(f"No data for param {param_idx}, sim {sim_idx}")
                        continue
                    df = all_data[key]
                    if len(df) == 0:
                        logger.warning(f"Empty data for param {param_idx}, sim {sim_idx}")
                        continue

                    if param_values is None:
                        param_values = df.iloc[0][self.static_covars].to_dict()

                    y_true = df[target_col].values.astype(np.float32)
                    X = self.prepare_input_features(df)
                    gru_pred = self.predict_with_model(gru_model, X)
                    lstm_pred = self.predict_with_model(lstm_model, X)

                    all_y_true.append(y_true)
                    all_gru_preds.append(gru_pred)
                    all_lstm_preds.append(lstm_pred)

                if len(all_y_true) == 0:
                    continue

                # Early-window screening (first year after the last_6_years start)
                # Map relative years since day 6*365
                T0 = len(all_y_true[0])
                timesteps = np.arange(T0)
                years_raw_full = (timesteps * self.window_size) / 365.0  # 0..~6
                early_mask = years_raw_full < 1.0

                if early_mask.any():
                    early_vals = []
                    for y in all_y_true:
                        vals = y[early_mask]
                        vals = vals[np.isfinite(vals)]
                        if vals.size:
                            early_vals.append(vals)
                    if early_vals:
                        early_mean = float(np.nanmean(np.concatenate(early_vals)))
                        if self.predictor == "prevalence":
                            if early_mean < self.prev_cutoff:
                                logger.info(f"Skip param {param_idx}: early mean prevalence {early_mean:.4f} < cutoff {self.prev_cutoff:.4f}")
                                continue
                        else:  # cases
                            if early_mean < self.cases_cutoff:
                                logger.info(f"Skip param {param_idx}: early mean cases {early_mean:.4f} < cutoff {self.cases_cutoff:.4f} per 1000/day")
                                continue

                # Build x-axis so that we display years 0–4 after dropping first 2 years (matching earlier viz)
                days = np.arange(len(all_y_true[0])) * self.window_size
                years_rel = days / 365.0
                year_2_idx = np.where(years_rel >= 2)[0][0] if (years_rel >= 2).any() else 0
                years_display = years_rel[year_2_idx:] - 2  # show 0..~4, with vline at 1 (=abs year 9)

                fig, ax = plt.subplots(figsize=(12, 8))

                # Plot each simulation (faint)
                for i, (y_true, p_gru, p_lstm) in enumerate(zip(all_y_true, all_gru_preds, all_lstm_preds)):
                    y_t = y_true[year_2_idx:]
                    g_t = p_gru[year_2_idx:]
                    l_t = p_lstm[year_2_idx:]
                    label_suffix = ' (individual)' if i == 0 else ''
                    ax.plot(years_display, y_t, color='gray', alpha=0.3, linewidth=0.8,
                            label=f'True{label_suffix}' if i == 0 else '')
                    ax.plot(years_display, g_t, color='lightcoral', alpha=0.3, linewidth=0.8,
                            label=f'GRU{label_suffix}' if i == 0 else '')
                    ax.plot(years_display, l_t, color='lightblue', alpha=0.3, linewidth=0.8,
                            label=f'LSTM{label_suffix}' if i == 0 else '')

                # Plot averages (bold)
                avg_y_true  = np.mean([y[year_2_idx:] for y in all_y_true], axis=0)
                avg_gru_pred = np.mean([p[year_2_idx:] for p in all_gru_preds], axis=0)
                avg_lstm_pred = np.mean([p[year_2_idx:] for p in all_lstm_preds], axis=0)
                ax.plot(years_display, avg_y_true, 'k-', label='True (average)', linewidth=2.5)
                ax.plot(years_display, avg_gru_pred, 'r-', label='GRU', linewidth=2.5, alpha=0.9)
                ax.plot(years_display, avg_lstm_pred, 'b-', label='LSTM', linewidth=2.5, alpha=0.9)

                ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Intervention Start')

                if self.predictor == "prevalence":
                    ax.set_ylim(0, 1)
                    ylab = "Prevalence"
                else:
                    ylab = "Cases per 1000 per day"

                ax.set_xlabel("Years", fontsize=12)
                ax.set_ylabel(ylab, fontsize=12)
                ax.set_xlim(0, 4)

                # Annotate static parameter values in the title
                if param_values:
                    param_text_lines = []
                    param_groups = [
                        ['eir', 'Q0', 'phi_bednets'],
                        ['dn0_use', 'dn0_future', 'seasonal', 'routine'],
                        ['itn_use', 'irs_use', 'itn_future', 'irs_future', 'lsm']
                    ]
                    for group in param_groups:
                        parts = []
                        for k in group:
                            if k in param_values:
                                val = param_values[k]
                                if isinstance(val, (int, np.integer)) or (isinstance(val, (float, np.floating)) and float(val).is_integer()):
                                    parts.append(f"{k}={int(val)}")
                                else:
                                    parts.append(f"{k}={float(val):.3f}")
                        if parts:
                            param_text_lines.append("  ".join(parts))
                    param_text = "\n".join(param_text_lines)
                    ax.set_title(
                        f'Parameter {param_idx} - {len(param_sims)} Simulations ({self.predictor.capitalize()})\n\n{param_text}',
                        fontsize=12, fontweight='bold', pad=20
                    )
                else:
                    ax.set_title(
                        f'Parameter {param_idx} - {len(param_sims)} Simulations ({self.predictor.capitalize()})',
                        fontsize=12, fontweight='bold'
                    )

                ax.legend(loc='best', fontsize=10)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        logger.info(f"PDF saved to {self.output_pdf}")

# ---------------------------
# CLI
# ---------------------------

def _resolve_db_param_indices(args: argparse.Namespace) -> Optional[List[int]]:
    if args.param_indices:
        return sorted(set(args.param_indices))
    if args.param_start is not None and args.param_end is not None:
        if args.param_end < args.param_start:
            raise ValueError("--param-end must be >= --param-start")
        return list(range(args.param_start, args.param_end + 1))
    if args.param_start is not None and args.param_count is not None:
        if args.param_count <= 0:
            raise ValueError("--param-count must be positive")
        end = args.param_start + args.param_count - 1
        return list(range(args.param_start, end + 1))
    return None

def main():
    parser = argparse.ArgumentParser(description="Visualize predictions for time series models (new setup)")

    # Core config + models
    parser.add_argument("--config", required=True, help="Path to args.json (contains db_path, table_name, predictor, etc.)")
    parser.add_argument("--gru-model", required=True, help="Path to GRU checkpoint (e.g., gru_final.pt)")
    parser.add_argument("--lstm-model", required=True, help="Path to LSTM checkpoint (e.g., lstm_final.pt)")
    parser.add_argument("--scaler", required=True, help="Path to static_scaler.pkl (fit on TRAIN only)")
    parser.add_argument("--output-pdf", default="test_predictions.pdf", help="Output PDF filename")

    # Performance / options
    parser.add_argument("--num-threads", type=int, default=16, help="DuckDB threads (default: 16)")
    parser.add_argument("--prev-cutoff", type=float, default=0.02,
                        help="If predictor=prevalence, skip params with mean true prevalence in first displayed year below this value")
    parser.add_argument("--cases-cutoff", type=float, default=0.1,
                        help="If predictor=cases, skip params with mean true cases < this per 1000 per day in first displayed year")

    # Legacy split-file mode
    parser.add_argument("--split-file", help="Path to train_val_test_split.csv (legacy selection)")
    parser.add_argument("--max-params", type=int, default=None, help="Max unique parameters to plot (legacy mode only)")
    parser.add_argument("--problematic_id", nargs="+", type=int, default=None,
                        help="List of parameter_index values to plot exclusively (legacy mode only)")

    # DB-based selection (preferred)
    parser.add_argument("--param-indices", nargs="+", type=int, help="Explicit list of parameter_index values to fetch from DB")
    parser.add_argument("--param-start", type=int, help="Start of inclusive parameter_index range for DB")
    parser.add_argument("--param-end", type=int, help="End of inclusive parameter_index range for DB")
    parser.add_argument("--param-count", type=int, help="Number of params from --param-start (inclusive)")

    args = parser.parse_args()
    db_param_indices = _resolve_db_param_indices(args)

    visualizer = TestVisualizer(
        config_path=args.config,
        gru_model_path=args.gru_model,
        lstm_model_path=args.lstm_model,
        scaler_path=args.scaler,
        output_pdf=args.output_pdf,
        split_file=args.split_file,
        max_params=args.max_params if (db_param_indices is None and args.problematic_id is None) else None,
        num_threads=args.num_threads,
        selected_params_from_split=args.problematic_id,
        prev_cutoff=args.prev_cutoff,
        cases_cutoff=args.cases_cutoff,
        db_param_indices=db_param_indices
    )

    visualizer.create_plots()

if __name__ == "__main__":
    main()
