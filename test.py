import argparse
import duckdb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import random
import math
import time
from tqdm import tqdm

##############################################################################
# 1. DuckDB Data Fetching
##############################################################################

def fetch_rolling_data(db_path, table_name, window_size, param_limit, sim_limit):
    print("[INFO] Connecting to DuckDB and fetching data...")
    start_time = time.time()

    con = duckdb.connect(db_path, read_only=True)
    con.execute("PRAGMA threads=8;")

    param_where_clause = ""
    if param_limit != "all":
        param_where_clause = f"WHERE parameter_index < {param_limit}"

    distinct_sims_subquery = f"""
        SELECT DISTINCT parameter_index, simulation_index
        FROM {table_name}
        {param_where_clause}
    """

    if sim_limit != "all":
        random_sims_subquery = f"""
            SELECT parameter_index, simulation_index
            FROM (
                SELECT
                    parameter_index,
                    simulation_index,
                    ROW_NUMBER() OVER (
                        PARTITION BY parameter_index
                        ORDER BY RANDOM()
                    ) AS rn
                FROM ({distinct_sims_subquery})
            )
            WHERE rn <= {sim_limit}
        """
    else:
        random_sims_subquery = distinct_sims_subquery

    cte_subquery = f"""
        SELECT
            t.parameter_index,
            t.simulation_index,
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


##############################################################################
# 2. Model Definitions (GRU and LSTM)
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
# 3. Dataset and DataLoader
##############################################################################

class TimeSeriesDataset(Dataset):

    def __init__(self, groups, lookback=30):
        self.samples = []
        self.lookback = lookback
        for g in groups:
            ts = g['time_series']  
            y  = g['targets']      
            T  = g['length']

            for start in range(0, T - lookback + 1):
                end = start + lookback
                self.samples.append({
                    'X': ts[start:end], 
                    'Y': y[start:end],   
                    'param_sim_id': g['param_sim_id']
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        X = item['X']
        Y = item['Y']
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)


def collate_fn(batch):
    Xs = []
    Ys = []
    for (x, y) in batch:
        Xs.append(x)
        Ys.append(y)

    Xs = torch.stack(Xs, dim=0)
    Xs = Xs.permute(1, 0, 2)

    Ys = torch.stack(Ys, dim=0)
    Ys = Ys.permute(1, 0)

    return Xs, Ys

##############################################################################
# 4. Training / Validation
##############################################################################

def train_model(model, train_loader, val_loader, epochs, optimizer, scheduler, device):
    criterion = nn.MSELoss()
    scaler = GradScaler()

    for epoch in range(1, epochs + 1):
        print(f"[INFO] Starting Epoch {epoch}/{epochs}")
        epoch_start_time = time.time()

        model.train()
        total_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"[INFO] Epoch {epoch} - Training", leave=False)

        for X, Y in train_loader_tqdm:
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                pred = model(X).squeeze(-1)
                loss = criterion(pred, Y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        print(f"[INFO] Epoch {epoch} - Average Training Loss: {avg_train_loss:.4f}")

        if scheduler is not None:
            scheduler.step()

        model.eval()
        val_loss = 0.0
        val_loader_tqdm = tqdm(val_loader, desc=f"[INFO] Epoch {epoch} - Validation", leave=False)

        with torch.no_grad():
            for X_val, Y_val in val_loader_tqdm:
                X_val = X_val.to(device, non_blocking=True)
                Y_val = Y_val.to(device, non_blocking=True)

                with autocast(device_type='cuda'):
                    pred_val = model(X_val).squeeze(-1)
                    loss_val = criterion(pred_val, Y_val)

                val_loss += loss_val.item()
                val_loader_tqdm.set_postfix(val_loss=loss_val.item())

        avg_val_loss = val_loss / len(val_loader)
        epoch_duration = time.time() - epoch_start_time

        print(f"[INFO] Epoch {epoch} Completed - Avg Validation Loss: {avg_val_loss:.4f} | Duration: {epoch_duration:.2f}s")

    print("[INFO] Training Completed")



def predict_full_sequence(model, full_ts, device):
    model.eval()
    with torch.no_grad(), autocast(device_type='cuda'):
        x_torch = torch.tensor(full_ts, dtype=torch.float32).unsqueeze(1).to(device)
        pred = model(x_torch).squeeze(-1).squeeze(-1).cpu().numpy()
    return pred


##############################################################################
# 5. Main Script
##############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", required=True, help="Path to DuckDB database file")
    parser.add_argument("--table-name", default="simulation_results", help="Table name inside DuckDB")
    parser.add_argument("--window-size", default=7, type=int, help="Window size for rolling average")
    parser.add_argument("--param-limit", default="all", help="Maximum parameter_index (exclusive) to include or 'all'")
    parser.add_argument("--sim-limit", default="all", help="Randomly sample this many simulation_index per parameter_index or 'all'")
    parser.add_argument("--epochs", default=20, type=int, help="Number of training epochs")
    parser.add_argument("--learning-rate", default=1e-3, type=float, help="Initial learning rate")
    parser.add_argument("--batch-size", default=512, type=int, help="Batch size for training")
    parser.add_argument("--hidden-size", default=64, type=int, help="Hidden size for RNNs")
    parser.add_argument("--num-layers", default=1, type=int, help="Number of RNN layers")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout probability")
    parser.add_argument("--lookback", default=30, type=int,
                        help="Lookback window (sequence length) for RNN inputs")
    parser.add_argument("--num-workers", default=0, type=int, help="Number of worker processes for DataLoader")
    parser.add_argument("--device", default=None, help="Choose device: 'cuda' or 'cpu'. If None, auto-detect.")
    parser.add_argument("--use-cyclical-time", action="store_true",
                        help="Whether to encode timesteps as sin/cos of day_of_year (mod 365).")
    parser.add_argument("--min-prevalence", default=0.01, type=float,
                        help="Exclude entire param-sim if the average prevalence is below this threshold.")
    args = parser.parse_args()

    df = fetch_rolling_data(
        db_path=args.db_path,
        table_name=args.table_name,
        window_size=args.window_size,
        param_limit=args.param_limit,
        sim_limit=args.sim_limit
    )

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("Using device:", device)

    group_cols = ["parameter_index", "simulation_index"]
    group_means = df.groupby(group_cols)["prevalence"].mean().reset_index()
    valid_groups = group_means[group_means["prevalence"] >= args.min_prevalence]
    valid_keys = set(zip(valid_groups["parameter_index"], valid_groups["simulation_index"]))

    df["param_sim"] = list(zip(df["parameter_index"], df["simulation_index"]))
    df = df[df["param_sim"].isin(valid_keys)]

    param_sim_groups = df.groupby(group_cols)

    static_covars = [
        "eir", "dn0_use", "dn0_future", "Q0", "phi_bednets",
        "seasonal", "routine", "itn_use", "irs_use",
        "itn_future", "irs_future", "lsm"
    ]

    # ----------------------------
    # 70/15/15 split
    # ----------------------------
    all_param_sims = list(param_sim_groups.groups.keys())
    random.shuffle(all_param_sims)

    n_total = len(all_param_sims)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)

    train_param_sims = set(all_param_sims[:n_train])
    val_param_sims   = set(all_param_sims[n_train : n_train + n_val])
    test_param_sims  = set(all_param_sims[n_train + n_val :])

    def build_data_list(param_sims):
        data_list = []
        for ps in param_sims:
            subdf = param_sim_groups.get_group(ps).sort_values("timesteps")
            T = len(subdf)

            static_vals = subdf.iloc[0][static_covars].values.astype(np.float32)

            t = subdf["timesteps"].values.astype(np.float32)

            if args.use_cyclical_time:
                day_of_year = t % 365.0
                sin_t = np.sin(2 * math.pi * day_of_year / 365.0)
                cos_t = np.cos(2 * math.pi * day_of_year / 365.0)
                X = np.zeros((T, 2 + len(static_covars)), dtype=np.float32)
                for i in range(T):
                    X[i, 0] = sin_t[i]
                    X[i, 1] = cos_t[i]
                    X[i, 2:] = static_vals
            else:
                X = np.zeros((T, 1 + len(static_covars)), dtype=np.float32)
                for i in range(T):
                    X[i, 0] = t[i]
                    X[i, 1:] = static_vals

            Y = subdf["prevalence"].values.astype(np.float32)

            data_list.append({
                "time_series": X,  
                "targets": Y,        
                "length": T,
                "param_sim_id": ps
            })
        return data_list

    train_groups = build_data_list(train_param_sims)
    val_groups   = build_data_list(val_param_sims)
    test_groups  = build_data_list(test_param_sims)

    train_dataset = TimeSeriesDataset(train_groups, lookback=args.lookback)
    val_dataset   = TimeSeriesDataset(val_groups,   lookback=args.lookback)

    if args.use_cyclical_time:
        input_size = 2 + len(static_covars)
    else:
        input_size = 1 + len(static_covars)

    use_pinned_mem = (device.type == "cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,   
        persistent_workers=True,
        prefetch_factor=4       
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    hidden_size = args.hidden_size
    output_size = 1
    dropout_prob = args.dropout
    num_layers = args.num_layers
    lr = args.learning_rate

    gru_model = GRUModel(input_size, hidden_size, output_size, dropout_prob, num_layers).to(device)
    lstm_model = LSTMModel(input_size, hidden_size, output_size, dropout_prob, num_layers).to(device)

    gru_optimizer = torch.optim.Adam(gru_model.parameters(), lr=lr)
    lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lr)

    gru_scheduler = torch.optim.lr_scheduler.StepLR(gru_optimizer, step_size=10, gamma=0.5)
    lstm_scheduler = torch.optim.lr_scheduler.StepLR(lstm_optimizer, step_size=10, gamma=0.5)

    print("\n========== Training GRU ==========")
    train_model(gru_model, train_loader, val_loader, args.epochs, gru_optimizer, gru_scheduler, device)

    print("\n========== Training LSTM ==========")
    train_model(lstm_model, train_loader, val_loader, args.epochs, lstm_optimizer, lstm_scheduler, device)

    val_param_indices = df['parameter_index'].unique()
    random.shuffle(val_param_indices)
    subset_for_plot = val_param_indices[:9] if len(val_param_indices) >= 9 else val_param_indices

    df_all_sims = fetch_rolling_data(
        db_path=args.db_path,
        table_name=args.table_name,
        window_size=args.window_size,
        param_limit=args.param_limit,
        sim_limit="all" 
    )

    param_groups = df_all_sims.groupby("parameter_index")

    n_plots = len(subset_for_plot)
    cols = 3
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10), sharex=False, sharey=False)
    axes = axes.flatten()

    for i, param_idx in enumerate(subset_for_plot):
        subdf_param = param_groups.get_group(param_idx)
        sim_groups = subdf_param.groupby("simulation_index")

        ax = axes[i]

        raw_param_index = subdf_param['parameter_index'].iloc[0]

        for sim_idx, sim_df in sim_groups:
            t = sim_df["timesteps"].values.astype(np.float32)
            y_true = sim_df["prevalence"].values.astype(np.float32)
            if sim_idx == list(sim_groups.groups.keys())[0]:
                ax.plot(t, y_true, color="black", alpha=0.5, linewidth=1, linestyle="--", label="True Prevalence")
            else:
                ax.plot(t, y_true, color="black", alpha=0.5, linewidth=1, linestyle="--")

        first_sim_idx = list(sim_groups.groups.keys())[0]
        sim_df = sim_groups.get_group(first_sim_idx).sort_values("timesteps")

        t = sim_df["timesteps"].values.astype(np.float32)
        static_vals = sim_df.iloc[0][static_covars].values.astype(np.float32)
        T = len(sim_df)

        if args.use_cyclical_time:
            day_of_year = t % 365.0
            sin_t = np.sin(2 * np.pi * day_of_year / 365.0)
            cos_t = np.cos(2 * np.pi * day_of_year / 365.0)
            X_full = np.zeros((T, 2 + len(static_covars)), dtype=np.float32)
            for j in range(T):
                X_full[j, 0] = sin_t[j]
                X_full[j, 1] = cos_t[j]
                X_full[j, 2:] = static_vals
        else:
            X_full = np.zeros((T, 1 + len(static_covars)), dtype=np.float32)
            for j in range(T):
                X_full[j, 0] = t[j]
                X_full[j, 1:] = static_vals

        y_gru = predict_full_sequence(gru_model, X_full, device)
        y_lstm = predict_full_sequence(lstm_model, X_full, device)

        ax.plot(t, y_gru, label="GRU", color="red")
        ax.plot(t, y_lstm, label="LSTM", color="blue")

        ax.set_title(f"Raw Parameter Index = {raw_param_index}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Prevalence")
        ax.legend()

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()

# python test.py \
#   --db-path /home/cosmo/net/malaria/Cosmo/segMINT/db.duckdb \
#   --table-name simulation_results \
#   --window-size 14 \
#   --param-limit 1024 \
#   --sim-limit 4 \
#   --lookback 100 \
#   --use-cyclical-time \
#   --epochs 64 \
#   --learning-rate 0.001 \
#   --batch-size 256 \
#   --hidden-size 128 \
#   --num-layers 2 \
#   --dropout 0.05 \
#   --num-workers 8 \
#   --device cuda \
#   --min-prevalence 0.01