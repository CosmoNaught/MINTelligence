"""
Time Series Forecasting with RNN Models (GRU and LSTM)
====================================================
A modular implementation for training and evaluating recurrent neural networks
on time series prevalence data.
"""

import argparse
import json
import logging
import math
import os
import random
import time
from typing import Dict, List, Tuple, Optional, Union, Any

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#######################
# Configuration Class #
#######################

class Config:
    """Configuration class to store all parameters."""
    
    def __init__(self, args: argparse.Namespace):
        """Initialize configuration from command line arguments."""
        # Data parameters
        self.db_path = args.db_path
        self.table_name = args.table_name
        self.window_size = args.window_size
        self.param_limit = args.param_limit
        self.sim_limit = args.sim_limit
        self.min_prevalence = args.min_prevalence
        self.use_cyclical_time = args.use_cyclical_time
        
        # Model parameters
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.lookback = args.lookback
        
        # Training parameters
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.patience = args.patience
        self.num_workers = args.num_workers
        self.device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # File paths
        self.output_dir = args.output_dir
        self.use_existing_split = args.use_existing_split
        self.split_file = args.split_file if args.split_file else os.path.join(args.output_dir, "train_val_test_split.csv")
        
        # Hyperparameter tuning
        self.run_tuning = args.run_tuning
        self.tuning_output_dir = args.tuning_output_dir
        self.tuning_timeout = args.tuning_timeout
        self.tuning_trials = args.tuning_trials
        self.use_tuned_parameters = args.use_tuned_parameters
        
        # Random seed
        self.seed = args.seed
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        if self.run_tuning or self.use_tuned_parameters:
            os.makedirs(self.tuning_output_dir, exist_ok=True)
            
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(vars(self), f, indent=4)
            
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        # Create a namespace object from the dictionary
        args = argparse.Namespace(**config_dict)
        return cls(args)

#######################
# Data Module         #
#######################

class DataModule:
    """Module to handle all data-related operations."""
    
    def __init__(self, config: Config):
        """Initialize the data module with configuration."""
        self.config = config
        self.static_covars = [
            "eir", "dn0_use", "dn0_future", "Q0", "phi_bednets",
            "seasonal", "routine", "itn_use", "irs_use",
            "itn_future", "irs_future", "lsm"
        ]
        self.df = None
        self.static_scaler = None
        self.train_param_sims = None
        self.val_param_sims = None
        self.test_param_sims = None
        self.train_params = None
        self.val_params = None
        self.test_params = None
        self.input_size = None
        
    def fetch_data(self) -> pd.DataFrame:
        """Fetch data from DuckDB database."""
        logger.info(f"Connecting to DuckDB and fetching data from {self.config.db_path}")
        start_time = time.time()

        con = duckdb.connect(self.config.db_path, read_only=True)
        con.execute("PRAGMA threads=8;")

        param_where_clause = ""
        if self.config.param_limit != "all":
            param_where_clause = f"WHERE parameter_index < {self.config.param_limit}"

        distinct_sims_subquery = f"""
            SELECT DISTINCT parameter_index, simulation_index, global_index
            FROM {self.config.table_name}
            {param_where_clause}
        """

        if self.config.sim_limit != "all":
            random_sims_subquery = f"""
                SELECT parameter_index, simulation_index, global_index
                FROM (
                    SELECT
                        parameter_index,
                        simulation_index,
                        global_index,
                        ROW_NUMBER() OVER (
                            PARTITION BY parameter_index
                            ORDER BY RANDOM()
                        ) AS rn
                    FROM ({distinct_sims_subquery})
                )
                WHERE rn <= {self.config.sim_limit}
            """
        else:
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
            FROM {self.config.table_name} t
            JOIN ({random_sims_subquery}) rs
            USING (parameter_index, simulation_index)
        """

        preceding = self.config.window_size - 1
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
              AND (cte.timesteps % {self.config.window_size}) = 0
            ORDER BY parameter_index, simulation_index, timesteps
        """

        df = con.execute(final_query).df()
        con.close()

        logger.info(f"Data fetched in {time.time() - start_time:.2f} seconds.")
        return df
        
    def prepare_data(self) -> None:
        """Prepare data for model training."""
        # 1. Fetch data
        self.df = self.fetch_data()
        
        # 2. Filter data based on prevalence threshold
        self._filter_by_prevalence()
        
        # 3. Create train/val/test split
        self._create_data_split()
        
        # 4. Create and fit feature scalers
        self._create_scalers()
        
        # 5. Determine input size for models
        self._set_input_size()
        
        logger.info("Data preparation completed successfully.")
        
    def _filter_by_prevalence(self) -> None:
        """Filter data based on prevalence threshold."""
        group_cols = ["parameter_index", "simulation_index"]
        group_means = self.df.groupby(group_cols)["prevalence"].mean().reset_index()
        valid_groups = group_means[group_means["prevalence"] >= self.config.min_prevalence]
        valid_keys = set(zip(valid_groups["parameter_index"], valid_groups["simulation_index"]))

        self.df["param_sim"] = list(zip(self.df["parameter_index"], self.df["simulation_index"]))
        self.df = self.df[self.df["param_sim"].isin(valid_keys)]
        
        logger.info(f"Filtered data to {len(valid_keys)} parameter-simulation pairs with prevalence >= {self.config.min_prevalence}")
        
    def _create_data_split(self) -> None:
        """Create or load train/validation/test split."""
        # Create a mapping from param_sim to global_index
        param_sim_to_global = {}
        for _, row in self.df.iterrows():
            param_idx = row["parameter_index"]
            sim_idx = row["simulation_index"]
            global_idx = row["global_index"]
            param_sim_to_global[(param_idx, sim_idx)] = global_idx
        
        # Load existing split if available
        if self.config.use_existing_split and os.path.exists(self.config.split_file):
            logger.info(f"Loading existing train/val/test split from {self.config.split_file}")
            self._load_existing_split()
        else:
            # Create new split
            logger.info("Creating new train/val/test split (70/15/15)")
            self._create_new_split(param_sim_to_global)
            
        logger.info(f"Data split: {len(self.train_param_sims)} train, {len(self.val_param_sims)} validation, {len(self.test_param_sims)} test parameter-simulation pairs")
        logger.info(f"Number of unique parameters: {len(self.train_params)} train, {len(self.val_params)} validation, {len(self.test_params)} test")
        
    def _load_existing_split(self) -> None:
        """Load existing train/validation/test split."""
        split_df = pd.read_csv(self.config.split_file)
        
        # Create sets for train/val/test params
        self.train_params = set(split_df[split_df["split"] == "train"]["parameter_index"])
        self.val_params = set(split_df[split_df["split"] == "validate"]["parameter_index"])
        self.test_params = set(split_df[split_df["split"] == "test"]["parameter_index"])
        
        # Create param_sim pairs
        self.train_param_sims = set()
        self.val_param_sims = set()
        self.test_param_sims = set()
        
        for _, row in split_df.iterrows():
            param_idx = row["parameter_index"]
            sim_idx = row["simulation_index"]
            split = row["split"]
            
            if split == "train":
                self.train_param_sims.add((param_idx, sim_idx))
            elif split == "validate":
                self.val_param_sims.add((param_idx, sim_idx))
            elif split == "test":
                self.test_param_sims.add((param_idx, sim_idx))
                
    def _create_new_split(self, param_sim_to_global: Dict) -> None:
        """Create new train/validation/test split."""
        param_sim_groups = self.df.groupby(["parameter_index", "simulation_index"])
        all_param_sims = list(param_sim_groups.groups.keys())
        
        # Split data by parameter index first
        unique_parameters = list(set([ps[0] for ps in all_param_sims]))
        random.shuffle(unique_parameters)

        n_params = len(unique_parameters)
        n_train_params = int(0.7 * n_params)
        n_val_params = int(0.15 * n_params)

        self.train_params = set(unique_parameters[:n_train_params])
        self.val_params = set(unique_parameters[n_train_params : n_train_params + n_val_params])
        self.test_params = set(unique_parameters[n_train_params + n_val_params :])

        # Assign all parameter-simulation pairs to their respective splits
        self.train_param_sims = set(ps for ps in all_param_sims if ps[0] in self.train_params)
        self.val_param_sims = set(ps for ps in all_param_sims if ps[0] in self.val_params)
        self.test_param_sims = set(ps for ps in all_param_sims if ps[0] in self.test_params)

        # Create a DataFrame to store the split information
        split_info = []
        for ps in all_param_sims:
            param_idx, sim_idx = ps
            global_idx = param_sim_to_global.get((param_idx, sim_idx), None)
            
            if param_idx in self.train_params:
                split = 'train'
            elif param_idx in self.val_params:
                split = 'validate'
            else:
                split = 'test'
            
            split_info.append({
                'parameter_index': param_idx,
                'simulation_index': sim_idx,
                'global_index': global_idx,
                'split': split
            })

        split_df = pd.DataFrame(split_info)
        split_df.to_csv(self.config.split_file, index=False)
        logger.info(f"Saved train/validation/test split information to {self.config.split_file}")
        
    def _create_scalers(self) -> None:
        """Create and fit feature scalers."""
        static_scalar_values = np.array([row[self.static_covars].values.astype(np.float32) 
                                         for _, row in self.df.iterrows()])
        self.static_scaler = StandardScaler()
        self.static_scaler.fit(static_scalar_values)
        
        # Save scaler for later use in predictions
        scaler_path = os.path.join(self.config.output_dir, "static_scaler.pkl")
        pd.to_pickle(self.static_scaler, scaler_path)
        logger.info(f"Feature scaler saved to {scaler_path}")
        
    def _set_input_size(self) -> None:
        """Determine input size for models based on feature encoding."""
        if self.config.use_cyclical_time:
            self.input_size = 2 + len(self.static_covars)  # sin, cos, static features
        else:
            self.input_size = 1 + len(self.static_covars)  # time, static features
        logger.info(f"Input size for models set to {self.input_size}")
            
    def build_data_list(self, param_sims: set) -> List[Dict]:
        """Build normalized data list for a set of parameter-simulation pairs."""
        param_sim_groups = self.df.groupby(["parameter_index", "simulation_index"])
        data_list = []
        for ps in param_sims:
            subdf = param_sim_groups.get_group(ps).sort_values("timesteps")
            T = len(subdf)

            # Get and scale static values
            static_vals = subdf.iloc[0][self.static_covars].values.astype(np.float32)
            static_vals = self.static_scaler.transform(static_vals.reshape(1, -1)).flatten()

            t = subdf["timesteps"].values.astype(np.float32)

            if self.config.use_cyclical_time:
                day_of_year = t % 365.0
                sin_t = np.sin(2 * math.pi * day_of_year / 365.0)
                cos_t = np.cos(2 * math.pi * day_of_year / 365.0)
                X = np.zeros((T, 2 + len(self.static_covars)), dtype=np.float32)
                for i in range(T):
                    X[i, 0] = sin_t[i]
                    X[i, 1] = cos_t[i]
                    X[i, 2:] = static_vals
            else:
                # Normalize timesteps
                t_min, t_max = np.min(t), np.max(t)
                t_norm = (t - t_min) / (t_max - t_min) if t_max > t_min else t
                
                X = np.zeros((T, 1 + len(self.static_covars)), dtype=np.float32)
                for i in range(T):
                    X[i, 0] = t_norm[i]
                    X[i, 1:] = static_vals

            Y = subdf["prevalence"].values.astype(np.float32)

            data_list.append({
                "time_series": X,  
                "targets": Y,        
                "length": T,
                "param_sim_id": ps
            })
        return data_list
        
    def create_datasets(self, lookback: int) -> Dict[str, Dataset]:
        """Create datasets for train, validation, and test sets."""
        # Build datasets with specified lookback
        train_groups = self.build_data_list(self.train_param_sims)
        val_groups = self.build_data_list(self.val_param_sims)
        test_groups = self.build_data_list(self.test_param_sims)

        train_dataset = TimeSeriesDataset(train_groups, lookback=lookback)
        val_dataset = TimeSeriesDataset(val_groups, lookback=lookback)
        test_dataset = TimeSeriesDataset(test_groups, lookback=lookback)
        
        return {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset
        }
        
    def create_dataloaders(self, datasets: Dict[str, Dataset], batch_size: int) -> Dict[str, DataLoader]:
        """Create data loaders from datasets."""
        train_loader = DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True,   
            persistent_workers=True if self.config.num_workers > 0 else False,
            prefetch_factor=4 if self.config.num_workers > 0 else None
        )

        val_loader = DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True if self.config.num_workers > 0 else False,
            prefetch_factor=4 if self.config.num_workers > 0 else None
        )
        
        test_loader = DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True if self.config.num_workers > 0 else False,
            prefetch_factor=4 if self.config.num_workers > 0 else None
        )
        
        return {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader
        }

###########################
# Dataset and DataLoader  #
###########################

class TimeSeriesDataset(Dataset):
    """Dataset for time series data."""
    
    def __init__(self, groups: List[Dict], lookback: int = 30):
        """Initialize time series dataset."""
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

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        item = self.samples[idx]
        X = item['X']
        Y = item['Y']
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Custom collate function for batching time series data."""
    Xs = []
    Ys = []
    for (x, y) in batch:
        Xs.append(x)
        Ys.append(y)

    Xs = torch.stack(Xs, dim=0)
    Xs = Xs.permute(1, 0, 2)  # (time, batch, features)

    Ys = torch.stack(Ys, dim=0)
    Ys = Ys.permute(1, 0)  # (time, batch)

    return Xs, Ys

#######################
# Model Module        #
#######################

class ModelFactory:
    """Factory class to create different types of models."""
    
    @staticmethod
    def create_model(model_type: str, input_size: int, hidden_size: int, output_size: int, 
                     dropout_prob: float, num_layers: int = 1) -> nn.Module:
        """Create a model of the specified type."""
        if model_type.lower() == "gru":
            return GRUModel(input_size, hidden_size, output_size, dropout_prob, num_layers)
        elif model_type.lower() == "lstm":
            return LSTMModel(input_size, hidden_size, output_size, dropout_prob, num_layers)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class GRUModel(nn.Module):
    """GRU Model for time series prediction."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 dropout_prob: float, num_layers: int = 1):
        """Initialize GRU model."""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        out, _ = self.gru(x)
        out = self.ln(out)
        out = self.dropout(out)
        out = self.fc(out)   
        out = self.activation(out)
        return out


class LSTMModel(nn.Module):
    """LSTM Model for time series prediction."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 dropout_prob: float, num_layers: int = 1):
        """Initialize LSTM model."""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        out, _ = self.lstm(x)  
        out = self.ln(out)
        out = self.dropout(out)
        out = self.fc(out) 
        out = self.activation(out)
        return out

#######################
# Training Module     #
#######################

class Trainer:
    """Trainer class for training and evaluating models."""
    
    def __init__(self, model: nn.Module, config: Config, output_dir: str, model_name: str):
        """Initialize trainer with model and configuration."""
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.output_dir = output_dir
        self.model_name = model_name
        self.criterion = nn.MSELoss()
        self.model.to(self.device)
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              optimizer: torch.optim.Optimizer, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
              epochs: Optional[int] = None) -> Tuple[nn.Module, float, int]:
        """Train model with early stopping and best model checkpointing."""
        epochs = epochs or self.config.epochs
        scaler = GradScaler()
        
        # Initialize best validation loss and patience counter for early stopping
        best_val_loss = float('inf')
        patience = self.config.patience
        patience_counter = 0
        best_epoch = 0
        
        # Path to save best model
        best_model_path = os.path.join(self.output_dir, f"{self.model_name}_best.pt")
        training_history = {'train_loss': [], 'val_loss': [], 'epochs': []}
        
        for epoch in range(1, epochs + 1):
            logger.info(f"Starting Epoch {epoch}/{epochs}")
            epoch_start_time = time.time()

            # Training phase
            self.model.train()
            total_train_loss = 0.0
            train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch} - Training", leave=False)

            for X, Y in train_loader_tqdm:
                X = X.to(self.device, non_blocking=True)
                Y = Y.to(self.device, non_blocking=True)

                optimizer.zero_grad()

                with autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
                    pred = self.model(X).squeeze(-1)
                    loss = self.criterion(pred, Y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_train_loss += loss.item()
                train_loader_tqdm.set_postfix(loss=loss.item())

            avg_train_loss = total_train_loss / len(train_loader)
            logger.info(f"Epoch {epoch} - Average Training Loss: {avg_train_loss:.6f}")

            # Validation phase
            self.model.eval()
            total_val_loss = 0.0
            val_predictions = []
            val_targets = []
            val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch} - Validation", leave=False)

            with torch.no_grad():
                for X_val, Y_val in val_loader_tqdm:
                    X_val = X_val.to(self.device, non_blocking=True)
                    Y_val = Y_val.to(self.device, non_blocking=True)

                    with autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
                        pred_val = self.model(X_val).squeeze(-1)
                        loss_val = self.criterion(pred_val, Y_val)

                    total_val_loss += loss_val.item()
                    val_loader_tqdm.set_postfix(val_loss=loss_val.item())
                    
                    # Collect predictions for additional metrics
                    val_predictions.append(pred_val.cpu().numpy())
                    val_targets.append(Y_val.cpu().numpy())

            avg_val_loss = total_val_loss / len(val_loader)
            epoch_duration = time.time() - epoch_start_time
            
            # Record training history
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(avg_val_loss)
            training_history['epochs'].append(epoch)

            # Apply learning rate scheduler if provided
            if scheduler is not None:
                # For ReduceLROnPlateau, pass the validation loss
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()

            # Check if this is the best model so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                patience_counter = 0
                
                # Save the best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'train_loss': avg_train_loss
                }, best_model_path)
                
                logger.info(f"New best model saved at epoch {epoch} with validation loss: {avg_val_loss:.6f}")
            else:
                patience_counter += 1
                logger.info(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")

            logger.info(f"Epoch {epoch} Completed - Avg Validation Loss: {avg_val_loss:.6f} | Duration: {epoch_duration:.2f}s")
            
            # Early stopping check
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs. Best epoch was {best_epoch} with validation loss: {best_val_loss:.6f}")
                break
        
        # Save training history
        history_path = os.path.join(self.output_dir, f"{self.model_name}_training_history.json")
        with open(history_path, 'w') as f:
            json.dump(training_history, f)
        
        logger.info(f"Training completed. Best model was at epoch {best_epoch} with validation loss: {best_val_loss:.6f}")
        
        # Load the best model for return
        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Also save a final model (in case we want both final and best)
        final_model_path = os.path.join(self.output_dir, f"{self.model_name}_final.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'train_loss': avg_train_loss,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss
        }, final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
        return self.model, best_val_loss, best_epoch

    def evaluate(self, data_loader: DataLoader, dataset_name: str = "Test") -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """Evaluate model on a dataset with comprehensive metrics."""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        
        # Evaluate with no gradients
        with torch.no_grad():
            for X, Y in tqdm(data_loader, desc=f"Evaluating on {dataset_name} Set"):
                X = X.to(self.device, non_blocking=True)
                Y = Y.to(self.device, non_blocking=True)
                
                with autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
                    pred = self.model(X).squeeze(-1)
                    loss = self.criterion(pred, Y)
                
                total_loss += loss.item()
                all_predictions.append(pred.cpu().numpy())
                all_targets.append(Y.cpu().numpy())
        
        # Concatenate predictions and targets
        all_predictions = np.concatenate([p.flatten() for p in all_predictions])
        all_targets = np.concatenate([t.flatten() for t in all_targets])
        
        # Calculate standard metrics
        mse = mean_squared_error(all_targets, all_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)
        
        # Calculate bounded metrics specific to prevalence data (0-1 range)
        # Symmetric Mean Absolute Percentage Error (bounded version)
        epsilon = 1e-7  # To avoid division by zero
        smape = 100 * np.mean(2 * np.abs(all_predictions - all_targets) / 
                             (np.abs(all_predictions) + np.abs(all_targets) + epsilon))
        
        # Calculate bias
        bias = np.mean(all_predictions - all_targets)
        
        # Log likelihood of Beta distribution (approximation for bounded variables)
        scaled_pred = np.clip(all_predictions, epsilon, 1-epsilon)
        scaled_targets = np.clip(all_targets, epsilon, 1-epsilon)
        log_likelihood = np.mean(np.log(scaled_pred) * scaled_targets + 
                               np.log(1 - scaled_pred) * (1 - scaled_targets))
        
        # Create results dictionary
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'smape': smape,
            'bias': bias,
            'log_likelihood': log_likelihood,
            'avg_loss': total_loss / len(data_loader)
        }
        
        # Print metrics
        logger.info(f"\n{dataset_name} Set Evaluation Results:")
        logger.info(f"    MSE:  {mse:.6f}")
        logger.info(f"    RMSE: {rmse:.6f}")
        logger.info(f"    MAE:  {mae:.6f}")
        logger.info(f"    R²:   {r2:.6f}")
        logger.info(f"    SMAPE: {smape:.2f}%")
        logger.info(f"    Bias: {bias:.6f}")
        logger.info(f"    Log-Likelihood: {log_likelihood:.6f}")
        
        return metrics, all_predictions, all_targets
    
    def predict_sequence(self, full_ts: np.ndarray) -> np.ndarray:
        """Predict a full time series sequence."""
        self.model.eval()
        with torch.no_grad(), autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu'):
            x_torch = torch.tensor(full_ts, dtype=torch.float32).unsqueeze(1).to(self.device)
            pred = self.model(x_torch).squeeze(-1).squeeze(-1).cpu().numpy()
        return pred
    
#######################
# Hyperparameter Opt  #
#######################

class HyperparameterOptimizer:
    """Class for hyperparameter optimization using Optuna."""
    
    def __init__(self, config: Config, data_module: DataModule):
        """Initialize hyperparameter optimizer with configuration and data module."""
        self.config = config
        self.data_module = data_module
        
    def objective_for_model(self, trial: optuna.Trial, model_type: str) -> float:
        """Objective function for Optuna optimization - Model-specific version."""
        # Sample hyperparameters
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512, 1024])
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256])
        num_layers = trial.suggest_int("num_layers", 1, 4)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lookback = trial.suggest_categorical("lookback", [13, 26, 52, 78])
        
        # Set device
        device = torch.device(self.config.device)
            
        # Print hyperparameters for this trial
        logger.info(f"\nTrial #{trial.number} for {model_type.upper()} with hyperparameters:")
        logger.info(f"    Learning Rate: {lr:.6f}")
        logger.info(f"    Batch Size: {batch_size}")
        logger.info(f"    Hidden Size: {hidden_size}")
        logger.info(f"    Number of Layers: {num_layers}")
        logger.info(f"    Dropout: {dropout:.2f}")
        logger.info(f"    Lookback: {lookback}")
        
        # Define output size
        output_size = 1
        
        # Build datasets and create data loaders with current lookback
        datasets = self.data_module.create_datasets(lookback)
        train_dataset = datasets["train"]
        val_dataset = datasets["val"]
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True if self.config.num_workers > 0 else False,
            prefetch_factor=4 if self.config.num_workers > 0 else None
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True if self.config.num_workers > 0 else False,
            prefetch_factor=4 if self.config.num_workers > 0 else None
        )
        
        # Initialize model
        model = ModelFactory.create_model(
            model_type, 
            self.data_module.input_size, 
            hidden_size, 
            output_size, 
            dropout, 
            num_layers
        ).to(device)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Initialize scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Create temporary directory for trial outputs
        trial_dir = os.path.join(self.config.tuning_output_dir, f"{model_type}_trial_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)
        
        # Set the number of epochs for hyperparameter tuning to be shorter than the default
        tuning_epochs = min(self.config.epochs, 16)  # Cap at 16 epochs to speed up tuning
        
        # Train model
        try:
            trainer = Trainer(model, self.config, trial_dir, model_type)
            model, best_val_loss, best_epoch = trainer.train(
                train_loader, val_loader, optimizer, scheduler, tuning_epochs
            )
            
            # Save trial results
            trial_results = {
                'trial_number': trial.number,
                'model_type': model_type,
                'hyperparameters': {
                    'learning_rate': lr,
                    'batch_size': batch_size,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'lookback': lookback
                },
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch
            }
            
            with open(os.path.join(trial_dir, f"{model_type}_trial_results.json"), 'w') as f:
                json.dump(convert_to_json_serializable(trial_results), f, indent=4)
            
            return best_val_loss
        
        except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError) as e:
            # Handle errors (like OOM)
            logger.error(f"Trial failed due to: {str(e)}")
            return float('inf')  # Return a bad score
            
    def run_optimization(self) -> Dict[str, Dict[str, Any]]:
        """Run hyperparameter optimization for both GRU and LSTM models."""
        logger.info("Starting hyperparameter optimization with Optuna for both GRU and LSTM models")
        
        # Dictionary to store best parameters for each model
        best_params = {}
        
        # Run optimization for each model type separately
        for model_type in ["gru", "lstm"]:
            logger.info(f"\nStarting optimization for {model_type.upper()} model")
            
            # Create study name based on dataset and model configuration
            study_name = f"{model_type}_optimization_{int(time.time())}"
            
            # Create sampler with seed for reproducibility
            sampler = TPESampler(seed=self.config.seed)
            
            # Create pruner to terminate unpromising trials early
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
            
            # Create Optuna study with in-memory storage
            study = optuna.create_study(
                study_name=study_name,
                direction="minimize",
                sampler=sampler,
                pruner=pruner
            )
            
            # Run optimization
            study.optimize(
                lambda trial: self.objective_for_model(trial, model_type),
                n_trials=self.config.tuning_trials,
                timeout=self.config.tuning_timeout // 2,  # Split timeout between the two models
                gc_after_trial=True,
                show_progress_bar=True
            )
            
            # Get best parameters
            model_best_params = study.best_params
            model_best_value = study.best_value
            logger.info(f"\nBest hyperparameters for {model_type.upper()}:")
            for param, value in model_best_params.items():
                logger.info(f"    {param}: {value}")
            logger.info(f"Best validation loss: {model_best_value:.6f}")
            
            # Save best parameters and full study results
            model_tuning_results = {
                'model_type': model_type,
                'best_params': model_best_params,
                'best_validation_loss': model_best_value,
                'study_name': study_name,
                'n_trials': len(study.trials),
                'optimization_time': time.time(),
                'trials': [
                    {
                        'number': t.number,
                        'params': t.params,
                        'value': t.value,
                        'state': str(t.state),
                        'datetime_start': str(t.datetime_start),
                        'datetime_complete': str(t.datetime_complete)
                    }
                    for t in study.trials
                ]
            }
            
            # Store best parameters for this model
            best_params[model_type] = model_best_params
            
            # Create model-specific directory
            model_tuning_dir = os.path.join(self.config.tuning_output_dir, model_type)
            os.makedirs(model_tuning_dir, exist_ok=True)
            
            # Save results to JSON file
            model_best_params_path = os.path.join(model_tuning_dir, "best_params.json")
            with open(model_best_params_path, 'w') as f:
                json.dump(convert_to_json_serializable(model_tuning_results), f, indent=4)
            logger.info(f"Best hyperparameters for {model_type.upper()} saved to {model_best_params_path}")
            
            # Plot optimization history
            try:
                history_fig = optuna.visualization.plot_optimization_history(study)
                history_fig.write_image(os.path.join(model_tuning_dir, "optimization_history.png"))
                param_importances = optuna.visualization.plot_param_importances(study)
                param_importances.write_image(os.path.join(model_tuning_dir, "param_importances.png"))
                parallel_coordinate = optuna.visualization.plot_parallel_coordinate(study)
                parallel_coordinate.write_image(os.path.join(model_tuning_dir, "parallel_coordinate.png"))
            except Exception as e:
                logger.warning(f"Could not generate visualization plots: {str(e)}")
        
        # Save combined best parameters
        combined_best_params_path = os.path.join(self.config.tuning_output_dir, "best_params.json")
        with open(combined_best_params_path, 'w') as f:
            json.dump(convert_to_json_serializable({"gru": best_params["gru"], "lstm": best_params["lstm"]}), f, indent=4)
        logger.info(f"Combined best hyperparameters saved to {combined_best_params_path}")
        
        return best_params

    @staticmethod
    def load_best_params(tuning_output_dir: str) -> Optional[Dict[str, Dict[str, Any]]]:
        """Load best hyperparameters from previous tuning runs."""
        best_params = {}
        
        # Check for combined best params file first
        combined_params_path = os.path.join(tuning_output_dir, "best_params.json")
        if os.path.exists(combined_params_path):
            with open(combined_params_path, 'r') as f:
                combined_params = json.load(f)
                if "gru" in combined_params and "lstm" in combined_params:
                    return combined_params
        
        # If combined file doesn't exist or is incomplete, try model-specific files
        for model_type in ["gru", "lstm"]:
            model_params_path = os.path.join(tuning_output_dir, model_type, "best_params.json")
            if os.path.exists(model_params_path):
                with open(model_params_path, 'r') as f:
                    tuning_results = json.load(f)
                    best_params[model_type] = tuning_results.get('best_params', None)
            else:
                logger.warning(f"No tuned parameters found for {model_type.upper()} model")
                best_params[model_type] = None
        
        # Return the parameters we found, or None if we didn't find any
        if best_params.get("gru") is not None or best_params.get("lstm") is not None:
            return best_params
        else:
            return None

#######################
# Visualization       #
#######################

class Visualizer:
    """Class for visualizing model results."""
    
    def __init__(self, config: Config, data_module: DataModule):
        """Initialize visualizer with configuration and data module."""
        self.config = config
        self.data_module = data_module
        
    def plot_training_history(self, output_dir: str, model_results: Dict[str, Dict[str, Any]]) -> None:
        """Plot combined training history for GRU and LSTM models."""
        gru_history_path = os.path.join(output_dir, "gru_training_history.json")
        lstm_history_path = os.path.join(output_dir, "lstm_training_history.json")
        
        if os.path.exists(gru_history_path) and os.path.exists(lstm_history_path):
            with open(gru_history_path, 'r') as f:
                gru_history = json.load(f)
            
            with open(lstm_history_path, 'r') as f:
                lstm_history = json.load(f)
            
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.plot(gru_history['epochs'], gru_history['train_loss'], label='Train Loss')
            plt.plot(gru_history['epochs'], gru_history['val_loss'], label='Validation Loss')
            plt.axvline(x=model_results["gru"]["best_epoch"], color='r', linestyle='--', alpha=0.5, 
                       label=f'Best Epoch ({model_results["gru"]["best_epoch"]})')
            plt.title('GRU Training History')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.plot(lstm_history['epochs'], lstm_history['train_loss'], label='Train Loss')
            plt.plot(lstm_history['epochs'], lstm_history['val_loss'], label='Validation Loss')
            plt.axvline(x=model_results["lstm"]["best_epoch"], color='r', linestyle='--', alpha=0.5, 
                       label=f'Best Epoch ({model_results["lstm"]["best_epoch"]})')
            plt.title('LSTM Training History')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir, "training_history.png")
            plt.savefig(plot_path)
            logger.info(f"Training history plot saved to {plot_path}")
            plt.close()
            
    def plot_model_comparison(self, output_dir: str, model_metrics: Dict[str, Dict[str, Any]]) -> None:
        """Plot comparison of model metrics."""
        plt.figure(figsize=(10, 6))
        
        metrics_to_plot = ['mse', 'rmse', 'mae', 'r2']
        x = np.arange(len(metrics_to_plot))
        width = 0.35
        
        gru_values = [model_metrics["gru"]["test_metrics"][m] for m in metrics_to_plot]
        lstm_values = [model_metrics["lstm"]["test_metrics"][m] for m in metrics_to_plot]
        
        plt.bar(x - width/2, gru_values, width, label='GRU')
        plt.bar(x + width/2, lstm_values, width, label='LSTM')
        
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.title('Model Performance Comparison on Test Set')
        plt.xticks(x, metrics_to_plot)
        plt.legend()
        plt.grid(alpha=0.3)
        
        plot_path = os.path.join(output_dir, "model_comparison.png")
        plt.savefig(plot_path)
        logger.info(f"Model comparison plot saved to {plot_path}")
        plt.close()
            
    def plot_test_predictions(self, output_dir: str, models: Dict[str, nn.Module], trainers: Dict[str, Trainer]) -> None:
        """Visualize test set predictions."""
        # Fetch test data for visualization
        df_test_sims = self.data_module.fetch_data()
        
        # Get test parameters
        test_param_indices = list(self.data_module.test_params)
        random.shuffle(test_param_indices)
        subset_for_plot = test_param_indices[:9] if len(test_param_indices) >= 9 else test_param_indices
        
        # Filter to only include chosen test parameters
        df_test_sims = df_test_sims[df_test_sims["parameter_index"].isin(subset_for_plot)]
        param_groups = df_test_sims.groupby("parameter_index")

        n_plots = len(subset_for_plot)
        cols = 3
        rows = (n_plots + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 10), sharex=False, sharey=False)
        axes = axes.flatten() if n_plots > 1 else [axes]

        # Create a list to store all plot data
        all_plot_data = []

        for i, param_idx in enumerate(subset_for_plot):
            if param_idx not in param_groups.groups:
                logger.warning(f"Parameter index {param_idx} not found in test data, skipping")
                continue
                
            subdf_param = param_groups.get_group(param_idx)
            sim_groups = subdf_param.groupby("simulation_index")
            
            if len(sim_groups) == 0:
                logger.warning(f"No simulation data for parameter {param_idx}, skipping")
                continue

            ax = axes[i]
            raw_param_index = subdf_param['parameter_index'].iloc[0]

            # Plot all simulations for this parameter
            for sim_idx, sim_df in sim_groups:
                # Check if this exact parameter-sim combination was in test set
                if (param_idx, sim_idx) in self.data_module.test_param_sims:
                    linestyle = "-"  # Solid line for test sims
                    alpha = 0.7
                    label = "True Prevalence (Test)" if sim_idx == list(sim_groups.groups.keys())[0] else None
                else:
                    linestyle = "--"  # Dashed line for other sims
                    alpha = 0.3
                    label = "True Prevalence (Other)" if sim_idx == list(sim_groups.groups.keys())[0] else None
                    
                t = sim_df["timesteps"].values.astype(np.float32)
                y_true = sim_df["prevalence"].values.astype(np.float32)
                ax.plot(t, y_true, color="black", alpha=alpha, linewidth=1, linestyle=linestyle, label=label)

            # For prediction, use only a test simulation
            valid_sim_indices = [sim_idx for sim_idx in sim_groups.groups.keys() 
                                if (param_idx, sim_idx) in self.data_module.test_param_sims]
            
            if not valid_sim_indices:
                logger.warning(f"No test simulations found for parameter {param_idx}, using first available sim")
                first_sim_idx = list(sim_groups.groups.keys())[0]
            else:
                first_sim_idx = valid_sim_indices[0]
                
            sim_df = sim_groups.get_group(first_sim_idx).sort_values("timesteps")
            
            # Get the global index for this param-sim
            global_idx = sim_df['global_index'].iloc[0]

            t = sim_df["timesteps"].values.astype(np.float32)
            static_vals = sim_df.iloc[0][self.data_module.static_covars].values.astype(np.float32)
            # Normalize static values
            static_vals = self.data_module.static_scaler.transform(static_vals.reshape(1, -1)).flatten()
            
            T = len(sim_df)

            if self.config.use_cyclical_time:
                day_of_year = t % 365.0
                sin_t = np.sin(2 * np.pi * day_of_year / 365.0)
                cos_t = np.cos(2 * np.pi * day_of_year / 365.0)
                X_full = np.zeros((T, 2 + len(self.data_module.static_covars)), dtype=np.float32)
                for j in range(T):
                    X_full[j, 0] = sin_t[j]
                    X_full[j, 1] = cos_t[j]
                    X_full[j, 2:] = static_vals
            else:
                # Normalize timesteps
                t_min, t_max = np.min(t), np.max(t)
                t_norm = (t - t_min) / (t_max - t_min) if t_max > t_min else t
                
                X_full = np.zeros((T, 1 + len(self.data_module.static_covars)), dtype=np.float32)
                for j in range(T):
                    X_full[j, 0] = t_norm[j]
                    X_full[j, 1:] = static_vals

            # Create base entry for plot data
            plot_data_entry = {
                'parameter_index': raw_param_index,
                'simulation_index': first_sim_idx,
                'global_index': global_idx,
                'is_test': (param_idx, first_sim_idx) in self.data_module.test_param_sims
            }

            # Make predictions with both models
            for model_type in ["gru", "lstm"]:
                y_pred = trainers[model_type].predict_sequence(X_full)
                
                # Plot predictions
                color = "red" if model_type == "gru" else "blue"
                ax.plot(t, y_pred, label=model_type.upper(), color=color)
                
                # Add to plot data
                for j in range(len(t)):
                    if j == 0:  # First iteration, create entry
                        if 'timestep' not in plot_data_entry:
                            plot_data_entry['timestep'] = t[j]
                            plot_data_entry['true_prevalence'] = sim_df["prevalence"].values[j]
                        plot_data_entry[f'{model_type}_prediction'] = y_pred[j]
                    else:  # Subsequent iterations, update existing entry
                        entry_copy = plot_data_entry.copy()
                        entry_copy['timestep'] = t[j]
                        entry_copy['true_prevalence'] = sim_df["prevalence"].values[j]
                        entry_copy[f'{model_type}_prediction'] = y_pred[j]
                        all_plot_data.append(entry_copy)
            
            # Only add the first entry once (since we've already added the rest in the loop)
            all_plot_data.append(plot_data_entry)
            
            test_status = "(Test)" if param_idx in self.data_module.test_params else "(Non-Test)"
            ax.set_title(f"Parameter Index = {raw_param_index} {test_status}")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Prevalence")
            ax.legend()

        # For any empty subplots, hide them
        for i in range(len(subset_for_plot), len(axes)):
            axes[i].axis('off')

        # Save the plot
        plot_path = os.path.join(output_dir, "test_predictions.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        logger.info(f"Saved test plot to {plot_path}")
        plt.close()

        # Save the plot data to CSV
        plot_data_df = pd.DataFrame(all_plot_data)
        csv_path = os.path.join(output_dir, "test_plot_data.csv")
        plot_data_df.to_csv(csv_path, index=False)
        logger.info(f"Saved test plot data to {csv_path}")

#######################
# Utility Functions   #
#######################

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed} for reproducibility")

def convert_to_json_serializable(obj: Any) -> Any:
    """Convert NumPy types to standard Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    return obj

#######################
# Main Script         #
#######################

def get_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(description="Time Series Forecasting with RNN Models")
    
    # Data parameters
    parser.add_argument("--db-path", required=True, help="Path to DuckDB database file")
    parser.add_argument("--table-name", default="simulation_results", help="Table name inside DuckDB")
    parser.add_argument("--window-size", default=7, type=int, help="Window size for rolling average")
    parser.add_argument("--param-limit", default="all", help="Maximum parameter_index (exclusive) to include or 'all'")
    parser.add_argument("--sim-limit", default="all", help="Randomly sample this many simulation_index per parameter_index or 'all'")
    parser.add_argument("--min-prevalence", default=0.01, type=float,
                        help="Exclude entire param-sim if the average prevalence is below this threshold.")
    parser.add_argument("--use-cyclical-time", action="store_true",
                        help="Whether to encode timesteps as sin/cos of day_of_year (mod 365).")
    
    # Model parameters
    parser.add_argument("--hidden-size", default=64, type=int, help="Hidden size for RNNs")
    parser.add_argument("--num-layers", default=1, type=int, help="Number of RNN layers")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout probability")
    parser.add_argument("--lookback", default=30, type=int,
                        help="Lookback window (sequence length) for RNN inputs")
    
    # Training parameters
    parser.add_argument("--epochs", default=100, type=int, help="Maximum number of training epochs")
    parser.add_argument("--learning-rate", default=1e-3, type=float, help="Initial learning rate")
    parser.add_argument("--batch-size", default=512, type=int, help="Batch size for training")
    parser.add_argument("--num-workers", default=0, type=int, help="Number of worker processes for DataLoader")
    parser.add_argument("--device", default=None, help="Choose device: 'cuda' or 'cpu'. If None, auto-detect.")
    parser.add_argument("--patience", default=16, type=int, help="Patience for early stopping")
    
    # File paths
    parser.add_argument("--output-dir", default="results", help="Directory to save model checkpoints and plot data")
    parser.add_argument("--use-existing-split", action="store_true", help="Use existing train/val/test split from CSV file")
    parser.add_argument("--split-file", default=None, help="Path to existing train/val/test split CSV file")
    
    # Hyperparameter tuning
    parser.add_argument("--run-tuning", action="store_true", help="Run hyperparameter tuning with Optuna")
    parser.add_argument("--tuning-output-dir", default="results_tuned", help="Directory to save tuning results")
    parser.add_argument("--tuning-timeout", type=int, default=3600*12, help="Timeout for tuning in seconds (default: 12 hours)")
    parser.add_argument("--tuning-trials", type=int, default=16, help="Number of trials for hyperparameter tuning")
    parser.add_argument("--use-tuned-parameters", action="store_true", 
                        help="Use previously tuned parameters instead of defaults or command line arguments")
    
    # Miscellaneous
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")
    
    return parser

def main() -> None:
    """Main function to run the script."""
    # Parse command-line arguments
    parser = get_parser()
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create configuration
    config = Config(args)
    config.save(os.path.join(config.output_dir, "args.json"))
    logger.info(f"Configuration saved to {os.path.join(config.output_dir, 'args.json')}")
    
    # Create data module and prepare data
    data_module = DataModule(config)
    data_module.prepare_data()
    
    # Dictionary to store hyperparameters for both models
    model_hyperparams = {
        "gru": {
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
            "lookback": config.lookback
        },
        "lstm": {
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
            "lookback": config.lookback
        }
    }
    
    # Handle hyperparameter tuning or loading tuned parameters
    if config.run_tuning:
        logger.info("Running hyperparameter optimization for both models")
        hyperparameter_optimizer = HyperparameterOptimizer(config, data_module)
        best_params = hyperparameter_optimizer.run_optimization()
        logger.info("Hyperparameter optimization completed")
        
        # Update hyperparameters for both models
        for model_type in ["gru", "lstm"]:
            if model_type in best_params:
                model_hyperparams[model_type] = best_params[model_type]
        
    elif config.use_tuned_parameters:
        logger.info("Attempting to load previously tuned parameters")
        loaded_params = HyperparameterOptimizer.load_best_params(config.tuning_output_dir)
        
        if loaded_params:
            logger.info("Using previously tuned parameters")
            
            # Update parameters for each model if available
            for model_type in ["gru", "lstm"]:
                if model_type in loaded_params and loaded_params[model_type] is not None:
                    model_hyperparams[model_type] = loaded_params[model_type]
                    logger.info(f"Loaded parameters for {model_type.upper()}:")
                    for param, value in loaded_params[model_type].items():
                        logger.info(f"    {param}: {value}")
                else:
                    logger.warning(f"No tuned parameters found for {model_type.upper()}. Using default parameters.")
        else:
            logger.warning("No tuned parameters found. Using default parameters for both models.")

    # Print final hyperparameters for both models
    for model_type in ["gru", "lstm"]:
        logger.info(f"\n{model_type.upper()} model hyperparameters:")
        for param, value in model_hyperparams[model_type].items():
            logger.info(f"    {param}: {value}")
    
    # Dictionary to store models, trainers, and metrics
    models = {}
    trainers = {}
    model_metrics = {}
    model_results = {}
    
    # Train and evaluate both models
    for model_type in ["gru", "lstm"]:
        logger.info(f"\n========== Training {model_type.upper()} Model ==========")
        
        # Get model hyperparameters
        params = model_hyperparams[model_type]
        
        # Create datasets and dataloaders
        datasets = data_module.create_datasets(params["lookback"])
        dataloaders = data_module.create_dataloaders(datasets, params["batch_size"])
        
        # Initialize model
        output_size = 1
        model = ModelFactory.create_model(
            model_type, 
            data_module.input_size, 
            params["hidden_size"], 
            output_size, 
            params["dropout"], 
            params["num_layers"]
        )
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
        
        # Initialize scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Initialize trainer
        trainer = Trainer(model, config, config.output_dir, model_type)
        
        # Train model
        model, best_loss, best_epoch = trainer.train(
            dataloaders["train"], dataloaders["val"], optimizer, scheduler
        )
        
        # Store model, trainer, and results
        models[model_type] = model
        trainers[model_type] = trainer
        model_results[model_type] = {
            "best_loss": best_loss,
            "best_epoch": best_epoch
        }
        
        # Evaluate model on test set
        logger.info(f"\n========== Evaluating {model_type.upper()} Model on Test Set ==========")
        test_metrics, test_preds, test_targets = trainer.evaluate(
            dataloaders["test"], f"{model_type.upper()} Test"
        )
        
        # Store test metrics
        model_metrics[model_type] = {
            "test_metrics": test_metrics,
            "test_preds": test_preds,
            "test_targets": test_targets
        }
    
    # Save test metrics for both models
    json_serializable_results = convert_to_json_serializable({
        "gru": model_metrics["gru"]["test_metrics"],
        "lstm": model_metrics["lstm"]["test_metrics"]
    })

    # Save to JSON file
    with open(os.path.join(config.output_dir, "test_metrics.json"), 'w') as f:
        json.dump(json_serializable_results, f, indent=4)
    
    # Create visualizer and generate plots
    visualizer = Visualizer(config, data_module)
    visualizer.plot_training_history(config.output_dir, model_results)
    visualizer.plot_model_comparison(config.output_dir, model_metrics)
    visualizer.plot_test_predictions(config.output_dir, models, trainers)
    
    logger.info("\nExperiment completed successfully!")

if __name__ == "__main__":
    main()