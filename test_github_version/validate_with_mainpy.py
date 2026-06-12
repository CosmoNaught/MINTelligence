#!/usr/bin/env python3
"""
Validate absorbing state fix for cases (v1.3.1) with main.py LSTM comparison.

Tests that:
1. Full Pipeline (estimint + minte) predictions
2. main.py LSTM predictions with true EIR
3. Filters out OOD cases (prevalence < 2%)
"""

import pickle
import sys
import json
import math
import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Import from locally installed packages
from estimint.run import run_xgb_model
from estimint.storage import load_xgb_model
from estimint.hbr import estimate_eir_with_mosquito_increase
import estimint
import minte
from minte import run_malaria_emulator, create_scenarios

MOSQUITO_INCREASE = 0.50  # +50% mosquito density

print(f"Using estimint v{estimint.__version__}")
print(f"Using minte v{minte.__version__}")

# Configuration
DB_PATH = "/home/cosmo/Documents/Repos/MINT_DATA/malaria_simulations_4096.duckdb"
TABLE_NAME = "simulation_results"
EDGE_CASE_CSV = "/home/cosmo/Documents/Repos/MINTelligence/post/edge_case_test_matched.csv"
WINDOW_SIZE = 14
INTERVENTION_DAY = 9 * 365

# Model paths for main.py LSTM
MAINPY_CASES_MODEL_PATH = "/home/cosmo/Documents/Repos/MINTelligence/cases/results_simple_stratified_cases/lstm_final.pt"
MAINPY_CASES_SCALER_PATH = "/home/cosmo/Documents/Repos/MINTelligence/cases/results_simple_stratified_cases/static_scaler.pkl"
MAINPY_PREVALENCE_MODEL_PATH = "/home/cosmo/Documents/Repos/MINTelligence/prevalence/results_simple_stratified_prevalence/lstm_final.pt"
MAINPY_PREVALENCE_SCALER_PATH = "/home/cosmo/Documents/Repos/MINTelligence/prevalence/results_simple_stratified_prevalence/static_scaler.pkl"

# CASES TO EXCLUDE (original exclusions + OOD)
EXCLUDED_CASES = {12, 15, 29}
MIN_PREV_Y9 = 0.02  # 2% prevalence threshold

# Load XGBoost model
print("Loading XGBoost model from estimint...")
FULL_MODEL = load_xgb_model()
print("Model loaded")

# Static covariates (same as main.py)
STATIC_COVARS = [
    "eir", "dn0_use", "dn0_future", "Q0", "phi_bednets",
    "seasonal", "routine", "itn_use", "irs_use",
    "itn_future", "irs_future", "lsm"
]
AFTER9_COVARS = ["dn0_future", "itn_future", "irs_future", "lsm", "routine"]


class LSTMModel(nn.Module):
    """LSTM Model for time series prediction (matches main.py)."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 dropout_prob: float, num_layers: int = 1, predictor: str = "prevalence"):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.predictor = predictor
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            dropout=dropout_prob if num_layers > 1 else 0.0
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


def inverse_transform_np(y: np.ndarray, predictor: str) -> np.ndarray:
    """Invert transform for metrics/plots."""
    if predictor == "prevalence":
        return 1.0 / (1.0 + np.exp(-y))  # sigmoid
    else:
        return np.expm1(y)


def load_mainpy_model(predictor):
    """Load the trained LSTM model from main.py."""
    if predictor == 'cases':
        model_path = MAINPY_CASES_MODEL_PATH
        scaler_path = MAINPY_CASES_SCALER_PATH
        tuned_params_path = "/home/cosmo/Documents/Repos/MINTelligence/cases/results_tuned_simple_stratified_cases/best_params.json"
    else:
        model_path = MAINPY_PREVALENCE_MODEL_PATH
        scaler_path = MAINPY_PREVALENCE_SCALER_PATH
        tuned_params_path = "/home/cosmo/Documents/Repos/MINTelligence/prevalence/results_tuned_simple_stratified_prevalence/best_params.json"

    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Load tuned hyperparameters
    with open(tuned_params_path, 'r') as f:
        tuned_params = json.load(f)
    lstm_params = tuned_params['lstm']

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)

    # Model architecture (from tuned params - uses cyclical time)
    input_size = 2 + len(STATIC_COVARS) + 2  # sin/cos + static + post9/t_since9
    hidden_size = lstm_params['hidden_size']
    num_layers = lstm_params['num_layers']
    dropout = lstm_params['dropout']

    print(f"  {predictor}: hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout:.3f}")

    model = LSTMModel(input_size, hidden_size, 1, dropout, num_layers, predictor)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, scaler


def fetch_simulation_data_prevalence(con, param_idx, sim_idx, window_size):
    """Fetch prevalence simulation data aggregated by window."""
    last_6_years_day = 6 * 365
    W = window_size
    query = f"""
        WITH raw AS (
            SELECT parameter_index, simulation_index, timesteps AS abs_timesteps,
                CAST(n_detect_lm_0_1825 AS DOUBLE) AS n_detect,
                CAST(n_age_0_1825 AS DOUBLE) AS n_age,
                eir, dn0_use, dn0_future, Q0, phi_bednets, seasonal,
                routine, itn_use, irs_use, itn_future, irs_future, lsm
            FROM {TABLE_NAME}
            WHERE parameter_index = {param_idx}
              AND simulation_index = {sim_idx}
              AND timesteps >= {last_6_years_day}
        ),
        groups AS (
            SELECT parameter_index, simulation_index,
                FLOOR((abs_timesteps - {last_6_years_day}) / {W}) AS gid,
                SUM(n_detect) / NULLIF(SUM(n_age), 0) AS prevalence,
                MIN(abs_timesteps) AS abs_timesteps,
                MAX(eir) AS eir, MAX(dn0_use) AS dn0_use, MAX(dn0_future) AS dn0_future,
                MAX(Q0) AS Q0, MAX(phi_bednets) AS phi_bednets, MAX(seasonal) AS seasonal,
                MAX(routine) AS routine, MAX(itn_use) AS itn_use, MAX(irs_use) AS irs_use,
                MAX(itn_future) AS itn_future, MAX(irs_future) AS irs_future, MAX(lsm) AS lsm
            FROM raw GROUP BY 1,2,3
        )
        SELECT parameter_index, simulation_index,
            ROW_NUMBER() OVER (PARTITION BY parameter_index, simulation_index ORDER BY gid) AS timesteps,
            abs_timesteps, prevalence, eir, dn0_use, dn0_future, Q0, phi_bednets,
            seasonal, routine, itn_use, irs_use, itn_future, irs_future, lsm
        FROM groups ORDER BY gid
    """
    return con.execute(query).df()


def fetch_simulation_data_cases(con, param_idx, sim_idx, window_size):
    """Fetch cases simulation data aggregated by window (includes prevalence for EIR estimation)."""
    last_6_years_day = 6 * 365
    W = window_size
    query = f"""
        WITH raw AS (
            SELECT parameter_index, simulation_index, timesteps AS abs_timesteps,
                CAST(n_inc_clinical_0_36500 AS DOUBLE) AS n_inc,
                CAST(n_age_0_36500 AS DOUBLE) AS n_age,
                CAST(n_detect_lm_0_1825 AS DOUBLE) AS n_detect,
                CAST(n_age_0_1825 AS DOUBLE) AS n_age_prev,
                eir, dn0_use, dn0_future, Q0, phi_bednets, seasonal,
                routine, itn_use, irs_use, itn_future, irs_future, lsm
            FROM {TABLE_NAME}
            WHERE parameter_index = {param_idx}
              AND simulation_index = {sim_idx}
              AND timesteps >= {last_6_years_day}
        ),
        groups AS (
            SELECT parameter_index, simulation_index,
                FLOOR((abs_timesteps - {last_6_years_day}) / {W}) AS gid,
                1000.0 * SUM(n_inc) / NULLIF(SUM(n_age), 0) AS cases,
                SUM(n_detect) / NULLIF(SUM(n_age_prev), 0) AS prevalence,
                MIN(abs_timesteps) AS abs_timesteps,
                MAX(eir) AS eir, MAX(dn0_use) AS dn0_use, MAX(dn0_future) AS dn0_future,
                MAX(Q0) AS Q0, MAX(phi_bednets) AS phi_bednets, MAX(seasonal) AS seasonal,
                MAX(routine) AS routine, MAX(itn_use) AS itn_use, MAX(irs_use) AS irs_use,
                MAX(itn_future) AS itn_future, MAX(irs_future) AS irs_future, MAX(lsm) AS lsm
            FROM raw GROUP BY 1,2,3
        )
        SELECT parameter_index, simulation_index,
            ROW_NUMBER() OVER (PARTITION BY parameter_index, simulation_index ORDER BY gid) AS timesteps,
            abs_timesteps, cases, prevalence, eir, dn0_use, dn0_future, Q0, phi_bednets,
            seasonal, routine, itn_use, irs_use, itn_future, irs_future, lsm
        FROM groups ORDER BY gid
    """
    return con.execute(query).df()


def predict_with_full_pipeline(df, case_row, window_size, predictor, mosquito_increase=0.0):
    """Predict using Full Pipeline: XGBoost -> LSTM (optionally with mosquito increase)."""
    # Extract annual mean prevalence over year 9 (365 days before intervention)
    year9_start = INTERVENTION_DAY - 365
    if 'prevalence' in df.columns:
        mask_y9 = (df["abs_timesteps"].values >= year9_start) & (df["abs_timesteps"].values < INTERVENTION_DAY)
        if mask_y9.any():
            prev_y9 = df.loc[mask_y9, "prevalence"].mean()
        else:
            # Fallback to closest point if no data in range
            year9_idx = np.argmin(np.abs(df["abs_timesteps"].values - INTERVENTION_DAY))
            prev_y9 = df.iloc[year9_idx]["prevalence"]
    else:
        prev_y9 = 0.1  # fallback

    # Get current/future values
    row_before = df[df["abs_timesteps"] < INTERVENTION_DAY].iloc[-1] if (df["abs_timesteps"] < INTERVENTION_DAY).any() else df.iloc[0]
    row_after = df[df["abs_timesteps"] >= INTERVENTION_DAY].iloc[0] if (df["abs_timesteps"] >= INTERVENTION_DAY).any() else df.iloc[-1]

    dn0_current = float(row_before.get('dn0_use', 0.0))
    dn0_future = float(row_after.get('dn0_future', 0.0))
    itn_current = float(row_before.get('itn_use', 0.0))
    itn_future = float(row_after.get('itn_future', 0.0))
    irs_current = float(row_before.get('irs_use', 0.0))
    irs_future = float(row_after.get('irs_future', 0.0))

    try:
        # Step 1: XGBoost estimates EIR from prevalence
        X = pd.DataFrame({
            'prev_y9': [prev_y9],
            'dn0_use': [dn0_current],
            'Q0': [float(case_row['actual_Q0'])],
            'phi_bednets': [float(case_row['actual_phi'])],
            'seasonal': [1.0 if 'Seasonal' in case_row['description'] else 0.0],
            'itn_use': [itn_current],
            'irs_use': [irs_current],
        })

        estimated_eir = run_xgb_model(X, FULL_MODEL)[0]
        eir_baseline = float(estimated_eir)

        # Optionally adjust EIR for mosquito density increase
        if mosquito_increase > 0:
            hbr_result = estimate_eir_with_mosquito_increase(
                prevalence=prev_y9,
                mosquito_increase=mosquito_increase,
                dn0_use=dn0_current,
                Q0=float(case_row['actual_Q0']),
                phi_bednets=float(case_row['actual_phi']),
                seasonal=1.0 if 'Seasonal' in case_row['description'] else 0.0,
                itn_use=itn_current,
                irs_use=irs_current,
            )
            estimated_eir = hbr_result["eir_new"]

        # Step 2: LSTM predicts from estimated EIR
        scenario = create_scenarios(
            eir=[estimated_eir],
            dn0_use=[dn0_current],
            dn0_future=[dn0_future],
            Q0=[float(case_row['actual_Q0'])],
            phi_bednets=[float(case_row['actual_phi'])],
            seasonal=[1.0 if 'Seasonal' in case_row['description'] else 0.0],
            routine=[0.0],
            itn_use=[itn_current],
            irs_use=[irs_current],
            itn_future=[itn_future],
            irs_future=[irs_future],
            lsm=[float(case_row.get('actual_lsm', 0.0))]
        )

        results = run_malaria_emulator(
            scenarios=scenario,
            predictor=predictor,
            window_size=window_size,
            device='cpu',
            time_steps=int(len(df) * window_size),
            use_cache=True,
            benchmark=False
        )

        # Align with simulation timesteps
        api_preds = results[predictor].values
        n_sim = len(df)
        if len(api_preds) < n_sim:
            api_preds = np.pad(api_preds, (0, n_sim - len(api_preds)), mode='edge')
        else:
            api_preds = api_preds[:n_sim]

        return api_preds, prev_y9, eir_baseline, float(estimated_eir)

    except Exception as e:
        print(f"      Warning: Full pipeline prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return np.full(len(df), np.nan), prev_y9, np.nan, np.nan


def predict_with_mainpy_lstm(df, mainpy_model, mainpy_scaler, predictor):
    """Predict using main.py LSTM with TRUE EIR from simulation data."""
    T = len(df)
    abs_t = df["abs_timesteps"].values.astype(np.float32)

    # Build covariate matrix (using TRUE EIR from simulation)
    base_static = df.iloc[0][STATIC_COVARS].values.astype(np.float32)
    raw_matrix = np.tile(base_static, (T, 1))

    # Gate future vars before day 9*365
    post_mask = (abs_t >= INTERVENTION_DAY)
    for cov in AFTER9_COVARS:
        if cov in STATIC_COVARS:
            j = STATIC_COVARS.index(cov)
            raw_matrix[~post_mask, j] = 0.0

    # Scale with train-fitted scaler
    scaled_matrix = mainpy_scaler.transform(raw_matrix)

    # Dynamic event features
    post9 = (abs_t >= INTERVENTION_DAY).astype(np.float32)
    t_since9_years = np.maximum(0.0, abs_t - INTERVENTION_DAY) / 365.0

    # Cyclical time encoding (same as main.py with use_cyclical_time=True)
    day_of_year = abs_t % 365.0
    sin_t = np.sin(2 * math.pi * day_of_year / 365.0)
    cos_t = np.cos(2 * math.pi * day_of_year / 365.0)

    X_full = np.zeros((T, 2 + len(STATIC_COVARS) + 2), dtype=np.float32)
    X_full[:, 0] = sin_t
    X_full[:, 1] = cos_t
    X_full[:, 2:2+len(STATIC_COVARS)] = scaled_matrix
    X_full[:, -2] = post9
    X_full[:, -1] = t_since9_years

    # Predict
    with torch.no_grad():
        x_torch = torch.tensor(X_full, dtype=torch.float32).unsqueeze(1)  # (T, 1, features)
        pred_t = mainpy_model(x_torch).squeeze(-1).squeeze(-1)

    pred = pred_t.cpu().numpy()
    return inverse_transform_np(pred, predictor)


def run_validation(predictor, output_pdf, output_csv, mainpy_model, mainpy_scaler,
                   mainpy_model_prev=None, mainpy_scaler_prev=None):
    """Run validation for a specific predictor."""
    print(f"\n{'='*80}")
    print(f"VALIDATING: {predictor.upper()}")
    print(f"{'='*80}")

    edge_cases = pd.read_csv(EDGE_CASE_CSV)
    edge_cases = edge_cases[~edge_cases['case_id'].isin(EXCLUDED_CASES)]
    print(f"Loaded {len(edge_cases)} edge cases (after original exclusions)")

    con = duckdb.connect(DB_PATH, read_only=True)
    con.execute("PRAGMA memory_limit='8GB';")
    con.execute("PRAGMA threads=4;")

    summary_stats = []
    ood_cases = []

    with PdfPages(output_pdf) as pdf:
        for _, case in edge_cases.iterrows():
            param_idx = int(case['parameter_index'])
            desc = case['description']
            case_id = case['case_id']

            # Get simulation indices
            sim_indices_df = con.execute(f"""
                SELECT DISTINCT simulation_index FROM {TABLE_NAME}
                WHERE parameter_index = {param_idx} ORDER BY simulation_index
            """).df()

            if len(sim_indices_df) == 0:
                print(f"  Case {case_id}: No simulations found")
                continue

            sim_indices = sim_indices_df['simulation_index'].tolist()

            # Get data for first simulation to check prev_y9
            sim_idx = sim_indices[0]
            if predictor == 'prevalence':
                df = fetch_simulation_data_prevalence(con, param_idx, sim_idx, WINDOW_SIZE)
                target_col = 'prevalence'
            else:
                df = fetch_simulation_data_cases(con, param_idx, sim_idx, WINDOW_SIZE)
                target_col = 'cases'

            if len(df) == 0:
                print(f"  Case {case_id}: No data")
                continue

            # Check annual mean prevalence at year 9 (OOD filter)
            year9_start = INTERVENTION_DAY - 365
            if 'prevalence' in df.columns:
                mask_y9 = (df["abs_timesteps"].values >= year9_start) & (df["abs_timesteps"].values < INTERVENTION_DAY)
                prev_y9 = df.loc[mask_y9, "prevalence"].mean() if mask_y9.any() else None
            else:
                prev_y9 = None

            if prev_y9 is not None and prev_y9 < MIN_PREV_Y9:
                print(f"  Case {case_id}: SKIPPED (OOD - prev_y9={prev_y9:.4f} < {MIN_PREV_Y9})")
                ood_cases.append({
                    'case_id': case_id,
                    'description': desc,
                    'prev_y9': prev_y9,
                    'reason': f'prev_y9 < {MIN_PREV_Y9}'
                })
                continue

            print(f"\nCase {case_id}: {desc}")

            all_y_true, all_pipeline_preds, all_mainpy_preds = [], [], []
            all_prev_true = []  # prevalence from malariasim (for horizontal lines)
            estimated_eir = None
            pipeline_prev_preds = None
            mainpy_prev_preds = None

            for sim_idx in sim_indices:
                if predictor == 'prevalence':
                    df = fetch_simulation_data_prevalence(con, param_idx, sim_idx, WINDOW_SIZE)
                    target_col = 'prevalence'
                else:
                    df = fetch_simulation_data_cases(con, param_idx, sim_idx, WINDOW_SIZE)
                    target_col = 'cases'

                if len(df) == 0:
                    continue

                y_true = df[target_col].values.astype(np.float32)
                all_y_true.append(y_true)

                # Collect malariasim prevalence for horizontal lines
                if predictor == 'cases':
                    all_prev_true.append(df['prevalence'].values.astype(np.float32))

                # Get pipeline predictions (only once)
                if sim_idx == sim_indices[0]:
                    pipeline_preds, prev_y9, eir_baseline, estimated_eir = predict_with_full_pipeline(
                        df, case, WINDOW_SIZE, predictor
                    )
                    # +50% mosquito density prediction
                    pipeline_preds_mosquito, _, _, estimated_eir_mosquito = predict_with_full_pipeline(
                        df, case, WINDOW_SIZE, predictor, mosquito_increase=MOSQUITO_INCREASE
                    )
                    mainpy_preds = predict_with_mainpy_lstm(df, mainpy_model, mainpy_scaler, predictor)
                    all_pipeline_preds.append(pipeline_preds)
                    all_mainpy_preds.append(mainpy_preds)

                    # Get prevalence predictions for horizontal lines (cases mode)
                    if predictor == 'cases' and mainpy_model_prev is not None:
                        try:
                            pipeline_prev_preds, _, _, _ = predict_with_full_pipeline(
                                df, case, WINDOW_SIZE, 'prevalence')
                        except Exception:
                            pipeline_prev_preds = np.full(len(df), np.nan)
                        try:
                            mainpy_prev_preds = predict_with_mainpy_lstm(
                                df, mainpy_model_prev, mainpy_scaler_prev, 'prevalence')
                        except Exception:
                            mainpy_prev_preds = np.full(len(df), np.nan)
                else:
                    all_pipeline_preds.append(all_pipeline_preds[0])
                    all_mainpy_preds.append(all_mainpy_preds[0])

            if not all_y_true:
                print(f"  No data")
                continue

            # Check for negative predictions
            neg_count_pipeline = np.sum(all_pipeline_preds[0] < 0)
            min_pred_pipeline = np.min(all_pipeline_preds[0])
            neg_count_mainpy = np.sum(all_mainpy_preds[0] < 0)
            min_pred_mainpy = np.min(all_mainpy_preds[0])
            print(f"  Pipeline: min={min_pred_pipeline:.4f}, neg_count={neg_count_pipeline}")
            print(f"  main.py:  min={min_pred_mainpy:.4f}, neg_count={neg_count_mainpy}")

            # Calculate metrics
            T0 = len(all_y_true[0])
            days = np.arange(T0) * WINDOW_SIZE
            years_rel = days / 365.0
            year_2_idx = np.where(years_rel >= 2)[0][0] if (years_rel >= 2).any() else 0
            years_display = years_rel[year_2_idx:] - 2

            avg_y = np.mean([y[year_2_idx:] for y in all_y_true], axis=0)
            avg_pipeline = np.mean([p[year_2_idx:] for p in all_pipeline_preds], axis=0) if not np.all(np.isnan(all_pipeline_preds[0])) else np.full_like(avg_y, np.nan)
            avg_mainpy = np.mean([p[year_2_idx:] for p in all_mainpy_preds], axis=0)
            avg_mosquito = pipeline_preds_mosquito[year_2_idx:] if (pipeline_preds_mosquito is not None and not np.all(np.isnan(pipeline_preds_mosquito))) else np.full_like(avg_y, np.nan)

            # Compute pre-intervention prevalence means for horizontal lines (yr 0-1)
            mask_0_1 = (years_display >= 0) & (years_display <= 1)

            if predictor == 'prevalence':
                # Trajectories ARE prevalence
                hline_prev_sim = np.nanmean(avg_y[mask_0_1])
                hline_prev_pipeline = np.nanmean(avg_pipeline[mask_0_1]) if not np.all(np.isnan(avg_pipeline)) else np.nan
                hline_prev_mainpy = np.nanmean(avg_mainpy[mask_0_1])
            else:
                # Cases mode: use separately-collected prevalence data
                avg_prev_sim = np.mean([p[year_2_idx:] for p in all_prev_true], axis=0)
                hline_prev_sim = np.nanmean(avg_prev_sim[mask_0_1])
                hline_prev_pipeline = np.nanmean(pipeline_prev_preds[year_2_idx:][mask_0_1]) if (
                    pipeline_prev_preds is not None and not np.all(np.isnan(pipeline_prev_preds))) else np.nan
                hline_prev_mainpy = np.nanmean(mainpy_prev_preds[year_2_idx:][mask_0_1]) if (
                    mainpy_prev_preds is not None and not np.all(np.isnan(mainpy_prev_preds))) else np.nan

            if not np.all(np.isnan(avg_pipeline)):
                mae_pipeline = np.mean(np.abs(avg_pipeline - avg_y))
                rmse_pipeline = np.sqrt(np.mean((avg_pipeline - avg_y)**2))
            else:
                mae_pipeline, rmse_pipeline = np.nan, np.nan

            mae_mainpy = np.mean(np.abs(avg_mainpy - avg_y))
            rmse_mainpy = np.sqrt(np.mean((avg_mainpy - avg_y)**2))

            # Mosquito scenario metrics
            if not np.all(np.isnan(avg_mosquito)):
                mae_pipeline_mosquito = np.mean(np.abs(avg_mosquito - avg_y))
            else:
                mae_pipeline_mosquito = np.nan

            summary_stats.append({
                'case_id': case_id,
                'description': desc,
                'prev_y9': prev_y9,
                'estimated_eir': estimated_eir,
                'estimated_eir_mosquito': estimated_eir_mosquito,
                'true_eir': case['actual_eir'],
                'mae_pipeline': mae_pipeline,
                'rmse_pipeline': rmse_pipeline,
                'mae_pipeline_mosquito': mae_pipeline_mosquito,
                'mae_mainpy': mae_mainpy,
                'rmse_mainpy': rmse_mainpy,
                'min_pred_pipeline': min_pred_pipeline,
                'min_pred_mainpy': min_pred_mainpy,
                'neg_count_pipeline': neg_count_pipeline,
                'neg_count_mainpy': neg_count_mainpy,
                'hline_prev_sim': hline_prev_sim,
                'hline_prev_pipeline': hline_prev_pipeline,
                'hline_prev_mainpy': hline_prev_mainpy,
                'prev_error': abs(prev_y9 - hline_prev_pipeline) if (prev_y9 is not None and not np.isnan(prev_y9) and not np.isnan(hline_prev_pipeline)) else np.nan,
            })

            eir_mosq_str = f"{estimated_eir_mosquito:.1f}" if not np.isnan(estimated_eir_mosquito) else "N/A"
            print(f"  Pipeline MAE={mae_pipeline:.4f}, main.py MAE={mae_mainpy:.4f} | +{int(MOSQUITO_INCREASE*100)}% mosq EIR={eir_mosq_str}")

            # Create plot
            fig, ax = plt.subplots(figsize=(12, 8))

            # Plot individual simulations
            for i, y_true in enumerate(all_y_true):
                y_t = y_true[year_2_idx:]
                ax.plot(years_display, y_t, color='gray', alpha=0.3, linewidth=0.8,
                        label='MalariaSim (individual)' if i == 0 else '')

            ax.plot(years_display, avg_y, 'k-', label='MalariaSim (avg)', linewidth=2.5)

            if not np.all(np.isnan(avg_pipeline)):
                ax.plot(years_display, avg_pipeline, 'r-',
                       label=f'Full Pipeline v{minte.__version__} (MAE={mae_pipeline:.3f}, EIR={estimated_eir:.1f})',
                       linewidth=2.5, alpha=0.85)

            if not np.all(np.isnan(avg_mosquito)):
                eir_mosq_str = f"{estimated_eir_mosquito:.1f}" if not np.isnan(estimated_eir_mosquito) else "N/A"
                ax.plot(years_display, avg_mosquito, color='darkorange', linestyle='-',
                       label=f'+{int(MOSQUITO_INCREASE*100)}% mosquitoes (EIR={eir_mosq_str})',
                       linewidth=2.5, alpha=0.85)

            ax.plot(years_display, avg_mainpy, 'g-',
                   label=f'main.py LSTM (MAE={mae_mainpy:.3f})',
                   linewidth=2.5, alpha=0.85)

            ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Intervention')
            ax.axhline(y=0, color='red', linestyle=':', alpha=0.3, linewidth=1, label='Zero line')

            # Add horizontal prevalence lines (pre-intervention mean, x=0 to x=1)
            if predictor == 'prevalence':
                target_ax = ax
            else:
                ax2 = ax.twinx()
                ax2.set_ylim(0, 1)
                ax2.set_ylabel("Prevalence", fontsize=10, color='gray')
                ax2.tick_params(axis='y', labelcolor='gray')
                target_ax = ax2

            if prev_y9 is not None and not np.isnan(prev_y9):
                target_ax.hlines(prev_y9, 0, 1, colors='blue', linestyles='solid',
                                 linewidth=2.0, alpha=0.8,
                                 label=f'Actual prev_y9 (input): {prev_y9:.3f}')
            if not np.isnan(hline_prev_sim):
                target_ax.hlines(hline_prev_sim, 0, 1, colors='black', linestyles='dashed',
                                 linewidth=1.5, alpha=0.7,
                                 label=f'MalariaSim prev (yr 0-1): {hline_prev_sim:.3f}')
            if not np.isnan(hline_prev_pipeline):
                target_ax.hlines(hline_prev_pipeline, 0, 1, colors='red', linestyles='dashed',
                                 linewidth=1.5, alpha=0.7,
                                 label=f'Pipeline prev (yr 0-1): {hline_prev_pipeline:.3f}')
            if not np.isnan(hline_prev_mainpy):
                target_ax.hlines(hline_prev_mainpy, 0, 1, colors='green', linestyles='dashed',
                                 linewidth=1.5, alpha=0.7,
                                 label=f'main.py prev (yr 0-1): {hline_prev_mainpy:.3f}')

            # Red shaded error box between actual prev_y9 and pipeline prev estimate
            if prev_y9 is not None and not np.isnan(prev_y9) and not np.isnan(hline_prev_pipeline):
                target_ax.fill_between([0, 1], prev_y9, hline_prev_pipeline,
                                       color='red', alpha=0.15,
                                       label=f'Prev error: {abs(prev_y9 - hline_prev_pipeline):.3f}')

            if predictor == 'prevalence':
                ax.set_ylim(0, 1)
                ylabel = "Prevalence"
            else:
                ax.set_ylim(bottom=0)
                ylabel = "Cases per 1000 per day"

            ax.set_xlim(0, 4)
            ax.set_xlabel("Years (0 = burn-in end)", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)

            title = f"Case {case_id}: {desc}\n"
            est_eir_str = f"{estimated_eir:.1f}" if estimated_eir is not None and not np.isnan(estimated_eir) else "N/A"
            title += f"Predictor: {predictor.upper()} | True EIR: {case['actual_eir']:.1f} | Est EIR: {est_eir_str}\n"
            title += f"prev_y9: {prev_y9:.3f} | [minte v{minte.__version__}]"
            ax.set_title(title, fontsize=9, fontweight='bold')

            if predictor == 'cases' and mainpy_model_prev is not None:
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8)
            else:
                ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # Prevalence matching check summary page
        if summary_stats:
            df_summary = pd.DataFrame(summary_stats)
            cases_sorted = df_summary.sort_values('case_id')

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 11))

            # Plot 1: Input prev_y9 vs Pipeline estimated prev (scatter)
            valid_mask = ~cases_sorted['prev_error'].isna()
            cs_valid = cases_sorted[valid_mask]

            ax1.scatter(cs_valid['prev_y9'], cs_valid['hline_prev_pipeline'],
                       alpha=0.7, s=60, c='red', edgecolors='darkred', zorder=3)
            prev_min = min(cs_valid['prev_y9'].min(), cs_valid['hline_prev_pipeline'].min()) * 0.9
            prev_max = max(cs_valid['prev_y9'].max(), cs_valid['hline_prev_pipeline'].max()) * 1.1
            ax1.plot([prev_min, prev_max], [prev_min, prev_max], 'k--', alpha=0.5,
                    linewidth=1.5, label='Perfect match')
            for _, row in cs_valid.iterrows():
                ax1.annotate(str(int(row['case_id'])),
                            (row['prev_y9'], row['hline_prev_pipeline']),
                            fontsize=6, alpha=0.7, textcoords='offset points',
                            xytext=(3, 3))
            ax1.set_xlabel('Input Prevalence (prev_y9)', fontsize=11)
            ax1.set_ylabel('Pipeline Estimated Prevalence (yr 0-1)', fontsize=11)
            ax1.set_title('Prevalence Matching: Input vs Pipeline Output', fontsize=12, fontweight='bold')
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)

            # Plot 2: Prevalence error per case (bar chart)
            x = np.arange(len(cs_valid))
            errors = (cs_valid['hline_prev_pipeline'] - cs_valid['prev_y9']).values
            colors = ['red' if e > 0 else 'blue' for e in errors]
            ax2.bar(x, errors, color=colors, alpha=0.7, edgecolor='gray', linewidth=0.5)
            ax2.axhline(0, color='black', linewidth=0.8)
            ax2.set_xlabel('Case ID', fontsize=11)
            ax2.set_ylabel('Prevalence Error (pipeline - input)', fontsize=11)
            ax2.set_title('Prevalence Error by Case', fontsize=12, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(cs_valid['case_id'].astype(int), rotation=90, fontsize=7)
            ax2.grid(True, alpha=0.3, axis='y')

            # Plot 3: Input prev_y9 vs MalariaSim avg prev (scatter) — sanity check
            ax3.scatter(cs_valid['prev_y9'], cs_valid['hline_prev_sim'],
                       alpha=0.7, s=60, c='black', edgecolors='gray', zorder=3)
            sim_min = min(cs_valid['prev_y9'].min(), cs_valid['hline_prev_sim'].min()) * 0.9
            sim_max = max(cs_valid['prev_y9'].max(), cs_valid['hline_prev_sim'].max()) * 1.1
            ax3.plot([sim_min, sim_max], [sim_min, sim_max], 'k--', alpha=0.5,
                    linewidth=1.5, label='Perfect match')
            for _, row in cs_valid.iterrows():
                ax3.annotate(str(int(row['case_id'])),
                            (row['prev_y9'], row['hline_prev_sim']),
                            fontsize=6, alpha=0.7, textcoords='offset points',
                            xytext=(3, 3))
            ax3.set_xlabel('Input Prevalence (prev_y9)', fontsize=11)
            ax3.set_ylabel('MalariaSim Avg Prevalence (yr 0-1)', fontsize=11)
            ax3.set_title('Sanity Check: Input vs MalariaSim Prevalence', fontsize=12, fontweight='bold')
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)

            # Plot 4: Summary text
            ax4.axis('off')
            mean_abs_err = cs_valid['prev_error'].mean()
            max_abs_err = cs_valid['prev_error'].max()
            median_abs_err = cs_valid['prev_error'].median()
            r2_prev = 1 - np.sum((cs_valid['hline_prev_pipeline'] - cs_valid['prev_y9'])**2) / np.sum((cs_valid['prev_y9'] - cs_valid['prev_y9'].mean())**2)
            mean_bias = (cs_valid['hline_prev_pipeline'] - cs_valid['prev_y9']).mean()
            sim_vs_input_corr = np.corrcoef(cs_valid['prev_y9'], cs_valid['hline_prev_sim'])[0, 1]

            summary_text = f"""
PREVALENCE MATCHING CHECK - {predictor.upper()}
{'='*55}

Version: minte v{minte.__version__}, estimint v{estimint.__version__}
OOD Filter: prev_y9 >= {MIN_PREV_Y9} (2%)
Cases excluded as OOD: {len(ood_cases)}
Cases validated: {len(cs_valid)}

INPUT (prev_y9) vs PIPELINE ESTIMATED PREVALENCE:
  R²:              {r2_prev:.4f}
  Mean Abs Error:   {mean_abs_err:.4f}
  Median Abs Error: {median_abs_err:.4f}
  Max Abs Error:    {max_abs_err:.4f}
  Mean Bias:        {mean_bias:+.4f}

SANITY CHECK (input vs MalariaSim avg):
  Correlation:      {sim_vs_input_corr:.4f}

PASS/FAIL (mean abs prev error < 0.05):
  {'PASS' if mean_abs_err < 0.05 else 'FAIL'} (mean abs error = {mean_abs_err:.4f})
"""
            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')

            plt.suptitle(f'Prevalence Matching: {predictor.upper()} - minte v{minte.__version__}',
                        fontsize=14, fontweight='bold', y=0.995)
            plt.tight_layout(rect=[0, 0, 1, 0.99])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    con.close()
    print(f"\nPDF saved: {output_pdf}")

    if summary_stats:
        df_summary = pd.DataFrame(summary_stats)
        df_summary.to_csv(output_csv, index=False)
        print(f"CSV saved: {output_csv}")

        # Save OOD cases
        if ood_cases:
            ood_csv = output_csv.replace('.csv', '_ood_excluded.csv')
            pd.DataFrame(ood_cases).to_csv(ood_csv, index=False)
            print(f"OOD cases saved: {ood_csv}")

    return pd.DataFrame(summary_stats) if summary_stats else pd.DataFrame()


def main():
    print(f"\n{'#'*80}")
    print(f"# VALIDATION WITH MAIN.PY LSTM COMPARISON FOR minte v{minte.__version__}")
    print(f"# OOD Filter: prevalence at year 9 >= {MIN_PREV_Y9} (2%)")
    print(f"{'#'*80}")

    # Load main.py LSTM models
    print("\nLoading main.py LSTM models...")
    mainpy_model_prev, mainpy_scaler_prev = load_mainpy_model('prevalence')
    mainpy_model_cases, mainpy_scaler_cases = load_mainpy_model('cases')
    print("Models loaded")

    # Run for prevalence
    df_prev = run_validation(
        'prevalence',
        '/home/cosmo/Documents/Repos/MINTelligence/test_github_version/validation_prevalence_v1.4.0.pdf',
        '/home/cosmo/Documents/Repos/MINTelligence/test_github_version/validation_prevalence_v1.4.0.csv',
        mainpy_model_prev,
        mainpy_scaler_prev
    )

    # Clear cache between predictors
    from minte import clear_cache
    clear_cache()

    # Run for cases
    df_cases = run_validation(
        'cases',
        '/home/cosmo/Documents/Repos/MINTelligence/test_github_version/validation_cases_v1.4.0.pdf',
        '/home/cosmo/Documents/Repos/MINTelligence/test_github_version/validation_cases_v1.4.0.csv',
        mainpy_model_cases,
        mainpy_scaler_cases,
        mainpy_model_prev=mainpy_model_prev,
        mainpy_scaler_prev=mainpy_scaler_prev
    )

    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")

    if len(df_prev) > 0:
        print(f"\nPrevalence:")
        print(f"  Cases validated: {len(df_prev)}")
        print(f"  Pipeline Mean MAE: {df_prev['mae_pipeline'].mean():.4f}")
        print(f"  main.py Mean MAE:  {df_prev['mae_mainpy'].mean():.4f}")

    if len(df_cases) > 0:
        print(f"\nCases:")
        print(f"  Cases validated: {len(df_cases)}")
        print(f"  Pipeline Mean MAE: {df_cases['mae_pipeline'].mean():.4f}")
        print(f"  main.py Mean MAE:  {df_cases['mae_mainpy'].mean():.4f}")

    print(f"\nExperiment completed successfully!")


if __name__ == "__main__":
    main()
