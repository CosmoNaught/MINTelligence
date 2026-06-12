#!/usr/bin/env python3
"""
Validate with multiple mosquito density scenarios: -50%, -25%, 0%, +25%, +50%.

Outputs named *_alt_v1.4.0.{pdf,csv}
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

from estimint.run import run_xgb_model
from estimint.storage import load_xgb_model
from estimint.hbr import estimate_eir_with_mosquito_delta
import estimint
import minte
from minte import run_malaria_emulator, create_scenarios

# ── Multiple mosquito scenarios ──────────────────────────────────────────────
MOSQUITO_INCREASES = [-0.50, -0.25, 0.0, 0.25, 0.50]
MOSQ_COLORS = {
    -0.50: "tab:blue",
    -0.25: "tab:cyan",
     0.00: "red",         # baseline pipeline = red as before
     0.25: "darkorange",
     0.50: "tab:brown",
}
MOSQ_LABELS = {
    -0.50: "-50% mosq",
    -0.25: "-25% mosq",
     0.00: "Baseline (0%)",
     0.25: "+25% mosq",
     0.50: "+50% mosq",
}

print(f"Using estimint v{estimint.__version__}")
print(f"Using minte v{minte.__version__}")

# Configuration
DB_PATH = "/home/cosmo/Documents/Repos/MINT_DATA/malaria_simulations_4096.duckdb"
TABLE_NAME = "simulation_results"
EDGE_CASE_CSV = "/home/cosmo/Documents/Repos/MINTelligence/post/edge_case_test_matched.csv"
WINDOW_SIZE = 14
INTERVENTION_DAY = 9 * 365

MAINPY_CASES_MODEL_PATH = "/home/cosmo/Documents/Repos/MINTelligence/cases/results_simple_stratified_cases/lstm_final.pt"
MAINPY_CASES_SCALER_PATH = "/home/cosmo/Documents/Repos/MINTelligence/cases/results_simple_stratified_cases/static_scaler.pkl"
MAINPY_PREVALENCE_MODEL_PATH = "/home/cosmo/Documents/Repos/MINTelligence/prevalence/results_simple_stratified_prevalence/lstm_final.pt"
MAINPY_PREVALENCE_SCALER_PATH = "/home/cosmo/Documents/Repos/MINTelligence/prevalence/results_simple_stratified_prevalence/static_scaler.pkl"

EXCLUDED_CASES = {12, 15, 29}
MIN_PREV_Y9 = 0.02

print("Loading XGBoost model from estimint...")
FULL_MODEL = load_xgb_model()
print("Model loaded")

STATIC_COVARS = [
    "eir", "dn0_use", "dn0_future", "Q0", "phi_bednets",
    "seasonal", "routine", "itn_use", "irs_use",
    "itn_future", "irs_future", "lsm"
]
AFTER9_COVARS = ["dn0_future", "itn_future", "irs_future", "lsm", "routine"]


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 dropout_prob, num_layers=1, predictor="prevalence"):
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

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.ln(out)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.activation(out)
        return out


def inverse_transform_np(y, predictor):
    if predictor == "prevalence":
        return 1.0 / (1.0 + np.exp(-y))
    else:
        return np.expm1(y)


def load_mainpy_model(predictor):
    if predictor == 'cases':
        model_path = MAINPY_CASES_MODEL_PATH
        scaler_path = MAINPY_CASES_SCALER_PATH
        tuned_params_path = "/home/cosmo/Documents/Repos/MINTelligence/cases/results_tuned_simple_stratified_cases/best_params.json"
    else:
        model_path = MAINPY_PREVALENCE_MODEL_PATH
        scaler_path = MAINPY_PREVALENCE_SCALER_PATH
        tuned_params_path = "/home/cosmo/Documents/Repos/MINTelligence/prevalence/results_tuned_simple_stratified_prevalence/best_params.json"

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    with open(tuned_params_path, 'r') as f:
        tuned_params = json.load(f)
    lstm_params = tuned_params['lstm']

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)

    input_size = 2 + len(STATIC_COVARS) + 2
    hidden_size = lstm_params['hidden_size']
    num_layers = lstm_params['num_layers']
    dropout = lstm_params['dropout']

    print(f"  {predictor}: hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout:.3f}")

    model = LSTMModel(input_size, hidden_size, 1, dropout, num_layers, predictor)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, scaler


def fetch_simulation_data_prevalence(con, param_idx, sim_idx, window_size):
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


def predict_with_full_pipeline(df, case_row, window_size, predictor, mosquito_delta=0.0):
    """Predict using Full Pipeline: XGBoost -> LSTM (optionally with mosquito increase)."""
    year9_start = INTERVENTION_DAY - 365
    if 'prevalence' in df.columns:
        mask_y9 = (df["abs_timesteps"].values >= year9_start) & (df["abs_timesteps"].values < INTERVENTION_DAY)
        if mask_y9.any():
            prev_y9 = df.loc[mask_y9, "prevalence"].mean()
        else:
            year9_idx = np.argmin(np.abs(df["abs_timesteps"].values - INTERVENTION_DAY))
            prev_y9 = df.iloc[year9_idx]["prevalence"]
    else:
        prev_y9 = 0.1

    row_before = df[df["abs_timesteps"] < INTERVENTION_DAY].iloc[-1] if (df["abs_timesteps"] < INTERVENTION_DAY).any() else df.iloc[0]
    row_after = df[df["abs_timesteps"] >= INTERVENTION_DAY].iloc[0] if (df["abs_timesteps"] >= INTERVENTION_DAY).any() else df.iloc[-1]

    dn0_current = float(row_before.get('dn0_use', 0.0))
    dn0_future = float(row_after.get('dn0_future', 0.0))
    itn_current = float(row_before.get('itn_use', 0.0))
    itn_future = float(row_after.get('itn_future', 0.0))
    irs_current = float(row_before.get('irs_use', 0.0))
    irs_future = float(row_after.get('irs_future', 0.0))

    try:
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

        # Apply mosquito density change (positive OR negative)
        if mosquito_delta != 0:
            hbr_result = estimate_eir_with_mosquito_delta(
                prevalence=prev_y9,
                mosquito_delta=mosquito_delta,
                dn0_use=dn0_current,
                Q0=float(case_row['actual_Q0']),
                phi_bednets=float(case_row['actual_phi']),
                seasonal=1.0 if 'Seasonal' in case_row['description'] else 0.0,
                itn_use=itn_current,
                irs_use=irs_current,
            )
            estimated_eir = hbr_result["eir_new"]

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
    T = len(df)
    abs_t = df["abs_timesteps"].values.astype(np.float32)

    base_static = df.iloc[0][STATIC_COVARS].values.astype(np.float32)
    raw_matrix = np.tile(base_static, (T, 1))

    post_mask = (abs_t >= INTERVENTION_DAY)
    for cov in AFTER9_COVARS:
        if cov in STATIC_COVARS:
            j = STATIC_COVARS.index(cov)
            raw_matrix[~post_mask, j] = 0.0

    scaled_matrix = mainpy_scaler.transform(raw_matrix)

    post9 = (abs_t >= INTERVENTION_DAY).astype(np.float32)
    t_since9_years = np.maximum(0.0, abs_t - INTERVENTION_DAY) / 365.0

    day_of_year = abs_t % 365.0
    sin_t = np.sin(2 * math.pi * day_of_year / 365.0)
    cos_t = np.cos(2 * math.pi * day_of_year / 365.0)

    X_full = np.zeros((T, 2 + len(STATIC_COVARS) + 2), dtype=np.float32)
    X_full[:, 0] = sin_t
    X_full[:, 1] = cos_t
    X_full[:, 2:2+len(STATIC_COVARS)] = scaled_matrix
    X_full[:, -2] = post9
    X_full[:, -1] = t_since9_years

    with torch.no_grad():
        x_torch = torch.tensor(X_full, dtype=torch.float32).unsqueeze(1)
        pred_t = mainpy_model(x_torch).squeeze(-1).squeeze(-1)

    pred = pred_t.cpu().numpy()
    return inverse_transform_np(pred, predictor)


def run_validation(predictor, output_pdf, output_csv, mainpy_model, mainpy_scaler,
                   mainpy_model_prev=None, mainpy_scaler_prev=None):
    print(f"\n{'='*80}")
    print(f"VALIDATING: {predictor.upper()}")
    print(f"Mosquito scenarios: {[f'{int(m*100):+d}%' for m in MOSQUITO_INCREASES]}")
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

            sim_indices_df = con.execute(f"""
                SELECT DISTINCT simulation_index FROM {TABLE_NAME}
                WHERE parameter_index = {param_idx} ORDER BY simulation_index
            """).df()

            if len(sim_indices_df) == 0:
                print(f"  Case {case_id}: No simulations found")
                continue

            sim_indices = sim_indices_df['simulation_index'].tolist()

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

            all_y_true = []
            all_prev_true = []
            mainpy_preds = None

            # Dict: mosquito_increase -> (preds, eir_baseline, estimated_eir)
            mosq_results = {}

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

                if predictor == 'cases':
                    all_prev_true.append(df['prevalence'].values.astype(np.float32))

                if sim_idx == sim_indices[0]:
                    # Run all mosquito scenarios
                    for mi in MOSQUITO_INCREASES:
                        preds, py9, eir_base, eir_est = predict_with_full_pipeline(
                            df, case, WINDOW_SIZE, predictor, mosquito_delta=mi
                        )
                        mosq_results[mi] = {
                            'preds': preds,
                            'eir_baseline': eir_base,
                            'estimated_eir': eir_est,
                            'prev_y9': py9,
                        }

                    mainpy_preds = predict_with_mainpy_lstm(df, mainpy_model, mainpy_scaler, predictor)

            if not all_y_true:
                print(f"  No data")
                continue

            # Metrics
            T0 = len(all_y_true[0])
            days = np.arange(T0) * WINDOW_SIZE
            years_rel = days / 365.0
            year_2_idx = np.where(years_rel >= 2)[0][0] if (years_rel >= 2).any() else 0
            years_display = years_rel[year_2_idx:] - 2

            avg_y = np.mean([y[year_2_idx:] for y in all_y_true], axis=0)
            avg_mainpy = mainpy_preds[year_2_idx:]

            # Build row for CSV
            row_stats = {
                'case_id': case_id,
                'description': desc,
                'prev_y9': mosq_results[0.0]['prev_y9'],
                'true_eir': case['actual_eir'],
                'eir_baseline': mosq_results[0.0]['eir_baseline'],
            }

            for mi in MOSQUITO_INCREASES:
                pct_key = f"{int(mi*100):+d}"
                mr = mosq_results[mi]
                avg_mosq = mr['preds'][year_2_idx:]
                if not np.all(np.isnan(avg_mosq)):
                    mae_val = np.mean(np.abs(avg_mosq - avg_y))
                    rmse_val = np.sqrt(np.mean((avg_mosq - avg_y)**2))
                else:
                    mae_val, rmse_val = np.nan, np.nan
                row_stats[f'estimated_eir_{pct_key}'] = mr['estimated_eir']
                row_stats[f'mae_pipeline_{pct_key}'] = mae_val
                row_stats[f'rmse_pipeline_{pct_key}'] = rmse_val

            mae_mainpy = np.mean(np.abs(avg_mainpy - avg_y))
            rmse_mainpy = np.sqrt(np.mean((avg_mainpy - avg_y)**2))
            row_stats['mae_mainpy'] = mae_mainpy
            row_stats['rmse_mainpy'] = rmse_mainpy

            summary_stats.append(row_stats)

            eir_strs = " | ".join(
                f"{int(mi*100):+d}%: EIR={mosq_results[mi]['estimated_eir']:.1f}"
                for mi in MOSQUITO_INCREASES
            )
            print(f"  {eir_strs}")
            print(f"  main.py MAE={mae_mainpy:.4f}")

            # ── Plot ─────────────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(14, 9))

            for i, y_true in enumerate(all_y_true):
                y_t = y_true[year_2_idx:]
                ax.plot(years_display, y_t, color='gray', alpha=0.3, linewidth=0.8,
                        label='MalariaSim (individual)' if i == 0 else '')

            ax.plot(years_display, avg_y, 'k-', label='MalariaSim (avg)', linewidth=2.5)

            # Plot each mosquito scenario
            for mi in MOSQUITO_INCREASES:
                mr = mosq_results[mi]
                avg_mosq = mr['preds'][year_2_idx:]
                if np.all(np.isnan(avg_mosq)):
                    continue
                mae_val = np.mean(np.abs(avg_mosq - avg_y))
                pct_str = f"{int(mi*100):+d}%"
                lbl = f"{MOSQ_LABELS[mi]} (MAE={mae_val:.3f}, EIR={mr['estimated_eir']:.1f})"
                lw = 2.8 if mi == 0.0 else 2.0
                ax.plot(years_display, avg_mosq, color=MOSQ_COLORS[mi],
                        linewidth=lw, alpha=0.85, label=lbl)

            ax.plot(years_display, avg_mainpy, 'g-',
                    label=f'main.py LSTM (MAE={mae_mainpy:.3f})',
                    linewidth=2.5, alpha=0.85)

            ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Intervention')
            ax.axhline(y=0, color='red', linestyle=':', alpha=0.3, linewidth=1)

            if predictor == 'prevalence':
                ax.set_ylim(0, 1)
                ylabel = "Prevalence"
            else:
                ax.set_ylim(bottom=0)
                ylabel = "Cases per 1000 per day"

            ax.set_xlim(0, 4)
            ax.set_xlabel("Years (0 = burn-in end)", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)

            eir_base = mosq_results[0.0]['estimated_eir']
            eir_base_str = f"{eir_base:.1f}" if not np.isnan(eir_base) else "N/A"
            title = f"Case {case_id}: {desc}\n"
            title += f"Predictor: {predictor.upper()} | True EIR: {case['actual_eir']:.1f} | Est EIR (0%): {eir_base_str}\n"
            title += f"prev_y9: {mosq_results[0.0]['prev_y9']:.3f} | [minte v{minte.__version__}]"
            ax.set_title(title, fontsize=9, fontweight='bold')

            ax.legend(loc='best', fontsize=7.5)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    con.close()
    print(f"\nPDF saved: {output_pdf}")

    if summary_stats:
        df_summary = pd.DataFrame(summary_stats)
        df_summary.to_csv(output_csv, index=False)
        print(f"CSV saved: {output_csv}")

        if ood_cases:
            ood_csv = output_csv.replace('.csv', '_ood_excluded.csv')
            pd.DataFrame(ood_cases).to_csv(ood_csv, index=False)
            print(f"OOD cases saved: {ood_csv}")

    return pd.DataFrame(summary_stats) if summary_stats else pd.DataFrame()


def main():
    print(f"\n{'#'*80}")
    print(f"# ALT VALIDATION: Multiple mosquito scenarios for minte v{minte.__version__}")
    print(f"# Scenarios: {[f'{int(m*100):+d}%' for m in MOSQUITO_INCREASES]}")
    print(f"# OOD Filter: prevalence at year 9 >= {MIN_PREV_Y9} (2%)")
    print(f"{'#'*80}")

    print("\nLoading main.py LSTM models...")
    mainpy_model_prev, mainpy_scaler_prev = load_mainpy_model('prevalence')
    mainpy_model_cases, mainpy_scaler_cases = load_mainpy_model('cases')
    print("Models loaded")

    base = '/home/cosmo/Documents/Repos/MINTelligence/test_github_version'

    # Run for prevalence
    df_prev = run_validation(
        'prevalence',
        f'{base}/validation_prevalence_alt_v1.4.0.pdf',
        f'{base}/validation_prevalence_alt_v1.4.0.csv',
        mainpy_model_prev,
        mainpy_scaler_prev
    )

    from minte import clear_cache
    clear_cache()

    # Run for cases
    df_cases = run_validation(
        'cases',
        f'{base}/validation_cases_alt_v1.4.0.pdf',
        f'{base}/validation_cases_alt_v1.4.0.csv',
        mainpy_model_cases,
        mainpy_scaler_cases,
        mainpy_model_prev=mainpy_model_prev,
        mainpy_scaler_prev=mainpy_scaler_prev
    )

    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")

    for name, df_res in [("Prevalence", df_prev), ("Cases", df_cases)]:
        if len(df_res) > 0:
            print(f"\n{name} (n={len(df_res)}):")
            for mi in MOSQUITO_INCREASES:
                pct_key = f"{int(mi*100):+d}"
                col = f'mae_pipeline_{pct_key}'
                if col in df_res.columns:
                    print(f"  {int(mi*100):+d}% mosq  Mean MAE: {df_res[col].mean():.4f}")
            print(f"  main.py       Mean MAE: {df_res['mae_mainpy'].mean():.4f}")

    print(f"\nExperiment completed successfully!")


if __name__ == "__main__":
    main()
