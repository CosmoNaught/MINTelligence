#!/usr/bin/env python3
"""
Validate MINTverse (estimint + minte) pipeline against MalariaSim
for ALL validation-split scenarios from train_val_test_split.csv.

Generates:
  - validation_results.csv       : scenario parameters and results
  - validation_cases.pdf         : cases per 1000 comparison
  - validation_prevalence.pdf    : prevalence comparison

Colour scheme:
  Trajectories:
    gray thin      – MalariaSim individual sims
    black solid    – MalariaSim average dynamics
    green solid    – MINTverse (estimint -> minte) dynamics
  Horizontal lines (x = 0 to 1):
    black dashed   – MalariaSim actual avg prevalence (yr 0-1)
    green dashed   – MINTverse actual avg prevalence (yr 0-1)
    blue dashed    – Input prevalence (prev_y9, what was given to model)
  Shading (x = 0 to 1):
    red solid      – gap between input (blue) and MalariaSim (black dashed)
    red hatched    – gap between input (blue) and MINTverse (green dashed)
"""

import os
import numpy as np
import pandas as pd
import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from estimint.run import run_xgb_model
from estimint.storage import load_xgb_model
from estimint.hbr import estimate_eir_with_mosquito_increase
import estimint
import minte
from minte import run_malaria_emulator, create_scenarios, clear_cache

MOSQUITO_INCREASES = [0.20, 0.40, 0.60, 0.80, 1.00]
MOSQUITO_COLORS = ['#f0c929', '#e8850c', '#d94e1f', '#b5223b', '#7a0177']

print(f"Using estimint v{estimint.__version__} (PyPI)")
print(f"Using minte v{minte.__version__} (PyPI)")

# ── Configuration ──────────────────────────────────────────────────────────────
DB_PATH = "/home/cosmo/Documents/Repos/MINT_DATA/malaria_simulations_4096.duckdb"
TABLE_NAME = "simulation_results"
SPLIT_CSV = "/home/cosmo/Documents/Repos/MINTelligence/cases/results_simple_stratified_cases/train_val_test_split.csv"
WINDOW_SIZE = 14
INTERVENTION_DAY = 9 * 365  # 3285

OUTPUT_DIR = "/home/cosmo/Documents/Repos/MINTelligence/validation_plots"
MIN_PREV_Y9 = 0.02

# Load XGBoost model
print("Loading XGBoost model from estimint (PyPI)...")
FULL_MODEL = load_xgb_model()
print("Model loaded")


# ── Data fetching ──────────────────────────────────────────────────────────────

def fetch_simulation_data_prevalence(con, param_idx, sim_idx):
    last_6y = 6 * 365
    W = WINDOW_SIZE
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
              AND timesteps >= {last_6y}
        ),
        groups AS (
            SELECT parameter_index, simulation_index,
                FLOOR((abs_timesteps - {last_6y}) / {W}) AS gid,
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


def fetch_simulation_data_cases(con, param_idx, sim_idx):
    last_6y = 6 * 365
    W = WINDOW_SIZE
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
              AND timesteps >= {last_6y}
        ),
        groups AS (
            SELECT parameter_index, simulation_index,
                FLOOR((abs_timesteps - {last_6y}) / {W}) AS gid,
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


# ── Pipeline prediction ────────────────────────────────────────────────────────

def predict_with_pipeline(df, predictor):
    """MINTverse pipeline: estimint XGBoost -> minte LSTM.

    All parameters are read directly from the DuckDB simulation data.
    """
    # Annual mean prevalence over year 9
    year9_start = INTERVENTION_DAY - 365
    if 'prevalence' in df.columns:
        mask_y9 = (df["abs_timesteps"].values >= year9_start) & (df["abs_timesteps"].values < INTERVENTION_DAY)
        prev_y9 = df.loc[mask_y9, "prevalence"].mean() if mask_y9.any() else 0.1
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
    routine_val = float(row_before.get('routine', 0.0))
    Q0 = float(row_before.get('Q0', 0.0))
    phi_bednets = float(row_before.get('phi_bednets', 0.0))
    seasonal = float(row_before.get('seasonal', 0.0))
    lsm = float(row_before.get('lsm', 0.0))

    try:
        # Step 1: estimint XGBoost -> EIR
        X = pd.DataFrame({
            'prev_y9': [prev_y9],
            'dn0_use': [dn0_current],
            'Q0': [Q0],
            'phi_bednets': [phi_bednets],
            'seasonal': [seasonal],
            'itn_use': [itn_current],
            'irs_use': [irs_current],
        })
        estimated_eir = run_xgb_model(X, FULL_MODEL)[0]

        # Step 2: minte LSTM -> trajectory
        scenario = create_scenarios(
            eir=[estimated_eir],
            dn0_use=[dn0_current],
            dn0_future=[dn0_future],
            Q0=[Q0],
            phi_bednets=[phi_bednets],
            seasonal=[seasonal],
            routine=[routine_val],
            itn_use=[itn_current],
            irs_use=[irs_current],
            itn_future=[itn_future],
            irs_future=[irs_future],
            lsm=[lsm]
        )

        results = run_malaria_emulator(
            scenarios=scenario,
            predictor=predictor,
            window_size=WINDOW_SIZE,
            device='cpu',
            time_steps=int(len(df) * WINDOW_SIZE),
            use_cache=True,
            benchmark=False
        )

        preds = results[predictor].values
        n_sim = len(df)
        if len(preds) < n_sim:
            preds = np.pad(preds, (0, n_sim - len(preds)), mode='edge')
        else:
            preds = preds[:n_sim]

        return preds, prev_y9, estimated_eir

    except Exception as e:
        print(f"      Warning: Pipeline prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return np.full(len(df), np.nan), prev_y9, np.nan


def predict_with_hbr_increase(df, predictor, mosquito_increase=0.20):
    """MINTverse pipeline with +X% mosquito density via HBR model.

    Returns (preds, eir_new) or (nans, nan) on failure.
    """
    year9_start = INTERVENTION_DAY - 365
    if 'prevalence' in df.columns:
        mask_y9 = (df["abs_timesteps"].values >= year9_start) & (df["abs_timesteps"].values < INTERVENTION_DAY)
        prev_y9 = df.loc[mask_y9, "prevalence"].mean() if mask_y9.any() else 0.1
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
    routine_val = float(row_before.get('routine', 0.0))
    Q0 = float(row_before.get('Q0', 0.0))
    phi_bednets = float(row_before.get('phi_bednets', 0.0))
    seasonal = float(row_before.get('seasonal', 0.0))
    lsm = float(row_before.get('lsm', 0.0))

    try:
        hbr_result = estimate_eir_with_mosquito_increase(
            prevalence=prev_y9,
            mosquito_increase=mosquito_increase,
            dn0_use=dn0_current, Q0=Q0, phi_bednets=phi_bednets,
            seasonal=seasonal, itn_use=itn_current, irs_use=irs_current,
        )
        eir_new = hbr_result["eir_new"]

        scenario = create_scenarios(
            eir=[eir_new],
            dn0_use=[dn0_current], dn0_future=[dn0_future],
            Q0=[Q0], phi_bednets=[phi_bednets], seasonal=[seasonal],
            routine=[routine_val],
            itn_use=[itn_current], irs_use=[irs_current],
            itn_future=[itn_future], irs_future=[irs_future],
            lsm=[lsm],
        )

        results = run_malaria_emulator(
            scenarios=scenario, predictor=predictor,
            window_size=WINDOW_SIZE, device='cpu',
            time_steps=int(len(df) * WINDOW_SIZE),
            use_cache=True, benchmark=False,
        )

        preds = results[predictor].values
        n_sim = len(df)
        if len(preds) < n_sim:
            preds = np.pad(preds, (0, n_sim - len(preds)), mode='edge')
        else:
            preds = preds[:n_sim]

        return preds, eir_new

    except Exception as e:
        print(f"      Warning: HBR prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return np.full(len(df), np.nan), np.nan


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_validation_param_indices():
    """Load split CSV and return sorted list of unique parameter_indices for validation."""
    split_df = pd.read_csv(SPLIT_CSV)
    val_df = split_df[split_df['split'] == 'validate']
    param_indices = sorted(val_df['parameter_index'].unique().tolist())
    print(f"Loaded {len(param_indices)} unique parameter_indices for validation "
          f"({len(val_df)} total rows)")
    return param_indices


# ── Validation loop ────────────────────────────────────────────────────────────

def run_validation(predictor, output_pdf):
    print(f"\n{'='*80}")
    print(f"VALIDATING: {predictor.upper()}")
    print(f"{'='*80}")

    param_indices = get_validation_param_indices()

    con = duckdb.connect(DB_PATH, read_only=True)
    con.execute("PRAGMA memory_limit='8GB'; PRAGMA threads=4;")

    summary_stats = []
    ood_scenarios = []

    with PdfPages(output_pdf) as pdf:
        for seq_num, param_idx in enumerate(param_indices, 1):
            # Get simulation indices for this parameter set
            sim_indices_df = con.execute(f"""
                SELECT DISTINCT simulation_index FROM {TABLE_NAME}
                WHERE parameter_index = {param_idx} ORDER BY simulation_index
            """).df()

            if len(sim_indices_df) == 0:
                continue

            sim_indices = sim_indices_df['simulation_index'].tolist()
            sim_idx = sim_indices[0]

            # Fetch first sim for OOD check and parameter extraction
            if predictor == 'prevalence':
                df = fetch_simulation_data_prevalence(con, param_idx, sim_idx)
                target_col = 'prevalence'
            else:
                df = fetch_simulation_data_cases(con, param_idx, sim_idx)
                target_col = 'cases'

            if len(df) == 0:
                continue

            # OOD filter: annual mean prevalence at year 9
            year9_start = INTERVENTION_DAY - 365
            mask_y9 = (df["abs_timesteps"].values >= year9_start) & (df["abs_timesteps"].values < INTERVENTION_DAY)
            prev_y9 = df.loc[mask_y9, "prevalence"].mean() if mask_y9.any() else None

            if prev_y9 is not None and prev_y9 < MIN_PREV_Y9:
                ood_scenarios.append({'param_idx': param_idx, 'prev_y9': prev_y9})
                continue

            # Extract static parameters from first sim
            row_before = df[df["abs_timesteps"] < INTERVENTION_DAY].iloc[-1] if (df["abs_timesteps"] < INTERVENTION_DAY).any() else df.iloc[0]
            scenario_params = {
                'eir': float(row_before.get('eir', 0.0)),
                'Q0': float(row_before.get('Q0', 0.0)),
                'phi_bednets': float(row_before.get('phi_bednets', 0.0)),
                'seasonal': float(row_before.get('seasonal', 0.0)),
                'dn0_use': float(row_before.get('dn0_use', 0.0)),
                'itn_use': float(row_before.get('itn_use', 0.0)),
                'irs_use': float(row_before.get('irs_use', 0.0)),
                'lsm': float(row_before.get('lsm', 0.0)),
            }

            print(f"Scenario {seq_num}/{len(param_indices)}: param_idx={param_idx} | EIR={scenario_params['eir']:.1f}")

            # Collect simulations
            all_y_true = []
            all_prev_true = []
            pipeline_preds = None
            pipeline_prev_preds = None
            estimated_eir = None
            # HBR results: list of (preds, eir_new) per increase level
            hbr_results = [None] * len(MOSQUITO_INCREASES)

            for sim_idx in sim_indices:
                if predictor == 'prevalence':
                    df = fetch_simulation_data_prevalence(con, param_idx, sim_idx)
                    target_col = 'prevalence'
                else:
                    df = fetch_simulation_data_cases(con, param_idx, sim_idx)
                    target_col = 'cases'

                if len(df) == 0:
                    continue

                all_y_true.append(df[target_col].values.astype(np.float32))

                if predictor == 'cases':
                    all_prev_true.append(df['prevalence'].values.astype(np.float32))

                # Pipeline predictions (once per scenario)
                if pipeline_preds is None:
                    pipeline_preds, prev_y9, estimated_eir = predict_with_pipeline(
                        df, predictor)

                    # Also get prevalence predictions for cases mode
                    if predictor == 'cases':
                        pipeline_prev_preds, _, _ = predict_with_pipeline(
                            df, 'prevalence')

                    # HBR at all mosquito increase levels
                    for mi_idx, mi in enumerate(MOSQUITO_INCREASES):
                        preds_mi, eir_mi = predict_with_hbr_increase(
                            df, predictor, mosquito_increase=mi)
                        hbr_results[mi_idx] = (preds_mi, eir_mi)

            if not all_y_true:
                continue

            # Compute display arrays
            T0 = len(all_y_true[0])
            days = np.arange(T0) * WINDOW_SIZE
            years_rel = days / 365.0
            year_2_idx = np.where(years_rel >= 2)[0][0] if (years_rel >= 2).any() else 0
            years_display = years_rel[year_2_idx:] - 2

            avg_y = np.mean([y[year_2_idx:] for y in all_y_true], axis=0)
            avg_pipeline = np.mean([pipeline_preds[year_2_idx:]], axis=0) if not np.all(np.isnan(pipeline_preds)) else np.full_like(avg_y, np.nan)

            # HBR display arrays per level
            avg_hbr_list = []
            for hr in hbr_results:
                if hr is not None and not np.all(np.isnan(hr[0])):
                    avg_hbr_list.append(hr[0][year_2_idx:])
                else:
                    avg_hbr_list.append(np.full_like(avg_y, np.nan))

            # Horizontal line values (yr 0-1 average)
            mask_0_1 = (years_display >= 0) & (years_display <= 1)

            if predictor == 'prevalence':
                hline_sim = np.nanmean(avg_y[mask_0_1])
                hline_pipeline = np.nanmean(avg_pipeline[mask_0_1]) if not np.all(np.isnan(avg_pipeline)) else np.nan
            else:
                avg_prev_sim = np.mean([p[year_2_idx:] for p in all_prev_true], axis=0)
                hline_sim = np.nanmean(avg_prev_sim[mask_0_1])
                hline_pipeline = np.nanmean(pipeline_prev_preds[year_2_idx:][mask_0_1]) if (
                    pipeline_prev_preds is not None and not np.all(np.isnan(pipeline_prev_preds))) else np.nan

            # Metrics
            if not np.all(np.isnan(avg_pipeline)):
                mae_pipeline = np.mean(np.abs(avg_pipeline - avg_y))
                rmse_pipeline = np.sqrt(np.mean((avg_pipeline - avg_y)**2))
            else:
                mae_pipeline, rmse_pipeline = np.nan, np.nan

            prev_error_sim = abs(prev_y9 - hline_sim) if not np.isnan(hline_sim) else np.nan
            prev_error_pipeline = abs(prev_y9 - hline_pipeline) if not np.isnan(hline_pipeline) else np.nan

            # HBR stats per level
            row_stats = {
                'param_idx': param_idx,
                'eir': scenario_params['eir'],
                'Q0': scenario_params['Q0'],
                'phi_bednets': scenario_params['phi_bednets'],
                'seasonal': scenario_params['seasonal'],
                'lsm': scenario_params['lsm'],
                'dn0_use': scenario_params['dn0_use'],
                'itn_use': scenario_params['itn_use'],
                'irs_use': scenario_params['irs_use'],
                'prev_y9': prev_y9,
                'estimated_eir': estimated_eir,
                'mae_pipeline': mae_pipeline,
                'rmse_pipeline': rmse_pipeline,
                'hline_sim': hline_sim,
                'hline_pipeline': hline_pipeline,
                'prev_error_sim': prev_error_sim,
                'prev_error_pipeline': prev_error_pipeline,
            }
            eir_parts = []
            for mi_idx, mi in enumerate(MOSQUITO_INCREASES):
                pct = int(mi * 100)
                hr = hbr_results[mi_idx]
                eir_val = hr[1] if hr is not None else np.nan
                row_stats[f'hbr_eir_{pct}pct'] = eir_val
                if predictor == 'prevalence' and not np.all(np.isnan(avg_hbr_list[mi_idx])):
                    row_stats[f'hbr_prev_{pct}pct'] = np.nanmean(avg_hbr_list[mi_idx][mask_0_1])
                else:
                    row_stats[f'hbr_prev_{pct}pct'] = np.nan
                eir_parts.append(f"+{pct}%={eir_val:.0f}" if not np.isnan(eir_val) else f"+{pct}%=N/A")

            summary_stats.append(row_stats)

            print(f"  MAE={mae_pipeline:.4f} | prev_y9={prev_y9:.3f} | base EIR={estimated_eir:.0f} | {' | '.join(eir_parts)}")

            # ── Plot ───────────────────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(12, 8))

            # Individual simulations
            for i, y_true in enumerate(all_y_true):
                ax.plot(years_display, y_true[year_2_idx:], color='gray', alpha=0.3,
                        linewidth=0.8, label='MalariaSim (individual)' if i == 0 else '')

            # Averages
            ax.plot(years_display, avg_y, 'k-', label='MalariaSim (avg)', linewidth=2.5)

            if not np.all(np.isnan(avg_pipeline)):
                ax.plot(years_display, avg_pipeline, color='green', linestyle='-',
                        label=f'MINTverse (MAE={mae_pipeline:.3f})',
                        linewidth=2.5, alpha=0.85)

            for mi_idx, mi in enumerate(MOSQUITO_INCREASES):
                if not np.all(np.isnan(avg_hbr_list[mi_idx])):
                    hr = hbr_results[mi_idx]
                    eir_val = hr[1] if hr is not None else np.nan
                    eir_str = f"{eir_val:.1f}" if not np.isnan(eir_val) else "N/A"
                    ax.plot(years_display, avg_hbr_list[mi_idx],
                            color=MOSQUITO_COLORS[mi_idx], linestyle='-',
                            label=f'+{mi:.0%} mosq (EIR={eir_str})',
                            linewidth=1.8, alpha=0.85)

            ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Intervention')

            # Horizontal prevalence lines (x=0 to x=1)
            if predictor == 'prevalence':
                target_ax = ax
            else:
                ax2 = ax.twinx()
                ax2.set_ylim(0, 1)
                ax2.set_ylabel("Prevalence", fontsize=10, color='gray')
                ax2.tick_params(axis='y', labelcolor='gray')
                target_ax = ax2

            # Input prevalence (blue dashed)
            if prev_y9 is not None and not np.isnan(prev_y9):
                target_ax.hlines(prev_y9, 0, 1, colors='blue', linestyles='dashed',
                                 linewidth=2.0, alpha=0.9,
                                 label=f'Input prev (prev_y9): {prev_y9:.3f}')

            # MalariaSim actual avg prevalence (black dashed)
            if not np.isnan(hline_sim):
                target_ax.hlines(hline_sim, 0, 1, colors='black', linestyles='dashed',
                                 linewidth=1.5, alpha=0.7,
                                 label=f'MalariaSim avg prev (yr 0-1): {hline_sim:.3f}')

            # MINTverse actual avg prevalence (green dashed)
            if not np.isnan(hline_pipeline):
                target_ax.hlines(hline_pipeline, 0, 1, colors='green', linestyles='dashed',
                                 linewidth=1.5, alpha=0.7,
                                 label=f'MINTverse avg prev (yr 0-1): {hline_pipeline:.3f}')

            # Red solid fill: gap between input (blue) and MalariaSim (black dashed)
            if prev_y9 is not None and not np.isnan(prev_y9) and not np.isnan(hline_sim):
                target_ax.fill_between([0, 1], prev_y9, hline_sim,
                                       color='red', alpha=0.15,
                                       label=f'Input vs MalariaSim: {prev_error_sim:.3f}')

            # Red hatched fill: gap between input (blue) and MINTverse (green dashed)
            if prev_y9 is not None and not np.isnan(prev_y9) and not np.isnan(hline_pipeline):
                target_ax.fill_between([0, 1], prev_y9, hline_pipeline,
                                       facecolor='red', alpha=0.10,
                                       hatch='///', edgecolor='red', linewidth=0.5,
                                       label=f'Input vs MINTverse: {prev_error_pipeline:.3f}')

            # Axis formatting
            if predictor == 'prevalence':
                ax.set_ylim(0, 1)
                ylabel = "Prevalence"
            else:
                ax.set_ylim(bottom=0)
                ylabel = "Cases per 1000 per day"

            ax.set_xlim(0, 4)
            ax.set_xlabel("Time (years)", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)

            seas_str = 'Y' if scenario_params['seasonal'] > 0.5 else 'N'
            est_eir_str = f"{estimated_eir:.1f}" if estimated_eir is not None and not np.isnan(estimated_eir) else "N/A"
            title = (f"Param {param_idx} | EIR={scenario_params['eir']:.1f} | "
                     f"Est={est_eir_str} | Q0={scenario_params['Q0']:.2f} | "
                     f"Seasonal={seas_str}\n")
            title += f"prev_y9: {prev_y9:.3f} | estimint v{estimint.__version__} + minte v{minte.__version__}"
            ax.set_title(title, fontsize=9, fontweight='bold')

            if predictor == 'cases':
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8)
            else:
                ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        # ── Summary page ────────────────────────────────────────────────────
        if summary_stats:
            df_summary = pd.DataFrame(summary_stats)
            cs = df_summary.dropna(subset=['mae_pipeline'])

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 11))

            # Panel 1: Input prev_y9 vs MINTverse estimated prev (scatter)
            ax1.scatter(cs['prev_y9'], cs['hline_pipeline'],
                       alpha=0.4, s=15, c='green', edgecolors='none', zorder=3)
            pmin = min(cs['prev_y9'].min(), cs['hline_pipeline'].min()) * 0.9
            pmax = max(cs['prev_y9'].max(), cs['hline_pipeline'].max()) * 1.1
            ax1.plot([pmin, pmax], [pmin, pmax], 'k--', alpha=0.5, linewidth=1.5, label='Perfect match')
            ax1.set_xlabel('Input Prevalence (prev_y9)', fontsize=11)
            ax1.set_ylabel('MINTverse Estimated Prevalence (yr 0-1)', fontsize=11)
            ax1.set_title('Prevalence Matching: Input vs MINTverse', fontsize=12, fontweight='bold')
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)

            # Panel 2: Histogram of prevalence errors
            errors = (cs['hline_pipeline'] - cs['prev_y9']).values
            ax2.hist(errors, bins=40, color='salmon', edgecolor='darkred', alpha=0.7)
            ax2.axvline(0, color='black', linewidth=1.5, linestyle='-')
            ax2.axvline(np.mean(errors), color='blue', linewidth=1.2, linestyle='--',
                        label=f'Mean: {np.mean(errors):+.4f}')
            ax2.axvline(np.median(errors), color='green', linewidth=1.2, linestyle=':',
                        label=f'Median: {np.median(errors):+.4f}')
            ax2.set_xlabel('Prevalence Error (MINTverse - input)', fontsize=11)
            ax2.set_ylabel('Count', fontsize=11)
            ax2.set_title('Distribution of Prevalence Errors', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3, axis='y')

            # Panel 3: Input prev_y9 vs MalariaSim avg prev (sanity check)
            ax3.scatter(cs['prev_y9'], cs['hline_sim'],
                       alpha=0.4, s=15, c='black', edgecolors='none', zorder=3)
            smin = min(cs['prev_y9'].min(), cs['hline_sim'].min()) * 0.9
            smax = max(cs['prev_y9'].max(), cs['hline_sim'].max()) * 1.1
            ax3.plot([smin, smax], [smin, smax], 'k--', alpha=0.5, linewidth=1.5, label='Perfect match')
            ax3.set_xlabel('Input Prevalence (prev_y9)', fontsize=11)
            ax3.set_ylabel('MalariaSim Avg Prevalence (yr 0-1)', fontsize=11)
            ax3.set_title('Sanity Check: Input vs MalariaSim', fontsize=12, fontweight='bold')
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)

            # Panel 4: MAE histogram
            mae_vals = cs['mae_pipeline'].values
            ax4.hist(mae_vals, bins=40, color='cornflowerblue', edgecolor='navy', alpha=0.7)
            ax4.axvline(np.mean(mae_vals), color='red', linewidth=1.5, linestyle='--',
                        label=f'Mean MAE: {np.mean(mae_vals):.4f}')
            ax4.axvline(np.median(mae_vals), color='green', linewidth=1.5, linestyle=':',
                        label=f'Median MAE: {np.median(mae_vals):.4f}')
            # Add stats annotation
            stats_text = (f"N = {len(mae_vals)}\n"
                         f"Mean = {np.mean(mae_vals):.4f}\n"
                         f"Median = {np.median(mae_vals):.4f}\n"
                         f"Max = {np.max(mae_vals):.4f}\n"
                         f"<0.01: {np.mean(mae_vals < 0.01):.0%}\n"
                         f"<0.05: {np.mean(mae_vals < 0.05):.0%}")
            ax4.text(0.95, 0.95, stats_text, transform=ax4.transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='right',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            ax4.set_xlabel(f'Mean Absolute Error ({predictor})', fontsize=11)
            ax4.set_ylabel('Count', fontsize=11)
            ax4.set_title(f'MAE Distribution ({predictor.upper()})', fontsize=12, fontweight='bold')
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3, axis='y')

            plt.suptitle(f'MINTverse Validation: {predictor.upper()} - estimint v{estimint.__version__} + minte v{minte.__version__}',
                        fontsize=12, fontweight='bold', y=0.995)
            plt.tight_layout(rect=[0, 0, 1, 0.99])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    con.close()
    print(f"\nPDF saved: {output_pdf}")
    return pd.DataFrame(summary_stats) if summary_stats else pd.DataFrame()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'#'*80}")
    print(f"# MINTverse FULL VALIDATION (estimint v{estimint.__version__} + minte v{minte.__version__})")
    print(f"{'#'*80}")

    # Prevalence
    df_prev = run_validation(
        'prevalence',
        f'{OUTPUT_DIR}/validation_prevalence.pdf',
    )

    clear_cache()

    # Cases
    df_cases = run_validation(
        'cases',
        f'{OUTPUT_DIR}/validation_cases.pdf',
    )

    # Save combined results CSV
    all_results = []
    if len(df_prev) > 0:
        df_prev_out = df_prev.copy()
        df_prev_out['predictor'] = 'prevalence'
        all_results.append(df_prev_out)
    if len(df_cases) > 0:
        df_cases_out = df_cases.copy()
        df_cases_out['predictor'] = 'cases'
        all_results.append(df_cases_out)

    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        csv_path = f'{OUTPUT_DIR}/validation_results.csv'
        df_all.to_csv(csv_path, index=False)
        print(f"\nResults CSV saved: {csv_path}")

    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")

    if len(df_prev) > 0:
        print(f"\nPrevalence:")
        print(f"  Scenarios validated: {len(df_prev)}")
        print(f"  Mean MAE: {df_prev['mae_pipeline'].mean():.4f}")
        print(f"  Median MAE: {df_prev['mae_pipeline'].median():.4f}")

    if len(df_cases) > 0:
        print(f"\nCases per 1000:")
        print(f"  Scenarios validated: {len(df_cases)}")
        print(f"  Mean MAE: {df_cases['mae_pipeline'].mean():.4f}")
        print(f"  Median MAE: {df_cases['mae_pipeline'].median():.4f}")

    print(f"\nOutputs:")
    print(f"  {OUTPUT_DIR}/validation_prevalence.pdf")
    print(f"  {OUTPUT_DIR}/validation_cases.pdf")
    print(f"  {OUTPUT_DIR}/validation_results.csv")


if __name__ == "__main__":
    main()
