#!/usr/bin/env python3
"""
Latent-input conditioning ablation for the inverse-conditioned dynamics pipeline.

Holds the forward emulator (minte) FIXED and varies only the EIR value fed to it,
across four conditioning modes:

  oracle       R1  true simulator EIR (from the DB)            -> isolates forward error
  raw          R2  raw XGBoost EIR, no QMAP, no smoothing       -> inverse error propagation
  qmap         R3  QMAP-calibrated EIR, no smoothing            -> calibration alone
  qmap_smooth  R4  QMAP + PCHIP/Gaussian smoothing (DEPLOYED)   -> headline pipeline

Everything else (scenario construction, ground-truth averaging, MAE window) is copied
verbatim from run_full_validation.py so the numbers are directly comparable to
validation_results.csv. HBR and plotting are stripped (irrelevant to this ablation).

Output: ablation_results.csv  (long form: one row per param_idx x predictor x mode)

Run (from MINTelligence/):
  PYTHONPATH=/home/cosmo/Documents/Repos/estimint/src \
  /home/cosmo/Documents/Repos/minte/.venv/bin/python validation_plots/run_ablation.py [--limit N]
"""

import os
import argparse
import numpy as np
import pandas as pd
import duckdb
import xgboost as xgb

from estimint.run import run_xgb_model, _predict_direct
from estimint.storage import load_xgb_model
import estimint
import minte
from minte import run_malaria_emulator, create_scenarios, clear_cache

# ── Configuration (identical to run_full_validation.py) ─────────────────────────
DB_PATH = "/home/cosmo/Documents/Repos/MINT_DATA/malaria_simulations_4096.duckdb"
TABLE_NAME = "simulation_results"
SPLIT_CSV = "/home/cosmo/Documents/Repos/MINTelligence/cases/results_simple_stratified_cases/train_val_test_split.csv"
WINDOW_SIZE = 14
INTERVENTION_DAY = 9 * 365
OUTPUT_DIR = "/home/cosmo/Documents/Repos/MINTelligence/validation_plots"
MIN_PREV_Y9 = 0.02

MODES = ["oracle", "raw", "qmap", "qmap_smooth"]

print(f"estimint {estimint.__file__}")
print(f"minte v{getattr(minte, '__version__', '?')}")
print("Loading XGBoost model from estimint...")
FULL_MODEL = load_xgb_model()
FEATURES = FULL_MODEL["features"]
print(f"Model loaded. features={FEATURES}")


# ── Data fetching (verbatim from run_full_validation.py) ────────────────────────

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


# ── Observables + per-mode EIR + forward pass ───────────────────────────────────

def build_observables(df):
    """Extract the deployment observables (verbatim logic from predict_with_pipeline)."""
    year9_start = INTERVENTION_DAY - 365
    if 'prevalence' in df.columns:
        mask_y9 = (df["abs_timesteps"].values >= year9_start) & (df["abs_timesteps"].values < INTERVENTION_DAY)
        prev_y9 = df.loc[mask_y9, "prevalence"].mean() if mask_y9.any() else 0.1
    else:
        prev_y9 = 0.1

    row_before = df[df["abs_timesteps"] < INTERVENTION_DAY].iloc[-1] if (df["abs_timesteps"] < INTERVENTION_DAY).any() else df.iloc[0]
    row_after = df[df["abs_timesteps"] >= INTERVENTION_DAY].iloc[0] if (df["abs_timesteps"] >= INTERVENTION_DAY).any() else df.iloc[-1]

    obs = dict(
        prev_y9=prev_y9,
        dn0_current=float(row_before.get('dn0_use', 0.0)),
        dn0_future=float(row_after.get('dn0_future', 0.0)),
        itn_current=float(row_before.get('itn_use', 0.0)),
        itn_future=float(row_after.get('itn_future', 0.0)),
        irs_current=float(row_before.get('irs_use', 0.0)),
        irs_future=float(row_after.get('irs_future', 0.0)),
        routine_val=float(row_before.get('routine', 0.0)),
        Q0=float(row_before.get('Q0', 0.0)),
        phi_bednets=float(row_before.get('phi_bednets', 0.0)),
        seasonal=float(row_before.get('seasonal', 0.0)),
        lsm=float(row_before.get('lsm', 0.0)),
        true_eir=float(row_before.get('eir', 0.0)),
    )
    return obs


def eir_for_mode(mode, obs):
    """Compute the EIR scalar fed to the forward emulator for a given conditioning mode."""
    X_df = pd.DataFrame({
        'prev_y9': [obs['prev_y9']],
        'dn0_use': [obs['dn0_current']],
        'Q0': [obs['Q0']],
        'phi_bednets': [obs['phi_bednets']],
        'seasonal': [obs['seasonal']],
        'itn_use': [obs['itn_current']],
        'irs_use': [obs['irs_current']],
    })
    if mode == 'oracle':
        return float(obs['true_eir'])
    Xmat = X_df[FEATURES].values.astype(np.float64)
    if mode == 'raw':
        log10 = FULL_MODEL["booster"].predict(xgb.DMatrix(Xmat))
        return float(np.power(10.0, log10)[0])
    if mode == 'qmap':
        return float(_predict_direct(Xmat, FULL_MODEL)[0])
    if mode == 'qmap_smooth':
        return float(run_xgb_model(X_df, FULL_MODEL)[0])
    raise ValueError(f"unknown mode {mode}")


def forward(eir_used, obs, predictor, n_df):
    scenario = create_scenarios(
        eir=[eir_used],
        dn0_use=[obs['dn0_current']],
        dn0_future=[obs['dn0_future']],
        Q0=[obs['Q0']],
        phi_bednets=[obs['phi_bednets']],
        seasonal=[obs['seasonal']],
        routine=[obs['routine_val']],
        itn_use=[obs['itn_current']],
        irs_use=[obs['irs_current']],
        itn_future=[obs['itn_future']],
        irs_future=[obs['irs_future']],
        lsm=[obs['lsm']],
    )
    results = run_malaria_emulator(
        scenarios=scenario, predictor=predictor,
        window_size=WINDOW_SIZE, device='cpu',
        time_steps=int(n_df * WINDOW_SIZE),
        use_cache=True, benchmark=False,
    )
    preds = results[predictor].values
    if len(preds) < n_df:
        preds = np.pad(preds, (0, n_df - len(preds)), mode='edge')
    else:
        preds = preds[:n_df]
    return preds


# ── Validation loop ─────────────────────────────────────────────────────────────

def get_validation_param_indices():
    split_df = pd.read_csv(SPLIT_CSV)
    val_df = split_df[split_df['split'] == 'validate']
    return sorted(val_df['parameter_index'].unique().tolist())


def run_validation(predictor, limit=None):
    print(f"\n{'='*70}\nABLATION: {predictor.upper()}\n{'='*70}")
    param_indices = get_validation_param_indices()
    if limit:
        param_indices = param_indices[:limit]

    con = duckdb.connect(DB_PATH, read_only=True)
    con.execute("PRAGMA memory_limit='8GB'; PRAGMA threads=4;")

    rows = []
    for seq_num, param_idx in enumerate(param_indices, 1):
        sim_indices_df = con.execute(f"""
            SELECT DISTINCT simulation_index FROM {TABLE_NAME}
            WHERE parameter_index = {param_idx} ORDER BY simulation_index
        """).df()
        if len(sim_indices_df) == 0:
            continue
        sim_indices = sim_indices_df['simulation_index'].tolist()

        fetch = fetch_simulation_data_prevalence if predictor == 'prevalence' else fetch_simulation_data_cases
        target_col = 'prevalence' if predictor == 'prevalence' else 'cases'

        df0 = fetch(con, param_idx, sim_indices[0])
        if len(df0) == 0:
            continue

        # OOD filter (identical to harness): annual mean prevalence at year 9
        year9_start = INTERVENTION_DAY - 365
        mask_y9 = (df0["abs_timesteps"].values >= year9_start) & (df0["abs_timesteps"].values < INTERVENTION_DAY)
        prev_y9_ood = df0.loc[mask_y9, "prevalence"].mean() if mask_y9.any() else None
        if prev_y9_ood is not None and prev_y9_ood < MIN_PREV_Y9:
            continue

        # Ground truth: average target across all sims, sliced from year_2_idx
        all_y_true = []
        for sim_idx in sim_indices:
            dfi = fetch(con, param_idx, sim_idx)
            if len(dfi) == 0:
                continue
            all_y_true.append(dfi[target_col].values.astype(np.float32))
        if not all_y_true:
            continue

        T0 = len(all_y_true[0])
        days = np.arange(T0) * WINDOW_SIZE
        years_rel = days / 365.0
        year_2_idx = np.where(years_rel >= 2)[0][0] if (years_rel >= 2).any() else 0
        avg_y = np.mean([y[year_2_idx:] for y in all_y_true], axis=0)

        obs = build_observables(df0)
        n_df = len(df0)

        base = dict(
            param_idx=param_idx, predictor=predictor,
            eir=obs['true_eir'], Q0=obs['Q0'], phi_bednets=obs['phi_bednets'],
            seasonal=obs['seasonal'], lsm=obs['lsm'], dn0_use=obs['dn0_current'],
            itn_use=obs['itn_current'], irs_use=obs['irs_current'], prev_y9=obs['prev_y9'],
        )

        line = [f"param={param_idx} EIR={obs['true_eir']:.1f}"]
        for mode in MODES:
            try:
                eir_used = eir_for_mode(mode, obs)
                preds = forward(eir_used, obs, predictor, n_df)
                avg_pred = preds[year_2_idx:]
                if np.all(np.isnan(avg_pred)):
                    mae = rmse = np.nan
                else:
                    mae = float(np.mean(np.abs(avg_pred - avg_y)))
                    rmse = float(np.sqrt(np.mean((avg_pred - avg_y) ** 2)))
            except Exception as e:
                eir_used, mae, rmse = np.nan, np.nan, np.nan
                print(f"      [{mode}] failed: {e}")
            rows.append({**base, 'mode': mode, 'eir_used': eir_used, 'mae': mae, 'rmse': rmse})
            line.append(f"{mode}:eir={eir_used:.1f},mae={mae:.4f}")
        print(f"  {seq_num}/{len(param_indices)} " + " | ".join(line))

    con.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap number of param indices (smoke test)")
    ap.add_argument("--predictors", nargs="+", default=["prevalence", "cases"])
    ap.add_argument("--out", default=os.path.join(OUTPUT_DIR, "ablation_results.csv"))
    args = ap.parse_args()

    all_rows = []
    for predictor in args.predictors:
        all_rows += run_validation(predictor, limit=args.limit)
        clear_cache()

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} rows -> {args.out}")

    # Quick aggregate preview (mean MAE per predictor x mode)
    if len(df):
        piv = df.groupby(['predictor', 'mode'])['mae'].mean().unstack().reindex(columns=MODES)
        print("\nMean MAE by mode:")
        print(piv.to_string())


if __name__ == "__main__":
    main()
