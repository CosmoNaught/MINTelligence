#!/usr/bin/env python3
"""
HBR Pipeline Validation: run MalariaSim at adjusted EIRs and compare to MINT.

For 5 random parameter sets:
  - Baseline: MalariaSim from DuckDB vs MINT
  - -50% mosquitoes: run MalariaSim at HBR-adjusted EIR vs MINT
  - +50% mosquitoes: run MalariaSim at HBR-adjusted EIR vs MINT
"""

import os
import sys
import subprocess
import tempfile
import pickle
import json
import random
import argparse

import duckdb
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from estimint.run import run_xgb_model
from estimint.storage import load_xgb_model
from estimint.hbr import estimate_eir_with_mosquito_delta
import estimint
import minte
from minte import run_malaria_emulator, create_scenarios

print(f"Using estimint v{estimint.__version__}")
print(f"Using minte v{minte.__version__}")

# ── Config ───────────────────────────────────────────────────────────────────
DB_PATH = "/home/cosmo/Documents/Repos/MINT_DATA/malaria_simulations_4096.duckdb"
HBR_DB_PATH = "/home/cosmo/Documents/Repos/MINT_DATA/HBR_malaria_simulations_4096.duckdb"
VAL_RDS_DIR = "/home/cosmo/Documents/Repos/MINT_DATA/val_malsim"
VAL_DB_PATH = "/home/cosmo/Documents/Repos/MINT_DATA/val_malsim.duckdb"
TABLE_NAME = "simulation_results"
EDGE_CASE_CSV = "/home/cosmo/Documents/Repos/MINTelligence/post/edge_case_test_matched.csv"
BEDNET_PARAMS_CSV = "/home/cosmo/Documents/Repos/minte-demo/data/bednet_params.csv"
RUN_MALARIASIM_R = "/home/cosmo/Documents/Repos/minte-demo/run_malariasim.R"
WINDOW_SIZE = 14
INTERVENTION_DAY = 9 * 365
EXCLUDED_CASES = {12, 15, 29}
MIN_PREV_Y9 = 0.02
N_CASES = 5
N_REPS = 1
SEED = 42

SCENARIOS = {
    "-50%": -0.50,
    "baseline": 0.0,
    "+50%": 0.50,
}
SCENARIO_COLORS = {
    "-50%": "tab:blue",
    "baseline": "black",
    "+50%": "tab:red",
}

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading XGBoost model from estimint...")
FULL_MODEL = load_xgb_model()
print("Model loaded")


# ── Fetch baseline MalariaSim from DuckDB ────────────────────────────────────
def fetch_sim_data(con, param_idx, sim_idx, predictor):
    """Fetch MalariaSim data from DuckDB, windowed at 14 days."""
    last_6y = 6 * 365
    W = WINDOW_SIZE
    if predictor == "prevalence":
        metric_expr = "SUM(n_detect_lm_0_1825) / NULLIF(SUM(n_age_0_1825), 0) AS metric"
    else:
        metric_expr = "1000.0 * SUM(n_inc_clinical_0_36500) / NULLIF(SUM(n_age_0_36500), 0) AS metric"

    query = f"""
        WITH raw AS (
            SELECT timesteps AS abs_timesteps,
                CAST(n_detect_lm_0_1825 AS DOUBLE) AS n_detect_lm_0_1825,
                CAST(n_age_0_1825 AS DOUBLE) AS n_age_0_1825,
                CAST(n_inc_clinical_0_36500 AS DOUBLE) AS n_inc_clinical_0_36500,
                CAST(n_age_0_36500 AS DOUBLE) AS n_age_0_36500,
                eir, dn0_use, dn0_future, Q0, phi_bednets, seasonal,
                routine, itn_use, irs_use, itn_future, irs_future, lsm
            FROM {TABLE_NAME}
            WHERE parameter_index = {param_idx}
              AND simulation_index = {sim_idx}
              AND timesteps >= {last_6y}
        ),
        groups AS (
            SELECT FLOOR((abs_timesteps - {last_6y}) / {W}) AS gid,
                {metric_expr},
                SUM(n_detect_lm_0_1825) / NULLIF(SUM(n_age_0_1825), 0) AS prevalence,
                MIN(abs_timesteps) AS abs_timesteps,
                MAX(eir) AS eir, MAX(dn0_use) AS dn0_use, MAX(dn0_future) AS dn0_future,
                MAX(Q0) AS Q0, MAX(phi_bednets) AS phi_bednets, MAX(seasonal) AS seasonal,
                MAX(routine) AS routine, MAX(itn_use) AS itn_use, MAX(irs_use) AS irs_use,
                MAX(itn_future) AS itn_future, MAX(irs_future) AS irs_future, MAX(lsm) AS lsm
            FROM raw GROUP BY 1
        )
        SELECT ROW_NUMBER() OVER (ORDER BY gid) AS timesteps,
            abs_timesteps, metric, prevalence, eir, dn0_use, dn0_future, Q0, phi_bednets,
            seasonal, routine, itn_use, irs_use, itn_future, irs_future, lsm
        FROM groups ORDER BY gid
    """
    return con.execute(query).df()


def get_avg_sim(con, param_idx, predictor, n_sims=4):
    """Get average MalariaSim across reps."""
    all_metrics = []
    ref_df = None
    for si in range(1, n_sims + 1):
        df = fetch_sim_data(con, param_idx, si, predictor)
        if ref_df is None:
            ref_df = df
        all_metrics.append(df["metric"].values)

    min_len = min(len(m) for m in all_metrics)
    avg = np.mean([m[:min_len] for m in all_metrics], axis=0)
    ref_df = ref_df.iloc[:min_len].copy()
    ref_df["metric"] = avg
    return ref_df


# ── Run MalariaSim via R ─────────────────────────────────────────────────────
def run_malariasim_r(eir, params_from_db, bednet_params_csv, n_reps=4,
                     save_rds_as=None):
    """Run malariasimulation R package for a given EIR and return processed results.
    If save_rds_as is provided, copy the .rds to that path for persistent storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write scenario CSV
        scenario_csv = os.path.join(tmpdir, "scenario.csv")
        bp = pd.read_csv(bednet_params_csv)

        def lookup_bednet(dn0_val):
            idx = (bp["dn0"] - dn0_val).abs().idxmin()
            return bp.loc[idx, "rn0"], bp.loc[idx, "gamman"]

        rn0_use, gamman_use = lookup_bednet(params_from_db["dn0_use"])
        rn0_future, gamman_future = lookup_bednet(params_from_db["dn0_future"])

        scenario = pd.DataFrame([{
            "eir": eir,
            "dn0_use": params_from_db["dn0_use"],
            "dn0_future": params_from_db["dn0_future"],
            "rn0_use": rn0_use,
            "rn0_future": rn0_future,
            "gamman_use": gamman_use,
            "gamman_future": gamman_future,
            "Q0": params_from_db["Q0"],
            "phi_bednets": params_from_db["phi_bednets"],
            "seasonal": int(params_from_db["seasonal"]),
            "routine": int(params_from_db["routine"]),
            "itn_use": params_from_db["itn_use"],
            "irs_use": params_from_db["irs_use"],
            "itn_future": params_from_db["itn_future"],
            "irs_future": params_from_db["irs_future"],
            "lsm": params_from_db["lsm"],
        }])
        scenario.to_csv(scenario_csv, index=False)

        output_dir = os.path.join(tmpdir, "results")
        os.makedirs(output_dir, exist_ok=True)

        # Run R script
        cmd = [
            "Rscript", RUN_MALARIASIM_R,
            "--csv", scenario_csv,
            "--cores", "16",
            "--reps", str(n_reps),
            "--output", output_dir,
            "--bednet-params", bednet_params_csv,
        ]
        print(f"    Running MalariaSim (EIR={eir:.2f})...", end="", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f" FAILED")
            print(f"    STDERR: {result.stderr[:500]}")
            return None
        print(f" done")

        rds_file = os.path.join(output_dir, "simulation_results_1.rds")

        # Save .rds persistently if requested
        if save_rds_as:
            import shutil
            os.makedirs(os.path.dirname(save_rds_as), exist_ok=True)
            shutil.copy2(rds_file, save_rds_as)
            print(f"    Saved .rds -> {save_rds_as}")

        # Extract ALL columns (including mosquito) to CSV
        csv_out = os.path.join(tmpdir, "sim_output.csv")
        r_extract = f"""
        suppressPackageStartupMessages(library(data.table))
        obj <- readRDS("{rds_file}")
        results_list <- obj$outputs
        all_dfs <- list()
        for (k in seq_along(results_list)) {{
            r <- results_list[[k]]
            dt <- data.table(
                rep = k,
                timestep = r$timestep,
                n_detect_lm_0_1825 = r$n_detect_lm_0_1825,
                n_age_0_1825 = r$n_age_0_1825,
                n_inc_clinical_0_36500 = r$n_inc_clinical_0_36500,
                n_age_0_36500 = r$n_age_0_36500,
                EIR_Anopheles = r$EIR_Anopheles,
                total_M_Anopheles = r$total_M_Anopheles,
                Im_Anopheles_count = r$Im_Anopheles_count,
                Sm_Anopheles_count = r$Sm_Anopheles_count,
                Pm_Anopheles_count = r$Pm_Anopheles_count,
                n_bitten = r$n_bitten
            )
            all_dfs[[k]] <- dt
        }}
        combined <- rbindlist(all_dfs)
        fwrite(combined, "{csv_out}")
        """
        r_result = subprocess.run(
            ["Rscript", "-e", r_extract],
            capture_output=True, text=True, timeout=60
        )
        if r_result.returncode != 0:
            print(f"    R extract failed: {r_result.stderr[:500]}")
            return None

        sim_df = pd.read_csv(csv_out)
        return sim_df


def process_malariasim_output(sim_df, predictor, window_size=14):
    """Process raw MalariaSim output into windowed prevalence/cases, averaged across reps."""
    last_6y = 6 * 365
    all_rep_metrics = []

    for rep_id in sim_df["rep"].unique():
        rep_data = sim_df[sim_df["rep"] == rep_id].copy()
        rep_data = rep_data[rep_data["timestep"] >= last_6y].copy()
        rep_data["gid"] = ((rep_data["timestep"] - last_6y) // window_size).astype(int)

        grouped = rep_data.groupby("gid").agg({
            "timestep": "min",
            "n_detect_lm_0_1825": "sum",
            "n_age_0_1825": "sum",
            "n_inc_clinical_0_36500": "sum",
            "n_age_0_36500": "sum",
        }).reset_index()

        grouped["prevalence"] = grouped["n_detect_lm_0_1825"] / grouped["n_age_0_1825"].replace(0, np.nan)
        grouped["cases"] = 1000.0 * grouped["n_inc_clinical_0_36500"] / grouped["n_age_0_36500"].replace(0, np.nan)

        if predictor == "prevalence":
            all_rep_metrics.append(grouped["prevalence"].values)
        else:
            all_rep_metrics.append(grouped["cases"].values)

    min_len = min(len(m) for m in all_rep_metrics)
    avg = np.mean([m[:min_len] for m in all_rep_metrics], axis=0)

    # Build a reference df with abs_timesteps
    rep1 = sim_df[sim_df["rep"] == 1].copy()
    rep1 = rep1[rep1["timestep"] >= last_6y].copy()
    rep1["gid"] = ((rep1["timestep"] - last_6y) // window_size).astype(int)
    abs_ts = rep1.groupby("gid")["timestep"].min().values[:min_len]

    result_df = pd.DataFrame({
        "abs_timesteps": abs_ts,
        "metric": avg,
    })
    return result_df


# ── MINT predictions ─────────────────────────────────────────────────────────
def predict_mint(df_baseline, case_row, predictor, mosquito_delta=0.0):
    """Run MINT pipeline: prevalence→EIR (optionally via HBR)→emulator."""
    year9_start = INTERVENTION_DAY - 365
    mask_y9 = (df_baseline["abs_timesteps"].values >= year9_start) & \
              (df_baseline["abs_timesteps"].values < INTERVENTION_DAY)
    if mask_y9.any():
        prev_y9 = df_baseline.loc[mask_y9, "prevalence"].mean()
    else:
        idx = np.argmin(np.abs(df_baseline["abs_timesteps"].values - INTERVENTION_DAY))
        prev_y9 = df_baseline.iloc[idx]["prevalence"]

    row_before = df_baseline[df_baseline["abs_timesteps"] < INTERVENTION_DAY]
    row_before = row_before.iloc[-1] if len(row_before) > 0 else df_baseline.iloc[0]
    row_after = df_baseline[df_baseline["abs_timesteps"] >= INTERVENTION_DAY]
    row_after = row_after.iloc[0] if len(row_after) > 0 else df_baseline.iloc[-1]

    dn0_current = float(row_before["dn0_use"])
    dn0_future_val = float(row_after["dn0_future"])
    itn_current = float(row_before["itn_use"])
    itn_future_val = float(row_after["itn_future"])
    irs_current = float(row_before["irs_use"])
    irs_future_val = float(row_after["irs_future"])
    seasonal_val = 1.0 if "Seasonal" in case_row["description"] else 0.0

    X = pd.DataFrame({
        "prev_y9": [prev_y9],
        "dn0_use": [dn0_current],
        "Q0": [float(case_row["actual_Q0"])],
        "phi_bednets": [float(case_row["actual_phi"])],
        "seasonal": [seasonal_val],
        "itn_use": [itn_current],
        "irs_use": [irs_current],
    })
    estimated_eir = float(run_xgb_model(X, FULL_MODEL)[0])

    if mosquito_delta != 0:
        hbr_result = estimate_eir_with_mosquito_delta(
            prevalence=prev_y9,
            mosquito_delta=mosquito_delta,
            dn0_use=dn0_current,
            Q0=float(case_row["actual_Q0"]),
            phi_bednets=float(case_row["actual_phi"]),
            seasonal=seasonal_val,
            itn_use=itn_current,
            irs_use=irs_current,
        )
        estimated_eir = float(hbr_result["eir_new"])

    scenario = create_scenarios(
        eir=[estimated_eir],
        dn0_use=[dn0_current],
        dn0_future=[dn0_future_val],
        Q0=[float(case_row["actual_Q0"])],
        phi_bednets=[float(case_row["actual_phi"])],
        seasonal=[seasonal_val],
        routine=[0.0],
        itn_use=[itn_current],
        irs_use=[irs_current],
        itn_future=[itn_future_val],
        irs_future=[irs_future_val],
        lsm=[float(case_row.get("actual_lsm", 0.0))],
    )

    n_timesteps = len(df_baseline) * WINDOW_SIZE
    results = run_malaria_emulator(
        scenarios=scenario,
        predictor=predictor,
        window_size=WINDOW_SIZE,
        device="cpu",
        time_steps=n_timesteps,
        use_cache=True,
        benchmark=False,
    )

    preds = results[predictor].values
    n_sim = len(df_baseline)
    if len(preds) < n_sim:
        preds = np.pad(preds, (0, n_sim - len(preds)), mode="edge")
    else:
        preds = preds[:n_sim]

    return preds, estimated_eir, prev_y9


# ── Build DuckDB from saved validation .rds files ─────────────────────────────
def build_val_duckdb(all_data):
    """Build a DuckDB with all validation MalariaSim runs (including mosquito columns)."""
    import shutil

    print(f"\nBuilding validation DuckDB: {VAL_DB_PATH}")
    rds_files = sorted(Glob_path(VAL_RDS_DIR, "*.rds"))
    if not rds_files:
        print("  No .rds files found, skipping DuckDB build")
        return

    all_csvs = []
    for rds_path in rds_files:
        fname = os.path.basename(rds_path)
        # Parse case_id, predictor, scenario from filename
        # e.g. case37_prevalence_minus50pct.rds
        parts = fname.replace(".rds", "").split("_")
        case_id = int(parts[0].replace("case", ""))
        predictor = parts[1]
        scenario_raw = "_".join(parts[2:])

        csv_tmp = os.path.join(VAL_RDS_DIR, fname.replace(".rds", ".csv"))

        r_extract = f"""
        suppressPackageStartupMessages(library(data.table))
        obj <- readRDS("{rds_path}")
        results_list <- obj$outputs
        all_dfs <- list()
        for (k in seq_along(results_list)) {{
            r <- results_list[[k]]
            dt <- data.table(
                simulation_index = k,
                timesteps = r$timestep,
                n_detect_lm_0_1825 = r$n_detect_lm_0_1825,
                n_age_0_1825 = r$n_age_0_1825,
                n_inc_clinical_0_36500 = r$n_inc_clinical_0_36500,
                n_age_0_36500 = r$n_age_0_36500,
                n_bitten = r$n_bitten,
                EIR_Anopheles = r$EIR_Anopheles,
                total_M_Anopheles = r$total_M_Anopheles,
                Im_Anopheles_count = r$Im_Anopheles_count,
                Sm_Anopheles_count = r$Sm_Anopheles_count,
                Pm_Anopheles_count = r$Pm_Anopheles_count
            )
            all_dfs[[k]] <- dt
        }}
        combined <- rbindlist(all_dfs)
        fwrite(combined, "{csv_tmp}")
        """
        result = subprocess.run(
            ["Rscript", "-e", r_extract],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            print(f"  Failed to extract {fname}: {result.stderr[:200]}")
            continue

        df = pd.read_csv(csv_tmp)
        df["case_id"] = case_id
        df["predictor"] = predictor
        df["scenario"] = scenario_raw
        all_csvs.append(df)
        os.remove(csv_tmp)  # clean up temp csv

    if not all_csvs:
        print("  No data extracted, skipping DuckDB build")
        return

    combined = pd.concat(all_csvs, ignore_index=True)

    # Look up the EIR used for each case/scenario from all_data
    eir_lookup = {}
    for pred in ["prevalence", "cases"]:
        if pred in all_data:
            for cd in all_data[pred]:
                for label, sr in cd["scenario_results"].items():
                    if label != "baseline":
                        key = (cd["case_id"], pred,
                               label.replace("+", "plus").replace("-", "minus").replace("%", "pct"))
                        eir_lookup[key] = sr["eir"]

    combined["eir_used"] = combined.apply(
        lambda r: eir_lookup.get((r["case_id"], r["predictor"], r["scenario"]), np.nan),
        axis=1
    )

    # Also look up parameter_index
    param_lookup = {}
    for pred in ["prevalence", "cases"]:
        if pred in all_data:
            for cd in all_data[pred]:
                param_lookup[cd["case_id"]] = cd["parameter_index"]
    combined["parameter_index"] = combined["case_id"].map(param_lookup)

    # Write DuckDB
    if os.path.exists(VAL_DB_PATH):
        os.remove(VAL_DB_PATH)
    con = duckdb.connect(VAL_DB_PATH)
    con.execute("CREATE TABLE simulation_results AS SELECT * FROM combined")
    count = con.execute("SELECT COUNT(*) FROM simulation_results").fetchone()[0]
    con.close()
    print(f"  DuckDB saved: {VAL_DB_PATH} ({count} rows)")


def Glob_path(directory, pattern):
    """Simple glob helper."""
    import glob
    return glob.glob(os.path.join(directory, pattern))


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot", action="store_true", help="Replot from saved pickle data")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    edge_cases = pd.read_csv(EDGE_CASE_CSV)
    edge_cases = edge_cases[~edge_cases["case_id"].isin(EXCLUDED_CASES)].reset_index(drop=True)

    con = duckdb.connect(DB_PATH, read_only=True)

    # Filter OOD cases (prev_y9 < 0.02)
    valid_cases = []
    for _, row in edge_cases.iterrows():
        df_check = fetch_sim_data(con, int(row["parameter_index"]), 1, "prevalence")
        y9_start = INTERVENTION_DAY - 365
        mask = (df_check["abs_timesteps"].values >= y9_start) & \
               (df_check["abs_timesteps"].values < INTERVENTION_DAY)
        if mask.any():
            prev = df_check.loc[mask, "prevalence"].mean()
        else:
            prev = 0.0
        if prev >= MIN_PREV_Y9:
            valid_cases.append(row)
    valid_cases = pd.DataFrame(valid_cases)
    print(f"\n{len(valid_cases)} valid cases (prev_y9 >= {MIN_PREV_Y9})")

    # Pick 5 random, then swap case 9→13 and case 32→8
    selected = valid_cases.sample(n=N_CASES, random_state=SEED).reset_index(drop=True)
    REPLACEMENTS = {9: 13, 32: 8}
    for old_id, new_id in REPLACEMENTS.items():
        mask = selected["case_id"] == old_id
        if mask.any():
            new_row = valid_cases[valid_cases["case_id"] == new_id]
            if len(new_row) > 0:
                idx = selected.index[mask][0]
                selected.loc[idx] = new_row.iloc[0]
                print(f"  Replaced case {old_id} with case {new_id}")
    selected = selected.reset_index(drop=True)
    print(f"Selected {N_CASES} cases:")
    for _, row in selected.iterrows():
        print(f"  Case {int(row['case_id'])}: {row['description']} (param_idx={int(row['parameter_index'])}, EIR={row['actual_eir']:.2f})")

    pickle_path = os.path.join(OUTPUT_DIR, "hbr_val_data.pkl")

    if args.replot and os.path.exists(pickle_path):
        print(f"\n--replot: loading cached data from {pickle_path}")
        with open(pickle_path, "rb") as f:
            all_data = pickle.load(f)
    else:
        all_data = {}
        for predictor in ["prevalence", "cases"]:
            print(f"\n{'='*80}")
            print(f"VALIDATING: {predictor.upper()}")
            print(f"{'='*80}")

            all_data[predictor] = []

            for _, case_row in selected.iterrows():
                case_id = int(case_row["case_id"])
                param_idx = int(case_row["parameter_index"])
                desc = case_row["description"]

                print(f"\nCase {case_id}: {desc}")

                df_baseline = get_avg_sim(con, param_idx, predictor)

                params_row = con.execute(f"""
                    SELECT DISTINCT eir, dn0_use, dn0_future, Q0, phi_bednets, seasonal,
                        routine, itn_use, irs_use, itn_future, irs_future, lsm
                    FROM {TABLE_NAME} WHERE parameter_index = {param_idx} LIMIT 1
                """).fetchdf().iloc[0].to_dict()

                mint_baseline, eir_baseline, prev_y9 = predict_mint(
                    df_baseline, case_row, predictor, mosquito_delta=0.0
                )
                print(f"  Baseline: true EIR={params_row['eir']:.2f}, estimated EIR={eir_baseline:.2f}, prev_y9={prev_y9:.4f}")

                scenario_results = {}
                scenario_results["baseline"] = {
                    "malariasim": df_baseline["metric"].values,
                    "mint": mint_baseline,
                    "eir": eir_baseline,
                    "abs_ts": df_baseline["abs_timesteps"].values,
                }

                for label, mosq_inc in [("-50%", -0.50), ("+50%", 0.50)]:
                    mint_preds, adj_eir, _ = predict_mint(
                        df_baseline, case_row, predictor, mosquito_delta=mosq_inc
                    )
                    print(f"  {label}: adjusted EIR={adj_eir:.2f}")

                    params_adjusted = params_row.copy()
                    params_adjusted["eir"] = adj_eir

                    rds_name = f"case{case_id}_{predictor}_{label.replace('+','plus').replace('-','minus').replace('%','pct')}.rds"
                    rds_path = os.path.join(VAL_RDS_DIR, rds_name)

                    sim_raw = run_malariasim_r(
                        adj_eir, params_adjusted, BEDNET_PARAMS_CSV, n_reps=N_REPS,
                        save_rds_as=rds_path,
                    )

                    if sim_raw is not None:
                        sim_processed = process_malariasim_output(sim_raw, predictor, WINDOW_SIZE)
                        msim_vals = sim_processed["metric"].values
                        msim_ts = sim_processed["abs_timesteps"].values
                    else:
                        print(f"    WARNING: MalariaSim failed for {label}, skipping")
                        msim_vals = np.full_like(mint_preds, np.nan)
                        msim_ts = df_baseline["abs_timesteps"].values[:len(mint_preds)]

                    n = min(len(msim_vals), len(mint_preds))
                    scenario_results[label] = {
                        "malariasim": msim_vals[:n],
                        "mint": mint_preds[:n],
                        "eir": adj_eir,
                        "abs_ts": msim_ts[:n] if len(msim_ts) >= n else df_baseline["abs_timesteps"].values[:n],
                    }

                # Compute MAE for year 9+ (post-intervention)
                for label in ["baseline", "-50%", "+50%"]:
                    sr = scenario_results[label]
                    n = min(len(sr["malariasim"]), len(sr["mint"]))
                    ts = sr["abs_ts"][:n]
                    post_mask = ts >= INTERVENTION_DAY
                    if post_mask.any():
                        mae = np.nanmean(np.abs(sr["malariasim"][:n][post_mask] - sr["mint"][:n][post_mask]))
                    else:
                        mae = np.nan
                    sr["mae"] = mae
                    print(f"  {label:>10s} MAE (post-intervention): {mae:.4f} | EIR={sr['eir']:.2f}")

                all_data[predictor].append({
                    "case_id": case_id,
                    "description": desc,
                    "parameter_index": param_idx,
                    "true_eir": params_row["eir"],
                    "prev_y9": prev_y9,
                    "scenario_results": scenario_results,
                })

        # Save pickle for replot
        with open(pickle_path, "wb") as f:
            pickle.dump(all_data, f)
        print(f"\nPickle saved: {pickle_path}")

        con.close()

    # ── Build DuckDB from saved .rds files ──
    if not args.replot and os.path.exists(VAL_RDS_DIR):
        build_val_duckdb(all_data)

    # ── Generate plots and CSVs from all_data ──
    for predictor in ["prevalence", "cases"]:
        pdf_path = os.path.join(OUTPUT_DIR, f"hbr_val_{predictor}.pdf")
        csv_rows = []

        with PdfPages(pdf_path) as pdf:
            for case_data in all_data[predictor]:
                case_id = case_data["case_id"]
                desc = case_data["description"]
                param_idx = case_data["parameter_index"]
                true_eir = case_data["true_eir"]
                prev_y9 = case_data["prev_y9"]
                scenario_results = case_data["scenario_results"]

                for label in ["baseline", "-50%", "+50%"]:
                    sr = scenario_results[label]
                    csv_rows.append({
                        "case_id": case_id,
                        "description": desc,
                        "parameter_index": param_idx,
                        "predictor": predictor,
                        "scenario": label,
                        "mosquito_change": SCENARIOS[label],
                        "eir_used": sr["eir"],
                        "true_eir": true_eir,
                        "mae_post_intervention": sr["mae"],
                    })

                # ── Plot: single plot per case, all 3 scenarios overlaid ──
                # Only last 4 years (year 8-12, i.e. abs_ts >= 8*365)
                # Colors: blue=-50%, black=baseline, red=+50%
                # Solid=MalariaSim, dashed=emulator
                last_4y_start = 8 * 365

                fig, ax = plt.subplots(1, 1, figsize=(10, 6))

                mae_strs = []
                for label in ["-50%", "baseline", "+50%"]:
                    sr = scenario_results[label]
                    n = min(len(sr["malariasim"]), len(sr["mint"]))
                    ts = sr["abs_ts"][:n]

                    mask = ts >= last_4y_start
                    ts_4y = ts[mask]
                    msim_4y = sr["malariasim"][:n][mask]
                    mint_4y = sr["mint"][:n][mask]

                    x_years = (ts_4y - last_4y_start) / 365.0

                    color = SCENARIO_COLORS[label]
                    ax.plot(x_years, msim_4y, color=color, linestyle="-", linewidth=1.5,
                            label=f"{label} MalariaSim", alpha=0.85)
                    ax.plot(x_years, mint_4y, color=color, linestyle="--", linewidth=1.5,
                            label=f"{label} Emulator", alpha=0.85)

                    mae_strs.append(f"{label}: {sr['mae']:.4f}")

                # Intervention line at year 1 (year 9 = last_4y_start + 365)
                ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.5, label="Intervention (Y9)")

                ax.set_title(
                    f"Case {case_id}: {desc}\n"
                    f"True EIR={true_eir:.2f} | MAE: {', '.join(mae_strs)}",
                    fontsize=11, fontweight="bold"
                )
                ax.set_xlabel("Years after Year 8")
                ax.set_ylabel(predictor.capitalize())
                if predictor == "prevalence":
                    ax.set_ylim(0, 1)
                ax.legend(fontsize=8, loc="best", ncol=2)
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                pdf.savefig(fig, dpi=150)
                plt.close(fig)

        print(f"PDF saved: {pdf_path}")

        csv_path = os.path.join(OUTPUT_DIR, f"hbr_val_{predictor}.csv")
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        print(f"CSV saved: {csv_path}")

    # ── Summary ──
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    all_csv = pd.read_csv(os.path.join(OUTPUT_DIR, "hbr_val_prevalence.csv"))
    all_csv_cases = pd.read_csv(os.path.join(OUTPUT_DIR, "hbr_val_cases.csv"))
    combined = pd.concat([all_csv, all_csv_cases])
    summary = combined.groupby(["predictor", "scenario"])["mae_post_intervention"].mean()
    print(summary.to_string())
    print("\nDone!")


if __name__ == "__main__":
    main()
