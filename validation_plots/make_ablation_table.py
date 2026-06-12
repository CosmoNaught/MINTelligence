#!/usr/bin/env python3
"""
Reduce ablation_results.csv into the latent-input conditioning ablation table.

Rows  = conditioning modes (oracle / raw / qmap / qmap_smooth)
Cols  = prevalence MAE, incidence MAE, high-EIR incidence MAE, ITN incidence MAE
        (high-EIR incidence MAE reported at several thresholds so a defensible
         cut can be chosen; ITN stratum = itn_use > 0)

Pass-rates are deliberately NOT reported for incidence: incidence is on a
cases-per-1000-per-day scale, so the prevalence 0.05 tolerance is not meaningful
for it. Prevalence pass-rate (MAE < 0.05) is shown for reference only.
"""
import argparse
import numpy as np
import pandas as pd

MODES = ["oracle", "raw", "qmap", "qmap_smooth"]
MODE_LABEL = {
    "oracle": "True simulator EIR (oracle)",
    "raw": "Raw inverse EIR",
    "qmap": "QMAP inverse EIR",
    "qmap_smooth": "QMAP + smoothing (deployed)",
}
HIGH_EIR_THRESHOLDS = [50, 100, 200, 300]


def mae(df):
    s = df["mae"].dropna()
    return s.mean() if len(s) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/home/cosmo/Documents/Repos/MINTelligence/validation_plots/ablation_results.csv")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)

    prev = df[df["predictor"] == "prevalence"]
    inc = df[df["predictor"] == "cases"]

    print(f"n prevalence scenarios/mode: {prev.groupby('mode').size().to_dict()}")
    print(f"n incidence  scenarios/mode: {inc.groupby('mode').size().to_dict()}")
    print(f"n incidence with itn_use>0 : {(inc[inc['mode']=='oracle']['itn_use']>0).sum()}")
    for t in HIGH_EIR_THRESHOLDS:
        print(f"n incidence with eir>{t:>3}      : {(inc[inc['mode']=='oracle']['eir']>t).sum()}")
    print()

    # ── Console table ──
    header = f"{'mode':<28} {'prevMAE':>9} {'incMAE':>9} " + " ".join(f"{'inc>'+str(t):>9}" for t in HIGH_EIR_THRESHOLDS) + f" {'incITN':>9} {'prevPass':>9}"
    print(header)
    print("-" * len(header))
    table = {}
    for m in MODES:
        pm = prev[prev["mode"] == m]
        im = inc[inc["mode"] == m]
        prev_mae = mae(pm)
        inc_mae = mae(im)
        inc_high = {t: mae(im[im["eir"] > t]) for t in HIGH_EIR_THRESHOLDS}
        inc_itn = mae(im[im["itn_use"] > 0])
        prev_pass = (pm["mae"].dropna() < 0.05).mean()
        table[m] = dict(prev_mae=prev_mae, inc_mae=inc_mae, inc_high=inc_high, inc_itn=inc_itn, prev_pass=prev_pass)
        highs = " ".join(f"{inc_high[t]:>9.4f}" for t in HIGH_EIR_THRESHOLDS)
        print(f"{m:<28} {prev_mae:>9.4f} {inc_mae:>9.4f} {highs} {inc_itn:>9.4f} {prev_pass:>9.1%}")

    # ── LaTeX table (high-EIR threshold = 100; change HIGH_EIR_TEX below if desired) ──
    HIGH_EIR_TEX = 100
    print(f"\n%% ---- LaTeX (high-EIR = eir > {HIGH_EIR_TEX}) ----")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Conditioning input & Prevalence MAE & Incidence MAE & High-EIR inc.\ MAE & ITN inc.\ MAE \\")
    print(r"\midrule")
    for m in MODES:
        t = table[m]
        print(f"{MODE_LABEL[m]} & {t['prev_mae']:.4f} & {t['inc_mae']:.4f} & {t['inc_high'][HIGH_EIR_TEX]:.4f} & {t['inc_itn']:.4f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
