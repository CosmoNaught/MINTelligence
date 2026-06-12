#!/usr/bin/env python3
"""
Regenerate the headline + stratified breakdown tables from ablation_results.csv
for a chosen conditioning mode (default: qmap = QMAP-only, no smoothing).

Validation: run with --mode qmap_smooth and confirm the numbers match the
existing paper tables (tab:trajectory, tab:breakdown_*) before trusting the
qmap-mode regeneration.
"""
import argparse
import numpy as np
import pandas as pd

EIR_EDGES = [1, 3, 10, 30, 100, 300, np.inf]
EIR_LABELS = ["0--1", "1--3", "3--10", "10--30", "30--100", "100--300", "300+"]


def eir_bin(e):
    for edge, lab in zip(EIR_EDGES, EIR_LABELS):
        if e <= edge:
            return lab
    return EIR_LABELS[-1]


def intv_label(r):
    parts = []
    if r["itn_use"] > 0: parts.append("ITN")
    if r["irs_use"] > 0: parts.append("IRS")
    if r["lsm"] > 0: parts.append("LSM")
    return "None" if not parts else "{+}".join(parts)


def agg(df):
    s = df["mae"].dropna()
    return len(s), s.mean(), (s < 0.05).mean()


def breakdown(prev, inc, keyfn, order):
    prev = prev.copy(); inc = inc.copy()
    prev["k"] = prev.apply(keyfn, axis=1)
    inc["k"] = inc.apply(keyfn, axis=1)
    rows = []
    for k in order:
        n_p, mae_p, pass_p = agg(prev[prev["k"] == k])
        n_i, mae_i, pass_i = agg(inc[inc["k"] == k])
        rows.append((k, n_p, mae_p, pass_p, n_i, mae_i, pass_i))
    return rows


def print_block(title, rows):
    print(f"\n{title}")
    print(f"{'group':<16} {'nP':>4} {'prevMAE':>8} {'prevPass':>8}   {'nI':>4} {'incMAE':>8} {'incPass':>8}")
    for k, n_p, mae_p, pass_p, n_i, mae_i, pass_i in rows:
        print(f"{k:<16} {n_p:>4} {mae_p:>8.4f} {pass_p:>8.0%}   {n_i:>4} {mae_i:>8.4f} {pass_i:>8.0%}")


def latex_breakdown(rows, col1, caption, label):
    print(f"\n%% --- {label} ---")
    print(r"\begin{table}[!htbp]")
    print(r"    \centering")
    print(f"    \\caption{{{caption}}}")
    print(f"    \\label{{{label}}}")
    print(r"    \begin{tabular}{lccccccc}")
    print(r"        \toprule")
    print(r"        & \multicolumn{3}{c}{Prevalence} & & \multicolumn{3}{c}{Incidence} \\")
    print(r"        \cmidrule(lr){2-4} \cmidrule(lr){6-8}")
    print(f"        {col1} & $n$ & MAE & Pass & & $n$ & MAE & Pass \\\\")
    print(r"        \midrule")
    for k, n_p, mae_p, pass_p, n_i, mae_i, pass_i in rows:
        print(f"        {k} & {n_p} & {mae_p:.4f} & {pass_p:.0%} & & {n_i} & {mae_i:.4f} & {pass_i:.0%} \\\\".replace("%", r"\%"))
    print(r"        \bottomrule")
    print(r"    \end{tabular}")
    print(r"\end{table}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/home/cosmo/Documents/Repos/MINTelligence/validation_plots/ablation_results.csv")
    ap.add_argument("--mode", default="qmap")
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["mode"] == args.mode]
    prev = df[df["predictor"] == "prevalence"]
    inc = df[df["predictor"] == "cases"]

    # Headline
    n_p, mae_p, pass_p = agg(prev)
    n_i, mae_i, pass_i = agg(inc)
    print(f"=== mode = {args.mode} ===")
    print(f"HEADLINE  prevalence: n={n_p} MAE={mae_p:.4f} pass={pass_p:.1%}")
    print(f"HEADLINE  incidence : n={n_i} MAE={mae_i:.4f} pass={pass_i:.1%}")

    eir_rows = breakdown(prev, inc, lambda r: eir_bin(r["eir"]), EIR_LABELS)
    seas_rows = breakdown(prev, inc, lambda r: "Seasonal" if r["seasonal"] > 0.5 else "Perennial", ["Perennial", "Seasonal"])
    intv_order = ["None", "ITN", "IRS", "LSM", "ITN{+}IRS", "ITN{+}LSM", "IRS{+}LSM", "ITN{+}IRS{+}LSM"]
    intv_rows = breakdown(prev, inc, intv_label, intv_order)

    print_block("BY EIR BIN", eir_rows)
    print_block("BY SEASONALITY", seas_rows)
    print_block("BY INTERVENTION", intv_rows)

    if args.latex:
        latex_breakdown(eir_rows, "EIR bin",
                        r"Validation breakdown by EIR bin. $n$ = number of scenarios, MAE = mean absolute error, Pass = fraction with MAE $< 0.05$.",
                        "tab:breakdown_eir")
        latex_breakdown(seas_rows, "Setting", r"Validation breakdown by seasonality.", "tab:breakdown_season")
        latex_breakdown(intv_rows, "Intervention", r"Validation breakdown by intervention type.", "tab:breakdown_intv")


if __name__ == "__main__":
    main()
