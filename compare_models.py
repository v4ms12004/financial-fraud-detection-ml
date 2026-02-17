#!/usr/bin/env python3
"""
compare_models.py — Compare RandomForest vs XGBoost

Run:
    python compare_models.py

Requires:
    outputs_rf/rf_metrics.csv
    outputs_xgboost/xgb_metrics.csv
"""

import os
import sys
import pandas as pd

RF_METRICS = os.path.join("outputs_rf", "rf_metrics.csv")
XGB_METRICS = os.path.join("outputs_xgboost", "xgb_metrics.csv")
OUTDIR = "outputs_compare"


def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def load_metrics(path: str, model_name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing metrics file: {path}")

    df = pd.read_csv(path)

    # Ensure required columns exist
    required = {"precision", "recall", "f1", "auprc", "threshold"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    df = df.copy()

    # Overwrite or create model column safely
    df["model"] = model_name

    # Keep only standard columns in consistent order
    df = df[["model", "precision", "recall", "f1", "auprc", "threshold"]]

    return df


def main():
    ensure_outdir(OUTDIR)

    rf = load_metrics(RF_METRICS, "RandomForest")
    xgb = load_metrics(XGB_METRICS, "XGBoost")

    comparison = pd.concat([rf, xgb], ignore_index=True)

    # Sort by AUPRC first, then F1
    comparison = comparison.sort_values(["auprc", "f1"], ascending=[False, False]).reset_index(drop=True)

    best = comparison.iloc[0]

    # Save CSV
    comparison.to_csv(os.path.join(OUTDIR, "model_comparison.csv"), index=False)

    # Save Markdown table
    md_table = comparison.to_markdown(index=False)
    with open(os.path.join(OUTDIR, "model_comparison.md"), "w", encoding="utf-8") as f:
        f.write(md_table)

    # Save summary
    with open(os.path.join(OUTDIR, "report_summary.txt"), "w", encoding="utf-8") as f:
        f.write("Model Comparison Summary (Test Set)\n")
        f.write("==================================\n\n")
        f.write(comparison.to_string(index=False))
        f.write("\n\n")
        f.write(
            f"Best model by AUPRC: {best['model']} "
            f"(AUPRC={best['auprc']:.6f}, F1={best['f1']:.6f})\n"
        )

    print("\n=== Model Comparison (sorted by AUPRC) ===")
    print(comparison.to_string(index=False))
    print(f"\nBest model by AUPRC: {best['model']} (AUPRC={best['auprc']:.6f})")

    print("\nSaved to folder: outputs_compare/")


if __name__ == "__main__":
    sys.exit(main())