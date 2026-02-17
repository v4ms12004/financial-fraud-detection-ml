#!/usr/bin/env python3
"""
eda_plots.py — Generate report-ready plots:
1) Fraud rate over time (step) + moving average
2) Fraud rate by transaction type

Run:
    python eda_plots.py

Requires:
    transaction_data.csv in the SAME folder as this script.

Outputs:
    outputs_eda/
      - fraud_rate_over_time.png
      - fraud_rate_by_type.png
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = "transaction_data.csv"
OUTDIR = "outputs_eda"
MOVING_AVG_WINDOW = 50   # steps window (adjust if you want smoother/rougher)


def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)


def apply_column_aliases(df):
    aliases = {
        "newbalanceorg": "newbalanceorig",
        "oldbalanceorig": "oldbalanceorg",
        "oldbalancedestination": "oldbalancedest",
        "newbalancedestination": "newbalancedest",
        "is_fraud": "isfraud",
        "fraud": "isfraud",
        "is_flagged_fraud": "isflaggedfraud",
    }
    rename_map = {c: aliases[c] for c in df.columns if c in aliases}
    return df.rename(columns=rename_map)


def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, CSV_FILE)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    df = apply_column_aliases(df)
    return df


def plot_fraud_rate_over_time(df):
    # Fraud rate per step
    by_step = df.groupby("step")["isfraud"].mean().reset_index(name="fraud_rate")
    by_step["fraud_rate_ma"] = by_step["fraud_rate"].rolling(MOVING_AVG_WINDOW, min_periods=1).mean()

    plt.figure()
    plt.plot(by_step["step"], by_step["fraud_rate"], label="Fraud rate")
    plt.plot(by_step["step"], by_step["fraud_rate_ma"], label=f"Moving avg (window={MOVING_AVG_WINDOW})")
    plt.xlabel("Step (time index)")
    plt.ylabel("Fraud rate")
    plt.title("Fraud Rate Over Time (Step)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fraud_rate_over_time.png"), dpi=200)
    plt.close()


def plot_fraud_rate_by_type(df):
    # Fraud rate per type
    by_type = df.groupby("type")["isfraud"].mean().sort_values(ascending=False)

    plt.figure()
    by_type.plot(kind="bar")
    plt.xlabel("Transaction type")
    plt.ylabel("Fraud rate")
    plt.title("Fraud Rate by Transaction Type")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fraud_rate_by_type.png"), dpi=200)
    plt.close()


def main():
    ensure_outdir(OUTDIR)
    df = load_data()

    # Minimal safety: ensure expected cols exist
    for c in ["step", "type", "isfraud"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    plot_fraud_rate_over_time(df)
    plot_fraud_rate_by_type(df)

    print(f"Saved EDA plots to: {OUTDIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())