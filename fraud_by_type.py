#!/usr/bin/env python3
"""
fraud_by_type.py
================
Generates a bar chart showing fraud vs non-fraud counts
by transaction type. Matches poster styling exactly:
  - Background: #f6f6f6
  - Font: Arial 24pt
  - White background save

OUTPUT:
    outputs_eda/fraud_by_transaction_type.png

USAGE:
    python fraud_by_type.py
    (transaction_data.csv must be in the same folder)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

CSV_FILE = "transaction_data.csv"
OUTDIR   = "outputs_eda"
BG       = "#f6f6f6"
FONT     = "Arial"
FS       = 24

def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path   = os.path.join(script_dir, CSV_FILE)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    if "isfraud" not in df.columns and "is_fraud" in df.columns:
        df = df.rename(columns={"is_fraud": "isfraud"})
    return df

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    plt.rcParams["font.family"] = FONT

    print("Loading data...")
    df = load_data()
    print(f"  Shape: {df.shape}")

    # Count fraud and non-fraud per transaction type
    grouped = df.groupby(["type", "isfraud"]).size().unstack(fill_value=0)
    grouped.columns = ["Non-Fraud", "Fraud"]
    grouped = grouped.loc[["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]]

    x      = np.arange(len(grouped))
    width  = 0.38
    labels = grouped.index.tolist()

    print("Transaction type breakdown:")
    for t in labels:
        total  = grouped.loc[t, "Non-Fraud"] + grouped.loc[t, "Fraud"]
        pct    = grouped.loc[t, "Fraud"] / total * 100
        print(f"  {t:<12} Non-Fraud: {grouped.loc[t,'Non-Fraud']:>9,}  "
              f"Fraud: {grouped.loc[t,'Fraud']:>6,}  ({pct:.2f}%)")

    # ── Chart ──
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    bars_nf = ax.bar(
        x - width / 2,
        grouped["Non-Fraud"],
        width,
        label="Non-Fraud",
        color="#4A90D9",
        edgecolor="none",
        zorder=3
    )
    bars_f = ax.bar(
        x + width / 2,
        grouped["Fraud"],
        width,
        label="Fraud",
        color="#C0392B",
        edgecolor="none",
        zorder=3
    )

    # Fraud count labels on top of red bars
    for bar, t in zip(bars_f, labels):
        val = grouped.loc[t, "Fraud"]
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 15000,
                f"{val:,}",
                ha="center", va="bottom",
                fontsize=FS - 6,
                fontfamily=FONT,
                color="#C0392B",
                fontweight="bold"
            )

    # Axes
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FS, fontfamily=FONT)
    ax.set_xlabel("Transaction Type", fontsize=FS, fontfamily=FONT, labelpad=10)
    ax.set_ylabel("Number of Transactions", fontsize=FS, fontfamily=FONT, labelpad=10)
    ax.set_title("Fraud vs Non-Fraud by Transaction Type",
                 fontsize=FS, fontweight="bold", pad=14, fontfamily=FONT)
    ax.tick_params(axis="y", labelsize=FS - 2)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6
                             else f"{int(v/1e3)}K" if v >= 1e3 else str(int(v)))
    )
    ax.grid(True, axis="y", color="#DDDDDD", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color("#CCCCCC")
        sp.set_linewidth(0.8)

    # Note about zero-fraud types — placed inside the plot top-left area
    ax.text(
        0.02, 0.97,
        "CASH_IN, DEBIT & PAYMENT\nhave zero fraud cases",
        transform=ax.transAxes,
        fontsize=FS - 8,
        fontfamily=FONT,
        color="#888888",
        fontstyle="italic",
        ha="left", va="top"
    )

    ax.legend(
        fontsize=FS - 2,
        framealpha=0.9,
        edgecolor="#CCCCCC",
        facecolor=BG,
        prop={"family": FONT, "size": FS - 2}
    )

    plt.tight_layout()
    out_path = os.path.join(OUTDIR, "fraud_by_transaction_type.png")
    plt.savefig(out_path, dpi=250, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())