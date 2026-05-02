#!/usr/bin/env python3
"""
fraud_by_time_of_day.py
========================
Shows fraud rate (%) and average fraud transaction amount ($)
by time of day segment across the 24-hour cycle.

Time segments:
  Night     : 00:00 - 05:59  (steps mod 24 in 0-5)
  Morning   : 06:00 - 11:59  (steps mod 24 in 6-11)
  Afternoon : 12:00 - 17:59  (steps mod 24 in 12-17)
  Evening   : 18:00 - 23:59  (steps mod 24 in 18-23)

X-axis: Time of day segment (original step variable derived)
Left Y-axis: Fraud rate (%)
Right Y-axis: Average fraud amount ($)

Matches poster styling:
  - Background: #f6f6f6
  - Font: Arial 24pt

OUTPUT:
    outputs_eda/fraud_by_time_of_day.png

USAGE:
    python fraud_by_time_of_day.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_FILE = "transaction_data.csv"
OUTDIR   = "outputs_eda"
BG       = "#f6f6f6"
FONT     = "Arial"
FS       = 24

# Time of day segments
SEGMENTS = {
    "Night\n(00–06)":     range(0,  6),
    "Morning\n(06–12)":   range(6,  12),
    "Afternoon\n(12–18)": range(12, 18),
    "Evening\n(18–24)":   range(18, 24),
}

def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path   = os.path.join(script_dir, CSV_FILE)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    if "is_fraud" in df.columns:
        df = df.rename(columns={"is_fraud": "isfraud"})
    return df

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    plt.rcParams["font.family"] = FONT

    print("Loading data...")
    df = load_data()
    print(f"  Shape: {df.shape}")

    # Derive hour of day from step (step mod 24)
    df["hour"] = (df["step"] - 1) % 24

    # Assign time segment
    def assign_segment(hour):
        for seg, hours in SEGMENTS.items():
            if hour in hours:
                return seg
        return "Unknown"

    df["segment"] = df["hour"].apply(assign_segment)

    # Compute stats per segment
    seg_order = list(SEGMENTS.keys())
    stats = []
    for seg in seg_order:
        subset = df[df["segment"] == seg]
        total      = len(subset)
        n_fraud    = subset["isfraud"].sum()
        fraud_rate = n_fraud / total * 100 if total > 0 else 0
        avg_fraud_amt = subset[subset["isfraud"] == 1]["amount"].mean() if n_fraud > 0 else 0
        stats.append({
            "segment":      seg,
            "total":        total,
            "n_fraud":      n_fraud,
            "fraud_rate":   fraud_rate,
            "avg_fraud_amt": avg_fraud_amt / 1000,  # in thousands
        })
        print(f"  {seg.replace(chr(10),' '):<25} "
              f"Total: {total:>9,}  Fraud: {n_fraud:>5,}  "
              f"Rate: {fraud_rate:.4f}%  "
              f"Avg fraud amt: ${avg_fraud_amt:>12,.0f}")

    stats_df = pd.DataFrame(stats)
    x = np.arange(len(seg_order))

    # ── Dual axis chart ──
    fig, ax1 = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(BG)
    ax1.set_facecolor(BG)

    # Bars — fraud rate (left axis)
    bar_width = 0.5
    bars = ax1.bar(
        x, stats_df["fraud_rate"],
        width=bar_width,
        color="#2E86C1",
        edgecolor="none",
        zorder=3,
        label="Fraud Rate (%)"
    )

    # Rate labels on top of bars
    for bar, rate in zip(bars, stats_df["fraud_rate"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{rate:.3f}%",
            ha="center", va="bottom",
            fontsize=FS - 6, fontfamily=FONT,
            fontweight="bold", color="#2E86C1"
        )

    ax1.set_xlabel("Time of Day", fontsize=FS, fontfamily=FONT, labelpad=10)
    ax1.set_ylabel("Fraud Rate (%)", fontsize=FS, fontfamily=FONT,
                   color="#2E86C1", labelpad=10)
    ax1.tick_params(axis="y", labelsize=FS - 4, labelcolor="#2E86C1")
    ax1.set_xticks(x)
    ax1.set_xticklabels(seg_order, fontsize=FS - 2, fontfamily=FONT)
    ax1.set_ylim(0, max(stats_df["fraud_rate"]) * 1.35)
    ax1.grid(True, axis="y", color="#DDDDDD", lw=0.8, zorder=0)
    ax1.set_axisbelow(True)
    for sp in ax1.spines.values():
        sp.set_color("#CCCCCC")
        sp.set_linewidth(0.8)

    # Line — avg fraud amount (right axis)
    ax2 = ax1.twinx()
    ax2.set_facecolor(BG)
    ax2.plot(
        x, stats_df["avg_fraud_amt"],
        color="#C0392B", lw=2.5,
        marker="o", markersize=9,
        markerfacecolor="#C0392B",
        zorder=4,
        label="Avg. Fraud Amount"
    )

    # Amount labels next to dots
    for xi, amt in zip(x, stats_df["avg_fraud_amt"]):
        ax2.text(
            xi + 0.07,
            amt + 8,
            f"${amt:.0f}K",
            ha="left", va="bottom",
            fontsize=FS - 8, fontfamily=FONT,
            color="#C0392B", fontweight="bold"
        )

    ax2.set_ylabel("Avg. Fraud Amount ($000s)",
                   fontsize=FS, fontfamily=FONT,
                   color="#C0392B", labelpad=10)
    ax2.tick_params(axis="y", labelsize=FS - 4, labelcolor="#C0392B")
    ax2.set_ylim(0, max(stats_df["avg_fraud_amt"]) * 1.4)
    ax2.spines["right"].set_color("#C0392B")
    ax2.spines["right"].set_linewidth(0.8)
    for sp in ["top", "left", "bottom"]:
        ax2.spines[sp].set_visible(False)

    # Title
    ax1.set_title(
        "Fraud Rate & Average Fraud Amount by Time of Day",
        fontsize=FS, fontweight="bold",
        pad=14, fontfamily=FONT
    )

    # Combined legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        fontsize=FS - 6,
        framealpha=0.9,
        edgecolor="#f6f6f6",
        facecolor=BG,
        loc="upper left",
        prop={"family": FONT, "size": FS - 6}
    )

    plt.tight_layout()
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        OUTDIR,
        "fraud_by_time_of_day.png"
    )
    plt.savefig(out_path, dpi=250, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())