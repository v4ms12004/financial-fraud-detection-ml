#!/usr/bin/env python3
"""
feature_importance_chart.py
============================
Generates a horizontal bar chart of XGBoost + Panel feature importance.
Uses values from outputs_synthetic_test/feature_importance_synthetic.csv
or falls back to hardcoded values from the project run.

Matches poster styling:
  - Background: #f6f6f6
  - Font: Arial 24pt

OUTPUT:
    outputs_panel_xgb/feature_importance_chart.png

USAGE:
    python feature_importance_chart.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

OUTDIR = "outputs_panel_xgb"
BG     = "#f6f6f6"
FONT   = "Arial"
FS     = 24

# Feature importance values from project run
# Source: outputs_synthetic_test/feature_importance_synthetic.csv
# Names converted from Python column names to readable labels
FEATURES = [
    ("Destination Account Age",           0.3845),
    ("Sender Prior Txn Count",            0.2766),
    ("Receiver Prior Txn Count",          0.0868),
    ("Dest. Account Type (Merchant)",     0.0592),
    ("Sender Account Age",                0.0552),
    ("Transaction Type: TRANSFER",        0.0414),
    ("Transaction Type: PAYMENT",         0.0171),
    ("Time Since Receiver's Last Txn",    0.0140),
    ("Sender Cumulative Amount Sent",      0.0128),
    ("Transaction Type: CASH_OUT",        0.0114),
    ("Sender Balance Change",             0.0108),
    ("Receiver Balance Change",           0.0095),
]

# Try loading from CSV if available — panel_xgb first (most recent definitive run)
CSV_PATHS = [
    os.path.join("outputs_panel_xgb",      "feature_importance_top30.csv"),
    os.path.join("outputs_synthetic_test", "feature_importance_synthetic.csv"),
]

NAME_MAP = {
    "dest_tenure":           "Destination Account Age",
    "orig_txn_count_prev":   "Sender Prior Txn Count",
    "dest_txn_count_prev":   "Receiver Prior Txn Count",
    "dest_prefix_M":         "Dest. Account Type (Merchant)",
    "orig_tenure":           "Sender Account Age",
    "type_TRANSFER":         "Transaction Type: TRANSFER",
    "type_PAYMENT":          "Transaction Type: PAYMENT",
    "dest_time_since_prev":  "Time Since Receiver's Last Txn",
    "orig_cum_amount_prev":  "Sender Cumulative Amount Sent",
    "type_CASH_OUT":         "Transaction Type: CASH_OUT",
    "delta_org":             "Sender Balance Change",
    "delta_dest":            "Receiver Balance Change",
    "abs_delta_dest":        "Abs. Receiver Balance Change",
    "oldbalancedest":        "Receiver Balance Before",
    "amount_over_oldorg":    "Amount / Sender Balance Ratio",
    "newbalanceorig":        "Sender Balance After",
    "newbalancedest":        "Receiver Balance After",
    "step":                  "Time Step",
    "dest_cum_in_amount_prev": "Receiver Cumulative Amount Recv.",
    "amount_over_olddest":   "Amount / Receiver Balance Ratio",
    "orig_time_since_prev":  "Time Since Sender's Last Txn",
    "dest_avg_in_amount_prev": "Receiver Avg. Amount Received",
    "oldbalanceorg":         "Sender Balance Before",
    "abs_delta_org":         "Abs. Sender Balance Change",
    "orig_avg_amount_prev":  "Sender Avg. Amount Sent",
    "amount":                "Transaction Amount",
    "org_balance_minus_amount_gap": "Balance-Amount Gap",
    "type_DEBIT":            "Transaction Type: DEBIT",
}

def load_features():
    for path in CSV_PATHS:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path  = os.path.join(script_dir, path)
        if os.path.exists(full_path):
            try:
                df = pd.read_csv(full_path, header=0, index_col=0)
                # CSV format is: unnamed index col, single value col
                # Reset so feature name becomes a column
                df = df.reset_index()
                df.columns = ["feature", "importance"]
                df["importance"] = pd.to_numeric(df["importance"], errors="coerce")
                df = df.dropna(subset=["importance"])
                df = df.sort_values("importance", ascending=False).head(12)
                df["feature"] = df["feature"].map(lambda x: NAME_MAP.get(x, x))
                print(f"  Loaded from {path}")
                print(f"  Top feature: {df.iloc[0]['feature']} ({df.iloc[0]['importance']:.4f})")
                return list(zip(df["feature"], df["importance"]))
            except Exception as e:
                print(f"  Failed to load {path}: {e}")
    print("  Using hardcoded values from project run.")
    return FEATURES

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    plt.rcParams["font.family"] = FONT

    features = load_features()

    # Sort ascending for horizontal bar (bottom = least important)
    features_sorted = sorted(features, key=lambda x: x[1])
    names  = [f[0] for f in features_sorted]
    values = [f[1] for f in features_sorted]
    pcts   = [v * 100 for v in values]

    n = len(names)
    y = np.arange(n)

    # Color scheme:
    # Top 3 features (highest importance) = dark teal  — the 75% story
    # Rest = steel blue
    colors = []
    top3_threshold = sorted(values, reverse=True)[2]
    for v in values:
        if v >= top3_threshold:
            colors.append("#1A5276")   # dark navy — top 3
        else:
            colors.append("#5DADE2")   # lighter blue — rest

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    bars = ax.barh(y, pcts, color=colors, edgecolor="none", height=0.65, zorder=3)

    # Value labels at end of each bar
    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center", ha="left",
            fontsize=FS - 6,
            fontfamily=FONT,
            color="#333333",
            fontweight="bold"
        )

    # Bracket annotation for top 3 = 75%
    top3_indices = [i for i, v in enumerate(values) if v >= top3_threshold]
    y_lo = min(top3_indices) - 0.4
    y_hi = max(top3_indices) + 0.4
    x_br = max(pcts) + 6.5

    ax.annotate(
        "",
        xy=(x_br, y_lo), xytext=(x_br, y_hi),
        arrowprops=dict(arrowstyle="-", color="#1A5276", lw=1.5)
    )
    ax.text(
        x_br + 0.8,
        (y_lo + y_hi) / 2,
        "75% of\ndecisions",
        va="center", ha="left",
        fontsize=FS - 8,
        fontfamily=FONT,
        color="#1A5276",
        fontweight="bold"
    )

    # Axes
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=FS - 2, fontfamily=FONT)
    ax.set_xlabel("Feature Importance (%)", fontsize=FS, fontfamily=FONT, labelpad=10)
    #ax.set_title("XGBoost + Panel Features — Feature Importance",
    #             fontsize=FS, fontweight="bold", pad=14, fontfamily=FONT)
    ax.tick_params(axis="x", labelsize=FS - 4)
    ax.set_xlim(0, max(pcts) + 12)

    ax.grid(True, axis="x", color="#DDDDDD", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color("#CCCCCC")
        sp.set_linewidth(0.8)

    # Legend for color coding
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1A5276", label="Top 3 features (75% combined)"),
        Patch(facecolor="#5DADE2", label="Remaining features"),
    ]
    ax.legend(
        handles=legend_elements,
        fontsize=FS - 6,
        framealpha=0.9,
        edgecolor="#CCCCCC",
        facecolor=BG,
        loc="lower right",
        prop={"family": FONT, "size": FS - 6}
    )

    plt.tight_layout()
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        OUTDIR,
        "feature_importance_chart.png"
    )
    plt.savefig(out_path, dpi=250, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())