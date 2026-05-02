#!/usr/bin/env python3
"""
confusion_matrix_chart.py
==========================
Generates a clean confusion matrix visual for the poster.
Uses exact values from the time-sorted XGBoost + Panel model run.

Values:
  TN: 1,268,101  (legitimate, correctly ignored)
  TP:     4,201  (fraud, correctly caught)
  FP:       169  (legitimate, wrongly flagged)
  FN:        53  (fraud, missed)

Matches poster styling:
  - Background: #f6f6f6
  - Font: Arial 24pt

OUTPUT:
    outputs_compare/confusion_matrix_chart.png

USAGE:
    python confusion_matrix_chart.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = "outputs_compare"
BG     = "#f6f6f6"
FONT   = "Arial"
FS     = 24

# Exact values from XGBoost + Panel model (panel_temporal_xgb.py)
# Temporal split by unique time steps, 80/20
TN = 121_896
TP =   1_654
FP =      30
FN =       0

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    plt.rcParams["font.family"] = FONT

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_aspect("equal")

    # Cell colors
    # TN — light blue    (correct, large)
    # TP — strong green  (correct, fraud caught)
    # FP — amber         (wrong, false alarm)
    # FN — strong red    (wrong, fraud missed — worst outcome)
    cell_colors = [
        ["#AED6F1", "#E74C3C"],   # top row:    TN (predicted 0, actual 0), FN (predicted 0, actual 1)
        ["#F39C12", "#1E8449"],   # bottom row: FP (predicted 1, actual 0), TP (predicted 1, actual 1)
    ]

    # Values and labels for each cell
    # Layout: rows = Predicted, cols = Actual
    cells = [
        # (value_str, label_str, text_color)
        [(f"{TN:,}",  "True Negative\nLegitimate\ncorrectly ignored",    "#1A2530"),
         (f"{FN:,}",  "False Negative\nFraud\nmissed",                  "#FFFFFF")],
        [(f"{FP:,}",  "False Positive\nLegitimate\nwrongly flagged",     "#FFFFFF"),
         (f"{TP:,}",  "True Positive\nFraud\ncaught",                   "#FFFFFF")],
    ]

    n = 2
    for row in range(n):
        for col in range(n):
            # Draw cell rectangle
            rect = plt.Rectangle(
                [col, n - 1 - row], 1, 1,
                facecolor=cell_colors[row][col],
                edgecolor="#f6f6f6",
                linewidth=3,
                zorder=2
            )
            ax.add_patch(rect)

            val_str, lbl_str, txt_color = cells[row][col]

            # Big number
            ax.text(
                col + 0.5, n - 1 - row + 0.62,
                val_str,
                ha="center", va="center",
                fontsize=FS + 2,
                fontfamily=FONT,
                fontweight="bold",
                color=txt_color,
                zorder=3
            )
            # Label below number
            ax.text(
                col + 0.5, n - 1 - row + 0.25,
                lbl_str,
                ha="center", va="center",
                fontsize=FS - 10,
                fontfamily=FONT,
                color=txt_color,
                linespacing=1.4,
                zorder=3
            )

    # Axis labels
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(["Predicted\nNot Fraud", "Predicted\nFraud"],
                        fontsize=FS - 4, fontfamily=FONT)
    ax.set_yticklabels(["Actual\nFraud", "Actual\nNot Fraud"],
                        fontsize=FS - 4, fontfamily=FONT)
    ax.tick_params(length=0)

    ax.set_xlabel("Predicted Label", fontsize=FS, fontfamily=FONT, labelpad=12)
    ax.set_ylabel("Actual Label",    fontsize=FS, fontfamily=FONT, labelpad=12)
    #ax.set_title("Confusion Matrix — XGBoost + Panel Features",
    #             fontsize=FS, fontweight="bold", pad=16, fontfamily=FONT)

    for sp in ax.spines.values():
        sp.set_visible(False)

    # Summary stats below chart
    total       = TN + TP + FP + FN
    fraud_total = TP + FN
    recall_pct  = TP / fraud_total * 100
    prec_pct    = TP / (TP + FP)  * 100

    fig.text(
        0.5, -0.04,
        f"Recall: {recall_pct:.1f}%  ({TP:,} of {fraud_total:,} fraud cases caught)   |   "
        f"Precision: {prec_pct:.1f}%  ({FP:,} false alarms out of {TP+FP:,} flagged)",
        ha="center", va="top",
        fontsize=FS - 8,
        fontfamily=FONT,
        color="#555555"
    )

    plt.tight_layout()
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        OUTDIR,
        "confusion_matrix_chart.png"
    )
    plt.savefig(out_path, dpi=250, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())