#!/usr/bin/env python3
"""
compare_pr_curves_zoomed.py
============================
Same as compare_pr_curves.py but produces a ZOOMED chart focusing
on the top-right corner (recall 0.85-1.0, precision 0.85-1.0)
where the three models actually differ.

Run this AFTER compare_pr_curves.py has already produced the metrics CSV,
OR run standalone — it will re-run all three models.

OUTPUT:
    outputs_compare/pr_curve_three_models_zoomed.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyArrowPatch

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve
from xgboost import XGBClassifier

# -------------------------------------------------------
# SETTINGS
# -------------------------------------------------------
CSV_FILE      = "transaction_data.csv"
OUTDIR        = "outputs_compare"
TRAIN_FRAC    = 0.80
RANDOM_STATE  = 42
N_TREES_RF    = 300
N_TREES_XGB   = 500
MAX_DEPTH     = 6
LEARNING_RATE = 0.08

# Zoom window — this is where the curves actually diverge
ZOOM_RECALL_MIN    = 0.85
ZOOM_PRECISION_MIN = 0.88

# -------------------------------------------------------
# Utilities (identical to compare_pr_curves.py)
# -------------------------------------------------------

def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path   = os.path.join(script_dir, CSV_FILE)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    aliases = {
        "newbalanceorg": "newbalanceorig", "oldbalanceorig": "oldbalanceorg",
        "is_fraud": "isfraud", "fraud": "isfraud",
        "is_flagged_fraud": "isflaggedfraud",
    }
    df = df.rename(columns={c: aliases[c] for c in df.columns if c in aliases})
    return df

def time_split(df, frac=0.80):
    df = df.sort_values("step").reset_index(drop=True)
    cut = int(len(df) * frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()

def scale_pos(y):
    return int((y == 0).sum()) / max(int((y == 1).sum()), 1)

def basic_features(df_raw):
    df = df_raw.copy()
    df = df.sort_values("step").reset_index(drop=True)
    df["delta_org"]  = df["newbalanceorig"] - df["oldbalanceorg"]
    df["delta_dest"] = df["newbalancedest"] - df["oldbalancedest"]
    df["abs_delta_org"]  = df["delta_org"].abs()
    df["abs_delta_dest"] = df["delta_dest"].abs()
    df["org_balance_minus_amount_gap"] = (
        (df["oldbalanceorg"] - df["newbalanceorig"]) - df["amount"]
    )
    df = df.drop(columns=["nameorig", "namedest", "isflaggedfraud"], errors="ignore")
    df = pd.get_dummies(df, columns=["type"], drop_first=True)
    return df

def panel_features(df_raw):
    df = df_raw.copy()
    df = df.sort_values("step", kind="mergesort").reset_index(drop=True)
    df["orig_prefix"] = df["nameorig"].astype(str).str.slice(0, 1)
    df["dest_prefix"] = df["namedest"].astype(str).str.slice(0, 1)
    orig_id, _ = pd.factorize(df["nameorig"], sort=False)
    dest_id, _ = pd.factorize(df["namedest"], sort=False)
    df["orig_id"] = orig_id.astype("int32")
    df["dest_id"] = dest_id.astype("int32")
    df["delta_org"]  = (df["newbalanceorig"] - df["oldbalanceorg"]).astype("float32")
    df["delta_dest"] = (df["newbalancedest"] - df["oldbalancedest"]).astype("float32")
    df["abs_delta_org"]  = df["delta_org"].abs()
    df["abs_delta_dest"] = df["delta_dest"].abs()
    df["org_balance_minus_amount_gap"] = (
        (df["oldbalanceorg"] - df["newbalanceorig"]) - df["amount"]
    ).astype("float32")
    df["amount_over_oldorg"]  = (df["amount"] / (df["oldbalanceorg"]  + 1.0)).astype("float32")
    df["amount_over_olddest"] = (df["amount"] / (df["oldbalancedest"] + 1.0)).astype("float32")
    df["orig_txn_count_prev"] = df.groupby("orig_id").cumcount().astype("int32")
    orig_cum = df.groupby("orig_id")["amount"].cumsum()
    df["orig_cum_amount_prev"] = (orig_cum - df["amount"]).astype("float32")
    df["orig_avg_amount_prev"] = (
        df["orig_cum_amount_prev"] / (df["orig_txn_count_prev"] + 1.0)
    ).astype("float32")
    df["orig_tenure"] = (
        df["step"] - df.groupby("orig_id")["step"].transform("min")
    ).astype("int32")
    df["orig_time_since_prev"] = (
        df["step"] - df.groupby("orig_id")["step"].shift(1)
    ).fillna(0).astype("int32")
    df["dest_txn_count_prev"] = df.groupby("dest_id").cumcount().astype("int32")
    dest_cum = df.groupby("dest_id")["amount"].cumsum()
    df["dest_cum_in_amount_prev"] = (dest_cum - df["amount"]).astype("float32")
    df["dest_avg_in_amount_prev"] = (
        df["dest_cum_in_amount_prev"] / (df["dest_txn_count_prev"] + 1.0)
    ).astype("float32")
    df["dest_tenure"] = (
        df["step"] - df.groupby("dest_id")["step"].transform("min")
    ).astype("int32")
    df["dest_time_since_prev"] = (
        df["step"] - df.groupby("dest_id")["step"].shift(1)
    ).fillna(0).astype("int32")
    df["type"]        = df["type"].astype("category")
    df["orig_prefix"] = df["orig_prefix"].astype("category")
    df["dest_prefix"] = df["dest_prefix"].astype("category")
    df = pd.get_dummies(df, columns=["type", "orig_prefix", "dest_prefix"], drop_first=True)
    df = df.drop(
        columns=["nameorig", "namedest", "isflaggedfraud", "orig_id", "dest_id"],
        errors="ignore"
    )
    return df

# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print("Loading data...")
    df_raw = load_data()
    print(f"  Shape: {df_raw.shape} | Fraud: {df_raw['isfraud'].sum():,}")

    print("\nBuilding basic features...")
    df_basic = basic_features(df_raw)
    train_b, test_b = time_split(df_basic)
    X_tr_b = train_b.drop(columns=["isfraud"])
    y_tr_b = train_b["isfraud"].astype(int).values
    X_te_b = test_b.drop(columns=["isfraud"])
    y_te_b = test_b["isfraud"].astype(int).values

    print("Building panel features...")
    df_panel = panel_features(df_raw)
    train_p, test_p = time_split(df_panel)
    X_tr_p = train_p.drop(columns=["isfraud"])
    y_tr_p = train_p["isfraud"].astype(int).values
    X_te_p = test_p.drop(columns=["isfraud"])
    y_te_p = test_p["isfraud"].astype(int).values

    print("\n[1/3] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=N_TREES_RF, class_weight="balanced_subsample",
        random_state=RANDOM_STATE, n_jobs=-1, max_features="sqrt",
    )
    rf.fit(X_tr_b, y_tr_b)
    prob_rf  = rf.predict_proba(X_te_b)[:, 1]
    auprc_rf = average_precision_score(y_te_b, prob_rf)
    p_rf, r_rf, _ = precision_recall_curve(y_te_b, prob_rf)
    print(f"  AUPRC: {auprc_rf:.6f}")

    print("\n[2/3] Training XGBoost baseline...")
    xgb_b = XGBClassifier(
        n_estimators=N_TREES_XGB, max_depth=MAX_DEPTH, learning_rate=LEARNING_RATE,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        scale_pos_weight=scale_pos(y_tr_b), eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist",
    )
    xgb_b.fit(X_tr_b, y_tr_b)
    prob_xgb_b  = xgb_b.predict_proba(X_te_b)[:, 1]
    auprc_xgb_b = average_precision_score(y_te_b, prob_xgb_b)
    p_xgb_b, r_xgb_b, _ = precision_recall_curve(y_te_b, prob_xgb_b)
    print(f"  AUPRC: {auprc_xgb_b:.6f}")

    print("\n[3/3] Training XGBoost + Panel Features...")
    xgb_p = XGBClassifier(
        n_estimators=N_TREES_XGB, max_depth=MAX_DEPTH, learning_rate=LEARNING_RATE,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        scale_pos_weight=scale_pos(y_tr_p), eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist",
    )
    xgb_p.fit(X_tr_p, y_tr_p)
    prob_xgb_p  = xgb_p.predict_proba(X_te_p)[:, 1]
    auprc_xgb_p = average_precision_score(y_te_p, prob_xgb_p)
    p_xgb_p, r_xgb_p, _ = precision_recall_curve(y_te_p, prob_xgb_p)
    print(f"  AUPRC: {auprc_xgb_p:.6f}")

    # ── Single zoomed chart ──
    print("\nGenerating zoomed PR curve chart...")

    BG   = "#f6f6f6"
    FONT = "Helvetica Neue"
    FS   = 24

    plt.rcParams["font.family"] = FONT

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # Three model curves
    ax.plot(r_rf,    p_rf,
            color="#888888", lw=2.0, ls="--",
            label=f"Random Forest  (AUPRC = {auprc_rf:.4f})")
    ax.plot(r_xgb_b, p_xgb_b,
            color="#2E86C1", lw=2.2, ls="-.",
            label=f"XGBoost baseline  (AUPRC = {auprc_xgb_b:.4f})")
    ax.plot(r_xgb_p, p_xgb_p,
            color="#1A7A4A", lw=2.8, ls="-",
            label=f"XGBoost + Panel Features  (AUPRC = {auprc_xgb_p:.4f})")

    # Zoomed window
    ax.set_xlim(ZOOM_RECALL_MIN, 1.005)
    ax.set_ylim(ZOOM_PRECISION_MIN, 1.005)
    ax.set_xlabel("Recall",     fontsize=FS, fontfamily=FONT)
    ax.set_ylabel("Precision",  fontsize=FS, fontfamily=FONT)
    ax.set_title("Precision–Recall Curve: Model Comparison",
                 fontsize=FS, fontweight="bold", pad=14, fontfamily=FONT)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
    ax.tick_params(labelsize=FS)
    ax.grid(True, color="#DDDDDD", lw=0.8)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color("#CCCCCC")
        sp.set_linewidth(0.8)

    ax.legend(loc="lower left", fontsize=FS,
              framealpha=0.9, edgecolor="#CCCCCC",
              facecolor=BG,
              prop={"family": FONT, "size": FS})

    plt.tight_layout()
    out_path = os.path.join(OUTDIR, "pr_curve_three_models_zoomed.png")
    plt.savefig(out_path, dpi=250, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {out_path}")

    # Save metrics
    pd.DataFrame([
        {"model": "Random Forest",       "auprc": auprc_rf},
        {"model": "XGBoost (baseline)",  "auprc": auprc_xgb_b},
        {"model": "XGBoost + Panel",     "auprc": auprc_xgb_p},
    ]).to_csv(os.path.join(OUTDIR, "three_model_metrics.csv"), index=False)

    print("\n" + "="*50)
    print(f"  Random Forest        AUPRC: {auprc_rf:.6f}")
    print(f"  XGBoost baseline     AUPRC: {auprc_xgb_b:.6f}")
    print(f"  XGBoost + Panel      AUPRC: {auprc_xgb_p:.6f}")
    print("="*50)
    print(f"\nSaved to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())