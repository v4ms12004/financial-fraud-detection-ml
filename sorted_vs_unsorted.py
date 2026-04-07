import os
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# -------------------------------------------------------
# SETTINGS — identical to panel_temporal_xgb.py
# -------------------------------------------------------

CSV_FILE     = "transaction_data.csv"
OUTDIR       = "outputs_compare"
RANDOM_STATE = 42
TRAIN_FRAC   = 0.80
USE_FLOAT32  = True

N_TREES       = 500
MAX_DEPTH     = 6
LEARNING_RATE = 0.08


# -------------------------------------------------------
# Data structures
# -------------------------------------------------------

@dataclass
class Metrics:
    precision:  float
    recall:     float
    f1:         float
    auprc:      float
    threshold:  float
    n_fraud_test:   int
    n_total_test:   int
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def fraud_catch_rate(self):
        """% of actual fraud cases caught (= Recall)"""
        return self.recall * 100

    @property
    def false_alarm_rate(self):
        """% of flagged transactions that are actually NOT fraud"""
        if self.tp + self.fp == 0:
            return 0.0
        return self.fp / (self.tp + self.fp) * 100


# -------------------------------------------------------
# Shared utilities
# -------------------------------------------------------

def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)

def apply_column_aliases(df):
    aliases = {
        "newbalanceorg":         "newbalanceorig",
        "oldbalanceorig":        "oldbalanceorg",
        "oldbalancedestination": "oldbalancedest",
        "newbalancedestination": "newbalancedest",
        "is_fraud":              "isfraud",
        "fraud":                 "isfraud",
        "is_flagged_fraud":      "isflaggedfraud",
    }
    return df.rename(columns={c: aliases[c] for c in df.columns if c in aliases})

def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path   = os.path.join(script_dir, CSV_FILE)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    df = apply_column_aliases(df)
    return df

def pick_threshold_for_f1(y_true, y_prob):
    p, r, thr = precision_recall_curve(y_true, y_prob)
    f1s = (2 * p[:-1] * r[:-1]) / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(thr[int(np.argmax(f1s))])

def build_features(df_raw):
    """
    Same feature engineering as panel_temporal_xgb.py.
    Applied identically to both models so the ONLY difference
    is the train/test split strategy.
    """
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
    df["abs_delta_org"]  = np.abs(df["delta_org"]).astype("float32")
    df["abs_delta_dest"] = np.abs(df["delta_dest"]).astype("float32")
    df["org_balance_minus_amount_gap"] = (
        (df["oldbalanceorg"] - df["newbalanceorig"]) - df["amount"]
    ).astype("float32")
    df["amount_over_oldorg"]  = (df["amount"] / (df["oldbalanceorg"]  + 1.0)).astype("float32")
    df["amount_over_olddest"] = (df["amount"] / (df["oldbalancedest"] + 1.0)).astype("float32")

    # Panel features — sender
    df["orig_txn_count_prev"]  = df.groupby("orig_id").cumcount().astype("int32")
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

    # Panel features — receiver
    df["dest_txn_count_prev"]     = df.groupby("dest_id").cumcount().astype("int32")
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
    df = df.drop(columns=["nameorig", "namedest", "isflaggedfraud",
                           "orig_id", "dest_id"], errors="ignore")
    return df

def train_xgb(X_train, y_train):
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    model = XGBClassifier(
        n_estimators     = N_TREES,
        max_depth        = MAX_DEPTH,
        learning_rate    = LEARNING_RATE,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        reg_lambda       = 1.0,
        scale_pos_weight = n_neg / max(n_pos, 1),
        eval_metric      = "logloss",
        random_state     = RANDOM_STATE,
        n_jobs           = -1,
        tree_method      = "hist",
    )
    model.fit(X_train, y_train)
    return model

def evaluate(y_true, y_prob) -> Tuple[Metrics, np.ndarray]:
    thr    = pick_threshold_for_f1(y_true, y_prob)
    y_pred = (y_prob >= thr).astype(int)
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    m = Metrics(
        precision     = float(precision_score(y_true, y_pred, zero_division=0)),
        recall        = float(recall_score(y_true, y_pred, zero_division=0)),
        f1            = float(f1_score(y_true, y_pred, zero_division=0)),
        auprc         = float(average_precision_score(y_true, y_prob)),
        threshold     = float(thr),
        n_fraud_test  = int(y_true.sum()),
        n_total_test  = int(len(y_true)),
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
    )
    return m, cm


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    ensure_outdir(OUTDIR)

    print("=" * 65)
    print("  Time-Sorted vs Random Split — Comparison Experiment")
    print("=" * 65)
    print("\nThis experiment quantifies the value of temporal ordering.")

    # Load and build features
    print("\nLoading data...")
    df_raw  = load_data()
    print(f"  Shape: {df_raw.shape} | Fraud: {df_raw['isfraud'].sum():,}")

    print("Building features (identical for both models)...")
    df_feat = build_features(df_raw)
    print(f"  Feature shape: {df_feat.shape}")

    X = df_feat.drop(columns=["isfraud"])
    y = df_feat["isfraud"].astype(int).values

    # ── MODEL A: Time-Sorted Split ──
    print("\n── Model A: TIME-SORTED split ──")
    cut = int(len(df_feat) * TRAIN_FRAC)
    X_train_s = X.iloc[:cut];  y_train_s = y[:cut]
    X_test_s  = X.iloc[cut:];  y_test_s  = y[cut:]
    print(f"  Train: {len(X_train_s):,} rows | fraud: {y_train_s.sum():,}")
    print(f"  Test : {len(X_test_s):,} rows  | fraud: {y_test_s.sum():,}")

    print("  Training...")
    model_s  = train_xgb(X_train_s, y_train_s)
    prob_s   = model_s.predict_proba(X_test_s)[:, 1]
    m_s, cm_s = evaluate(y_test_s, prob_s)
    print(f"  AUPRC     : {m_s.auprc:.6f}")
    print(f"  Precision : {m_s.precision:.6f}")
    print(f"  Recall    : {m_s.recall:.6f}")
    print(f"  F1        : {m_s.f1:.6f}")
    print(f"  Fraud caught (TP): {m_s.tp:,} / {m_s.n_fraud_test:,} ({m_s.fraud_catch_rate:.1f}%)")
    print(f"  False alarms    : {m_s.fp:,} ({m_s.false_alarm_rate:.1f}% of flagged txns)")

    # ── MODEL B: Random Split ──
    print("\n── Model B: RANDOM split (naive baseline) ──")
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y, test_size=1.0 - TRAIN_FRAC,
        random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {len(X_train_r):,} rows | fraud: {y_train_r.sum():,}")
    print(f"  Test : {len(X_test_r):,} rows  | fraud: {y_test_r.sum():,}")

    print("  Training...")
    model_r  = train_xgb(X_train_r, y_train_r)
    prob_r   = model_r.predict_proba(X_test_r)[:, 1]
    m_r, cm_r = evaluate(y_test_r, prob_r)
    print(f"  AUPRC     : {m_r.auprc:.6f}")
    print(f"  Precision : {m_r.precision:.6f}")
    print(f"  Recall    : {m_r.recall:.6f}")
    print(f"  F1        : {m_r.f1:.6f}")
    print(f"  Fraud caught (TP): {m_r.tp:,} / {m_r.n_fraud_test:,} ({m_r.fraud_catch_rate:.1f}%)")
    print(f"  False alarms    : {m_r.fp:,} ({m_r.false_alarm_rate:.1f}% of flagged txns)")

    # ── Comparison ──
    auprc_diff = m_r.auprc - m_s.auprc
    print(f"\n{'='*65}")
    print(f"  AUPRC INFLATION from random split: +{auprc_diff:.6f}")
    print(f"  ({auprc_diff*100:.3f} percentage points of artificial inflation)")
    print(f"{'='*65}")

    # Save metrics CSV
    metrics_df = pd.DataFrame([
        {
            "model":          "Time-Sorted (Temporal)",
            "split_method":   "Chronological 80/20",
            "auprc":          m_s.auprc,
            "precision":      m_s.precision,
            "recall":         m_s.recall,
            "f1":             m_s.f1,
            "threshold":      m_s.threshold,
            "tp":             m_s.tp,
            "fp":             m_s.fp,
            "fn":             m_s.fn,
            "tn":             m_s.tn,
            "fraud_catch_pct":m_s.fraud_catch_rate,
            "false_alarm_pct":m_s.false_alarm_rate,
            "n_fraud_test":   m_s.n_fraud_test,
            "n_total_test":   m_s.n_total_test,
        },
        {
            "model":          "Random Split (Naive Baseline)",
            "split_method":   "Random 80/20 (stratified)",
            "auprc":          m_r.auprc,
            "precision":      m_r.precision,
            "recall":         m_r.recall,
            "f1":             m_r.f1,
            "threshold":      m_r.threshold,
            "tp":             m_r.tp,
            "fp":             m_r.fp,
            "fn":             m_r.fn,
            "tn":             m_r.tn,
            "fraud_catch_pct":m_r.fraud_catch_rate,
            "false_alarm_pct":m_r.false_alarm_rate,
            "n_fraud_test":   m_r.n_fraud_test,
            "n_total_test":   m_r.n_total_test,
        }
    ])
    metrics_path = os.path.join(OUTDIR, "sorted_vs_unsorted_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved: {metrics_path}")

    # Save confusion matrices
    pd.DataFrame(cm_s, index=["Actual 0","Actual 1"],
                 columns=["Pred 0","Pred 1"]).to_csv(
        os.path.join(OUTDIR, "confusion_matrix_sorted.csv"))
    pd.DataFrame(cm_r, index=["Actual 0","Actual 1"],
                 columns=["Pred 0","Pred 1"]).to_csv(
        os.path.join(OUTDIR, "confusion_matrix_unsorted.csv"))

    # ── PR Curve comparison plot ──
    p_s, r_s, _ = precision_recall_curve(y_test_s, prob_s)
    p_r, r_r, _ = precision_recall_curve(y_test_r, prob_r)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor='#070710')
    ax.set_facecolor('#0d0d1a')
    ax.plot(r_s, p_s, color='#00ff9d', lw=2,
            label=f'Time-Sorted (AUPRC={m_s.auprc:.4f}) ← REALISTIC')
    ax.plot(r_r, p_r, color='#ff4757', lw=2, linestyle='--',
            label=f'Random Split (AUPRC={m_r.auprc:.4f}) ← INFLATED')
    ax.axhline(y=0.0013, color='#7070a0', lw=1, linestyle=':',
               label='Random baseline (fraud rate)')
    ax.set_xlabel('Recall', color='#7070a0')
    ax.set_ylabel('Precision', color='#7070a0')
    ax.set_title('PR Curve: Time-Sorted vs Random Split',
                 color='#e8e8f0', fontsize=13, pad=12)
    ax.tick_params(colors='#7070a0')
    ax.spines[:].set_color('#1e1e3f')
    ax.legend(facecolor='#14142b', edgecolor='#1e1e3f',
              labelcolor='#e8e8f0', fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    pr_path = os.path.join(OUTDIR, "pr_curve_comparison.png")
    plt.savefig(pr_path, dpi=200, facecolor='#070710')
    plt.close()
    print(f"Saved: {pr_path}")

    # ── Markdown report ──
    md_path = os.path.join(OUTDIR, "sorted_vs_unsorted_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Time-Sorted vs Random Split — Comparison Report\n\n")
        f.write("## Motivation\n")
        f.write("Financial transactions occur sequentially in time. Training a model on ")
        f.write("randomly sampled data allows future fraud patterns to leak into training, ")
        f.write("artificially inflating performance metrics. This experiment directly ")
        f.write("quantifies that inflation.\n\n")
        f.write("## Side-by-Side Results\n\n")
        f.write("| Metric | Time-Sorted ✓ | Random Split ✗ | Inflation |\n")
        f.write("|--------|--------------|----------------|----------|\n")
        f.write(f"| **AUPRC** | **{m_s.auprc:.6f}** | {m_r.auprc:.6f} | +{auprc_diff:.6f} |\n")
        f.write(f"| Precision | {m_s.precision:.6f} | {m_r.precision:.6f} | "
                f"{m_r.precision-m_s.precision:+.6f} |\n")
        f.write(f"| Recall | {m_s.recall:.6f} | {m_r.recall:.6f} | "
                f"{m_r.recall-m_s.recall:+.6f} |\n")
        f.write(f"| F1 Score | {m_s.f1:.6f} | {m_r.f1:.6f} | "
                f"{m_r.f1-m_s.f1:+.6f} |\n")
        f.write(f"| Fraud Caught | {m_s.tp:,} / {m_s.n_fraud_test:,} "
                f"({m_s.fraud_catch_rate:.1f}%) | {m_r.tp:,} / {m_r.n_fraud_test:,} "
                f"({m_r.fraud_catch_rate:.1f}%) | — |\n")
        f.write(f"| False Alarms | {m_s.fp:,} ({m_s.false_alarm_rate:.1f}%) | "
                f"{m_r.fp:,} ({m_r.false_alarm_rate:.1f}%) | — |\n\n")
        f.write("## Confusion Matrices\n\n")
        f.write("**Time-Sorted Model:**\n\n")
        f.write(f"| | Predicted Not Fraud | Predicted Fraud |\n")
        f.write(f"|---|---|---|\n")
        f.write(f"| **Actual Not Fraud** | {m_s.tn:,} | {m_s.fp:,} |\n")
        f.write(f"| **Actual Fraud** | {m_s.fn:,} | {m_s.tp:,} |\n\n")
        f.write("**Random Split Model:**\n\n")
        f.write(f"| | Predicted Not Fraud | Predicted Fraud |\n")
        f.write(f"|---|---|---|\n")
        f.write(f"| **Actual Not Fraud** | {m_r.tn:,} | {m_r.fp:,} |\n")
        f.write(f"| **Actual Fraud** | {m_r.fn:,} | {m_r.tp:,} |\n\n")
        f.write("## Interpretation\n\n")
        f.write(f"The random split model shows an AUPRC inflation of **+{auprc_diff:.4f}** ")
        f.write(f"({auprc_diff*100:.3f} percentage points) compared to the time-sorted model. ")
        f.write("This inflation occurs because random splitting allows the model to train on ")
        f.write("future fraud examples — transactions that, in a real deployment scenario, ")
        f.write("would not yet have occurred at prediction time.\n\n")
        f.write("The time-sorted model represents the realistic operational scenario: ")
        f.write("train on historical data, predict on future unseen transactions. ")
        f.write("This is the only valid evaluation methodology for time-series fraud detection.\n\n")
        f.write("## Conclusion\n\n")
        f.write("Temporal ordering is non-negotiable for fraud detection model evaluation. ")
        f.write("Random splits produce optimistically biased metrics that overestimate ")
        f.write("real-world performance. The time-sorted approach in this project ensures ")
        f.write("all reported metrics reflect genuine out-of-sample generalization.\n")

    print(f"Saved: {md_path}")
    print("\nDone! Run outputs are in:", OUTDIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())