#!/usr/bin/env python3
"""
xgb_model.py — XGBoost fraud model (PaySim-like dataset)

How to run:
    python xgb_model.py

Requirements:
    transaction_data.csv must be in the SAME folder as this script.

Outputs:
    outputs_xgboost/
      - xgb_metrics.csv
      - xgb_precision_recall_curve.png
      - xgb_feature_importance_top20.csv
      - report_summary.txt
"""

import os
import sys
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError("xgboost is not installed. Install with: pip install xgboost")


# -----------------------------
# SETTINGS (edit here if needed)
# -----------------------------
CV_SPLITS = 5
CV_SAMPLE_TRAIN_ROWS = 800_000

CSV_FILE = "transaction_data.csv"
OUTPUT_DIR = "outputs_xgboost"

TRAIN_FRAC = 0.80
RANDOM_STATE = 42

N_TREES = 500
MAX_DEPTH = 6
LEARNING_RATE = 0.08

DOWNSAMPLE_RATIO = None   # Set to e.g. 50 if you want to downsample training majority class


# -----------------------------
# Data structure
# -----------------------------
@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float
    auprc: float
    threshold: float


# -----------------------------
# Helper functions
# -----------------------------
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


def add_features(df):
    out = df.copy()

    out["delta_org"] = out["newbalanceorig"] - out["oldbalanceorg"]
    out["delta_dest"] = out["newbalancedest"] - out["oldbalancedest"]
    out["abs_delta_org"] = np.abs(out["delta_org"])
    out["abs_delta_dest"] = np.abs(out["delta_dest"])
    out["org_balance_minus_amount_gap"] = (
        (out["oldbalanceorg"] - out["newbalanceorig"]) - out["amount"]
    )

    out = out.drop(columns=["nameorig", "namedest", "isflaggedfraud"], errors="ignore")
    out = pd.get_dummies(out, columns=["type"], drop_first=True)

    return out


def time_split(df):
    df_sorted = df.sort_values("step").reset_index(drop=True)
    cut = int(len(df_sorted) * TRAIN_FRAC)
    return df_sorted.iloc[:cut].copy(), df_sorted.iloc[cut:].copy()


def downsample_train(train_df, ratio):
    fraud = train_df[train_df["isfraud"] == 1]
    nonfraud = train_df[train_df["isfraud"] == 0]

    max_nonfraud = int(ratio * len(fraud))
    if len(nonfraud) <= max_nonfraud:
        return train_df

    nonfraud_down = nonfraud.sample(n=max_nonfraud, random_state=RANDOM_STATE)
    return pd.concat([fraud, nonfraud_down]).sample(frac=1, random_state=RANDOM_STATE)


def pick_threshold_for_f1(y_true, y_prob):
    p, r, thr = precision_recall_curve(y_true, y_prob)
    f1s = (2 * p[:-1] * r[:-1]) / np.maximum(p[:-1] + r[:-1], 1e-12)
    return thr[int(np.argmax(f1s))]

def time_series_cv_auprc_xgb(train_df, n_splits=5, sample_rows=800_000):
    if sample_rows is not None and len(train_df) > sample_rows:
        train_df = train_df.iloc[:sample_rows].copy()

    X = train_df.drop(columns=["isfraud"])
    y = train_df["isfraud"].astype(int).values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        n_neg = int((y_tr == 0).sum())
        n_pos = int((y_tr == 1).sum())
        spw = n_neg / max(n_pos, 1)

        model = XGBClassifier(
            n_estimators=N_TREES,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )

        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_va)[:, 1]

        auprc = average_precision_score(y_va, y_prob)
        scores.append(float(auprc))

        print(f"CV fold {fold}/{n_splits} AUPRC: {auprc:.6f}")

    scores = np.array(scores, dtype=float)
    return scores.tolist(), float(scores.mean()), float(scores.std())

# -----------------------------
# Main execution
# -----------------------------
def main():
    print("Loading data...")
    df_raw = load_data()
    print(f"Loaded shape: {df_raw.shape}")
    print("Class counts:\n", df_raw["isfraud"].value_counts())

    print("Feature engineering...")
    df = add_features(df_raw)

    train_df, test_df = time_split(df)

    print("\nRunning TimeSeriesSplit CV on training set...")
    cv_scores, cv_mean, cv_std = time_series_cv_auprc_xgb(
    train_df,
    n_splits=CV_SPLITS,
    sample_rows=CV_SAMPLE_TRAIN_ROWS
    )
    print(f"CV AUPRC mean={cv_mean:.6f}, std={cv_std:.6f}")

    if DOWNSAMPLE_RATIO is not None:
        train_df = downsample_train(train_df, DOWNSAMPLE_RATIO)

    X_train = train_df.drop(columns=["isfraud"])
    y_train = train_df["isfraud"].values
    X_test = test_df.drop(columns=["isfraud"])
    y_test = test_df["isfraud"].values

    # Scale weight for imbalance
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    print("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=N_TREES,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )
    xgb.fit(X_train, y_train)

    print("Evaluating...")
    y_prob = xgb.predict_proba(X_test)[:, 1]
    threshold = pick_threshold_for_f1(y_test, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = Metrics(
        precision=precision_score(y_test, y_pred, zero_division=0),
        recall=recall_score(y_test, y_pred, zero_division=0),
        f1=f1_score(y_test, y_pred, zero_division=0),
        auprc=average_precision_score(y_test, y_prob),
        threshold=float(threshold),
    )

    cm = confusion_matrix(y_test, y_pred)

    ensure_outdir(OUTPUT_DIR)

    # Save metrics
    pd.DataFrame([metrics.__dict__]).to_csv(
        os.path.join(OUTPUT_DIR, "xgb_metrics.csv"), index=False
    )

    # Save PR curve
    p, r, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(r, p)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("XGBoost PR Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "xgb_precision_recall_curve.png"), dpi=200)
    plt.close()

    # Feature importance
    importance = pd.Series(xgb.feature_importances_, index=X_train.columns)
    importance.sort_values(ascending=False).head(20).to_csv(
        os.path.join(OUTPUT_DIR, "xgb_feature_importance_top20.csv")
    )

    pd.DataFrame({"cv_fold": list(range(1, len(cv_scores)+1)), "auprc": cv_scores}).to_csv(
    os.path.join(OUTPUT_DIR, "xgb_cv_auprc.csv"), index=False
    )

    # Save summary text
    with open(os.path.join(OUTPUT_DIR, "report_summary.txt"), "w", encoding="utf-8") as f:
        f.write("XGBoost — Test-set Results\n")
        f.write("=========================\n\n")
        f.write(str(pd.DataFrame([metrics.__dict__])) + "\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")

        f.write("\nTimeSeriesSplit CV (train-only):\n")
        f.write(f"- folds: {CV_SPLITS}\n")
        f.write(f"- AUPRC mean: {cv_mean:.6f}\n")
        f.write(f"- AUPRC std : {cv_std:.6f}\n")
        f.write("- per-fold saved to xgb_cv_auprc.csv\n")

    print("\nMetrics:\n", metrics)
    print("\nConfusion Matrix:\n", cm)
    print(f"\nOutputs saved to '{OUTPUT_DIR}' folder.")


if __name__ == "__main__":
    sys.exit(main())