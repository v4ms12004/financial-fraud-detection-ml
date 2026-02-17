#!/usr/bin/env python3
"""
rf_model.py — Random Forest fraud model

How to run:
    python rf_model.py

Requirements:
    transaction_data.csv must be in the SAME folder as this script.
"""

import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# -----------------------------
# SETTINGS (edit here if needed)
# -----------------------------
CSV_FILE = "transaction_data.csv"
OUTPUT_DIR = "outputs_rf"
TRAIN_FRAC = 0.80
RANDOM_STATE = 42
N_TREES = 300
DOWNSAMPLE_RATIO = None   # Set to e.g. 50 if you want downsampling


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

    if DOWNSAMPLE_RATIO is not None:
        train_df = downsample_train(train_df, DOWNSAMPLE_RATIO)

    X_train = train_df.drop(columns=["isfraud"])
    y_train = train_df["isfraud"].values
    X_test = test_df.drop(columns=["isfraud"])
    y_test = test_df["isfraud"].values

    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=N_TREES,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features="sqrt",
    )
    rf.fit(X_train, y_train)

    print("Evaluating...")
    y_prob = rf.predict_proba(X_test)[:, 1]
    threshold = pick_threshold_for_f1(y_test, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = Metrics(
        precision=precision_score(y_test, y_pred),
        recall=recall_score(y_test, y_pred),
        f1=f1_score(y_test, y_pred),
        auprc=average_precision_score(y_test, y_prob),
        threshold=threshold,
    )

    cm = confusion_matrix(y_test, y_pred)

    ensure_outdir(OUTPUT_DIR)

    # Save metrics
    pd.DataFrame([metrics.__dict__]).to_csv(
        os.path.join(OUTPUT_DIR, "rf_metrics.csv"), index=False
    )

    # Save PR curve
    p, r, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(r, p)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Random Forest PR Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rf_precision_recall_curve.png"), dpi=200)
    plt.close()

    # Feature importance
    importance = pd.Series(rf.feature_importances_, index=X_train.columns)
    importance.sort_values(ascending=False).head(20).to_csv(
        os.path.join(OUTPUT_DIR, "rf_feature_importance_top20.csv")
    )

    print("\nMetrics:\n", metrics)
    print("\nConfusion Matrix:\n", cm)
    print(f"\nOutputs saved to '{OUTPUT_DIR}' folder.")


if __name__ == "__main__":
    sys.exit(main())