#!/usr/bin/env python3
"""
Outputs (created in outputs_panel_xgb/):
    - xgb_panel_metrics_main.csv
    - pr_curve_main.png
    - feature_importance_top30.csv
    - temporal_sensitivity.csv
    - temporal_sensitivity.md
    - report_summary.txt
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

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

from xgboost import XGBClassifier


# -----------------------------
# SETTINGS
# -----------------------------
CSV_FILE = "transaction_data.csv"
OUTDIR = "outputs_panel_xgb"

RANDOM_STATE = 42

# Main split (for your "primary" reported result)
MAIN_TRAIN_FRAC = 0.80

# Temporal sensitivity testing:
# If your professor suspects "patterns repeat around 30%", this is a good set:
SENSITIVITY_TRAIN_FRACS = [0.30, 0.60, 0.90]  # train on first X% of time
SENSITIVITY_TEST_WIDTH = 0.10                 # test on the next 10% window after train end

# XGBoost params (reasonable defaults for big tabular data)
N_TREES = 500
MAX_DEPTH = 6
LEARNING_RATE = 0.08

# Keep feature engineering efficient
USE_FLOAT32 = True

# Optional: if you want faster iteration while developing, set e.g. 2_000_000
ROW_LIMIT = None  # None means use full dataset


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float
    auprc: float
    threshold: float


# -----------------------------
# Utilities
# -----------------------------
def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def apply_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize common variants
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


def load_data() -> pd.DataFrame:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, CSV_FILE)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # dtype hints to reduce memory
    dtype: Dict[str, str] = {
        "step": "int32",
        "amount": "float32" if USE_FLOAT32 else "float64",
        "oldbalanceorg": "float32" if USE_FLOAT32 else "float64",
        "newbalanceorig": "float32" if USE_FLOAT32 else "float64",
        "oldbalancedest": "float32" if USE_FLOAT32 else "float64",
        "newbalancedest": "float32" if USE_FLOAT32 else "float64",
        "isfraud": "int8",
        "isflaggedfraud": "int8",
        # name columns and type loaded as object then optimized after
    }

    df = pd.read_csv(csv_path, nrows=ROW_LIMIT, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    df = apply_column_aliases(df)

    # Apply dtypes where possible
    for col, dt in dtype.items():
        if col in df.columns:
            df[col] = df[col].astype(dt, copy=False)

    # Ensure required columns exist
    required = [
        "step", "type", "amount",
        "nameorig", "namedest",
        "oldbalanceorg", "newbalanceorig",
        "oldbalancedest", "newbalancedest",
        "isfraud"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def pick_threshold_for_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    p, r, thr = precision_recall_curve(y_true, y_prob)
    # thr has length len(p)-1
    f1s = (2 * p[:-1] * r[:-1]) / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(thr[int(np.argmax(f1s))])


# -----------------------------
# Feature Engineering (Panel + Parsed)
# -----------------------------
def build_panel_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Builds leakage-safe, past-only panel features.
    Key idea: for each transaction, features only use information from previous transactions.

    We DO NOT use isFraud in features.
    """
    df = df_raw.copy()

    # Sort by time first; stable sort helps consistent cum features
    df = df.sort_values(["step"], kind="mergesort").reset_index(drop=True)

    # Parse hidden meaning: prefix of account IDs (often 'C' vs 'M')
    # These capture customer vs merchant-style accounts in PaySim-like data.
    df["orig_prefix"] = df["nameorig"].astype(str).str.slice(0, 1)
    df["dest_prefix"] = df["namedest"].astype(str).str.slice(0, 1)

    # Convert large-cardinality IDs to integer codes (much faster groupby)
    orig_id, orig_uniques = pd.factorize(df["nameorig"], sort=False)
    dest_id, dest_uniques = pd.factorize(df["namedest"], sort=False)
    df["orig_id"] = orig_id.astype("int32", copy=False)
    df["dest_id"] = dest_id.astype("int32", copy=False)

    # Basic balance-based features (transaction-level)
    df["delta_org"] = (df["newbalanceorig"] - df["oldbalanceorg"]).astype("float32" if USE_FLOAT32 else "float64")
    df["delta_dest"] = (df["newbalancedest"] - df["oldbalancedest"]).astype("float32" if USE_FLOAT32 else "float64")
    df["abs_delta_org"] = np.abs(df["delta_org"]).astype("float32" if USE_FLOAT32 else "float64")
    df["abs_delta_dest"] = np.abs(df["delta_dest"]).astype("float32" if USE_FLOAT32 else "float64")

    # "Accounting consistency" style feature (often informative in simulated fraud)
    df["org_balance_minus_amount_gap"] = (
        (df["oldbalanceorg"] - df["newbalanceorig"]) - df["amount"]
    ).astype("float32" if USE_FLOAT32 else "float64")

    # Ratio-type features (avoid divide by zero)
    df["amount_over_oldorg"] = (df["amount"] / (df["oldbalanceorg"] + 1.0)).astype("float32" if USE_FLOAT32 else "float64")
    df["amount_over_olddest"] = (df["amount"] / (df["oldbalancedest"] + 1.0)).astype("float32" if USE_FLOAT32 else "float64")

    # ---- Panel (past-only) features for sender (orig) ----
    # count so far (before this transaction)
    df["orig_txn_count_prev"] = df.groupby("orig_id").cumcount().astype("int32", copy=False)

    # cumulative amount so far (before this transaction)
    orig_cum_amt = df.groupby("orig_id")["amount"].cumsum()
    df["orig_cum_amount_prev"] = (orig_cum_amt - df["amount"]).astype("float32" if USE_FLOAT32 else "float64")

    # average amount so far (before this txn)
    df["orig_avg_amount_prev"] = (
        df["orig_cum_amount_prev"] / (df["orig_txn_count_prev"] + 1.0)
    ).astype("float32" if USE_FLOAT32 else "float64")

    # time since first seen (tenure) and time since last seen
    orig_first_step = df.groupby("orig_id")["step"].transform("min")
    df["orig_tenure"] = (df["step"] - orig_first_step).astype("int32", copy=False)

    orig_prev_step = df.groupby("orig_id")["step"].shift(1)
    df["orig_time_since_prev"] = (df["step"] - orig_prev_step).fillna(0).astype("int32", copy=False)

    # ---- Panel (past-only) features for receiver (dest) ----
    df["dest_txn_count_prev"] = df.groupby("dest_id").cumcount().astype("int32", copy=False)

    dest_cum_in_amt = df.groupby("dest_id")["amount"].cumsum()
    df["dest_cum_in_amount_prev"] = (dest_cum_in_amt - df["amount"]).astype("float32" if USE_FLOAT32 else "float64")

    df["dest_avg_in_amount_prev"] = (
        df["dest_cum_in_amount_prev"] / (df["dest_txn_count_prev"] + 1.0)
    ).astype("float32" if USE_FLOAT32 else "float64")

    dest_first_step = df.groupby("dest_id")["step"].transform("min")
    df["dest_tenure"] = (df["step"] - dest_first_step).astype("int32", copy=False)

    dest_prev_step = df.groupby("dest_id")["step"].shift(1)
    df["dest_time_since_prev"] = (df["step"] - dest_prev_step).fillna(0).astype("int32", copy=False)

    # ---- One-hot encode categorical features ----
    # type, prefixes
    df["type"] = df["type"].astype("category")
    df["orig_prefix"] = df["orig_prefix"].astype("category")
    df["dest_prefix"] = df["dest_prefix"].astype("category")

    df = pd.get_dummies(df, columns=["type", "orig_prefix", "dest_prefix"], drop_first=True)

    # Drop raw high-cardinality identifiers and flagged fraud (not needed)
    df = df.drop(columns=["nameorig", "namedest", "isflaggedfraud"], errors="ignore")

    # Also drop internal IDs (we don't want model to memorize them)
    df = df.drop(columns=["orig_id", "dest_id"], errors="ignore")

    return df


# -----------------------------
# Temporal Splits (Sensitivity / Distribution Shift)
# -----------------------------
def split_by_time_fraction(df: pd.DataFrame, train_frac: float, test_width: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by time fraction using sorted unique steps.
    Train = earliest steps up to train_frac
    Test  = immediately following window of width test_width

    This matches the "if it repeats at ~30%, evaluate in windows around that boundary" idea.
    """
    steps = np.sort(df["step"].unique())
    n = len(steps)
    train_end_idx = int(np.floor(train_frac * n))
    train_end_idx = max(1, min(train_end_idx, n - 1))

    test_end_idx = int(np.floor((train_frac + test_width) * n))
    test_end_idx = max(train_end_idx + 1, min(test_end_idx, n))

    train_steps = steps[:train_end_idx]
    test_steps = steps[train_end_idx:test_end_idx]

    train_df = df[df["step"].isin(train_steps)].copy()
    test_df = df[df["step"].isin(test_steps)].copy()
    return train_df, test_df


# -----------------------------
# Model Training / Evaluation
# -----------------------------
def train_xgb(X_train: pd.DataFrame, y_train: np.ndarray) -> XGBClassifier:
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = XGBClassifier(
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
    model.fit(X_train, y_train)
    return model


def evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[Metrics, np.ndarray]:
    thr = pick_threshold_for_f1(y_true, y_prob)
    y_pred = (y_prob >= thr).astype(int)

    m = Metrics(
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        auprc=float(average_precision_score(y_true, y_prob)),
        threshold=float(thr),
    )
    cm = confusion_matrix(y_true, y_pred)
    return m, cm


def save_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, outpath: str, title: str) -> None:
    p, r, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(r, p)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ensure_outdir(OUTDIR)

    print("Loading data...")
    df_raw = load_data()
    print(f"Loaded shape: {df_raw.shape}")
    print("Class counts:\n", df_raw["isfraud"].value_counts())

    print("\nBuilding panel + parsed features (past-only)...")
    df_feat = build_panel_features(df_raw)
    print(f"Feature dataset shape: {df_feat.shape}")

    # -------------------------
    # Main evaluation split (80/20)
    # -------------------------
    print("\nMAIN split evaluation (time-based)...")
    train_main, test_main = split_by_time_fraction(df_feat, MAIN_TRAIN_FRAC, 1.0 - MAIN_TRAIN_FRAC)
    X_train = train_main.drop(columns=["isfraud"])
    y_train = train_main["isfraud"].astype(int).values
    X_test = test_main.drop(columns=["isfraud"])
    y_test = test_main["isfraud"].astype(int).values

    print(f"Main train rows: {len(train_main):,} | Main test rows: {len(test_main):,}")

    model = train_xgb(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics, cm = evaluate_probs(y_test, y_prob)
    print("\nMain metrics:", metrics)
    print("Confusion matrix:\n", cm)

    pd.DataFrame([metrics.__dict__]).to_csv(os.path.join(OUTDIR, "xgb_panel_metrics_main.csv"), index=False)
    save_pr_curve(y_test, y_prob, os.path.join(OUTDIR, "pr_curve_main.png"), "XGBoost (Panel Features) PR Curve — Main Split")

    # feature importance
    imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    imp.head(30).to_csv(os.path.join(OUTDIR, "feature_importance_top30.csv"))

    # -------------------------
    # Temporal sensitivity tests (distribution shift style)
    # -------------------------
    print("\nRunning temporal sensitivity tests...")
    rows: List[Dict[str, object]] = []

    for frac in SENSITIVITY_TRAIN_FRACS:
        tr, te = split_by_time_fraction(df_feat, frac, SENSITIVITY_TEST_WIDTH)
        if len(te) == 0 or len(tr) == 0:
            continue

        X_tr = tr.drop(columns=["isfraud"])
        y_tr = tr["isfraud"].astype(int).values
        X_te = te.drop(columns=["isfraud"])
        y_te = te["isfraud"].astype(int).values

        m = train_xgb(X_tr, y_tr)
        prob = m.predict_proba(X_te)[:, 1]
        met, _ = evaluate_probs(y_te, prob)

        rows.append({
            "train_frac_end": frac,
            "test_width": SENSITIVITY_TEST_WIDTH,
            "train_rows": len(tr),
            "test_rows": len(te),
            "test_auprc": met.auprc,
            "test_precision": met.precision,
            "test_recall": met.recall,
            "test_f1": met.f1,
            "threshold": met.threshold,
        })

        print(f"Train 0–{int(frac*100)}% | Test next {int(SENSITIVITY_TEST_WIDTH*100)}% -> AUPRC={met.auprc:.6f}")

    sens = pd.DataFrame(rows)
    sens_path = os.path.join(OUTDIR, "temporal_sensitivity.csv")
    sens.to_csv(sens_path, index=False)

    # Also save a markdown table (no tabulate dependency)
    md_path = os.path.join(OUTDIR, "temporal_sensitivity.md")
    if not sens.empty:
        headers = sens.columns.tolist()
        md = "| " + " | ".join(headers) + " |\n"
        md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for _, r in sens.iterrows():
            vals = []
            for c in headers:
                v = r[c]
                if isinstance(v, float):
                    vals.append(f"{v:.6f}")
                else:
                    vals.append(str(v))
            md += "| " + " | ".join(vals) + " |\n"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
    else:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("| (no results) |\n|---|\n")

    # Summary for your report
    summary_path = os.path.join(OUTDIR, "report_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("XGBoost with Panel + Parsed Features — Summary\n")
        f.write("============================================\n\n")

        f.write("What features were added:\n")
        f.write("- Parsed ID prefixes (orig_prefix, dest_prefix) to capture hidden entity type (e.g., customer vs merchant)\n")
        f.write("- Sender (orig) past-only behavior: prior txn count, prior cumulative amount, avg amount so far, tenure, time since previous txn\n")
        f.write("- Receiver (dest) past-only behavior: prior txn count, prior cumulative inflow, avg inflow so far, tenure, time since previous txn\n")
        f.write("- Balance/amount derived features: deltas, absolute deltas, accounting gap, amount-to-balance ratios\n\n")

        f.write("Main split (time-based) results:\n")
        f.write(f"- Train fraction: {MAIN_TRAIN_FRAC:.2f}\n")
        f.write(f"- Precision: {metrics.precision:.6f}\n")
        f.write(f"- Recall   : {metrics.recall:.6f}\n")
        f.write(f"- F1       : {metrics.f1:.6f}\n")
        f.write(f"- AUPRC    : {metrics.auprc:.6f}\n")
        f.write(f"- Threshold: {metrics.threshold:.6f}\n\n")

        f.write("Temporal sensitivity tests (distribution-shift style):\n")
        f.write(f"- Train ends at: {SENSITIVITY_TRAIN_FRACS}\n")
        f.write(f"- Test window width: {SENSITIVITY_TEST_WIDTH:.2f}\n")
        f.write(f"- Saved to: temporal_sensitivity.csv and temporal_sensitivity.md\n\n")

        f.write("Saved outputs:\n")
        f.write("- xgb_panel_metrics_main.csv\n")
        f.write("- pr_curve_main.png\n")
        f.write("- feature_importance_top30.csv\n")
        f.write("- temporal_sensitivity.csv\n")
        f.write("- temporal_sensitivity.md\n")

    print("\nDone.")
    print(f"Outputs saved to: {OUTDIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())