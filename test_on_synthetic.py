import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
from xgboost import XGBClassifier

# -------------------------------------------------------
# SETTINGS
# -------------------------------------------------------

# Input: synthetic dataset produced by generate_synthetic_data.py
SYNTHETIC_CSV = "advanced_synthetic_data.csv"

# Output directory
OUTDIR = "outputs_synthetic_test"

# Your known PaySim results to compare against
# Update these if you have more precise numbers from outputs_panel_xgb/
PAYSIM_KNOWN = {
    "auprc":     0.998,
    "precision": None,   # Fill in if you have it
    "recall":    None,   # Fill in if you have it
    "f1":        None,   # Fill in if you have it
    "source":    "PaySim (panel_temporal_xgb.py, 80/20 temporal split, ~0.998 AUPRC)",
}

# Also try to load exact PaySim results from your existing output CSVs if present
PAYSIM_METRICS_CSV = os.path.join("outputs_panel_xgb", "xgb_panel_metrics_main.csv")
PAYSIM_SENSITIVITY_CSV = os.path.join("outputs_panel_xgb", "temporal_sensitivity.csv")

# Model hyperparameters — IDENTICAL to panel_temporal_xgb.py
# (must match so the comparison is fair)
N_TREES       = 500
MAX_DEPTH     = 6
LEARNING_RATE = 0.08
RANDOM_STATE  = 42
USE_FLOAT32   = True

# Train/test split — identical to panel_temporal_xgb.py
MAIN_TRAIN_FRAC     = 0.80
SENSITIVITY_FRACS   = [0.30, 0.60, 0.90]
SENSITIVITY_WIDTH   = 0.10

# -------------------------------------------------------
# Data structures
# -------------------------------------------------------

@dataclass
class Metrics:
    precision: float
    recall:    float
    f1:        float
    auprc:     float
    threshold: float
    n_fraud:   int
    n_total:   int

# -------------------------------------------------------
# Utilities (mirrors panel_temporal_xgb.py exactly)
# -------------------------------------------------------

def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def apply_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "newbalanceorg":        "newbalanceorig",
        "oldbalanceorig":       "oldbalanceorg",
        "oldbalancedestination":"oldbalancedest",
        "newbalancedestination":"newbalancedest",
        "is_fraud":             "isfraud",
        "fraud":                "isfraud",
        "is_flagged_fraud":     "isflaggedfraud",
    }
    rename_map = {c: aliases[c] for c in df.columns if c in aliases}
    return df.rename(columns=rename_map)

def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"\nFile not found: {csv_path}\n"
            f"Please run generate_synthetic_data.py first.\n"
        )
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    df = apply_column_aliases(df)

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
    f1s = (2 * p[:-1] * r[:-1]) / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(thr[int(np.argmax(f1s))])

# -------------------------------------------------------
# Feature Engineering — IDENTICAL to panel_temporal_xgb.py
# -------------------------------------------------------

def build_panel_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Exact copy of the feature engineering from panel_temporal_xgb.py.
    Must be identical so the comparison is fair.
    """
    df = df_raw.copy()
    df = df.sort_values(["step"], kind="mergesort").reset_index(drop=True)

    df["orig_prefix"] = df["nameorig"].astype(str).str.slice(0, 1)
    df["dest_prefix"] = df["namedest"].astype(str).str.slice(0, 1)

    orig_id, _ = pd.factorize(df["nameorig"], sort=False)
    dest_id, _ = pd.factorize(df["namedest"], sort=False)
    df["orig_id"] = orig_id.astype("int32", copy=False)
    df["dest_id"] = dest_id.astype("int32", copy=False)

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
    df["orig_txn_count_prev"]   = df.groupby("orig_id").cumcount().astype("int32")
    orig_cum_amt                = df.groupby("orig_id")["amount"].cumsum()
    df["orig_cum_amount_prev"]  = (orig_cum_amt - df["amount"]).astype("float32")
    df["orig_avg_amount_prev"]  = (
        df["orig_cum_amount_prev"] / (df["orig_txn_count_prev"] + 1.0)
    ).astype("float32")
    orig_first_step             = df.groupby("orig_id")["step"].transform("min")
    df["orig_tenure"]           = (df["step"] - orig_first_step).astype("int32")
    orig_prev_step              = df.groupby("orig_id")["step"].shift(1)
    df["orig_time_since_prev"]  = (df["step"] - orig_prev_step).fillna(0).astype("int32")

    # Panel features — receiver
    df["dest_txn_count_prev"]      = df.groupby("dest_id").cumcount().astype("int32")
    dest_cum_in_amt                = df.groupby("dest_id")["amount"].cumsum()
    df["dest_cum_in_amount_prev"]  = (dest_cum_in_amt - df["amount"]).astype("float32")
    df["dest_avg_in_amount_prev"]  = (
        df["dest_cum_in_amount_prev"] / (df["dest_txn_count_prev"] + 1.0)
    ).astype("float32")
    dest_first_step                = df.groupby("dest_id")["step"].transform("min")
    df["dest_tenure"]              = (df["step"] - dest_first_step).astype("int32")
    dest_prev_step                 = df.groupby("dest_id")["step"].shift(1)
    df["dest_time_since_prev"]     = (df["step"] - dest_prev_step).fillna(0).astype("int32")

    # One-hot encode categoricals
    df["type"]        = df["type"].astype("category")
    df["orig_prefix"] = df["orig_prefix"].astype("category")
    df["dest_prefix"] = df["dest_prefix"].astype("category")
    df = pd.get_dummies(df, columns=["type", "orig_prefix", "dest_prefix"], drop_first=True)

    df = df.drop(columns=["nameorig", "namedest", "isflaggedfraud"], errors="ignore")
    df = df.drop(columns=["orig_id", "dest_id"], errors="ignore")

    return df

# -------------------------------------------------------
# Temporal splitting — IDENTICAL to panel_temporal_xgb.py
# -------------------------------------------------------

def split_by_time_fraction(
    df: pd.DataFrame, train_frac: float, test_width: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    steps = np.sort(df["step"].unique())
    n = len(steps)
    train_end_idx = int(np.floor(train_frac * n))
    train_end_idx = max(1, min(train_end_idx, n - 1))
    test_end_idx  = int(np.floor((train_frac + test_width) * n))
    test_end_idx  = max(train_end_idx + 1, min(test_end_idx, n))
    train_steps = steps[:train_end_idx]
    test_steps  = steps[train_end_idx:test_end_idx]
    return (
        df[df["step"].isin(train_steps)].copy(),
        df[df["step"].isin(test_steps)].copy(),
    )

# -------------------------------------------------------
# Model training — IDENTICAL to panel_temporal_xgb.py
# -------------------------------------------------------

def train_xgb(X_train: pd.DataFrame, y_train: np.ndarray) -> XGBClassifier:
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    model = XGBClassifier(
        n_estimators     = N_TREES,
        max_depth        = MAX_DEPTH,
        learning_rate    = LEARNING_RATE,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        reg_lambda       = 1.0,
        scale_pos_weight = scale_pos_weight,
        eval_metric      = "logloss",
        random_state     = RANDOM_STATE,
        n_jobs           = -1,
        tree_method      = "hist",
    )
    model.fit(X_train, y_train)
    return model

def evaluate_probs(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[Metrics, np.ndarray]:
    thr    = pick_threshold_for_f1(y_true, y_prob)
    y_pred = (y_prob >= thr).astype(int)
    m = Metrics(
        precision = float(precision_score(y_true, y_pred, zero_division=0)),
        recall    = float(recall_score(y_true, y_pred, zero_division=0)),
        f1        = float(f1_score(y_true, y_pred, zero_division=0)),
        auprc     = float(average_precision_score(y_true, y_prob)),
        threshold = float(thr),
        n_fraud   = int(y_true.sum()),
        n_total   = int(len(y_true)),
    )
    cm = confusion_matrix(y_true, y_pred)
    return m, cm

# -------------------------------------------------------
# Comparison report
# -------------------------------------------------------

def load_paysim_results() -> dict:
    """Try to load exact PaySim metrics from saved CSV, fall back to known values."""
    result = dict(PAYSIM_KNOWN)
    if os.path.exists(PAYSIM_METRICS_CSV):
        try:
            df = pd.read_csv(PAYSIM_METRICS_CSV)
            result["auprc"]     = float(df["auprc"].iloc[0])
            result["precision"] = float(df["precision"].iloc[0])
            result["recall"]    = float(df["recall"].iloc[0])
            result["f1"]        = float(df["f1"].iloc[0])
            result["source"]    = f"PaySim (loaded from {PAYSIM_METRICS_CSV})"
            print(f"  Loaded exact PaySim results from {PAYSIM_METRICS_CSV}")
        except Exception as e:
            print(f"  Could not parse {PAYSIM_METRICS_CSV}: {e}. Using known values.")
    else:
        print(f"  {PAYSIM_METRICS_CSV} not found — using remembered PaySim AUPRC ~0.998")
    return result

def interpret_result(paysim_auprc: float, synth_auprc: float) -> str:
    """
    Interpret the comparison result for the report.
    """
    diff      = paysim_auprc - synth_auprc
    pct_drop  = (diff / paysim_auprc) * 100

    if pct_drop < 2:
        verdict = "GENERALIZES WELL"
        interp = (
            f"The model maintains strong AUPRC on the synthetic dataset "
            f"(drop of only {pct_drop:.1f}%). This suggests the model is detecting "
            f"genuine fraud signals — specifically the balance-depletion pattern, "
            f"account prefix structure, and panel behavioral features — rather than "
            f"PaySim simulator artifacts. The fraud detection pipeline appears robust "
            f"and likely to generalize beyond PaySim-generated data."
        )
    elif pct_drop < 10:
        verdict = "MODERATE DEGRADATION"
        interp = (
            f"The model shows a moderate AUPRC drop of {pct_drop:.1f}% on synthetic "
            f"data. This suggests partial overfitting to PaySim-specific patterns. "
            f"Some fraud signals (e.g., balance deltas, account prefixes) appear "
            f"transferable, but others may be tied to PaySim's specific statistical "
            f"distributions or deterministic agent logic. Feature ablation studies "
            f"could help identify which features are driving the drop."
        )
    else:
        verdict = "SIGNIFICANT DEGRADATION — POSSIBLE SIMULATOR OVERFITTING"
        interp = (
            f"The model shows a significant AUPRC drop of {pct_drop:.1f}% on synthetic "
            f"data. This is strong evidence that the model overfit to PaySim's internal "
            f"simulator rules rather than learning generalizable fraud behavior. "
            f"The near-perfect PaySim AUPRC (~0.998) was likely achieved by learning "
            f"deterministic patterns unique to the PaySim agent-based simulation "
            f"(e.g., exact balance-drain mechanics, specific step distributions), "
            f"rather than features that would transfer to real-world fraud."
        )
    return verdict, interp, diff, pct_drop

def write_comparison_report(
    synth_metrics: Metrics,
    synth_cm: np.ndarray,
    synth_sensitivity: pd.DataFrame,
    paysim: dict,
    outdir: str,
) -> None:
    paysim_auprc = paysim["auprc"]
    synth_auprc  = synth_metrics.auprc
    verdict, interp, diff, pct_drop = interpret_result(paysim_auprc, synth_auprc)

    # ---- Plain text report ----
    txt_path = os.path.join(outdir, "comparison_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("  PAYSIM vs SYNTHETIC — FRAUD DETECTION COMPARISON REPORT\n")
        f.write("=" * 65 + "\n\n")

        f.write("RESEARCH QUESTION\n")
        f.write("-" * 40 + "\n")
        f.write("Is the model detecting genuine fraud behavior, or has it\n")
        f.write("overfit to the internal rules of the PaySim simulator?\n\n")

        f.write("SIDE-BY-SIDE METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Metric':<20} {'PaySim':>12} {'Synthetic':>12} {'Difference':>12}\n")
        f.write("-" * 60 + "\n")

        def fmt(v): return f"{v:.6f}" if v is not None else "N/A"

        f.write(f"{'AUPRC':<20} {fmt(paysim_auprc):>12} {fmt(synth_auprc):>12} {fmt(-diff):>12}\n")

        if paysim.get("precision") is not None:
            p_diff = synth_metrics.precision - paysim["precision"]
            f.write(f"{'Precision':<20} {fmt(paysim['precision']):>12} {fmt(synth_metrics.precision):>12} {fmt(p_diff):>12}\n")
        else:
            f.write(f"{'Precision':<20} {'N/A':>12} {fmt(synth_metrics.precision):>12} {'N/A':>12}\n")

        if paysim.get("recall") is not None:
            r_diff = synth_metrics.recall - paysim["recall"]
            f.write(f"{'Recall':<20} {fmt(paysim['recall']):>12} {fmt(synth_metrics.recall):>12} {fmt(r_diff):>12}\n")
        else:
            f.write(f"{'Recall':<20} {'N/A':>12} {fmt(synth_metrics.recall):>12} {'N/A':>12}\n")

        if paysim.get("f1") is not None:
            f1_diff = synth_metrics.f1 - paysim["f1"]
            f.write(f"{'F1 Score':<20} {fmt(paysim['f1']):>12} {fmt(synth_metrics.f1):>12} {fmt(f1_diff):>12}\n")
        else:
            f.write(f"{'F1 Score':<20} {'N/A':>12} {fmt(synth_metrics.f1):>12} {'N/A':>12}\n")

        f.write("\n")
        f.write(f"{'AUPRC Drop':<20} {pct_drop:>11.2f}%\n")
        f.write(f"{'Fraud cases (test)':<20} {'N/A':>12} {synth_metrics.n_fraud:>12}\n")
        f.write(f"{'Total test rows':<20} {'N/A':>12} {synth_metrics.n_total:>12}\n")

        f.write("\n")
        f.write("VERDICT\n")
        f.write("-" * 40 + "\n")
        f.write(f"{verdict}\n\n")

        f.write("INTERPRETATION\n")
        f.write("-" * 40 + "\n")
        f.write(interp + "\n\n")

        f.write("CONFUSION MATRIX (Synthetic test set)\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'':>20} Predicted 0   Predicted 1\n")
        f.write(f"{'Actual 0':>20} {synth_cm[0,0]:>12,} {synth_cm[0,1]:>12,}\n")
        f.write(f"{'Actual 1':>20} {synth_cm[1,0]:>12,} {synth_cm[1,1]:>12,}\n\n")

        f.write("TEMPORAL SENSITIVITY (Synthetic)\n")
        f.write("-" * 40 + "\n")
        if not synth_sensitivity.empty:
            f.write(synth_sensitivity.to_string(index=False) + "\n\n")

        f.write("DATA SOURCES\n")
        f.write("-" * 40 + "\n")
        f.write(f"PaySim   : {paysim['source']}\n")
        f.write(f"Synthetic: {SYNTHETIC_CSV} (generated by generate_synthetic_data.py)\n")
        f.write("\nMODEL SETTINGS (identical for both runs)\n")
        f.write("-" * 40 + "\n")
        f.write(f"n_estimators     : {N_TREES}\n")
        f.write(f"max_depth        : {MAX_DEPTH}\n")
        f.write(f"learning_rate    : {LEARNING_RATE}\n")
        f.write(f"train_frac       : {MAIN_TRAIN_FRAC}\n")
        f.write(f"random_state     : {RANDOM_STATE}\n")

    print(f"  Saved: {txt_path}")

    # ---- Markdown report ----
    md_path = os.path.join(outdir, "comparison_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# PaySim vs Synthetic — Fraud Detection Comparison\n\n")
        f.write("## Research Question\n")
        f.write("> Is the model detecting genuine fraud behavior, or has it overfit ")
        f.write("to the internal rules of the PaySim simulator?\n\n")

        f.write("## Side-by-Side Metrics\n\n")
        f.write("| Metric | PaySim | Synthetic | Difference |\n")
        f.write("|--------|--------|-----------|------------|\n")
        f.write(f"| **AUPRC** | {fmt(paysim_auprc)} | {fmt(synth_auprc)} | {fmt(-diff)} |\n")

        if paysim.get("precision") is not None:
            p_diff = synth_metrics.precision - paysim["precision"]
            f.write(f"| Precision | {fmt(paysim['precision'])} | {fmt(synth_metrics.precision)} | {fmt(p_diff)} |\n")
        else:
            f.write(f"| Precision | N/A | {fmt(synth_metrics.precision)} | N/A |\n")

        if paysim.get("recall") is not None:
            r_diff = synth_metrics.recall - paysim["recall"]
            f.write(f"| Recall | {fmt(paysim['recall'])} | {fmt(synth_metrics.recall)} | {fmt(r_diff)} |\n")
        else:
            f.write(f"| Recall | N/A | {fmt(synth_metrics.recall)} | N/A |\n")

        if paysim.get("f1") is not None:
            f1_diff = synth_metrics.f1 - paysim["f1"]
            f.write(f"| F1 Score | {fmt(paysim['f1'])} | {fmt(synth_metrics.f1)} | {fmt(f1_diff)} |\n")
        else:
            f.write(f"| F1 Score | N/A | {fmt(synth_metrics.f1)} | N/A |\n")

        f.write(f"\n**AUPRC Drop: {pct_drop:.2f}%**\n\n")

        f.write("## Verdict\n\n")
        f.write(f"**{verdict}**\n\n")
        f.write(f"{interp}\n\n")

        f.write("## Temporal Sensitivity (Synthetic)\n\n")
        if not synth_sensitivity.empty:
            headers = list(synth_sensitivity.columns)
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            for _, row in synth_sensitivity.iterrows():
                vals = []
                for c in headers:
                    v = row[c]
                    vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
                f.write("| " + " | ".join(vals) + " |\n")
        f.write("\n")

        f.write("## Confusion Matrix (Synthetic test set)\n\n")
        f.write("| | Predicted 0 | Predicted 1 |\n")
        f.write("|---|---|---|\n")
        f.write(f"| **Actual 0** | {synth_cm[0,0]:,} | {synth_cm[0,1]:,} |\n")
        f.write(f"| **Actual 1** | {synth_cm[1,0]:,} | {synth_cm[1,1]:,} |\n\n")

        f.write("## Model Settings (identical for both runs)\n\n")
        f.write(f"- `n_estimators`: {N_TREES}\n")
        f.write(f"- `max_depth`: {MAX_DEPTH}\n")
        f.write(f"- `learning_rate`: {LEARNING_RATE}\n")
        f.write(f"- `train_frac`: {MAIN_TRAIN_FRAC}\n")
        f.write(f"- `random_state`: {RANDOM_STATE}\n")

    print(f"  Saved: {md_path}")

# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main() -> int:
    ensure_outdir(OUTDIR)

    print("=" * 65)
    print("  Testing panel_temporal_xgb pipeline on synthetic dataset")
    print("=" * 65)

    # Load PaySim reference results
    print("\nLoading PaySim reference results...")
    paysim = load_paysim_results()
    print(f"  PaySim AUPRC (reference): {paysim['auprc']:.6f}")

    # Load synthetic data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path   = os.path.join(script_dir, SYNTHETIC_CSV)
    print(f"\nLoading synthetic data from: {csv_path}")
    df_raw = load_data(csv_path)
    print(f"  Shape   : {df_raw.shape}")
    print(f"  Fraud   : {df_raw['isfraud'].sum():,} ({df_raw['isfraud'].mean()*100:.4f}%)")
    print(f"  No-Fraud: {(df_raw['isfraud']==0).sum():,}")

    # Feature engineering
    print("\nBuilding panel features (identical to panel_temporal_xgb.py)...")
    df_feat = build_panel_features(df_raw)
    print(f"  Feature dataset shape: {df_feat.shape}")

    # Main 80/20 temporal split
    print(f"\nRunning main {int(MAIN_TRAIN_FRAC*100)}/{int((1-MAIN_TRAIN_FRAC)*100)} temporal split...")
    train_main, test_main = split_by_time_fraction(df_feat, MAIN_TRAIN_FRAC, 1.0 - MAIN_TRAIN_FRAC)

    X_train = train_main.drop(columns=["isfraud"])
    y_train = train_main["isfraud"].astype(int).values
    X_test  = test_main.drop(columns=["isfraud"])
    y_test  = test_main["isfraud"].astype(int).values

    print(f"  Train rows : {len(train_main):,} | fraud: {y_train.sum():,}")
    print(f"  Test rows  : {len(test_main):,}  | fraud: {y_test.sum():,}")

    if y_test.sum() == 0:
        print("\nWARNING: No fraud cases in the test split.")
        print("This can happen with small datasets. Try increasing TARGET_TRANSACTIONS")
        print("in generate_synthetic_data.py and regenerating.")
        return 1

    print("\nTraining XGBoost on synthetic data...")
    model  = train_xgb(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nEvaluating...")
    synth_metrics, synth_cm = evaluate_probs(y_test, y_prob)

    print(f"\n{'='*40}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*40}")
    print(f"  PaySim  AUPRC : {paysim['auprc']:.6f}  (reference)")
    print(f"  Synth   AUPRC : {synth_metrics.auprc:.6f}  (this run)")
    print(f"  Drop         : {(paysim['auprc'] - synth_metrics.auprc):.6f}")
    print(f"  Precision     : {synth_metrics.precision:.6f}")
    print(f"  Recall        : {synth_metrics.recall:.6f}")
    print(f"  F1            : {synth_metrics.f1:.6f}")
    print(f"{'='*40}")

    # Save main metrics
    metrics_path = os.path.join(OUTDIR, "synthetic_metrics_main.csv")
    pd.DataFrame([synth_metrics.__dict__]).to_csv(metrics_path, index=False)
    print(f"\n  Saved metrics : {metrics_path}")

    # Precision-recall curve
    p, r, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(r, p, label=f"Synthetic AUPRC={synth_metrics.auprc:.4f}")
    plt.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5, label="Precision=0.9")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Synthetic Dataset")
    plt.legend()
    plt.tight_layout()
    pr_path = os.path.join(OUTDIR, "pr_curve_synthetic.png")
    plt.savefig(pr_path, dpi=200)
    plt.close()
    print(f"  Saved PR curve: {pr_path}")

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=X_train.columns)
    imp_path = os.path.join(OUTDIR, "feature_importance_synthetic.csv")
    imp.sort_values(ascending=False).head(30).to_csv(imp_path)
    print(f"  Saved feature importance: {imp_path}")

    # Temporal sensitivity
    print("\nRunning temporal sensitivity tests on synthetic data...")
    sens_rows = []
    for frac in SENSITIVITY_FRACS:
        tr, te = split_by_time_fraction(df_feat, frac, SENSITIVITY_WIDTH)
        if len(tr) == 0 or len(te) == 0:
            continue
        y_tr = tr["isfraud"].astype(int).values
        y_te = te["isfraud"].astype(int).values
        if y_te.sum() == 0:
            print(f"  Skipping frac={frac} — no fraud in test window")
            continue
        m_s  = train_xgb(tr.drop(columns=["isfraud"]), y_tr)
        prob = m_s.predict_proba(te.drop(columns=["isfraud"]))[:, 1]
        met, _ = evaluate_probs(y_te, prob)
        sens_rows.append({
            "train_frac_end": frac,
            "test_width":     SENSITIVITY_WIDTH,
            "train_rows":     len(tr),
            "test_rows":      len(te),
            "test_auprc":     met.auprc,
            "test_precision": met.precision,
            "test_recall":    met.recall,
            "test_f1":        met.f1,
        })
        print(f"  Train 0-{int(frac*100)}% | Test next {int(SENSITIVITY_WIDTH*100)}% "
              f"=> AUPRC={met.auprc:.6f}")

    synth_sensitivity = pd.DataFrame(sens_rows)
    sens_path = os.path.join(OUTDIR, "synthetic_temporal_sensitivity.csv")
    synth_sensitivity.to_csv(sens_path, index=False)
    print(f"  Saved sensitivity: {sens_path}")

    # Full comparison report
    print("\nWriting comparison report...")
    write_comparison_report(synth_metrics, synth_cm, synth_sensitivity, paysim, OUTDIR)

    print("\nDone! All outputs saved to:", OUTDIR)
    print("\nNext steps:")
    print("  1. Open comparison_report.md for your report/notebook")
    print("  2. Check pr_curve_synthetic.png against your PaySim PR curve")
    print("  3. Note the verdict and use it to discuss simulator overfitting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
