#!/usr/bin/env python3
"""
generate_realistic_synthetic.py
================================
Generates a HARDER, more realistic synthetic transaction dataset that
deliberately challenges the fraud detection model with real-world complications.

COMPLICATIONS ADDED (vs generate_synthetic_data.py):
------------------------------------------------------
1. PARTIAL DRAINS
   Fraudster takes 60-95% of victim balance (not 100%).
   This breaks:
     - org_balance_minus_amount_gap (no longer exactly 0)
     - amount_over_oldorg (no longer exactly 1.0)
     - delta_org (not a perfect full-balance depletion)

2. BALANCE NOISE
   Small random discrepancies added to balance updates
   (simulating fees, FX rounding, processing delays).
   This breaks:
     - All accounting consistency features
     - Exact balance delta matching

3. SPLIT FRAUD TRANSACTIONS
   ~30% of fraud chains split the transfer across 2-3 steps
   instead of one clean TRANSFER → CASH_OUT.
   This breaks:
     - The clean two-step sequential pattern
     - Panel cumulative features (spread across time)

4. LEGITIMATE LARGE TRANSFERS
   ~2% of normal transactions are large TRANSFERs that
   superficially resemble fraud (high amount, C→C, large delta).
   This breaks:
     - The model's reliance on amount/transfer type as a signal

5. NORMAL ACCOUNTS HITTING ZERO
   ~1% of normal accounts legitimately drain to zero balance
   (e.g., account closure, full withdrawal).
   This breaks:
     - "Balance goes to zero = fraud" heuristic

PURPOSE:
    A perfect score on the clean synthetic data (AUPRC=1.0) is suspicious
    because the fraud pattern was too deterministic. This harder dataset
    tests whether the model's learned signals are robust to real-world noise
    and ambiguity, or whether they collapse without clean deterministic patterns.

EXPECTED OUTCOME:
    AUPRC should drop from ~1.0 toward a more realistic range (0.75-0.95).
    If it stays near 1.0, the model is remarkably robust.
    If it drops significantly, we've identified which features were fragile.

OUTPUT:
    realistic_synthetic_data.csv    — harder dataset for testing
    realistic_generator_summary.txt — documents all parameters

USAGE:
    python generate_realistic_synthetic.py
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple

# -------------------------------------------------------
# SETTINGS
# -------------------------------------------------------

RANDOM_SEED = 13

N_STEPS      = 600
N_CUSTOMERS  = 3_000
N_MERCHANTS  = 500

TARGET_TRANSACTIONS = 500_000
TARGET_FRAUD_RATE   = 0.0013
N_FRAUD_CHAINS      = int(TARGET_TRANSACTIONS * TARGET_FRAUD_RATE)

# Balance / amount distributions (same as clean synthetic)
BALANCE_MEAN = 250_000
BALANCE_STD  = 400_000
BALANCE_MIN  = 0
BALANCE_MAX  = 15_000_000

AMOUNT_MEAN  = 120_000
AMOUNT_STD   = 280_000
AMOUNT_MIN   = 1.0
AMOUNT_MAX   = 3_000_000

TYPE_PROBS = {
    "CASH_IN":   0.20,
    "CASH_OUT":  0.33,
    "DEBIT":     0.02,
    "PAYMENT":   0.36,
    "TRANSFER":  0.09,
}

# -------------------------------------------------------
# COMPLICATION SETTINGS
# -------------------------------------------------------

# 1. Partial drains: fraudster takes this fraction of victim balance
PARTIAL_DRAIN_MIN = 0.60   # At least 60% drained
PARTIAL_DRAIN_MAX = 0.95   # At most 95% drained

# 2. Balance noise: small fee/rounding discrepancy added to balances
#    Applied as a fraction of transaction amount
BALANCE_NOISE_RATE    = 0.02   # Up to 2% of amount as noise
BALANCE_NOISE_PROB    = 0.70   # 70% of transactions get some noise

# 3. Split fraud transactions: fraction of fraud chains that split
SPLIT_FRAUD_PROB      = 0.30   # 30% of fraud chains use 2-3 transfers
SPLIT_MIN_PARTS       = 2
SPLIT_MAX_PARTS       = 3

# 4. Legitimate large transfers resembling fraud
LEGIT_LARGE_TRANSFER_PROB = 0.02   # 2% of normal transactions
LEGIT_LARGE_AMOUNT_MIN    = 0.70   # Takes 70-99% of sender balance
LEGIT_LARGE_AMOUNT_MAX    = 0.99

# 5. Normal accounts hitting zero balance
LEGIT_ZERO_DRAIN_PROB = 0.01   # 1% of normal transactions fully drain

OUTPUT_FILE  = "realistic_synthetic_data.csv"
SUMMARY_FILE = "realistic_generator_summary.txt"


# -------------------------------------------------------
# Account
# -------------------------------------------------------

@dataclass
class Account:
    name:    str
    balance: float


def generate_customer_ids(n: int, seed_offset: int = 9_000_000) -> List[str]:
    return [f"C{seed_offset + i}" for i in range(n)]


def generate_merchant_ids(n: int, seed_offset: int = 5_000_000) -> List[str]:
    return [f"M{seed_offset + i}" for i in range(n)]


def sample_balance(rng: np.random.Generator) -> float:
    raw = rng.normal(BALANCE_MEAN, BALANCE_STD)
    return float(np.clip(raw, BALANCE_MIN, BALANCE_MAX))


def sample_amount(rng: np.random.Generator) -> float:
    raw = abs(rng.normal(AMOUNT_MEAN, AMOUNT_STD))
    return float(np.clip(raw, AMOUNT_MIN, AMOUNT_MAX))


def add_balance_noise(amount: float, rng: np.random.Generator) -> float:
    """
    COMPLICATION 2: Balance noise.
    Simulates fees, rounding, FX discrepancies.
    Returns a small noise value to add to balance updates.
    """
    if rng.random() < BALANCE_NOISE_PROB:
        noise = rng.uniform(0, BALANCE_NOISE_RATE) * amount
        # Noise can be positive or negative
        return float(noise * rng.choice([-1, 1]))
    return 0.0


# -------------------------------------------------------
# Normal transaction simulation
# -------------------------------------------------------

def simulate_normal_transactions(
    customers: List[Account],
    merchants: List[Account],
    n_transactions: int,
    rng: np.random.Generator,
) -> List[dict]:
    type_list    = list(TYPE_PROBS.keys())
    type_weights = list(TYPE_PROBS.values())
    steps_per_tx = N_STEPS / n_transactions
    rows = []

    for i in range(n_transactions):
        step    = int(i * steps_per_tx) + 1
        tx_type = rng.choice(type_list, p=type_weights)
        orig    = customers[rng.integers(0, len(customers))]

        if tx_type in ("CASH_IN", "PAYMENT"):
            dest = merchants[rng.integers(0, len(merchants))]
        elif tx_type == "TRANSFER":
            dest = customers[rng.integers(0, len(customers))]
            while dest.name == orig.name:
                dest = customers[rng.integers(0, len(customers))]
        elif tx_type == "CASH_OUT":
            dest = merchants[rng.integers(0, len(merchants))]
        else:
            dest = merchants[rng.integers(0, len(merchants))]

        # ---- COMPLICATION 4: Legitimate large transfers ----
        # Some normal TRANSFERs take most of the sender's balance,
        # superficially resembling fraud
        if tx_type == "TRANSFER" and rng.random() < LEGIT_LARGE_TRANSFER_PROB:
            drain_frac = rng.uniform(LEGIT_LARGE_AMOUNT_MIN, LEGIT_LARGE_AMOUNT_MAX)
            amount = orig.balance * drain_frac if orig.balance > 100 else sample_amount(rng)
        # ---- COMPLICATION 5: Normal accounts hitting zero ----
        elif rng.random() < LEGIT_ZERO_DRAIN_PROB and orig.balance > 100:
            amount = orig.balance  # Full legitimate drain
        else:
            amount = sample_amount(rng)

        old_bal_orig = orig.balance
        old_bal_dest = dest.balance

        # ---- COMPLICATION 2: Balance noise on normal transactions ----
        noise = add_balance_noise(amount, rng)

        if tx_type in ("CASH_OUT", "TRANSFER", "PAYMENT", "DEBIT"):
            actual_amount = min(amount, orig.balance)
            orig.balance  = max(0.0, orig.balance - actual_amount + noise)
            if tx_type == "TRANSFER":
                dest.balance += actual_amount + noise
        elif tx_type == "CASH_IN":
            orig.balance  += amount + noise
            actual_amount  = amount
        else:
            actual_amount = amount

        rows.append({
            "step":           step,
            "type":           tx_type,
            "amount":         round(actual_amount, 2),
            "nameOrig":       orig.name,
            "oldbalanceOrg":  round(old_bal_orig, 2),
            "newbalanceOrig": round(orig.balance, 2),
            "nameDest":       dest.name,
            "oldbalanceDest": round(old_bal_dest, 2),
            "newbalanceDest": round(dest.balance, 2),
            "isFraud":        0,
            "isFlaggedFraud": 0,
        })

    return rows


# -------------------------------------------------------
# Fraud simulation with all complications
# -------------------------------------------------------

def simulate_fraud_chain_realistic(
    victim: Account,
    mule: Account,
    step: int,
    rng: np.random.Generator,
) -> List[dict]:
    """
    Realistic fraud chain with three complications:

    COMPLICATION 1 — Partial drain:
        Fraudster takes 60-95% of victim balance (not 100%).

    COMPLICATION 2 — Balance noise:
        Small discrepancies in balance updates.

    COMPLICATION 3 — Split transactions:
        30% of chains split the TRANSFER into 2-3 smaller steps
        spread across consecutive time steps.
    """
    rows = []

    if victim.balance < 100:
        return []  # Not worth targeting

    # ---- COMPLICATION 1: Partial drain ----
    drain_frac      = rng.uniform(PARTIAL_DRAIN_MIN, PARTIAL_DRAIN_MAX)
    total_to_steal  = victim.balance * drain_frac

    # ---- COMPLICATION 3: Split transactions ----
    use_split = rng.random() < SPLIT_FRAUD_PROB
    if use_split:
        n_parts = int(rng.integers(SPLIT_MIN_PARTS, SPLIT_MAX_PARTS + 1))
        # Split amounts unevenly (not perfectly equal — more realistic)
        raw_splits = rng.dirichlet(np.ones(n_parts))
        split_amounts = (raw_splits * total_to_steal).tolist()
    else:
        n_parts       = 1
        split_amounts = [total_to_steal]

    # ---- Execute TRANSFER(s): victim → mule ----
    for part_idx, transfer_amount in enumerate(split_amounts):
        transfer_amount = max(1.0, round(transfer_amount, 2))

        # ---- COMPLICATION 2: Balance noise on fraud transactions ----
        noise = add_balance_noise(transfer_amount, rng)

        old_bal_victim = victim.balance
        old_bal_mule   = mule.balance

        victim.balance = max(0.0, victim.balance - transfer_amount + noise)
        mule.balance  += transfer_amount + noise

        rows.append({
            "step":           step + part_idx,
            "type":           "TRANSFER",
            "amount":         transfer_amount,
            "nameOrig":       victim.name,
            "oldbalanceOrg":  round(old_bal_victim, 2),
            "newbalanceOrig": round(victim.balance, 2),
            "nameDest":       mule.name,
            "oldbalanceDest": round(old_bal_mule, 2),
            "newbalanceDest": round(mule.balance, 2),
            "isFraud":        1,
            "isFlaggedFraud": 0,
        })

    # ---- Execute CASH_OUT: mule → merchant ----
    cashout_step   = step + n_parts   # Happens after all transfers
    cashout_amount = mule.balance
    merchant_id    = f"M{rng.integers(5_000_000, 5_000_000 + 500)}"

    # ---- COMPLICATION 2: Noise on cash-out too ----
    noise_co = add_balance_noise(cashout_amount, rng)

    old_bal_mule_co = mule.balance
    mule.balance    = max(0.0, mule.balance - cashout_amount + noise_co)

    rows.append({
        "step":           cashout_step,
        "type":           "CASH_OUT",
        "amount":         round(cashout_amount, 2),
        "nameOrig":       mule.name,
        "oldbalanceOrg":  round(old_bal_mule_co, 2),
        "newbalanceOrig": round(mule.balance, 2),
        "nameDest":       merchant_id,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "isFraud":        1,
        "isFlaggedFraud": 0,
    })

    return rows


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main() -> int:
    print("=" * 65)
    print("  Realistic Synthetic Transaction Data Generator")
    print("=" * 65)
    print("\nComplications enabled:")
    print(f"  1. Partial drains        : {int(PARTIAL_DRAIN_MIN*100)}-{int(PARTIAL_DRAIN_MAX*100)}% of balance stolen")
    print(f"  2. Balance noise         : up to {int(BALANCE_NOISE_RATE*100)}% fee/rounding noise ({int(BALANCE_NOISE_PROB*100)}% of txns)")
    print(f"  3. Split fraud txns      : {int(SPLIT_FRAUD_PROB*100)}% of chains split into {SPLIT_MIN_PARTS}-{SPLIT_MAX_PARTS} transfers")
    print(f"  4. Legit large transfers : {int(LEGIT_LARGE_TRANSFER_PROB*100)}% of normal TRANSFERs take 70-99% of balance")
    print(f"  5. Legit zero drains     : {int(LEGIT_ZERO_DRAIN_PROB*100)}% of normal txns fully drain sender balance")

    rng = np.random.default_rng(RANDOM_SEED)

    # Generate accounts
    print("\nGenerating accounts...")
    customer_ids = generate_customer_ids(N_CUSTOMERS)
    merchant_ids = generate_merchant_ids(N_MERCHANTS)
    customers = [Account(name=cid, balance=sample_balance(rng)) for cid in customer_ids]
    merchants = [Account(name=mid, balance=0.0) for mid in merchant_ids]
    print(f"  Customers : {len(customers):,}")
    print(f"  Merchants : {len(merchants):,}")

    # Normal transactions
    n_normal = TARGET_TRANSACTIONS - (N_FRAUD_CHAINS * 3)  # Extra room for split chains
    print(f"\nSimulating {n_normal:,} normal transactions...")
    normal_rows = simulate_normal_transactions(customers, merchants, n_normal, rng)
    print(f"  Generated {len(normal_rows):,} normal rows")

    # Count legit complications in normal transactions
    n_large_legit = sum(
        1 for r in normal_rows
        if r["type"] == "TRANSFER" and r["amount"] > 0.70 * r["oldbalanceOrg"]
        and r["oldbalanceOrg"] > 100
    )
    n_zero_drain = sum(
        1 for r in normal_rows
        if r["newbalanceOrig"] == 0.0 and r["isFraud"] == 0
    )
    print(f"  Large legit transfers (look like fraud) : {n_large_legit:,}")
    print(f"  Normal txns that hit zero balance       : {n_zero_drain:,}")

    # Fraud chains
    print(f"\nSimulating {N_FRAUD_CHAINS} realistic fraud chains...")
    eligible_victims = [c for c in customers if c.balance > 1000]

    actual_fraud_chains = min(N_FRAUD_CHAINS, len(eligible_victims))
    rng_indices  = rng.choice(len(eligible_victims), size=actual_fraud_chains, replace=False)
    fraud_victims = [eligible_victims[i] for i in rng_indices]
    mule_accounts = [
        Account(name=f"C{8_000_000 + i}", balance=0.0)
        for i in range(actual_fraud_chains)
    ]
    fraud_steps = sorted(rng.integers(1, N_STEPS - 4, size=actual_fraud_chains).tolist())

    fraud_rows = []
    skipped    = 0
    n_split    = 0
    for victim, mule, step in zip(fraud_victims, mule_accounts, fraud_steps):
        chain = simulate_fraud_chain_realistic(victim, mule, int(step), rng)
        if chain:
            fraud_rows.extend(chain)
            # Count split chains (more than 2 rows = split)
            if len(chain) > 2:
                n_split += 1
        else:
            skipped += 1

    print(f"  Generated {len(fraud_rows):,} fraud rows from {actual_fraud_chains - skipped} chains")
    print(f"  Split chains (2-3 transfers) : {n_split}")
    print(f"  Single transfer chains       : {actual_fraud_chains - skipped - n_split}")
    print(f"  Skipped (zero balance)       : {skipped}")

    # Combine and sort
    print("\nCombining and sorting...")
    all_rows = normal_rows + fraud_rows
    df = pd.DataFrame(all_rows)
    df = df.sort_values("step").reset_index(drop=True)

    # Verify schema
    expected_cols = [
        "step", "type", "amount",
        "nameOrig", "oldbalanceOrg", "newbalanceOrig",
        "nameDest", "oldbalanceDest", "newbalanceDest",
        "isFraud", "isFlaggedFraud"
    ]
    assert list(df.columns) == expected_cols, f"Schema mismatch: {list(df.columns)}"

    # Summary stats
    total      = len(df)
    n_fraud    = int(df["isFraud"].sum())
    fraud_rate = n_fraud / total * 100
    type_counts = df["type"].value_counts()

    print("\n" + "=" * 65)
    print("DATASET SUMMARY")
    print("=" * 65)
    print(f"  Total rows      : {total:,}")
    print(f"  Fraud rows      : {n_fraud:,}  ({fraud_rate:.4f}%)")
    print(f"  Non-fraud rows  : {total - n_fraud:,}")
    print(f"\n  Transaction type breakdown:")
    for t, c in type_counts.items():
        print(f"    {t:<12} : {c:>8,}  ({c/total*100:.1f}%)")

    fraud_df = df[df["isFraud"] == 1]
    print(f"\n  Fraud transaction types: {fraud_df['type'].value_counts().to_dict()}")

    print(f"\n  Amount stats (all):")
    print(f"    Mean : {df['amount'].mean():>12,.2f}")
    print(f"    Std  : {df['amount'].std():>12,.2f}")
    print(f"    Max  : {df['amount'].max():>12,.2f}")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
    print(f"\nSaving to: {out_path}")
    df.to_csv(out_path, index=False)
    print(f"  Saved {total:,} rows")

    # Save summary
    summary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), SUMMARY_FILE)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Realistic Synthetic Generator — Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write("COMPLICATIONS ADDED\n")
        f.write("-" * 30 + "\n")
        f.write(f"1. Partial drains        : {PARTIAL_DRAIN_MIN*100:.0f}-{PARTIAL_DRAIN_MAX*100:.0f}% of victim balance stolen\n")
        f.write(f"   Breaks: org_balance_minus_amount_gap, amount_over_oldorg\n\n")
        f.write(f"2. Balance noise         : up to {BALANCE_NOISE_RATE*100:.0f}% fee/rounding noise on {BALANCE_NOISE_PROB*100:.0f}% of txns\n")
        f.write(f"   Breaks: all accounting consistency features\n\n")
        f.write(f"3. Split fraud txns      : {SPLIT_FRAUD_PROB*100:.0f}% of chains use {SPLIT_MIN_PARTS}-{SPLIT_MAX_PARTS} transfers\n")
        f.write(f"   Breaks: clean TRANSFER->CASHOUT two-step pattern\n\n")
        f.write(f"4. Legit large transfers : {LEGIT_LARGE_TRANSFER_PROB*100:.0f}% of normal TRANSFERs take 70-99% of balance\n")
        f.write(f"   Breaks: amount/transfer type as a strong fraud signal\n\n")
        f.write(f"5. Legit zero drains     : {LEGIT_ZERO_DRAIN_PROB*100:.0f}% of normal txns fully drain sender\n")
        f.write(f"   Breaks: balance goes to zero = fraud heuristic\n\n")
        f.write("GENERATION STATS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total rows      : {total:,}\n")
        f.write(f"Fraud rows      : {n_fraud:,} ({fraud_rate:.4f}%)\n")
        f.write(f"Split chains    : {n_split}\n")
        f.write(f"Skipped chains  : {skipped}\n\n")
        f.write("HOW TO TEST\n")
        f.write("-" * 30 + "\n")
        f.write("Update SYNTHETIC_CSV in test_on_synthetic.py to:\n")
        f.write(f"  SYNTHETIC_CSV = '{OUTPUT_FILE}'\n")
        f.write("Then run: python test_on_synthetic.py\n\n")
        f.write("EXPECTED OUTCOME\n")
        f.write("-" * 30 + "\n")
        f.write("AUPRC should drop from ~1.0 toward a more realistic range.\n")
        f.write("If AUPRC stays near 1.0: model is remarkably robust.\n")
        f.write("If AUPRC drops significantly: fragile features identified.\n")

    print(f"  Summary saved: {summary_path}")
    print("\nDone.")
    print(f"\nNext step: update SYNTHETIC_CSV in test_on_synthetic.py to:")
    print(f"  SYNTHETIC_CSV = '{OUTPUT_FILE}'")
    print("Then run: python test_on_synthetic.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())