#!/usr/bin/env python3
"""
generate_advanced_synthetic.py
================================
Third-generation synthetic dataset generator. Specifically designed to
defeat the top panel features identified from feature importance analysis:

    dest_tenure          38.5%  <-- mule has 0 tenure (brand new account)
    orig_txn_count_prev  27.7%  <-- victim has low activity before fraud
    dest_txn_count_prev   8.7%  <-- mule has 0 prior transactions

These three features alone accounted for 75% of model decisions.
All five complications from generate_realistic_synthetic.py are retained
since they attack the remaining 25% (balance-based features).

NEW IN THIS VERSION:
--------------------
6. MULE ACCOUNT WARM-UP
   Mule accounts participate in 10-50 normal transactions BEFORE
   the fraud event. This makes dest_tenure and dest_txn_count_prev
   look indistinguishable from normal accounts.

7. VICTIM ACCOUNT WARM-UP
   Victim accounts are specifically chosen from the most active
   customers (high orig_txn_count_prev) so the fraud transaction
   doesn't stand out as coming from a low-activity account.

8. MULE ACCOUNTS LOOK LIKE REGULAR ACCOUNTS
   Mule accounts send and receive money normally before fraud,
   building realistic cumulative amount and timing history.

RETAINED FROM generate_realistic_synthetic.py:
-----------------------------------------------
1. Partial drains (60-95% of balance stolen)
2. Balance noise (fees/rounding discrepancies)
3. Split fraud transactions (2-3 transfers)
4. Legitimate large transfers resembling fraud
5. Normal accounts occasionally hitting zero

OUTPUT:
    advanced_synthetic_data.csv
    advanced_generator_summary.txt

USAGE:
    python generate_advanced_synthetic.py
    Then update test_on_synthetic.py:
        SYNTHETIC_CSV = "advanced_synthetic_data.csv"
    Then run: python test_on_synthetic.py
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# -------------------------------------------------------
# SETTINGS (same as before for comparability)
# -------------------------------------------------------

RANDOM_SEED = 17

N_STEPS     = 600
N_CUSTOMERS = 3_000
N_MERCHANTS = 500

TARGET_TRANSACTIONS = 500_000
TARGET_FRAUD_RATE   = 0.0013
N_FRAUD_CHAINS      = int(TARGET_TRANSACTIONS * TARGET_FRAUD_RATE)

BALANCE_MEAN = 250_000
BALANCE_STD  = 400_000
BALANCE_MIN  = 0
BALANCE_MAX  = 15_000_000

AMOUNT_MEAN = 120_000
AMOUNT_STD  = 280_000
AMOUNT_MIN  = 1.0
AMOUNT_MAX  = 3_000_000

TYPE_PROBS = {
    "CASH_IN":   0.20,
    "CASH_OUT":  0.33,
    "DEBIT":     0.02,
    "PAYMENT":   0.36,
    "TRANSFER":  0.09,
}

# -------------------------------------------------------
# COMPLICATION SETTINGS (retained from v2)
# -------------------------------------------------------

PARTIAL_DRAIN_MIN         = 0.60
PARTIAL_DRAIN_MAX         = 0.95
BALANCE_NOISE_RATE        = 0.02
BALANCE_NOISE_PROB        = 0.70
SPLIT_FRAUD_PROB          = 0.30
SPLIT_MIN_PARTS           = 2
SPLIT_MAX_PARTS           = 3
LEGIT_LARGE_TRANSFER_PROB = 0.02
LEGIT_LARGE_AMOUNT_MIN    = 0.70
LEGIT_LARGE_AMOUNT_MAX    = 0.99
LEGIT_ZERO_DRAIN_PROB     = 0.01

# -------------------------------------------------------
# NEW: WARM-UP SETTINGS (v3 additions)
# -------------------------------------------------------

# Mule account warm-up: how many normal txns before fraud
MULE_WARMUP_MIN = 10    # Minimum prior transactions for mule
MULE_WARMUP_MAX = 50    # Maximum prior transactions for mule

# Mule warm-up step range: spread across early steps
MULE_WARMUP_STEP_MIN = 1
MULE_WARMUP_STEP_MAX = 200  # Warmup happens in first third of simulation

# Victim selection: choose from most active accounts
# (top X% by number of transactions)
VICTIM_MIN_ACTIVITY_PERCENTILE = 0.60  # Victim must be in top 40% most active

# Mule warm-up transaction types (realistic for a "normal" account)
MULE_WARMUP_TYPE_PROBS = {
    "CASH_IN":   0.35,   # Deposits (from merchants/banks)
    "PAYMENT":   0.40,   # Spending
    "TRANSFER":  0.15,   # Occasional transfers
    "CASH_OUT":  0.10,   # Some withdrawals
}

OUTPUT_FILE  = "advanced_synthetic_data.csv"
SUMMARY_FILE = "advanced_generator_summary.txt"


# -------------------------------------------------------
# Account
# -------------------------------------------------------

@dataclass
class Account:
    name:      str
    balance:   float
    txn_count: int   = 0   # Track activity level
    first_step: int  = -1  # Track tenure


def generate_customer_ids(n: int, seed_offset: int = 9_000_000) -> List[str]:
    return [f"C{seed_offset + i}" for i in range(n)]


def generate_merchant_ids(n: int, seed_offset: int = 5_000_000) -> List[str]:
    return [f"M{seed_offset + i}" for i in range(n)]


def generate_mule_ids(n: int, seed_offset: int = 8_000_000) -> List[str]:
    return [f"C{seed_offset + i}" for i in range(n)]


def sample_balance(rng: np.random.Generator) -> float:
    raw = rng.normal(BALANCE_MEAN, BALANCE_STD)
    return float(np.clip(raw, BALANCE_MIN, BALANCE_MAX))


def sample_amount(rng: np.random.Generator) -> float:
    raw = abs(rng.normal(AMOUNT_MEAN, AMOUNT_STD))
    return float(np.clip(raw, AMOUNT_MIN, AMOUNT_MAX))


def add_balance_noise(amount: float, rng: np.random.Generator) -> float:
    if rng.random() < BALANCE_NOISE_PROB:
        noise = rng.uniform(0, BALANCE_NOISE_RATE) * amount
        return float(noise * rng.choice([-1, 1]))
    return 0.0


# -------------------------------------------------------
# NEW: Mule account warm-up simulation
# -------------------------------------------------------

def simulate_mule_warmup(
    mule: Account,
    customers: List[Account],
    merchants: List[Account],
    n_warmup_txns: int,
    rng: np.random.Generator,
) -> List[dict]:
    """
    NEW COMPLICATION 6 & 8: Mule account warm-up.

    Generates realistic pre-fraud transaction history for a mule account.
    This makes dest_tenure and dest_txn_count_prev look like a normal account,
    directly defeating the top two features (38.5% + 8.7% = 47.2% importance).

    The mule:
    - Receives money (CASH_IN from merchants, incoming TRANSFERs)
    - Spends money (PAYMENT, CASH_OUT)
    - Builds a realistic balance history over time
    - Gets a realistic first_step (tenure) spread across early simulation
    """
    type_list    = list(MULE_WARMUP_TYPE_PROBS.keys())
    type_weights = list(MULE_WARMUP_TYPE_PROBS.values())

    # Spread warmup transactions across early steps
    warmup_steps = sorted(
        rng.integers(MULE_WARMUP_STEP_MIN, MULE_WARMUP_STEP_MAX, size=n_warmup_txns)
        .tolist()
    )

    rows = []
    for step in warmup_steps:
        tx_type = rng.choice(type_list, p=type_weights)
        noise   = add_balance_noise(sample_amount(rng), rng)

        if tx_type == "CASH_IN":
            # Mule receives money from a merchant (deposit)
            merchant = merchants[rng.integers(0, len(merchants))]
            amount   = sample_amount(rng)
            old_bal_mule = mule.balance
            mule.balance += amount + noise
            if mule.first_step == -1:
                mule.first_step = step
            mule.txn_count += 1
            rows.append({
                "step":           step,
                "type":           "CASH_IN",
                "amount":         round(amount, 2),
                "nameOrig":       mule.name,
                "oldbalanceOrg":  round(old_bal_mule, 2),
                "newbalanceOrig": round(mule.balance, 2),
                "nameDest":       merchant.name,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "isFraud":        0,
                "isFlaggedFraud": 0,
            })

        elif tx_type == "PAYMENT":
            # Mule pays a merchant
            merchant = merchants[rng.integers(0, len(merchants))]
            amount   = min(sample_amount(rng), mule.balance)
            if amount < 1.0:
                continue
            old_bal_mule = mule.balance
            mule.balance = max(0.0, mule.balance - amount + noise)
            if mule.first_step == -1:
                mule.first_step = step
            mule.txn_count += 1
            rows.append({
                "step":           step,
                "type":           "PAYMENT",
                "amount":         round(amount, 2),
                "nameOrig":       mule.name,
                "oldbalanceOrg":  round(old_bal_mule, 2),
                "newbalanceOrig": round(mule.balance, 2),
                "nameDest":       merchant.name,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "isFraud":        0,
                "isFlaggedFraud": 0,
            })

        elif tx_type == "TRANSFER":
            # Mule sends money to another customer (looks normal)
            dest = customers[rng.integers(0, len(customers))]
            amount = min(sample_amount(rng), mule.balance * 0.3)
            if amount < 1.0:
                continue
            old_bal_mule = mule.balance
            old_bal_dest = dest.balance
            mule.balance = max(0.0, mule.balance - amount + noise)
            dest.balance += amount
            if mule.first_step == -1:
                mule.first_step = step
            mule.txn_count += 1
            rows.append({
                "step":           step,
                "type":           "TRANSFER",
                "amount":         round(amount, 2),
                "nameOrig":       mule.name,
                "oldbalanceOrg":  round(old_bal_mule, 2),
                "newbalanceOrig": round(mule.balance, 2),
                "nameDest":       dest.name,
                "oldbalanceDest": round(old_bal_dest, 2),
                "newbalanceDest": round(dest.balance, 2),
                "isFraud":        0,
                "isFlaggedFraud": 0,
            })

        elif tx_type == "CASH_OUT":
            # Mule withdraws some cash
            merchant = merchants[rng.integers(0, len(merchants))]
            amount   = min(sample_amount(rng), mule.balance * 0.2)
            if amount < 1.0:
                continue
            old_bal_mule = mule.balance
            mule.balance = max(0.0, mule.balance - amount + noise)
            if mule.first_step == -1:
                mule.first_step = step
            mule.txn_count += 1
            rows.append({
                "step":           step,
                "type":           "CASH_OUT",
                "amount":         round(amount, 2),
                "nameOrig":       mule.name,
                "oldbalanceOrg":  round(old_bal_mule, 2),
                "newbalanceOrig": round(mule.balance, 2),
                "nameDest":       merchant.name,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "isFraud":        0,
                "isFlaggedFraud": 0,
            })

    return rows


# -------------------------------------------------------
# Normal transactions
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

        # Complication 4: legit large transfers
        if tx_type == "TRANSFER" and rng.random() < LEGIT_LARGE_TRANSFER_PROB:
            drain_frac = rng.uniform(LEGIT_LARGE_AMOUNT_MIN, LEGIT_LARGE_AMOUNT_MAX)
            amount = orig.balance * drain_frac if orig.balance > 100 else sample_amount(rng)
        # Complication 5: legit zero drain
        elif rng.random() < LEGIT_ZERO_DRAIN_PROB and orig.balance > 100:
            amount = orig.balance
        else:
            amount = sample_amount(rng)

        old_bal_orig = orig.balance
        old_bal_dest = dest.balance
        noise        = add_balance_noise(amount, rng)  # Complication 2

        if tx_type in ("CASH_OUT", "TRANSFER", "PAYMENT", "DEBIT"):
            actual_amount = min(amount, orig.balance)
            orig.balance  = max(0.0, orig.balance - actual_amount + noise)
            if tx_type == "TRANSFER":
                dest.balance += actual_amount + noise
        elif tx_type == "CASH_IN":
            orig.balance += amount + noise
            actual_amount = amount
        else:
            actual_amount = amount

        # Track activity for victim selection later
        orig.txn_count += 1
        if orig.first_step == -1:
            orig.first_step = step

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
# Fraud simulation (with all complications + warm-up)
# -------------------------------------------------------

def simulate_fraud_chain_advanced(
    victim: Account,
    mule: Account,
    step: int,
    rng: np.random.Generator,
) -> List[dict]:
    """
    Full fraud chain with all complications retained from v2.
    Mule accounts have been pre-warmed up before this is called,
    so dest_tenure and dest_txn_count_prev look normal.
    """
    rows = []

    if victim.balance < 100:
        return []

    # Complication 1: partial drain
    drain_frac     = rng.uniform(PARTIAL_DRAIN_MIN, PARTIAL_DRAIN_MAX)
    total_to_steal = victim.balance * drain_frac

    # Complication 3: split transactions
    use_split = rng.random() < SPLIT_FRAUD_PROB
    if use_split:
        n_parts      = int(rng.integers(SPLIT_MIN_PARTS, SPLIT_MAX_PARTS + 1))
        raw_splits   = rng.dirichlet(np.ones(n_parts))
        split_amounts = (raw_splits * total_to_steal).tolist()
    else:
        n_parts       = 1
        split_amounts = [total_to_steal]

    # TRANSFER(s): victim → mule
    for part_idx, transfer_amount in enumerate(split_amounts):
        transfer_amount = max(1.0, round(transfer_amount, 2))
        noise           = add_balance_noise(transfer_amount, rng)  # Complication 2

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

    # CASH_OUT: mule → merchant
    cashout_step    = step + n_parts
    cashout_amount  = mule.balance
    merchant_id     = f"M{rng.integers(5_000_000, 5_000_000 + 500)}"
    noise_co        = add_balance_noise(cashout_amount, rng)

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
    print("  Advanced Synthetic Generator (v3 — Tenure-Aware)")
    print("=" * 65)
    print("\nComplications active:")
    print(f"  1. Partial drains        : {int(PARTIAL_DRAIN_MIN*100)}-{int(PARTIAL_DRAIN_MAX*100)}% of balance stolen")
    print(f"  2. Balance noise         : up to {int(BALANCE_NOISE_RATE*100)}% noise on {int(BALANCE_NOISE_PROB*100)}% of txns")
    print(f"  3. Split fraud txns      : {int(SPLIT_FRAUD_PROB*100)}% of chains split into {SPLIT_MIN_PARTS}-{SPLIT_MAX_PARTS} transfers")
    print(f"  4. Legit large transfers : {int(LEGIT_LARGE_TRANSFER_PROB*100)}% of normal TRANSFERs drain 70-99%")
    print(f"  5. Legit zero drains     : {int(LEGIT_ZERO_DRAIN_PROB*100)}% of normal txns fully drain sender")
    print(f"  6. Mule account warm-up  : {MULE_WARMUP_MIN}-{MULE_WARMUP_MAX} prior txns per mule account")
    print(f"  7. Active victim select  : victims from top {int((1-VICTIM_MIN_ACTIVITY_PERCENTILE)*100)}% most active accounts")

    rng = np.random.default_rng(RANDOM_SEED)

    # Generate accounts
    print("\nGenerating accounts...")
    customer_ids = generate_customer_ids(N_CUSTOMERS)
    merchant_ids = generate_merchant_ids(N_MERCHANTS)
    mule_ids     = generate_mule_ids(N_FRAUD_CHAINS)

    customers = [Account(name=cid, balance=sample_balance(rng)) for cid in customer_ids]
    merchants = [Account(name=mid, balance=0.0) for mid in merchant_ids]
    mules     = [Account(name=mid, balance=sample_balance(rng) * 0.3) for mid in mule_ids]

    print(f"  Customers : {len(customers):,}")
    print(f"  Merchants : {len(merchants):,}")
    print(f"  Mules     : {len(mules):,} (will be warmed up)")

    # Phase 1: Warm up mule accounts FIRST
    # This generates pre-fraud transaction history for each mule
    print(f"\nPhase 1: Warming up {len(mules):,} mule accounts...")
    warmup_rows = []
    for mule in mules:
        n_warmup = int(rng.integers(MULE_WARMUP_MIN, MULE_WARMUP_MAX + 1))
        warmup   = simulate_mule_warmup(mule, customers, merchants, n_warmup, rng)
        warmup_rows.extend(warmup)
    print(f"  Generated {len(warmup_rows):,} mule warm-up transaction rows")
    print(f"  Average warm-up txns per mule: {len(warmup_rows)/len(mules):.1f}")

    # Phase 2: Normal transactions (bulk of dataset)
    n_normal = TARGET_TRANSACTIONS - (N_FRAUD_CHAINS * 3) - len(warmup_rows)
    n_normal = max(n_normal, 100_000)
    print(f"\nPhase 2: Simulating {n_normal:,} normal transactions...")
    normal_rows = simulate_normal_transactions(customers, merchants, n_normal, rng)
    print(f"  Generated {len(normal_rows):,} normal rows")

    # Phase 3: Select victims from most active customers
    # NEW COMPLICATION 7: Victims are high-activity accounts so
    # orig_txn_count_prev looks normal at time of fraud
    print(f"\nPhase 3: Selecting active victims (top {int((1-VICTIM_MIN_ACTIVITY_PERCENTILE)*100)}% by activity)...")
    activity_threshold = np.quantile(
        [c.txn_count for c in customers],
        VICTIM_MIN_ACTIVITY_PERCENTILE
    )
    active_customers = [
        c for c in customers
        if c.txn_count >= activity_threshold and c.balance > 1000
    ]
    print(f"  Active eligible victims: {len(active_customers):,} "
          f"(min {int(activity_threshold)} prior txns)")

    actual_fraud_chains = min(N_FRAUD_CHAINS, len(active_customers), len(mules))
    rng_indices  = rng.choice(len(active_customers), size=actual_fraud_chains, replace=False)
    fraud_victims = [active_customers[i] for i in rng_indices]
    fraud_mules   = mules[:actual_fraud_chains]

    # Spread fraud events in the LATER half of the simulation
    # (after mule warm-up which happens in first ~200 steps)
    fraud_steps = sorted(
        rng.integers(MULE_WARMUP_STEP_MAX + 10, N_STEPS - 4, size=actual_fraud_chains)
        .tolist()
    )

    print(f"\nPhase 4: Simulating {actual_fraud_chains} advanced fraud chains...")
    fraud_rows = []
    skipped    = 0
    n_split    = 0
    for victim, mule, step in zip(fraud_victims, fraud_mules, fraud_steps):
        chain = simulate_fraud_chain_advanced(victim, mule, int(step), rng)
        if chain:
            fraud_rows.extend(chain)
            if len(chain) > 2:
                n_split += 1
        else:
            skipped += 1

    print(f"  Generated {len(fraud_rows):,} fraud rows")
    print(f"  Split chains    : {n_split}")
    print(f"  Skipped chains  : {skipped}")

    # Combine ALL rows and sort by step
    print("\nCombining all phases and sorting by step...")
    all_rows = warmup_rows + normal_rows + fraud_rows
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

    # Summary
    total      = len(df)
    n_fraud    = int(df["isFraud"].sum())
    fraud_rate = n_fraud / total * 100

    print("\n" + "=" * 65)
    print("DATASET SUMMARY")
    print("=" * 65)
    print(f"  Total rows        : {total:,}")
    print(f"  Warm-up rows      : {len(warmup_rows):,}  (mule pre-history, isFraud=0)")
    print(f"  Normal rows       : {len(normal_rows):,}")
    print(f"  Fraud rows        : {n_fraud:,}  ({fraud_rate:.4f}%)")
    print(f"\n  Transaction types:")
    for t, c in df["type"].value_counts().items():
        print(f"    {t:<12} : {c:>8,}  ({c/total*100:.1f}%)")
    fraud_df = df[df["isFraud"] == 1]
    print(f"\n  Fraud types: {fraud_df['type'].value_counts().to_dict()}")

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
    print(f"\nSaving to: {out_path}")
    df.to_csv(out_path, index=False)
    print(f"  Saved {total:,} rows, {len(df.columns)} columns")

    # Save summary
    summary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), SUMMARY_FILE)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Advanced Synthetic Generator v3 — Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write("KEY INSIGHT FROM FEATURE IMPORTANCE\n")
        f.write("-" * 40 + "\n")
        f.write("dest_tenure         : 38.5% importance  <- mule was brand new\n")
        f.write("orig_txn_count_prev : 27.7% importance  <- victim was low activity\n")
        f.write("dest_txn_count_prev :  8.7% importance  <- mule had 0 prior txns\n")
        f.write("Total from these 3  : 75.0%\n\n")
        f.write("FIX APPLIED\n")
        f.write("-" * 40 + "\n")
        f.write(f"Mule warm-up txns   : {MULE_WARMUP_MIN}-{MULE_WARMUP_MAX} prior transactions\n")
        f.write(f"Victim selection    : top {int((1-VICTIM_MIN_ACTIVITY_PERCENTILE)*100)}% most active accounts\n")
        f.write(f"Fraud timing        : steps {MULE_WARMUP_STEP_MAX+10}+ (after warm-up period)\n\n")
        f.write("ALL COMPLICATIONS\n")
        f.write("-" * 40 + "\n")
        f.write(f"1. Partial drains        : {PARTIAL_DRAIN_MIN*100:.0f}-{PARTIAL_DRAIN_MAX*100:.0f}%\n")
        f.write(f"2. Balance noise         : up to {BALANCE_NOISE_RATE*100:.0f}%\n")
        f.write(f"3. Split fraud txns      : {SPLIT_FRAUD_PROB*100:.0f}% of chains\n")
        f.write(f"4. Legit large transfers : {LEGIT_LARGE_TRANSFER_PROB*100:.0f}% of normal TRANSFERs\n")
        f.write(f"5. Legit zero drains     : {LEGIT_ZERO_DRAIN_PROB*100:.0f}% of normal txns\n")
        f.write(f"6. Mule account warm-up  : {MULE_WARMUP_MIN}-{MULE_WARMUP_MAX} prior txns\n")
        f.write(f"7. Active victim select  : top {int((1-VICTIM_MIN_ACTIVITY_PERCENTILE)*100)}% by txn count\n\n")
        f.write("GENERATION STATS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total rows      : {total:,}\n")
        f.write(f"Fraud rows      : {n_fraud:,} ({fraud_rate:.4f}%)\n")
        f.write(f"Split chains    : {n_split}\n")
        f.write(f"Skipped chains  : {skipped}\n\n")
        f.write("HOW TO TEST\n")
        f.write("-" * 40 + "\n")
        f.write("Update SYNTHETIC_CSV in test_on_synthetic.py:\n")
        f.write(f"  SYNTHETIC_CSV = '{OUTPUT_FILE}'\n")
        f.write("Then run: python test_on_synthetic.py\n\n")
        f.write("INTERPRETATION TABLE\n")
        f.write("-" * 40 + "\n")
        f.write("AUPRC >= 0.97  : Model is remarkably robust\n")
        f.write("AUPRC 0.85-0.97: Healthy — generalizes with some struggle\n")
        f.write("AUPRC 0.70-0.85: Significant drop — some features fragile\n")
        f.write("AUPRC < 0.70   : Model relied heavily on tenure/count signals\n")

    print(f"  Summary saved: {summary_path}")
    print("\nDone.")
    print(f"\nNext step:")
    print(f"  1. Update SYNTHETIC_CSV in test_on_synthetic.py to '{OUTPUT_FILE}'")
    print(f"  2. Run: python test_on_synthetic.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())