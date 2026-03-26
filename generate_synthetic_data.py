import os
import sys
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple

# -------------------------------------------------------
# SETTINGS — deliberately different from PaySim
# -------------------------------------------------------

RANDOM_SEED = 7       # PaySim used a different internal seed
N_STEPS = 600         # PaySim had 743 steps; we use fewer to save time
N_CUSTOMERS = 3_000   # PaySim had millions of unique accounts; we scale down
N_MERCHANTS = 500     # Merchant accounts (M-prefix)

# Target dataset size (approximate — actual size depends on step activity)
# PaySim had 6.36M rows; we generate a smaller but structurally equivalent dataset
TARGET_TRANSACTIONS = 500_000

# Fraud settings — same pattern as PaySim, different rate
# PaySim fraud rate: ~0.13% (8213 / 6362620)
TARGET_FRAUD_RATE = 0.0013   # Keep same rate for comparability
N_FRAUD_CHAINS = int(TARGET_TRANSACTIONS * TARGET_FRAUD_RATE)  # ~650 fraud chains

# Balance distribution — DIFFERENT from PaySim to change statistical fingerprint
# PaySim mean amount ~179k; we shift this distribution
BALANCE_MEAN = 250_000      # Different from PaySim's ~833k mean balance
BALANCE_STD  = 400_000      # Different spread
BALANCE_MIN  = 0
BALANCE_MAX  = 15_000_000   # Different cap (PaySim had ~59M max)

# Transaction amount distribution — DIFFERENT from PaySim
AMOUNT_MEAN  = 120_000      # PaySim mean was ~179k
AMOUNT_STD   = 280_000
AMOUNT_MIN   = 1.0
AMOUNT_MAX   = 3_000_000    # PaySim had ~92M max; we cap lower

# Transaction type probabilities — slightly different from PaySim mix
# PaySim: CASH_IN 22%, CASH_OUT 35%, DEBIT 0.6%, PAYMENT 34%, TRANSFER 8%
TYPE_PROBS = {
    "CASH_IN":   0.20,
    "CASH_OUT":  0.33,
    "DEBIT":     0.02,
    "PAYMENT":   0.36,
    "TRANSFER":  0.09,
}

OUTPUT_FILE   = "synthetic_transaction_data.csv"
SUMMARY_FILE  = "generator_summary.txt"

# -------------------------------------------------------
# Account generation
# -------------------------------------------------------

def generate_customer_ids(n: int, prefix: str = "C", seed_offset: int = 9_000_000) -> List[str]:
    """
    Generate unique customer account IDs with C-prefix.
    Offset ensures no overlap with PaySim IDs (which start from lower numbers).
    """
    ids = [f"{prefix}{seed_offset + i}" for i in range(n)]
    return ids

def generate_merchant_ids(n: int, seed_offset: int = 5_000_000) -> List[str]:
    """
    Generate unique merchant account IDs with M-prefix.
    """
    ids = [f"M{seed_offset + i}" for i in range(n)]
    return ids

def sample_balance(rng: np.random.Generator) -> float:
    """Sample an initial account balance from a log-normal-like distribution."""
    raw = rng.normal(BALANCE_MEAN, BALANCE_STD)
    return float(np.clip(raw, BALANCE_MIN, BALANCE_MAX))

def sample_amount(rng: np.random.Generator) -> float:
    """Sample a transaction amount."""
    raw = abs(rng.normal(AMOUNT_MEAN, AMOUNT_STD))
    return float(np.clip(raw, AMOUNT_MIN, AMOUNT_MAX))

# -------------------------------------------------------
# Normal (non-fraud) transaction simulation
# -------------------------------------------------------

@dataclass
class Account:
    name: str
    balance: float

def simulate_normal_transactions(
    customers: List[Account],
    merchants: List[Account],
    n_transactions: int,
    rng: np.random.Generator,
) -> List[dict]:
    """
    Simulate normal (non-fraudulent) transactions across all types.
    Accounts are mutated in place so balances evolve realistically over time.
    """
    type_list = list(TYPE_PROBS.keys())
    type_weights = list(TYPE_PROBS.values())

    # We'll spread transactions across N_STEPS
    steps_per_tx = N_STEPS / n_transactions
    rows = []

    for i in range(n_transactions):
        step = int(i * steps_per_tx) + 1
        tx_type = rng.choice(type_list, p=type_weights)

        # Pick originator
        orig = customers[rng.integers(0, len(customers))]

        # Pick destination
        if tx_type in ("CASH_IN", "PAYMENT"):
            # Merchant destinations
            dest = merchants[rng.integers(0, len(merchants))]
        elif tx_type == "TRANSFER":
            # Customer to customer
            dest = customers[rng.integers(0, len(customers))]
            while dest.name == orig.name:
                dest = customers[rng.integers(0, len(customers))]
        elif tx_type == "CASH_OUT":
            # Customer to merchant (ATM-style)
            dest = merchants[rng.integers(0, len(merchants))]
        else:  # DEBIT
            dest = merchants[rng.integers(0, len(merchants))]

        amount = sample_amount(rng)

        # Record pre-transaction balances
        old_bal_orig = orig.balance
        old_bal_dest = dest.balance

        # Update balances realistically
        if tx_type in ("CASH_OUT", "TRANSFER", "PAYMENT", "DEBIT"):
            # Money leaves originator
            actual_amount = min(amount, orig.balance)  # Can't overdraft
            orig.balance = max(0.0, orig.balance - actual_amount)
            if tx_type in ("CASH_IN", "TRANSFER"):
                dest.balance += actual_amount
        elif tx_type == "CASH_IN":
            # Money enters originator (deposit)
            orig.balance += amount
            actual_amount = amount
        else:
            actual_amount = amount

        new_bal_orig = orig.balance
        new_bal_dest = dest.balance

        rows.append({
            "step": step,
            "type": tx_type,
            "amount": round(actual_amount, 2),
            "nameOrig": orig.name,
            "oldbalanceOrg": round(old_bal_orig, 2),
            "newbalanceOrig": round(new_bal_orig, 2),
            "nameDest": dest.name,
            "oldbalanceDest": round(old_bal_dest, 2),
            "newbalanceDest": round(new_bal_dest, 2),
            "isFraud": 0,
            "isFlaggedFraud": 0,
        })

    return rows

# -------------------------------------------------------
# Fraud transaction simulation (same PaySim pattern)
# -------------------------------------------------------

def simulate_fraud_chain(
    victim: Account,
    mule: Account,
    step: int,
    rng: np.random.Generator,
) -> List[dict]:
    """
    Simulates one complete PaySim-style fraud chain:

    Step 1 — TRANSFER: Fraudster moves victim's money to mule account.
    Step 2 — CASH_OUT: Fraudster drains the mule account entirely.

    Both transactions are flagged isFraud = 1.
    Mule account balance is zeroed out (closed) after CASH_OUT.

    This is deliberately identical to PaySim's fraud mechanism so we can
    test whether the model learned the fraud BEHAVIOR or the simulator ARTIFACTS.
    """
    rows = []

    # Fraudster drains the victim account fully (PaySim behavior)
    transfer_amount = victim.balance
    if transfer_amount < 1.0:
        # Victim has no money — skip this chain
        return []

    # ---- Step 1: TRANSFER victim → mule ----
    old_bal_victim = victim.balance
    old_bal_mule   = mule.balance

    victim.balance = 0.0          # Fully drained
    mule.balance  += transfer_amount

    new_bal_victim = victim.balance
    new_bal_mule   = mule.balance

    rows.append({
        "step": step,
        "type": "TRANSFER",
        "amount": round(transfer_amount, 2),
        "nameOrig": victim.name,
        "oldbalanceOrg": round(old_bal_victim, 2),
        "newbalanceOrig": round(new_bal_victim, 2),
        "nameDest": mule.name,
        "oldbalanceDest": round(old_bal_mule, 2),
        "newbalanceDest": round(new_bal_mule, 2),
        "isFraud": 1,
        "isFlaggedFraud": 0,
    })

    # ---- Step 2: CASH_OUT mule → merchant (next step) ----
    cashout_step   = step + 1
    cashout_amount = mule.balance

    # Pick a random merchant as the CASH_OUT destination
    # (PaySim behavior: mule cashes out to external entity)
    merchant_id = f"M{rng.integers(5_000_000, 5_000_000 + 500)}"

    old_bal_mule_2 = mule.balance
    mule.balance   = 0.0  # Mule fully drained and closed

    rows.append({
        "step": cashout_step,
        "type": "CASH_OUT",
        "amount": round(cashout_amount, 2),
        "nameOrig": mule.name,
        "oldbalanceOrg": round(old_bal_mule_2, 2),
        "newbalanceOrig": round(mule.balance, 2),
        "nameDest": merchant_id,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "isFraud": 1,
        "isFlaggedFraud": 0,
    })

    return rows

# -------------------------------------------------------
# Main generation pipeline
# -------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("Synthetic Transaction Data Generator")
    print("=" * 60)
    print(f"\nSettings:")
    print(f"  Random seed     : {RANDOM_SEED}  (PaySim used a different seed)")
    print(f"  Steps           : {N_STEPS}       (PaySim: 743)")
    print(f"  Target size     : {TARGET_TRANSACTIONS:,} transactions")
    print(f"  Fraud chains    : {N_FRAUD_CHAINS} (~{TARGET_FRAUD_RATE*100:.2f}% fraud rate)")
    print(f"  Balance mean    : {BALANCE_MEAN:,}  (PaySim: ~833k)")
    print(f"  Amount mean     : {AMOUNT_MEAN:,}   (PaySim: ~179k)")

    rng = np.random.default_rng(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # --- Generate accounts ---
    print("\nGenerating accounts...")
    customer_ids  = generate_customer_ids(N_CUSTOMERS)
    merchant_ids  = generate_merchant_ids(N_MERCHANTS)

    customers = [Account(name=cid, balance=sample_balance(rng)) for cid in customer_ids]
    merchants = [Account(name=mid, balance=0.0) for mid in merchant_ids]

    print(f"  Customers : {len(customers):,}")
    print(f"  Merchants : {len(merchants):,}")

    # --- Normal transactions ---
    n_normal = TARGET_TRANSACTIONS - (N_FRAUD_CHAINS * 2)  # Each fraud chain = 2 rows
    print(f"\nSimulating {n_normal:,} normal transactions...")
    normal_rows = simulate_normal_transactions(customers, merchants, n_normal, rng)
    print(f"  Generated {len(normal_rows):,} normal transaction rows")

    # --- Fraud chains ---
    print(f"\nSimulating {N_FRAUD_CHAINS} fraud chains (PaySim pattern)...")

    # Select victim accounts that have meaningful balances
    eligible_victims = [c for c in customers if c.balance > 1000]
    if len(eligible_victims) < N_FRAUD_CHAINS:
        print(f"  WARNING: Only {len(eligible_victims)} eligible victims found. "
              f"Reducing fraud chains to match.")
        actual_fraud_chains = len(eligible_victims)
    else:
        actual_fraud_chains = N_FRAUD_CHAINS

    # Randomly select victims and create dedicated mule accounts
    rng_indices = rng.choice(len(eligible_victims), size=actual_fraud_chains, replace=False)
    fraud_victims = [eligible_victims[i] for i in rng_indices]

    # Mule accounts: fresh C-prefix accounts not in the customer pool
    mule_accounts = [
        Account(name=f"C{8_000_000 + i}", balance=0.0)
        for i in range(actual_fraud_chains)
    ]

    # Spread fraud events across the timeline
    fraud_steps = sorted(rng.integers(1, N_STEPS - 1, size=actual_fraud_chains).tolist())

    fraud_rows = []
    skipped = 0
    for victim, mule, step in zip(fraud_victims, mule_accounts, fraud_steps):
        chain = simulate_fraud_chain(victim, mule, int(step), rng)
        if chain:
            fraud_rows.extend(chain)
        else:
            skipped += 1

    actual_fraud_txns = len(fraud_rows)
    print(f"  Generated {actual_fraud_txns} fraud transaction rows "
          f"({skipped} chains skipped — victim had zero balance)")

    # --- Combine and sort by step ---
    print("\nCombining and sorting all transactions by step...")
    all_rows = normal_rows + fraud_rows
    df = pd.DataFrame(all_rows)
    df = df.sort_values("step").reset_index(drop=True)

    # --- Verify schema matches PaySim ---
    expected_cols = [
        "step", "type", "amount",
        "nameOrig", "oldbalanceOrg", "newbalanceOrig",
        "nameDest", "oldbalanceDest", "newbalanceDest",
        "isFraud", "isFlaggedFraud"
    ]
    assert list(df.columns) == expected_cols, \
        f"Schema mismatch! Got: {list(df.columns)}"

    # --- Summary statistics ---
    total      = len(df)
    n_fraud    = df["isFraud"].sum()
    fraud_rate = n_fraud / total * 100
    type_counts = df["type"].value_counts()

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"  Total rows      : {total:,}")
    print(f"  Fraud rows      : {n_fraud:,}  ({fraud_rate:.4f}%)")
    print(f"  Non-fraud rows  : {total - n_fraud:,}")
    print(f"\n  Transaction type breakdown:")
    for t, c in type_counts.items():
        print(f"    {t:<12} : {c:>8,}  ({c/total*100:.1f}%)")

    print(f"\n  Amount stats:")
    print(f"    Mean  : {df['amount'].mean():>12,.2f}")
    print(f"    Std   : {df['amount'].std():>12,.2f}")
    print(f"    Min   : {df['amount'].min():>12,.2f}")
    print(f"    Max   : {df['amount'].max():>12,.2f}")

    fraud_df = df[df["isFraud"] == 1]
    print(f"\n  Fraud-only transaction types:")
    print(f"    {fraud_df['type'].value_counts().to_dict()}")

    # --- Save dataset ---
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
    print(f"\nSaving dataset to: {out_path}")
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df):,} rows, {len(df.columns)} columns")

    # --- Save summary ---
    summary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), SUMMARY_FILE)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Synthetic Dataset Generator — Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write("PURPOSE\n")
        f.write("-------\n")
        f.write("Test whether the PaySim-trained model detects real fraud\n")
        f.write("behavior or PaySim simulator artifacts.\n\n")
        f.write("FRAUD PATTERN (identical to PaySim)\n")
        f.write("------------------------------------\n")
        f.write("1. Fraudster takes over a victim C-account\n")
        f.write("2. Full TRANSFER of victim balance to mule C-account\n")
        f.write("3. CASH_OUT of mule account to external merchant\n")
        f.write("4. Mule balance reduced to 0 (account closed)\n\n")
        f.write("WHAT IS DIFFERENT FROM PAYSIM\n")
        f.write("------------------------------\n")
        f.write(f"- Random seed        : {RANDOM_SEED} (PaySim: different)\n")
        f.write(f"- Steps              : {N_STEPS} (PaySim: 743)\n")
        f.write(f"- Balance mean       : {BALANCE_MEAN:,} (PaySim: ~833k)\n")
        f.write(f"- Amount mean        : {AMOUNT_MEAN:,} (PaySim: ~179k)\n")
        f.write(f"- Account ID range   : C9000000+ (PaySim: lower range)\n")
        f.write(f"- Total transactions : {total:,} (PaySim: 6,362,620)\n\n")
        f.write("GENERATION SETTINGS\n")
        f.write("-------------------\n")
        f.write(f"N_CUSTOMERS   : {N_CUSTOMERS:,}\n")
        f.write(f"N_MERCHANTS   : {N_MERCHANTS:,}\n")
        f.write(f"N_FRAUD_CHAINS: {actual_fraud_chains}\n")
        f.write(f"FRAUD_RATE    : {fraud_rate:.4f}%\n\n")
        f.write("HOW TO USE\n")
        f.write("----------\n")
        f.write("1. Run this script to generate synthetic_transaction_data.csv\n")
        f.write("2. Rename or symlink it as transaction_data.csv\n")
        f.write("3. Run panel_temporal_xgb.py (trained on PaySim) on this data\n")
        f.write("4. Compare AUPRC with PaySim test results\n\n")
        f.write("INTERPRETATION\n")
        f.write("--------------\n")
        f.write("High AUPRC on this dataset => model learned generalizable fraud\n")
        f.write("Low AUPRC on this dataset  => model overfit to PaySim artifacts\n")

    print(f"  Summary saved to: {summary_path}")
    print("\nDone. Next step: run panel_temporal_xgb.py using this dataset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
