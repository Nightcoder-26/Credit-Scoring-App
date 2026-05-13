"""
create_dataset.py
-----------------
Run this ONCE to add the 'credit_risk' target column to the dataset.
The dataset you provided is the Kaggle German Credit dataset - it has all
features but is missing the Risk/target column.

This script generates a realistic target using domain-knowledge rules
that match the original German Credit dataset's 70% good / 30% bad split.

Usage:
    python data/create_dataset.py
"""

import pandas as pd
import numpy as np
import os

# -- Load the raw CSV (no target) ------------------------------
raw_path = os.path.join(os.path.dirname(__file__), "german_credit_raw.csv")
out_path = os.path.join(os.path.dirname(__file__), "german_credit.csv")

if not os.path.exists(raw_path):
    print(f"ERROR: {raw_path} not found.")
    print("Please rename your uploaded CSV to: data/german_credit_raw.csv")
    exit(1)

df = pd.read_csv(raw_path, index_col=0)
print(f"Loaded {len(df)} rows | Columns: {list(df.columns)}")

# -- Generate credit_risk target using domain heuristics -------
np.random.seed(42)

def compute_risk_score(row):
    """
    Computes a default-risk score (higher = more risky).
    Based on known German Credit domain patterns.
    """
    score = 0.0

    # Checking account: no account / little = higher risk
    ca = str(row.get("Checking account", "NA")).lower()
    if ca in ["na", "nan", "none", ""]:
        score += 2.0
    elif ca == "little":
        score += 1.5
    elif ca == "moderate":
        score += 0.5
    elif ca in ["rich", "quite rich"]:
        score -= 0.5

    # Saving accounts: little / none = higher risk
    sa = str(row.get("Saving accounts", "NA")).lower()
    if sa in ["na", "nan", "none", ""]:
        score += 1.5
    elif sa == "little":
        score += 1.0
    elif sa == "moderate":
        score += 0.0
    elif sa in ["rich", "quite rich"]:
        score -= 1.0

    # Credit amount: higher = riskier
    amount = row.get("Credit amount", 2500)
    score += (amount - 2500) / 5000

    # Duration: longer = riskier
    duration = row.get("Duration", 18)
    score += (duration - 18) / 24

    # Age: younger = riskier
    age = row.get("Age", 35)
    score -= (age - 25) / 40

    # Housing
    housing = str(row.get("Housing", "own")).lower()
    if housing == "rent":
        score += 0.5
    elif housing == "free":
        score += 0.2

    # Job: 0 (unskilled non-resident) = higher risk
    job = row.get("Job", 2)
    score += (2 - job) * 0.3

    return score

# Compute risk scores
df["_risk_score"] = df.apply(compute_risk_score, axis=1)

# Convert to binary target with ~70% good (1) / 30% bad (0)
threshold = np.percentile(df["_risk_score"], 70)
df["credit_risk"] = (df["_risk_score"] <= threshold).astype(int)
df = df.drop(columns=["_risk_score"])

# -- Report ----------------------------------------------------
counts = df["credit_risk"].value_counts()
print(f"\nTarget distribution:")
print(f"  Creditworthy (1): {counts.get(1,0)} ({counts.get(1,0)/len(df)*100:.0f}%)")
print(f"  Default      (0): {counts.get(0,0)} ({counts.get(0,0)/len(df)*100:.0f}%)")

# -- Save ------------------------------------------------------
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print("You can now run: python train_model.py")
