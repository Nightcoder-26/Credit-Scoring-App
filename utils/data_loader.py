"""
data_loader.py
--------------
Handles loading, cleaning, and preprocessing of credit datasets.

SUPPORTED DATASET FORMATS
--------------------------
Option A (Automatic): UCI German Credit Dataset is auto-downloaded
    saved at: data/german.data

Option B (Manual CSV): Drop your own CSV file at:
    data/german_credit.csv
  Required columns (any subset will work, missing ones are imputed):
    age, duration, credit_amount, installment_rate, residence_since,
    existing_credits, num_dependents, credit_risk (target: 0 or 1)

The loader auto-detects which file is present and uses it.
"""

import pandas as pd
import numpy as np
import os
import requests
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# ---------------------------------------------
# Column definitions for German Credit Dataset
# ---------------------------------------------
GERMAN_COLUMNS = [
    "checking_account", "duration", "credit_history", "purpose",
    "credit_amount", "savings_account", "employment", "installment_rate",
    "personal_status", "other_debtors", "residence_since", "property",
    "age", "other_installments", "housing", "existing_credits",
    "job", "num_dependents", "telephone", "foreign_worker", "credit_risk",
]

NUMERIC_COLUMNS = [
    "duration", "credit_amount", "installment_rate",
    "residence_since", "age", "existing_credits", "num_dependents",
]

CATEGORICAL_COLUMNS = [
    "checking_account", "credit_history", "purpose", "savings_account",
    "employment", "personal_status", "other_debtors", "property",
    "other_installments", "housing", "job", "telephone", "foreign_worker",
]

# Human-readable label maps for UCI coded values
CHECKING_ACCOUNT_MAP = {"A11": "< 0 DM", "A12": "0-200 DM", "A13": "> 200 DM", "A14": "No Account"}
CREDIT_HISTORY_MAP   = {"A30": "No credits taken", "A31": "All credits paid duly",
                         "A32": "Existing credits paid duly", "A33": "Delay in past", "A34": "Critical account"}
PURPOSE_MAP          = {"A40": "Car (new)", "A41": "Car (used)", "A42": "Furniture/equipment",
                         "A43": "Radio/TV", "A44": "Domestic appliances", "A45": "Repairs",
                         "A46": "Education", "A47": "Vacation", "A48": "Retraining",
                         "A49": "Business", "A410": "Others"}
SAVINGS_MAP          = {"A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM",
                         "A64": "> 1000 DM", "A65": "Unknown/No savings"}
EMPLOYMENT_MAP       = {"A71": "Unemployed", "A72": "< 1 year", "A73": "1-4 years",
                         "A74": "4-7 years", "A75": "> 7 years"}
HOUSING_MAP          = {"A151": "Rent", "A152": "Own", "A153": "Free"}
JOB_MAP              = {"A171": "Unskilled non-resident", "A172": "Unskilled resident",
                         "A173": "Skilled", "A174": "Highly skilled"}


# ---------------------------------------------
# Dataset Loading
# ---------------------------------------------

def load_and_clean_data(data_path: str = "data/german.data") -> pd.DataFrame:
    """
    Main entry point. Auto-detects dataset format:
      1. If any CSV exists in data/ folder      load it directly
      2. If data/german.data exists             load UCI space-separated file
      3. Otherwise                              download from UCI ML Repository
    Returns a cleaned DataFrame ready for EDA.
    """
    os.makedirs("data", exist_ok=True)

    # -- Look for any user-supplied CSV ------------------------
    csv_candidates = [
        "data/german_credit.csv",
        "data/german_credit_data.csv",
        "data/german.csv",
        "data/credit.csv",
    ]
    for csv_path in csv_candidates:
        if os.path.exists(csv_path):
            print(f"[FOLDER] Loading dataset: {csv_path}")
            return _load_csv(csv_path)

    # -- Fall back to UCI .data file ---------------------------
    uci_path = "data/german.data"
    if not os.path.exists(uci_path):
        _download_uci(uci_path)

    if os.path.exists(uci_path):
        df = _load_uci(uci_path)
    else:
        print("[WARN]  Download failed. Generating realistic sample data as fallback.")
        df = _generate_sample_data()

    return _clean(df)


def _download_uci(save_path: str):
    """Downloads the UCI German Credit dataset."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    print("[DOWNLOADING]  Downloading German Credit Dataset from UCI ML Repository...")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(r.content)
        print(f"[OK] Dataset saved   {save_path}")
    except Exception as e:
        print(f"[WARN]  Download failed: {e}")


def _load_uci(path: str) -> pd.DataFrame:
    """Loads the raw UCI space-separated .data file."""
    df = pd.read_csv(path, sep=" ", header=None, names=GERMAN_COLUMNS)
    # Convert target: 1=Good   1 (creditworthy), 2=Bad   0 (default)
    df["credit_risk"] = df["credit_risk"].map({1: 1, 2: 0})
    # Apply human-readable labels
    df["checking_account"] = df["checking_account"].map(CHECKING_ACCOUNT_MAP).fillna(df["checking_account"])
    df["credit_history"]   = df["credit_history"].map(CREDIT_HISTORY_MAP).fillna(df["credit_history"])
    df["purpose"]          = df["purpose"].map(PURPOSE_MAP).fillna(df["purpose"])
    df["savings_account"]  = df["savings_account"].map(SAVINGS_MAP).fillna(df["savings_account"])
    df["employment"]       = df["employment"].map(EMPLOYMENT_MAP).fillna(df["employment"])
    df["housing"]          = df["housing"].map(HOUSING_MAP).fillna(df["housing"])
    df["job"]              = df["job"].map(JOB_MAP).fillna(df["job"])
    return df


def _load_csv(path: str) -> pd.DataFrame:  # noqa: C901
    """
    Loads a user-supplied CSV - supports both the standard format and
    the Kaggle German Credit format (Age, Sex, Job, Housing, Saving accounts,
    Checking account, Credit amount, Duration, Purpose, Risk).

    If no target column is found, auto-generates 'credit_risk' using
    domain-knowledge heuristics (~70% creditworthy / 30% default).
    """
    df = pd.read_csv(path)

    # -- Normalise column names --------------------------------
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Drop unnamed index column if present
    df = df.loc[:, ~df.columns.str.startswith("unnamed")]

    # -- Map Kaggle column names   standard names --------------
    kaggle_map = {
        "age":              "age",
        "sex":              "sex",
        "job":              "job",
        "housing":          "housing",
        "saving_accounts":  "savings_account",
        "checking_account": "checking_account",
        "credit_amount":    "credit_amount",
        "duration":         "duration",
        "purpose":          "purpose",
        "risk":             "credit_risk",   # Kaggle target column
    }
    df = df.rename(columns={k: v for k, v in kaggle_map.items() if k in df.columns})

    # -- Resolve target column ---------------------------------
    if "credit_risk" not in df.columns:
        # Try other common names
        for alt in ["target", "label", "class", "default", "creditworthy", "good_bad"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "credit_risk"})
                break
        else:
            # No target found   auto-generate from domain heuristics
            print("[WARN]  No target column found. Auto-generating 'credit_risk' using domain heuristics.")
            df["credit_risk"] = _generate_target(df)

    # -- Normalise target values -------------------------------
    if df["credit_risk"].dtype == object:
        good_vals = {"good", "1", "yes", "creditworthy", "low"}
        df["credit_risk"] = df["credit_risk"].str.strip().str.lower().apply(
            lambda x: 1 if x in good_vals else 0
        )
    else:
        unique_vals = set(df["credit_risk"].dropna().unique())
        if unique_vals <= {1, 2}:          # UCI 1/2 convention
            df["credit_risk"] = df["credit_risk"].map({1: 1, 2: 0})

    dist = df["credit_risk"].value_counts(normalize=True) * 100
    print(f"[OK] Target distribution - Creditworthy: {dist.get(1,0):.0f}%  |  Default: {dist.get(0,0):.0f}%")
    return _clean(df)


def _generate_target(df: pd.DataFrame) -> pd.Series:
    """
    Generates a realistic credit_risk target (0/1) from Kaggle-format features.
    Uses domain knowledge to achieve ~70% creditworthy / 30% default split.
    """
    np.random.seed(42)
    score = np.zeros(len(df))

    # Checking account: NA/little = riskier
    if "checking_account" in df.columns:
        ca = df["checking_account"].fillna("NA").str.lower()
        score += ca.map({"na": 2.0, "little": 1.5, "moderate": 0.5,
                         "quite rich": -0.5, "rich": -1.0}).fillna(0)

    # Savings account: NA/little = riskier
    if "savings_account" in df.columns:
        sa = df["savings_account"].fillna("NA").str.lower()
        score += sa.map({"na": 1.5, "little": 1.0, "moderate": 0.0,
                         "quite rich": -0.5, "rich": -1.0}).fillna(0)

    # Credit amount: higher = riskier
    if "credit_amount" in df.columns:
        score += (df["credit_amount"].fillna(2500) - 2500) / 5000

    # Duration: longer = riskier
    if "duration" in df.columns:
        score += (df["duration"].fillna(18) - 18) / 24

    # Age: younger = riskier
    if "age" in df.columns:
        score -= (df["age"].fillna(35) - 25) / 40

    # Housing: rent = riskier
    if "housing" in df.columns:
        h = df["housing"].fillna("own").str.lower()
        score += h.map({"rent": 0.5, "free": 0.2, "own": 0.0}).fillna(0)

    # Job level: 0 = unskilled non-resident = riskier
    if "job" in df.columns:
        score += (2 - df["job"].fillna(2)) * 0.3

    # Threshold at 70th percentile   70% good, 30% bad
    threshold = np.percentile(score, 70)
    return (score <= threshold).astype(int)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans a DataFrame: drops duplicates, imputes missing values."""
    original = len(df)
    df = df.drop_duplicates()
    if len(df) < original:
        print(f"[INFO]  Removed {original - len(df)} duplicate rows.")

    # Impute numeric columns with median
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Impute categorical columns with mode
    for col in df.select_dtypes(include="object").columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def _generate_sample_data(n: int = 1000) -> pd.DataFrame:
    """Fallback: generates realistic synthetic data if real dataset unavailable."""
    np.random.seed(42)
    ages     = np.random.randint(19, 75, n)
    amounts  = np.random.randint(250, 18000, n)
    duration = np.random.randint(4, 72, n)

    prob_good = np.clip(0.3 + 0.005 * (ages - 30) - 0.000008 * amounts, 0.15, 0.92)
    risk      = np.where(np.random.random(n) < prob_good, 1, 2)

    return pd.DataFrame({
        "checking_account":   np.random.choice(list(CHECKING_ACCOUNT_MAP.values()), n),
        "duration":           duration,
        "credit_history":     np.random.choice(list(CREDIT_HISTORY_MAP.values()), n),
        "purpose":            np.random.choice(["Car (new)", "Car (used)", "Furniture/equipment",
                                                "Radio/TV", "Education", "Business"], n),
        "credit_amount":      amounts,
        "savings_account":    np.random.choice(list(SAVINGS_MAP.values()), n),
        "employment":         np.random.choice(list(EMPLOYMENT_MAP.values()), n),
        "installment_rate":   np.random.randint(1, 5, n),
        "personal_status":    np.random.choice(["Male single", "Female", "Male married"], n),
        "other_debtors":      np.random.choice(["None", "Co-applicant", "Guarantor"], n),
        "residence_since":    np.random.randint(1, 5, n),
        "property":           np.random.choice(["Real estate", "Savings/insurance", "Car/other", "Unknown"], n),
        "age":                ages,
        "other_installments": np.random.choice(["None", "Bank", "Stores"], n),
        "housing":            np.random.choice(list(HOUSING_MAP.values()), n),
        "existing_credits":   np.random.randint(1, 5, n),
        "job":                np.random.choice(list(JOB_MAP.values()), n),
        "num_dependents":     np.random.randint(1, 3, n),
        "telephone":          np.random.choice(["Yes", "No"], n),
        "foreign_worker":     np.random.choice(["Yes", "No"], n),
        "credit_risk":        np.where(risk == 1, 1, 0),
    })


# ---------------------------------------------
# Feature Engineering
# ---------------------------------------------

def engineer_features(df: pd.DataFrame):
    """
    Creates derived financial features and encodes categoricals.
    Returns (engineered_df, label_encoders_dict).
    """
    df = df.copy()

    # -- Derived features --------------------------------------
    df["debt_duration_ratio"] = df["credit_amount"] / (df["duration"] + 1)
    
    if "installment_rate" in df.columns:
        df["installment_burden"]  = df["installment_rate"] * df["credit_amount"] / 1000

    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 25, 35, 50, 65, 100],
            labels=["Very Young", "Young Adult", "Middle-aged", "Senior", "Elderly"]
        ).astype(str)

    if "credit_amount" in df.columns:
        df["credit_tier"] = pd.cut(
            df["credit_amount"],
            bins=[0, 1000, 5000, 10000, 20000],
            labels=["Low", "Medium", "High", "Very High"]
        ).astype(str)

    # -- Encode categoricals -----------------------------------
    le = LabelEncoder()
    label_encoders = {}
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    return df, label_encoders


def get_train_test_split(df: pd.DataFrame, target: str = "credit_risk", test_size: float = 0.2):
    """
    Splits data 80/20 (stratified), scales features.
    Returns: X_train, X_test, y_train, y_test, feature_names, scaler
    """
    feature_cols = [c for c in df.columns if c != target]
    X, y = df[feature_cols], df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    return X_train_s, X_test_s, y_train.values, y_test.values, list(X.columns), scaler
