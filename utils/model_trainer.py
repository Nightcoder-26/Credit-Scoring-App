"""
model_trainer.py
----------------
Handles training, evaluation, hyperparameter tuning,
and persistence of multiple classification models.
"""

import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, Any, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARN]  XGBoost not installed. Skipping XGBoost model.")


# --------------------------------------------------------------
# Model Definitions
# --------------------------------------------------------------

def get_models() -> Dict[str, Any]:
    """Returns a dictionary of all models to train."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42,
            class_weight="balanced", n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
    }
    
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100, random_state=42,
            eval_metric="logloss", use_label_encoder=False
        )
    
    return models


# --------------------------------------------------------------
# Training & Evaluation
# --------------------------------------------------------------

def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[Dict, Dict]:
    """
    Trains all models and evaluates them on the test set.
    Returns:
        trained_models: dict of model_name -> fitted model
        results: dict of model_name -> metrics dict
    """
    models = get_models()
    trained_models = {}
    results = {}
    
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        results[name] = evaluate_model(model, X_test, y_test)
        print(f"   [OK] {name} - Accuracy: {results[name]['accuracy']:.4f} | AUC: {results[name]['roc_auc']:.4f}")
    
    return trained_models, results


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Computes a full suite of classification metrics for a given model.
    Returns a dictionary with all key metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    metrics = {
        "accuracy":        accuracy_score(y_test, y_pred),
        "precision":       precision_score(y_test, y_pred, zero_division=0),
        "recall":          recall_score(y_test, y_pred, zero_division=0),
        "f1_score":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":         roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "y_pred":          y_pred,
        "y_prob":          y_prob,
    }
    return metrics


def get_best_model(results: Dict) -> str:
    """Returns the name of the best model based on ROC-AUC score."""
    return max(results, key=lambda k: results[k]["roc_auc"])


# --------------------------------------------------------------
# Hyperparameter Tuning
# --------------------------------------------------------------

def tune_best_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[Any, Dict]:
    """
    Runs GridSearchCV on the best-performing model to optimize hyperparameters.
    Returns the best tuned model and its metrics.
    """
    param_grids = {
        "Random Forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "max_features": ["sqrt", "log2"],
        },
        "Gradient Boosting": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "subsample": [0.8, 1.0],
        },
        "Decision Tree": {
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 5, 10],
            "criterion": ["gini", "entropy"],
        },
        "Logistic Regression": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs", "liblinear"],
        },
    }
    
    base_models = get_models()
    
    if model_name not in param_grids:
        print(f"[WARN]  No param grid for {model_name}. Using default params.")
        return base_models.get(model_name), {}
    
    print(f"[TUNE] Tuning {model_name} with GridSearchCV (cv=5)...")
    grid_search = GridSearchCV(
        estimator=base_models[model_name],
        param_grid=param_grids[model_name],
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"   [OK] Best Params: {best_params}")
    print(f"   [OK] Best CV AUC: {best_score:.4f}")
    
    metrics = evaluate_model(best_model, X_test, y_test)
    return best_model, metrics


# --------------------------------------------------------------
# Model Persistence
# --------------------------------------------------------------

def save_model_bundle(
    model,
    scaler,
    feature_names: list,
    label_encoders: dict,
    model_name: str,
    save_dir: str = "models"
) -> str:
    """
    Saves the model, scaler, feature names, and encoders as a bundle.
    Returns the path to the saved bundle.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "label_encoders": label_encoders,
        "model_name": model_name,
    }
    
    safe_name = model_name.lower().replace(" ", "_")
    bundle_path = os.path.join(save_dir, f"{safe_name}_bundle.pkl")
    joblib.dump(bundle, bundle_path)
    
    # Also save the best model separately as credit_model.pkl (for easy access)
    joblib.dump(bundle, "credit_model.pkl")
    print(f"[SAVED] Model bundle saved: {bundle_path}")
    print(f"[SAVED] Also saved as: credit_model.pkl")
    
    return bundle_path


def load_model_bundle(bundle_path: str = "credit_model.pkl") -> dict:
    """Loads and returns the saved model bundle."""
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(
            f"Model file '{bundle_path}' not found. "
            "Please run train_model.py first."
        )
    return joblib.load(bundle_path)


# --------------------------------------------------------------
# Prediction Interface
# --------------------------------------------------------------

def predict_credit_risk(bundle: dict, input_data: dict) -> dict:
    """
    Given a loaded model bundle and a dict of user inputs,
    returns a prediction with risk score and confidence.
    
    input_data keys must match feature_names in the bundle.
    """
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]
    
    # Build input vector in correct feature order
    row = [input_data.get(feat, 0) for feat in feature_names]
    X = np.array(row).reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    
    prob_good = float(probabilities[1])
    prob_bad  = float(probabilities[0])
    
    return {
        "prediction": int(prediction),
        "label": "[OK] Creditworthy" if prediction == 1 else "  Not Creditworthy",
        "confidence": max(prob_good, prob_bad) * 100,
        "prob_creditworthy": prob_good * 100,
        "prob_default": prob_bad * 100,
        "risk_level": (
            "Low Risk" if prob_bad < 0.25 else
            "Moderate Risk" if prob_bad < 0.5 else
            "High Risk" if prob_bad < 0.75 else
            "Very High Risk"
        ),
    }
