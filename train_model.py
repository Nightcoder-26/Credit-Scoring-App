"""
train_model.py
==============
Standalone training pipeline for the Credit Scoring Model.

Run this script FIRST to train models and generate credit_model.pkl:
    python train_model.py
"""

import sys
import os
import warnings
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from utils.data_loader import load_and_clean_data, engineer_features, get_train_test_split
from utils.model_trainer import (
    train_all_models, get_best_model, tune_best_model, save_model_bundle
)


def print_banner():
    print("\n" + "=" * 65)
    print("      CREDIT SCORING MODEL - TRAINING PIPELINE")
    print("   CodeAlpha Internship Project")
    print("=" * 65 + "\n")


def print_section(title: str):
    print(f"\n{'-'*55}")
    print(f"  {title}")
    print(f"{'-'*55}")


def main():
    print_banner()

    # Step 1: Load & Clean Data
    print_section("STEP 1: Loading & Cleaning Data")
    df_raw = load_and_clean_data("data/german.data")
    print(f"[OK] Dataset shape: {df_raw.shape}")
    print(f"[OK] Target distribution:\n{df_raw['credit_risk'].value_counts().to_string()}")
    print(f"\n[SAMPLE] Sample rows:")
    print(df_raw.head(3).to_string())

    # Save cleaned data
    os.makedirs("data", exist_ok=True)
    df_raw.to_csv("data/german_credit_cleaned.csv", index=False)
    print("\n[SAVED] Cleaned data saved -> data/german_credit_cleaned.csv")

    # Step 2: Feature Engineering
    print_section("STEP 2: Feature Engineering")
    df_engineered, label_encoders = engineer_features(df_raw)
    print(f"[OK] Features after engineering: {df_engineered.shape[1]}")
    print(f"[OK] New derived features:")
    new_feats = ["debt_duration_ratio", "installment_burden", "age_group", "credit_tier"]
    for f in new_feats:
        if f in df_engineered.columns:
            print(f"   * {f}")

    # Step 3: Train/Test Split
    print_section("STEP 3: Train/Test Split (80/20, stratified)")
    X_train, X_test, y_train, y_test, feature_names, scaler = get_train_test_split(df_engineered)
    print(f"[OK] Training samples:  {X_train.shape[0]}")
    print(f"[OK] Test samples:      {X_test.shape[0]}")
    print(f"[OK] Total features:    {len(feature_names)}")
    print(f"   Features: {', '.join(feature_names[:8])}...")

    # Step 4: Train All Models
    print_section("STEP 4: Training All Models")
    trained_models, results = train_all_models(X_train, y_train, X_test, y_test)

    # Step 5: Model Comparison
    print_section("STEP 5: Model Performance Comparison")
    metrics_df = pd.DataFrame({
        name: {
            "Accuracy":  f"{res['accuracy']:.4f}",
            "Precision": f"{res['precision']:.4f}",
            "Recall":    f"{res['recall']:.4f}",
            "F1 Score":  f"{res['f1_score']:.4f}",
            "ROC-AUC":   f"{res['roc_auc']:.4f}",
        }
        for name, res in results.items()
    }).T

    print(metrics_df.to_string())
    metrics_df.to_csv("data/model_comparison.csv")
    print("\n[SAVED] Comparison table saved -> data/model_comparison.csv")

    # Step 6: Select & Tune Best Model
    print_section("STEP 6: Hyperparameter Tuning")
    best_name = get_best_model(results)
    print(f"\n[BEST] Best model by ROC-AUC: {best_name}")
    print(f"   Base AUC: {results[best_name]['roc_auc']:.4f}")

    tuned_model, tuned_metrics = tune_best_model(
        best_name, X_train, y_train, X_test, y_test
    )
    print(f"\n[OK] Tuned Model Results:")
    print(f"   Accuracy:  {tuned_metrics['accuracy']:.4f}")
    print(f"   Precision: {tuned_metrics['precision']:.4f}")
    print(f"   Recall:    {tuned_metrics['recall']:.4f}")
    print(f"   F1 Score:  {tuned_metrics['f1_score']:.4f}")
    print(f"   ROC-AUC:   {tuned_metrics['roc_auc']:.4f}")
    print(f"\n{tuned_metrics['classification_report']}")

    # Compare with base model
    base_auc  = results[best_name]['roc_auc']
    tuned_auc = tuned_metrics['roc_auc']
    improvement = (tuned_auc - base_auc) * 100
    imp_str = "[UP]" if improvement > 0 else "[DOWN]"
    print(f"{imp_str} AUC improvement after tuning: {improvement:+.2f}%")

    # Choose best between base and tuned
    final_model = tuned_model if tuned_auc >= base_auc else trained_models[best_name]
    final_name  = best_name + " (Tuned)" if tuned_auc >= base_auc else best_name

    # Step 7: Save Model Bundle
    print_section("STEP 7: Saving Model")
    save_model_bundle(
        model=final_model,
        scaler=scaler,
        feature_names=feature_names,
        label_encoders=label_encoders,
        model_name=final_name,
        save_dir="models",
    )

    # Also save metadata
    import json
    metadata = {
        "best_model": final_name,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "train_samples": int(X_train.shape[0]),
        "test_samples":  int(X_test.shape[0]),
        "final_metrics": {
            "accuracy":  round(float(tuned_metrics["accuracy"]), 4),
            "precision": round(float(tuned_metrics["precision"]), 4),
            "recall":    round(float(tuned_metrics["recall"]), 4),
            "f1_score":  round(float(tuned_metrics["f1_score"]), 4),
            "roc_auc":   round(float(tuned_metrics["roc_auc"]), 4),
        }
    }
    with open("models/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n[SAVED] Metadata saved -> models/model_metadata.json")

    # Done
    print("\n" + "=" * 65)
    print("  [OK] TRAINING COMPLETE!")
    print(f"  [BEST] Final Model: {final_name}")
    print(f"  [TARGET] Final ROC-AUC: {max(base_auc, tuned_auc):.4f}")
    print("  [SAVED] Model saved: credit_model.pkl")
    print("  [RUN] Command: streamlit run app.py")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
