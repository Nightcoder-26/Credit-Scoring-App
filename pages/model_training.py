"""
pages/model_training.py
------------------------
Model training, evaluation, comparison, and hyperparameter tuning page.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.data_loader import load_and_clean_data, engineer_features, get_train_test_split
from utils.model_trainer import (
    train_all_models, evaluate_model, get_best_model,
    tune_best_model, save_model_bundle
)
from utils.visualizations import (
    plot_model_comparison, plot_confusion_matrix,
    plot_roc_curves, plot_feature_importance, plot_metrics_radar
)


@st.cache_data(show_spinner="  Preprocessing data...")
def get_processed_data():
    df_raw = load_and_clean_data("data/german.data")
    df_eng, encoders = engineer_features(df_raw)
    X_tr, X_te, y_tr, y_te, feats, scaler = get_train_test_split(df_eng)
    return X_tr, X_te, y_tr, y_te, feats, scaler, df_raw, encoders


def render():
    st.markdown("## [AI] Model Training & Evaluation")
    st.markdown("Train multiple ML classifiers, compare performance, and fine-tune the best model.")

    # -- Load data ---------------------------------------------
    X_train, X_test, y_train, y_test, feature_names, scaler, df_raw, encoders = get_processed_data()

    st.info(
        f"  Dataset ready: **{X_train.shape[0]}** training | **{X_test.shape[0]}** test samples | "
        f"**{len(feature_names)}** features"
    )

    # -- Train Models ------------------------------------------
    st.markdown("###   Train Models")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        Click **Train All Models** to fit:
        - Logistic Regression
        - Decision Tree
        - Random Forest
        - Gradient Boosting
        - XGBoost *(if installed)*
        """)
    with col2:
        train_btn = st.button("[RUN] Train All Models", type="primary", use_container_width=True)

    if train_btn or "trained_models" in st.session_state:
        if train_btn:
            with st.spinner("  Training models... this may take ~30 seconds"):
                progress_bar = st.progress(0)
                trained_models, results = train_all_models(X_train, y_train, X_test, y_test)
                progress_bar.progress(100)
            
            st.session_state["trained_models"] = trained_models
            st.session_state["results"] = results
            st.session_state["feature_names"] = feature_names
            st.session_state["y_test"] = y_test
            st.session_state["X_train"] = X_train
            st.session_state["y_train"] = y_train
            st.session_state["X_test"] = X_test
            st.session_state["scaler"] = scaler
            st.session_state["encoders"] = encoders
            st.success("[OK] All models trained successfully!")
        
        trained_models = st.session_state["trained_models"]
        results        = st.session_state["results"]
        feature_names  = st.session_state["feature_names"]
        y_test         = st.session_state["y_test"]

        # -- Metrics Table -------------------------------------
        st.markdown("### [DATA] Performance Metrics")
        
        metrics_data = []
        best_name = get_best_model(results)
        
        for name, res in results.items():
            row = {
                "Model":     name,
                "Accuracy":  f"{res['accuracy']:.4f}",
                "Precision": f"{res['precision']:.4f}",
                "Recall":    f"{res['recall']:.4f}",
                "F1 Score":  f"{res['f1_score']:.4f}",
                "ROC-AUC":   f"{res['roc_auc']:.4f}",
                "Best":      "[BEST]" if name == best_name else "",
            }
            metrics_data.append(row)
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # Download table
        csv = metrics_df.to_csv(index=False).encode("utf-8")
        st.download_button("[DOWNLOADING] Download Metrics CSV", data=csv,
                           file_name="model_metrics.csv", mime="text/csv")

        # -- Charts ---------------------------------------------
        st.markdown("### [UP] Visual Comparison")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Bar Chart", "Radar Chart", "ROC Curves", "Confusion Matrices"])
        
        with tab1:
            fig = plot_model_comparison(results)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = plot_metrics_radar(results)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            fig = plot_roc_curves(results, y_test)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            cols = st.columns(min(3, len(results)))
            for i, (name, res) in enumerate(results.items()):
                with cols[i % 3]:
                    fig = plot_confusion_matrix(res["confusion_matrix"], name)
                    st.plotly_chart(fig, use_container_width=True)

        # -- Feature Importance --------------------------------
        st.markdown("###   Feature Importance")
        
        models_with_importance = {
            k: v for k, v in trained_models.items()
            if hasattr(v, "feature_importances_") or hasattr(v, "coef_")
        }
        
        if models_with_importance:
            selected_model_fi = st.selectbox(
                "Select model for feature importance",
                list(models_with_importance.keys()),
                key="fi_model"
            )
            fig = plot_feature_importance(
                models_with_importance[selected_model_fi],
                feature_names,
                selected_model_fi
            )
            st.plotly_chart(fig, use_container_width=True)

        # -- Hyperparameter Tuning -----------------------------
        st.markdown("---")
        st.markdown("###   Hyperparameter Tuning")
        
        best_name_display = get_best_model(results)
        st.markdown(f"[BEST] **Best base model:** `{best_name_display}` (ROC-AUC: {results[best_name_display]['roc_auc']:.4f})")
        
        tune_col1, tune_col2 = st.columns([3, 1])
        with tune_col1:
            tune_model_choice = st.selectbox(
                "Model to tune",
                list(results.keys()),
                index=list(results.keys()).index(best_name_display),
                key="tune_choice"
            )
        with tune_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            tune_btn = st.button("[TUNE] Run GridSearchCV", type="primary", use_container_width=True)
        
        if tune_btn:
            with st.spinner(f"[TUNE] Tuning {tune_model_choice} with 5-fold CV... (this takes 1 3 minutes)"):
                tuned_model, tuned_metrics = tune_best_model(
                    tune_model_choice,
                    st.session_state["X_train"],
                    st.session_state["y_train"],
                    st.session_state["X_test"],
                    st.session_state["y_test"],
                )
            
            st.session_state["tuned_model"] = tuned_model
            st.session_state["tuned_metrics"] = tuned_metrics
            st.session_state["tuned_name"] = tune_model_choice
            st.success(f"[OK] Tuning complete!")
        
        if "tuned_model" in st.session_state:
            tm = st.session_state["tuned_metrics"]
            base_auc  = results[st.session_state["tuned_name"]]["roc_auc"]
            tuned_auc = tm["roc_auc"]
            delta     = tuned_auc - base_auc
            
            st.markdown("#### Tuned Model Results")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy",  f"{tm['accuracy']:.4f}")
            c2.metric("Precision", f"{tm['precision']:.4f}")
            c3.metric("Recall",    f"{tm['recall']:.4f}")
            c4.metric("F1 Score",  f"{tm['f1_score']:.4f}")
            c5.metric("ROC-AUC",   f"{tuned_auc:.4f}", delta=f"{delta:+.4f}")
            
            # Show confusion matrix for tuned model
            col_a, col_b = st.columns(2)
            with col_a:
                fig = plot_confusion_matrix(tm["confusion_matrix"], f"{st.session_state['tuned_name']} (Tuned)")
                st.plotly_chart(fig, use_container_width=True)
            with col_b:
                fig = plot_feature_importance(
                    st.session_state["tuned_model"],
                    st.session_state["feature_names"],
                    f"{st.session_state['tuned_name']} (Tuned)"
                )
                st.plotly_chart(fig, use_container_width=True)

        # -- Save Model ----------------------------------------
        st.markdown("---")
        st.markdown("### [SAVED] Save Best Model")
        
        save_col1, save_col2 = st.columns([3, 1])
        with save_col1:
            st.markdown("Save the trained model bundle (model + scaler + feature names) to `credit_model.pkl`.")
        with save_col2:
            save_btn = st.button("[SAVED] Save Model", type="primary", use_container_width=True)
        
        if save_btn:
            if "tuned_model" in st.session_state:
                final_model = st.session_state["tuned_model"]
                final_name  = st.session_state["tuned_name"] + " (Tuned)"
            else:
                final_name  = get_best_model(results)
                final_model = trained_models[final_name]
            
            path = save_model_bundle(
                model=final_model,
                scaler=st.session_state["scaler"],
                feature_names=st.session_state["feature_names"],
                label_encoders=st.session_state["encoders"],
                model_name=final_name,
            )
            st.session_state["model_saved"] = True
            st.success(f"[OK] Model saved to `credit_model.pkl` and `models/` folder!")
    
    else:
        st.markdown("""
        <div style="background:#1A1D2E; border-radius:14px; padding:2rem; text-align:center; 
                    border:2px dashed #6C63FF;">
            <div style="font-size:2.5rem;">[AI]</div>
            <div style="font-size:1.1rem; color:#9CA3AF; margin-top:0.5rem;">
                Click <b style="color:#6C63FF;">Train All Models</b> to begin
            </div>
        </div>
        """, unsafe_allow_html=True)
