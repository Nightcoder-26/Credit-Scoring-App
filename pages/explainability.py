"""
pages/explainability.py
------------------------
SHAP-based model explainability and feature importance analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, os
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.data_loader import load_and_clean_data, engineer_features, get_train_test_split
from utils.model_trainer import load_model_bundle
from utils.visualizations import plot_feature_importance

# Try importing SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


@st.cache_data(show_spinner="  Preparing data for explainability...")
def get_data_for_shap():
    df_raw = load_and_clean_data("data/german.data")
    df_eng, encoders = engineer_features(df_raw)
    X_tr, X_te, y_tr, y_te, feats, scaler = get_train_test_split(df_eng)
    return X_tr, X_te, y_tr, y_te, feats, scaler


def render():
    st.markdown("## [SHAP] Explainable AI - Model Interpretability")
    st.markdown("""
    Understanding **why** the model makes a prediction is as important as the prediction itself.
    This section provides feature importance charts and SHAP (SHapley Additive exPlanations) analysis.
    """)

    # -- Load model --------------------------------------------
    try:
        bundle = load_model_bundle("credit_model.pkl")
    except FileNotFoundError:
        st.error("[WARN] No model found. Please train the model first (Model Training page or `python train_model.py`).")
        return

    model        = bundle["model"]
    feature_names = bundle["feature_names"]
    scaler       = bundle["scaler"]
    model_name   = bundle.get("model_name", "Trained Model")

    st.success(f"[OK] Loaded: **{model_name}**")

    # -- Load test data ----------------------------------------
    X_train, X_test, y_train, y_test, feats, _ = get_data_for_shap()

    # -- Tab layout --------------------------------------------
    tab1, tab2, tab3 = st.tabs(["[DATA] Feature Importance", "[PREDICT] SHAP Analysis", "  Decision Logic"])

    # -- Tab 1: Feature Importance -----------------------------
    with tab1:
        st.markdown("### [BEST] Feature Importance Rankings")
        st.markdown("Which features drive the model's predictions most strongly?")

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            imp_type = "Tree-based Gini Importance"
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
            imp_type = "Logistic Regression Coefficient Magnitude"
        else:
            st.warning("This model type does not expose feature importances directly.")
            return

        st.caption(f"*Importance type: {imp_type}*")

        # Top N slider
        top_n = st.slider("Number of top features to display", 5, len(feature_names), 15, key="top_n_fi")
        fig = plot_feature_importance(model, feature_names, model_name, top_n=top_n)
        st.plotly_chart(fig, use_container_width=True)

        # Table
        idx = np.argsort(importances)[::-1]
        fi_df = pd.DataFrame({
            "Rank":      range(1, len(feature_names) + 1),
            "Feature":   [feature_names[i] for i in idx],
            "Importance":np.round(importances[idx], 6),
            "% of Total":np.round(importances[idx] / importances.sum() * 100, 2),
        })
        st.dataframe(fi_df, use_container_width=True, hide_index=True, height=350)

        csv = fi_df.to_csv(index=False).encode("utf-8")
        st.download_button("[DOWNLOADING] Download Feature Importance CSV", data=csv,
                           file_name="feature_importance.csv", mime="text/csv")

    # -- Tab 2: SHAP -------------------------------------------
    with tab2:
        st.markdown("### [PREDICT] SHAP Values - Global & Local Explanations")

        if not SHAP_AVAILABLE:
            st.warning("""
            [WARN] **SHAP not installed.** To enable SHAP analysis:
            ```bash
            pip install shap
            ```
            Then restart the app.
            """)
            st.markdown("""
            #### What SHAP provides:
            - **Global explanations**: Which features matter most across all predictions
            - **Local explanations**: Why a specific prediction was made
            - **Force plots**: Visual breakdown of how each feature pushed the prediction
            - **Beeswarm plots**: Feature impact distribution across all samples
            """)
            return

        n_samples = st.slider("Samples for SHAP computation (more = slower)", 50, 300, 100, step=50)
        
        with st.spinner("  Computing SHAP values... this may take a moment"):
            X_sample = X_test[:n_samples]
            
            try:
                # Use TreeExplainer for tree-based models, KernelExplainer as fallback
                if hasattr(model, "feature_importances_"):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)
                    # For binary classifiers, shap_values may be a list [class0, class1]
                    if isinstance(shap_values, list):
                        shap_values_pos = shap_values[1]
                    else:
                        shap_values_pos = shap_values
                else:
                    explainer = shap.KernelExplainer(
                        model.predict_proba, shap.sample(X_train, 50)
                    )
                    shap_values = explainer.shap_values(X_sample, nsamples=100)
                    shap_values_pos = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values

                # Mean absolute SHAP per feature
                mean_shap = np.abs(shap_values_pos).mean(axis=0)
                shap_df = pd.DataFrame({
                    "Feature":    feature_names[:len(mean_shap)],
                    "Mean |SHAP|": np.round(mean_shap, 5),
                }).sort_values("Mean |SHAP|", ascending=False)

                # SHAP bar chart
                fig = px.bar(
                    shap_df.head(15), x="Mean |SHAP|", y="Feature",
                    orientation="h",
                    color="Mean |SHAP|",
                    color_continuous_scale="plasma",
                    title="<b>SHAP Feature Importance (Mean |SHAP| Value)</b>",
                    template="plotly_dark",
                )
                fig.update_layout(height=450, margin=dict(l=160))
                st.plotly_chart(fig, use_container_width=True)

                # SHAP table
                st.dataframe(shap_df, hide_index=True, use_container_width=True, height=300)

                # Per-sample SHAP waterfall (local explanation)
                st.markdown("#### [TARGET] Local Explanation - Single Sample")
                sample_idx = st.slider("Select sample index for local explanation", 0, n_samples - 1, 0)
                
                sample_shap = shap_values_pos[sample_idx]
                sample_x    = X_sample[sample_idx]
                
                local_df = pd.DataFrame({
                    "Feature":    feature_names[:len(sample_shap)],
                    "SHAP Value": np.round(sample_shap, 5),
                    "Feature Value": np.round(sample_x[:len(sample_shap)], 3),
                }).sort_values("SHAP Value", key=abs, ascending=False).head(15)

                colors = ["#43D9AD" if v > 0 else "#FF6B6B" for v in local_df["SHAP Value"]]
                
                fig = go.Figure(go.Bar(
                    x=local_df["SHAP Value"],
                    y=local_df["Feature"],
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v:+.4f}" for v in local_df["SHAP Value"]],
                    textposition="outside",
                ))
                fig.update_layout(
                    title=f"<b>Local SHAP - Sample #{sample_idx}</b><br>"
                          f"<sup>Green = pushes toward Creditworthy, Red = pushes toward Default</sup>",
                    template="plotly_dark",
                    height=420,
                    margin=dict(l=180),
                    xaxis_title="SHAP Value (impact on prediction)",
                )
                st.plotly_chart(fig, use_container_width=True)
                
                pred_prob = model.predict_proba(X_sample[sample_idx:sample_idx+1])[0]
                st.info(f"**Sample #{sample_idx}** - Model prediction: "
                        f"Creditworthy prob = **{pred_prob[1]*100:.1f}%** | "
                        f"Default prob = **{pred_prob[0]*100:.1f}%**")

            except Exception as e:
                st.error(f"SHAP computation error: {e}")
                st.info("Tip: Some model types need specific SHAP explainers. Try training a Random Forest or Gradient Boosting model.")

    # -- Tab 3: Decision Logic ---------------------------------
    with tab3:
        st.markdown("###   Model Decision Logic")

        st.markdown("""
        ####   How Tree-Based Models Decide

        The Random Forest and Gradient Boosting models use **ensemble decision trees**:
        1. Each tree makes a prediction based on feature thresholds
        2. All trees vote, and the majority wins
        3. Feature importance = how often a feature is used for splitting, weighted by improvement

        #### [DOWN] How Logistic Regression Decides

        Logistic Regression computes a **linear combination** of features:
        - `score = w f  + w f  + ... + w f  + bias`
        - The score is passed through a **sigmoid** to get a probability
        - Larger coefficient = more important feature

        ####   Key Risk Factors (German Credit Domain Knowledge)
        """)

        risk_factors = [
            ("[BANK]", "Checking Account Status",  "High Risk",  "Negative or no balance strongly predicts default"),
            (" ", "Credit History",            "High Risk",  "Past delays or critical accounts increase risk"),
            (" ", "Loan Duration",             "Medium Risk", "Longer loans = more repayment uncertainty"),
            (" ", "Credit Amount",             "Medium Risk", "Higher amounts relative to income = more stress"),
            (" ", "Employment Duration",       "High Risk",  "Unemployed or < 1 year = less stability"),
            (" ", "Installment Rate",          "Medium Risk", "Higher % of income spent = more financial burden"),
            ("[HOME]", "Housing Status",            "Low Risk",   "Own property = lower default risk generally"),
            (" ", "Savings Account Balance",   "High Risk",  "No savings = no buffer during financial shocks"),
            (" ", "Age",                       "Low Risk",   "Older applicants tend to have more stability"),
        ]

        for icon, factor, risk, explanation in risk_factors:
            color = "#FF6B6B" if "High" in risk else ("#FFB347" if "Medium" in risk else "#43D9AD")
            st.markdown(f"""
            <div style="background:#1A1D2E; border-radius:10px; padding:0.9rem 1rem; 
                        margin-bottom:0.6rem; border-left:4px solid {color};">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="font-size:1.1rem;">{icon}</span>
                        <b style="color:#FFFFFF; margin-left:0.5rem;">{factor}</b>
                        <span style="background:{color}22; color:{color}; border-radius:4px; 
                                     padding:0.1rem 0.5rem; font-size:0.72rem; margin-left:0.5rem;">
                            {risk}
                        </span>
                    </div>
                </div>
                <div style="color:#9CA3AF; font-size:0.82rem; margin-top:0.4rem; margin-left:1.8rem;">
                    {explanation}
                </div>
            </div>
            """, unsafe_allow_html=True)
