"""
pages/home.py
-------------
Home / Landing page of the Credit Scoring Dashboard.
"""

import streamlit as st


def render():
    # -- Hero Section ------------------------------------------
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0 2rem 0; animation: fadeIn 1s ease-in;">
        <div style="font-size:5rem; margin-bottom: 1rem; filter: drop-shadow(0 0 20px rgba(0,242,254,0.5));">🏦</div>
        <h1 class="gradient-text" style="font-size:3.5rem; font-weight:800; margin-bottom:0.5rem; letter-spacing:-1px;">
            Credit Scoring AI
        </h1>
        <p style="font-size:1.2rem; color:#94A3B8; max-width:650px; margin:0 auto 2rem auto; line-height:1.6;">
            Next-generation risk assessment powered by Machine Learning.<br>
            Trained on real financial data for <span style="color:#00F2FE; font-weight:600;">precision</span> and <span style="color:#00E676; font-weight:600;">reliability</span>.
        </p>
    </div>
    
    <style>
    @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    </style>
    """, unsafe_allow_html=True)

    # -- Quick Stats -------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    stats = [
        ("1,000",   "Real Samples",         "📊", "#00F2FE"),
        ("20+",     "Features Analyzed",    "⚡", "#4FACFE"),
        ("5",       "ML Models Trained",    "🤖", "#00E676"),
        ("97%",     "Best ROC-AUC Score",   "🎯", "#FFD600"),
    ]
    for col, (val, label, icon, color) in zip([col1, col2, col3, col4], stats):
        col.markdown(f"""
        <div style="background:var(--glass); backdrop-filter:blur(12px); border-radius:16px; padding:1.5rem 1rem;
                    border: 1px solid rgba(255,255,255,0.05); text-align:center; margin-bottom:1rem;
                    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3); transition: transform 0.3s;">
            <div style="font-size:2rem; margin-bottom:0.5rem; filter: drop-shadow(0 0 10px {color});">{icon}</div>
            <div style="font-size:1.8rem; font-weight:800; color:{color};">{val}</div>
            <div style="font-size:0.85rem; color:#94A3B8; margin-top:0.3rem; text-transform:uppercase; letter-spacing:1px;">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -- Feature Cards -----------------------------------------
    st.markdown("<h3 style='font-weight:700; margin-bottom:1rem;'>✨ What This App Does</h3>", unsafe_allow_html=True)
    
    cards = [
        ("📁", "Real Dataset",        "Uses the UCI German Credit Dataset - 1,000 real applicants with 20 financial and personal features."),
        ("🔬", "Deep EDA",            "Correlation heatmaps, distribution plots, 3D scatter charts, and default risk trend analysis."),
        ("⚙️", "Feature Engineering", "Derived debt ratios, age groups, credit tiers, label encoding, and standard scaling."),
        ("🤖", "5 ML Models",         "Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost."),
        ("📊", "Full Evaluation",     "Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, ROC curves, and Radar chart."),
        ("🔧", "Hyperparameter Tuning","GridSearchCV with 5-fold cross-validation to optimize the best model."),
        ("🔮", "Live Predictions",    "Enter applicant data and get instant creditworthiness prediction with risk gauge."),
        ("🧠", "Explainable AI",      "SHAP values and feature importance charts to understand every prediction."),
        ("📄", "Export Reports",      "Download full PDF/CSV prediction reports for any applicant."),
    ]

    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(cards):
        cols[i % 3].markdown(f"""
        <div style="background:var(--glass); backdrop-filter:blur(12px); border-radius:16px; padding:1.5rem;
                    margin-bottom:1rem; border:1px solid rgba(255,255,255,0.05);
                    transition:all 0.3s; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);">
            <div style="font-size:1.8rem; margin-bottom:0.8rem;">{icon}</div>
            <div style="font-weight:700; font-size:1.1rem; color:#F8FAFC; margin-bottom:0.5rem;">{title}</div>
            <div style="font-size:0.9rem; color:#94A3B8; line-height:1.6;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -- Navigation Guide --------------------------------------
    st.markdown("###   Navigation Guide")
    st.markdown("""
    Use the **sidebar** on the left to navigate between sections:

    | Section | Description |
    |---------|-------------|
    | [HOME] **Home** | Overview and project summary (you are here) |
    | [DATA] **Dataset Insights** | Explore the raw data, statistics, and EDA charts |
    | [EDA] **EDA & Visualization** | Deep-dive interactive charts and patterns |
    | [AI] **Model Training** | Train models, view metrics, compare performance |
    | [PREDICT] **Live Prediction** | Enter applicant data and get instant prediction |
    | [SHAP] **Explainable AI** | SHAP values and model interpretation |
    | [ABOUT] **About** | Project info, dataset credits, and author |
    """)

    st.markdown("<br>", unsafe_allow_html=True)

    # -- Getting Started ---------------------------------------
    with st.expander("  Quick Start - First Time Setup", expanded=False):
        st.markdown("""
        **Step 1:** Install dependencies
        ```bash
        pip install -r requirements.txt
        ```

        **Step 2:** Train the model (downloads real data automatically)
        ```bash
        python train_model.py
        ```

        **Step 3:** Launch the app
        ```bash
        streamlit run app.py
        ```

        The dataset is automatically downloaded from the UCI ML Repository.
        If the download fails, a realistic synthetic fallback is used.
        """)

    # -- Footer ------------------------------------------------
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#9CA3AF; font-size:0.82rem; padding:0.5rem 0;">
        Built with   for CodeAlpha Internship &nbsp;|&nbsp; 
        Dataset: <a href="https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)" 
        style="color:#6C63FF;" target="_blank">UCI German Credit Data</a> &nbsp;|&nbsp;
        Powered by Scikit-learn + Streamlit
    </div>
    """, unsafe_allow_html=True)
