"""
pages/about.py
--------------
About page: project info, dataset credits, methodology, and author section.
"""

import streamlit as st


def render():
    st.markdown("## [ABOUT] About This Project")

    # -- Project Overview --------------------------------------
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1A1D2E, #12152a);
                border-radius:16px; padding:2rem; border:1px solid #2D3148; margin-bottom:1.5rem;">
        <h3 style="color:#6C63FF; margin-top:0;">[BANK] Credit Scoring Model Using Machine Learning</h3>
        <p style="color:#9CA3AF; line-height:1.7; margin:0;">
            A production-ready, end-to-end Machine Learning system that predicts credit risk using real 
            financial and personal data. Built with industry-standard tools and best practices, this project 
            demonstrates the complete ML lifecycle from data ingestion to model deployment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### [TARGET] Project Objectives")
        objectives = [
            "Predict creditworthiness from applicant financial data",
            "Train & compare multiple ML classification algorithms",
            "Provide interpretable, explainable AI predictions",
            "Build a polished, interactive Streamlit dashboard",
            "Follow production ML engineering best practices",
        ]
        for obj in objectives:
            st.markdown(f"[OK] {obj}")

        st.markdown("### [DATA] Dataset")
        st.markdown("""
        **German Credit Dataset** (Statlog)
        - **Source:** UCI Machine Learning Repository
        - **Creator:** Prof. Dr. Hans Hofmann, University of Hamburg
        - **Samples:** 1,000 real credit applicants
        - **Features:** 20 financial and personal attributes
        - **Target:** Creditworthy (Good) vs Default (Bad)
        - **Year:** 1994 (widely used benchmark to this day)
        
        [  Dataset Link](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
        """)

    with col2:
        st.markdown("###   Tech Stack")
        tech_items = [
            (" ", "Python 3.10+",         "Core language"),
            ("[DATA]", "Pandas & NumPy",        "Data manipulation"),
            ("[AI]", "Scikit-learn",          "ML models & preprocessing"),
            (" ", "XGBoost",               "Gradient boosted trees"),
            ("[SHAP]", "SHAP",                  "Model explainability"),
            ("[UP]", "Plotly & Seaborn",      "Interactive visualizations"),
            (" ", "Streamlit",             "Web dashboard"),
            ("[SAVED]", "Joblib",                "Model persistence"),
        ]
        for icon, tech, desc in tech_items:
            st.markdown(f"""
            <div style="background:#1A1D2E; border-radius:8px; padding:0.6rem 0.8rem; 
                        margin-bottom:0.4rem; display:flex; align-items:center;">
                <span style="font-size:1.1rem; margin-right:0.6rem;">{icon}</span>
                <div>
                    <b style="color:#FFFFFF;">{tech}</b>
                    <span style="color:#9CA3AF; font-size:0.8rem; margin-left:0.5rem;">- {desc}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # -- ML Pipeline -------------------------------------------
    st.markdown("---")
    st.markdown("###   ML Pipeline Overview")
    
    pipeline_steps = [
        ("1", "Data Ingestion",      "Download German Credit Dataset from UCI ML Repository"),
        ("2", "Data Cleaning",       "Handle missing values, duplicates, and map coded values to readable labels"),
        ("3", "Feature Engineering", "Create debt ratios, age groups, credit tiers; label-encode categoricals"),
        ("4", "Train/Test Split",    "80/20 stratified split with StandardScaler normalization"),
        ("5", "Model Training",      "Train 5 classifiers: LR, DT, RF, GBM, XGBoost"),
        ("6", "Evaluation",          "Compare Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix"),
        ("7", "Hyperparameter Tuning","GridSearchCV with 5-fold cross-validation on best model"),
        ("8", "Model Saving",        "Save final bundle (model + scaler + encoders) with Joblib"),
        ("9", "Prediction API",      "Interactive form   encoded input   scaled   predict   display"),
        ("10","Explainability",       "SHAP TreeExplainer for global and local model interpretation"),
    ]

    for step, title, desc in pipeline_steps:
        st.markdown(f"""
        <div style="display:flex; gap:1rem; margin-bottom:0.8rem; align-items:flex-start;">
            <div style="background:#6C63FF; color:#FFF; border-radius:50%; width:2rem; height:2rem;
                        display:flex; align-items:center; justify-content:center; 
                        font-weight:800; font-size:0.8rem; flex-shrink:0; margin-top:0.1rem;">
                {step}
            </div>
            <div>
                <b style="color:#FFFFFF;">{title}</b>
                <div style="color:#9CA3AF; font-size:0.85rem;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # -- Model Results -----------------------------------------
    st.markdown("---")
    st.markdown("### [UP] Expected Model Results")
    
    import pandas as pd
    results_data = {
        "Model":         ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost"],
        "Accuracy":      ["~75%", "~72%", "~79%", "~78%", "~80%"],
        "ROC-AUC":       ["~0.78", "~0.72", "~0.83", "~0.82", "~0.84"],
        "F1 Score":      ["~0.84", "~0.80", "~0.87", "~0.86", "~0.87"],
        "Best For":      ["Baseline", "Interpretability", "Robustness", "Accuracy", "Performance"],
    }
    st.dataframe(pd.DataFrame(results_data), use_container_width=True, hide_index=True)
    st.caption("*Results are approximate and depend on random seed, preprocessing choices, and tuning.*")

    # -- Future Improvements -----------------------------------
    st.markdown("---")
    st.markdown("### [PREDICT] Future Improvements")
    
    futures = [
        (" ", "Neural Networks",     "Add deep learning model with PyTorch/Keras for comparison"),
        (" ", "Class Imbalance",     "Apply SMOTE oversampling or class weighting for better recall"),
        (" ", "Automated Retraining","Schedule periodic retraining when new data arrives"),
        (" ", "More Datasets",       "Integrate Give Me Some Credit / LendingClub dataset"),
        (" ", "REST API",            "Expose predictions as FastAPI REST endpoint"),
        (" ", "Mobile UI",           "Build React Native or Flutter mobile frontend"),
        ("[DATA]", "Drift Detection",     "Monitor prediction drift and data distribution shifts"),
        (" ", "Auth & Logging",      "Add user authentication and prediction audit trail"),
    ]

    col1, col2 = st.columns(2)
    for i, (icon, title, desc) in enumerate(futures):
        [col1, col2][i % 2].markdown(f"""
        <div style="background:#1A1D2E; border-radius:10px; padding:0.8rem 1rem; 
                    margin-bottom:0.6rem; border-left:3px solid #6C63FF;">
            <div style="font-size:1rem;">{icon} <b style="color:#FFFFFF;">{title}</b></div>
            <div style="color:#9CA3AF; font-size:0.8rem; margin-top:0.2rem;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # -- Deployment --------------------------------------------
    st.markdown("---")
    st.markdown("### [RUN] Deployment Guide")
    
    dep_tab1, dep_tab2, dep_tab3 = st.tabs(["Streamlit Cloud", "Render", "HuggingFace Spaces"])
    
    with dep_tab1:
        st.markdown("""
        #### Deploy to Streamlit Cloud (Free & Easiest)
        
        1. Push your project to a public GitHub repo
        2. Go to [share.streamlit.io](https://share.streamlit.io)
        3. Click **New App**   connect your GitHub repo
        4. Set **Main file path** to `app.py`
        5. Click **Deploy** - done!
        
        > [WARN] Streamlit Cloud has ~1GB RAM. Add `@st.cache_data` to all data loading functions (already done!).
        
        **Required files in repo root:**
        - `app.py`
        - `requirements.txt`
        - `credit_model.pkl` (commit the trained model)
        """)

    with dep_tab2:
        st.markdown("""
        #### Deploy to Render (Free tier available)
        
        1. Push to GitHub
        2. Go to [render.com](https://render.com)   New Web Service
        3. Connect your GitHub repository
        4. Set **Build Command**: `pip install -r requirements.txt`
        5. Set **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
        6. Choose free tier   Deploy
        
        > Render free tier spins down after 15 min of inactivity.
        """)

    with dep_tab3:
        st.markdown("""
        #### Deploy to HuggingFace Spaces (Free GPU available)
        
        1. Create account at [huggingface.co](https://huggingface.co)
        2. New Space   Choose **Streamlit** SDK
        3. Upload your project files
        4. Create `README.md` with `sdk: streamlit` in the YAML header:
        ```yaml
        ---
        title: Credit Scoring AI
        emoji: [BANK]
        colorFrom: purple
        colorTo: pink
        sdk: streamlit
        sdk_version: 1.35.0
        app_file: app.py
        pinned: true
        ---
        ```
        5. Push or upload   Space auto-builds!
        
        > HuggingFace Spaces is great for ML demos and gives free CPU resources.
        """)

    # -- Author ------------------------------------------------
    st.markdown("---")
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1A1D2E,#12152a); border-radius:16px; 
                padding:2rem; border:1px solid #6C63FF44; text-align:center;">
        <div style="font-size:3rem; margin-bottom:0.5rem;"> </div>
        <h3 style="color:#6C63FF; margin:0 0 0.5rem 0;">Built for CodeAlpha Internship</h3>
        <p style="color:#9CA3AF; margin:0 0 1rem 0;">
            This project was developed as part of the <b style="color:#FFFFFF;">CodeAlpha</b> 
            Machine Learning Internship program.
        </p>
        <div style="display:flex; justify-content:center; gap:1rem; flex-wrap:wrap;">
            <span style="background:#6C63FF22; color:#6C63FF; padding:0.3rem 0.8rem; 
                         border-radius:20px; font-size:0.85rem;">[BEST] Internship Project</span>
            <span style="background:#43D9AD22; color:#43D9AD; padding:0.3rem 0.8rem; 
                         border-radius:20px; font-size:0.85rem;">[AI] Machine Learning</span>
            <span style="background:#FF658422; color:#FF6584; padding:0.3rem 0.8rem; 
                         border-radius:20px; font-size:0.85rem;">[DATA] Data Science</span>
            <span style="background:#FFB34722; color:#FFB347; padding:0.3rem 0.8rem; 
                         border-radius:20px; font-size:0.85rem;">  Full Stack AI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; color:#9CA3AF; font-size:0.82rem; margin-top:1rem;">
          Dataset: UCI German Credit (Hofmann, 1994) &nbsp;|&nbsp; 
          Stack: Python   Scikit-learn   XGBoost   SHAP   Plotly   Streamlit
    </div>
    """, unsafe_allow_html=True)
