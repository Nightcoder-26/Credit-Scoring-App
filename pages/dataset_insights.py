"""
pages/dataset_insights.py
--------------------------
Dataset loading, preview, statistics, and data quality overview.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.data_loader import load_and_clean_data, NUMERIC_COLUMNS


@st.cache_data(show_spinner="[DOWNLOADING] Loading dataset...")
def get_data():
    return load_and_clean_data("data/german.data")


def render():
    st.markdown("## [DATA] Dataset Insights")
    st.markdown("Exploring the **UCI German Credit Dataset** - a benchmark dataset for credit risk classification.")

    df = get_data()

    # -- Top-Level Stats ---------------------------------------
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("[SAMPLE] Total Samples",    f"{len(df):,}")
    col2.metric("  Total Features",   f"{df.shape[1] - 1}")
    col3.metric("[OK] Creditworthy",     f"{(df['credit_risk']==1).sum():,}  ({(df['credit_risk']==1).mean()*100:.0f}%)")
    col4.metric("  Default",          f"{(df['credit_risk']==0).sum():,}  ({(df['credit_risk']==0).mean()*100:.0f}%)")
    col5.metric("  Missing Values",   f"{df.isnull().sum().sum()}")

    st.markdown("---")

    # -- Tabs --------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(["  Raw Preview", "[UP] Statistics", "  Data Quality", "  Column Info"])

    with tab1:
        st.markdown("#### Dataset Preview")
        
        # Filter controls
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            risk_filter = st.selectbox("Filter by Credit Risk", ["All", "Creditworthy (1)", "Default (0)"], key="risk_filter")
        with c2:
            n_rows = st.slider("Rows to show", 5, 100, 20, key="n_rows_slider")
        with c3:
            st.markdown("<br>", unsafe_allow_html=True)
            show_all = st.checkbox("All columns", value=True)
        
        display_df = df.copy()
        if risk_filter == "Creditworthy (1)":
            display_df = display_df[display_df["credit_risk"] == 1]
        elif risk_filter == "Default (0)":
            display_df = display_df[display_df["credit_risk"] == 0]
        
        if not show_all:
            display_cols = ["age", "duration", "credit_amount", "credit_history",
                           "purpose", "employment", "housing", "credit_risk"]
            display_cols = [c for c in display_cols if c in display_df.columns]
            display_df = display_df[display_cols]
        
        st.dataframe(
            display_df.head(n_rows),
            use_container_width=True,
            height=400,
        )
        
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "[DOWNLOADING] Download Full Dataset (CSV)",
                data=csv,
                file_name="german_credit_data.csv",
                mime="text/csv",
            )

    with tab2:
        st.markdown("#### Descriptive Statistics - Numeric Features")
        
        numeric_df = df.select_dtypes(include=np.number)
        stats = numeric_df.describe().T
        stats["skewness"] = numeric_df.skew()
        stats["kurtosis"] = numeric_df.kurtosis()
        stats = stats.round(3)
        
        st.dataframe(stats, use_container_width=True)
        
        st.markdown("#### Numeric Feature Distributions")
        num_cols_available = [c for c in NUMERIC_COLUMNS if c in df.columns]
        
        import plotly.express as px
        selected_col = st.selectbox("Select feature to explore", num_cols_available, key="dist_col")
        
        df_copy = df.copy()
        df_copy["Status"] = df_copy["credit_risk"].map({1: "Creditworthy", 0: "Default"})
        
        fig = px.histogram(
            df_copy, x=selected_col, color="Status", nbins=30, barmode="overlay",
            color_discrete_map={"Creditworthy": "#43D9AD", "Default": "#FF6B6B"},
            opacity=0.75,
            template="plotly_dark",
            title=f"Distribution of <b>{selected_col}</b>",
        )
        fig.update_layout(height=350, margin=dict(t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("#### Data Quality Report")
        
        quality_data = []
        for col in df.columns:
            quality_data.append({
                "Column":         col,
                "Type":           str(df[col].dtype),
                "Non-Null Count": df[col].notna().sum(),
                "Null Count":     df[col].isnull().sum(),
                "Null %":         f"{df[col].isnull().mean()*100:.1f}%",
                "Unique Values":  df[col].nunique(),
                "Sample Values":  str(df[col].dropna().unique()[:3].tolist()),
            })
        
        quality_df = pd.DataFrame(quality_data)
        st.dataframe(quality_df, use_container_width=True, height=420)
        
        # Missing value summary
        total_missing = df.isnull().sum().sum()
        if total_missing == 0:
            st.success("[OK] No missing values detected in this dataset. It is complete and clean.")
        else:
            st.warning(f"[WARN] {total_missing} missing values found. These were imputed during preprocessing.")

        # Duplicate check
        dupes = df.duplicated().sum()
        if dupes == 0:
            st.success("[OK] No duplicate rows found.")
        else:
            st.warning(f"[WARN] {dupes} duplicate rows found and removed during cleaning.")

    with tab4:
        st.markdown("#### Feature Reference Guide")
        
        col_info = [
            ("checking_account",   "Categorical", "Status of existing checking account"),
            ("duration",           "Numeric",     "Loan duration in months"),
            ("credit_history",     "Categorical", "Past credit payment history"),
            ("purpose",            "Categorical", "Purpose of the credit/loan"),
            ("credit_amount",      "Numeric",     "Credit amount in Deutsche Marks (DM)"),
            ("savings_account",    "Categorical", "Savings account balance category"),
            ("employment",         "Categorical", "Duration of current employment"),
            ("installment_rate",   "Numeric",     "Installment as % of disposable income"),
            ("personal_status",    "Categorical", "Personal status and gender"),
            ("other_debtors",      "Categorical", "Presence of guarantors or co-applicants"),
            ("residence_since",    "Numeric",     "Years at current residence"),
            ("property",           "Categorical", "Type of property owned"),
            ("age",                "Numeric",     "Applicant age in years"),
            ("other_installments", "Categorical", "Other installment plans (bank, stores, none)"),
            ("housing",            "Categorical", "Housing status (rent/own/free)"),
            ("existing_credits",   "Numeric",     "Number of existing credits at this bank"),
            ("job",                "Categorical", "Job skill level category"),
            ("num_dependents",     "Numeric",     "Number of financial dependents"),
            ("telephone",          "Categorical", "Registered telephone (yes/no)"),
            ("foreign_worker",     "Categorical", "Is the applicant a foreign worker?"),
            ("credit_risk",        "Target (0/1)","0 = Default, 1 = Creditworthy"),
        ]
        
        info_df = pd.DataFrame(col_info, columns=["Feature", "Type", "Description"])
        st.dataframe(info_df, use_container_width=True, height=500)
        
        st.info("[INFO] The dataset uses coded values (e.g., A11, A32) which are mapped to human-readable labels during preprocessing.")
