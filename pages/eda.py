"""
pages/eda.py
------------
Exploratory Data Analysis with rich interactive Plotly charts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, os
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.data_loader import load_and_clean_data
from utils.visualizations import (
    plot_class_distribution,
    plot_age_distribution,
    plot_credit_amount_by_purpose,
    plot_correlation_heatmap,
    plot_duration_vs_amount,
    plot_age_vs_amount_risk,
)


@st.cache_data(show_spinner="[DATA] Loading data for EDA...")
def get_data():
    return load_and_clean_data("data/german.data")


def render():
    st.markdown("## [EDA] Exploratory Data Analysis")
    st.markdown("Interactive visualizations to uncover patterns, distributions, and risk factors in the dataset.")

    df = get_data()
    df_vis = df.copy()
    df_vis["Status"] = df_vis["credit_risk"].map({1: "Creditworthy", 0: "Default"})

    # -- Section 1: Class & Distribution ----------------------
    st.markdown("### [UP] Class Distribution & Key Distributions")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = plot_class_distribution(df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = plot_age_distribution(df)
        st.plotly_chart(fig, use_container_width=True)

    # -- Section 2: Financial Analysis ------------------------
    st.markdown("###   Credit Amount Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = plot_credit_amount_by_purpose(df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = plot_duration_vs_amount(df)
        st.plotly_chart(fig, use_container_width=True)

    # -- Section 3: Correlation --------------------------------
    st.markdown("###   Correlation Analysis")
    fig = plot_correlation_heatmap(df)
    st.plotly_chart(fig, use_container_width=True)

    # -- Section 4: Risk Trends --------------------------------
    st.markdown("### [DOWN] Default Risk Trends by Feature")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Default rate by employment duration
        if "employment" in df.columns and df["employment"].dtype == "object":
            emp_risk = df.groupby("employment")["credit_risk"].agg(["mean", "count"]).reset_index()
            emp_risk.columns = ["Employment", "Default Rate (%)", "Count"]
            emp_risk["Default Rate (%)"] = (1 - emp_risk["Default Rate (%)"]) * 100
            emp_risk = emp_risk.sort_values("Default Rate (%)", ascending=False)
            
            fig = px.bar(
                emp_risk, x="Employment", y="Default Rate (%)",
                color="Default Rate (%)",
                color_continuous_scale="RdYlGn_r",
                title="<b>Default Rate by Employment Duration</b>",
                template="plotly_dark", text="Count",
            )
            fig.update_traces(texttemplate="n=%{text}", textposition="outside")
            fig.update_layout(title_font_size=15, xaxis_tickangle=-30, height=380)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Numeric credit_amount binned analysis
            df_vis["amount_bin"] = pd.cut(df_vis["credit_amount"], bins=5)
            risk_by_bin = df_vis.groupby("amount_bin", observed=True)["credit_risk"].mean().reset_index()
            risk_by_bin.columns = ["Amount Range", "Creditworthy Rate"]
            risk_by_bin["Amount Range"] = risk_by_bin["Amount Range"].astype(str)
            fig = px.bar(
                risk_by_bin, x="Amount Range", y="Creditworthy Rate",
                color="Creditworthy Rate", color_continuous_scale="RdYlGn",
                title="<b>Creditworthy Rate by Credit Amount</b>",
                template="plotly_dark",
            )
            fig.update_layout(title_font_size=15, height=380)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Default rate by housing
        if "housing" in df.columns and df["housing"].dtype == "object":
            housing_risk = df.groupby("housing")["credit_risk"].agg(["mean","count"]).reset_index()
            housing_risk.columns = ["Housing", "Creditworthy Rate", "Count"]
            housing_risk["Default Rate (%)"] = (1 - housing_risk["Creditworthy Rate"]) * 100
            
            fig = px.pie(
                housing_risk, values="Count", names="Housing",
                title="<b>Applicants by Housing Status</b>",
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Vivid,
                hole=0.4,
            )
            fig.update_layout(title_font_size=15, height=380)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Duration distribution by credit risk
            fig = px.violin(
                df_vis, x="Status", y="duration",
                color="Status",
                color_discrete_map={"Creditworthy": "#43D9AD", "Default": "#FF6B6B"},
                box=True, points="outliers",
                title="<b>Loan Duration Distribution by Risk</b>",
                template="plotly_dark",
            )
            fig.update_layout(title_font_size=15, height=380)
            st.plotly_chart(fig, use_container_width=True)

    # -- Section 5: 3D Visualization ---------------------------
    st.markdown("###   3D Risk Landscape")
    fig = plot_age_vs_amount_risk(df)
    st.plotly_chart(fig, use_container_width=True)

    # -- Section 6: Custom Chart Builder ----------------------
    st.markdown("---")
    st.markdown("###   Custom Chart Builder")
    st.markdown("Build your own chart by selecting features to explore:")
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "credit_risk"]
    
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        x_col = st.selectbox("X-Axis", numeric_cols, index=0, key="custom_x")
    with cc2:
        y_col = st.selectbox("Y-Axis", numeric_cols, index=1, key="custom_y")
    with cc3:
        chart_type = st.selectbox("Chart Type", ["Scatter", "Box", "Histogram", "Violin"], key="chart_type")
    
    df_custom = df_vis.copy()
    
    if chart_type == "Scatter":
        fig = px.scatter(
            df_custom, x=x_col, y=y_col, color="Status",
            color_discrete_map={"Creditworthy": "#43D9AD", "Default": "#FF6B6B"},
            opacity=0.6, template="plotly_dark",
            title=f"<b>{x_col} vs {y_col}</b>",
            trendline="ols",
        )
    elif chart_type == "Box":
        fig = px.box(
            df_custom, x="Status", y=x_col, color="Status",
            color_discrete_map={"Creditworthy": "#43D9AD", "Default": "#FF6B6B"},
            template="plotly_dark",
            title=f"<b>{x_col} Distribution by Credit Risk</b>",
        )
    elif chart_type == "Histogram":
        fig = px.histogram(
            df_custom, x=x_col, color="Status", nbins=30, barmode="overlay",
            color_discrete_map={"Creditworthy": "#43D9AD", "Default": "#FF6B6B"},
            opacity=0.75, template="plotly_dark",
            title=f"<b>{x_col} Histogram</b>",
        )
    else:  # Violin
        fig = px.violin(
            df_custom, x="Status", y=x_col, color="Status", box=True,
            color_discrete_map={"Creditworthy": "#43D9AD", "Default": "#FF6B6B"},
            template="plotly_dark",
            title=f"<b>{x_col} Violin Plot</b>",
        )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
