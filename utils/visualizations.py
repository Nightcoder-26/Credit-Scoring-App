"""
visualizations.py
-----------------
All chart-generation functions used across the Streamlit app.
Returns Plotly figures for interactive display.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc


# -- Color Palette ----------------------------------------------
COLORS = {
    "primary":    "#6C63FF",
    "secondary":  "#FF6584",
    "success":    "#43D9AD",
    "warning":    "#FFB347",
    "danger":     "#FF6B6B",
    "bg_dark":    "#0F1117",
    "bg_card":    "#1A1D2E",
    "text":       "#FFFFFF",
    "muted":      "#9CA3AF",
    "good":       "#43D9AD",
    "bad":        "#FF6B6B",
}

CHART_TEMPLATE = "plotly_dark"
COLOR_SEQ = px.colors.qualitative.Vivid


# --------------------------------------------------------------
# EDA Charts
# --------------------------------------------------------------

def plot_class_distribution(df: pd.DataFrame) -> go.Figure:
    """Donut chart of creditworthy vs not creditworthy."""
    counts = df["credit_risk"].value_counts().reset_index()
    counts.columns = ["credit_risk", "count"]
    counts["label"] = counts["credit_risk"].map({1: "Creditworthy", 0: "Default"})
    
    fig = px.pie(
        counts, values="count", names="label",
        color="label",
        color_discrete_map={"Creditworthy": COLORS["good"], "Default": COLORS["bad"]},
        hole=0.55,
        title="<b>Credit Risk Class Distribution</b>",
        template=CHART_TEMPLATE,
    )
    fig.update_traces(textposition="outside", textinfo="percent+label",
                      hovertemplate="%{label}: %{value} samples (%{percent})")
    fig.update_layout(
        title_font_size=18,
        showlegend=True,
        margin=dict(t=60, b=20),
        annotations=[dict(text="<b>Risk<br>Split</b>", x=0.5, y=0.5,
                          font_size=14, showarrow=False, font_color=COLORS["text"])]
    )
    return fig


def plot_age_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of age distribution colored by credit risk."""
    df_copy = df.copy()
    df_copy["Status"] = df_copy["credit_risk"].map({1: "Creditworthy", 0: "Default"})
    
    fig = px.histogram(
        df_copy, x="age", color="Status", nbins=30, barmode="overlay",
        color_discrete_map={"Creditworthy": COLORS["good"], "Default": COLORS["bad"]},
        opacity=0.75,
        title="<b>Age Distribution by Credit Risk</b>",
        labels={"age": "Age (years)", "count": "Count"},
        template=CHART_TEMPLATE,
    )
    fig.update_layout(title_font_size=18, bargap=0.05)
    return fig


def plot_credit_amount_by_purpose(df: pd.DataFrame) -> go.Figure:
    """Box plot of credit amount per purpose."""
    if "purpose" not in df.columns or df["purpose"].dtype != "object":
        # Encoded data - just do histogram
        fig = px.histogram(
            df, x="credit_amount", nbins=40,
            color_discrete_sequence=[COLORS["primary"]],
            title="<b>Credit Amount Distribution</b>",
            labels={"credit_amount": "Credit Amount (DM)"},
            template=CHART_TEMPLATE,
        )
    else:
        fig = px.box(
            df, x="purpose", y="credit_amount",
            color_discrete_sequence=[COLORS["primary"]],
            title="<b>Credit Amount by Purpose</b>",
            labels={"credit_amount": "Credit Amount (DM)", "purpose": "Purpose"},
            template=CHART_TEMPLATE,
        )
        fig.update_xaxes(tickangle=-30)
    
    fig.update_layout(title_font_size=18)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Correlation heatmap for numeric features."""
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="RdBu_r",
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        hovertemplate="<b>%{x}</b>   <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="<b>Feature Correlation Heatmap</b>",
        title_font_size=18,
        template=CHART_TEMPLATE,
        height=550,
        xaxis=dict(tickangle=-45),
    )
    return fig


def plot_duration_vs_amount(df: pd.DataFrame) -> go.Figure:
    """Scatter: loan duration vs credit amount colored by risk."""
    df_copy = df.copy()
    df_copy["Status"] = df_copy["credit_risk"].map({1: "Creditworthy", 0: "Default"})
    
    fig = px.scatter(
        df_copy, x="duration", y="credit_amount",
        color="Status", opacity=0.6, size_max=8,
        color_discrete_map={"Creditworthy": COLORS["good"], "Default": COLORS["bad"]},
        title="<b>Loan Duration vs Credit Amount</b>",
        labels={"duration": "Duration (months)", "credit_amount": "Credit Amount (DM)"},
        template=CHART_TEMPLATE,
        marginal_x="histogram", marginal_y="violin",
    )
    fig.update_layout(title_font_size=18)
    return fig


def plot_age_vs_amount_risk(df: pd.DataFrame) -> go.Figure:
    """3D scatter of age, credit amount, and duration with risk coloring."""
    df_copy = df.copy()
    df_copy["Status"] = df_copy["credit_risk"].map({1: "Creditworthy", 0: "Default"})
    
    fig = px.scatter_3d(
        df_copy, x="age", y="credit_amount", z="duration",
        color="Status",
        color_discrete_map={"Creditworthy": COLORS["good"], "Default": COLORS["bad"]},
        opacity=0.6,
        title="<b>3D Risk Landscape: Age   Amount   Duration</b>",
        labels={"age": "Age", "credit_amount": "Amount (DM)", "duration": "Duration (mo)"},
        template=CHART_TEMPLATE,
        height=500,
    )
    fig.update_layout(title_font_size=16)
    return fig


# --------------------------------------------------------------
# Model Evaluation Charts
# --------------------------------------------------------------

def plot_model_comparison(results: dict) -> go.Figure:
    """Grouped bar chart comparing all model metrics."""
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    
    model_names = list(results.keys())
    
    fig = go.Figure()
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [results[m][metric] for m in model_names]
        fig.add_trace(go.Bar(
            name=label,
            x=model_names,
            y=values,
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
            marker_color=COLOR_SEQ[i % len(COLOR_SEQ)],
        ))
    
    fig.update_layout(
        barmode="group",
        title="<b>Model Performance Comparison</b>",
        title_font_size=18,
        yaxis=dict(range=[0, 1.12], title="Score"),
        xaxis_title="Model",
        template=CHART_TEMPLATE,
        legend=dict(orientation="h", y=1.05),
        height=480,
    )
    return fig


def plot_confusion_matrix(cm: np.ndarray, model_name: str) -> go.Figure:
    """Styled confusion matrix heatmap."""
    labels = ["Default (0)", "Creditworthy (1)"]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale=[[0, "#1A1D2E"], [1, COLORS["primary"]]],
        text=cm,
        texttemplate="<b>%{text}</b>",
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{text}<extra></extra>",
        showscale=False,
    ))
    fig.update_layout(
        title=f"<b>Confusion Matrix - {model_name}</b>",
        title_font_size=16,
        template=CHART_TEMPLATE,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=380,
    )
    return fig


def plot_roc_curves(results: dict, y_test: np.ndarray) -> go.Figure:
    """Multi-model ROC curve overlay."""
    fig = go.Figure()
    
    # Random classifier baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines", name="Random Classifier",
        line=dict(color=COLORS["muted"], dash="dash"),
    ))
    
    for i, (name, res) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        auc_score = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"{name} (AUC={auc_score:.3f})",
            line=dict(color=COLOR_SEQ[i % len(COLOR_SEQ)], width=2.5),
        ))
    
    fig.update_layout(
        title="<b>ROC Curves - All Models</b>",
        title_font_size=18,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template=CHART_TEMPLATE,
        legend=dict(x=0.6, y=0.05),
        height=480,
    )
    return fig


def plot_feature_importance(model, feature_names: list, model_name: str, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of feature importances."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return go.Figure()
    
    # Sort and select top N
    idx = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in idx]
    top_importances = importances[idx]
    
    # Color gradient
    colors = [f"rgba(108,99,255,{0.4 + 0.6 * (i / top_n)})" for i in range(top_n, 0, -1)]
    
    fig = go.Figure(go.Bar(
        x=top_importances[::-1],
        y=top_features[::-1],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.4f}" for v in top_importances[::-1]],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"<b>Top {top_n} Feature Importances - {model_name}</b>",
        title_font_size=16,
        xaxis_title="Importance Score",
        template=CHART_TEMPLATE,
        height=max(350, top_n * 28),
        margin=dict(l=180),
    )
    return fig


def plot_metrics_radar(results: dict) -> go.Figure:
    """Radar chart comparing models across all metrics."""
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    
    fig = go.Figure()
    for i, (name, res) in enumerate(results.items()):
        values = [res[m] for m in metrics]
        values_closed = values + [values[0]]
        labels_closed = metric_labels + [metric_labels[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            fill="toself",
            name=name,
            line_color=COLOR_SEQ[i % len(COLOR_SEQ)],
            fillcolor=COLOR_SEQ[i % len(COLOR_SEQ)],
            opacity=0.3,
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="<b>Model Performance Radar</b>",
        title_font_size=18,
        template=CHART_TEMPLATE,
        height=480,
        showlegend=True,
    )
    return fig


# --------------------------------------------------------------
# Prediction UI
# --------------------------------------------------------------

def plot_risk_gauge(prob_default: float) -> go.Figure:
    """Animated gauge chart showing credit risk probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob_default,
        number={"suffix": "%", "font": {"size": 36, "color": COLORS["text"]}},
        delta={"reference": 30, "increasing": {"color": COLORS["danger"]},
               "decreasing": {"color": COLORS["success"]}},
        title={"text": "<b>Default Risk %</b>", "font": {"size": 16, "color": COLORS["muted"]}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": COLORS["muted"]},
            "bar": {"color": COLORS["primary"]},
            "bgcolor": COLORS["bg_card"],
            "borderwidth": 2,
            "bordercolor": COLORS["muted"],
            "steps": [
                {"range": [0, 25],  "color": "#1a3a2a"},
                {"range": [25, 50], "color": "#3a3a1a"},
                {"range": [50, 75], "color": "#3a2010"},
                {"range": [75, 100],"color": "#3a1010"},
            ],
            "threshold": {
                "line": {"color": COLORS["secondary"], "width": 4},
                "thickness": 0.85,
                "value": prob_default,
            },
        },
    ))
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=300,
        margin=dict(t=80, b=20, l=20, r=20),
        paper_bgcolor=COLORS["bg_dark"],
    )
    return fig


def plot_confidence_bars(prob_creditworthy: float, prob_default: float) -> go.Figure:
    """Horizontal bar chart showing prediction confidence."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[prob_creditworthy],
        y=["Creditworthy"],
        orientation="h",
        marker_color=COLORS["good"],
        text=[f"{prob_creditworthy:.1f}%"],
        textposition="inside",
        name="Creditworthy",
    ))
    fig.add_trace(go.Bar(
        x=[prob_default],
        y=["Default Risk"],
        orientation="h",
        marker_color=COLORS["danger"],
        text=[f"{prob_default:.1f}%"],
        textposition="inside",
        name="Default Risk",
    ))
    fig.update_layout(
        barmode="overlay",
        title="<b>Prediction Confidence</b>",
        xaxis=dict(range=[0, 100], title="Probability (%)"),
        template=CHART_TEMPLATE,
        height=200,
        showlegend=False,
        margin=dict(t=50, b=40),
    )
    return fig
