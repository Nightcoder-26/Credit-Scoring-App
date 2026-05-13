"""
app.py
======
Main entry point for the Credit Scoring AI - Streamlit Dashboard.

Run with:
    streamlit run app.py
"""

import streamlit as st
import os
import sys

# -- Page Configuration ----------------------------------------
st.set_page_config(
    page_title="Credit Scoring AI | CodeAlpha",
    page_icon="[BANK]",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com",
        "Report a bug": None,
        "About": "Credit Scoring AI - CodeAlpha ML Internship Project",
    },
)

# -- Global CSS / Theme ----------------------------------------
st.markdown("""
<style>
/* -- Google Fonts -- */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* -- Root Variables -- */
:root {
    --primary:   #00F2FE;
    --secondary: #4FACFE;
    --success:   #00E676;
    --warning:   #FFD600;
    --danger:    #FF1744;
    --bg-dark:   #0A0E17;
    --glass:     rgba(255, 255, 255, 0.03);
    --border:    rgba(255, 255, 255, 0.1);
    --text:      #F8FAFC;
    --muted:     #94A3B8;
}

/* -- Global Font -- */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
    background-color: var(--bg-dark) !important;
    color: var(--text) !important;
}

/* -- Main App Background -- */
.stApp {
    background: radial-gradient(circle at 15% 50%, rgba(79, 172, 254, 0.1), transparent 25%),
                radial-gradient(circle at 85% 30%, rgba(0, 242, 254, 0.15), transparent 25%);
    background-color: var(--bg-dark);
}

/* -- Remove default padding -- */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1200px;
}

/* -- Sidebar -- */
[data-testid="stSidebar"] {
    background: rgba(10, 14, 23, 0.7) !important;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-right: 1px solid var(--border);
}

/* -- Metric Cards (Glassmorphism) -- */
[data-testid="metric-container"] {
    background: var(--glass);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px 0 rgba(79, 172, 254, 0.2);
    border: 1px solid rgba(79, 172, 254, 0.3);
}
[data-testid="metric-container"] label {
    color: var(--muted) !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(to right, var(--primary), var(--secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* -- Buttons -- */
.stButton > button {
    background: linear-gradient(135deg, var(--secondary) 0%, var(--primary) 100%) !important;
    color: #0A0E17 !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(0, 242, 254, 0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 8px 25px rgba(0, 242, 254, 0.5) !important;
}

/* -- DataFrames -- */
[data-testid="stDataFrame"] {
    border-radius: 16px;
    border: 1px solid var(--border);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
}

/* -- Tabs -- */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: var(--glass);
    border-radius: 12px 12px 0 0;
    border: 1px solid var(--border);
    border-bottom: none;
    color: var(--muted);
    font-weight: 600;
    padding: 0.8rem 1.5rem;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(79, 172, 254, 0.2), rgba(0, 242, 254, 0.1)) !important;
    color: var(--primary) !important;
    border-top: 2px solid var(--primary) !important;
}

/* -- Hide headers and default sidebar nav -- */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebarNav"] { display: none !important; }

/* -- Animated gradient text -- */
.gradient-text {
    background: linear-gradient(-45deg, #00F2FE, #4FACFE, #00E676, #FFD600);
    background-size: 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: animated_text 5s ease-in-out infinite;
}

@keyframes animated_text {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
</style>
""", unsafe_allow_html=True)

# -- Add project root to path ----------------------------------
sys.path.insert(0, os.path.dirname(__file__))

# -- Import page modules ---------------------------------------
from pages import home, dataset_insights, eda, model_training, prediction, explainability, about

# -- Sidebar Navigation ----------------------------------------
with st.sidebar:
    # Logo / Title
    st.markdown("""
    <div style="text-align:center; padding: 1.2rem 0 1rem 0;">
        <div style="font-size:2.5rem;">[BANK]</div>
        <div style="font-weight:800; font-size:1.1rem; color:#FFFFFF; margin:0.3rem 0 0.1rem 0;">
            Credit Scoring AI
        </div>
        <div style="font-size:0.72rem; color:#9CA3AF;">CodeAlpha ML Internship</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation
    st.markdown(
        "<div style='font-size:0.7rem; color:#9CA3AF; text-transform:uppercase; "
        "letter-spacing:0.08em; margin-bottom:0.5rem;'>Navigation</div>",
        unsafe_allow_html=True
    )
    
    pages = {
        "[HOME]  Home":               "home",
        "[DATA]  Dataset Insights":   "dataset",
        "[EDA]  EDA & Visualization": "eda",
        "[AI]  Model Training":     "training",
        "[PREDICT]  Live Prediction":    "prediction",
        "[SHAP]  Explainable AI":     "explainability",
        "[ABOUT]  About":              "about",
    }
    
    selected = st.radio(
        "nav",
        list(pages.keys()),
        label_visibility="collapsed",
        key="nav_radio",
    )
    
    st.markdown("---")
    
    # Model Status
    model_exists = os.path.exists("credit_model.pkl")
    if model_exists:
        import joblib
        try:
            bundle = joblib.load("credit_model.pkl")
            mname = bundle.get("model_name", "Unknown")
            st.markdown(f"""
            <div style="background:#0d2a1e; border:1px solid #43D9AD; border-radius:10px; 
                        padding:0.8rem; margin-bottom:0.5rem;">
                <div style="color:#43D9AD; font-size:0.8rem; font-weight:600;">[OK] Model Ready</div>
                <div style="color:#9CA3AF; font-size:0.72rem; margin-top:0.2rem;">{mname}</div>
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            st.markdown("""
            <div style="background:#2a1010; border:1px solid #FF6B6B; border-radius:10px; 
                        padding:0.8rem; margin-bottom:0.5rem;">
                <div style="color:#FF6B6B; font-size:0.8rem; font-weight:600;">[WARN] Model Corrupted</div>
                <div style="color:#9CA3AF; font-size:0.72rem; margin-top:0.2rem;">Re-train the model</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#2a1a08; border:1px solid #FFB347; border-radius:10px; 
                    padding:0.8rem; margin-bottom:0.5rem;">
            <div style="color:#FFB347; font-size:0.8rem; font-weight:600;">[WARN] No Model Found</div>
            <div style="color:#9CA3AF; font-size:0.72rem; margin-top:0.2rem;">
                Go to Model Training or run:<br>
                <code style="color:#FFB347;">python train_model.py</code>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick stats sidebar
    st.markdown("""
    <div style="font-size:0.7rem; color:#9CA3AF; text-transform:uppercase; 
                letter-spacing:0.08em; margin-bottom:0.5rem;">Quick Reference</div>
    """, unsafe_allow_html=True)
    
    quick_stats = [
        ("[SAMPLE]", "Dataset",   "German Credit (UCI)"),
        (" ", "Samples",   "1,000"),
        (" ", "Features",  "20 + 4 derived"),
        ("[TARGET]", "Target",    "Creditworthy / Default"),
        (" ", "Models",    "5 classifiers"),
    ]
    for icon, label, val in quick_stats:
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center;
                    padding:0.3rem 0; border-bottom:1px solid #2D3148;">
            <span style="color:#9CA3AF; font-size:0.78rem;">{icon} {label}</span>
            <span style="color:#FFFFFF; font-size:0.78rem; font-weight:500;">{val}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; color:#9CA3AF; font-size:0.7rem;">
        v1.0.0 &nbsp;|&nbsp; Python   Streamlit<br>
        <a href="https://github.com" style="color:#6C63FF;">GitHub</a> &nbsp;|&nbsp;
        <a href="https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)" 
           style="color:#6C63FF;">Dataset</a>
    </div>
    """, unsafe_allow_html=True)

# -- Page Router -----------------------------------------------
page_key = pages[selected]

if page_key == "home":
    home.render()
elif page_key == "dataset":
    dataset_insights.render()
elif page_key == "eda":
    eda.render()
elif page_key == "training":
    model_training.render()
elif page_key == "prediction":
    prediction.render()
elif page_key == "explainability":
    explainability.render()
elif page_key == "about":
    about.render()
