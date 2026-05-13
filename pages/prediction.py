"""
pages/prediction.py
--------------------
Live credit risk prediction page with animated gauge, confidence bars,
risk breakdown, and downloadable PDF/CSV report.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, os
import json
from datetime import datetime
from io import StringIO, BytesIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.model_trainer import load_model_bundle, predict_credit_risk
from utils.visualizations import plot_risk_gauge, plot_confidence_bars


def _load_bundle():
    """Try loading saved model; returns None if not found."""
    try:
        return load_model_bundle("credit_model.pkl")
    except FileNotFoundError:
        return None


def _encode_input(raw_inputs: dict, bundle: dict) -> dict:
    """
    Converts human-readable form values into model-ready numeric values
    matching the feature space of the trained model.
    """
    checking_map   = {"< 0 DM": 0, "0-200 DM": 1, "> 200 DM": 2, "No Account": 3}
    credit_hist_map= {"No credits taken": 0, "All credits paid duly": 1,
                      "Existing credits paid duly": 2, "Delay in past": 3, "Critical account": 4}
    purpose_map    = {"Car (new)": 0, "Car (used)": 1, "Furniture/equipment": 2,
                      "Radio/TV": 3, "Education": 4, "Business": 5, "Others": 6}
    savings_map    = {"< 100 DM": 0, "100-500 DM": 1, "500-1000 DM": 2,
                      "> 1000 DM": 3, "Unknown/No savings": 4}
    employment_map = {"Unemployed": 0, "< 1 year": 1, "1-4 years": 2,
                      "4-7 years": 3, "> 7 years": 4}
    housing_map    = {"Rent": 0, "Own": 1, "Free": 2}
    job_map        = {"Unskilled non-resident": 0, "Unskilled resident": 1,
                      "Skilled": 2, "Highly skilled": 3}
    other_inst_map = {"Bank": 0, "Stores": 1, "None": 2}
    personal_map   = {"Male single": 0, "Female": 1, "Male married": 2, "Male divorced": 3}
    property_map   = {"Real estate": 0, "Savings/insurance": 1, "Car/other": 2, "Unknown": 3}
    other_debt_map = {"None": 0, "Co-applicant": 1, "Guarantor": 2}
    tel_map        = {"No": 0, "Yes": 1}
    foreign_map    = {"Yes": 0, "No": 1}

    age = raw_inputs["age"]
    credit_amount = raw_inputs["credit_amount"]
    duration = raw_inputs["duration"]
    install_rate = raw_inputs["installment_rate"]

    encoded = {
        "checking_account":   checking_map.get(raw_inputs.get("checking_account", "No Account"), 3),
        "duration":           duration,
        "credit_history":     credit_hist_map.get(raw_inputs.get("credit_history", "Existing credits paid duly"), 2),
        "purpose":            purpose_map.get(raw_inputs.get("purpose", "Others"), 6),
        "credit_amount":      credit_amount,
        "savings_account":    savings_map.get(raw_inputs.get("savings_account", "Unknown/No savings"), 4),
        "employment":         employment_map.get(raw_inputs.get("employment", "1-4 years"), 2),
        "installment_rate":   install_rate,
        "personal_status":    personal_map.get(raw_inputs.get("personal_status", "Male single"), 0),
        "other_debtors":      other_debt_map.get(raw_inputs.get("other_debtors", "None"), 0),
        "residence_since":    raw_inputs.get("residence_since", 2),
        "property":           property_map.get(raw_inputs.get("property", "Real estate"), 0),
        "age":                age,
        "other_installments": other_inst_map.get(raw_inputs.get("other_installments", "None"), 2),
        "housing":            housing_map.get(raw_inputs.get("housing", "Own"), 1),
        "existing_credits":   raw_inputs.get("existing_credits", 1),
        "job":                job_map.get(raw_inputs.get("job", "Skilled"), 2),
        "num_dependents":     raw_inputs.get("num_dependents", 1),
        "telephone":          tel_map.get(raw_inputs.get("telephone", "No"), 0),
        "foreign_worker":     foreign_map.get(raw_inputs.get("foreign_worker", "No"), 1),
        # Derived features
        "debt_duration_ratio":  credit_amount / (duration + 1),
        "installment_burden":   install_rate * credit_amount / 1000,
        "age_group":            2 if 35 <= age < 50 else (1 if 25 <= age < 35 else (3 if 50 <= age < 65 else 0)),
        "credit_tier":          0 if credit_amount <= 1000 else (1 if credit_amount <= 5000 else (2 if credit_amount <= 10000 else 3)),
    }
    return encoded


def _generate_csv_report(raw_inputs: dict, result: dict) -> bytes:
    """Generate a CSV prediction report."""
    report = {**raw_inputs, **{
        "prediction":        result["label"],
        "risk_level":        result["risk_level"],
        "prob_creditworthy": f"{result['prob_creditworthy']:.1f}%",
        "prob_default":      f"{result['prob_default']:.1f}%",
        "confidence":        f"{result['confidence']:.1f}%",
        "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }}
    buf = StringIO()
    pd.DataFrame([report]).to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def render():
    st.markdown("## [PREDICT] Live Credit Risk Prediction")
    st.markdown("Enter applicant details below to get an **instant AI-powered credit risk assessment**.")

    # -- Model check -------------------------------------------
    bundle = _load_bundle()
    if bundle is None:
        st.error("""
        [WARN] **No trained model found.**
        
        Please run one of the following first:
        ```bash
        python train_model.py
        ```
        Or go to the **[AI] Model Training** page and click **Train All Models   Save Model**.
        """)
        return

    model_name = bundle.get("model_name", "Unknown Model")
    st.success(f"[OK] Model loaded: **{model_name}**")

    st.markdown("---")

    # -- Input Form --------------------------------------------
    with st.form("prediction_form"):
        st.markdown("###   Applicant Information")
        
        # Row 1: Core Demographics
        st.markdown("####   Personal Details")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age (years)", 18, 80, 35, help="Applicant's age")
        with c2:
            personal_status = st.selectbox(
                "Personal Status",
                ["Male single", "Female", "Male married", "Male divorced"]
            )
        with c3:
            num_dependents = st.selectbox("Number of Dependents", [1, 2], index=0)

        # Row 2: Employment
        st.markdown("####   Employment & Housing")
        c1, c2, c3 = st.columns(3)
        with c1:
            employment = st.selectbox(
                "Employment Duration",
                ["Unemployed", "< 1 year", "1-4 years", "4-7 years", "> 7 years"],
                index=2
            )
        with c2:
            housing = st.selectbox("Housing Status", ["Rent", "Own", "Free"], index=1)
        with c3:
            job = st.selectbox(
                "Job Category",
                ["Unskilled non-resident", "Unskilled resident", "Skilled", "Highly skilled"],
                index=2
            )
        
        # Row 3: Financial
        st.markdown("####   Financial Details")
        c1, c2, c3 = st.columns(3)
        with c1:
            credit_amount = st.number_input(
                "Credit Amount (DM)", min_value=100, max_value=20000, value=3000, step=100
            )
        with c2:
            duration = st.slider("Loan Duration (months)", 4, 72, 24)
        with c3:
            installment_rate = st.selectbox(
                "Installment Rate (% of income)", [1, 2, 3, 4], index=1,
                help="Monthly payment as % of disposable income"
            )

        # Row 4: Accounts & History
        st.markdown("#### [BANK] Banking & Credit History")
        c1, c2, c3 = st.columns(3)
        with c1:
            checking_account = st.selectbox(
                "Checking Account Status",
                ["< 0 DM", "0-200 DM", "> 200 DM", "No Account"],
                index=3
            )
        with c2:
            savings_account = st.selectbox(
                "Savings Account",
                ["< 100 DM", "100-500 DM", "500-1000 DM", "> 1000 DM", "Unknown/No savings"],
                index=4
            )
        with c3:
            credit_history = st.selectbox(
                "Credit History",
                ["No credits taken", "All credits paid duly", "Existing credits paid duly",
                 "Delay in past", "Critical account"],
                index=2
            )

        # Row 5: Additional
        st.markdown("#### [SAMPLE] Additional Details")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            purpose = st.selectbox(
                "Loan Purpose",
                ["Car (new)", "Car (used)", "Furniture/equipment", "Radio/TV",
                 "Education", "Business", "Others"],
                index=3
            )
        with c2:
            existing_credits = st.selectbox("Existing Credits at Bank", [1, 2, 3, 4], index=0)
        with c3:
            other_installments = st.selectbox("Other Installment Plans", ["None", "Bank", "Stores"], index=0)
        with c4:
            telephone = st.selectbox("Telephone Registered", ["No", "Yes"])

        c1, c2, c3 = st.columns(3)
        with c1:
            property_type = st.selectbox(
                "Property Owned",
                ["Real estate", "Savings/insurance", "Car/other", "Unknown"],
                index=0
            )
        with c2:
            other_debtors = st.selectbox("Other Debtors/Guarantors", ["None", "Co-applicant", "Guarantor"], index=0)
        with c3:
            residence_since = st.selectbox("Residence Duration (years)", [1, 2, 3, 4], index=1)
        
        foreign_worker = st.selectbox("Foreign Worker", ["No", "Yes"], index=0)

        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.form_submit_button("[PREDICT] Predict Credit Risk", type="primary", use_container_width=True)

    # -- Prediction --------------------------------------------
    if submit:
        raw_inputs = {
            "age":               age,
            "personal_status":   personal_status,
            "num_dependents":    num_dependents,
            "employment":        employment,
            "housing":           housing,
            "job":               job,
            "credit_amount":     credit_amount,
            "duration":          duration,
            "installment_rate":  installment_rate,
            "checking_account":  checking_account,
            "savings_account":   savings_account,
            "credit_history":    credit_history,
            "purpose":           purpose,
            "existing_credits":  existing_credits,
            "other_installments":other_installments,
            "telephone":         telephone,
            "property":          property_type,
            "other_debtors":     other_debtors,
            "residence_since":   residence_since,
            "foreign_worker":    foreign_worker,
        }

        encoded = _encode_input(raw_inputs, bundle)
        result  = predict_credit_risk(bundle, encoded)

        st.markdown("---")
        st.markdown("## [TARGET] Prediction Result")

        # -- Big Result Banner ---------------------------------
        color = "#43D9AD" if result["prediction"] == 1 else "#FF6B6B"
        bg    = "#0d2a1e" if result["prediction"] == 1 else "#2a0d0d"
        icon  = "[OK]" if result["prediction"] == 1 else " "
        
        st.markdown(f"""
        <div style="background:{bg}; border:2px solid {color}; border-radius:16px;
                    padding:1.8rem 2rem; text-align:center; margin-bottom:1.5rem;">
            <div style="font-size:3rem;">{icon}</div>
            <div style="font-size:2rem; font-weight:800; color:{color}; margin:0.3rem 0;">
                {result["label"]}
            </div>
            <div style="font-size:1rem; color:#9CA3AF;">
                Risk Level: <b style="color:{color};">{result["risk_level"]}</b> &nbsp;|&nbsp;
                Confidence: <b style="color:#FFFFFF;">{result["confidence"]:.1f}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # -- Gauge + Confidence Bars ---------------------------
        col1, col2 = st.columns(2)
        with col1:
            fig = plot_risk_gauge(result["prob_default"])
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = plot_confidence_bars(result["prob_creditworthy"], result["prob_default"])
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk breakdown table
            st.markdown("#### [DATA] Risk Breakdown")
            breakdown = {
                "Metric": ["Default Probability", "Creditworthy Probability", "Risk Level", "Confidence"],
                "Value":  [
                    f"{result['prob_default']:.1f}%",
                    f"{result['prob_creditworthy']:.1f}%",
                    result["risk_level"],
                    f"{result['confidence']:.1f}%",
                ]
            }
            st.dataframe(pd.DataFrame(breakdown), hide_index=True, use_container_width=True)

        # -- Applicant Summary ---------------------------------
        st.markdown("#### [SAMPLE] Applicant Summary")
        summary_cols = {
            "Age": age, "Employment": employment, "Housing": housing,
            "Credit Amount (DM)": f"{credit_amount:,}", "Duration": f"{duration} months",
            "Checking Account": checking_account, "Savings Account": savings_account,
            "Credit History": credit_history, "Purpose": purpose,
        }
        c1, c2, c3 = st.columns(3)
        items = list(summary_cols.items())
        for i, (k, v) in enumerate(items):
            [c1, c2, c3][i % 3].markdown(f"""
            <div style="background:#1A1D2E; border-radius:8px; padding:0.6rem 0.8rem; margin-bottom:0.5rem;">
                <div style="font-size:0.72rem; color:#9CA3AF;">{k}</div>
                <div style="font-weight:600; color:#FFFFFF;">{v}</div>
            </div>
            """, unsafe_allow_html=True)

        # -- Advisory Notes ------------------------------------
        st.markdown("#### [TIP] Advisory Notes")
        if result["prediction"] == 1:
            st.success("""
            [OK] **This applicant appears creditworthy.**
            - Strong financial profile with acceptable risk indicators.
            - Recommended for standard loan approval with routine verification.
            - Consider offering competitive rates based on credit history.
            """)
        else:
            st.error("""
              **This applicant shows elevated credit risk.**
            - Financial indicators suggest potential repayment difficulties.
            - Recommend additional income verification and collateral assessment.
            - Consider conditional approval with higher interest rate or reduced loan amount.
            """)

        # -- Download Report -----------------------------------
        st.markdown("---")
        st.markdown("###   Download Report")
        csv_report = _generate_csv_report(raw_inputs, result)
        
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                label="  Download CSV Report",
                data=csv_report,
                file_name=f"credit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_d2:
            json_data = json.dumps({**raw_inputs, **result}, indent=2, default=str)
            st.download_button(
                label="[SAMPLE] Download JSON Report",
                data=json_data.encode("utf-8"),
                file_name=f"credit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )
        
        st.caption("[WARN] This prediction is AI-generated and should be reviewed by a qualified credit analyst before making final decisions.")
