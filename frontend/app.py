import os
import io
import json
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="RiskLens AI | Financial Risk & Compliance Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise API Endpoint Configuration
API_URL = os.getenv("RISKLENS_API_URL", "https://risklens-ai-2-t0zj.onrender.com").rstrip("/")

def api_call(method: str, endpoint: str, json_data: dict = None, files: dict = None, timeout: int = 45):
    """Centralized HTTP API client for communication with FastAPI backend."""
    url = f"{API_URL}{endpoint}"
    try:
        if method.upper() == "GET":
            resp = requests.get(url, timeout=timeout)
        elif method.upper() == "POST":
            if files:
                resp = requests.post(url, files=files, timeout=timeout)
            else:
                resp = requests.post(url, json=json_data, timeout=timeout)
        else:
            return None, f"Unsupported HTTP method {method}"

        if resp.status_code == 200:
            return resp.json(), None
        else:
            try:
                err_detail = resp.json().get("detail", resp.text)
            except Exception:
                err_detail = resp.text
            return None, f"API Error ({resp.status_code}): {err_detail}"
    except requests.exceptions.RequestException as e:
        return None, "🚨 RiskLens backend is temporarily unavailable. Please try again later."

# Custom Enterprise Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 1.2rem;
        font-weight: 400;
    }
    .workflow-card {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-left: 5px solid #0284C7;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 22px;
    }
    .workflow-diagram {
        font-size: 0.92rem;
        font-weight: 700;
        color: #0F172A;
        font-family: monospace;
        white-space: pre-wrap;
    }
    .workflow-caption {
        font-size: 0.85rem;
        color: #475569;
        margin-top: 6px;
    }
    .metric-box {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        text-align: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 14px;
        border-radius: 16px;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .badge-medium {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 14px;
        border-radius: 16px;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .badge-low {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 4px 14px;
        border-radius: 16px;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .review-alert-banner {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        color: #991B1B;
        padding: 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 16px;
    }
    .context-box {
        background-color: #F0F9FF;
        border: 1px solid #BAE6FD;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 16px;
        font-size: 0.9rem;
        color: #0369A1;
    }
</style>
""", unsafe_allow_html=True)

# --- Friendly Feature Name Mapping ---
FRIENDLY_FEATURE_MAP = {
    "failed_login_attempts": "Multiple Failed Login Attempts",
    "device_risk_score": "High Device Risk",
    "credit_utilization": "High Credit Utilization",
    "debt_to_income": "High Debt-to-Income Ratio",
    "transaction_amount": "Large Transaction Amount",
    "account_age_days": "Account Age",
    "time_pattern": "Night Transaction Pattern",
    "is_night_transaction": "Late Night Transaction",
    "avg_monthly_income": "Monthly Income Level",
    "interest_rate": "Interest Rate Level",
    "transaction_frequency": "High Transaction Frequency",
    "merchant_category_Crypto_Exchange": "Crypto Merchant Category",
    "merchant_category_Wire_Transfer": "Wire Transfer Merchant Category",
    "merchant_category_Gambling": "Gambling Merchant Category",
    "location_International_HighRisk": "High-Risk International Location",
    "location_Online_Unverified": "Unverified Online Location",
    "ae_anomaly_score": "Deep Behavior Anomaly"
}

# Session State Setup
if 'has_submitted_data' not in st.session_state:
    st.session_state.has_submitted_data = False
if 'active_tx_payload' not in st.session_state:
    st.session_state.active_tx_payload = None
if 'active_cust_id' not in st.session_state:
    st.session_state.active_cust_id = None
if 'active_cust_name' not in st.session_state:
    st.session_state.active_cust_name = None
if 'uploaded_batch_df' not in st.session_state:
    st.session_state.uploaded_batch_df = None
if 'active_prediction' not in st.session_state:
    st.session_state.active_prediction = None
if 'active_explanation' not in st.session_state:
    st.session_state.active_explanation = None

# --- TOP PERSISTENT BRANDING & WORKFLOW ---
st.markdown("<div class='main-title'>RiskLens AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Explainable Financial Risk & Compliance Intelligence Platform</div>", unsafe_allow_html=True)

st.markdown("""
<div class='workflow-card'>
    <div class='workflow-diagram'>
CUSTOMER TRANSACTION ──┐
                       ├──► ML + AUTOENCODER ──► RISK & ANOMALY ──► SHAP EXPLANATION ──┐
BANK POLICY PDF/TXT  ──┘                                                                ├──► EVIDENCE-BASED DECISION ──► HUMAN REVIEW
FINANCIAL NEWS TEXT  ────────────────────────────────────────► FinBERT CONTEXT ─────────┘
    </div>
    <div class='workflow-caption'>
        <b>Parallel Input Architecture:</b> Customer transaction data drives ML risk prediction & PyTorch anomaly scoring. Uploaded bank policies provide grounded RAG evidence to justify compliance decisions.
    </div>
</div>
""", unsafe_allow_html=True)

# Persistent Top Navigation Tabs
nav_tabs = st.tabs([
    "Risk Assessment",
    "Model Performance",
    "Policy Evidence",
    "Financial News",
    "What-If Analysis",
    "Human Review"
])

# --- SIDEBAR INPUT CONTROL LAYER ---
st.sidebar.image("https://img.icons8.com/isometric/100/search.png", width=50)
st.sidebar.title("RiskLens AI")
st.sidebar.caption("Parallel Workflow Inputs")

st.sidebar.markdown("### 1. Customer & Transaction Data")
tx_input_mode = st.sidebar.radio("Transaction Input Method", ["Manual Entry", "Upload CSV File"])

# Optional Sample Preset Selector
with st.sidebar.expander("🔬 Load Sample Demo Preset"):
    preset_choice = st.selectbox(
        "Select Sample Customer Profile",
        ["None (Custom Entry)", "Sample CUST_0042 (High Risk)", "Sample CUST_0018 (Medium Suspicious)", "Sample CUST_0005 (Low Risk)"]
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### 2. Policy & Evidence Source")
st.sidebar.caption("Upload bank policies, risk guidelines, or annual reports.")
policy_doc_file = st.sidebar.file_uploader("Upload Policy (PDF/TXT)", type=["pdf", "txt"])

if policy_doc_file is not None:
    doc_res, err = api_call(
        "POST", 
        "/upload-document", 
        files={"file": (policy_doc_file.name, policy_doc_file.read())}
    )
    if err:
        st.sidebar.error(err)
    else:
        st.sidebar.success(f"Indexed {doc_res.get('chunks_indexed', 0)} passages from '{policy_doc_file.name}'.")

# Global Navigation Helpers
def reset_to_input():
    st.session_state.has_submitted_data = False
    st.session_state.active_tx_payload = None
    st.session_state.active_cust_id = None
    st.session_state.active_cust_name = None
    st.session_state.uploaded_batch_df = None
    st.session_state.active_prediction = None
    st.session_state.active_explanation = None

# --- TAB 1: RISK ASSESSMENT (HOME / MAIN DASHBOARD) ---
with nav_tabs[0]:
    st.subheader("Risk Assessment & Customer Intelligence")
    
    # Render Input Form Section
    st.markdown("### Customer & Transaction Data Input")
    
    if tx_input_mode == "Manual Entry":
        # Handle Sample Preset Pre-population
        if preset_choice == "Sample CUST_0042 (High Risk)":
            def_id, def_name = "CUST_0042", "Acme International"
            def_amt, def_freq, def_cat, def_loc = 14500.0, 24, "Wire_Transfer", "International_HighRisk"
            def_logins, def_device, def_income, def_dti, def_ir, def_util, def_age, def_night = 5, 0.92, 6000.0, 0.75, 12.5, 0.88, 120, True
        elif preset_choice == "Sample CUST_0018 (Medium Suspicious)":
            def_id, def_name = "CUST_0018", "Apex Tech Partners"
            def_amt, def_freq, def_cat, def_loc = 3200.0, 9, "Crypto_Exchange", "Online_Unverified"
            def_logins, def_device, def_income, def_dti, def_ir, def_util, def_age, def_night = 2, 0.55, 7500.0, 0.48, 8.5, 0.65, 365, False
        elif preset_choice == "Sample CUST_0005 (Low Risk)":
            def_id, def_name = "CUST_0005", "Standard Retail Inc"
            def_amt, def_freq, def_cat, def_loc = 120.0, 2, "Grocery", "Domestic"
            def_logins, def_device, def_income, def_dti, def_ir, def_util, def_age, def_night = 0, 0.12, 9500.0, 0.20, 4.5, 0.18, 730, False
        else:
            def_id, def_name = "", ""
            def_amt, def_freq, def_cat, def_loc = 4500.0, 12, "Crypto_Exchange", "International_HighRisk"
            def_logins, def_device, def_income, def_dti, def_ir, def_util, def_age, def_night = 4, 0.85, 6000.0, 0.65, 8.5, 0.85, 180, True

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            cust_id_input = st.text_input("Customer ID", value=def_id, placeholder="e.g. CUST_0042")
        with col_m2:
            cust_name_input = st.text_input("Customer Name", value=def_name, placeholder="e.g. Example Customer")

        st.markdown("---")
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            st.markdown("##### Transaction Attributes")
            in_amt = st.number_input("Transaction Amount ($)", value=def_amt, min_value=1.0, step=100.0)
            in_freq = st.slider("Transaction Frequency (24h)", 1, 50, def_freq)
            in_cat = st.selectbox(
                "Merchant Category",
                ['Grocery', 'Utilities', 'Electronics', 'Luxury_Goods', 'Crypto_Exchange', 'Wire_Transfer', 'Gambling', 'Travel'],
                index=['Grocery', 'Utilities', 'Electronics', 'Luxury_Goods', 'Crypto_Exchange', 'Wire_Transfer', 'Gambling', 'Travel'].index(def_cat)
            )
            in_loc = st.selectbox(
                "Transaction Location",
                ['Domestic', 'International_LowRisk', 'International_HighRisk', 'Online_Unverified'],
                index=['Domestic', 'International_LowRisk', 'International_HighRisk', 'Online_Unverified'].index(def_loc)
            )
            in_night = st.checkbox("Night Transaction (22:00-06:00)", value=def_night)

        with col_f2:
            st.markdown("##### Security & Device Signals")
            in_logins = st.slider("Failed Login Attempts", 0, 10, def_logins)
            in_device = st.slider("Device Risk Score", 0.0, 1.0, def_device, 0.05)
            in_age = st.number_input("Account Age (Days)", value=def_age, min_value=1, step=30)

        with col_f3:
            st.markdown("##### Financial Metrics")
            in_income = st.number_input("Average Monthly Income ($)", value=def_income, min_value=500.0, step=500.0)
            in_dti = st.slider("Debt-to-Income Ratio", 0.05, 0.95, def_dti, 0.05)
            in_util = st.slider("Credit Utilization Ratio", 0.05, 0.99, def_util, 0.05)
            in_ir = st.slider("Interest Rate (%)", 3.0, 20.0, def_ir, 0.5)

        st.write("")
        btn_submit_manual = st.button("Submit Customer Risk Assessment", type="primary")

        if btn_submit_manual:
            c_id = cust_id_input.strip() if cust_id_input.strip() else "CUST_0042"
            c_name = cust_name_input.strip() if cust_name_input.strip() else "Example Customer"
            
            payload = {
                "customer_id": c_id,
                "customer_name": c_name,
                "transaction_amount": in_amt,
                "transaction_frequency": in_freq,
                "merchant_category": in_cat,
                "location": in_loc,
                "time_pattern": 2 if in_night else 14,
                "is_night_transaction": 1 if in_night else 0,
                "is_weekend": 0,
                "account_age_days": in_age,
                "avg_monthly_income": in_income,
                "debt_to_income": in_dti,
                "interest_rate": in_ir,
                "credit_utilization": in_util,
                "failed_login_attempts": in_logins,
                "device_risk_score": in_device
            }
            
            with st.spinner("Analyzing risk with RiskLens backend..."):
                pred_res, p_err = api_call("POST", "/predict-risk", json_data=payload)
                expl_res, e_err = api_call("POST", "/explain-risk", json_data=payload)

            if p_err or e_err:
                st.error(p_err or e_err)
            else:
                st.session_state.active_cust_id = c_id
                st.session_state.active_cust_name = c_name
                st.session_state.active_tx_payload = payload
                st.session_state.active_prediction = pred_res
                st.session_state.active_explanation = expl_res
                st.session_state.has_submitted_data = True

    else:
        st.caption("Upload a CSV file containing customer transaction records.")
        csv_file = st.file_uploader("Upload Transactions CSV", type=["csv"])
        
        if csv_file is not None:
            with st.spinner("Processing batch CSV with RiskLens backend..."):
                batch_res, b_err = api_call(
                    "POST", 
                    "/upload-transactions-csv", 
                    files={"file": (csv_file.name, csv_file.read())}
                )
            if b_err:
                st.error(b_err)
            else:
                predictions = batch_res.get("predictions", [])
                df_scored = pd.DataFrame(predictions)
                st.session_state.uploaded_batch_df = df_scored
                
                st.success(f"CSV Processed: {batch_res.get('message', 'Success')}")
                st.markdown("#### Transaction Batch Risk Summary")
                b_c1, b_c2, b_c3, b_c4 = st.columns(4)
                b_c1.metric("Total Transactions", batch_res.get("total_transactions", len(df_scored)))
                b_c2.metric("High Risk Count", batch_res.get("high_risk_count", 0))
                b_c3.metric("Medium Risk Count", int((df_scored["risk_level"] == "MEDIUM").sum()) if not df_scored.empty else 0)
                b_c4.metric("Average Risk Score", f"{df_scored['risk_score'].mean():.2%}" if not df_scored.empty else "0.0%")
                
                if not df_scored.empty:
                    display_table = df_scored[['customer_id', 'customer_name', 'transaction_amount', 'risk_score', 'risk_level']].rename(columns={
                        'customer_id': 'Customer ID',
                        'customer_name': 'Customer Name',
                        'transaction_amount': 'Transaction Amount ($)',
                        'risk_score': 'Risk Score',
                        'risk_level': 'Risk Level'
                    })
                    st.dataframe(display_table)
                    
                    st.markdown("#### Select Customer for Full Pipeline Evaluation")
                    selected_row = st.selectbox(
                        "Select Customer Transaction",
                        options=list(range(len(df_scored))),
                        format_func=lambda idx: f"{df_scored.iloc[idx]['customer_id']} ({df_scored.iloc[idx]['customer_name']}) - ${df_scored.iloc[idx]['transaction_amount']:,.2f} [Risk: {df_scored.iloc[idx]['risk_level']}]"
                    )
                    
                    if st.button("Inspect Selected Customer Pipeline", type="primary"):
                        selected_tx = df_scored.iloc[selected_row].to_dict()
                        c_id = str(selected_tx.get('customer_id', 'CUST_0042'))
                        c_name = str(selected_tx.get('customer_name', 'Anonymous Customer'))
                        
                        with st.spinner("Fetching customer pipeline telemetry..."):
                            pred_res, p_err = api_call("POST", "/predict-risk", json_data=selected_tx)
                            expl_res, e_err = api_call("POST", "/explain-risk", json_data=selected_tx)
                            
                        if p_err or e_err:
                            st.error(p_err or e_err)
                        else:
                            st.session_state.active_cust_id = c_id
                            st.session_state.active_cust_name = c_name
                            st.session_state.active_tx_payload = selected_tx
                            st.session_state.active_prediction = pred_res
                            st.session_state.active_explanation = expl_res
                            st.session_state.has_submitted_data = True

    # Display Initial Neutral State if No Customer Data Submitted
    if not st.session_state.has_submitted_data:
        st.write("")
        st.info("ℹ️ **No customer data submitted.** Enter customer transaction details above or upload a CSV file, then click **'Submit Customer Risk Assessment'** to view risk score and explanations.")

    # Render Customer Risk Results when Submitted
    if st.session_state.has_submitted_data and st.session_state.active_tx_payload is not None:
        st.markdown("---")
        
        # Navigation Bar
        col_n1, col_n2 = st.columns([1, 4])
        with col_n1:
            if st.button("← Start New Analysis"):
                reset_to_input()
                st.rerun()

        tx_payload = st.session_state.active_tx_payload
        c_id = st.session_state.active_cust_id or "CUST_0042"
        c_name = st.session_state.active_cust_name or "Anonymous Customer"

        prediction = st.session_state.active_prediction
        explanation = st.session_state.active_explanation

        if prediction and explanation:
            requires_review = prediction.get('requires_human_review', (
                prediction.get('risk_level') == "HIGH" or 
                prediction.get('anomaly_status') == "HIGHLY ANOMALOUS" or 
                prediction.get('combined_risk_level') == "HIGH"
            ))

            # Workflow Breadcrumb Indicator
            st.markdown("""
            <div style='background-color:#F1F5F9; padding:8px 14px; border-radius:6px; font-weight:700; font-size:0.85rem; color:#334155; margin-bottom:16px;'>
                INPUT ➔ RISK PREDICTION ➔ ANOMALY DETECTION ➔ EXPLANATION ➔ POLICY EVIDENCE ➔ HUMAN REVIEW
            </div>
            """, unsafe_allow_html=True)

            if requires_review:
                st.markdown(
                    "<div class='review-alert-banner'>🚨 REQUIRES HUMAN REVIEW<br/>"
                    f"<span style='font-size: 0.9rem; font-weight: 400;'>Customer {c_id} exhibits elevated risk ({int(prediction['risk_score']*100)}%) or deep anomaly signals requiring analyst decision.</span></div>",
                    unsafe_allow_html=True
                )

            # Customer Header Profile
            st.markdown(f"#### Customer Profile: **{c_id}** ({c_name})")

            # Risk Score Metric Cards
            r_col1, r_col2, r_col3 = st.columns(3)
            
            with r_col1:
                st.markdown("##### Supervised Risk Score")
                st.markdown(f"<div class='metric-box'><h2 style='color:#1E3A8A; margin:0;'>{int(prediction['risk_score']*100)}%</h2><p style='margin:0; font-weight:700;'>{prediction['risk_level']}</p></div>", unsafe_allow_html=True)

            with r_col2:
                st.markdown("##### Anomaly Status")
                st.markdown(f"<div class='metric-box'><h2 style='color:#7C3AED; margin:0;'>{int(prediction['anomaly_score']*100)}%</h2><p style='margin:0; font-weight:700;'>{prediction['anomaly_status']}</p></div>", unsafe_allow_html=True)

            with r_col3:
                st.markdown("##### Combined Risk Level")
                comb_level = prediction['combined_risk_level']
                b_class = "badge-high" if comb_level == "HIGH" else ("badge-medium" if comb_level == "MEDIUM" else "badge-low")
                st.markdown(f"<div class='metric-box'><h2 style='color:#0F172A; margin:0;'>{int(prediction['combined_risk_score']*100)}%</h2><p style='margin:4px;'><span class='{b_class}'>{comb_level} RISK</span></p></div>", unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("Why is this customer high risk?")
            st.caption("Key risk factors influencing the model's prediction.")

            factors = explanation.get('important_factors', [])
            s_col1, s_col2 = st.columns(2)

            with s_col1:
                st.markdown("##### Risk-Increasing Factors")
                inc_f = [f for f in factors if f.startswith("+")]
                if inc_f:
                    for f in inc_f:
                        clean_f = f.replace("+", "").strip()
                        st.error(f"• {clean_f}")
                else:
                    st.write("No major risk-increasing factors.")

            with s_col2:
                st.markdown("##### Risk-Reducing Factors")
                red_f = [f for f in factors if f.startswith("-")]
                if red_f:
                    for f in red_f:
                        clean_f = f.replace("-", "").strip()
                        st.success(f"• {clean_f}")
                else:
                    st.write("Standard account parameters.")

            # Clean SHAP Factor Chart
            shap_raw = dict(list(explanation.get('shap_values', {}).items())[:6])
            friendly_shap = {FRIENDLY_FEATURE_MAP.get(k, k.replace("_", " ").title()): v for k, v in shap_raw.items()}
            df_shap = pd.DataFrame({"Risk Factor": list(friendly_shap.keys()), "Impact": list(friendly_shap.values())}).sort_values("Impact", ascending=True)
            
            fig_shap = px.bar(df_shap, x="Impact", y="Risk Factor", orientation="h", title="SHAP Risk Factor Contribution Chart", color="Impact", color_continuous_scale=["#10B981", "#EF4444"])
            fig_shap.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_shap, use_container_width=True)

# --- TAB 2: MODEL PERFORMANCE ---
with nav_tabs[1]:
    st.subheader("Model Performance & Algorithm Evaluation")
    st.caption("Comparative evaluation metrics across risk classifiers and PyTorch Deep Autoencoder.")
    
    with st.spinner("Fetching model evaluation metrics from backend..."):
        eval_res, eval_err = api_call("GET", "/model-evaluation")
        
    if eval_err:
        st.error(eval_err)
    elif eval_res:
        all_m = eval_res.get("all_model_metrics", {})
        if all_m:
            rows = []
            for name, m in all_m.items():
                rows.append({
                    "Model Algorithm": name.replace("_", " ").title(),
                    "Accuracy": f"{m.get('accuracy', 0.0):.4f}",
                    "Precision": f"{m.get('precision', 0.0):.4f}",
                    "Recall": f"{m.get('recall', 0.0):.4f}",
                    "F1 Score": f"{m.get('f1_score', 0.0):.4f}",
                    "ROC-AUC": f"{m.get('roc_auc', 0.0):.4f}"
                })
            st.table(pd.DataFrame(rows))
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown("##### ROC-AUC Performance Curves")
                fig_roc = go.Figure()
                for name, m in all_m.items():
                    c = m.get("roc_curve", {})
                    if c:
                        fig_roc.add_trace(go.Scatter(x=c.get("fpr", []), y=c.get("tpr", []), mode='lines', name=f"{name.title()} ({m.get('roc_auc', 0):.2f})"))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name='Baseline'))
                fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=320)
                st.plotly_chart(fig_roc, use_container_width=True)
                
            with col_m2:
                st.markdown("##### Autoencoder Anomaly Summary")
                ae_m = all_m.get("autoencoder", all_m.get("AUTOENCODER", {}))
                st.json({
                    "Reconstruction Baseline Mean": f"{ae_m.get('reconstruction_mean', 0.5):.4f}",
                    "Reconstruction Baseline Std": f"{ae_m.get('reconstruction_std', 0.8):.4f}",
                    "Autoencoder Accuracy": f"{ae_m.get('accuracy', 0.0):.4f}",
                    "Autoencoder ROC-AUC": f"{ae_m.get('roc_auc', 0.0):.4f}"
                })

# --- TAB 3: POLICY EVIDENCE (CONTEXT-AWARE CUSTOM RAG) ---
with nav_tabs[2]:
    st.subheader("Policy Evidence Search (Custom Grounded RAG)")
    st.caption("Retrieve bank policies, compliance rules, and risk guidelines grounded in customer risk telemetry.")
    
    if st.session_state.has_submitted_data and st.session_state.active_tx_payload is not None:
        c_id = st.session_state.active_cust_id or "CUST_0042"
        c_name = st.session_state.active_cust_name or "Anonymous Customer"
        pred = st.session_state.active_prediction or {}
        expl = st.session_state.active_explanation or {}
        
        st.markdown(f"""
        <div class='context-box'>
            📋 <b>Active Customer Risk Context:</b> {c_id} ({c_name}) | <b>Risk Score:</b> {int(pred.get('risk_score', 0.0)*100)}% ({pred.get('risk_level', 'LOW')}) | <b>Anomaly Status:</b> {pred.get('anomaly_status', 'NORMAL')}
        </div>
        """, unsafe_allow_html=True)
        
        default_q = f"Why is Customer {c_id} considered high risk according to our bank policy?"
        customer_context = {
            "customer_id": c_id,
            "customer_name": c_name,
            "risk_score": pred.get('risk_score', 0.0),
            "risk_level": pred.get('risk_level', 'LOW'),
            "anomaly_status": pred.get('anomaly_status', 'NORMAL'),
            "important_factors": expl.get('important_factors', [])
        }
    else:
        st.info("ℹ️ No customer active. RAG queries will retrieve general policy evidence.")
        default_q = "What policy requirement was triggered for high-risk international transactions?"
        customer_context = None

    rag_q_input = st.text_input("Enter Policy Evidence Question", value=default_q)
    
    if st.button("Search Policy Evidence & Synthesize", type="primary"):
        with st.spinner("Retrieving grounded policy evidence from backend..."):
            rag_res, r_err = api_call(
                "POST", 
                "/retrieve-evidence", 
                json_data={
                    "query": rag_q_input,
                    "top_k": 4,
                    "transaction_context": customer_context
                }
            )
        if r_err:
            st.error(r_err)
        elif rag_res:
            st.markdown("#### Evidence-Based Policy Answer")
            st.markdown(rag_res.get('answer', ''))
            
            retrieved_docs = rag_res.get('evidence', [])
            if retrieved_docs:
                st.markdown("#### Policy Evidence Cards")
                for idx, ev in enumerate(retrieved_docs, 1):
                    doc_n = ev.get('document_name', 'Bank_Risk_Policy.txt')
                    page_n = ev.get('page_number', 1)
                    content_n = ev.get('content', '').strip()
                    sec_n = "Authentication" if "login" in content_n.lower() else ("Device Risk" if "device" in content_n.lower() else "Compliance Guidelines")
                    
                    st.markdown(f"""
                    <div style='background-color:#F8FAFC; border:1px solid #E2E8F0; border-left:4px solid #10B981; border-radius:6px; padding:12px; margin-bottom:12px;'>
                        <div style='font-weight:700; font-size:0.95rem; color:#0F172A;'>✓ Rule {idx} — {sec_n}</div>
                        <div style='font-size:0.9rem; color:#334155; margin:6px 0;'>{content_n}</div>
                        <div style='font-size:0.82rem; color:#64748B;'><b>Source:</b> {doc_n} (Page {page_n}) &nbsp;|&nbsp; <b>Section:</b> {sec_n}</div>
                    </div>
                    """, unsafe_allow_html=True)

# --- TAB 4: FINANCIAL NEWS ---
with nav_tabs[3]:
    st.subheader("Financial News Analysis (FinBERT NLP)")
    st.caption("Analyze external financial news articles and regulatory filings for sentiment & market impact.")
    
    news_text_in = st.text_area("Enter Financial News Text", value="Apex Financial Group faces quarterly loss as non-performing assets rise 18%.", height=90)
    if st.button("Analyze News Sentiment"):
        with st.spinner("Analyzing news sentiment with FinBERT backend..."):
            news_out, n_err = api_call("POST", "/analyze-news", json_data={"text": news_text_in})
            
        if n_err:
            st.error(n_err)
        elif news_out:
            n_c1, n_c2, n_c3 = st.columns(3)
            n_c1.metric("Sentiment", news_out.get('sentiment', 'Neutral'))
            n_c2.metric("Target Entity", news_out.get('entity', 'Company'))
            n_c3.metric("Risk Impact Level", news_out.get('impact', 'Low Risk'))

# --- TAB 5: WHAT-IF ANALYSIS ---
with nav_tabs[4]:
    st.subheader("What-If Analysis (Counterfactual Scenario Analysis)")
    st.caption("Change financial conditions to see how the estimated risk could change.")
    
    if st.session_state.has_submitted_data and st.session_state.active_tx_payload is not None:
        base_p = st.session_state.active_tx_payload
        c_id = st.session_state.active_cust_id or "CUST_0042"
    else:
        st.info("ℹ️ No customer data submitted. Using baseline reference profile.")
        base_p = {
            "transaction_amount": 4500.0, "transaction_frequency": 12, "merchant_category": "Crypto_Exchange",
            "location": "International_HighRisk", "time_pattern": 2, "is_night_transaction": 1, "is_weekend": 0,
            "account_age_days": 180, "avg_monthly_income": 6000.0, "debt_to_income": 0.65, "interest_rate": 8.5,
            "credit_utilization": 0.88, "failed_login_attempts": 4, "device_risk_score": 0.85
        }
        c_id = "Baseline Profile"

    st.markdown(f"##### Counterfactual Scenario Adjuster ({c_id})")
    w_col1, w_col2, w_col3, w_col4 = st.columns(4)
    
    with w_col1:
        scen_amt = st.number_input("Scenario Amount ($)", value=float(base_p.get('transaction_amount', 4500.0) * 1.5), step=500.0)
    with w_col2:
        scen_income = st.number_input("Scenario Income ($)", value=float(base_p.get('avg_monthly_income', 6000.0)), step=500.0)
    with w_col3:
        scen_ir = st.slider("Scenario Interest Rate (%)", 3.0, 22.0, float(base_p.get('interest_rate', 8.5) + 2.0), 0.5)
    with w_col4:
        scen_dti = st.slider("Scenario Debt-to-Income", 0.05, 0.95, float(base_p.get('debt_to_income', 0.65)), 0.05)

    if st.button("Run Counterfactual Simulation", type="primary"):
        with st.spinner("Simulating scenario deltas on backend..."):
            sim_out, s_err = api_call(
                "POST", 
                "/simulate", 
                json_data={
                    "base_transaction": base_p,
                    "modified_params": {
                        "transaction_amount": scen_amt,
                        "avg_monthly_income": scen_income,
                        "interest_rate": scen_ir,
                        "debt_to_income": scen_dti
                    }
                }
            )
        if s_err:
            st.error(s_err)
        elif sim_out:
            wc_1, wc_2, wc_3, wc_4 = st.columns(4)
            wc_1.metric("Current Risk Score", f"{int(sim_out['baseline']['risk_score']*100)}%")
            wc_2.metric("Scenario Risk Score", f"{int(sim_out['scenario']['risk_score']*100)}%")
            wc_3.metric("Risk Delta", f"{sim_out['delta']['percentage_change']:+.1f}%")
            wc_4.metric("Direction", sim_out['delta']['direction'])

            st.markdown("##### Impact Summary")
            st.info(sim_out.get('impact_summary', ''))

# --- TAB 6: HUMAN REVIEW ---
with nav_tabs[5]:
    st.subheader("Human Review Portal")
    st.caption("Human analyst case evaluation, policy evidence verification, and decision logging.")
    
    # Fetch Past Decision History from Backend
    with st.spinner("Fetching case decision history..."):
        rev_hist_res, h_err = api_call("GET", "/reviews")
    past_logs = rev_hist_res.get("reviews", []) if rev_hist_res else []
    
    if st.session_state.has_submitted_data and st.session_state.active_tx_payload is not None:
        c_id = st.session_state.active_cust_id or "CUST_0042"
        c_name = st.session_state.active_cust_name or "Anonymous Customer"
        tx_p = st.session_state.active_tx_payload
        pred = st.session_state.active_prediction or {}
        expl = st.session_state.active_explanation or {}
        
        col_hr1, col_hr2 = st.columns([1.5, 1])
        
        with col_hr1:
            st.markdown("##### Customer Case Details")
            st.json({
                "Customer ID": c_id,
                "Customer Name": c_name,
                "Risk Score": f"{int(pred.get('risk_score', 0.0)*100)}% ({pred.get('risk_level', 'LOW')})",
                "Anomaly Status": f"{int(pred.get('anomaly_score', 0.0)*100)}% ({pred.get('anomaly_status', 'NORMAL')})",
                "Combined Assessment": f"{pred.get('combined_risk_level', 'LOW')} RISK",
                "Top Risk Factors": [f.replace("+", "").strip() for f in expl.get('important_factors', []) if f.startswith("+")][:3]
            })
            
            st.markdown("##### Analyst Decision Form")
            with st.form("human_analyst_decision_form"):
                analyst_id_val = st.text_input("Analyst ID", value="Analyst_J_Doe")
                analyst_notes = st.text_area("Analyst Notes / Rationale", value="High risk score and failed login attempts violate standard policy limits.")
                
                b1, b2, b3 = st.columns(3)
                act_app = b1.form_submit_button("Approve")
                act_rej = b2.form_submit_button("Reject")
                act_req = b3.form_submit_button("Request Verification")
                
                d_choice = None
                if act_app:
                    d_choice = "APPROVE"
                elif act_rej:
                    d_choice = "REJECT"
                elif act_req:
                    d_choice = "REQUEST VERIFICATION"
                    
                if d_choice:
                    sub_res, sub_err = api_call(
                        "POST", 
                        "/review-transaction", 
                        json_data={
                            "transaction_id": c_id,
                            "transaction_details": tx_p,
                            "risk_score": pred.get('risk_score', 0.0),
                            "anomaly_score": pred.get('anomaly_score', 0.0),
                            "decision": d_choice,
                            "comments": analyst_notes,
                            "analyst_id": analyst_id_val,
                            "shap_factors": expl.get('important_factors', [])
                        }
                    )
                    if sub_err:
                        st.error(sub_err)
                    else:
                        rec_record = sub_res.get("review_record", {})
                        st.success(f"Decision '{d_choice}' logged for Customer {c_id}. Case ID: {rec_record.get('review_id', 'REV_0001')}.")
                        st.rerun()

        with col_hr2:
            st.markdown("##### Case Decision History")
            if past_logs:
                for r in reversed(past_logs[-5:]):
                    ico = "🔴" if r.get('decision') == "REJECT" else ("🟢" if r.get('decision') == "APPROVE" else "🟡")
                    with st.expander(f"{ico} {r.get('transaction_id')} - {r.get('decision')}"):
                        st.write(f"**Analyst:** {r.get('analyst_id')}")
                        st.write(f"**Notes:** {r.get('comments')}")
            else:
                st.info("No cases reviewed yet.")
    else:
        st.markdown("##### Case Decision History")
        if past_logs:
            for r in reversed(past_logs[-5:]):
                ico = "🔴" if r.get('decision') == "REJECT" else ("🟢" if r.get('decision') == "APPROVE" else "🟡")
                with st.expander(f"{ico} {r.get('transaction_id')} - {r.get('decision')}"):
                    st.write(f"**Analyst:** {r.get('analyst_id')}")
                    st.write(f"**Notes:** {r.get('comments')}")
        else:
            st.info("ℹ️ **No customer transaction submitted.** Submit customer details in the **Risk Assessment** tab to evaluate cases requiring human review.")
