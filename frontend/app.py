import os
import io
import json
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="LedgerMind AI | Financial Risk Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise API Endpoint Configuration (Default to local port 8001 for local testing)
API_URL = os.getenv("RISKLENS_API_URL", "http://127.0.0.1:8001").rstrip("/")

def api_call(method: str, endpoint: str, json_data: dict = None, files: dict = None, timeout: int = 45):
    """Centralized HTTP API client for communication with FastAPI backend with terminal debugging logs."""
    url = f"{API_URL}{endpoint}"
    print(f"\n[FRONTEND API CALL] {method.upper()} {url}")
    if json_data:
        print(f"[REQUEST PAYLOAD] {json.dumps(json_data, indent=2)}")
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

        print(f"[RESPONSE STATUS] {resp.status_code}")
        if resp.status_code == 200:
            res_json = resp.json()
            if isinstance(res_json, dict):
                print(f"[RESPONSE JSON KEYS] {list(res_json.keys())}")
            else:
                print(f"[RESPONSE TYPE] {type(res_json)}")
            return res_json, None
        else:
            try:
                err_detail = resp.json().get("detail", resp.text)
            except Exception:
                err_detail = resp.text
            print(f"[API ERROR DETAIL] {err_detail}")
            return None, f"API Error ({resp.status_code}) at {url}: {err_detail}"
    except requests.exceptions.RequestException as e:
        print(f"[API CONNECTION ERROR] {str(e)}")
        return None, f"Unable to communicate with LedgerMind API backend at {url}. (Details: {str(e)})"

# --- Custom Enterprise Styling ---
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.1rem;
        letter-spacing: -0.02em;
    }
    .sub-title {
        font-size: 1.15rem;
        color: #0284C7;
        margin-bottom: 0.3rem;
        font-weight: 600;
    }
    .main-desc {
        font-size: 0.95rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .welcome-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 24px 28px;
        margin-bottom: 24px;
        text-align: center;
    }
    .welcome-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 6px;
    }
    .welcome-sub {
        font-size: 1.05rem;
        color: #475569;
    }
    .workflow-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #0284C7;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .workflow-diagram {
        font-size: 0.9rem;
        font-weight: 700;
        color: #0F172A;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        white-space: pre-wrap;
        line-height: 1.5;
    }
    .workflow-caption {
        font-size: 0.88rem;
        color: #475569;
        margin-top: 10px;
        line-height: 1.4;
    }
    .status-banner {
        background-color: #F0F9FF;
        border: 1px solid #BAE6FD;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 20px;
        font-size: 0.92rem;
        color: #0369A1;
    }
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 18px;
        border: 1px solid #E2E8F0;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-value-high {
        color: #DC2626;
        font-size: 2rem;
        font-weight: 800;
        margin: 4px 0;
    }
    .metric-value-med {
        color: #D97706;
        font-size: 2rem;
        font-weight: 800;
        margin: 4px 0;
    }
    .metric-value-low {
        color: #16A34A;
        font-size: 2rem;
        font-weight: 800;
        margin: 4px 0;
    }
    .badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 14px;
        border-radius: 16px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
    }
    .badge-medium {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 14px;
        border-radius: 16px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
    }
    .badge-low {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 4px 14px;
        border-radius: 16px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
    }
    .review-alert-banner {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        color: #991B1B;
        padding: 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 20px;
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

# --- Strict Business-Friendly SHAP Feature Label Map ---
FRIENDLY_FEATURE_MAP = {
    "failed_login_attempts": "Multiple Failed Login Attempts",
    "device_risk_score": "High Device Risk",
    "credit_utilization": "High Credit Utilization",
    "credit_utilization_ratio": "High Credit Utilization",
    "debt_to_income": "High Debt-to-Income Ratio",
    "debt_to_income_ratio": "High Debt-to-Income Ratio",
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
    "merchant_category_Luxury_Goods": "Luxury Goods Merchant Category",
    "location_International_HighRisk": "High-Risk International Location",
    "location_Online_Unverified": "Unverified Online Location",
    "ae_anomaly_score": "Deep Behavior Anomaly"
}

def clean_feature_label(raw_feature_str: str) -> str:
    """Map raw technical feature string to clean business-friendly label."""
    clean_str = raw_feature_str.replace("+", "").replace("-", "").strip()
    if clean_str in FRIENDLY_FEATURE_MAP:
        return FRIENDLY_FEATURE_MAP[clean_str]
    for key, label in FRIENDLY_FEATURE_MAP.items():
        if key in clean_str:
            return label
    return clean_str.replace("_", " ").title()

# --- Session State Setup (Parallel Coexistence) ---
if 'uploaded_batch_df' not in st.session_state:
    st.session_state['uploaded_batch_df'] = None
if 'batch_total' not in st.session_state:
    st.session_state['batch_total'] = 0
if 'batch_high_risk' not in st.session_state:
    st.session_state['batch_high_risk'] = 0
if 'batch_msg' not in st.session_state:
    st.session_state['batch_msg'] = None

if 'policy_indexed' not in st.session_state:
    st.session_state['policy_indexed'] = False
if 'indexed_doc_name' not in st.session_state:
    st.session_state['indexed_doc_name'] = None
if 'indexed_chunks_count' not in st.session_state:
    st.session_state['indexed_chunks_count'] = 0

if 'has_submitted_data' not in st.session_state:
    st.session_state['has_submitted_data'] = False
if 'selected_customer_id' not in st.session_state:
    st.session_state['selected_customer_id'] = None
if 'active_cust_id' not in st.session_state:
    st.session_state['active_cust_id'] = None
if 'active_cust_name' not in st.session_state:
    st.session_state['active_cust_name'] = None
if 'active_tx_payload' not in st.session_state:
    st.session_state['active_tx_payload'] = None
if 'active_prediction' not in st.session_state:
    st.session_state['active_prediction'] = None
if 'active_explanation' not in st.session_state:
    st.session_state['active_explanation'] = None
if 'rag_response' not in st.session_state:
    st.session_state['rag_response'] = None
if 'news_response' not in st.session_state:
    st.session_state['news_response'] = None
if 'sim_response' not in st.session_state:
    st.session_state['sim_response'] = None

def update_active_customer_from_id(cust_id: str):
    """Canonical function to load & set telemetry for the selected customer ID."""
    st.session_state['selected_customer_id'] = str(cust_id)
    df = st.session_state.get('uploaded_batch_df')
    if df is not None and not df.empty:
        matching_rows = df[df['customer_id'].astype(str) == str(cust_id)]
        if not matching_rows.empty:
            selected_tx = matching_rows.iloc[0].to_dict()
            c_id = str(selected_tx.get('customer_id', cust_id))
            c_name = str(selected_tx.get('customer_name', c_id))
            
            pred_res, p_err = api_call("POST", "/predict-risk", json_data=selected_tx)
            expl_res, e_err = api_call("POST", "/explain-risk", json_data=selected_tx)
            
            if not p_err and not e_err:
                st.session_state['active_cust_id'] = c_id
                st.session_state['active_cust_name'] = c_name
                st.session_state['active_tx_payload'] = selected_tx
                st.session_state['active_prediction'] = pred_res
                st.session_state['active_explanation'] = expl_res
                st.session_state['has_submitted_data'] = True
                st.session_state['rag_response'] = None
                st.session_state['sim_response'] = None

def reset_analysis_state():
    """Reset customer selection analysis state while preserving uploaded CSV & indexed policy."""
    st.session_state['has_submitted_data'] = False
    st.session_state['selected_customer_id'] = None
    st.session_state['active_cust_id'] = None
    st.session_state['active_cust_name'] = None
    st.session_state['active_tx_payload'] = None
    st.session_state['active_prediction'] = None
    st.session_state['active_explanation'] = None
    st.session_state['rag_response'] = None
    st.session_state['news_response'] = None
    st.session_state['sim_response'] = None

def reset_all_state():
    """Complete system reset."""
    reset_analysis_state()
    st.session_state['uploaded_batch_df'] = None
    st.session_state['batch_total'] = 0
    st.session_state['batch_high_risk'] = 0
    st.session_state['batch_msg'] = None
    st.session_state['policy_indexed'] = False
    st.session_state['indexed_doc_name'] = None
    st.session_state['indexed_chunks_count'] = 0

def render_customer_selector(location_key: str = "main"):
    """Render single canonical customer selector dropdown."""
    if st.session_state.get('uploaded_batch_df') is not None and not st.session_state['uploaded_batch_df'].empty:
        df = st.session_state['uploaded_batch_df']
        cust_options = []
        cust_id_map = {}
        
        for idx, row in df.iterrows():
            cid = str(row.get('customer_id', f'CUST_{idx+1:03d}'))
            cname = str(row.get('customer_name', cid))
            label = f"{cid} — {cname}" if cname and cname != cid else f"{cid} — Customer {idx+1:03d}"
            label += f" [Risk: {row.get('risk_level', 'LOW')}]"
            cust_options.append(label)
            cust_id_map[label] = cid
            
        current_cid = st.session_state.get('selected_customer_id') or list(cust_id_map.values())[0]
        current_index = 0
        for i, opt_label in enumerate(cust_options):
            if cust_id_map[opt_label] == current_cid:
                current_index = i
                break
                
        selected_label = st.selectbox(
            "Select Customer for Detailed Analysis",
            options=cust_options,
            index=current_index,
            key=f"customer_selector_{location_key}"
        )
        
        selected_cid = cust_id_map[selected_label]
        if selected_cid != st.session_state.get('selected_customer_id'):
            update_active_customer_from_id(selected_cid)
            st.rerun()

# --- TOP BRANDING & WORKFLOW BANNER ---
st.markdown("<div class='main-title'>LedgerMind AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Financial Risk Intelligence Platform</div>", unsafe_allow_html=True)
st.markdown("<div class='main-desc'>AI-powered financial risk detection, explanation, evidence retrieval, and decision support.</div>", unsafe_allow_html=True)

# Architecture Workflow Card
st.markdown("""
<div class='workflow-card'>
    <div class='workflow-diagram'>
Transaction Data ➔ Risk Prediction ➔ Anomaly Detection ➔ Risk Explanation ➔ Policy & Evidence Search ➔ Financial News Context ➔ What-If Analysis ➔ Human Review
    </div>
    <div class='workflow-caption'>
        <b>ML predicted the risk. SHAP explains the prediction. RAG retrieved the supporting policy evidence.</b>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 6 MAIN NAVIGATION TABS ---
nav_tabs = st.tabs([
    "1. Risk & Anomaly Assessment",
    "2. Model Performance",
    "3. Policy & Evidence Search",
    "4. Financial News Analysis",
    "5. What-If Risk Analysis",
    "6. Human Review"
])

# =========================================================
# --- SIDEBAR: CHOOSE YOUR DATA (3 INDEPENDENT SECTIONS) ---
# =========================================================
st.sidebar.image("https://img.icons8.com/isometric/100/shield.png", width=50)
st.sidebar.title("Choose Your Data")
st.sidebar.caption("Input methods are independent and build a unified risk intelligence profile.")

# SECTION A: CUSTOM TRANSACTION ANALYSIS (SINGLE CUSTOMER MANUAL INPUT)
st.sidebar.markdown("### A. Custom Transaction Analysis")
st.sidebar.caption("Analyze one customer manually.")

with st.sidebar.expander("📝 Enter Custom Transaction Details", expanded=False):
    with st.form("custom_single_tx_form"):
        c_id_in = st.text_input("Customer ID (Optional)", value="", placeholder="e.g. CUSTOM_001")
        c_name_in = st.text_input("Customer Name (Optional)", value="", placeholder="e.g. John Doe")
        
        st.markdown("**Transaction Information**")
        c_amt = st.number_input("Transaction Amount ($)", value=4500.0, step=500.0)
        c_freq = st.slider("Transaction Frequency (24h)", 1, 50, 12)
        c_cat = st.selectbox("Merchant Category", ['Grocery', 'Utilities', 'Electronics', 'Luxury_Goods', 'Crypto_Exchange', 'Wire_Transfer', 'Gambling', 'Travel'], index=4)
        c_loc = st.selectbox("Transaction Location", ['Domestic', 'International_LowRisk', 'International_HighRisk', 'Online_Unverified'], index=2)
        c_night = st.checkbox("Night Transaction", value=True)
        
        st.markdown("**Security & Account Information**")
        c_logins = st.slider("Failed Login Attempts", 0, 10, 4)
        c_device = st.slider("Device Risk Score", 0.0, 1.0, 0.85, 0.05)
        c_age = st.number_input("Account Age (Days)", value=180, min_value=1)
        
        st.markdown("**Financial Information**")
        c_income = st.number_input("Monthly Income ($)", value=6000.0, step=500.0)
        c_dti = st.slider("Debt-to-Income Ratio", 0.05, 0.95, 0.65, 0.05)
        c_util = st.slider("Credit Utilization Ratio", 0.05, 0.99, 0.85, 0.05)
        c_ir = st.slider("Interest Rate (%)", 3.0, 20.0, 8.5, 0.5)
        
        btn_custom_sub = st.form_submit_button("Analyze Custom Transaction", type="primary")
        if btn_custom_sub:
            cust_id_final = c_id_in.strip() if c_id_in.strip() else "CUSTOM_001"
            cust_name_final = c_name_in.strip() if c_name_in.strip() else cust_id_final
            
            payload = {
                "customer_id": cust_id_final, "customer_name": cust_name_final,
                "transaction_amount": c_amt, "transaction_frequency": c_freq,
                "merchant_category": c_cat, "location": c_loc,
                "time_pattern": 2 if c_night else 14, "is_night_transaction": 1 if c_night else 0, "is_weekend": 0,
                "account_age_days": c_age, "avg_monthly_income": c_income, "debt_to_income": c_dti,
                "interest_rate": c_ir, "credit_utilization": c_util, "failed_login_attempts": c_logins,
                "device_risk_score": c_device
            }
            with st.spinner(f"Running custom risk analysis for {cust_id_final}..."):
                pred_res, p_err = api_call("POST", "/predict-risk", json_data=payload)
                expl_res, e_err = api_call("POST", "/explain-risk", json_data=payload)
                
            if p_err or e_err:
                st.sidebar.error(p_err or e_err)
            else:
                st.session_state['active_cust_id'] = cust_id_final
                st.session_state['active_cust_name'] = cust_name_final
                st.session_state['active_tx_payload'] = payload
                st.session_state['active_prediction'] = pred_res
                st.session_state['active_explanation'] = expl_res
                st.session_state['has_submitted_data'] = True
                st.session_state['selected_customer_id'] = cust_id_final
                st.session_state['rag_response'] = None
                st.sidebar.success(f"Analyzed custom customer '{cust_id_final}'.")

# SECTION B: UPLOAD TRANSACTION DATA (CSV)
st.sidebar.markdown("---")
st.sidebar.markdown("### B. Upload Transaction Data")
st.sidebar.caption("Upload a CSV containing multiple customer transactions.")
csv_file_input = st.sidebar.file_uploader("Upload Transaction Data (CSV)", type=["csv"], key="sidebar_csv_uploader")

if csv_file_input is not None:
    if st.sidebar.button("Analyze Transactions", type="primary", key="btn_analyze_csv"):
        with st.spinner("Processing CSV transactions with backend..."):
            batch_res, b_err = api_call(
                "POST",
                "/upload-transactions-csv",
                files={"file": (csv_file_input.name, csv_file_input.getvalue(), csv_file_input.type or "text/csv")}
            )
        if b_err:
            st.sidebar.error(b_err)
        else:
            predictions = batch_res.get("predictions", [])
            df_scored = pd.DataFrame(predictions)
            st.session_state['uploaded_batch_df'] = df_scored
            st.session_state['batch_msg'] = batch_res.get('message', 'Success')
            st.session_state['batch_total'] = batch_res.get("total_transactions", len(df_scored))
            st.session_state['batch_high_risk'] = batch_res.get("high_risk_count", 0)
            
            # Default active customer to first row in CSV
            if not df_scored.empty:
                first_cid = str(df_scored.iloc[0].get('customer_id', 'CUST_001'))
                update_active_customer_from_id(first_cid)
                
            st.sidebar.success(f"Processed {len(df_scored)} transactions.")

# SECTION C: UPLOAD BANK / COMPANY POLICY
st.sidebar.markdown("---")
st.sidebar.markdown("### C. Upload Bank / Company Policy")
st.sidebar.caption("Upload a PDF or TXT containing bank policies, compliance rules, risk guidelines, annual reports, or internal procedures.")
policy_file_input = st.sidebar.file_uploader("Upload Policy Document (PDF/TXT)", type=["pdf", "txt"], key="sidebar_doc_uploader")

if policy_file_input is not None:
    if st.sidebar.button("Process Policy Document", key="btn_process_doc"):
        with st.spinner("Indexing policy document into FAISS vector store..."):
            doc_res, d_err = api_call(
                "POST",
                "/upload-document",
                files={"file": (policy_file_input.name, policy_file_input.getvalue(), policy_file_input.type or "application/octet-stream")}
            )
        if d_err:
            st.sidebar.error(d_err)
        else:
            st.session_state['policy_indexed'] = True
            st.session_state['indexed_doc_name'] = policy_file_input.name
            st.session_state['indexed_chunks_count'] = doc_res.get('chunks_indexed', 0)
            st.sidebar.success(f"Document successfully processed. Policy evidence is ready for customer-specific risk questions.")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 System Reset"):
    reset_all_state()
    st.rerun()


# =========================================================
# --- TAB 1: RISK & ANOMALY ASSESSMENT ---
# =========================================================
with nav_tabs[0]:
    st.subheader("Risk & Anomaly Assessment")
    
    # 1. PARALLEL STATUS BANNER
    csv_status = f"✓ CSV Batch Analyzed ({st.session_state['batch_total']} records)" if st.session_state['uploaded_batch_df'] is not None else "⏳ Awaiting Transaction CSV Upload"
    doc_status = f"✓ Policy Document Processed ({st.session_state['indexed_doc_name']})" if st.session_state['policy_indexed'] else "⏳ Awaiting Policy Document Upload"
    cust_status = f"✓ Active Customer: {st.session_state['active_cust_id']}" if st.session_state['has_submitted_data'] else "⏳ Select Customer for Evaluation"
    
    st.markdown(f"""
    <div class='status-banner'>
        <b>Parallel Workflow Status:</b><br/>
        • <b>Transaction Data:</b> {csv_status}<br/>
        • <b>Policy Source:</b> {doc_status}<br/>
        • <b>Active Profile:</b> {cust_status}
    </div>
    """, unsafe_allow_html=True)

    # 2. INITIAL WELCOME STATE (If no data submitted yet)
    if st.session_state['uploaded_batch_df'] is None and not st.session_state['has_submitted_data']:
        st.markdown("""
        <div class='welcome-card'>
            <div class='welcome-title'>Welcome to LedgerMind AI</div>
            <div class='welcome-sub'>Enter transaction details or upload transaction data to begin.</div>
        </div>
        """, unsafe_allow_html=True)
        st.info("💡 **Getting Started:** Use **Section A** for a custom transaction, **Section B** for a CSV batch, or **Section C** to upload policy documents.")

    # 3. TRANSACTION RISK SUMMARY (If CSV Processed)
    if st.session_state['uploaded_batch_df'] is not None:
        df_scored = st.session_state['uploaded_batch_df']
        st.markdown("### Transaction Risk Summary")
        
        b_c1, b_c2, b_c3, b_c4 = st.columns(4)
        b_c1.metric("Total Transactions", st.session_state.get("batch_total", len(df_scored)))
        b_c2.metric("High-Risk Transactions", st.session_state.get("batch_high_risk", 0))
        b_c3.metric("Average Risk Score", f"{df_scored['risk_score'].mean():.1%}" if not df_scored.empty else "0.0%")
        
        req_rev_cnt = int(df_scored.get("requires_human_review", pd.Series([False]*len(df_scored))).sum()) if not df_scored.empty else 0
        b_c4.metric("Transactions Requiring Review", req_rev_cnt)
        
        if not df_scored.empty:
            display_table = df_scored[['customer_id', 'customer_name', 'transaction_amount', 'risk_score', 'risk_level', 'anomaly_score', 'review_status']].rename(columns={
                'customer_id': 'Customer ID',
                'customer_name': 'Customer Name',
                'transaction_amount': 'Transaction Amount ($)',
                'risk_score': 'Risk Score',
                'risk_level': 'Risk Level',
                'anomaly_score': 'Anomaly Score',
                'review_status': 'Review Status'
            })
            st.dataframe(display_table, use_container_width=True)
            
            st.markdown("---")
            st.markdown("### Select Customer")
            st.caption("Choose a customer to view their individual risk assessment and policy-based explanation.")
            render_customer_selector(location_key="tab1")

    # 4. ACTIVE CUSTOMER RISK PROFILE & SHAP EXPLANATION
    if st.session_state['has_submitted_data'] and st.session_state['active_tx_payload'] is not None:
        st.markdown("---")
        c_id = st.session_state['active_cust_id'] or "CUST_001"
        c_name = st.session_state['active_cust_name'] or c_id
        prediction = st.session_state['active_prediction']
        explanation = st.session_state['active_explanation']

        if prediction and explanation:
            requires_review = prediction.get('requires_human_review', False)

            if requires_review:
                st.markdown(
                    "<div class='review-alert-banner'>🚨 REQUIRES HUMAN REVIEW<br/>"
                    f"<span style='font-size: 0.9rem; font-weight: 400;'>Customer {c_id} exhibits elevated risk ({int(prediction['risk_score']*100)}%) or deep behavior anomaly requiring analyst review.</span></div>",
                    unsafe_allow_html=True
                )

            st.markdown(f"### Customer Risk Assessment: **{c_id}** ({c_name})")

            # Risk Cards
            r_col1, r_col2, r_col3 = st.columns(3)
            
            with r_col1:
                st.markdown("##### Overall Risk")
                r_val_class = "metric-value-high" if prediction['risk_level'] == "HIGH" else ("metric-value-med" if prediction['risk_level'] == "MEDIUM" else "metric-value-low")
                st.markdown(f"<div class='metric-card'><div class='{r_val_class}'>{int(prediction['risk_score']*100)}%</div><p style='margin:0;'><span class='badge-{prediction['risk_level'].lower()}'>{prediction['risk_level']}</span></p></div>", unsafe_allow_html=True)

            with r_col2:
                st.markdown("##### Anomaly Score")
                st.markdown(f"<div class='metric-card'><div class='metric-value-med'>{int(prediction['anomaly_score']*100)}%</div><p style='margin:0; font-weight:700;'>{prediction['anomaly_status']}</p></div>", unsafe_allow_html=True)

            with r_col3:
                st.markdown("##### Final Assessment")
                comb_level = prediction['combined_risk_level']
                b_class = "badge-high" if comb_level == "HIGH" else ("badge-medium" if comb_level == "MEDIUM" else "badge-low")
                st.markdown(f"<div class='metric-card'><div class='metric-value-high'>{int(prediction['combined_risk_score']*100)}%</div><p style='margin:4px;'><span class='{b_class}'>{comb_level} RISK</span></p></div>", unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("Why is this customer risky?")
            st.caption("Key factors influencing the model's risk prediction.")

            raw_factors = explanation.get('important_factors', [])
            s_col1, s_col2 = st.columns(2)

            with s_col1:
                st.markdown("##### Risk-Increasing Factors")
                inc_f = [f for f in raw_factors if f.startswith("+")]
                if inc_f:
                    for f in inc_f:
                        st.error(f"• {clean_feature_label(f)}")
                else:
                    st.write("No major risk-increasing factors.")

            with s_col2:
                st.markdown("##### Risk-Reducing Factors")
                red_f = [f for f in raw_factors if f.startswith("-")]
                if red_f:
                    for f in red_f:
                        st.success(f"• {clean_feature_label(f)}")
                else:
                    st.write("Standard account parameters.")

            # Clean Plotly SHAP Chart with STRICT FRIENDLY LABELS on Y-Axis
            shap_raw = dict(list(explanation.get('shap_values', {}).items())[:6])
            friendly_shap = {clean_feature_label(k): v for k, v in shap_raw.items()}
            df_shap = pd.DataFrame({"Risk Factor": list(friendly_shap.keys()), "Impact": list(friendly_shap.values())}).sort_values("Impact", ascending=True)
            
            fig_shap = px.bar(
                df_shap, 
                x="Impact", 
                y="Risk Factor", 
                orientation="h", 
                title="SHAP Risk Factor Contribution Chart", 
                color="Impact", 
                color_continuous_scale=["#10B981", "#EF4444"]
            )
            fig_shap.update_layout(height=280, margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig_shap, use_container_width=True)

# =========================================================
# --- TAB 2: MODEL PERFORMANCE ---
# =========================================================
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
                ae_m = eval_res.get("all_model_metrics", {}).get("autoencoder", {})
                st.json({
                    "Reconstruction Baseline Mean": f"{ae_m.get('reconstruction_mean', 0.5):.4f}",
                    "Reconstruction Baseline Std": f"{ae_m.get('reconstruction_std', 0.8):.4f}",
                    "Autoencoder Accuracy": f"{ae_m.get('accuracy', 0.0):.4f}",
                    "Autoencoder ROC-AUC": f"{ae_m.get('roc_auc', 0.0):.4f}"
                })

# =========================================================
# --- TAB 3: POLICY & EVIDENCE SEARCH (CUSTOMER-SPECIFIC RAG) ---
# =========================================================
with nav_tabs[2]:
    st.subheader("Policy & Evidence Search")
    
    # Architecture Boundary Notice
    st.info("ℹ️ **Architecture Boundary:** ML predicts customer risk score; SHAP explains the prediction. RAG searches the uploaded bank policy to retrieve supporting compliance evidence. RAG does NOT generate the risk score.")
    
    # 1. CUSTOMER SELECTOR AT TOP OF TAB 3
    if st.session_state.get('uploaded_batch_df') is not None:
        st.markdown("### Selected Customer")
        render_customer_selector(location_key="tab3")

    has_customer = st.session_state.get('has_submitted_data', False) and st.session_state.get('active_tx_payload') is not None
    has_policy = st.session_state.get('policy_indexed', False)

    if not has_customer:
        st.warning("⚠️ Analyze a transaction or upload transaction data to select a customer for policy-based risk explanation.")
    else:
        c_id = st.session_state.get('active_cust_id', 'CUST_001')
        c_name = st.session_state.get('active_cust_name', c_id)
        pred = st.session_state.get('active_prediction') or {}
        expl = st.session_state.get('active_explanation') or {}

        # 2. CUSTOMER RISK CONTEXT
        st.markdown("---")
        st.markdown("### Customer Risk Context")
        
        cr_col1, cr_col2, cr_col3, cr_col4 = st.columns(4)
        cr_col1.metric("Customer ID", f"{c_id}")
        cr_col2.metric("Risk Score", f"{int(pred.get('risk_score', 0.0)*100)}%")
        cr_col3.metric("Risk Level", pred.get('risk_level', 'LOW'))
        cr_col4.metric("Anomaly Score", f"{int(pred.get('anomaly_score', 0.0)*100)}%")

        raw_inc_factors = [f for f in expl.get('important_factors', []) if f.startswith("+")]
        friendly_factors = [clean_feature_label(f) for f in raw_inc_factors]
        
        st.markdown("**Top Risk Factors:**")
        if friendly_factors:
            for ff in friendly_factors[:4]:
                st.markdown(f"• **{ff}**")
        else:
            st.markdown("• Standard account parameters")

        # 3. ASK A RISK QUESTION
        st.markdown("---")
        st.markdown("### Ask a Risk Question")
        st.caption("Ask why this customer is considered risky based on the uploaded bank/company policy.")

        if has_policy:
            doc_name_str = st.session_state.get('indexed_doc_name', 'Bank_Risk_Policy.txt')
            st.success(f"✓ Policy document ready for evidence search ({doc_name_str}).")
        else:
            st.info("👈 **Policy Document Not Uploaded Yet.** Upload a bank/company policy in Section C of the sidebar (or below) to enable RAG policy evidence search.")
            
        tab3_policy_file = st.file_uploader("Upload Bank / Company Policy (PDF/TXT)", type=["pdf", "txt"], key="tab3_policy_uploader")
        if tab3_policy_file is not None and not st.session_state.get('policy_indexed', False):
            if st.button("Process Policy Document Now", key="tab3_btn_process"):
                with st.spinner("Indexing policy document into FAISS vector store..."):
                    doc_res, d_err = api_call(
                        "POST",
                        "/upload-document",
                        files={"file": (tab3_policy_file.name, tab3_policy_file.getvalue(), tab3_policy_file.type or "application/octet-stream")}
                    )
                if d_err:
                    st.error(d_err)
                else:
                    st.session_state['policy_indexed'] = True
                    st.session_state['indexed_doc_name'] = tab3_policy_file.name
                    st.session_state['indexed_chunks_count'] = doc_res.get('chunks_indexed', 0)
                    st.success("Document successfully processed! Policy evidence is ready.")
                    st.rerun()

        default_rag_question = "Why is this customer considered high risk according to the uploaded policy?"
        rag_q_input = st.text_input("Question", value=default_rag_question, key=f"rag_q_input_{c_id}")

        btn_retrieve = st.button("Retrieve Policy Evidence", type="primary")

        if btn_retrieve:
            # Auto-process pending file in uploader if present and not yet indexed
            pending_file = tab3_policy_file or policy_file_input
            if pending_file is not None and not st.session_state.get('policy_indexed', False):
                with st.spinner(f"Processing uploaded policy file '{pending_file.name}'..."):
                    doc_res, d_err = api_call(
                        "POST",
                        "/upload-document",
                        files={"file": (pending_file.name, pending_file.getvalue(), pending_file.type or "application/octet-stream")}
                    )
                if not d_err:
                    st.session_state['policy_indexed'] = True
                    st.session_state['indexed_doc_name'] = pending_file.name
                    st.session_state['indexed_chunks_count'] = doc_res.get('chunks_indexed', 0)

            customer_context_payload = {
                "customer_id": c_id,
                "customer_name": c_name,
                "risk_score": pred.get('risk_score', 0.0),
                "risk_level": pred.get('risk_level', 'LOW'),
                "anomaly_status": pred.get('anomaly_status', 'NORMAL'),
                "important_factors": friendly_factors
            }
            
            with st.spinner(f"Retrieving grounded policy evidence for {c_id}..."):
                rag_res, r_err = api_call(
                    "POST", 
                    "/retrieve-evidence", 
                    json_data={
                        "query": rag_q_input,
                        "top_k": 4,
                        "transaction_context": customer_context_payload
                    }
                )
            if r_err:
                st.error(r_err)
            else:
                st.session_state['rag_response'] = rag_res

        # 4. RAG RESULT SECTION
        if st.session_state.get('rag_response') is not None:
            rag_res = st.session_state['rag_response']
            st.markdown("---")
            st.markdown("### Policy-Based Risk Explanation")
            
            st.markdown(f"**Customer:** {c_id} ({c_name})")
            st.markdown(f"**Question:** *\"{rag_q_input}\"*")
            
            st.markdown("#### ANSWER")
            st.markdown(rag_res.get('answer', 'No policy answer generated.'))
            
            retrieved_docs = rag_res.get('evidence', [])
            if retrieved_docs:
                st.markdown("#### SUPPORTING EVIDENCE")
                for idx, ev in enumerate(retrieved_docs, 1):
                    doc_n = ev.get('document_name', st.session_state.get('indexed_doc_name') or 'Bank_Risk_Policy.txt')
                    page_n = ev.get('page_number', 1)
                    content_n = ev.get('content', '').strip()
                    sec_n = "Authentication Risk" if "login" in content_n.lower() else ("Device Risk" if "device" in content_n.lower() else "Credit & Exposure Guidelines")
                    
                    st.markdown(f"""
                    <div style='background-color:#FFFFFF; border:1px solid #E2E8F0; border-left:4px solid #10B981; border-radius:6px; padding:14px; margin-bottom:12px;'>
                        <div style='font-weight:700; font-size:0.95rem; color:#0F172A;'>✓ Evidence Excerpt {idx} — {sec_n}</div>
                        <div style='font-size:0.9rem; color:#334155; margin:6px 0;'>{content_n}</div>
                        <div style='font-size:0.82rem; color:#64748B;'><b>SOURCE DOCUMENT:</b> {doc_n} &nbsp;|&nbsp; <b>PAGE / SECTION:</b> Page {page_n} ({sec_n})</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.caption("ℹ️ *ML predicted the risk. SHAP explains the prediction. RAG retrieved the supporting policy evidence.*")

# =========================================================
# --- TAB 4: FINANCIAL NEWS ANALYSIS ---
# =========================================================
with nav_tabs[3]:
    st.subheader("Financial News Analysis")
    st.caption("Analyze external financial news articles and regulatory filings for market risk sentiment.")
    
    if st.session_state.get('has_submitted_data') and st.session_state.get('active_cust_id'):
        st.caption(f"Currently active customer profile: **{st.session_state['active_cust_id']}** ({st.session_state.get('active_cust_name')})")
        
    news_text_in = st.text_area("Enter Financial News Text", value="Apex Financial Group faces quarterly loss as non-performing assets rise 18%.", height=90)
    if st.button("Analyze Financial News", type="primary"):
        with st.spinner("Analyzing news sentiment with FinBERT NLP backend..."):
            news_out, n_err = api_call("POST", "/analyze-news", json_data={"text": news_text_in})
            
        if n_err:
            st.error(n_err)
        elif news_out:
            st.session_state['news_response'] = news_out

    if st.session_state.get('news_response') is not None:
        news_out = st.session_state['news_response']
        st.markdown("---")
        st.markdown("### Financial News Analysis Results")
        n_c1, n_c2, n_c3 = st.columns(3)
        n_c1.metric("Sentiment", news_out.get('sentiment', 'Neutral'))
        n_c2.metric("Target Entity / Company", news_out.get('entity', 'Company'))
        n_c3.metric("Risk Impact Level", news_out.get('impact', 'Low Risk'))

# =========================================================
# --- TAB 5: WHAT-IF RISK ANALYSIS ---
# =========================================================
with nav_tabs[4]:
    st.subheader("What-If Risk Analysis")
    st.markdown("Change financial conditions to see how the estimated risk could change.")
    
    if st.session_state.get('has_submitted_data') and st.session_state.get('active_tx_payload') is not None:
        base_p = st.session_state['active_tx_payload']
        c_id = st.session_state.get('active_cust_id', 'CUST_001')
    else:
        st.info("ℹ️ No customer data selected. Using baseline reference profile.")
        base_p = {
            "transaction_amount": 4500.0, "transaction_frequency": 12, "merchant_category": "Crypto_Exchange",
            "location": "International_HighRisk", "time_pattern": 2, "is_night_transaction": 1, "is_weekend": 0,
            "account_age_days": 180, "avg_monthly_income": 6000.0, "debt_to_income": 0.65, "interest_rate": 8.5,
            "credit_utilization": 0.88, "failed_login_attempts": 4, "device_risk_score": 0.85
        }
        c_id = "Baseline Reference"

    st.markdown(f"##### Counterfactual Scenario Adjuster ({c_id})")
    w_col1, w_col2, w_col3, w_col4 = st.columns(4)
    
    with w_col1:
        scen_amt = st.number_input("Scenario Amount ($)", value=float(base_p.get('transaction_amount', 4500.0) * 1.5), step=500.0)
    with w_col2:
        scen_income = st.number_input("Scenario Monthly Income ($)", value=float(base_p.get('avg_monthly_income', 6000.0)), step=500.0)
    with w_col3:
        scen_ir = st.slider("Scenario Interest Rate (%)", 3.0, 22.0, float(base_p.get('interest_rate', 8.5) + 2.0), 0.5)
    with w_col4:
        scen_dti = st.slider("Scenario Debt-to-Income", 0.05, 0.95, float(base_p.get('debt_to_income', 0.65)), 0.05)

    if st.button("Run What-If Analysis", type="primary"):
        with st.spinner("Simulating counterfactual scenario deltas..."):
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
            st.session_state['sim_response'] = sim_out

    if st.session_state.get('sim_response') is not None:
        sim_out = st.session_state['sim_response']
        st.markdown("---")
        st.markdown("### What-If Simulation Results")
        wc_1, wc_2, wc_3, wc_4 = st.columns(4)
        wc_1.metric("Current Risk", f"{int(sim_out['baseline']['risk_score']*100)}%")
        wc_2.metric("Scenario Risk", f"{int(sim_out['scenario']['risk_score']*100)}%")
        wc_3.metric("Risk Change", f"{sim_out['delta']['percentage_change']:+.1f}%")
        wc_4.metric("Direction", sim_out['delta']['direction'])

        st.markdown("##### Explanation")
        st.info(sim_out.get('impact_summary', ''))

# =========================================================
# --- TAB 6: HUMAN REVIEW ---
# =========================================================
with nav_tabs[5]:
    st.subheader("Human Review")
    st.caption("Analyst case decision portal — Demonstration workflow for human-in-the-loop review.")
    
    # Fetch Past Case History from Backend
    with st.spinner("Fetching case review logs..."):
        rev_hist_res, h_err = api_call("GET", "/reviews")
    past_logs = rev_hist_res.get("reviews", []) if rev_hist_res else []
    
    if st.session_state.get('has_submitted_data') and st.session_state.get('active_tx_payload') is not None:
        c_id = st.session_state.get('active_cust_id', 'CUST_001')
        c_name = st.session_state.get('active_cust_name', c_id)
        tx_p = st.session_state['active_tx_payload']
        pred = st.session_state.get('active_prediction') or {}
        expl = st.session_state.get('active_explanation') or {}
        
        col_hr1, col_hr2 = st.columns([1.5, 1])
        
        with col_hr1:
            st.markdown("##### Active Customer Case Details")
            st.json({
                "Customer ID": c_id,
                "Customer Name": c_name,
                "Risk Level": pred.get('risk_level', 'LOW'),
                "Risk Score": f"{int(pred.get('risk_score', 0.0)*100)}%",
                "Anomaly Score": f"{int(pred.get('anomaly_score', 0.0)*100)}% ({pred.get('anomaly_status', 'NORMAL')})",
                "Top Risk Factors": [clean_feature_label(f) for f in expl.get('important_factors', []) if f.startswith("+")][:3]
            })
            
            st.markdown("##### Review Case Action")
            with st.form("human_analyst_decision_form"):
                analyst_id_val = st.text_input("Analyst ID", value="Analyst_J_Doe")
                analyst_notes = st.text_area("Analyst Rationale / Notes", value="High risk score and failed login attempts violate compliance thresholds.")
                
                b1, b2, b3 = st.columns(3)
                act_app = b1.form_submit_button("Approve")
                act_rej = b2.form_submit_button("Reject")
                act_esc = b3.form_submit_button("Escalate")
                
                d_choice = None
                if act_app:
                    d_choice = "APPROVE"
                elif act_rej:
                    d_choice = "REJECT"
                elif act_esc:
                    d_choice = "ESCALATE"
                    
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
                        st.success(f"Case '{d_choice}' logged for Customer {c_id}. Decision Record ID: {rec_record.get('review_id', 'REV_0001')}.")
                        st.rerun()

        with col_hr2:
            st.markdown("##### Case Decision History")
            if past_logs:
                for r in reversed(past_logs[-5:]):
                    ico = "🔴" if r.get('decision') == "REJECT" else ("🟢" if r.get('decision') == "APPROVE" else "ABSTAIN")
                    with st.expander(f"{ico} {r.get('transaction_id')} - {r.get('decision')}"):
                        st.write(f"**Analyst:** {r.get('analyst_id')}")
                        st.write(f"**Notes:** {r.get('comments')}")
            else:
                st.info("No cases reviewed yet.")
    else:
        st.markdown("##### Case Decision History")
        if past_logs:
            for r in reversed(past_logs[-5:]):
                ico = "🔴" if r.get('decision') == "REJECT" else ("🟢" if r.get('decision') == "APPROVE" else "ABSTAIN")
                with st.expander(f"{ico} {r.get('transaction_id')} - {r.get('decision')}"):
                    st.write(f"**Analyst:** {r.get('analyst_id')}")
                    st.write(f"**Notes:** {r.get('comments')}")
        else:
            st.info("ℹ️ **No customer transaction submitted.** Analyze transactions in **Tab 1** to select a customer requiring review.")
