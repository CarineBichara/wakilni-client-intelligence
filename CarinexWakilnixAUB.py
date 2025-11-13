# ===================== Wakilni Client Intelligence - Technical + Executive Edition (Enhanced Final Version) =====================
# Tabs: Overview | Predictions | Recommendations | Validation | Feedback Insights
# Author: Carine Bichara (MSBA Capstone 2025, AUB)

import os
import io
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from pathlib import Path
from base64 import b64encode
from datetime import datetime
import plotly.express as px
import warnings

try:
    from wordcloud import WordCloud
    from PIL import Image
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

warnings.filterwarnings("ignore")

# -------------------------------- THEME / BRANDING --------------------------------
PRIMARY = "#8B0000"   # Wakilni Dark Red
LIGHT   = "#FFC1C1"   # Soft Pink/Red
NEUTRAL = "#fff5f5"   # Off-white background
MUTED   = "#5b5b5b"
ACCENT  = "#FFF8B3"

px.defaults.color_discrete_sequence = [PRIMARY, LIGHT, "#d46a6a", "#ff9999"]

st.set_page_config(page_title="Wakilni Client Intelligence - Predictions", layout="wide", initial_sidebar_state="expanded")

# -------------------------------- CSS --------------------------------
st.markdown(
    f"""
    <style>
        html, body, [class*="css"] {{ font-size: 16px; color: #0b1f44; }}
        .main {{ background-color: {NEUTRAL}; }}
        .header-container {{
            display:flex; align-items:center; justify-content:space-between;
            background-color:{NEUTRAL}; padding:22px 10px 6px 10px; margin-top:-10px; margin-bottom:6px;
        }}
        .header-center {{ text-align:center; flex-grow:1; }}
        .header-center h1 {{ color:{PRIMARY}; font-weight:800; font-size:34px; margin:0; letter-spacing:0.2px; }}
        .header-center p {{ margin-top:6px; font-size:14px; color:{MUTED}; }}
        .header-line {{ border-top:1px solid #e7c9c9; margin:10px 0 2px 0; }}
        .section-title {{ color:{PRIMARY}; font-weight:800; font-size:22px; margin:18px 0 10px 0; }}
        .info-box {{
            background-color: #f0f5ff; padding: 18px; margin-bottom: 16px; border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.06); border-left: 6px solid #3366ff;
        }}
        .action-box {{
            background-color: #fff8f0; padding: 18px; margin-bottom: 16px; border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.06); border-left: 6px solid #ff9933;
        }}
        .success-box {{
            background-color: #e8f5e9; padding: 18px; margin-bottom: 16px; border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.06); border-left: 6px solid #4caf50;
        }}
        [data-testid="stFileUploader"] section {{
            background-color:{ACCENT} !important; border-radius:12px !important; padding:10px 20px !important;
            border:1px solid {PRIMARY} !important;
        }}
        [data-testid="stFileUploader"] button {{
            background-color:{PRIMARY} !important; color:#fff !important; border:none !important; border-radius:6px !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------- HEADER --------------------------------
AUB_LOGO = Path("AUB_Logo_OSB_Horizontal_RGB.png")
WAKILNI_LOGO = Path("wakilni-logo-v2.png")

def _b64(path: Path) -> str:
    try:
        with open(path, "rb") as f:
            return b64encode(f.read()).decode()
    except Exception:
        return ""

aub_b64 = _b64(AUB_LOGO)
wakilni_b64 = _b64(WAKILNI_LOGO)

st.markdown(
    f"""
    <div class="header-container">
        <img src="data:image/png;base64,{aub_b64}" style="height:58px;object-fit:contain;">
        <div class="header-center">
            <h1>Wakilni Client Intelligence</h1>
            <p>Prediction Engine - Auto-Revalidated Monthly - Pipeline-integrated</p>
        </div>
        <img src="data:image/png;base64,{wakilni_b64}" style="height:50px;object-fit:contain;">
    </div>
    <div class="header-line"></div>
    """,
    unsafe_allow_html=True
)

# ---------- Power BI link ----------
POWER_BI_URL = "https://app.powerbi.com/links/0k7REKF7TJ?ctid=c7ba5b1a-41b6-43e9-a120-6ff654ada137&pbi_source=linkShare"
st.markdown(f'<a style="background:{PRIMARY};color:#fff;padding:10px 14px;border-radius:8px;text-decoration:none;font-weight:600;" href="{POWER_BI_URL}" target="_blank">Open full Power BI report</a>', unsafe_allow_html=True)

# -------------------------------- MODEL LOAD --------------------------------

MODEL_PATH = "churn_pipeline.pkl"   # <-- correct path

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()

model = load_model()

# -------------------------------- FILE UPLOAD --------------------------------
st.subheader("Upload Enriched Client Dataset (Prediction Only)")
uploaded = st.file_uploader("Upload the enriched CSV", type="csv")

if not uploaded:
    st.markdown(
        f"""
        <div class='info-box'>
            <b>How to use:</b> Upload your enriched client CSV to generate churn probabilities,
            segment clients by risk, and export high-risk lists.
        </div>
        """, unsafe_allow_html=True
    )
    st.stop()

# -------------------------------- DATA PREPARATION --------------------------------
df = pd.read_csv(uploaded)
if df.empty:
    st.error("Uploaded file is empty.")
    st.stop()

df.columns = df.columns.str.strip().astype(str)
expected_cols = getattr(model, "feature_names_in_", None)
if expected_cols is None and hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
    expected_cols = getattr(model.named_steps["preprocessor"], "feature_names_in_", None)

if expected_cols is not None:
    expected_cols = list(expected_cols)
    X = df.reindex(columns=expected_cols, fill_value=0)
else:
    X = df.select_dtypes(include=[np.number]).fillna(0)

# -------------------------------- PREDICTION --------------------------------
try:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        proba = model.predict(X).astype(float)
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

df["Churn_Probability"] = proba
df["Churn_Prediction"] = (proba > 0.5).astype(int)
df_pred = df.copy()

# -------------------------------- SIDEBAR --------------------------------
with st.sidebar:
    st.markdown("### Filters")
    risk_hi = st.slider("High-risk threshold", 0.50, 0.95, 0.70, 0.01)
    risk_med = st.slider("Medium-risk threshold", 0.20, float(risk_hi), 0.50, 0.01)

# -------------------------------- TABS --------------------------------
tabs = st.tabs(["Overview", "Predictions", "Recommendations", "Validation", "Feedback Insights"])

# ------------------------------- OVERVIEW TAB -------------------------------
with tabs[0]:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)

    total = len(df_pred)
    hi_rate = (df_pred["Churn_Probability"] > risk_hi).mean()
    med_rate = ((df_pred["Churn_Probability"] > risk_med) & (df_pred["Churn_Probability"] <= risk_hi)).mean()
    lo_rate = (df_pred["Churn_Probability"] <= risk_med).mean()
    avg_p = df_pred["Churn_Probability"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total clients scored", f"{total:,}")
    col2.metric("Avg churn probability", f"{avg_p:.2f}")
    col3.metric(f"High-risk (>{risk_hi:.2f})", f"{hi_rate:.1%}")
    col4.metric(f"Medium-risk ({risk_med:.2f}-{risk_hi:.2f})", f"{med_rate:.1%}")
    col5.metric(f"Low-risk (≤{risk_med:.2f})", f"{lo_rate:.1%}")

    hist = px.histogram(df_pred, x="Churn_Probability", nbins=30, color_discrete_sequence=[PRIMARY])
    hist.add_vline(x=risk_med, line_dash="dash", line_color="#666")
    hist.add_vline(x=risk_hi, line_dash="dash", line_color=PRIMARY)
    hist.update_layout(plot_bgcolor=NEUTRAL, paper_bgcolor=NEUTRAL)
    st.plotly_chart(hist, use_container_width=True)

    # --- Both grouped breakdowns ---
    if "qadaa" in df_pred.columns:
        g_qadaa = df_pred.groupby("qadaa")["Churn_Probability"].mean().reset_index().sort_values("Churn_Probability", ascending=False)
        fig_q = px.bar(g_qadaa, x="qadaa", y="Churn_Probability", title="Average Churn Probability by Qadaa",
                       color_discrete_sequence=[PRIMARY])
        fig_q.update_traces(texttemplate='%{y:.2f}', textposition='outside')
        fig_q.update_layout(plot_bgcolor=NEUTRAL, paper_bgcolor=NEUTRAL)
        st.plotly_chart(fig_q, use_container_width=True)

    if "industry" in df_pred.columns:
        g_ind = df_pred.groupby("industry")["Churn_Probability"].mean().reset_index().sort_values("Churn_Probability", ascending=False)
        fig_i = px.bar(g_ind, x="industry", y="Churn_Probability", title="Average Churn Probability by Industry",
                       color_discrete_sequence=[PRIMARY])
        fig_i.update_traces(texttemplate='%{y:.2f}', textposition='outside')
        fig_i.update_layout(plot_bgcolor=NEUTRAL, paper_bgcolor=NEUTRAL, xaxis_tickangle=-45)
        st.plotly_chart(fig_i, use_container_width=True)

# ------------------------------- PREDICTIONS TAB -------------------------------
with tabs[1]:
    st.markdown('<div class="section-title">High-Risk Clients</div>', unsafe_allow_html=True)
    hi = df_pred[df_pred["Churn_Probability"] > risk_hi]
    cols = [c for c in ["acc_ref", "client", "industry", "qadaa"] if c in df_pred.columns] + ["Churn_Probability"]

    if hi.empty:
        st.info("No clients above threshold.")
    else:
        st.dataframe(hi[cols].sort_values("Churn_Probability", ascending=False), use_container_width=True)
        st.download_button("Download High-Risk Clients (CSV)", hi.to_csv(index=False), "high_risk_clients.csv")

# ------------------------------- RECOMMENDATIONS TAB -------------------------------
with tabs[2]:
    st.markdown('<div class="section-title">Strategic Recommendations</div>', unsafe_allow_html=True)
    any_rate = (df_pred["Churn_Probability"] > risk_med).mean()
    st.markdown(f"<div class='info-box'><b>Status:</b> {any_rate:.1%} of clients at medium or higher risk.</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class='action-box'>
            <b>Action Plan</b>
            <ul>
                <li><b>High-risk (>{risk_hi:.2f})</b>: Immediate outreach; assign AM follow-up within 48h.</li>
                <li><b>Medium-risk ({risk_med:.2f}-{risk_hi:.2f})</b>: Monitor weekly; enhance loyalty nudges.</li>
                <li><b>Low-risk (≤{risk_med:.2f})</b>: Maintain SLA satisfaction; explore upsell/cross-sell pilots.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True
    )

# ------------------------------- VALIDATION TAB -------------------------------
with tabs[3]:
    st.markdown('<div class="section-title">Human Validation</div>', unsafe_allow_html=True)
    sample = df_pred[df_pred["Churn_Probability"] > risk_hi].nlargest(10, "Churn_Probability")

    if sample.empty:
        st.info("No high-risk clients to validate.")
    else:
        if 'validation_data' not in st.session_state:
            st.session_state.validation_data = {}
        for idx, row in sample.iterrows():
            cid = row.get("acc_ref", f"Client_{idx}")
            val = st.radio(f"{cid} - p={row['Churn_Probability']:.2f}",
                           ["Uncertain", "Correct high-risk", "False alarm"],
                           key=f"val_{idx}", horizontal=True)
            st.session_state.validation_data[str(idx)] = {
                "client": cid, "probability": float(row["Churn_Probability"]),
                "validation": val, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        val_df = pd.DataFrame.from_dict(st.session_state.validation_data, orient="index")
        st.download_button("Download Validation Labels", val_df.to_csv(index=False), "validation_labels.csv")
# ------------------------------- FEEDBACK INSIGHTS TAB -------------------------------
with tabs[4]:
    st.markdown('<div class="section-title">Feedback Insights (NLP-based)</div>', unsafe_allow_html=True)

    # Avoid SessionInfo crash
    if "feedback_uploaded" not in st.session_state:
        st.session_state.feedback_uploaded = None

    fb_file = st.file_uploader(
        "Upload Feedback CSV", 
        type="csv", 
        key="feedback_upload_key"
    )

    # Store the file once
    if fb_file is not None:
        st.session_state.feedback_uploaded = fb_file

    # If nothing uploaded
    if st.session_state.feedback_uploaded is None:
        st.info("Upload the NLP-enriched feedback CSV to explore sentiment and trends.")
        st.stop()

    # Load file safely
    df_fb = pd.read_csv(st.session_state.feedback_uploaded)
    df_fb.columns = df_fb.columns.str.strip().str.lower()

    # Continue normally...
    text_col = next((c for c in ["feedback", "comment", "text", "full_feedback"] if c in df_fb.columns), None)

    st.metric("Total Feedback Entries", len(df_fb))

    if "sentiment_label_rule" in df_fb.columns:
        fig_sent = px.pie(df_fb, names="sentiment_label_rule",
                          title="Sentiment Distribution",
                          color_discrete_sequence=[PRIMARY, LIGHT, "#ff9999"])
        st.plotly_chart(fig_sent, use_container_width=True)

    # Trend over time
    if "feedback_date" in df_fb.columns:
        df_fb["feedback_date"] = pd.to_datetime(df_fb["feedback_date"], errors="coerce")
        trend = df_fb.groupby(
            [df_fb["feedback_date"].dt.to_period("M"), "sentiment_label_rule"]
        ).size().reset_index(name="count")
        trend["feedback_date"] = trend["feedback_date"].astype(str)

        fig_trend = px.line(
            trend, x="feedback_date", y="count",
            color="sentiment_label_rule",
            title="Monthly Sentiment Trend",
            color_discrete_sequence=[PRIMARY, LIGHT, "#ff9999"]
        )
        st.plotly_chart(fig_trend, use_container_width=True)




