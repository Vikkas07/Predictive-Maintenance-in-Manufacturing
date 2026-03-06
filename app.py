import streamlit as st
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

st.set_page_config(
    page_title="PredictaMaint | Industrial Maintenance Intelligence",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, p, div, span, label {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

/* ── Hide Streamlit default chrome ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header[data-testid="stHeader"] { display: none !important; }

/* ── Top bar ── */
.top-bar {
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 52px;
    background: #080a0f;
    border-bottom: 1px solid #1a1f2e;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 1.5rem 0 1.5rem;
    z-index: 999;
}
.top-bar-left { display: flex; flex-direction: column; justify-content: center; }
.top-bar-brand {
    font-size: 0.9rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.01em;
    line-height: 1.2;
}
.top-bar-sub {
    font-size: 0.65rem;
    color: #5a6072;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    line-height: 1.2;
}
.top-bar-right { display: flex; flex-direction: column; align-items: flex-end; justify-content: center; }
.top-bar-plant {
    font-size: 0.8rem;
    font-weight: 600;
    color: #c8cdd8;
    line-height: 1.3;
}
.top-bar-meta {
    font-size: 0.65rem;
    color: #5a6072;
    letter-spacing: 0.02em;
    line-height: 1.3;
}
.top-bar-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #2a9d8f;
    margin-right: 5px;
    vertical-align: middle;
}

/* ── Push content below top bar ── */
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 52px !important;
}
.main .block-container {
    padding-top: 72px !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 1600px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #080a0f !important;
    border-right: 1px solid #1a1f2e !important;
}
[data-testid="stSidebarNav"] a {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #5a6072 !important;
    padding: 0.55rem 1rem !important;
    border-radius: 5px !important;
    letter-spacing: 0.01em !important;
    display: flex !important;
    align-items: center !important;
}
[data-testid="stSidebarNav"] a:hover {
    background-color: #0f1117 !important;
    color: #c8cdd8 !important;
}
[data-testid="stSidebarNav"] a[aria-current="page"] {
    background-color: #0f1117 !important;
    color: #ffffff !important;
    border-left: 2px solid #e63946 !important;
    font-weight: 600 !important;
}

/* ── Sidebar brand block ── */
.sidebar-brand {
    padding: 1rem 1rem 0.75rem 1rem;
    border-bottom: 1px solid #1a1f2e;
    margin-bottom: 0.5rem;
}
.sidebar-brand-title {
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    margin: 0 !important;
}
.sidebar-brand-sub {
    font-size: 0.65rem !important;
    color: #5a6072 !important;
    margin: 0.15rem 0 0 0 !important;
    letter-spacing: 0.03em !important;
}

/* ── Sidebar system status ── */
.sys-status {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    padding: 0.75rem 1rem;
    border-top: 1px solid #1a1f2e;
    background: #080a0f;
}
.sys-status-title {
    font-size: 0.6rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: #3a3f52 !important;
    margin-bottom: 0.5rem !important;
}
.sys-status-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: #5a6072;
    padding: 0.18rem 0;
}
.sys-status-row b { color: #2a9d8f; font-weight: 500; }

/* ── Global metric / card styles ── */
[data-testid="stMetric"] {
    background: #0a0d14;
    border: 1px solid #1a1f2e;
    border-radius: 6px;
    padding: 0.9rem 1.1rem;
}
[data-testid="stMetricLabel"] {
    font-size: 0.68rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
    color: #5a6072 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.7rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
}

hr { border-color: #1a1f2e !important; }

/* ── Buttons ── */
.stButton > button {
    background: #e63946 !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}
.stButton > button:hover { background: #c1121f !important; }

/* ── Page section headers ── */
.dash-section-header {
    font-size: 0.65rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.13em !important;
    color: #3a3f52 !important;
    border-bottom: 1px solid #1a1f2e !important;
    padding-bottom: 0.4rem !important;
    margin-bottom: 1rem !important;
    margin-top: 1.6rem !important;
}

/* ── Page title standard ── */
.dash-page-title {
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    margin: 0 0 0.2rem 0 !important;
    letter-spacing: -0.02em !important;
    line-height: 1.15 !important;
}
.dash-page-sub {
    font-size: 0.8rem !important;
    color: #5a6072 !important;
    margin: 0 0 1.5rem 0 !important;
    line-height: 1.5 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Top bar (global, fixed) ────────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
    <div class="top-bar-left">
        <span class="top-bar-brand">PredictaMaint</span>
        <span class="top-bar-sub">Predictive Maintenance Intelligence</span>
    </div>
    <div class="top-bar-right">
        <span class="top-bar-plant">AI4I Manufacturing Dataset</span>
        <span class="top-bar-meta">
            <span class="top-bar-dot"></span>System Active &nbsp;&bull;&nbsp; Model: XGBoost &nbsp;&bull;&nbsp; Threshold: 30%
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar brand + system status ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <p class="sidebar-brand-title">PredictaMaint</p>
        <p class="sidebar-brand-sub">Maintenance Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sys-status">
        <div class="sys-status-title">System Status</div>
        <div class="sys-status-row"><span>Prediction Engine</span><b>Active</b></div>
        <div class="sys-status-row"><span>Data Pipeline</span><b>Active</b></div>
        <div class="sys-status-row"><span>SHAP Explainer</span><b>Active</b></div>
    </div>
    """, unsafe_allow_html=True)

pg = st.navigation([
    st.Page("pages/1_Overview.py",          title="Overview"),
    st.Page("pages/2_Live_Prediction.py",   title="Live Prediction"),
    st.Page("pages/3_Risk_Analysis.py",     title="Risk Analysis"),
    st.Page("pages/4_Maintenance_Planner.py", title="Maintenance Planner"),
])

pg.run()