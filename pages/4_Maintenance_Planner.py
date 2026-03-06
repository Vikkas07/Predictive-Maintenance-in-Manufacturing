import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os
from datetime import date, timedelta

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from utils import load_model
from config import FEATURE_COLUMNS

# ── Page-level styles ──────────────────────────────────────────────────────────
st.markdown("""
<style>
/* KPI tiles */
.kpi-tile {
    background: #0a0d14;
    border: 1px solid #1a1f2e;
    border-radius: 7px;
    padding: 1rem 1.2rem 0.9rem;
    height: 100%;
}
.kpi-tile-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.55rem;
}
.kpi-tile-label {
    font-size: 0.72rem !important;
    color: #5a6072 !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
}
.kpi-dot-c { width:8px;height:8px;border-radius:50%;background:#e63946;flex-shrink:0; }
.kpi-dot-h { width:8px;height:8px;border-radius:50%;background:#f4a261;flex-shrink:0; }
.kpi-dot-m { width:8px;height:8px;border-radius:50%;background:#e9c46a;flex-shrink:0; }
.kpi-dot-n { width:8px;height:8px;border-radius:50%;background:#5a6072;flex-shrink:0; }
.kpi-tile-value {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    line-height: 1 !important;
    margin-bottom: 0.2rem !important;
}
.kpi-tile-sub { font-size: 0.7rem !important; color: #5a6072 !important; }

/* Section header */
.s-hdr {
    font-size: 0.63rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.13em !important;
    color: #3a3f52 !important;
    border-bottom: 1px solid #1a1f2e !important;
    padding-bottom: 0.38rem !important;
    margin-bottom: 0.9rem !important;
    margin-top: 1.5rem !important;
}

/* Filter bar */
.filter-bar {
    background: #0a0d14;
    border: 1px solid #1a1f2e;
    border-radius: 7px;
    padding: 0.7rem 1.1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1rem;
}
.filter-bar-label {
    font-size: 0.72rem !important;
    color: #5a6072 !important;
    display: flex;
    align-items: center;
    gap: 0.35rem;
    white-space: nowrap;
}

/* Queue header */
.q-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.65rem;
}
.q-title {
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    color: #ffffff !important;
}
.q-sub { font-size: 0.72rem !important; color: #5a6072 !important; }

/* Task card — matches reference */
.tc {
    background: #0a0d14;
    border: 1px solid #1a1f2e;
    border-radius: 7px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
}
.tc-top {
    display: flex;
    align-items: flex-start;
    gap: 0.85rem;
}
.rank-col { display:flex; flex-direction:column; align-items:center; flex-shrink:0; gap:3px; }
.rb {
    width: 34px; height: 34px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: 700; flex-shrink: 0;
}
.rb-C { background:rgba(230,57,70,0.15); color:#e63946; border:1.5px solid rgba(230,57,70,0.5); }
.rb-H { background:rgba(244,162,97,0.15); color:#f4a261; border:1.5px solid rgba(244,162,97,0.5); }
.rb-M { background:rgba(233,196,106,0.13); color:#e9c46a; border:1.5px solid rgba(233,196,106,0.42); }
.rb-N { background:rgba(42,157,143,0.12); color:#2a9d8f; border:1.5px solid rgba(42,157,143,0.4); }
.pl-C { font-size:0.55rem !important; font-weight:700 !important; letter-spacing:0.1em !important; text-transform:uppercase !important; color:#e63946 !important; text-align:center; }
.pl-H { font-size:0.55rem !important; font-weight:700 !important; letter-spacing:0.1em !important; text-transform:uppercase !important; color:#f4a261 !important; text-align:center; }
.pl-M { font-size:0.55rem !important; font-weight:700 !important; letter-spacing:0.1em !important; text-transform:uppercase !important; color:#e9c46a !important; text-align:center; }
.pl-N { font-size:0.55rem !important; font-weight:700 !important; letter-spacing:0.1em !important; text-transform:uppercase !important; color:#2a9d8f !important; text-align:center; }

.tc-body { flex: 1; min-width: 0; }
.tc-name {
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    color: #ffffff !important;
    margin: 0 0 0.15rem !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.tc-meta-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.72rem !important;
    color: #5a6072 !important;
    margin-bottom: 0.5rem;
}
.tc-meta-row .sep { color: #1a1f2e; }
.tc-meta-row b { color: #8b92a5 !important; font-weight: 500 !important; }
.tc-time { font-size:0.72rem !important; color:#5a6072 !important; white-space:nowrap; flex-shrink:0; margin-top:3px; }

.tc-desc {
    font-size: 0.78rem !important;
    color: #8b92a5 !important;
    line-height: 1.55 !important;
    margin: 0.5rem 0 0.6rem 3rem !important;
}

/* Risk warning bar */
.rwb-C {
    background: rgba(230,57,70,0.08);
    border: 1px solid rgba(230,57,70,0.25);
    border-radius: 4px;
    padding: 0.35rem 0.7rem;
    font-size: 0.72rem !important;
    color: #e63946 !important;
    margin: 0 0 0.6rem 3rem !important;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.rwb-H {
    background: rgba(244,162,97,0.07);
    border: 1px solid rgba(244,162,97,0.22);
    border-radius: 4px;
    padding: 0.35rem 0.7rem;
    font-size: 0.72rem !important;
    color: #f4a261 !important;
    margin: 0 0 0.6rem 3rem !important;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.rwb-M {
    background: rgba(233,196,106,0.06);
    border: 1px solid rgba(233,196,106,0.18);
    border-radius: 4px;
    padding: 0.35rem 0.7rem;
    font-size: 0.72rem !important;
    color: #e9c46a !important;
    margin: 0 0 0.6rem 3rem !important;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* Task footer row */
.tc-footer {
    display: flex;
    flex-wrap: wrap;
    gap: 1.2rem;
    font-size: 0.7rem !important;
    color: #5a6072 !important;
    margin-left: 3rem;
    padding-top: 0.5rem;
    border-top: 1px solid #1a1f2e;
}
.tc-footer span b { color: #c8cdd8 !important; font-weight: 600 !important; }

/* Sensor flags */
.sv-w { color: #f4a261 !important; font-weight: 700 !important; }
.sv-c { color: #e63946 !important; font-weight: 700 !important; }

/* Export button - ghost */
.stDownloadButton > button {
    background: transparent !important;
    color: #5a6072 !important;
    border: 1px solid #1a1f2e !important;
    border-radius: 5px !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    padding: 0.28rem 0.75rem !important;
    letter-spacing: 0.02em !important;
    line-height: 1.5 !important;
}
.stDownloadButton > button:hover {
    border-color: #3a3f52 !important;
    color: #c8cdd8 !important;
    background: #0f1117 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#5a6072", size=10),
    margin=dict(l=0, r=0, t=30, b=0),
)

SENSOR_RANGES = {
    "Air_temperature_K":     (295, 304),
    "Process_temperature_K": (305, 314),
    "Rotational_speed_rpm":  (1168, 2886),
    "Torque_Nm":             (3.8, 76.6),
    "Tool_wear_min":         (0, 253),
}

RISK_MSG = {
    "Critical": "Model confidence exceeds decision threshold. Sensor telemetry indicates imminent failure. Halt operation immediately.",
    "High":     "Readings approach failure boundary. Elevated torque or thermal stress detected by model.",
    "Moderate": "Moderate sensor deviation detected. Proactive inspection recommended within 7 days.",
}

ACTIONS = {
    "Critical": "Halt machine. Dispatch maintenance team immediately. Inspect tool wear, torque load, and thermal profile. Replace components as required.",
    "High":     "Schedule inspection within 24 hours. Prepare replacement tooling. Assign technician and verify lubrication status.",
    "Moderate": "Schedule inspection within 7 days. Review tool wear trend and operating load history.",
    "Normal":   "Continue normal operation. Readings within acceptable range. Review per scheduled maintenance interval.",
}

EST_H = {"Critical": "4–6h", "High": "2–4h", "Moderate": "1–2h", "Normal": "0.5h"}

FAILURE_TYPES = {
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failure",
}

LEVEL_CODES = {"Critical": "C", "High": "H", "Moderate": "M", "Normal": "N"}

# ── Data ───────────────────────────────────────────────────────────────────────
@st.cache_data
def load_fleet():
    base = os.path.dirname(os.path.dirname(__file__))
    df = pd.read_csv(os.path.join(base, "data", "step1_clean.csv"))
    df["Type_L"] = (df["Type"] == "L").astype(int)
    df["Type_M"] = (df["Type"] == "M").astype(int)
    df = df.rename(columns={
        "Air temperature [K]":     "Air_temperature_K",
        "Process temperature [K]": "Process_temperature_K",
        "Rotational speed [rpm]":  "Rotational_speed_rpm",
        "Torque [Nm]":             "Torque_Nm",
        "Tool wear [min]":         "Tool_wear_min",
    })
    return df

@st.cache_data
def build_schedule(_model, threshold, df):
    probs = _model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    df = df.copy()
    df["failure_prob"] = probs

    def classify(p):
        if p >= threshold:         return "Critical"
        elif p >= threshold * 0.6: return "High"
        elif p >= threshold * 0.3: return "Moderate"
        return "Normal"

    today = date.today()
    due_map = {
        "Critical": today,
        "High":     today + timedelta(days=1),
        "Moderate": today + timedelta(days=7),
        "Normal":   today + timedelta(days=30),
    }
    df["risk_level"]      = df["failure_prob"].apply(classify)
    df["maintenance_due"] = df["risk_level"].map(due_map)

    failure_cols = ["TWF","HDF","PWF","OSF","RNF"]
    def dom_failure(row):
        flags = [FAILURE_TYPES[c] for c in failure_cols if c in row and row[c] == 1]
        return ", ".join(flags) if flags else "None detected"
    df["failure_type"] = df.apply(dom_failure, axis=1)

    df = df.sort_values("failure_prob", ascending=False).reset_index(drop=True)
    df["priority_rank"] = df.index + 1
    return df

model, threshold, _ = load_model()
raw_df   = load_fleet()
schedule = build_schedule(model, threshold, raw_df)
rc       = schedule["risk_level"].value_counts()

# ── Export CSV ─────────────────────────────────────────────────────────────────
export_df = schedule[[
    "priority_rank","Product ID","Type","risk_level","failure_prob",
    "maintenance_due","failure_type","Tool_wear_min","Torque_Nm","Air_temperature_K"
]].rename(columns={
    "priority_rank":"Rank","Product ID":"Machine ID","risk_level":"Risk Level",
    "failure_prob":"Failure Probability","maintenance_due":"Maintenance Due",
    "failure_type":"Failure Type","Tool_wear_min":"Tool Wear (min)",
    "Torque_Nm":"Torque (Nm)","Air_temperature_K":"Air Temp (K)"
}).copy()
export_df["Failure Probability"] = (export_df["Failure Probability"]*100).round(2).astype(str)+"%"

# ── PAGE HEADER ────────────────────────────────────────────────────────────────
# Use st.header so Streamlit controls the font rendering — no custom CSS class needed
h_col, btn_col = st.columns([5, 1])
with h_col:
    st.markdown("## Maintenance Planner")
    st.markdown(
        "<p style='font-size:0.82rem;color:#5a6072;margin:-0.8rem 0 1.2rem 0;'>Risk-ranked service queue generated from ML model predictions across the monitored fleet.</p>",
        unsafe_allow_html=True
    )
with btn_col:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.download_button(
        "↓ Export CSV",
        data=export_df.to_csv(index=False),
        file_name=f"maintenance_plan_{date.today().isoformat()}.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ── KPI TILES ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
tiles = [
    ("Critical Priority",  rc.get("Critical",0),  "Halt & inspect immediately",    "kpi-dot-c"),
    ("High Priority",      rc.get("High",0),       "Inspect within 24 hours",       "kpi-dot-h"),
    ("Moderate Priority",  rc.get("Moderate",0),   "Schedule within 7 days",        "kpi-dot-m"),
    ("Normal / Routine",   rc.get("Normal",0),     "Standard maintenance interval", "kpi-dot-n"),
]
for col, (label, val, sub, dot) in zip([k1,k2,k3,k4], tiles):
    with col:
        st.markdown(f"""
        <div class='kpi-tile'>
            <div class='kpi-tile-header'>
                <span class='kpi-tile-label'>{label}</span>
                <span class='{dot}'></span>
            </div>
            <div class='kpi-tile-value'>{val:,}</div>
            <div class='kpi-tile-sub'>{sub}</div>
        </div>""", unsafe_allow_html=True)

# ── WORKLOAD CHARTS ────────────────────────────────────────────────────────────
st.markdown("<div class='s-hdr'>Workload Analysis</div>", unsafe_allow_html=True)
ch1, ch2 = st.columns(2)

with ch1:
    fig_wl = go.Figure(go.Bar(
        x=["Today","24 h","7 days","30+ days"],
        y=[rc.get("Critical",0), rc.get("High",0), rc.get("Moderate",0), rc.get("Normal",0)],
        marker=dict(color=["#e63946","#f4a261","#e9c46a","#2a9d8f"], opacity=0.85),
        hovertemplate="%{x}<br>Machines: %{y}<extra></extra>",
    ))
    fig_wl.update_layout(
        **CHART_LAYOUT, height=200,
        xaxis=dict(showgrid=False, tickfont=dict(size=10, color="#5a6072")),
        yaxis=dict(showgrid=True, gridcolor="#1a1f2e", title="Machines", tickfont=dict(color="#5a6072")),
        title=dict(text="Machines by Maintenance Window", font=dict(size=11, color="#5a6072"), x=0),
    )
    st.plotly_chart(fig_wl, use_container_width=True, config={"displayModeBar": False})

with ch2:
    tr = schedule.groupby("Type")["failure_prob"].mean().reset_index()
    fig_type = go.Figure(go.Bar(
        x=tr["Type"], y=(tr["failure_prob"]*100).round(2),
        marker=dict(color="#e63946", opacity=0.8),
        hovertemplate="Type %{x}<br>Avg Risk: %{y:.2f}%<extra></extra>",
    ))
    fig_type.update_layout(
        **CHART_LAYOUT, height=200,
        xaxis=dict(title="Machine Type", showgrid=False, tickfont=dict(color="#5a6072")),
        yaxis=dict(title="Avg Failure Prob (%)", showgrid=True, gridcolor="#1a1f2e", tickfont=dict(color="#5a6072")),
        title=dict(text="Avg Predicted Risk by Machine Type", font=dict(size=11, color="#5a6072"), x=0),
    )
    st.plotly_chart(fig_type, use_container_width=True, config={"displayModeBar": False})

# ── FILTERS ────────────────────────────────────────────────────────────────────
st.markdown("<div class='s-hdr'>Filters</div>", unsafe_allow_html=True)

f1, f2, f3 = st.columns([1,1,1])
with f1:
    risk_filter = st.selectbox("Priority", ["All","Critical","High","Moderate","Normal"])
with f2:
    type_filter = st.selectbox("Machine Type", ["All","L","M","H"])
with f3:
    top_n = st.selectbox("Show", [25,50,100,250], index=1, format_func=lambda x: f"Top {x} machines")

filtered = schedule.copy()
if risk_filter != "All":
    filtered = filtered[filtered["risk_level"] == risk_filter]
if type_filter != "All":
    filtered = filtered[filtered["Type"] == type_filter]
filtered = filtered.head(top_n).reset_index(drop=True)

# ── QUEUE / TABLE HEADER ───────────────────────────────────────────────────────
st.markdown("<div class='s-hdr'>Maintenance Queue</div>", unsafe_allow_html=True)

qc1, qc2 = st.columns([3,1])
with qc1:
    st.markdown(
        f"<p style='font-size:0.73rem;color:#5a6072;margin:0 0 0.7rem 0;'>"
        f"Showing <b style='color:#8b92a5'>{len(filtered)}</b> machines — sorted by predicted failure probability.</p>",
        unsafe_allow_html=True
    )
with qc2:
    view_mode = st.selectbox("view", ["☰  Queue", "⊞  Table"], label_visibility="collapsed")

# ── TABLE VIEW ─────────────────────────────────────────────────────────────────
if "Table" in view_mode:
    tdf = filtered[[
        "priority_rank","Product ID","Type","risk_level","failure_prob",
        "maintenance_due","failure_type","Tool_wear_min","Torque_Nm","Air_temperature_K"
    ]].rename(columns={
        "priority_rank":"Rank","Product ID":"Machine ID","risk_level":"Risk Level",
        "failure_prob":"Failure Prob %","maintenance_due":"Due Date",
        "failure_type":"Failure Type","Tool_wear_min":"Wear (min)",
        "Torque_Nm":"Torque (Nm)","Air_temperature_K":"Air Temp (K)"
    }).copy()
    tdf["Failure Prob %"] = (tdf["Failure Prob %"]*100).round(1).astype(str)+"%"
    tdf["Wear (min)"]     = tdf["Wear (min)"].astype(int)
    tdf["Torque (Nm)"]    = tdf["Torque (Nm)"].round(1)
    tdf["Air Temp (K)"]   = tdf["Air Temp (K)"].round(1)
    tdf["Due Date"]       = tdf["Due Date"].astype(str)
    st.dataframe(tdf, use_container_width=True, hide_index=True,
        column_config={
            "Rank":        st.column_config.NumberColumn("Rank", width="small"),
            "Failure Type":st.column_config.TextColumn("Failure Type", width="large"),
        }
    )

# ── QUEUE VIEW ─────────────────────────────────────────────────────────────────
else:
    def sv_cls(key, val):
        lo, hi = SENSOR_RANGES[key]
        if val > hi*1.05 or val < lo*0.95: return "sv-c"
        if val > hi or val < lo:           return "sv-w"
        return ""

    for _, row in filtered.iterrows():
        rank   = int(row["priority_rank"])
        lvl    = row["risk_level"]
        lc     = LEVEL_CODES[lvl]
        prob   = float(row["failure_prob"])
        mid    = row["Product ID"]
        mtype  = row["Type"]
        wear   = int(row["Tool_wear_min"])
        torque = round(float(row["Torque_Nm"]), 1)
        air_t  = round(float(row["Air_temperature_K"]), 1)
        proc_t = round(float(row["Process_temperature_K"]), 1)
        speed  = int(row["Rotational_speed_rpm"])
        due    = str(row["maintenance_due"])
        ftype  = row["failure_type"]

        risk_bar = ""
        if lvl in RISK_MSG:
            risk_bar = f"<div class='rwb-{lc}'><span>&#9888;</span> Risk if delayed: {RISK_MSG[lvl]}</div>"

        footer_items = [
            f"Machine ID: <b>{mid}</b>",
            f"Type: <b>{mtype}</b>",
            f"Scheduled: <b>{due}</b>",
            f"Est. Time: <b>{EST_H[lvl]}</b>",
            f"Tool Wear: <b class='{sv_cls('Tool_wear_min', wear)}'>{wear} min</b>",
            f"Torque: <b class='{sv_cls('Torque_Nm', torque)}'>{torque} Nm</b>",
            f"Air Temp: <b class='{sv_cls('Air_temperature_K', air_t)}'>{air_t} K</b>",
            f"Process Temp: <b>{proc_t} K</b>",
            f"Speed: <b class='{sv_cls('Rotational_speed_rpm', speed)}'>{speed} rpm</b>",
            f"Failure Type: <b>{ftype}</b>",
        ]
        footer_html = "".join(f"<span>{i}</span>" for i in footer_items)

        st.markdown(f"""
        <div class='tc'>
            <div class='tc-top'>
                <div class='rank-col'>
                    <div class='rb rb-{lc}'>{rank}</div>
                    <div class='pl-{lc}'>{lvl}</div>
                </div>
                <div class='tc-body'>
                    <div class='tc-name'>{mid} &mdash; Type {mtype}</div>
                    <div class='tc-meta-row'>
                        <span>&#9651; Failure Probability: <b>{prob:.1%}</b></span>
                        <span class='sep'>|</span>
                        <span>Scheduled: <b>{due}</b></span>
                        <span class='sep'>|</span>
                        <span>{ftype}</span>
                    </div>
                </div>
                <div class='tc-time'>&#9203; {EST_H[lvl]}</div>
            </div>
            <div class='tc-desc'>{ACTIONS[lvl]}</div>
            {risk_bar}
            <div class='tc-footer'>{footer_html}</div>
        </div>
        """, unsafe_allow_html=True)