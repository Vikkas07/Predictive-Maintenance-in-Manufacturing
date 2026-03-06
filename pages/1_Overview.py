import streamlit as st
import pandas as pd
import numpy as np
import sys, os
import plotly.graph_objects as go
import plotly.express as px

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from config import MODEL_PATH, FAILURE_THRESHOLD
from utils import load_model, predict_failure

# ── Data loader ────────────────────────────────────────────────────────────────
@st.cache_data
def load_fleet_data():
    base = os.path.dirname(os.path.dirname(__file__))
    df = pd.read_csv(os.path.join(base, "data", "step1_clean.csv"))

    # Encode Type for model
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
def compute_fleet_risk(_model, _threshold, df):
    from config import FEATURE_COLUMNS
    features = df[FEATURE_COLUMNS]
    probs = _model.predict_proba(features)[:, 1]
    df = df.copy()
    df["failure_prob"] = probs

    def classify(p):
        if p >= _threshold:
            return "Critical"
        elif p >= _threshold * 0.6:
            return "High"
        elif p >= _threshold * 0.3:
            return "Moderate"
        else:
            return "Normal"

    df["risk_level"] = df["failure_prob"].apply(classify)
    return df

# ── Plot helpers ───────────────────────────────────────────────────────────────
COLORS = {
    "Critical": "#e63946",
    "High":     "#f4a261",
    "Moderate": "#e9c46a",
    "Normal":   "#2a9d8f",
}

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#8b92a5", size=11),
    margin=dict(l=0, r=0, t=28, b=0),
)

def risk_donut(counts):
    labels = list(counts.keys())
    values = list(counts.values())
    colors = [COLORS[l] for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.72,
        marker=dict(colors=colors, line=dict(color="#0a0c10", width=2)),
        textinfo="none",
        hovertemplate="%{label}: %{value} machines (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        showlegend=True,
        legend=dict(orientation="v", x=1.02, y=0.5,
                    font=dict(size=11, color="#8b92a5")),
        height=220,
        annotations=[dict(
            text=f"<b>{sum(values)}</b><br><span style='font-size:10px'>Machines</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#ffffff"),
        )]
    )
    return fig

def failure_type_bar(df):
    types = {
        "Tool Wear (TWF)":     df["TWF"].sum(),
        "Heat Dissipation (HDF)": df["HDF"].sum(),
        "Power Failure (PWF)": df["PWF"].sum(),
        "Overstrain (OSF)":    df["OSF"].sum(),
        "Random (RNF)":        df["RNF"].sum(),
    }
    fig = go.Figure(go.Bar(
        x=list(types.values()),
        y=list(types.keys()),
        orientation="h",
        marker=dict(color="#e63946", opacity=0.85),
        hovertemplate="%{y}: %{x} failures<extra></extra>",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        height=220,
        xaxis=dict(showgrid=True, gridcolor="#1e2130", zeroline=False),
        yaxis=dict(showgrid=False),
    )
    return fig

def prob_histogram(df):
    fig = go.Figure(go.Histogram(
        x=df["failure_prob"],
        nbinsx=40,
        marker=dict(color="#e63946", opacity=0.75),
        hovertemplate="Probability: %{x:.2f}<br>Count: %{y}<extra></extra>",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        height=200,
        xaxis=dict(title="Failure Probability", showgrid=False, color="#8b92a5"),
        yaxis=dict(title="Machines",            showgrid=True, gridcolor="#1e2130"),
    )
    return fig

def sensor_box(df):
    sensors = ["Air_temperature_K", "Torque_Nm", "Tool_wear_min", "Rotational_speed_rpm"]
    labels  = ["Air Temp (K)", "Torque (Nm)", "Tool Wear (min)", "Speed (rpm)"]

    fig = go.Figure()
    for s, l in zip(sensors, labels):
        fig.add_trace(go.Box(
            y=df[s], name=l,
            marker_color="#e63946",
            line_color="#e63946",
            fillcolor="rgba(230,57,70,0.12)",
            boxmean=True,
        ))
    fig.update_layout(
        **CHART_LAYOUT,
        height=240,
        showlegend=False,
        yaxis=dict(showgrid=True, gridcolor="#1e2130"),
        xaxis=dict(showgrid=False),
    )
    return fig

def machine_type_risk(df):
    grp = df.groupby(["Type", "risk_level"]).size().reset_index(name="count")
    fig = px.bar(
        grp, x="Type", y="count", color="risk_level",
        color_discrete_map=COLORS,
        barmode="stack",
    )
    fig.update_layout(
        **CHART_LAYOUT,
        height=220,
        xaxis=dict(title="Machine Type", showgrid=False, color="#8b92a5"),
        yaxis=dict(title="Count", showgrid=True, gridcolor="#1e2130"),
        legend=dict(title="", orientation="h", y=-0.25, font=dict(size=10)),
    )
    return fig

# ── Page ───────────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
        .kpi-card {
            background: #0f1117;
            border: 1px solid #1e2130;
            border-radius: 6px;
            padding: 1.1rem 1.4rem;
        }
        .kpi-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #8b92a5;
            margin-bottom: 0.3rem;
        }
        .kpi-value {
            font-size: 2rem;
            font-weight: 700;
            color: #ffffff;
            line-height: 1.1;
        }
        .kpi-sub {
            font-size: 0.72rem;
            color: #8b92a5;
            margin-top: 0.25rem;
        }
        .kpi-critical { border-left: 3px solid #e63946; }
        .kpi-warning  { border-left: 3px solid #f4a261; }
        .kpi-ok       { border-left: 3px solid #2a9d8f; }
        .kpi-neutral  { border-left: 3px solid #4a5568; }
        .section-header {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #8b92a5;
            border-bottom: 1px solid #1e2130;
            padding-bottom: 0.4rem;
            margin-bottom: 1rem;
            margin-top: 1.8rem;
        }
        .status-dot-ok       { display:inline-block; width:8px; height:8px; border-radius:50%; background:#2a9d8f; margin-right:6px; }
        .status-dot-critical { display:inline-block; width:8px; height:8px; border-radius:50%; background:#e63946; margin-right:6px; }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("## Fleet Monitoring Overview")
st.markdown(
    "<span style='color:#8b92a5; font-size:0.85rem;'>Real-time risk intelligence across the monitored machine population</span>",
    unsafe_allow_html=True
)

# Load data
with st.spinner("Loading fleet data..."):
    model, threshold, _ = load_model()
    raw_df = load_fleet_data()
    df = compute_fleet_risk(model, threshold, raw_df)

total      = len(df)
failures   = int(df["Machine failure"].sum())
failure_pct = round(failures / total * 100, 1)
avg_risk   = round(df["failure_prob"].mean() * 100, 1)

risk_counts = df["risk_level"].value_counts().to_dict()
for lvl in ["Critical", "High", "Moderate", "Normal"]:
    risk_counts.setdefault(lvl, 0)

critical_count = risk_counts["Critical"]
high_count     = risk_counts["High"]

# ── KPI Row ────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Key Performance Indicators</div>", unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(f"""
        <div class='kpi-card kpi-neutral'>
            <div class='kpi-label'>Total Machines</div>
            <div class='kpi-value'>{total:,}</div>
            <div class='kpi-sub'>Monitored fleet</div>
        </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
        <div class='kpi-card kpi-critical'>
            <div class='kpi-label'>Critical Risk</div>
            <div class='kpi-value'>{critical_count:,}</div>
            <div class='kpi-sub'>Immediate action required</div>
        </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""
        <div class='kpi-card kpi-warning'>
            <div class='kpi-label'>High Risk</div>
            <div class='kpi-value'>{high_count:,}</div>
            <div class='kpi-sub'>Schedule inspection</div>
        </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""
        <div class='kpi-card kpi-ok'>
            <div class='kpi-label'>Fleet Failure Rate</div>
            <div class='kpi-value'>{failure_pct}%</div>
            <div class='kpi-sub'>{failures} confirmed failures</div>
        </div>""", unsafe_allow_html=True)

with k5:
    st.markdown(f"""
        <div class='kpi-card kpi-neutral'>
            <div class='kpi-label'>Avg Failure Probability</div>
            <div class='kpi-value'>{avg_risk}%</div>
            <div class='kpi-sub'>Across all machines</div>
        </div>""", unsafe_allow_html=True)

# ── Risk Distribution + Failure Types ─────────────────────────────────────────
st.markdown("<div class='section-header'>Risk Distribution</div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns([1.1, 1.2, 1.2])

with c1:
    st.markdown("<span style='color:#8b92a5; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em;'>Fleet Risk Breakdown</span>", unsafe_allow_html=True)
    st.plotly_chart(risk_donut(risk_counts), use_container_width=True, config={"displayModeBar": False})

with c2:
    st.markdown("<span style='color:#8b92a5; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em;'>Failure Type Distribution</span>", unsafe_allow_html=True)
    st.plotly_chart(failure_type_bar(df), use_container_width=True, config={"displayModeBar": False})

with c3:
    st.markdown("<span style='color:#8b92a5; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em;'>Risk by Machine Type</span>", unsafe_allow_html=True)
    st.plotly_chart(machine_type_risk(df), use_container_width=True, config={"displayModeBar": False})

# ── Sensor Health + Probability Distribution ───────────────────────────────────
st.markdown("<div class='section-header'>Sensor Telemetry & Model Output</div>", unsafe_allow_html=True)

s1, s2 = st.columns([1.4, 1])

with s1:
    st.markdown("<span style='color:#8b92a5; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em;'>Sensor Value Distribution (Fleet)</span>", unsafe_allow_html=True)
    st.plotly_chart(sensor_box(df), use_container_width=True, config={"displayModeBar": False})

with s2:
    st.markdown("<span style='color:#8b92a5; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em;'>Failure Probability Histogram</span>", unsafe_allow_html=True)
    st.plotly_chart(prob_histogram(df), use_container_width=True, config={"displayModeBar": False})

# ── Critical Machines Table ────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Critical & High Risk Machines</div>", unsafe_allow_html=True)

alert_df = df[df["risk_level"].isin(["Critical", "High"])].copy()
alert_df = alert_df[[
    "Product ID", "Type", "risk_level", "failure_prob",
    "Air_temperature_K", "Torque_Nm", "Tool_wear_min"
]].rename(columns={
    "Product ID":        "Machine ID",
    "Type":              "Type",
    "risk_level":        "Risk Level",
    "failure_prob":      "Failure Probability",
    "Air_temperature_K": "Air Temp (K)",
    "Torque_Nm":         "Torque (Nm)",
    "Tool_wear_min":     "Tool Wear (min)",
})
alert_df["Failure Probability"] = (alert_df["Failure Probability"] * 100).round(1).astype(str) + "%"
alert_df = alert_df.sort_values("Risk Level").reset_index(drop=True)

if len(alert_df) == 0:
    st.markdown(
        "<span style='color:#2a9d8f;'><span class='status-dot-ok'></span>No critical or high risk machines detected.</span>",
        unsafe_allow_html=True
    )
else:
    st.dataframe(
        alert_df.head(50),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown(
        f"<span style='color:#8b92a5; font-size:0.75rem;'>Showing top 50 of {len(alert_df)} machines requiring attention.</span>",
        unsafe_allow_html=True
    )

# ── System Status ──────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>System Status</div>", unsafe_allow_html=True)

sc1, sc2, sc3 = st.columns(3)
with sc1:
    st.markdown("<span class='status-dot-ok'></span><span style='color:#8b92a5; font-size:0.82rem;'>Prediction Engine &nbsp; <b style='color:#2a9d8f;'>Operational</b></span>", unsafe_allow_html=True)
with sc2:
    st.markdown("<span class='status-dot-ok'></span><span style='color:#8b92a5; font-size:0.82rem;'>Data Pipeline &nbsp; <b style='color:#2a9d8f;'>Operational</b></span>", unsafe_allow_html=True)
with sc3:
    st.markdown("<span class='status-dot-ok'></span><span style='color:#8b92a5; font-size:0.82rem;'>SHAP Explainer &nbsp; <b style='color:#2a9d8f;'>Operational</b></span>", unsafe_allow_html=True)