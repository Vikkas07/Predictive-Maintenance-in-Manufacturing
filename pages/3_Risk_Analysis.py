import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import shap
import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from utils import load_model
from config import FEATURE_COLUMNS, FAILURE_THRESHOLD

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
        .page-title { font-size: 1.5rem; font-weight: 700; color: #ffffff; margin-bottom: 0.2rem; }
        .page-sub   { font-size: 0.85rem; color: #8b92a5; margin-bottom: 1.5rem; }
        .section-header {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #8b92a5;
            border-bottom: 1px solid #1e2130;
            padding-bottom: 0.4rem;
            margin-bottom: 1rem;
            margin-top: 1.8rem;
        }
        .info-card {
            background: #0f1117;
            border: 1px solid #1e2130;
            border-radius: 6px;
            padding: 1rem 1.2rem;
        }
        .info-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #8b92a5;
            margin-bottom: 0.3rem;
        }
        .info-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #ffffff;
        }
        .info-sub { font-size: 0.75rem; color: #8b92a5; margin-top: 0.2rem; }
        .shap-row {
            display: flex;
            align-items: center;
            padding: 0.45rem 0;
            border-bottom: 1px solid #1e2130;
            font-size: 0.82rem;
        }
        .shap-row:last-child { border-bottom: none; }
        .shap-feature { color: #c8cdd8; width: 210px; flex-shrink: 0; }
        .shap-value-pos { color: #e63946; font-weight: 600; width: 70px; text-align: right; flex-shrink: 0; }
        .shap-value-neg { color: #2a9d8f; font-weight: 600; width: 70px; text-align: right; flex-shrink: 0; }
        .shap-bar-wrap { flex: 1; padding: 0 12px; }
        .machine-detail-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.82rem;
            color: #8b92a5;
            padding: 0.3rem 0;
            border-bottom: 1px solid #1e2130;
        }
        .machine-detail-row:last-child { border-bottom: none; }
        .machine-detail-val { color: #ffffff; font-weight: 500; }
    </style>
""", unsafe_allow_html=True)

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#8b92a5", size=11),
    margin=dict(l=0, r=0, t=28, b=0),
)

COLORS = {
    "Critical": "#e63946",
    "High":     "#f4a261",
    "Moderate": "#e9c46a",
    "Normal":   "#2a9d8f",
}

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
def score_fleet(_model, threshold, df):
    probs = _model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    df = df.copy()
    df["failure_prob"] = probs

    def classify(p):
        if p >= threshold:               return "Critical"
        elif p >= threshold * 0.6:       return "High"
        elif p >= threshold * 0.3:       return "Moderate"
        else:                            return "Normal"

    df["risk_level"] = df["failure_prob"].apply(classify)
    return df

@st.cache_resource
def get_explainer(_model):
    return shap.Explainer(_model)

# ── Load ───────────────────────────────────────────────────────────────────────
model, threshold, _ = load_model()
raw_df  = load_fleet()
fleet   = score_fleet(model, threshold, raw_df)
explainer = get_explainer(model)

FEATURE_LABELS = {
    "Air_temperature_K":     "Air Temperature (K)",
    "Process_temperature_K": "Process Temperature (K)",
    "Rotational_speed_rpm":  "Rotational Speed (rpm)",
    "Torque_Nm":             "Torque (Nm)",
    "Tool_wear_min":         "Tool Wear (min)",
    "temp_diff":             "Temperature Differential (K)",
    "Type_L":                "Machine Type: L",
    "Type_M":                "Machine Type: M",
}

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("<div class='page-title'>Risk Analysis & Explainability</div>", unsafe_allow_html=True)
st.markdown("<div class='page-sub'>Fleet-wide failure risk distribution and SHAP-based diagnostic explanation for individual machines.</div>", unsafe_allow_html=True)

# ── Fleet KPIs ─────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Fleet Risk Summary</div>", unsafe_allow_html=True)

risk_counts = fleet["risk_level"].value_counts()
k1, k2, k3, k4 = st.columns(4)

for col, level, border in zip(
    [k1, k2, k3, k4],
    ["Critical", "High", "Moderate", "Normal"],
    ["#e63946", "#f4a261", "#e9c46a", "#2a9d8f"]
):
    count = int(risk_counts.get(level, 0))
    pct   = round(count / len(fleet) * 100, 1)
    with col:
        st.markdown(f"""
            <div class='info-card' style='border-left: 3px solid {border};'>
                <div class='info-label'>{level}</div>
                <div class='info-value'>{count:,}</div>
                <div class='info-sub'>{pct}% of fleet</div>
            </div>
        """, unsafe_allow_html=True)

# ── Risk Distribution Charts ───────────────────────────────────────────────────
st.markdown("<div class='section-header'>Risk Distribution</div>", unsafe_allow_html=True)

ch1, ch2 = st.columns(2)

with ch1:
    st.markdown("<span style='color:#8b92a5; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em;'>Failure Probability — Fleet Histogram</span>", unsafe_allow_html=True)
    fig_hist = go.Figure(go.Histogram(
        x=fleet["failure_prob"],
        nbinsx=50,
        marker=dict(color="#e63946", opacity=0.75),
        hovertemplate="Probability: %{x:.2f}<br>Machines: %{y}<extra></extra>",
    ))
    fig_hist.add_vline(
        x=threshold, line_dash="dash", line_color="#f4a261", line_width=1.5,
        annotation_text=f"Threshold ({threshold:.0%})",
        annotation_font_color="#f4a261",
        annotation_font_size=10,
    )
    fig_hist.update_layout(
        **CHART_LAYOUT, height=260,
        xaxis=dict(title="Failure Probability", showgrid=False, color="#8b92a5"),
        yaxis=dict(title="Machines",            showgrid=True, gridcolor="#1e2130"),
    )
    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

with ch2:
    st.markdown("<span style='color:#8b92a5; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em;'>Risk Level — Sorted Machine Profile</span>", unsafe_allow_html=True)
    sorted_probs = fleet["failure_prob"].sort_values().reset_index(drop=True)
    colors_sorted = sorted_probs.apply(
        lambda p: "#e63946" if p >= threshold else
                  "#f4a261" if p >= threshold * 0.6 else
                  "#e9c46a" if p >= threshold * 0.3 else "#2a9d8f"
    )
    fig_sorted = go.Figure(go.Bar(
        x=list(range(len(sorted_probs))),
        y=sorted_probs,
        marker=dict(color=colors_sorted),
        hovertemplate="Machine rank: %{x}<br>Probability: %{y:.3f}<extra></extra>",
    ))
    fig_sorted.add_hline(
        y=threshold, line_dash="dash", line_color="#f4a261", line_width=1.5
    )
    fig_sorted.update_layout(
        **CHART_LAYOUT, height=260,
        xaxis=dict(title="Machine Index (sorted)", showgrid=False),
        yaxis=dict(title="Failure Probability", showgrid=True, gridcolor="#1e2130"),
    )
    st.plotly_chart(fig_sorted, use_container_width=True, config={"displayModeBar": False})

# ── Feature vs Risk Scatter ────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Sensor vs Failure Probability</div>", unsafe_allow_html=True)

sensor_options = {
    "Tool Wear (min)":         "Tool_wear_min",
    "Torque (Nm)":             "Torque_Nm",
    "Rotational Speed (rpm)":  "Rotational_speed_rpm",
    "Air Temperature (K)":     "Air_temperature_K",
    "Temperature Differential":"temp_diff",
}

sel_sensor = st.selectbox(
    "Select sensor to plot against failure probability",
    list(sensor_options.keys()),
    label_visibility="collapsed"
)
col = sensor_options[sel_sensor]

fig_scatter = px.scatter(
    fleet.sample(min(2000, len(fleet)), random_state=42),
    x=col,
    y="failure_prob",
    color="risk_level",
    color_discrete_map=COLORS,
    opacity=0.55,
    labels={col: sel_sensor, "failure_prob": "Failure Probability", "risk_level": "Risk Level"},
)
fig_scatter.add_hline(
    y=threshold, line_dash="dash", line_color="#f4a261", line_width=1.5,
    annotation_text="Decision threshold", annotation_font_color="#f4a261", annotation_font_size=10,
)
fig_scatter.update_layout(
    **CHART_LAYOUT, height=280,
    xaxis=dict(showgrid=True, gridcolor="#1e2130"),
    yaxis=dict(showgrid=True, gridcolor="#1e2130"),
    legend=dict(title="", orientation="h", y=-0.22),
)
st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})

# ── Individual Machine Explainability ──────────────────────────────────────────
st.markdown("<div class='section-header'>Individual Machine Diagnostic</div>", unsafe_allow_html=True)

left, right = st.columns([1, 1.6], gap="large")

with left:
    # Filter controls
    risk_filter = st.selectbox("Filter by risk level", ["All", "Critical", "High", "Moderate", "Normal"])
    subset = fleet if risk_filter == "All" else fleet[fleet["risk_level"] == risk_filter]

    machine_ids = subset["Product ID"].tolist()
    selected_id = st.selectbox("Select machine", machine_ids)

    machine_row = fleet[fleet["Product ID"] == selected_id].iloc[[0]]
    prob  = float(machine_row["failure_prob"].values[0])
    level = machine_row["risk_level"].values[0]

    level_color = COLORS[level]

    st.markdown(f"""
        <div class='info-card' style='border-left: 3px solid {level_color}; margin-top: 1rem;'>
            <div class='info-label'>Failure Probability</div>
            <div class='info-value' style='color:{level_color};'>{prob:.1%}</div>
            <div class='info-sub'>Risk classification: <b style='color:{level_color};'>{level}</b></div>
        </div>
    """, unsafe_allow_html=True)

    # Sensor readings
    sensor_display = {
        "Air Temperature":     f"{machine_row['Air_temperature_K'].values[0]:.1f} K",
        "Process Temperature": f"{machine_row['Process_temperature_K'].values[0]:.1f} K",
        "Temp Differential":   f"{machine_row['temp_diff'].values[0]:.1f} K",
        "Rotational Speed":    f"{int(machine_row['Rotational_speed_rpm'].values[0])} rpm",
        "Torque":              f"{machine_row['Torque_Nm'].values[0]:.1f} Nm",
        "Tool Wear":           f"{int(machine_row['Tool_wear_min'].values[0])} min",
        "Machine Type":        machine_row["Type"].values[0],
    }
    rows = "".join(
        f"<div class='machine-detail-row'><span>{k}</span><span class='machine-detail-val'>{v}</span></div>"
        for k, v in sensor_display.items()
    )
    st.markdown(f"""
        <div style='background:#0f1117; border:1px solid #1e2130; border-radius:6px; padding:1rem 1.2rem; margin-top:1rem;'>
            <div style='font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em; color:#8b92a5; margin-bottom:0.6rem;'>Sensor Readings</div>
            {rows}
        </div>
    """, unsafe_allow_html=True)

with right:
    with st.spinner("Computing SHAP values..."):
        shap_values = explainer(machine_row[FEATURE_COLUMNS])
        sv   = shap_values[0].values
        base = float(shap_values[0].base_values)
        feat_names = [FEATURE_LABELS[f] for f in FEATURE_COLUMNS]

        # Sort by absolute contribution
        order  = np.argsort(np.abs(sv))[::-1]
        sv_ord = sv[order]
        fn_ord = [feat_names[i] for i in order]
        fv_ord = [machine_row[FEATURE_COLUMNS].values[0][i] for i in order]

    st.markdown("<span style='color:#8b92a5; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em;'>SHAP Feature Contributions</span>", unsafe_allow_html=True)

    # Plotly waterfall
    measure = ["relative"] * len(sv_ord) + ["total"]
    x_labels = [f"{fn} = {fv:.2f}" if isinstance(fv, float) else f"{fn} = {fv}"
                for fn, fv in zip(fn_ord, fv_ord)] + ["Final Probability"]
    y_vals = list(sv_ord) + [base + sum(sv_ord)]

    bar_colors = [
        "#e63946" if v > 0 else "#2a9d8f"
        for v in sv_ord
    ] + ["#f4a261"]

    fig_shap = go.Figure(go.Bar(
        x=x_labels,
        y=[base] + list(np.cumsum(sv_ord)[:-1]) + [None],
        marker=dict(color="rgba(0,0,0,0)"),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig_shap = go.Figure()

    cumulative = base
    for i, (name, val, fval) in enumerate(zip(fn_ord, sv_ord, fv_ord)):
        label = f"{name} = {fval:.2f}" if isinstance(fval, float) else f"{name} = {fval}"
        fig_shap.add_trace(go.Bar(
            name=label,
            x=[label],
            y=[abs(val)],
            marker=dict(color="#e63946" if val > 0 else "#2a9d8f", opacity=0.85),
            hovertemplate=f"{label}<br>SHAP: {val:+.4f}<extra></extra>",
            showlegend=False,
        ))

    fig_shap.update_layout(
        **CHART_LAYOUT,
        height=320,
        barmode="group",
        xaxis=dict(showgrid=False, tickangle=-30, tickfont=dict(size=9)),
        yaxis=dict(title="Absolute SHAP Value", showgrid=True, gridcolor="#1e2130"),
        annotations=[
            dict(
                x=0.5, y=1.08, xref="paper", yref="paper",
                text=f"Base rate: {base:.3f} | Final: {base + sum(sv):.3f}",
                showarrow=False, font=dict(size=10, color="#8b92a5"),
            )
        ]
    )
    st.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar": False})

    # SHAP table
    shap_rows = ""
    max_abs = max(abs(sv_ord)) if len(sv_ord) > 0 else 1
    for name, val, fval in zip(fn_ord, sv_ord, fv_ord):
        fval_str  = f"{fval:.2f}" if isinstance(fval, float) else str(int(fval))
        pct       = abs(val) / max_abs * 100
        bar_color = "#e63946" if val > 0 else "#2a9d8f"
        direction = "Increases risk" if val > 0 else "Decreases risk"
        shap_rows += f"""
            <div class='shap-row'>
                <div class='shap-feature'>{name}<br><span style='font-size:0.72rem; color:#4a5568;'>Value: {fval_str}</span></div>
                <div class='shap-bar-wrap'>
                    <div style='background:{bar_color}; opacity:0.7; height:8px; border-radius:3px; width:{pct:.0f}%;'></div>
                </div>
                <div class='{'shap-value-pos' if val > 0 else 'shap-value-neg'}'>{val:+.4f}</div>
                <div style='font-size:0.7rem; color:#4a5568; width:110px; text-align:right;'>{direction}</div>
            </div>
        """
    st.markdown(f"""
        <div style='background:#0f1117; border:1px solid #1e2130; border-radius:6px; padding:1rem 1.2rem; margin-top:1rem;'>
            <div style='font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em; color:#8b92a5; margin-bottom:0.6rem;'>
                Contribution Breakdown &nbsp;|&nbsp; Red = increases risk &nbsp;|&nbsp; Teal = decreases risk
            </div>
            {shap_rows}
        </div>
    """, unsafe_allow_html=True)