import streamlit as st
import plotly.graph_objects as go
import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from utils import load_model, prepare_input, predict_failure, maintenance_decision

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
        .page-title { font-size: 1.5rem; font-weight: 700; color: #ffffff; margin-bottom: 0.2rem; }
        .page-sub   { font-size: 0.85rem; color: #8b92a5; margin-bottom: 1.5rem; }

        .input-panel {
            background: #0f1117;
            border: 1px solid #1e2130;
            border-radius: 6px;
            padding: 1.4rem 1.6rem;
        }
        .panel-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #8b92a5;
            border-bottom: 1px solid #1e2130;
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
        }

        .result-card {
            background: #0f1117;
            border: 1px solid #1e2130;
            border-radius: 6px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1rem;
        }
        .result-critical { border-left: 4px solid #e63946; }
        .result-warning  { border-left: 4px solid #f4a261; }
        .result-ok       { border-left: 4px solid #2a9d8f; }

        .decision-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #8b92a5;
            margin-bottom: 0.4rem;
        }
        .decision-value-critical { font-size: 1.3rem; font-weight: 700; color: #e63946; }
        .decision-value-warning  { font-size: 1.3rem; font-weight: 700; color: #f4a261; }
        .decision-value-ok       { font-size: 1.3rem; font-weight: 700; color: #2a9d8f; }

        .prob-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #8b92a5;
            margin-bottom: 0.3rem;
        }
        .prob-value {
            font-size: 2.8rem;
            font-weight: 700;
            color: #ffffff;
            line-height: 1;
        }

        .action-box {
            background: #12151f;
            border: 1px solid #1e2130;
            border-radius: 6px;
            padding: 1rem 1.2rem;
            margin-top: 1rem;
        }
        .action-title {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #8b92a5;
            margin-bottom: 0.6rem;
        }
        .action-item {
            font-size: 0.82rem;
            color: #c8cdd8;
            padding: 0.25rem 0;
            border-bottom: 1px solid #1e2130;
        }
        .action-item:last-child { border-bottom: none; }

        .sensor-summary {
            background: #0f1117;
            border: 1px solid #1e2130;
            border-radius: 6px;
            padding: 1rem 1.2rem;
            margin-top: 1rem;
        }
        .sensor-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.82rem;
            color: #8b92a5;
            padding: 0.3rem 0;
            border-bottom: 1px solid #1e2130;
        }
        .sensor-row:last-child { border-bottom: none; }
        .sensor-val { color: #ffffff; font-weight: 500; }
    </style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("<div class='page-title'>Live Machine Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='page-sub'>Enter current sensor readings to assess failure probability and receive a maintenance recommendation.</div>", unsafe_allow_html=True)

model, threshold, columns = load_model()

# ── Layout ─────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1.4], gap="large")

with left:
    st.markdown("<div class='panel-label'>Sensor Input</div>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        machine_type = st.selectbox(
            "Machine Type",
            ["L", "M", "H"],
            help="L = Light duty, M = Medium duty, H = Heavy duty"
        )

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            air = st.number_input("Air Temperature (K)", 290.0, 320.0, 300.0, step=0.1, format="%.1f")
            speed = st.number_input("Rotational Speed (rpm)", 1000, 3000, 1500, step=10)
            wear = st.number_input("Tool Wear (min)", 0, 300, 120, step=1)
        with c2:
            process = st.number_input("Process Temperature (K)", 300.0, 340.0, 310.0, step=0.1, format="%.1f")
            torque = st.number_input("Torque (Nm)", 10.0, 80.0, 40.0, step=0.5, format="%.1f")

        temp_diff = round(process - air, 2)
        st.markdown(
            f"<div style='font-size:0.78rem; color:#8b92a5; margin-top:0.5rem;'>Computed Temperature Differential: <b style='color:#fff;'>{temp_diff:.1f} K</b></div>",
            unsafe_allow_html=True
        )

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        submit = st.form_submit_button("Run Prediction", use_container_width=True)

    # Reference ranges
    st.markdown("""
        <div class='sensor-summary'>
            <div style='font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em; color:#8b92a5; margin-bottom:0.6rem;'>Normal Operating Ranges</div>
            <div class='sensor-row'><span>Air Temperature</span><span class='sensor-val'>295 – 304 K</span></div>
            <div class='sensor-row'><span>Process Temperature</span><span class='sensor-val'>305 – 314 K</span></div>
            <div class='sensor-row'><span>Rotational Speed</span><span class='sensor-val'>1168 – 2886 rpm</span></div>
            <div class='sensor-row'><span>Torque</span><span class='sensor-val'>3.8 – 76.6 Nm</span></div>
            <div class='sensor-row'><span>Tool Wear</span><span class='sensor-val'>0 – 253 min</span></div>
        </div>
    """, unsafe_allow_html=True)

with right:
    if not submit:
        st.markdown("""
            <div style='height:100%; display:flex; align-items:center; justify-content:center; padding:3rem 0;'>
                <div style='text-align:center; color:#4a5568;'>
                    <div style='font-size:2.5rem; margin-bottom:0.75rem;'>—</div>
                    <div style='font-size:0.85rem; letter-spacing:0.05em; text-transform:uppercase;'>Awaiting sensor input</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        data = {
            "Air_temperature_K":     air,
            "Process_temperature_K": process,
            "Rotational_speed_rpm":  speed,
            "Torque_Nm":             torque,
            "Tool_wear_min":         wear,
            "temp_diff":             temp_diff,
            "Type_L":                1 if machine_type == "L" else 0,
            "Type_M":                1 if machine_type == "M" else 0,
        }

        df_input = prepare_input(data)
        prob     = predict_failure(model, df_input)
        decision, color = maintenance_decision(prob, threshold)

        color_map = {"red": "critical", "orange": "warning", "green": "ok"}
        risk_class = color_map[color]

        # ── Probability gauge ──────────────────────────────────────────────────
        gauge_color = {"critical": "#e63946", "warning": "#f4a261", "ok": "#2a9d8f"}[risk_class]

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 1),
            number={"suffix": "%", "font": {"size": 36, "color": "#ffffff"}},
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#8b92a5", tickfont=dict(size=10, color="#8b92a5")),
                bar=dict(color=gauge_color, thickness=0.25),
                bgcolor="#0a0c10",
                bordercolor="#1e2130",
                borderwidth=1,
                steps=[
                    dict(range=[0,  threshold * 0.3  * 100], color="#12151f"),
                    dict(range=[threshold * 0.3  * 100, threshold * 0.6  * 100], color="#1a1f2e"),
                    dict(range=[threshold * 0.6  * 100, threshold * 100],        color="#1e2438"),
                    dict(range=[threshold * 100, 100],                            color="#2a1520"),
                ],
                threshold=dict(
                    line=dict(color="#e63946", width=2),
                    thickness=0.75,
                    value=threshold * 100
                )
            )
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b92a5"),
            height=240,
            margin=dict(l=20, r=20, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # ── Decision card ──────────────────────────────────────────────────────
        st.markdown(f"""
            <div class='result-card result-{risk_class}'>
                <div class='decision-label'>Maintenance Decision</div>
                <div class='decision-value-{risk_class}'>{decision}</div>
                <div style='font-size:0.78rem; color:#8b92a5; margin-top:0.5rem;'>
                    Failure probability: <b style='color:#fff;'>{prob:.1%}</b> &nbsp;|&nbsp;
                    Decision threshold: <b style='color:#fff;'>{threshold:.0%}</b> &nbsp;|&nbsp;
                    Machine type: <b style='color:#fff;'>{machine_type}</b>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # ── Recommended actions ────────────────────────────────────────────────
        actions = {
            "critical": [
                "Halt machine operation immediately",
                "Dispatch maintenance team — priority 1",
                "Inspect tool wear and replace if above 200 min",
                "Check torque calibration and bearing integrity",
                "Log incident and update maintenance records",
            ],
            "warning": [
                "Schedule inspection within 24 hours",
                "Monitor torque and temperature closely",
                "Prepare replacement tooling and components",
                "Notify shift supervisor",
            ],
            "ok": [
                "Continue normal operation",
                "Log reading for trend monitoring",
                "Next scheduled review as per maintenance plan",
            ],
        }

        action_items = "".join(
            f"<div class='action-item'>{a}</div>" for a in actions[risk_class]
        )
        st.markdown(f"""
            <div class='action-box'>
                <div class='action-title'>Recommended Actions</div>
                {action_items}
            </div>
        """, unsafe_allow_html=True)

        # ── Input summary ──────────────────────────────────────────────────────
        st.markdown(f"""
            <div class='sensor-summary' style='margin-top:1rem;'>
                <div style='font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em; color:#8b92a5; margin-bottom:0.6rem;'>Submitted Readings</div>
                <div class='sensor-row'><span>Air Temperature</span><span class='sensor-val'>{air} K</span></div>
                <div class='sensor-row'><span>Process Temperature</span><span class='sensor-val'>{process} K</span></div>
                <div class='sensor-row'><span>Temperature Differential</span><span class='sensor-val'>{temp_diff} K</span></div>
                <div class='sensor-row'><span>Rotational Speed</span><span class='sensor-val'>{speed} rpm</span></div>
                <div class='sensor-row'><span>Torque</span><span class='sensor-val'>{torque} Nm</span></div>
                <div class='sensor-row'><span>Tool Wear</span><span class='sensor-val'>{wear} min</span></div>
            </div>
        """, unsafe_allow_html=True)