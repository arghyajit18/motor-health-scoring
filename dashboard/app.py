"""
Streamlit dashboard: Motor Health Scoring from Current Signature.
Run with: streamlit run dashboard/app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Motor Health Scoring", layout="wide")

WEIGHTS = {"sideband_ratio": 1.0, "bearing_band": 1.0, "ecc_ratio": 1.0,
           "unbalance_pct": 0.8, "thd": 0.5, "crest": 0.5}
BASE_MU = {"sideband_ratio": 0.00038, "bearing_band": 0.00077,
           "ecc_ratio": 0.00045, "unbalance_pct": 0.03385}
BASE_SD = {"sideband_ratio": 0.00013, "bearing_band": 0.00012,
           "ecc_ratio": 0.00019, "unbalance_pct": 0.01771}
D50, SCALE = 1.97, 10.952


@st.cache_data
def load_data():
    trend = pd.read_csv("outputs/health_trend.csv", parse_dates=["date"])
    ranking = pd.read_csv("outputs/fleet_ranking.csv")
    alerts = pd.read_csv("outputs/alerts.csv", parse_dates=["date"])
    with open("outputs/metrics.json") as f:
        metrics = json.load(f)
    return trend, ranking, alerts, metrics


trend, ranking, alerts, metrics = load_data()

st.title("Motor Health Scoring from Current Signature")
st.caption("12-motor fleet, daily 1-second 3-phase readings at 2 kHz, FFT signature analysis")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Fleet mean health", f"{ranking['health'].mean():.0f}/100")
c2.metric("Critical motors", f"{int((ranking['status'] == 'Critical').sum())}")
c3.metric("Detection recall", f"{metrics['recall_alert_critical'] * 100:.0f}%")
lead = metrics["mean_early_warning_lead_days"]
c4.metric("Mean early warning", f"{lead:.0f} days" if lead is not None else "n/a")

st.divider()

st.subheader("Fleet Ranking (today)")
styled = ranking[["motor_id", "motor_type", "health", "status",
                  "suspected_cause", "action"]].sort_values("health")
st.dataframe(styled, use_container_width=True)
fig = px.bar(styled, x="motor_id", y="health", color="status",
             color_discrete_map={"Healthy": "#16a34a", "Watch": "#eab308",
                                 "Alert": "#f97316", "Critical": "#dc2626"},
             labels={"health": "Health (0-100)", "motor_id": "Motor"})
fig.update_layout(height=350)
st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Motor Detail")
mid = st.selectbox("Motor", options=sorted(trend["motor_id"].unique()))
g = trend[trend["motor_id"] == mid].sort_values("day")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=g["day"], y=g["health"], name="Health",
                          line=dict(color="#2563eb")))
for line, name, col in ((80, "Healthy", "green"), (60, "Watch", "orange"), (40, "Alert", "red")):
    fig2.add_hline(y=line, line_dash="dash", line_color=col, annotation_text=name)
fig2.update_layout(height=350, xaxis_title="Day", yaxis_title="Health (0-100)")
st.plotly_chart(fig2, use_container_width=True)

st.caption("Latest diagnosis: "
           f"{g.iloc[-1]['status']} — {g.iloc[-1]['suspected_cause']}. "
           f"{g.iloc[-1]['action']}.")

st.divider()

st.subheader("What-If Simulator: signature values to health score")
st.caption("Move the sliders to see how fault-indicator magnitudes drive the 0-100 score")
s1, s2 = st.columns(2)
with s1:
    adj_sb = st.slider("Rotor sideband ratio", 0.0, 0.15, 0.01, 0.001)
    adj_bb = st.slider("Bearing band energy", 0.0, 0.08, 0.005, 0.001)
with s2:
    adj_ecc = st.slider("Eccentricity ratio", 0.0, 0.12, 0.01, 0.001)
    adj_unb = st.slider("Unbalance (%)", 0.0, 10.0, 0.5, 0.1)
vals = {"sideband_ratio": adj_sb, "bearing_band": adj_bb,
        "ecc_ratio": adj_ecc, "unbalance_pct": adj_unb}
d = sum(max(0.0, (vals[c] - BASE_MU[c]) / BASE_SD[c]) * WEIGHTS[c] for c in vals)
health = float(np.clip(100 * np.exp(-max(0.0, d - D50) / SCALE), 0, 100))
status = "Healthy" if health >= 80 else ("Watch" if health >= 60
                                         else ("Alert" if health >= 40 else "Critical"))
a, b = st.columns(2)
with a:
    st.metric("Simulated health", f"{health:.0f}/100")
with b:
    st.metric("Status", status)

st.divider()

st.subheader("Alert Log")
st.dataframe(alerts.sort_values("date", ascending=False).head(20), use_container_width=True)
st.caption("Health = 100 * exp(-degradation/scale) from baseline z-scores; "
           "IsolationForest trained on early healthy readings cross-checks each score.")
