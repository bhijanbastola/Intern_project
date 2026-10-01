import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Nepal High-Flow Predictor", page_icon="🌊", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load("flood_model.joblib")


bundle = load_bundle()
model = bundle["model"]
features = bundle["features"]
stations = bundle["station_info"]
results = bundle["results"]

st.title("🌊 Nepal River High-Flow Predictor")
st.caption("Predicts whether tomorrow's river discharge will be unusually high "
           "(above the station's 95th percentile).")

tab_predict, tab_models = st.tabs(["Prediction", "Model comparison"])

# ---------------- Sidebar inputs ----------------
st.sidebar.header("Today's conditions")

station = st.sidebar.selectbox(
    "Station", stations.index,
    format_func=lambda s: f"{stations.loc[s, 'location']} (station {s})")
thr = float(stations.loc[station, "threshold"])
elevation = float(stations.loc[station, "elevation_m"])
st.sidebar.caption(f"High-flow threshold: {thr:,.1f} m³/s  |  Elevation: {elevation:,.0f} m")

month = st.sidebar.selectbox("Month", list(range(1, 13)), index=datetime.date.today().month - 1)

st.sidebar.subheader("River")
q_today = st.sidebar.number_input("Discharge today (m³/s)", min_value=0.0,
                                  value=round(thr * 0.5, 1), key=f"q_today_{station}")
q_yest = st.sidebar.number_input("Discharge yesterday (m³/s)", min_value=0.0,
                                 value=round(thr * 0.5, 1), key=f"q_yest_{station}")

st.sidebar.subheader("Rain")
rain_today = st.sidebar.number_input("Rain today (mm)", min_value=0.0, value=5.0)
rain_3d = st.sidebar.number_input("Rain, last 3 days incl. today (mm)", min_value=0.0, value=10.0)
rain_7d = st.sidebar.number_input("Rain, last 7 days incl. today (mm)", min_value=0.0, value=20.0)

st.sidebar.subheader("Soil and weather")
soil = st.sidebar.slider("Soil moisture today (m³/m³)", 0.05, 0.60, 0.30, 0.01)
soil_3d = st.sidebar.slider("Soil moisture, 3-day average (m³/m³)", 0.05, 0.60, 0.30, 0.01)
temp = st.sidebar.slider("Mean temperature (°C)", -10.0, 40.0, 20.0, 0.5)
humidity = st.sidebar.slider("Mean relative humidity (%)", 0, 100, 70)

# ---------------- Build the model input (same features as training) ----------------
row = {
    "discharge_ratio": q_today / thr,
    "discharge_yesterday": q_yest / thr,
    "discharge_change": (q_today - q_yest) / thr,
    "rain_log": np.log1p(rain_today),
    "rain_3d": rain_3d,
    "rain_7d": rain_7d,
    "soil_moisture_0_100cm_m3m3": soil,
    "soil_3d": soil_3d,
    "soil_x_rain": soil * rain_3d,
    "temperature_mean_c": temp,
    "relative_humidity_mean_pct": humidity,
    "elevation_m": elevation,
    "month": month,
}
X = pd.DataFrame([row])[features]
prob = float(model.predict_proba(X)[0, 1])

# ---------------- Prediction tab ----------------
with tab_predict:
    if rain_3d < rain_today or rain_7d < rain_3d:
        st.warning("Check your rain inputs: the 7-day total should be at least the 3-day total, "
                   "and the 3-day total at least today's rain.")

    if prob < 0.20:
        level, color = "LOW", "green"
    elif prob < 0.50:
        level, color = "MODERATE", "orange"
    else:
        level, color = "HIGH", "red"

    c1, c2, c3 = st.columns(3)
    c1.metric("Probability of high flow tomorrow", f"{prob:.1%}")
    c2.metric("Risk level", level)
    c3.metric("Today's discharge vs threshold", f"{q_today / thr:.0%}")

    st.progress(min(prob, 1.0))
    st.markdown(f"**Risk level: :{color}[{level}]**")

    with st.expander("Model inputs"):
        st.dataframe(X.T.rename(columns={0: "value"}))

    st.caption(f"Model used: {bundle['model_name']}. Educational project built on modelled "
               "discharge data, not an official flood warning.")

# ---------------- Model comparison tab ----------------
with tab_models:
    st.subheader("Test-set results (2026 data)")
    st.dataframe(results, use_container_width=True)
    st.success(f"Best model by PR-AUC: {bundle['model_name']}")
    st.caption("PR-AUC is the main metric because high-flow days are rare. "
               "The persistence baseline predicts 'high tomorrow' if today is already high.")