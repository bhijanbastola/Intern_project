import joblib
import numpy as np
import pandas as pd

CSV_PATH = "/Users/bhijanbastola/Desktop/Flood/Data/CORRECTED_2023_2026_NEPAL_FLOOD_WEATHER_KAGGLE.csv"      # <- change to your file name
TEST_START = "2026-01-01"

# 1. Load the saved model bundle (thresholds come from here, so they match the app)
bundle = joblib.load("flood_model.joblib")
model = bundle["model"]
features = bundle["features"]
stations = bundle["station_info"]

# 2. Load data and rebuild the columns we need
df = pd.read_csv(CSV_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["dhm_station", "date"]).reset_index(drop=True)

df["threshold"] = df["dhm_station"].map(stations["threshold"])
df["discharge_tomorrow"] = df.groupby("dhm_station")["river_discharge_m3s"].shift(-1)
df["target"] = (df["discharge_tomorrow"] > df["threshold"]).astype(int)
df["q_yesterday"] = df.groupby("dhm_station")["river_discharge_m3s"].shift(1)
df["rain_3d"] = df.groupby("dhm_station")["precipitation_mm"].rolling(3).sum().reset_index(level=0, drop=True)
df["rain_7d"] = df.groupby("dhm_station")["precipitation_mm"].rolling(7).sum().reset_index(level=0, drop=True)
df["soil_3d"] = df.groupby("dhm_station")["soil_moisture_0_100cm_m3m3"].rolling(3).mean().reset_index(level=0, drop=True)
df["month"] = df["date"].dt.month
df["ratio"] = df["river_discharge_m3s"] / df["threshold"]
df = df.dropna(subset=["q_yesterday", "rain_3d", "rain_7d", "soil_3d", "discharge_tomorrow"])


# 3. Convert a data row into the exact values you type into the app
def app_inputs(r):
    return {
        "Month": int(r["month"]),
        "Discharge today (m3/s)": round(float(r["river_discharge_m3s"]), 1),
        "Discharge yesterday (m3/s)": round(float(r["q_yesterday"]), 1),
        "Rain today (mm)": round(float(r["precipitation_mm"]), 1),
        "Rain last 3 days (mm)": round(float(r["rain_3d"]), 1),
        "Rain last 7 days (mm)": round(float(r["rain_7d"]), 1),
        "Soil moisture today": float(np.clip(round(r["soil_moisture_0_100cm_m3m3"], 2), 0.05, 0.60)),
        "Soil moisture 3-day avg": float(np.clip(round(r["soil_3d"], 2), 0.05, 0.60)),
        "Temperature (C)": float(np.clip(round(r["temperature_mean_c"] * 2) / 2, -10, 40)),
        "Humidity (%)": int(np.clip(round(r["relative_humidity_mean_pct"]), 0, 100)),
    }


# 4. Predict exactly the way the app does
def predict(station, inp):
    thr = float(stations.loc[station, "threshold"])
    row = {
        "discharge_ratio": inp["Discharge today (m3/s)"] / thr,
        "discharge_yesterday": inp["Discharge yesterday (m3/s)"] / thr,
        "discharge_change": (inp["Discharge today (m3/s)"] - inp["Discharge yesterday (m3/s)"]) / thr,
        "rain_log": np.log1p(inp["Rain today (mm)"]),
        "rain_3d": inp["Rain last 3 days (mm)"],
        "rain_7d": inp["Rain last 7 days (mm)"],
        "soil_moisture_0_100cm_m3m3": inp["Soil moisture today"],
        "soil_3d": inp["Soil moisture 3-day avg"],
        "soil_x_rain": inp["Soil moisture today"] * inp["Rain last 3 days (mm)"],
        "temperature_mean_c": inp["Temperature (C)"],
        "relative_humidity_mean_pct": inp["Humidity (%)"],
        "elevation_m": float(stations.loc[station, "elevation_m"]),
        "month": inp["Month"],
    }
    return float(model.predict_proba(pd.DataFrame([row])[features])[0, 1])


def risk_level(p):
    return "LOW" if p < 0.20 else "MODERATE" if p < 0.50 else "HIGH"


# 5. Pick demo days for every station
records = []
for st in stations.index:
    name = stations.loc[st, "location"]
    thr = float(stations.loc[st, "threshold"])
    all_st = df[df["dhm_station"] == st]
    recent = all_st[all_st["date"] >= TEST_START]
    pool = recent if len(recent) > 0 else all_st

    picks = []

    # Normal day: no high flow followed, discharge closest to 50% of threshold
    calm = pool[pool["target"] == 0]
    if len(calm) > 0:
        picks.append(("Normal day", calm.loc[(calm["ratio"] - 0.5).abs().idxmin()]))

    # High-flow day: a day before a real high-flow event, strongest model signal
    events = pool[pool["target"] == 1]
    if len(events) == 0:
        events = all_st[all_st["target"] == 1]   # fall back to earlier years
    if len(events) > 0:
        probs = [predict(st, app_inputs(r)) for _, r in events.iterrows()]
        picks.append(("High-flow day", events.iloc[int(np.argmax(probs))]))

    print("=" * 60)
    print(f"STATION {st}: {name}   (threshold {thr:,.1f} m3/s)")
    if len(picks) < 2:
        print("  (no high-flow event found for this station)")

    for label, r in picks:
        inp = app_inputs(r)
        p = predict(st, inp)
        actual = "HIGH flow happened next day" if r["target"] == 1 else "no high flow next day"
        print(f"\n  [{label}]  from {r['date'].date()}")
        for k, v in inp.items():
            print(f"    {k:<28} {v}")
        print(f"    -> Model: {p:.1%} ({risk_level(p)})  |  Actual: {actual}")
        records.append({"station": st, "location": name, "scenario": label,
                        "date": r["date"].date(), **inp,
                        "model_probability": round(p, 3), "risk": risk_level(p),
                        "actual_next_day": int(r["target"])})

demo = pd.DataFrame(records)
demo.to_csv("demo_values.csv", index=False)

# Readable text file: type these values into the app's sidebar
input_names = list(app_inputs(df.iloc[0]).keys())
with open("demo_values.txt", "w") as f:
    f.write("NEPAL HIGH-FLOW PREDICTOR - DEMO VALUES\n")
    f.write("Select the station, then type each value into the sidebar.\n")
    for st, group in demo.groupby("station"):
        f.write("\n" + "=" * 55 + "\n")
        f.write(f"STATION {st}: {stations.loc[st, 'location']}  "
                f"(threshold {stations.loc[st, 'threshold']:,.1f} m3/s)\n")
        for _, r in group.iterrows():
            f.write(f"\n  [{r['scenario']}]  (real day: {r['date']})\n")
            for k in input_names:
                f.write(f"    {k:<28} {r[k]}\n")
            f.write(f"    Expected app result: {r['model_probability']:.1%} ({r['risk']})\n")

print("\nSaved demo_values.csv and demo_values.txt")