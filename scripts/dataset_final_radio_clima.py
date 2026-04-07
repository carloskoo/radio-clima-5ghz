import os
import glob
import pandas as pd

RADIO_FILE = "dataset_modelado_radio.csv"
CLIMA_ROOT = "eber_burga"
CLIMA_PATTERN = os.path.join(CLIMA_ROOT, "*", "era5_mid_hourly_*.csv")

radio = pd.read_csv(RADIO_FILE)
radio["timestamp"] = pd.to_datetime(radio["timestamp"], errors="coerce")
radio = radio.dropna(subset=["timestamp"])
radio["timestamp_hour"] = radio["timestamp"].dt.floor("h")

radio_num_cols = [
    "dl_rssi", "dl_snr", "ul_rssi", "ul_snr",
    "rssi_mean", "snr_mean", "rssi_diff", "snr_diff"
]

radio_hourly = (
    radio.groupby("timestamp_hour")[radio_num_cols]
    .mean()
    .reset_index()
)

if "periodo" in radio.columns:
    periodo_hourly = (
        radio.groupby("timestamp_hour")["periodo"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
        .reset_index()
    )
    radio_hourly = radio_hourly.merge(periodo_hourly, on="timestamp_hour", how="left")

clima_files = sorted(glob.glob(CLIMA_PATTERN))
clima_list = []

for f in clima_files:
    df = pd.read_csv(f, sep=";")
    clima_list.append(df)

clima = pd.concat(clima_list, ignore_index=True)
clima["timestamp_hour"] = pd.to_datetime(clima["time_lima"], errors="coerce")
clima = clima.dropna(subset=["timestamp_hour"])

try:
    clima["timestamp_hour"] = clima["timestamp_hour"].dt.tz_localize(None)
except Exception:
    pass

clima["timestamp_hour"] = clima["timestamp_hour"].dt.floor("h")

for col in ["temp_c", "dewpoint_c", "precip_mm", "press_hpa", "wind_ms"]:
    clima[col] = pd.to_numeric(clima[col], errors="coerce")

clima_hourly = (
    clima.groupby("timestamp_hour")[["temp_c", "dewpoint_c", "precip_mm", "press_hpa", "wind_ms"]]
    .mean()
    .reset_index()
)

dataset_final = pd.merge(radio_hourly, clima_hourly, on="timestamp_hour", how="inner")
dataset_final = dataset_final.rename(columns={"timestamp_hour": "timestamp"})
dataset_final = dataset_final.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

dataset_final["rain_flag"] = (dataset_final["precip_mm"] > 0).astype(int)

def classify_rain(x):
    if x == 0:
        return "no_rain"
    elif x < 2.5:
        return "light_rain"
    elif x < 10:
        return "moderate_rain"
    else:
        return "heavy_rain"

dataset_final["rain_class"] = dataset_final["precip_mm"].apply(classify_rain)

dataset_final.to_csv("dataset_final_radio_clima.csv", index=False)
print("✔ dataset_final_radio_clima.csv generado")
