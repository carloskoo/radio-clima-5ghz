import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset_final_radio_clima.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Fig. 1
plt.figure(figsize=(12, 5))
plt.plot(df["timestamp"], df["dl_rssi"])
plt.title("RSSI Time Series")
plt.xlabel("Time")
plt.ylabel("RSSI (dBm)")
plt.tight_layout()
plt.savefig("fig1_rssi.png", dpi=300)
plt.close()

# Fig. 2
plt.figure(figsize=(12, 5))
plt.plot(df["timestamp"], df["precip_mm"])
plt.title("Precipitation Time Series")
plt.xlabel("Time")
plt.ylabel("Precipitation (mm)")
plt.tight_layout()
plt.savefig("fig2_lluvia.png", dpi=300)
plt.close()

# Fig. 3
plt.figure(figsize=(7, 5))
plt.scatter(df["precip_mm"], df["dl_rssi"], alpha=0.4)
plt.title("Precipitation vs RSSI")
plt.xlabel("Precipitation (mm)")
plt.ylabel("RSSI (dBm)")
plt.tight_layout()
plt.savefig("fig3_scatter.png", dpi=300)
plt.close()

# Fig. 4
orden = ["no_rain", "light_rain", "moderate_rain", "heavy_rain"]
data = [df[df["rain_class"] == c]["dl_rssi"].dropna() for c in orden]

plt.figure(figsize=(8, 5))
plt.boxplot(data, tick_labels=orden)
plt.title("RSSI vs Rain Class")
plt.xlabel("Rain Class")
plt.ylabel("RSSI (dBm)")
plt.tight_layout()
plt.savefig("fig4_boxplot.png", dpi=300)
plt.close()

# Fig. 5
cols = ["dl_rssi", "dl_snr", "temp_c", "dewpoint_c", "precip_mm", "press_hpa", "wind_ms"]
corr = df[cols].corr()

plt.figure(figsize=(8, 6))
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(cols)), cols, rotation=45)
plt.yticks(range(len(cols)), cols)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("fig5_corr.png", dpi=300)
plt.close()

print("✔ Figuras 1–5 generadas")
