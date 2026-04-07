import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DATA_FILE = "dataset_final_radio_clima.csv"

df = pd.read_csv(DATA_FILE)

if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

features = ["temp_c", "dewpoint_c", "precip_mm", "press_hpa", "wind_ms"]
target = "dl_rssi"

df_model = df[features + [target]].copy()

for col in features + [target]:
    df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

df_model = df_model.dropna()

X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

def evaluar_modelo(y_true, y_pred, nombre_modelo):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n--- {nombre_modelo} ---")
    print(f"R2   : {r2:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")

    return {
        "Modelo": nombre_modelo,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse
    }

res_lr = evaluar_modelo(y_test, y_pred_lr, "Linear Regression")
res_rf = evaluar_modelo(y_test, y_pred_rf, "Random Forest")

metricas_df = pd.DataFrame([res_lr, res_rf])
metricas_df.to_csv("metricas_modelos.csv", index=False)

importances = rf.feature_importances_

importance_df = pd.DataFrame({
    "Variable": features,
    "Importance": importances
})
importance_df.to_csv("feature_importance.csv", index=False)

# Fig. 6
model_names = ["Linear Regression", "Random Forest"]
r2_values = [res_lr["R2"], res_rf["R2"]]

plt.figure(figsize=(7, 5))
plt.bar(model_names, r2_values)
plt.title("Model Comparison by R²")
plt.xlabel("Model")
plt.ylabel("R²")
plt.tight_layout()
plt.savefig("fig6_model_comparison_r2.png", dpi=300)
plt.close()

# Fig. 7
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5)

min_val = min(y_test.min(), y_pred_rf.min())
max_val = max(y_test.max(), y_pred_rf.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    color='red',
    linestyle='--',
    linewidth=2,
    label='Ideal prediction (y = x)'
)

plt.xlabel("Observed RSSI (dBm)")
plt.ylabel("Predicted RSSI (dBm)")
plt.title("Observed vs Predicted RSSI (Random Forest)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig7_predicted_vs_actual.png", dpi=300)
plt.close()

# Fig. 8
sorted_idx = np.argsort(importances)[::-1]
features_sorted = [features[i] for i in sorted_idx]
importances_sorted = importances[sorted_idx]

plt.figure(figsize=(8, 5))
plt.bar(features_sorted, importances_sorted)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Variables")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("fig8_importance.png", dpi=300)
plt.close()

coef_df = pd.DataFrame({
    "Variable": features,
    "Coefficient": lr.coef_
})
coef_df.to_csv("linear_regression_coefficients.csv", index=False)

print("✔ Modelado y figuras 6–8 generadas")
