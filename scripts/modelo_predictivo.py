import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =========================================================
# 1. RUTAS DEL PROYECTO (GITHUB + LOCAL)
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(BASE_DIR, "data", "processed", "dataset_final_radio_clima.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# =========================================================
# 2. CARGAR DATASET
# =========================================================

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"No se encontró el archivo: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

print("Columnas detectadas:", df.columns.tolist())

if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# =========================================================
# 3. VARIABLES DEL MODELO
# =========================================================

features = ["temp_c", "dewpoint_c", "precip_mm", "press_hpa", "wind_ms"]
target = "dl_rssi"

faltantes = [col for col in features + [target] if col not in df.columns]
if faltantes:
    raise ValueError(f"Faltan columnas necesarias: {faltantes}")

df_model = df[features + [target]].copy()

for col in features + [target]:
    df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

df_model = df_model.dropna()

print("Filas disponibles para modelado:", len(df_model))

X = df_model[features]
y = df_model[target]

# =========================================================
# 4. HOLD-OUT SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Filas entrenamiento:", len(X_train))
print("Filas prueba:", len(X_test))

# =========================================================
# 5. DEFINIR MODELOS
# =========================================================

lr = LinearRegression()
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
gb = GradientBoostingRegressor(random_state=42)

# =========================================================
# 6. ENTRENAR MODELOS
# =========================================================

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_gb = gb.predict(X_test)

# =========================================================
# 7. FUNCIÓN DE EVALUACIÓN HOLD-OUT
# =========================================================

def evaluar_holdout(y_true, y_pred, nombre_modelo):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n--- {nombre_modelo} (Hold-out) ---")
    print(f"R2   : {r2:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")

    return {
        "Model": nombre_modelo,
        "Evaluation": "Hold-out",
        "R2_mean": r2,
        "R2_std": np.nan,
        "MAE_mean": mae,
        "MAE_std": np.nan,
        "RMSE_mean": rmse,
        "RMSE_std": np.nan
    }

res_lr_hold = evaluar_holdout(y_test, y_pred_lr, "Linear Regression")
res_rf_hold = evaluar_holdout(y_test, y_pred_rf, "Random Forest")
res_gb_hold = evaluar_holdout(y_test, y_pred_gb, "Gradient Boosting")

# =========================================================
# 8. CROSS-VALIDATION (5-FOLD)
# =========================================================

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluar_cv(model, X, y, nombre_modelo):
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
    mae_scores = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")
    rmse_scores = np.sqrt(
        -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error")
    )

    print(f"\n--- {nombre_modelo} (5-Fold CV) ---")
    print(f"R2   : {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
    print(f"MAE  : {mae_scores.mean():.4f} ± {mae_scores.std():.4f}")
    print(f"RMSE : {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")

    return {
        "Model": nombre_modelo,
        "Evaluation": "5-Fold CV",
        "R2_mean": r2_scores.mean(),
        "R2_std": r2_scores.std(),
        "MAE_mean": mae_scores.mean(),
        "MAE_std": mae_scores.std(),
        "RMSE_mean": rmse_scores.mean(),
        "RMSE_std": rmse_scores.std()
    }

res_lr_cv = evaluar_cv(lr, X, y, "Linear Regression")
res_rf_cv = evaluar_cv(rf, X, y, "Random Forest")
res_gb_cv = evaluar_cv(gb, X, y, "Gradient Boosting")

# =========================================================
# 9. EXPORTAR MÉTRICAS
# =========================================================

metricas_df = pd.DataFrame([
    res_lr_hold, res_rf_hold, res_gb_hold,
    res_lr_cv, res_rf_cv, res_gb_cv
])

metricas_path = os.path.join(TABLES_DIR, "metricas_modelos.csv")
metricas_df.to_csv(metricas_path, index=False)

print(f"\n✔ Métricas guardadas en: {metricas_path}")

# =========================================================
# 10. IMPORTANCIA DE VARIABLES (RANDOM FOREST)
# =========================================================

importances_rf = rf.feature_importances_

importance_df = pd.DataFrame({
    "Variable": features,
    "Importance": importances_rf
}).sort_values(by="Importance", ascending=False)

importance_path = os.path.join(TABLES_DIR, "feature_importance.csv")
importance_df.to_csv(importance_path, index=False)

print(f"✔ Importancia de variables guardada en: {importance_path}")

# =========================================================
# 11. COEFICIENTES LINEALES
# =========================================================

coef_df = pd.DataFrame({
    "Variable": features,
    "Coefficient": lr.coef_
}).sort_values(by="Coefficient", ascending=False)

coef_path = os.path.join(TABLES_DIR, "linear_regression_coefficients.csv")
coef_df.to_csv(coef_path, index=False)

print(f"✔ Coeficientes lineales guardados en: {coef_path}")

# =========================================================
# 12. FIGURA 6 — COMPARACIÓN DE MODELOS (HOLD-OUT R²)
# =========================================================

model_names = ["Linear Regression", "Random Forest", "Gradient Boosting"]
r2_values = [
    res_lr_hold["R2_mean"],
    res_rf_hold["R2_mean"],
    res_gb_hold["R2_mean"]
]

plt.figure(figsize=(8, 5))
plt.bar(model_names, r2_values)
plt.title("Model Comparison by R²")
plt.xlabel("Model")
plt.ylabel("R²")
plt.tight_layout()

fig6_path = os.path.join(FIGURES_DIR, "fig6_model_comparison_r2.png")
plt.savefig(fig6_path, dpi=300)
plt.close()

print(f"✔ Figura guardada en: {fig6_path}")

# =========================================================
# 13. FIGURA 7 — OBSERVED VS PREDICTED (RANDOM FOREST)
# =========================================================

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5)

min_val = min(y_test.min(), y_pred_rf.min())
max_val = max(y_test.max(), y_pred_rf.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    color="red",
    linestyle="--",
    linewidth=2,
    label="Ideal prediction (y = x)"
)

plt.xlabel("Observed RSSI (dBm)")
plt.ylabel("Predicted RSSI (dBm)")
plt.title("Observed vs Predicted RSSI (Random Forest)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

fig7_path = os.path.join(FIGURES_DIR, "fig7_predicted_vs_actual.png")
plt.savefig(fig7_path, dpi=300)
plt.close()

print(f"✔ Figura guardada en: {fig7_path}")

# =========================================================
# 14. FIGURA 8 — FEATURE IMPORTANCE (RANDOM FOREST)
# =========================================================

plt.figure(figsize=(8, 5))
plt.bar(importance_df["Variable"], importance_df["Importance"])
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Variables")
plt.ylabel("Importance")
plt.tight_layout()

fig8_path = os.path.join(FIGURES_DIR, "fig8_importance.png")
plt.savefig(fig8_path, dpi=300)
plt.close()

print(f"✔ Figura guardada en: {fig8_path}")

# =========================================================
# 15. TABLA PARA EL PAPER (CV)
# =========================================================

tabla_paper = metricas_df[metricas_df["Evaluation"] == "5-Fold CV"].copy()

tabla_paper["R² (mean ± std)"] = tabla_paper.apply(
    lambda row: f'{row["R2_mean"]:.4f} ± {row["R2_std"]:.4f}', axis=1
)
tabla_paper["MAE (mean ± std)"] = tabla_paper.apply(
    lambda row: f'{row["MAE_mean"]:.4f} ± {row["MAE_std"]:.4f}', axis=1
)
tabla_paper["RMSE (mean ± std)"] = tabla_paper.apply(
    lambda row: f'{row["RMSE_mean"]:.4f} ± {row["RMSE_std"]:.4f}', axis=1
)

tabla_paper = tabla_paper[
    ["Model", "R² (mean ± std)", "MAE (mean ± std)", "RMSE (mean ± std)"]
]

tabla_paper_path = os.path.join(TABLES_DIR, "tabla_cv_para_paper.csv")
tabla_paper.to_csv(tabla_paper_path, index=False)

print(f"✔ Tabla para paper guardada en: {tabla_paper_path}")

# =========================================================
# 16. RESUMEN FINAL
# =========================================================

mejor_holdout = metricas_df[metricas_df["Evaluation"] == "Hold-out"].sort_values(
    by="R2_mean", ascending=False
).iloc[0]["Model"]

mejor_cv = metricas_df[metricas_df["Evaluation"] == "5-Fold CV"].sort_values(
    by="R2_mean", ascending=False
).iloc[0]["Model"]

print("\n==================== RESUMEN FINAL ====================")
print(f"Mejor modelo Hold-out según R²: {mejor_holdout}")
print(f"Mejor modelo 5-Fold CV según R²: {mejor_cv}")
print("======================================================")
