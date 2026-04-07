# Radio-Clima-5GHz

Predictive modeling of signal attenuation in 5.8 GHz radio links using ERA5-Land climatological data in a high-altitude Andean environment.

---

## 📌 Overview

This repository presents an analysis of the influence of climatological variables on the performance of a 5.8 GHz point-to-point radio link deployed in a high-altitude Andean environment.

The study integrates real-world radio measurements (RSSI and SNR) with ERA5-Land climatological data to evaluate signal variability and develop predictive models using machine learning techniques.

---

## 📂 Repository Structure
radio-clima-5ghz/
│
├── data/
│ └── processed/
│ └── dataset_final_radio_clima.csv
│
├── results/
│ └── figures/
│ ├── fig1_rssi_timeseries.png
│ ├── fig2_precipitation_timeseries.png
│ ├── fig3_scatter.png
│ ├── fig4_boxplot.png
│ ├── fig5_correlation.png
│ ├── fig6_feature_importance.png
│ ├── fig7_predicted_vs_actual.png
│ └── fig8_model_comparison.png
│
├── scripts/
│ ├── dataset_canon.py
│ ├── clima_era5_incremental.py
│ ├── dataset_final_radio_clima.py
│ ├── graficos_canon.py
│ └── modelo_predictivo.py
│
├── README.md
├── LICENSE


---

## ⚙️ Requirements

Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn xarray netCDF4 cdsapi

Execution Workflow

Run the scripts in the following order:

python scripts/dataset_canon.py
python scripts/clima_era5_incremental.py
python scripts/dataset_final_radio_clima.py
python scripts/graficos_canon.py
python scripts/modelo_predictivo.py

Dataset Description

The dataset includes:

dl_rssi: Received Signal Strength Indicator
dl_snr: Signal-to-noise ratio
temp_c: Air temperature
dewpoint_c: Dew point temperature
precip_mm: Precipitation
press_hpa: Surface pressure
wind_ms: Wind speed
rain_flag: Binary rain indicator
rain_class: Rain intensity category

⚠️ Note: A subset of the dataset is provided for reproducibility purposes. The full dataset is available upon request.

Key Findings
Weak inverse relationship between precipitation and RSSI (≈ -0.207)
Significant influence of multivariable atmospheric conditions
Random Forest model outperforms linear regression:
R² = 0.3941
Lower MAE and RMSE
📊 Generated Results

This repository includes:

Time series analysis of RSSI and precipitation
Scatter and distribution plots
Correlation matrix
Feature importance analysis
Predictive modeling evaluation
