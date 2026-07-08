# Pearls AQI Predictor — Project Report

**City:** Lahore, Pakistan
**Date:** May 2026
**Stack:** Python · Hopsworks · GitHub Actions · Streamlit · Flask

---

## 1. Project Objective

Build a complete, end-to-end Air Quality Index (AQI) prediction system for Lahore using a serverless MLOps architecture. The system automatically ingests data, stores features in a cloud Feature Store, trains multiple ML models daily, registers the best model in a Model Registry, and serves real-time forecasts through an interactive web dashboard.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  EVERY HOUR — GitHub Actions                                    │
│  fetch_raw.py  →  feature_pipeline.py  →  Hopsworks FS         │
│  (AQICN + Open-Meteo)   (engineer features)   (aqi_features v1)│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  EVERY DAY 02:00 UTC — GitHub Actions                           │
│  training_pipeline.py                                           │
│  ├─ Read from Hopsworks Feature Store                           │
│  ├─ Train Ridge + Random Forest + LSTM                          │
│  ├─ Generate SHAP + LIME explanations                           │
│  └─ Save best model to Hopsworks Model Registry                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  LIVE — Streamlit Dashboard (app/dashboard.py)                  │
│  ├─ Load model from Hopsworks Model Registry                    │
│  ├─ Fetch recent features from Hopsworks Feature Store          │
│  ├─ Compute ML prediction (PM2.5 & AQI, 24h ahead)             │
│  ├─ Show real-time AQI gauge, WHO comparisons                   │
│  ├─ Show 3-day Open-Meteo forecast                              │
│  └─ Feature importance (SHAP/RF importances)                    │
│                                                                 │
│  Flask API (app/api.py)                                         │
│  └─ GET /predict  →  JSON: predicted PM2.5, AQI, alert         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Sources

| Source | API Endpoint | Data Provided |
|--------|--------------|---------------|
| **AQICN** | `api.waqi.info` | Real-time AQI, PM2.5, PM10, NO2, CO, O3, SO2 |
| **Open-Meteo Forecast** | `api.open-meteo.com` | Current weather: temp, humidity, wind, pressure |
| **Open-Meteo Air Quality** | `air-quality-api.open-meteo.com` | Hourly 3-day pollutant forecast |
| **Open-Meteo Archive** | `archive-api.open-meteo.com` | Historical weather & air quality (backfill) |
| **Hopsworks** | Cloud Feature Store | Centralized feature storage & retrieval |

---

## 4. Feature Engineering

### 4.1 Features in Feature Store (`feature_pipeline.py`)

| Feature | Type | Description |
|---------|------|-------------|
| `aqi` | Raw | Overall Air Quality Index |
| `pm25`, `pm10`, `no2`, `co`, `o3`, `so2` | Raw | Pollutant readings (µg/m³) |
| `temperature`, `humidity`, `wind_speed`, `pressure` | Raw | Weather conditions |
| `hour` | Time | Hour of day (0–23) |
| `day_of_week` | Time | Day of week (0=Mon, 6=Sun) |
| `month` | Time | Calendar month (1–12) |
| `is_weekend` | Time | Binary: Saturday or Sunday |
| `aqi_category` | Derived | AQI band (1=Good → 6=Hazardous) |
| `pm25_change` | Derived | PM2.5 change from previous reading |

### 4.2 Additional Features Computed at Training Time

| Feature | Description |
|---------|-------------|
| `pm25_lag1/3/6/24` | PM2.5 values 1, 3, 6, 24 hours ago |
| `pm25_roll3/6/24` | Rolling average PM2.5 over 3, 6, 24 hours |

**Target Variable:** PM2.5 value 24 hours ahead (`pm25.shift(-24)`)

---

## 5. Backfill

`pipelines/backfill.py` was run once to populate 6 months of historical data from the Open-Meteo Archive API (lat: 31.5497, lon: 74.3436). This dataset was used as training data before sufficient Feature Store history accumulated.

The data covers hourly readings of PM2.5, PM10, NO2, O3, CO, plus temperature, humidity, wind speed, and surface pressure, resulting in ~4,300 rows.

---

## 6. ML Models

Three models were trained and evaluated on an 80/20 chronological train/test split.

### 6.1 Ridge Regression
- L2 regularization (alpha = 1.0)
- Features scaled with `StandardScaler`
- Serves as linear baseline

### 6.2 Random Forest Regressor
- 100 estimators, max_depth=15, min_samples_split=5
- No scaling required
- Selected as the production model

### 6.3 LSTM Neural Network (Deep Learning)
- Architecture: LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16, ReLU) → Dense(1)
- Sequence length: 24 timesteps
- Optimizer: Adam, Loss: MSE
- EarlyStopping (patience=5, restore best weights)
- Optional: only trained when TensorFlow is available

### 6.4 Evaluation Results

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| Ridge Regression | 22.73 | 17.61 | −0.581 |
| Random Forest | 17.00 | 13.77 | 0.116 |

**Analysis:**
- Ridge Regression underperforms (negative R²), indicating the relationship is non-linear and a simple linear model cannot capture it adequately.
- Random Forest (R² = 0.116) is significantly better, though there is room for improvement. The low R² is largely due to the high variability of PM2.5 in Lahore (heavily affected by seasonal smog, crop burning, and traffic patterns that are not fully captured by the 6-month backfill dataset).
- With more historical data (12+ months) and additional features (crop burning calendar, traffic proxies, dew point), model performance is expected to improve substantially.

---

## 7. Model Interpretability

### 7.1 SHAP (SHapley Additive exPlanations)
- Used `shap.TreeExplainer` on the Random Forest model
- Generates a beeswarm summary plot (`models/shap_summary.png`)
- Displayed as an interactive bar chart on the Streamlit dashboard
- **Key finding:** PM2.5 lag features (especially 24h ago) are the strongest predictors, followed by humidity and wind speed.

### 7.2 LIME (Local Interpretable Model-agnostic Explanations)
- Used `lime.lime_tabular.LimeTabularExplainer` in regression mode
- Generates HTML explanation files for 3 individual test predictions (`models/lime_explanation_0/1/2.html`)
- Saves a summary bar-chart (`models/lime_summary.png`)
- LIME provides instance-level explanations complementing the global SHAP view.

---

## 8. MLOps Pipeline

### 8.1 Feature Store (Hopsworks)
- **Feature Group:** `aqi_features` (version 1)
- Hourly inserts via `feature_pipeline.py`
- Training pipeline reads the full history via `fg.read()`
- Dashboard fetches the last 30 rows for lag feature computation during real-time prediction

### 8.2 Model Registry (Hopsworks)
- **Model name:** `aqi_predictor`
- Registered after each daily training run via `mr.sklearn.create_model()`
- Artifacts uploaded: `best_model.pkl`, `scaler.pkl`, `shap_summary.png`, `lime_summary.png`, `evaluation_results.csv`
- Dashboard loads the best-performing version via `mr.get_best_model("aqi_predictor", "r2", "max")`

### 8.3 CI/CD (GitHub Actions)

| Workflow | Schedule | Actions |
|----------|----------|---------|
| `feature_pipeline.yml` | Every hour (`0 * * * *`) | Fetch data → engineer features → push to Hopsworks FS |
| `training_pipeline.yml` | Daily 02:00 UTC | Read FS → train models → SHAP/LIME → register in Model Registry |

Both workflows use GitHub Secrets (`AQICN_TOKEN`, `HOPSWORKS_API_KEY`) and upload failure logs as artifacts.

---

## 9. Web Application

### 9.1 Streamlit Dashboard (`app/dashboard.py`)

| Section | Description |
|---------|-------------|
| 🤖 ML Model Prediction | 24h ahead PM2.5 + AQI from Random Forest (loaded from Model Registry) |
| 📊 Current AQI | Plotly gauge with colored bands (Good → Hazardous) |
| ⚖️ Pollutants vs WHO | Grouped bar chart: current levels vs WHO 24-hr guidelines |
| 📈 3-Day Forecast | Hourly AQI time series with AQI band annotations |
| 📅 Daily Summary | 3-day cards: min/avg/max AQI per day |
| 🔬 24h Pollutants | PM2.5, PM10, NO2, O3 comparison for next 24 hours |
| 🧠 Feature Importance | Interactive horizontal bar chart, color-coded by category |

**Alerts:**
- AQI > 150: `st.error()` — HAZARD ALERT
- AQI > 100: `st.warning()` — WARNING for sensitive groups
- ML prediction > 150: separate 24h-ahead HAZARD alert

### 9.2 Flask REST API (`app/api.py`)

| Endpoint | Method | Response |
|----------|--------|----------|
| `/predict` | GET | `{current_aqi, predicted_pm25_24h, timestamp, alert}` |
| `/health` | GET | `{"status": "ok"}` |

---

## 10. Exploratory Data Analysis

`notebooks/EDA.ipynb` covers:
- Distribution analysis of AQI and pollutants
- Time-series trends (daily, weekly, seasonal patterns)
- Correlation heatmap between features
- AQI band frequency analysis
- Missing value assessment

**Key EDA findings:**
- PM2.5 is highly correlated with overall AQI in Lahore (r > 0.95)
- AQI peaks in November–January (winter smog season)
- Significant diurnal cycle: AQI rises at morning/evening rush hours
- Wind speed and humidity are the strongest weather-side predictors

---

## 11. Challenges and Solutions

| Challenge | Solution |
|-----------|----------|
| Requirements file encoding (UTF-16) | Rewrote `requirements.txt` as UTF-8 |
| Headless matplotlib in GitHub Actions | Set `matplotlib.use('Agg')` before pyplot import |
| Deprecated pandas `.append()` | Replaced with `pd.concat()` |
| models/ missing before CI run | Added `os.makedirs("models", exist_ok=True)` |
| Feature Store lag features require full history | Lag/rolling features computed at training time from full FS dataset; real-time dashboard reads last 30 rows from FS |

---

## 12. Conclusion

The system successfully delivers a complete, automated MLOps pipeline for AQI prediction:

- **Feature Store integration:** Hopsworks stores hourly features; training reads directly from it.
- **Model Registry:** Every trained model is tracked with metrics and artifacts in Hopsworks.
- **Automated pipelines:** GitHub Actions ensures zero-touch data ingestion and daily retraining.
- **Interactive dashboard:** Real-time gauge, 3-day forecasts, ML-model 24h predictions, SHAP/LIME explanations, and hazard alerts.
- **Multiple models:** Ridge Regression, Random Forest, and LSTM — covering statistical to deep learning approaches.

**Future improvements:**
1. Expand backfill to 24+ months to improve LSTM performance.
2. Add hyperparameter tuning (GridSearchCV / Keras Tuner).
3. Incorporate additional features (festival calendar, crop-burning events, traffic density).
4. Deploy dashboard to Streamlit Community Cloud with auto-restart.
5. Add XGBoost / LightGBM for comparison.
