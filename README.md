---
title: Lahore AQI Predictor
emoji: 🌫️
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# 🌫️ Lahore AQI Predictor
 
A real-time Air Quality Index (AQI) prediction dashboard for Lahore, Pakistan — built with Machine Learning, live API integrations, and an automated MLOps pipeline.
 
## 🔴 Live Demo
👉 [ishatahir290105/AQI-Prediction on Hugging Face Spaces](https://huggingface.co/spaces/ishatahir290105/AQI-Prediction)
 
---
 
## ✨ Features
 
- 📊 **Real-time AQI Dashboard** — live PM2.5, PM10, NO₂, O₃, CO readings for Lahore
- 🤖 **ML Prediction** — 24-hour ahead PM2.5 forecast using Random Forest & XGBoost
- 🌤️ **Weather Integration** — temperature, humidity, wind speed, pressure via Open-Meteo
- 📈 **72-hour Forecast** — hourly and daily AQI forecast charts
- 🔍 **Explainability** — SHAP & LIME analysis for model interpretability
- ⚙️ **Automated MLOps** — GitHub Actions runs feature pipeline hourly, retrains model daily
---
 
## 🏗️ Architecture
 
```
Live APIs (AQICN + Open-Meteo)
        ↓
Feature Pipeline (GitHub Actions — every hour)
        ↓
Hopsworks Feature Store
        ↓
Training Pipeline (GitHub Actions — daily 2AM UTC)
        ↓
best_model.pkl (Random Forest / XGBoost)
        ↓
Flask API → Frontend Dashboard (Hugging Face Spaces)
```
 
---
 
## 🤖 Model Performance
 
| Model | RMSE | MAE | R² |
|---|---|---|---|
| Random Forest | 11.69 | 8.62 | 0.947 |
| **XGBoost** ✅ | **5.72** | **4.30** | **0.987** |
 
Best model is auto-selected and saved as `best_model.pkl` after each training run.
 
---
 
## 🔌 API Endpoints
 
| Endpoint | Description |
|---|---|
| `GET /api/current` | Current AQI + weather data |
| `GET /api/forecast` | 72-hour hourly forecast |
| `GET /api/daily-forecast` | 3-day daily summary |
| `GET /api/predict` | ML model prediction (next 24h) |
| `GET /api/health` | Health check |
 
---
 
## 🛠️ Tech Stack
 
| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML Models | Scikit-learn, XGBoost, Random Forest |
| Explainability | SHAP, LIME |
| Feature Store | Hopsworks |
| Data Sources | AQICN API, Open-Meteo API |
| Frontend | HTML, CSS, JavaScript, Chart.js |
| MLOps | GitHub Actions (CI/CD) |
| Deployment | Docker, Hugging Face Spaces |
 
---
 
## ⚙️ MLOps Pipelines
 
**Feature Pipeline** — runs every hour via GitHub Actions
- Fetches live AQI + weather data from APIs
- Engineers 23 features (cyclical time encoding, lag features, rolling averages)
- Pushes to Hopsworks Feature Store
**Training Pipeline** — runs daily at 2AM UTC
- Reads historical data from Feature Store
- Trains Ridge, Random Forest, and XGBoost models
- Evaluates and saves best model
- Commits updated model artifacts back to repo
---
 
## 🔐 Environment Variables
 
Set these as Secrets in your HF Space or GitHub repo:
 
| Variable | Description |
|---|---|
| `AQICN_TOKEN` | API token from [aqicn.org/api](https://aqicn.org/api/) |
| `HOPSWORKS_API_KEY` | API key from [app.hopsworks.ai](https://app.hopsworks.ai/) |
 
---
 
## 👩‍💻 Author
 
**Isha Tahir** — BS Software Engineering, UMT Lahore  
Data Science Intern @ 10Pearls  
GitHub: [@ishatahir290105](https://github.com/ishatahir290105)
