from flask import Flask, jsonify, send_from_directory
import joblib
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder='../frontend')

# Load model with error handling
model = None
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl")
try:
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Model loaded successfully!")
    else:
        print(f"Model file not found at {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")

TOKEN = os.getenv("AQICN_TOKEN")
HOPSWORKS_KEY = os.getenv("HOPSWORKS_API_KEY")


def get_aqi_color(aqi):
    if aqi <= 50:
        return "good", "Good", "#57f1db"
    elif aqi <= 100:
        return "moderate", "Moderate", "#57f1db"
    elif aqi <= 150:
        return "sensitive", "Unhealthy (Sensitive)", "#ffad3a"
    elif aqi <= 200:
        return "unhealthy", "Unhealthy", "#ffb4ab"
    elif aqi <= 300:
        return "very_unhealthy", "Very Unhealthy", "#cebdff"
    else:
        return "hazardous", "Hazardous", "#ffb4ab"


def pm25_to_aqi(pm25):
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    if pm25 is None or (isinstance(pm25, float) and np.isnan(pm25)):
        return np.nan
    pm25 = max(0.0, float(pm25))
    for c_lo, c_hi, aqi_lo, aqi_hi in breakpoints:
        if c_lo <= pm25 <= c_hi:
            return round((aqi_hi - aqi_lo) / (c_hi - c_lo) * (pm25 - c_lo) + aqi_lo)
    return 500


def fetch_current_aqi():
    if not TOKEN:
        # Return dummy data if no token
        print("Warning: No AQICN_TOKEN found, returning dummy data!")
        return {
            "aqi": 120,
            "iaqi": {
                "pm25": {"v": 45},
                "pm10": {"v": 80},
                "no2": {"v": 30},
                "o3": {"v": 60},
                "co": {"v": 0.5}
            }
        }
    try:
        resp = requests.get(f"https://api.waqi.info/feed/geo:31.5497;74.3436/?token={TOKEN}", timeout=10).json()
        if resp.get("status") != "ok" or not isinstance(resp.get("data"), dict):
            raise ValueError("AQICN API error")
        return resp["data"]
    except Exception as e:
        print(f"Error fetching AQI data: {e}, returning dummy data!")
        return {
            "aqi": 120,
            "iaqi": {
                "pm25": {"v": 45},
                "pm10": {"v": 80},
                "no2": {"v": 30},
                "o3": {"v": 60},
                "co": {"v": 0.5}
            }
        }


def fetch_current_weather():
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": 31.5497, "longitude": 74.3436,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure",
                "forecast_days": 1,
            },
            timeout=10,
        ).json()
        cur = resp.get("current", {})
        return {
            "temperature": cur.get("temperature_2m", 25),
            "humidity": cur.get("relative_humidity_2m", 60),
            "wind_speed": cur.get("wind_speed_10m", 5),
            "pressure": cur.get("surface_pressure", 1013),
        }
    except Exception as e:
        print(f"Error fetching weather: {e}, returning dummy data!")
        return {"temperature": 25, "humidity": 60, "wind_speed": 5, "pressure": 1013}


def fetch_forecast():
    try:
        url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": 31.5497,
            "longitude": 74.3436,
            "hourly": "pm10,pm2_5,nitrogen_dioxide,ozone",
            "forecast_days": 4,
        }
        data = requests.get(url, params=params, timeout=10).json()
        df = pd.DataFrame({
            "time": pd.to_datetime(data["hourly"]["time"]),
            "pm25": data["hourly"]["pm2_5"],
            "pm10": data["hourly"]["pm10"],
            "no2": data["hourly"]["nitrogen_dioxide"],
            "o3": data["hourly"]["ozone"],
        })
        df["aqi"] = df["pm25"].apply(pm25_to_aqi)
        now = pd.Timestamp.now()
        df = df[df["time"] > now].head(72).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Error fetching forecast: {e}, returning dummy data!")
        # Generate dummy forecast data
        dummy = []
        now = datetime.now()
        for i in range(1, 73):
            dt = now + pd.Timedelta(hours=i)
            dummy.append({
                "time": dt,
                "pm25": 40 + np.random.randint(-20, 40),
                "pm10": 70 + np.random.randint(-30, 50),
                "no2": 25 + np.random.randint(-10, 20),
                "o3": 55 + np.random.randint(-15, 30)
            })
        df = pd.DataFrame(dummy)
        df["aqi"] = df["pm25"].apply(pm25_to_aqi)
        return df


def make_ml_prediction(current_iaqi, current_weather):
    if not model:
        # Dummy prediction if no model
        return 50.0, 100
    FEATURE_COLS = [
        "pm25", "pm10", "no2", "o3", "co",
        "temperature", "humidity", "wind_speed", "pressure",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "day_of_week", "is_weekend",
        "pm25_lag1", "pm25_lag3", "pm25_lag6", "pm25_lag24",
        "pm25_roll3", "pm25_roll6", "pm25_roll24",
        "pm25_change"
    ]

    now = datetime.now()
    curr_pm25 = float(current_iaqi.get("pm25", {}).get("v") or 0)

    features = {
        "pm25": curr_pm25,
        "pm10": float(current_iaqi.get("pm10", {}).get("v") or 0),
        "no2": float(current_iaqi.get("no2", {}).get("v") or 0),
        "o3": float(current_iaqi.get("o3", {}).get("v") or 0),
        "co": float(current_iaqi.get("co", {}).get("v") or 0),
        "temperature": float(current_weather.get("temperature", 25)),
        "humidity": float(current_weather.get("humidity", 60)),
        "wind_speed": float(current_weather.get("wind_speed", 5)),
        "pressure": float(current_weather.get("pressure", 1013)),
        "hour_sin": np.sin(2 * np.pi * now.hour / 24),
        "hour_cos": np.cos(2 * np.pi * now.hour / 24),
        "month_sin": np.sin(2 * np.pi * now.month / 12),
        "month_cos": np.cos(2 * np.pi * now.month / 12),
        "day_of_week": now.weekday(),
        "is_weekend": 1 if now.weekday() >= 5 else 0,
        "pm25_lag1": curr_pm25,
        "pm25_lag3": curr_pm25,
        "pm25_lag6": curr_pm25,
        "pm25_lag24": curr_pm25,
        "pm25_roll3": curr_pm25,
        "pm25_roll6": curr_pm25,
        "pm25_roll24": curr_pm25,
        "pm25_change": 0.0,
    }

    X = np.array([[features[f] for f in FEATURE_COLS]])
    pred_pm25 = max(0.0, float(model.predict(X)[0]))
    pred_aqi = pm25_to_aqi(pred_pm25)
    return pred_pm25, pred_aqi


# Frontend Routes
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(app.static_folder, path)


# API Routes
@app.route('/api/current', methods=['GET'])
def get_current_data():
    try:
        aqi_data = fetch_current_aqi()
        weather_data = fetch_current_weather()
        current_aqi = aqi_data["aqi"]
        emoji, label, color = get_aqi_color(current_aqi)
        iaqi = aqi_data.get("iaqi", {})

        return jsonify({
            "current_aqi": current_aqi,
            "aqi_label": label,
            "aqi_emoji": emoji,
            "aqi_color": color,
            "pm25": iaqi.get("pm25", {}).get("v", 45),
            "pm10": iaqi.get("pm10", {}).get("v", 80),
            "no2": iaqi.get("no2", {}).get("v", 30),
            "o3": iaqi.get("o3", {}).get("v", 60),
            "co": iaqi.get("co", {}).get("v", 0.5),
            "temperature": weather_data.get("temperature", 25),
            "humidity": weather_data.get("humidity", 60),
            "wind_speed": weather_data.get("wind_speed", 5),
            "pressure": weather_data.get("pressure", 1013),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        print(f"Error in /api/current: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    try:
        df = fetch_forecast()
        df["time"] = df["time"].dt.isoformat()
        return jsonify(df.to_dict("records"))
    except Exception as e:
        print(f"Error in /api/forecast: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/daily-forecast', methods=['GET'])
def get_daily_forecast():
    try:
        df = fetch_forecast()
        df["date"] = df["time"].dt.date
        daily = (
            df.groupby("date")
            .agg(MinAQI=("aqi", "min"), AvgAQI=("aqi", "mean"), MaxAQI=("aqi", "max"),
                 AvgPM25=("pm25", "mean"))
            .reset_index()
        )
        daily["date"] = daily["date"].apply(lambda x: x.strftime("%a, %b %d"))
        daily["AvgAQI"] = daily["AvgAQI"].round().astype(int)
        daily["AvgPM25"] = daily["AvgPM25"].round(1)

        result = []
        for _, row in daily.iterrows():
            emoji, label, _ = get_aqi_color(row["AvgAQI"])
            result.append({
                "date": row["date"],
                "aqi": int(row["AvgAQI"]),
                "min_aqi": int(row["MinAQI"]),
                "max_aqi": int(row["MaxAQI"]),
                "pm25": float(row["AvgPM25"]),
                "label": label,
                "emoji": emoji
            })

        return jsonify(result)
    except Exception as e:
        print(f"Error in /api/daily-forecast: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/predict', methods=['GET'])
def predict():
    try:
        aqi_data = fetch_current_aqi()
        weather_data = fetch_current_weather()
        pred_pm25, pred_aqi = make_ml_prediction(aqi_data.get("iaqi", {}), weather_data)
        pred_emoji, pred_label, _ = get_aqi_color(pred_aqi)

        return jsonify({
            "predicted_pm25_24h": round(float(pred_pm25), 1),
            "predicted_aqi": int(pred_aqi),
            "predicted_label": pred_label,
            "predicted_emoji": pred_emoji,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        print(f"Error in /api/predict: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")
