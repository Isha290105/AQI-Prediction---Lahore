"""
fetch_raw.py - Fetch real-time AQI and weather data for Lahore
This script fetches current air quality data from AQICN API and weather data from Open-Meteo API.
"""

import requests
import pandas as pd
from datetime import datetime
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
AQICN_TOKEN=os.getenv("AQICN_TOKEN")
CITY="lahore"
LAT=31.5497  # Lahore latitude
LON=74.3436  # Lahore longitude

# Retry configuration
MAX_RETRIES=3
RETRY_DELAY=2  # seconds


def fetch_with_retry(url,params=None,max_retries=MAX_RETRIES,retry_delay=RETRY_DELAY):
    for attempt in range(max_retries):
        try:
            response=requests.get(url,params=params,timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt<max_retries-1:
                time.sleep(retry_delay)
            else:
                print(f"All retries failed for URL: {url}")
                return None


def fetch_aqicn():
    print("Fetching AQI data from AQICN...")

    if not AQICN_TOKEN:
        print("AQICN_TOKEN not found in .env file!")
        return None

    url=f"https://api.waqi.info/feed/{CITY}/?token={AQICN_TOKEN}"
    data=fetch_with_retry(url)

    if not data or data.get("status")!="ok":
        print("AQICN API error:",data)
        return None

    d=data["data"]
    iaqi=d.get("iaqi",{})

    aqi_data={
        "timestamp":datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "aqi":d.get("aqi"),
        "pm25":iaqi.get("pm25", {}).get("v"),
        "pm10":iaqi.get("pm10", {}).get("v"),
        "no2":iaqi.get("no2", {}).get("v"),
        "co":iaqi.get("co", {}).get("v"),
        "o3":iaqi.get("o3", {}).get("v"),
        "so2":iaqi.get("so2", {}).get("v"),
    }

    print(f"AQI:{aqi_data['aqi']}, PM2.5: {aqi_data['pm25']}")
    return aqi_data


def fetch_weather():
    print("Fetching weather data from Open-Meteo...")

    url="https://api.open-meteo.com/v1/forecast"
    params={
        "latitude": LAT,
        "longitude": LON,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure",
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure",
        "forecast_days": 1
    }

    data=fetch_with_retry(url,params=params)

    if not data or "current" not in data:
        print("Open-Meteo API error:",data)
        return None

    current=data["current"]

    weather_data={
        "temperature":current.get("temperature_2m"),
        "humidity":current.get("relative_humidity_2m"),
        "wind_speed":current.get("wind_speed_10m"),
        "pressure":current.get("surface_pressure"),
    }

    print(f"Temperature: {weather_data['temperature']}°C, Humidity: {weather_data['humidity']}%")
    return weather_data


def fetch_all(save_to_file=True):
    print("\n"+"="*60)
    print("Starting Data Fetch")
    print("="*60)
    print(f"Location: {CITY.title()}")
    print(f"Coordinates: ({LAT}, {LON})")
    print(f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("="*60)
    print()

    # Fetch AQI data
    aqi_data=fetch_aqicn()
    if aqi_data is None:
        print("\nFailed to fetch AQI data. Aborting.")
        return None

    # Fetch weather data
    weather_data=fetch_weather()
    if weather_data is None:
        print("\nFailed to fetch weather data. Aborting.")
        return None

    # Combine data
    combined={**aqi_data,**weather_data}
    df=pd.DataFrame([combined])

    print("\n"+"="*60)
    print("Data Fetched Successfully")
    print("="*60)
    print("\nFetched Data:")
    print(df.to_string(index=False))
    print()

    # Save to file if requested
    if save_to_file:
        save_data(df)

    return df


def save_data(df):
    # Ensure data directory exists
    os.makedirs('data/raw', exist_ok=True)

    csv_path='data/raw/lahore_aqi_raw.csv'

    # Check if file exists to determine if we should append
    file_exists=os.path.exists(csv_path)

    try:
        if file_exists:
            # Append to existing file
            df.to_csv(csv_path, mode='a', header=False, index=False)
            print(f"Data appended to {csv_path}")
        else:
            # Create new file with header
            df.to_csv(csv_path, mode='w', header=True, index=False)
            print(f"Data saved to new file {csv_path}")

        # Show file stats
        existing_df = pd.read_csv(csv_path)
        print(f"Total records in file: {len(existing_df)}")

    except Exception as e:
        print(f"Error saving data to CSV: {e}")


if __name__ == "__main__":
    fetch_all(save_to_file=True)
