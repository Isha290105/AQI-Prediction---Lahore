"""
backfill.py - Historical Data Backfill Script
Fetches last 6 months of AQI and weather data from Open-Meteo APIs for training
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
LATITUDE = 31.5497  # Lahore coordinates
LONGITUDE = 74.3436
LOOKBACK_DAYS = 180  # 6 months


def fetch_historical_openmeteo(start_date, end_date):
    """
    Fetch historical air quality data from Open-Meteo API

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        DataFrame with air quality data
    """
    print("\nFetching historical air quality data...")
    print(f"  Period: {start_date} to {end_date}")

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "pm10,pm2_5,nitrogen_dioxide,ozone,carbon_monoxide",
        "start_date": start_date,
        "end_date": end_date,
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Check if response contains expected data
        if "hourly" not in data:
            print(f"ERROR: Unexpected API response format")
            return None

        # Create DataFrame
        df = pd.DataFrame({
            "timestamp": data["hourly"]["time"],
            "pm10": data["hourly"]["pm10"],
            "pm25": data["hourly"]["pm2_5"],
            "no2": data["hourly"]["nitrogen_dioxide"],
            "o3": data["hourly"]["ozone"],
            "co": data["hourly"]["carbon_monoxide"],
        })

        print(f"✓ Fetched {len(df)} air quality records")
        return df

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to fetch air quality data: {e}")
        return None
    except (KeyError, ValueError) as e:
        print(f"ERROR: Failed to parse air quality data: {e}")
        return None


def fetch_historical_weather(start_date, end_date):
    """
    Fetch historical weather data from Open-Meteo Archive API

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        DataFrame with weather data
    """
    print("\nFetching historical weather data...")
    print(f"  Period: {start_date} to {end_date}")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Check if response contains expected data
        if "hourly" not in data:
            print(f"ERROR: Unexpected API response format")
            return None

        # Create DataFrame
        df = pd.DataFrame({
            "timestamp": data["hourly"]["time"],
            "temperature": data["hourly"]["temperature_2m"],
            "humidity": data["hourly"]["relative_humidity_2m"],
            "wind_speed": data["hourly"]["wind_speed_10m"],
            "pressure": data["hourly"]["surface_pressure"],
        })

        print(f"✓ Fetched {len(df)} weather records")
        return df

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to fetch weather data: {e}")
        return None
    except (KeyError, ValueError) as e:
        print(f"ERROR: Failed to parse weather data: {e}")
        return None


def combine_and_save(aq_df, weather_df, output_path):
    """
    Combine air quality and weather data, then save to CSV

    Args:
        aq_df: Air quality DataFrame
        weather_df: Weather DataFrame
        output_path: Path to save combined CSV

    Returns:
        Combined DataFrame or None if failed
    """
    print("\nCombining datasets...")

    try:
        # Merge on timestamp
        combined = pd.merge(aq_df, weather_df, on="timestamp", how="inner")

        # Convert timestamp to datetime
        combined["timestamp"] = pd.to_datetime(combined["timestamp"])

        # Sort by timestamp
        combined = combined.sort_values("timestamp")

        # Check for missing data
        missing_count = combined.isnull().sum().sum()
        if missing_count > 0:
            print(f"⚠ Warning: {missing_count} missing values found")

        # Ensure data directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to CSV
        combined.to_csv(output_path, index=False)

        print(f"✓ Combined {len(combined)} records")
        print(f"✓ Saved to: {output_path}")

        return combined

    except Exception as e:
        print(f"ERROR: Failed to combine and save data: {e}")
        return None


def display_summary(df):
    """
    Display summary statistics of the backfilled data

    Args:
        df: Combined DataFrame
    """
    print("\n" + "="*60)
    print("BACKFILL SUMMARY")
    print("="*60)
    print(f"\nTotal Records: {len(df)}")
    print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData Shape: {df.shape}")

    print("\nFirst 5 records:")
    print(df.head())

    print("\nMissing Values:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "None")

    print("\nBasic Statistics:")
    print(df.describe())
    print("="*60)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("HISTORICAL DATA BACKFILL")
    print("="*60)
    print(f"Location: Lahore (Lat: {LATITUDE}, Lon: {LONGITUDE})")
    print(f"Lookback Period: {LOOKBACK_DAYS} days (6 months)")
    print("="*60)

    # Check if backfill data already exists
    output_path = "data/historical_backfill.csv"
    if os.path.exists(output_path):
        print(f"\n✓ Backfill data already exists at {output_path}")
        print("Skipping download. Delete the file to force re-download.")
        print("\n" + "="*60)
        print("BACKFILL SKIPPED (DATA EXISTS)")
        print("="*60)
        exit(0)

    # Calculate date range (last 6 months)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)

    end_str = end_date.strftime("%Y-%m-%d")
    start_str = start_date.strftime("%Y-%m-%d")

    print(f"\nDate Range: {start_str} to {end_str}")

    # Fetch air quality data
    aq_df = fetch_historical_openmeteo(start_str, end_str)
    if aq_df is None:
        print("\nERROR: Failed to fetch air quality data. Aborting backfill.")
        exit(1)

    # Fetch weather data
    weather_df = fetch_historical_weather(start_str, end_str)
    if weather_df is None:
        print("\nERROR: Failed to fetch weather data. Aborting backfill.")
        exit(1)

    # Combine and save
    output_path = "data/historical_backfill.csv"
    combined = combine_and_save(aq_df, weather_df, output_path)

    if combined is None:
        print("\nERROR: Failed to combine and save data. Aborting backfill.")
        exit(1)

    # Display summary
    display_summary(combined)

    print("\n" + "="*60)
    print("BACKFILL COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")
