"""
feature_pipeline.py - Feature Engineering and Hopsworks Integration
Processes raw AQI and weather data, engineers features, and pushes to Hopsworks Feature Store
"""

import pandas as pd
import numpy as np
from datetime import datetime
import hopsworks
import os
from dotenv import load_dotenv
from fetch_raw import fetch_all

RAW_CSV = "data/raw/lahore_aqi_raw.csv"

# Load environment variables
load_dotenv()


def engineer_features(df):
    print("\n"+"="*60)
    print("Starting Feature Engineering")
    print("="*60)

    # Ensure timestamp is datetime
    df["timestamp"]=pd.to_datetime(df["timestamp"])

    # Time-based features
    print("Creating time-based features...")
    df["hour"]=df["timestamp"].dt.hour
    df["day_of_week"]=df["timestamp"].dt.dayofweek
    df["month"]=df["timestamp"].dt.month
    df["is_weekend"]=df["day_of_week"].isin([5, 6]).astype(int)

    # AQI category mapping
    def aqi_category(aqi):
        """
        Map AQI value to category:
        1: Good (0-50)
        2: Moderate (51-100)
        3: Unhealthy for Sensitive Groups (101-150)
        4: Unhealthy (151-200)
        5: Very Unhealthy (201-300)
        6: Hazardous (301+)
        """
        if pd.isna(aqi):
            return 0
        elif aqi <= 50:
            return 1  # Good
        elif aqi <= 100:
            return 2  # Moderate
        elif aqi <= 150:
            return 3  # Unhealthy for sensitive
        elif aqi <= 200:
            return 4  # Unhealthy
        elif aqi <= 300:
            return 5  # Very Unhealthy
        else:
            return 6  # Hazardous

    print("Creating AQI category feature...")
    df["aqi_category"]=df["aqi"].apply(aqi_category)

    # AQI / PM2.5 change rate (vs previous reading in the raw CSV)
    print("Creating PM2.5 change rate feature...")
    pm25_change = 0.0
    if os.path.exists(RAW_CSV):
        hist = pd.read_csv(RAW_CSV)
        # The newest row was just appended; prev value is second-to-last
        if len(hist) >= 2 and "pm25" in hist.columns:
            prev_pm25 = pd.to_numeric(hist["pm25"].iloc[-2], errors="coerce")
            curr_pm25 = pd.to_numeric(df["pm25"].iloc[0], errors="coerce")
            if not (pd.isna(prev_pm25) or pd.isna(curr_pm25)):
                pm25_change = float(curr_pm25 - prev_pm25)
    df["pm25_change"] = pm25_change

    # Fill missing values using forward fill, then fill remaining with 0
    print("Handling missing values...")
    df=df.ffill().fillna(0)

    print("\nFeature Engineering Complete!")
    print(f"Total features: {len(df.columns)}")
    print(f"Features: {list(df.columns)}")
    print("="*60)

    return df


def push_to_feature_store(df):
    print("\n"+"="*60)
    print("Connecting to Hopsworks Feature Store")
    print("="*60)

    # Login to Hopsworks
    api_key=os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        print("ERROR: HOPSWORKS_API_KEY not found in .env file!")
        return

    try:
        project=hopsworks.login(api_key_value=api_key)
        print(f"✓ Connected to Hopsworks project: {project.name}")
    except Exception as e:
        print(f"ERROR: Failed to connect to Hopsworks: {e}")
        return

    # Get Feature Store
    try:
        fs=project.get_feature_store()
        print(f"✓ Connected to Feature Store")
    except Exception as e:
        print(f"ERROR: Failed to get Feature Store: {e}")
        return

    # Create or get Feature Group
    try:
        print("\nCreating/Getting Feature Group...")
        fg=fs.get_or_create_feature_group(
            name="aqi_features",
            version=1,
            description="AQI and weather features for Lahore with time-based engineered features",
            primary_key=["timestamp"],
            event_time="timestamp",
            online_enabled=False
        )
        print(f"✓ Feature Group 'aqi_features' ready")
    except Exception as e:
        print(f"ERROR: Failed to create Feature Group: {e}")
        return

    # Insert data into Feature Group
    try:
        print("\nInserting features into Feature Store...")
        fg.insert(df, write_options={"wait_for_job": True})
        print("Features pushed to Hopsworks successfully!")
        print(f"  - Records inserted: {len(df)}")
        print(f"  - Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    except Exception as e:
        print(f"ERROR: Failed to insert features: {e}")
        return

    print("="*60)
    print("Feature Store Update Complete!")
    print("="*60)


if __name__ == "__main__":
    print("\n"+"="*60)
    print("AQI FEATURE PIPELINE")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Step 1: Fetch raw data
    print("\nStep 1: Fetching raw data...")
    raw=fetch_all(save_to_file=True)

    if raw is None:
        print("\nERROR: Failed to fetch raw data. Pipeline aborted.")
        exit(1)

    # Step 2: Engineer features
    print("\nStep 2: Engineering features...")
    features=engineer_features(raw)

    # Step 3: Push to Feature Store
    print("\nStep 3: Pushing to Feature Store...")
    push_to_feature_store(features)

    print("\n"+"="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 +"\n")
