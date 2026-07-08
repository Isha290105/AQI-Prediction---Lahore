"""
training_pipeline.py - Model Training Pipeline for AQI Prediction
Trains machine learning models to predict PM2.5 levels 24 hours ahead
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from datetime import datetime
from dotenv import load_dotenv
import hopsworks

load_dotenv()

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: lime not installed. LIME analysis will be skipped.")

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: xgboost not installed. XGBoost training will be skipped.")


def load_data():
    """Load historical features from Hopsworks Feature Store (fallback: local CSV)."""
    print("\n" + "="*60)
    print("Loading Data")
    print("="*60)

    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        print("Warning: HOPSWORKS_API_KEY not set — falling back to local CSV.")
        return _load_from_csv()

    try:
        project = hopsworks.login(api_key_value=api_key)
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name="aqi_features", version=1)
        df = fg.read()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        print(f"✓ Loaded {len(df)} records from Feature Store")
        print(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"✓ Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Error reading from Feature Store: {e}")
        print("Falling back to local CSV...")
        return _load_from_csv()


def _load_from_csv():
    """Fallback: load historical backfill from local CSV."""
    data_path = "data/historical_backfill.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run pipelines/backfill.py first.")
        return None
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"✓ Loaded {len(df)} records from local CSV")
    return df


def create_features_and_target(df):
    """
    Engineer features and create target variable for training
    """
    print("\n" + "="*60)
    print("Feature Engineering")
    print("="*60)

    # Time-based features
    print("Creating time-based features...")
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Cyclical encoding for hour and month
    print("Adding cyclical encoding for time features...")
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Lag features — past PM2.5 values
    print("Creating lag features...")
    df["pm25_lag1"] = df["pm25"].shift(1)
    df["pm25_lag3"] = df["pm25"].shift(3)
    df["pm25_lag6"] = df["pm25"].shift(6)
    df["pm25_lag24"] = df["pm25"].shift(24)

    # Rolling averages
    print("Creating rolling average features...")
    df["pm25_roll3"] = df["pm25"].rolling(3).mean()
    df["pm25_roll6"] = df["pm25"].rolling(6).mean()
    df["pm25_roll24"] = df["pm25"].rolling(24).mean()

    # Change rate
    print("Creating rate of change feature...")
    df["pm25_change"] = df["pm25"].diff()

    # Target — PM2.5 value 24 hours ahead
    print("Creating target variable (24h ahead prediction)...")
    df["target"] = df["pm25"].shift(-24)

    # Drop rows with NaN values
    initial_rows = len(df)
    df = df.dropna()
    dropped_rows = initial_rows - len(df)
    print(f"✓ Dropped {dropped_rows} rows with missing values")

    # Define feature columns
    feature_cols = [
        # Direct features
        "pm25", "pm10", "no2", "o3", "co",
        # Weather features
        "temperature", "humidity", "wind_speed", "pressure",
        # Time features (cyclical + day of week + weekend)
        "hour_sin", "hour_cos", "month_sin", "month_cos", "day_of_week", "is_weekend",
        # Lag features
        "pm25_lag1", "pm25_lag3", "pm25_lag6", "pm25_lag24",
        # Rolling features
        "pm25_roll3", "pm25_roll6", "pm25_roll24",
        # Change rate
        "pm25_change"
    ]

    X = df[feature_cols]
    y = df["target"]

    print(f"✓ Feature matrix shape: {X.shape}")
    print(f"✓ Target vector shape: {y.shape}")
    print(f"✓ Total features: {len(feature_cols)}")

    return X, y, feature_cols


def evaluate(y_true, y_pred, model_name):
    """Evaluate model performance using multiple metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name} Results:")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  MAE  : {mae:.2f}")
    print(f"  R²   : {r2:.4f}")

    return {"model": model_name, "rmse": rmse, "mae": mae, "r2": r2}


def train_models(X, y):
    """Train multiple models using TimeSeriesSplit"""
    print("\n" + "="*60)
    print("Model Training with TimeSeriesSplit")
    print("="*60)

    # Time Series Split
    tscv = TimeSeriesSplit(n_splits=5)
    results = []

    # Standard Scaler for Ridge Regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Ridge Regression
    print("\n" + "-"*60)
    print("Training Ridge Regression...")
    print("-"*60)
    ridge = Ridge(alpha=1.0)
    ridge_scores = cross_val_score(ridge, X_scaled, y, cv=tscv, scoring='r2')
    print(f"R² across folds: {ridge_scores}")
    print(f"Mean R²: {ridge_scores.mean():.3f} ± {ridge_scores.std():.3f}")
    ridge.fit(X_scaled, y)

    # 2. Random Forest (tuned hyperparameters)
    print("\n" + "-"*60)
    print("Training Random Forest...")
    print("-"*60)
    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    rf_scores = cross_val_score(rf, X, y, cv=tscv, scoring='r2')
    print(f"R² across folds: {rf_scores}")
    print(f"Mean R²: {rf_scores.mean():.3f} ± {rf_scores.std():.3f}")
    rf.fit(X, y)
    rf_preds = rf.predict(X)
    rf_eval = evaluate(y, rf_preds, "Random Forest")
    results.append(rf_eval)

    best_model = rf
    best_name = "Random Forest"

    # 3. XGBoost (if available)
    if XGBOOST_AVAILABLE:
        print("\n" + "-"*60)
        print("Training XGBoost...")
        print("-"*60)
        xgb = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        xgb_scores = cross_val_score(xgb, X, y, cv=tscv, scoring='r2')
        print(f"R² across folds: {xgb_scores}")
        print(f"Mean R²: {xgb_scores.mean():.3f} ± {xgb_scores.std():.3f}")
        xgb.fit(X, y)
        xgb_preds = xgb.predict(X)
        xgb_eval = evaluate(y, xgb_preds, "XGBoost")
        results.append(xgb_eval)
        if xgb_eval["r2"] > rf_eval["r2"]:
            best_model = xgb
            best_name = "XGBoost"

    # Save best model
    print("\n" + "="*60)
    print(f"Saving Best Model ({best_name})")
    print("="*60)

    os.makedirs("models", exist_ok=True)

    try:
        joblib.dump(best_model, "models/best_model.pkl")
        joblib.dump(scaler, "models/scaler.pkl")
        print(f"✓ Best model ({best_name}) saved to models/best_model.pkl")
        print("✓ Scaler saved to models/scaler.pkl")
    except Exception as e:
        print(f"Error: Failed to save models: {e}")

    return best_model, scaler, X, results


def run_shap(model, X, feature_cols):
    """Generate SHAP plots for model interpretability"""
    print("\n" + "="*60)
    print("SHAP Analysis")
    print("="*60)

    try:
        print("Computing SHAP values...")
        if hasattr(model, 'get_booster'):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        print("Generating SHAP summary plot...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_cols, show=False)
        plt.tight_layout()

        output_path = "models/shap_summary.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ SHAP plot saved to {output_path}")
    except Exception as e:
        print(f"Error: Failed to generate SHAP plot: {e}")


def run_lime(model, X, feature_cols):
    """Generate LIME explanations"""
    print("\n" + "="*60)
    print("LIME Analysis")
    print("="*60)

    if not LIME_AVAILABLE:
        return

    try:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X.values,
            feature_names=feature_cols,
            mode="regression",
            random_state=42,
        )

        os.makedirs("models", exist_ok=True)
        n_explain = min(3, len(X))

        for i in range(n_explain):
            exp = explainer.explain_instance(
                X.iloc[i].values,
                model.predict,
                num_features=len(feature_cols),
            )
            html_path = f"models/lime_explanation_{i}.html"
            exp.save_to_file(html_path)
            print(f"✓ LIME explanation {i+1} saved to {html_path}")

        exp = explainer.explain_instance(
            X.iloc[0].values,
            model.predict,
            num_features=len(feature_cols),
        )
        fig = exp.as_pyplot_figure()
        plt.title("LIME Feature Importance (sample prediction)")
        plt.tight_layout()
        plt.savefig("models/lime_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("✓ LIME summary plot saved to models/lime_summary.png")
    except Exception as e:
        print(f"Error: Failed to generate LIME explanations: {e}")


def save_to_model_registry(model, scaler, best_metrics):
    """Upload the trained model and artifacts to Hopsworks Model Registry"""
    print("\n" + "="*60)
    print("Saving Model to Hopsworks Model Registry")
    print("="*60)

    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        print("Warning: HOPSWORKS_API_KEY not set — skipping Model Registry upload.")
        return

    try:
        project = hopsworks.login(api_key_value=api_key)
        mr = project.get_model_registry()

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/best_model.pkl")
        joblib.dump(scaler, "models/scaler.pkl")

        aqi_model = mr.sklearn.create_model(
            name="aqi_predictor",
            metrics={
                "rmse": round(float(best_metrics["rmse"]), 4),
                "mae": round(float(best_metrics["mae"]), 4),
                "r2": round(float(best_metrics["r2"]), 4),
            },
            description="AQI Predictor Model",
        )
        aqi_model.save("models/")
        print(f"✓ Model registered in Hopsworks Model Registry as 'aqi_predictor'")
    except Exception as e:
        print(f"Error: Failed to save model to registry: {e}")


def save_results(results):
    """Save evaluation results to CSV"""
    try:
        results_df = pd.DataFrame(results)
        output_path = "models/evaluation_results.csv"
        results_df.to_csv(output_path, index=False)
        print(f"✓ Evaluation results saved to {output_path}")
    except Exception as e:
        print(f"Error: Failed to save results: {e}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("AQI TRAINING PIPELINE (Improved)")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Step 1: Load data
    df = load_data()
    if df is None:
        print("\nError: Failed to load data. Pipeline aborted.")
        exit(1)

    # Step 2: Create features and target
    X, y, feature_cols = create_features_and_target(df)
    if X is None or y is None:
        print("\nError: Failed to create features. Pipeline aborted.")
        exit(1)

    print(f"\nDataset summary:")
    print(f"  Rows: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Target mean: {y.mean():.2f}")
    print(f"  Target std: {y.std():.2f}")

    # Step 3: Train models
    model, scaler, X, results = train_models(X, y)

    # Step 4: SHAP analysis
    run_shap(model, X, feature_cols)

    # Step 5: LIME analysis
    run_lime(model, X, feature_cols)

    # Step 6: Save evaluation results
    save_results(results)

    # Step 7: Upload to Hopsworks Model Registry
    best_metrics = results[-1] if results else None
    if best_metrics:
        save_to_model_registry(model, scaler, best_metrics)

    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")
