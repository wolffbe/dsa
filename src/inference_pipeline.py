#!/usr/bin/env python3
"""
Inference Pipeline: Makes predictions using model from Hopsworks Model Registry.
Stores predictions back to Feature Store for dashboard display.
"""

import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from config import ensure_directories, COMBINED_CSV, PREDICTIONS_CSV
from hopsworks_utils import (
    get_feature_store,
    get_model_registry,
    get_latest_model,
    save_prediction,
    get_prediction_history
)
from train import create_features
from inference import create_future_dates, prepare_future_features, save_predictions

FORECAST_DAYS = 7


def run_inference_pipeline():
    """
    Inference pipeline:
    1. Load latest model from Model Registry
    2. Fetch latest features from Feature Store
    3. Make predictions
    4. Save predictions to Feature Store
    """
    print("=" * 60)
    print("INFERENCE PIPELINE")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    ensure_directories()
    
    # Connect to Hopsworks
    print("\nConnecting to Hopsworks...")
    fs = get_feature_store()
    mr = get_model_registry()
    print(f"  Connected to: {fs.name}")
    
    # Load model
    print("\nLoading model from Model Registry...")
    try:
        model, feature_cols, version = get_latest_model(mr)
        print(f"  Loaded dsa_predictor v{version}")
        print(f"  Features: {len(feature_cols)}")
    except Exception as e:
        print(f"  ERROR: Could not load model: {e}")
        print("  Make sure you've run the training pipeline first.")
        return
    
    # Load historical data for feature preparation
    print("\nLoading historical data...")
    if not os.path.exists(COMBINED_CSV):
        print(f"  ERROR: {COMBINED_CSV} not found")
        return
    
    df = pd.read_csv(COMBINED_CSV)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Create features
    df = create_features(df)
    print(f"  Loaded {len(df)} historical records")
    print(f"  Latest date: {df['date'].max().strftime('%Y-%m-%d')}")
    
    # Prepare future dates
    print(f"\nGenerating {FORECAST_DAYS}-day forecast...")
    future_df, last_row, historical = create_future_dates(df, days=FORECAST_DAYS)
    prepared_future = prepare_future_features(future_df, last_row, df, feature_cols)
    
    # Make predictions
    X = prepared_future[feature_cols].values
    predictions = model.predict(X)
    predictions = np.maximum(predictions, 0)  # No negative counts
    
    # Display forecast
    print(f"\nForecast for next {FORECAST_DAYS} days:")
    print("-" * 40)
    for i, (_, row) in enumerate(future_df.iterrows()):
        date_str = row['date'].strftime('%Y-%m-%d (%A)')
        print(f"  {date_str}: {predictions[i]:,.0f}")
    print("-" * 40)
    print(f"  Average: {np.mean(predictions):,.0f}")
    
    # Save all predictions to Feature Store in a single batch
    print("\nSaving predictions to Feature Store...")
    from hopsworks_utils import save_predictions_batch
    save_predictions_batch(fs, future_df['date'].tolist(), predictions.tolist())
    print(f"  Saved {FORECAST_DAYS} predictions to Hopsworks")
    
    # Also save to local CSV for dashboard (Hopsworks read has bugs)
    print("\nSaving predictions to local CSV...")
    predictions_df = pd.DataFrame({
        'target_date': future_df['date'],
        'prediction_date': datetime.now(),
        'predicted': predictions,
        'actual': None
    })
    save_predictions(predictions_df)
    print(f"  Saved {FORECAST_DAYS} predictions to {PREDICTIONS_CSV}")
    
    print("\n" + "=" * 60)
    print("INFERENCE PIPELINE COMPLETE")
    print("=" * 60)
    
    return predictions, future_df


if __name__ == "__main__":
    run_inference_pipeline()
