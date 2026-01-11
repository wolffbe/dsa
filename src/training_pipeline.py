#!/usr/bin/env python3
"""
Training Pipeline: Trains model using Hopsworks Feature Store and registers in Model Registry.
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dotenv import load_dotenv
load_dotenv()

from config import ensure_directories, IMG_DIR, TRAINING_PLOT, FEATURE_IMPORTANCE_PLOT, COMBINED_CSV
from hopsworks_utils import (
    get_feature_store,
    get_model_registry,
    get_training_data,
    register_model
)
from train import plot_training_results, plot_feature_importance, create_features


FEATURE_COLS = [
    'dayofweek', 'dayofmonth', 'month', 'year', 'weekofyear',
    'is_weekend', 'is_month_start', 'is_month_end',
    'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14', 'lag_28',
    'rolling_mean_7', 'rolling_std_7', 'rolling_min_7', 'rolling_max_7',
    'rolling_mean_14', 'rolling_std_14', 'rolling_min_14', 'rolling_max_14',
    'rolling_mean_28', 'rolling_std_28', 'rolling_min_28', 'rolling_max_28',
    'trend_7', 'trend_14',
    'dow_effect', 'month_effect',
    'sentiment_lag_1', 'sentiment_lag_3', 'sentiment_lag_7',
    'sentiment_rolling_7', 'negative_news_rolling_7'
]


def run_training_pipeline(validation_days=30):
    """
    Training pipeline:
    1. Fetch training data from Feature Store
    2. Train Gradient Boosting model
    3. Register model in Model Registry
    """
    print("=" * 60)
    print("TRAINING PIPELINE")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    ensure_directories()
    
    # Connect to Hopsworks
    print("\nConnecting to Hopsworks...")
    fs = get_feature_store()
    mr = get_model_registry()
    print(f"  Connected to: {fs.name}")
    
    # Fetch training data
    print("\nFetching training data from Feature Store...")
    available_features = [f for f in FEATURE_COLS]
    
    try:
        df = get_training_data(fs, available_features)
        print(f"  Loaded {len(df)} samples from Feature Store")
    except Exception as e:
        print(f"  WARNING: Could not fetch from Feature Store: {e}")
        print("  Falling back to local CSV data...")
        
        # Fall back to local CSV
        if not os.path.exists(COMBINED_CSV):
            print(f"  ERROR: {COMBINED_CSV} not found. Run backfill.py first.")
            return
        
        df = pd.read_csv(COMBINED_CSV)
        df['date'] = pd.to_datetime(df['date'])
        df = create_features(df)
        df = df.dropna()
        print(f"  Loaded {len(df)} samples from local CSV")
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"  Loaded {len(df)} training samples")
    
    # Filter available features
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    print(f"  Using {len(available_features)} features")
    
    # Drop NaN
    df = df.dropna(subset=available_features + ['dsa_count'])
    print(f"  {len(df)} samples after dropping NaN")
    
    # Train/validation split
    split_idx = len(df) - validation_days
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    print(f"\nTraining: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    
    X_train = train_df[available_features]
    y_train = train_df['dsa_count']
    X_val = val_df[available_features]
    y_val = val_df['dsa_count']
    
    # Train model
    print("\nTraining Gradient Boosting model...")
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        min_samples_split=5,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    # Calculate MAPE safely (avoid division by zero)
    def safe_mape(y_true, y_pred):
        mask = y_true != 0
        if mask.sum() == 0:
            return 0.0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return float(mape) if np.isfinite(mape) else 999.99
    
    train_metrics = {
        'mae': float(mean_absolute_error(y_train, train_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_train, train_pred))),
        'r2': float(r2_score(y_train, train_pred)),
        'mape': safe_mape(y_train.values, train_pred)
    }
    
    val_metrics = {
        'mae': float(mean_absolute_error(y_val, val_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_val, val_pred))),
        'r2': float(r2_score(y_val, val_pred)),
        'mape': safe_mape(y_val.values, val_pred)
    }
    
    print(f"\nValidation Metrics:")
    print(f"  MAE:  {val_metrics['mae']:,.0f}")
    print(f"  RMSE: {val_metrics['rmse']:,.0f}")
    print(f"  R²:   {val_metrics['r2']:.4f}")
    print(f"  MAPE: {val_metrics['mape']:.2f}%")
    
    # Register model
    print("\nRegistering model in Model Registry...")
    model_obj = register_model(mr, model, available_features, val_metrics)
    
    # Create plots
    plot_training_results(val_df, val_pred, train_metrics, val_metrics)
    plot_feature_importance(model, available_features)
    
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nModel registered: dsa_predictor v{model_obj.version}")
    print(f"Validation R²: {val_metrics['r2']:.4f}")


if __name__ == "__main__":
    run_training_pipeline()
