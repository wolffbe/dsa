#!/usr/bin/env python3
"""
Hopsworks integration utilities for DSA prediction project.
Handles feature store and model registry operations.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

try:
    import hopsworks
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "hopsworks", "-q"])
    import hopsworks

from dotenv import load_dotenv
load_dotenv()


def get_hopsworks_project():
    """Connect to Hopsworks and return project handle."""
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT")
    
    if not api_key:
        raise ValueError("HOPSWORKS_API_KEY not found in .env file")
    
    project = hopsworks.login(api_key_value=api_key, project=project_name)
    return project


def get_feature_store():
    """Get Hopsworks feature store."""
    project = get_hopsworks_project()
    return project.get_feature_store()


def get_model_registry():
    """Get Hopsworks model registry."""
    project = get_hopsworks_project()
    return project.get_model_registry()


# ============================================================================
# FEATURE GROUP OPERATIONS
# ============================================================================

def create_dsa_feature_group(fs, df):
    """
    Create or update the DSA violations feature group.
    
    Features:
    - date_id (primary key, string format YYYY-MM-DD)
    - dsa_count: total violations
    - news_count: number of articles
    - sentiment_mean/min/max/std
    - negative_news_pct, positive_news_pct
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Use string date_id as primary key to avoid timestamp issues
    df['date_id'] = df['date'].dt.strftime('%Y-%m-%d')
    
    # Hopsworks requires a timestamp column for time-travel
    df['event_time'] = df['date']
    
    dsa_fg = fs.get_or_create_feature_group(
        name="dsa_daily_features",
        version=3,
        description="Daily DSA violation counts with news sentiment features",
        primary_key=["date_id"],
        event_time="event_time",
        online_enabled=False,
    )
    
    dsa_fg.insert(df, write_options={"wait_for_job": True})
    print(f"  Inserted {len(df)} rows into dsa_daily_features")
    
    return dsa_fg


def create_engineered_feature_group(fs, df):
    """
    Create feature group with engineered features (lag, rolling, etc).
    These are the features used for model training.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Use string date_id as primary key to avoid timestamp issues
    df['date_id'] = df['date'].dt.strftime('%Y-%m-%d')
    df['event_time'] = df['date']
    
    # Select only numeric columns and date
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_keep = ['date_id', 'date', 'event_time'] + numeric_cols
    df = df[[c for c in cols_to_keep if c in df.columns]]
    
    eng_fg = fs.get_or_create_feature_group(
        name="dsa_engineered_features",
        version=3,
        description="Engineered features for DSA prediction (lag, rolling, calendar)",
        primary_key=["date_id"],
        event_time="event_time",
        online_enabled=False,
    )
    
    eng_fg.insert(df, write_options={"wait_for_job": True})
    print(f"  Inserted {len(df)} rows into dsa_engineered_features")
    
    return eng_fg


def get_training_data(fs, feature_cols, start_date=None, end_date=None):
    """
    Retrieve training data from feature store.
    """
    fg = fs.get_feature_group("dsa_engineered_features", version=3)
    
    # Read all data from feature group using Hive to avoid Arrow Flight issues
    df = fg.read(read_options={"use_hive": True})
    
    # Convert date_id back to date
    if 'date_id' in df.columns:
        df['date'] = pd.to_datetime(df['date_id'])
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Apply time filter if specified
    if start_date:
        df = df[df['date'] >= start_date]
    if end_date:
        df = df[df['date'] <= end_date]
    
    # Select only requested columns (plus date and target)
    cols_needed = list(set(feature_cols + ['dsa_count', 'date']))
    available_cols = [c for c in cols_needed if c in df.columns]
    
    return df[available_cols]


def get_latest_features(fs, feature_cols):
    """
    Get the most recent feature values for inference.
    """
    fg = fs.get_feature_group("dsa_engineered_features", version=3)
    df = fg.read(read_options={"use_hive": True})
    
    # Convert date_id back to date
    if 'date_id' in df.columns:
        df['date'] = pd.to_datetime(df['date_id'])
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    df = df.sort_values('date', ascending=False)
    
    return df.head(1)[feature_cols]


# ============================================================================
# MODEL REGISTRY OPERATIONS  
# ============================================================================

def register_model(mr, model, feature_cols, metrics, model_name="dsa_predictor"):
    """
    Register a trained model in Hopsworks Model Registry.
    """
    import joblib
    import json
    import tempfile
    import os
    
    # Create temp directory for model artifacts
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save model
        model_path = os.path.join(tmpdir, "model.pkl")
        joblib.dump(model, model_path)
        
        # Save feature columns
        features_path = os.path.join(tmpdir, "feature_cols.json")
        with open(features_path, 'w') as f:
            json.dump(feature_cols, f)
        
        # Save metrics
        metrics_path = os.path.join(tmpdir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
        
        # Create model in registry
        dsa_model = mr.python.create_model(
            name=model_name,
            metrics=metrics,
            description="Gradient Boosting model for DSA violation prediction",
            input_example=None,
            model_schema=None,
        )
        
        # Upload artifacts
        dsa_model.save(tmpdir)
        
        print(f"  Registered model '{model_name}' version {dsa_model.version}")
        return dsa_model


def get_latest_model(mr, model_name="dsa_predictor"):
    """
    Download the latest model from registry.
    """
    import joblib
    import json
    
    model_obj = mr.get_model(model_name, version=None)  # Latest version
    model_dir = model_obj.download()
    
    model = joblib.load(os.path.join(model_dir, "model.pkl"))
    
    with open(os.path.join(model_dir, "feature_cols.json"), 'r') as f:
        feature_cols = json.load(f)
    
    return model, feature_cols, model_obj.version


# ============================================================================
# PREDICTION OPERATIONS
# ============================================================================

def save_prediction(fs, prediction_date, predicted_value, actual_value=None):
    """
    Save a prediction to the predictions feature group.
    """
    target_date = pd.to_datetime(prediction_date)
    
    # Use -1 as sentinel for missing actual values (Hopsworks doesn't support null)
    actual = float(actual_value) if actual_value is not None else -1.0
    error = float(actual_value - predicted_value) if actual_value is not None else -1.0
    
    df = pd.DataFrame([{
        'target_date_id': target_date.strftime('%Y-%m-%d'),
        'prediction_date': datetime.now(),
        'target_date': target_date,
        'predicted': float(predicted_value),
        'actual': actual,
        'error': error,
        'event_time': datetime.now(),
    }])
    
    pred_fg = fs.get_or_create_feature_group(
        name="dsa_predictions",
        version=4,
        description="DSA prediction history for accuracy tracking",
        primary_key=["target_date_id"],
        event_time="event_time",
        online_enabled=False,
    )
    
    pred_fg.insert(df, write_options={"wait_for_job": True})
    return pred_fg


def save_predictions_batch(fs, dates, predicted_values):
    """
    Save multiple predictions to the predictions feature group in a single batch.
    Much faster than individual inserts.
    """
    records = []
    for date, predicted in zip(dates, predicted_values):
        target_date = pd.to_datetime(date)
        records.append({
            'target_date_id': target_date.strftime('%Y-%m-%d'),
            'prediction_date': datetime.now(),
            'target_date': target_date,
            'predicted': float(predicted),
            'actual': -1.0,  # Sentinel for missing value
            'error': -1.0,   # Sentinel for missing value
            'event_time': datetime.now(),
        })
    
    df = pd.DataFrame(records)
    
    pred_fg = fs.get_or_create_feature_group(
        name="dsa_predictions",
        version=4,
        description="DSA prediction history for accuracy tracking",
        primary_key=["target_date_id"],
        event_time="event_time",
        online_enabled=False,
    )
    
    pred_fg.insert(df, write_options={"wait_for_job": True})
    return pred_fg


def get_prediction_history(fs, limit=30):
    """
    Get recent prediction history for dashboard display.
    """
    try:
        pred_fg = fs.get_feature_group("dsa_predictions", version=4)
        df = pred_fg.read(read_options={"use_hive": True})
        if 'target_date_id' in df.columns:
            df['target_date'] = pd.to_datetime(df['target_date_id'])
        elif 'target_date' in df.columns:
            df['target_date'] = pd.to_datetime(df['target_date'])
        
        # Remove duplicates - keep latest prediction for each target date
        df = df.sort_values('prediction_date', ascending=False)
        df = df.drop_duplicates(subset=['target_date_id'], keep='first')
        
        df = df.sort_values('target_date', ascending=False).head(limit)
        return df
    except Exception as e:
        print(f"Error reading predictions from Hopsworks: {e}")
        return pd.DataFrame()
