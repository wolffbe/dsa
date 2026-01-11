#!/usr/bin/env python3
"""
Feature Pipeline: Fetches data and uploads to Hopsworks Feature Store.
Run this daily to keep features up to date.
"""

import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from config import COMBINED_CSV, ensure_directories
from hopsworks_utils import (
    get_feature_store,
    create_dsa_feature_group,
    create_engineered_feature_group
)

# Import feature engineering from train.py
from train import create_features


def run_feature_pipeline(csv_path=COMBINED_CSV):
    """
    Main feature pipeline:
    1. Load combined DSA + news data
    2. Create engineered features
    3. Upload to Hopsworks Feature Store
    """
    print("=" * 60)
    print("FEATURE PIPELINE")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    if not os.path.exists(csv_path):
        print(f"  ERROR: {csv_path} not found. Run backfill.py first.")
        return
    
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    print(f"  Loaded {len(df)} records")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    # Connect to Hopsworks
    print("\nConnecting to Hopsworks...")
    fs = get_feature_store()
    print(f"  Connected to feature store: {fs.name}")
    
    # Upload raw daily features
    print("\nUploading daily features...")
    raw_cols = ['date', 'dsa_count', 'news_count', 'sentiment_mean', 
                'sentiment_min', 'sentiment_max', 'sentiment_std',
                'sentiment_neg_mean', 'sentiment_pos_mean',
                'negative_news_pct', 'positive_news_pct']
    raw_df = df[[c for c in raw_cols if c in df.columns]].copy()
    create_dsa_feature_group(fs, raw_df)
    
    # Create and upload engineered features
    print("\nCreating engineered features...")
    df_eng = create_features(df)
    print(f"  Created {len(df_eng.columns)} features")
    
    # Drop rows with NaN (from lag features)
    df_eng = df_eng.dropna()
    print(f"  {len(df_eng)} rows after dropping NaN")
    
    print("\nUploading engineered features...")
    create_engineered_feature_group(fs, df_eng)
    
    print("\n" + "=" * 60)
    print("FEATURE PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nFeature groups updated in Hopsworks")


if __name__ == "__main__":
    run_feature_pipeline()
