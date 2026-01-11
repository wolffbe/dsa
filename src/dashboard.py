#!/usr/bin/env python3
"""
Streamlit Dashboard for DSA Prediction Monitoring.
Displays real-time predictions, historical accuracy, and feature insights.

Run with: streamlit run src/dashboard.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    import streamlit as st
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "-q"])
    import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly", "-q"])
    import plotly.express as px
    import plotly.graph_objects as go

from dotenv import load_dotenv
load_dotenv()

from config import COMBINED_CSV, PREDICTIONS_CSV


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="DSA Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_historical_data():
    """Load historical DSA + news data."""
    if os.path.exists(COMBINED_CSV):
        df = pd.read_csv(COMBINED_CSV)
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date')
    return pd.DataFrame()


@st.cache_data(ttl=60)  # Cache for 1 minute
def load_predictions():
    """Load prediction history."""
    if os.path.exists(PREDICTIONS_CSV):
        df = pd.read_csv(PREDICTIONS_CSV)
        df['target_date'] = pd.to_datetime(df['target_date'])
        df['prediction_date'] = pd.to_datetime(df['prediction_date'])
        return df.sort_values('target_date', ascending=False)
    return pd.DataFrame()


def try_load_from_hopsworks():
    """Try to load data from Hopsworks Feature Store."""
    try:
        from hopsworks_utils import get_feature_store, get_prediction_history
        fs = get_feature_store()
        
        # Get predictions (version 4)
        pred_df = get_prediction_history(fs, limit=100)
        
        # Show debug info
        if pred_df.empty:
            st.warning("Predictions DataFrame is empty from Hopsworks")
        else:
            st.success(f"Loaded {len(pred_df)} predictions from Hopsworks")
        
        # Filter out sentinel values (-1 means no actual value yet)
        if not pred_df.empty and 'actual' in pred_df.columns:
            pred_df.loc[pred_df['actual'] == -1, 'actual'] = None
        
        # Get historical data from local CSV (Hopsworks read has issues)
        hist_df = load_historical_data()
        
        return hist_df, pred_df, True
    except Exception as e:
        st.error(f"Hopsworks error: {e}")
        return None, None, False


# ============================================================================
# DASHBOARD LAYOUT
# ============================================================================

def main():
    st.title("📊 DSA Content Moderation Prediction Dashboard")
    st.markdown("Monitoring EU Digital Services Act violations and prediction accuracy")
    
    # Sidebar
    st.sidebar.header("Settings")
    data_source = st.sidebar.radio(
        "Data Source",
        ["Local CSV", "Hopsworks Feature Store"],
        index=0
    )
    
    days_to_show = st.sidebar.slider("Days to display", 7, 365, 90)
    
    # Load data
    if data_source == "Hopsworks Feature Store":
        hist_df, pred_df, success = try_load_from_hopsworks()
        if not success:
            st.warning("Could not connect to Hopsworks. Falling back to local CSV.")
            hist_df = load_historical_data()
            pred_df = load_predictions()
    else:
        hist_df = load_historical_data()
        pred_df = load_predictions()
    
    if hist_df.empty:
        st.error("No historical data found. Run `make backfill` first.")
        return
    
    # Filter to recent data
    cutoff = datetime.now() - timedelta(days=days_to_show)
    hist_df = hist_df[hist_df['date'] >= cutoff]
    
    # ========================================================================
    # KEY METRICS
    # ========================================================================
    
    st.header("📈 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        latest_count = hist_df.iloc[-1]['dsa_count'] if len(hist_df) > 0 else 0
        prev_count = hist_df.iloc[-2]['dsa_count'] if len(hist_df) > 1 else latest_count
        delta = latest_count - prev_count
        st.metric(
            "Latest DSA Violations",
            f"{latest_count:,.0f}",
            delta=f"{delta:+,.0f}",
            delta_color="inverse"
        )
    
    with col2:
        avg_7d = hist_df.tail(7)['dsa_count'].mean() if len(hist_df) >= 7 else 0
        st.metric("7-Day Average", f"{avg_7d:,.0f}")
    
    with col3:
        avg_sentiment = hist_df['sentiment_mean'].mean() if 'sentiment_mean' in hist_df.columns else 0
        st.metric(
            "Avg News Sentiment",
            f"{avg_sentiment:.3f}",
            delta="Neutral" if abs(avg_sentiment) < 0.1 else ("Positive" if avg_sentiment > 0 else "Negative")
        )
    
    with col4:
        if not pred_df.empty and 'actual' in pred_df.columns:
            evaluated = pred_df[pred_df['actual'].notna()]
            if len(evaluated) > 0:
                mape = evaluated['pct_error'].mean() if 'pct_error' in evaluated.columns else 0
                st.metric("Prediction MAPE", f"{mape:.1f}%")
            else:
                st.metric("Prediction MAPE", "N/A")
        else:
            st.metric("Prediction MAPE", "N/A")
    
    # ========================================================================
    # HISTORICAL TREND
    # ========================================================================
    
    st.header("📉 Historical DSA Violations")
    
    fig = go.Figure()
    
    # Daily counts
    fig.add_trace(go.Scatter(
        x=hist_df['date'],
        y=hist_df['dsa_count'],
        mode='lines',
        name='Daily Count',
        line=dict(color='#3498db', width=1),
        fill='tozeroy',
        fillcolor='rgba(52, 152, 219, 0.2)'
    ))
    
    # 7-day rolling average
    rolling_avg = hist_df['dsa_count'].rolling(7).mean()
    fig.add_trace(go.Scatter(
        x=hist_df['date'],
        y=rolling_avg,
        mode='lines',
        name='7-Day Average',
        line=dict(color='#e74c3c', width=2, dash='dash')
    ))
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="DSA Violation Count",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # ========================================================================
    # PREDICTIONS
    # ========================================================================
    
    st.header("🔮 Predictions")
    
    if not pred_df.empty:
        # Get future predictions (not yet evaluated)
        future_preds = pred_df[pred_df['actual'].isna()].head(7)
        
        if len(future_preds) > 0:
            st.subheader("Upcoming Forecasts")
            
            cols = st.columns(min(7, len(future_preds)))
            for i, (_, row) in enumerate(future_preds.iterrows()):
                with cols[i]:
                    date_str = row['target_date'].strftime('%b %d')
                    day_name = row['target_date'].strftime('%a')
                    st.metric(
                        f"{day_name} {date_str}",
                        f"{row['predicted']:,.0f}"
                    )
        
        # Prediction accuracy
        evaluated = pred_df[pred_df['actual'].notna()]
        
        if len(evaluated) > 0:
            st.subheader("Prediction Accuracy")
            
            fig2 = go.Figure()
            
            fig2.add_trace(go.Scatter(
                x=evaluated['target_date'],
                y=evaluated['actual'],
                mode='lines+markers',
                name='Actual',
                line=dict(color='#2ecc71', width=2)
            ))
            
            fig2.add_trace(go.Scatter(
                x=evaluated['target_date'],
                y=evaluated['predicted'],
                mode='lines+markers',
                name='Predicted',
                line=dict(color='#e74c3c', width=2, dash='dash')
            ))
            
            fig2.update_layout(
                xaxis_title="Date",
                yaxis_title="DSA Count",
                hovermode='x unified',
                height=350
            )
            
            st.plotly_chart(fig2, width='stretch')
    else:
        st.info("No predictions available. Run `make infer` to generate forecasts.")
    
    # ========================================================================
    # NEWS SENTIMENT
    # ========================================================================
    
    if 'sentiment_mean' in hist_df.columns:
        st.header("📰 News Sentiment")
        
        fig3 = go.Figure()
        
        colors = ['#27ae60' if s > 0 else '#e74c3c' for s in hist_df['sentiment_mean']]
        
        fig3.add_trace(go.Bar(
            x=hist_df['date'],
            y=hist_df['sentiment_mean'],
            marker_color=colors,
            name='Sentiment'
        ))
        
        fig3.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig3.update_layout(
            xaxis_title="Date",
            yaxis_title="Average Sentiment (-1 to 1)",
            height=300
        )
        
        st.plotly_chart(fig3, width='stretch')
    
    # ========================================================================
    # DATA TABLE
    # ========================================================================
    
    with st.expander("📋 View Raw Data"):
        st.dataframe(
            hist_df.tail(30).sort_values('date', ascending=False),
            width='stretch'
        )
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.markdown("---")
    st.markdown(
        f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
        f"Data range: {hist_df['date'].min().strftime('%Y-%m-%d')} to {hist_df['date'].max().strftime('%Y-%m-%d')}*"
    )


if __name__ == "__main__":
    main()
