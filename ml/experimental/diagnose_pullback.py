
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ml.data_pipeline import calculate_features
from data_processor import BinanceDataProcessor

def diagnose():
    symbol = 'ADAUSDT' # Known to have signals in backtest
    interval = '1d'
    days = 120
    
    print(f"Diagnosing {symbol} ({interval}, {days} days)...")
    
    # 1. Load Data
    ohlcv_dir = Path(__file__).parent.parent.parent / 'data' / 'ohlcv'
    file_path = ohlcv_dir / f"{symbol}_USDT.parquet"
    
    if not file_path.exists():
        print("File not found!")
        return
        
    df = pd.read_parquet(file_path)
    print(f"Loaded {len(df)} rows.")
    
    # Rename/Sort
    if 'open_time' in df.columns: df = df.rename(columns={'open_time': 'timestamp'})
    if df['timestamp'].dtype == 'int64': df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    else: df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"Date Range Full: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
    
    # 2. Filter Date
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days+60) # Generous buffer
    if df['timestamp'].dt.tz is None: df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    
    df_filtered = df[df['timestamp'] >= cutoff]
    print(f"Filtered Rows: {len(df_filtered)}")
    if len(df_filtered) == 0:
        print("Filtered result empty!")
        return

    # 3. Resample
    df_filtered = df_filtered.set_index('timestamp')
    df_res = df_filtered.resample(interval).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna().reset_index()
    print(f"Resampled to {interval}: {len(df_res)} rows.")
    
    # 4. Calculate Features
    print("Calculating features...")
    try:
        df_feat = calculate_features(df_res.copy())
    except Exception as e:
        print(f"Feature calc failed: {e}")
        return
        
    print("Columns:", df_feat.columns[:10])
    
    # 5. Check Splits
    if 'macd_cross_up' not in df_feat.columns:
        print("macd_cross_up MISSING!")
    else:
        cross_up = df_feat['macd_cross_up'].sum()
        cross_down = df_feat['macd_cross_down'].sum()
        print(f"macd_cross_up count: {cross_up}")
        print(f"macd_cross_down count: {cross_down}")
        
        if cross_up > 0:
            print("Sample Cross Up Indices:", df_feat[df_feat['macd_cross_up'] == 1].index.tolist())

if __name__ == "__main__":
    diagnose()
