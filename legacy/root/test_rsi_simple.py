import pandas as pd
import numpy as np
import sys
from pathlib import Path

def calculate_rsi_simple(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1  + rs))

def debug():
    parquet_path = r'd:\Code\Projects\self-projects\macd-overlay - Copy\data\processed\features_1h_btc_context.parquet'
    gt_df = pd.read_parquet(parquet_path)
    
    idx = 500
    row = gt_df.iloc[idx]
    ts = row['timestamp']
    symbol = row['symbol']
    
    raw_ohlcv = gt_df[gt_df['symbol'] == symbol].copy().sort_values('timestamp')
    raw_idx = raw_ohlcv[raw_ohlcv['timestamp'] == ts].index[0]
    
    df = raw_ohlcv.copy()
    
    df['rsi_simple'] = calculate_rsi_simple(df['close'], 14)
    df['rsi_slope_s'] = df['rsi_simple'].diff(3)
    
    print(f"--- Simple RSI at {ts} ---")
    print(f"GT rsi_14:          {row['rsi_14']:.6f}")
    print(f"Simple RSI (shift): {df['rsi_simple'].shift(1).loc[raw_idx]:.6f}")
    
    print(f"\nGT rsi_slope:       {row['rsi_slope']:.6f}")
    print(f"Simple Slope(shift):{df['rsi_slope_s'].shift(1).loc[raw_idx]:.6f}")

if __name__ == "__main__":
    debug()
