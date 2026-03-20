import pandas as pd
import numpy as np
import sys
from pathlib import Path

def calculate_rsi_wilder(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    # Wilder uses alpha=1/N or span=2N-1
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1  + rs))

def debug():
    parquet_path = r'd:\Code\Projects\self-projects\macd-overlay - Copy\data\processed\features_1h_btc_context.parquet'
    gt_df = pd.read_parquet(parquet_path)
    
    idx = 500
    row = gt_df.iloc[idx]
    ts = row['timestamp']
    symbol = row['symbol']
    
    print(f"--- Symbol: {symbol} at {ts} ---")
    
    raw_ohlcv = gt_df[gt_df['symbol'] == symbol].copy().sort_values('timestamp')
    raw_idx = raw_ohlcv[raw_ohlcv['timestamp'] == ts].index[0]
    
    df = raw_ohlcv.copy()
    
    # Test Wilder RSI
    df['rsi_wilder'] = calculate_rsi_wilder(df['close'], 14)
    r_val = df['rsi_wilder'].loc[raw_idx]
    r_val_shifted = df['rsi_wilder'].shift(1).loc[raw_idx]
    
    print(f"GT rsi_14:          {row['rsi_14']:.6f}")
    print(f"Wilder RSI (raw):   {r_val:.6f}")
    print(f"Wilder RSI (shift): {r_val_shifted:.6f}")
    
    # Test rsi_slope with Wilder
    df['rsi_slope_w'] = df['rsi_wilder'].diff(3)
    rs_val = df['rsi_slope_w'].loc[raw_idx]
    rs_val_shifted = df['rsi_slope_w'].shift(1).loc[raw_idx]
    
    print(f"\nGT rsi_slope:       {row['rsi_slope']:.6f}")
    print(f"Wilder Slope(raw):  {rs_val:.6f}")
    print(f"Wilder Slope(shift):{rs_val_shifted:.6f}")

if __name__ == "__main__":
    debug()
