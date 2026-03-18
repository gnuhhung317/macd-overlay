import pandas as pd
import numpy as np
import sys
from pathlib import Path

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1  + rs))

def debug():
    parquet_path = r'd:\Code\Projects\self-projects\macd-overlay - Copy\data\processed\features_1h_btc_context.parquet'
    gt_df = pd.read_parquet(parquet_path)
    
    # Check index 500 to be safe
    idx = 500
    row = gt_df.iloc[idx]
    ts = row['timestamp']
    symbol = row['symbol']
    
    print(f"--- Symbol: {symbol} at {ts} ---")
    
    # Load 1000 candles for this symbol to have padding
    raw_ohlcv = gt_df[gt_df['symbol'] == symbol].copy().sort_values('timestamp')
    # Find index of ts in raw_ohlcv
    raw_idx = raw_ohlcv[raw_ohlcv['timestamp'] == ts].index[0]
    
    # Local subset to calculate
    # We need enough history
    df = raw_ohlcv.copy()
    
    # Test price_vs_sma_30
    df['sma_30'] = df['close'].rolling(30).mean()
    # Varying versions
    v1 = df['close'] / df['sma_30']
    v2 = df['close'].shift(1) / df['sma_30']
    v3 = (df['close'] / df['sma_30']).shift(1)
    
    # Test dist_to_ema_21_pct
    df['ema_21'] = df['close'].ewm(span=21, adjust=True).mean()
    d1 = (df['close'] - df['ema_21']) / df['close']
    d2 = (df['close'].shift(1) - df['ema_21']) / df['close'].shift(1)
    d3 = ((df['close'] - df['ema_21']) / df['close']).shift(1)
    
    # Test rsi_slope
    df['rsi'] = calculate_rsi(df['close'], 14)
    r1 = df['rsi'].diff(3)
    r2 = df['rsi'].diff(3) / 3
    r3 = df['rsi'].diff(1)
    
    print(f"GT price_vs_sma_30:    {row['price_vs_sma_30']:.6f}")
    print(f"v1 (close/sma):        {v1.loc[raw_idx]:.6f}")
    print(f"v2 (close.t-1/sma.t):  {v2.loc[raw_idx]:.6f}")
    print(f"v3 (close.t-1/sma.t-1):{v3.loc[raw_idx]:.6f}")
    
    print(f"\nGT dist_to_ema_21_pct: {row['dist_to_ema_21_pct']:.6f}")
    print(f"d1 (close-ema)/close:  {d1.loc[raw_idx]:.6f}")
    print(f"d2 (cl.t-1-ema.t)/cl.t-1: {d2.loc[raw_idx]:.6f}")
    print(f"d3 (cl.t-1-ema.t-1)/cl.t-1: {d3.loc[raw_idx]:.6f}")
    
    print(f"\nGT rsi_slope:         {row['rsi_slope']:.6f}")
    print(f"r1 (diff 3):           {r1.loc[raw_idx]:.6f}")
    print(f"r2 (diff 3 / 3):       {r2.loc[raw_idx]:.6f}")
    print(f"r3 (diff 1):           {r3.loc[raw_idx]:.6f}")

if __name__ == "__main__":
    debug()
