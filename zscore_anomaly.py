import pandas as pd
import numpy as np
from pathlib import Path
import os

def check_zscore_anomaly():
    files = list(Path('data/processed/symbols_v3').glob('*USDT.parquet'))
    files.sort(key=lambda x: os.path.getsize(x), reverse=True)
    limit = 50
    
    dfs = []
    print("Loading Z-Score Data...")
    for f in files[:limit]:
        df = pd.read_parquet(f)
        df['symbol'] = f.stem
        # returns over last 1h, 4h
        df['ret_1h'] = df['close'] / df['close'].shift(1) - 1
        df['ret_4h'] = df['close'] / df['close'].shift(4) - 1
        
        # Vol Z-score
        df['vol_24h_mean'] = df['volume'].shift(1).rolling(24).mean()
        df['vol_24h_std'] = df['volume'].shift(1).rolling(24).std()
        df['vol_zscore'] = (df['volume'] - df['vol_24h_mean']) / (df['vol_24h_std'] + 1e-9)
        
        # Target
        df['target_12h'] = df['close'].shift(-12) / df['close'] - 1
        df['target_24h'] = df['close'].shift(-24) / df['close'] - 1
        dfs.append(df)
        
    data = pd.concat(dfs, ignore_index=True)
    data = data.dropna(subset=['vol_zscore', 'target_12h', 'target_24h'])
    
    events = data[data['vol_zscore'] > 3.0].copy()
    
    print(f"Found {len(events)} extreme volume events (Z > 3) out of {len(data)} total rows.")
    
    # Are these events usually trend continuation or mean reversion?
    # Separate into pump events AND dump events
    pump_events = events[events['ret_1h'] > 0]
    dump_events = events[events['ret_1h'] < 0]
    
    print(f"\n--- PUMPS (Volume Z > 3 AND ret_1h > 0) ---")
    print(f"Count: {len(pump_events)}")
    print(f"Avg 12H Forward Return: {pump_events['target_12h'].mean() * 100:.3f}%")
    print(f"Avg 24H Forward Return: {pump_events['target_24h'].mean() * 100:.3f}%")
    print(f"12H Win Rate: {(pump_events['target_12h'] > 0).mean() * 100:.1f}%")
    
    print(f"\n--- DUMPS (Volume Z > 3 AND ret_1h < 0) ---")
    print(f"Count: {len(dump_events)}")
    print(f"Avg 12H Forward Return: {dump_events['target_12h'].mean() * 100:.3f}%")
    print(f"Avg 24H Forward Return: {dump_events['target_24h'].mean() * 100:.3f}%")
    print(f"12H Win Rate: {(dump_events['target_12h'] > 0).mean() * 100:.1f}%")

if __name__ == "__main__":
    check_zscore_anomaly()