import pandas as pd
import numpy as np
from pathlib import Path
import os

def check_simple_momentum():
    files = list(Path('data/processed/symbols_v3').glob('*USDT.parquet'))
    files.sort(key=lambda x: os.path.getsize(x), reverse=True)
    limit = 50
    
    dfs = []
    print("Loading Baseline CSM Data...")
    for f in files[:limit]:
        df = pd.read_parquet(f)
        df['symbol'] = f.stem
        # returns over last 24h
        df['ret_24h'] = df['close'] / df['close'].shift(24) - 1
        df['target_24h'] = df['close'].shift(-24) / df['close'] - 1
        dfs.append(df)
        
    data = pd.concat(dfs, ignore_index=True)
    data = data.dropna(subset=['ret_24h', 'target_24h'])
    
    # Let's test pure Cross Sectional Momentum (CSM)
    timestamps = np.sort(data['timestamp'].unique())
    
    rebalance_freq = 24
    equity = 100.0
    
    returns = []
    
    for i in range(0, len(timestamps) - rebalance_freq, rebalance_freq):
        ts = timestamps[i]
        chunk = data[data['timestamp'] == ts].copy()
        
        if len(chunk) < limit * 0.8:
            continue
            
        # simple strategy: Momentum (Trend following)
        # Buy highest past 24h return, Short lowest
        chunk = chunk.sort_values('ret_24h', ascending=False)
        top_k = 5
        longs = chunk.head(top_k)
        shorts = chunk.tail(top_k)
        
        long_return = longs['target_24h'].mean()
        short_return = shorts['target_24h'].mean()
        
        # We shorted the losers, so negative return is good
        strat_return = 0.5 * long_return - 0.5 * short_return
        
        fees = 0.0005 * 2 # 0.1% round trip
        strat_return -= fees
        
        equity *= (1 + strat_return)
        returns.append(strat_return)
        
    print(f"Trend CSM - Equity: ${equity:.2f}")

    equity = 100.0
    for i in range(0, len(timestamps) - rebalance_freq, rebalance_freq):
        ts = timestamps[i]
        chunk = data[data['timestamp'] == ts].copy()
        
        if len(chunk) < limit * 0.8:
            continue
            
        # Mean Reversion: Buy lowest past 24h return, Short highest past 24h return
        chunk = chunk.sort_values('ret_24h', ascending=True)
        top_k = 5
        longs = chunk.head(top_k)
        shorts = chunk.tail(top_k)
        
        long_return = longs['target_24h'].mean()
        short_return = shorts['target_24h'].mean()
        
        strat_return = 0.5 * long_return - 0.5 * short_return
        strat_return -= 0.001
        
        equity *= (1 + strat_return)
    
    print(f"Mean Rev CSM - Equity: ${equity:.2f}")

if __name__ == "__main__":
    check_simple_momentum()