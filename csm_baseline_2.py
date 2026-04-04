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
    
    timestamps = np.sort(data['timestamp'].unique())
    rebalance_freq = 24
    
    returns_gross_trend = []
    returns_gross_mr = []
    
    for i in range(0, len(timestamps) - rebalance_freq, rebalance_freq):
        ts = timestamps[i]
        chunk = data[data['timestamp'] == ts].copy()
        
        if len(chunk) < limit * 0.8:
            continue
            
        # TREND
        chunk_trend = chunk.sort_values('ret_24h', ascending=False)
        top_k = 5
        longs = chunk_trend.head(top_k)
        shorts = chunk_trend.tail(top_k)
        
        long_return = longs['target_24h'].mean()
        short_return = shorts['target_24h'].mean()
        
        strat_return = 0.5 * long_return - 0.5 * short_return
        returns_gross_trend.append(strat_return)
        
        # MR
        chunk_mr = chunk.sort_values('ret_24h', ascending=True)
        longs = chunk_mr.head(top_k)
        shorts = chunk_mr.tail(top_k)
        
        long_return = longs['target_24h'].mean()
        short_return = shorts['target_24h'].mean()
        
        strat_return = 0.5 * long_return - 0.5 * short_return
        returns_gross_mr.append(strat_return)

    print(f"Trend Gross Ret/Day: {np.mean(returns_gross_trend)*100*24/rebalance_freq:.3f}%")
    print(f"Mean Rev Gross Ret/Day: {np.mean(returns_gross_mr)*100*24/rebalance_freq:.3f}%")

if __name__ == "__main__":
    check_simple_momentum()