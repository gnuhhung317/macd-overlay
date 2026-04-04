import os
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

def load_data(limit=20):
    files = list(Path('data/processed/symbols_v3').glob('*USDT.parquet'))
    # Sort files by size to get the ones with most history/liquidity
    files.sort(key=lambda x: os.path.getsize(x), reverse=True)
    
    dfs = []
    for f in files[:limit]:
        df = pd.read_parquet(f)
        df['symbol'] = f.stem
        # Compute dynamic features as requested: "Z-Score khối lượng, độ lệch"
        df['vol_24h_mean'] = df['volume'].rolling(24).mean()
        df['vol_24h_std'] = df['volume'].rolling(24).std()
        df['vol_zscore'] = (df['volume'] - df['vol_24h_mean']) / (df['vol_24h_std'] + 1e-9)
        
        # Target: Return of the NEXT 12 hours!
        df['target_12h'] = df['close'].shift(-12) / df['close'] - 1
        
        # Momentum 12H, 24H Z-score
        df['ret_24h'] = df['close'] / df['close'].shift(24) - 1
        dfs.append(df)
        
    data = pd.concat(dfs, ignore_index=True)
    return data

def run_cross_sectional_research():
    print("Loading data...")
    df = load_data(limit=50) # Use top 50 biggest coins
    df = df.dropna(subset=['target_12h', 'vol_zscore', 'ret_24h', 'adx', 'rsi_14'])
    
    # Sort chronologically to prevent leakage
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    features = [
        'vol_zscore', 'ret_24h', 'adx', 'rsi_14', 'macd_slope', 
        'volume_compression', 'rs_vs_btc', 'dist_to_ema_21_pct'
    ]
    # some features might not exist perfectly depending on dataset, let's verify
    existing_cols = set(df.columns)
    use_features = [f for f in features if f in existing_cols]
    
    # Time splits
    train_end = pd.to_datetime('2024-06-01')
    
    # Make sure we only use valid data
    train_mask = df['timestamp'] < train_end
    test_mask = df['timestamp'] >= train_end
    
    print(f"Train size: {train_mask.sum()}, Test size: {test_mask.sum()}")
    
    X_train, y_train = df[train_mask][use_features], df[train_mask]['target_12h']
    X_test, y_test = df[test_mask][use_features], df[test_mask]['target_12h']
    
    print(f"Training XGBoost with features: {use_features}...")
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    
    print("Predicting...")
    df.loc[test_mask, 'pred'] = model.predict(X_test)
    
    # Eval Cross-Sectional Strategy
    test_df = df[test_mask].copy()
    test_df = test_df.dropna(subset=['pred'])
    
    results = []
    
    # We step through time and construct a portfolio
    timestamps = test_df['timestamp'].unique()
    timestamps = np.sort(timestamps)
    
    equity = 10000.0
    eq_curve = []
    
    # We rebalance every 12 hours!
    rebalance_freq = 12 
    
    # Map to fast lookup
    for i in range(0, len(timestamps) - rebalance_freq, rebalance_freq):
        ts = timestamps[i]
        chunk = test_df[test_df['timestamp'] == ts]
        if len(chunk) < 10:
            continue # not enough coins to rank
            
        # Rank by model prediction
        chunk = chunk.sort_values('pred', ascending=False)
        
        # BUY Top 3, SHORT Bottom 3
        top_k = 3
        longs = chunk.head(top_k)
        shorts = chunk.tail(top_k)
        
        # Calculate exactly what happens 12 hours later
        long_return = longs['target_12h'].mean()
        short_return = shorts['target_12h'].mean()
        
        # Strategy return (Beta neutral pure Alpha)
        # We shorted the bottom, so their negative return is our positive PnL
        # Longs: + return, Shorts: - return
        strat_return = 0.5 * long_return - 0.5 * short_return 
        
        # apply fees (e.g., 0.05% per side per trade)
        fees = 0.0005 * 2 # open and close
        strat_return -= fees
        
        equity *= (1 + strat_return)
        eq_curve.append({'time': ts, 'equity': equity, 'long_ret': long_return, 'short_ret': short_return})

    res_df = pd.DataFrame(eq_curve)
    res_df.to_csv('cs_momentum_results.csv', index=False)
    
    total_return = (equity / 10000.0 - 1) * 100
    print(f"FINAL EQUITY: ${equity:.2f} | RETURN: {total_return:.2f}%")
    
if __name__ == "__main__":
    run_cross_sectional_research()
