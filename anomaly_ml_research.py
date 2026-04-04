import pandas as pd
import numpy as np
from pathlib import Path
import os
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def engineer_features(limit=20):
    files = list(Path('data/processed/symbols_v3').glob('*USDT.parquet'))
    files.sort(key=lambda x: os.path.getsize(x), reverse=True)
    
    dfs = []
    print("Loading and Engineering Data...")
    for f in files[:limit]:
        df = pd.read_parquet(f)
        df['symbol'] = f.stem
        
        # Core Anomaly Features
        df['vol_24h_mean'] = df['volume'].shift(1).rolling(24).mean()
        df['vol_24h_std'] = df['volume'].shift(1).rolling(24).std()
        df['vol_zscore'] = (df['volume'] - df['vol_24h_mean']) / (df['vol_24h_std'] + 1e-9)
        
        # Momentum & Volatility
        df['ret_1h'] = df['close'] / df['close'].shift(1) - 1
        df['ret_4h'] = df['close'] / df['close'].shift(4) - 1
        df['ret_24h'] = df['close'] / df['close'].shift(24) - 1
        
        # Relative Context (distance from EMAs)
        df['dist_ema_21'] = df['close'] / df['ema_21'].shift(1) - 1
        df['dist_ema_200'] = df['close'] / df['ema_200'].shift(1) - 1
        
        # Volatility context
        df['atr_pct'] = df['atr_14'].shift(1) / df['close'].shift(1)
        
        # Shifted targets (predicting 24H return forward)
        df['target_24h'] = df['close'].shift(-24) / df['close'] - 1
        
        dfs.append(df)
        
    data = pd.concat(dfs, ignore_index=True)
    return data

def run_ensemble_anomaly_engine():
    data = engineer_features(limit=40)
    
    # FILTER FOR ANOMALIES ONLY (Z-SCORE > 2.5) -> High Rhythm Exhaustion
    events = data[data['vol_zscore'] > 2.5].copy()
    
    features = [
        'vol_zscore', 'ret_1h', 'ret_4h', 'ret_24h', 
        'dist_ema_21', 'dist_ema_200', 'atr_pct', 
        'rsi_14', 'adx', 'macd_slope'
    ]
    
    events = events.dropna(subset=features + ['target_24h'])
    events = events.sort_values('timestamp').reset_index(drop=True)
    
    # Train / Test temporal split
    train_end = pd.to_datetime('2024-01-01')
    
    train_mask = events['timestamp'] < train_end
    val_mask = (events['timestamp'] >= train_end) & (events['timestamp'] < pd.to_datetime('2024-06-01'))
    test_mask = events['timestamp'] >= pd.to_datetime('2024-06-01')
    
    X_train, y_train = events[train_mask][features], events[train_mask]['target_24h']
    X_val, y_val = events[val_mask][features], events[val_mask]['target_24h']
    X_test, y_test = events[test_mask][features], events[test_mask]['target_24h']
    
    print(f"Anomaly Events - Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # Extremely heavy ensemble conceptually
    print("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.03, subsample=0.8, n_jobs=-1, random_state=42)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)
    
    print("Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=150, max_depth=8, min_samples_leaf=10, n_jobs=-1, random_state=42)
    rf_model.fit(X_train, y_train)
    
    print("Ensembling...")
    xgb_pred = xgb_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)
    
    # Blend 50/50
    final_pred = (xgb_pred + rf_pred) / 2
    
    test_events = events[test_mask].copy()
    test_events['pred'] = final_pred
    
    # Trading Logic: Only take trades where predicted return > 1.5% (High threshold Signal)
    long_signals = test_events[test_events['pred'] > 0.015]
    short_signals = test_events[test_events['pred'] < -0.015]
    
    # Calculate PNL
    fees = 0.001 # 0.1% round trip
    
    long_pnl = long_signals['target_24h'].mean() - fees
    short_pnl = -(short_signals['target_24h'].mean()) - fees
    
    print(f"\n--- ENSEMBLE TRADING RESULTS (OOS) ---")
    print(f"LONG TRADES: {len(long_signals)} | Expected Net Return per trade: {long_pnl * 100:.3f}% | Win Rate: {(long_signals['target_24h'] > 0).mean()*100:.1f}%")
    print(f"SHORT TRADES: {len(short_signals)} | Expected Net Return per trade: {short_pnl * 100:.3f}% | Win Rate: {(short_signals['target_24h'] < 0).mean()*100:.1f}%")

if __name__ == "__main__":
    run_ensemble_anomaly_engine()