
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time

# Fix paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.backtest_timeframes import TimeframeBacktester, PROCESSED_DIR, BacktestConfig

def debug_high_confidence():
    start_time = time.time()
    
    config = BacktestConfig(initial_capital=10000)
    config.start_date = '2020-01-01'
    config.end_date = '2026-02-01'
    config.max_bars = 40
    
    print("\n" + "="*80)
    print("DEBUGGING HIGH CONFIDENCE (0.8-1.0) PERFORMANCE")
    print("="*80)
    
    timeframe = '8h'
    
    # --- STEP 1: LOAD DATA ---
    print(f"[{time.time()-start_time:.1f}s] Loading data...")
    data_path = PROCESSED_DIR / f'features_{timeframe}_full.parquet'
    df = pd.read_parquet(data_path)
    
    # Filter by date
    df = df.sort_values(['symbol', 'timestamp'])
    ref_ts = df['timestamp'].iloc[0]
    
    def make_compatible(ts, ref):
        if ref.tzinfo is not None and ts.tzinfo is None: return ts.tz_localize('UTC')
        if ref.tzinfo is None and ts.tzinfo is not None: return ts.tz_localize(None)
        return ts

    df = df[(df['timestamp'] >= make_compatible(pd.Timestamp(config.start_date), ref_ts)) & 
            (df['timestamp'] <= make_compatible(pd.Timestamp(config.end_date), ref_ts))]

    # --- STEP 2: LOAD MODELS ---
    bt = TimeframeBacktester(timeframe, config)
    
    # --- STEP 3: PREDICT ALL SIGNALS ---
    print(f"[{time.time()-start_time:.1f}s] Detecting and predicting signals...")
    crossover_mask = (df['macd_cross_up'] == 1) | (df['macd_cross_down'] == 1)
    df_signals = df[crossover_mask].copy()
    
    # Batch predict confidence
    X_batch = df_signals.fillna(0).replace([np.inf, -np.inf], 0)
    
    def get_preds(features, model, scaler):
        X = pd.DataFrame(index=X_batch.index)
        for f in features: X[f] = X_batch[f] if f in X_batch.columns else 0
        X_s = scaler.transform(X) if scaler else X
        return model.predict_proba(X_s)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_s)

    df_signals['confidence'] = get_preds(bt.entry_features, bt.entry_model, bt.entry_scaler)
    
    # Filter for interests: 0.7-0.8 vs 0.8-1.0
    hi_conf = df_signals[df_signals['confidence'] >= 0.7].copy()
    hi_conf['bracket'] = hi_conf['confidence'].apply(lambda c: '0.8-1.0' if c >= 0.8 else '0.7-0.8')
    
    print(f"[{time.time()-start_time:.1f}s] Found {len(hi_conf)} signals in 0.7+ range.")

    # --- STEP 4: ANALYZE MFE / MAE ---
    print(f"[{time.time()-start_time:.1f}s] Calculating MFE/MAE and Clustering...")
    df_by_sym = {s: g for s, g in df.groupby('symbol')}
    
    results = []
    for idx, row in hi_conf.iterrows():
        symbol = row['symbol']
        group = df_by_sym[symbol]
        pos = group.index.get_loc(idx)
        
        if pos >= len(group) - config.max_bars: continue
        
        future = group.iloc[pos+1 : pos+1+config.max_bars]
        entry_price = row['close']
        is_long = row['macd_cross_up'] == 1
        
        if is_long:
            mfe = (future['high'].max() - entry_price) / entry_price
            mae = (future['low'].min() - entry_price) / entry_price
        else:
            mfe = (entry_price - future['low'].min()) / entry_price
            mae = (entry_price - future['high'].max()) / entry_price
            
        results.append({
            'bracket': row['bracket'],
            'symbol': symbol,
            'timestamp': row['timestamp'],
            'mfe': mfe * 100,
            'mae': mae * 100,
            'hour': row['timestamp'].hour,
            'confidence': row['confidence']
        })
        
    df_res = pd.DataFrame(results)
    
    # --- REPORTING ---
    print("\n" + "="*50)
    print("COMPARISON: 0.7-0.8 VS 0.8-1.0")
    print("="*50)
    
    # 1. Win Rate & MFE/MAE
    summary = df_res.groupby('bracket').agg({
        'confidence': 'count',
        'mfe': ['mean', 'median'],
        'mae': ['mean', 'median']
    })
    print("\n[Trade Metrics]")
    print(summary)
    
    # 2. Symbol Clustering
    print("\n[Symbol Clustering - Top 5 for 0.8-1.0]")
    print(df_res[df_res['bracket'] == '0.8-1.0']['symbol'].value_counts().head(5))
    
    # 3. Time Clustering (Monthly)
    df_res['month'] = df_res['timestamp'].dt.to_period('M')
    print("\n[Time Clustering - Monthly Counts]")
    print(df_res.groupby(['month', 'bracket']).size().unstack().fillna(0))
    
    # 4. Session Analysis
    print("\n[Session Analysis - Mean confidence by Hour (0.7+)]")
    print(df_res.groupby('hour')['confidence'].mean().sort_values(ascending=False).head(5))

    # 5. Feature Correlation (Pseudo-SHAP using correlation)
    print(f"\n[{time.time()-start_time:.1f}s] Running Correlation Analysis...")
    feat_cols = [c for c in bt.entry_features if c in hi_conf.columns]
    corrs = hi_conf[feat_cols + ['confidence']].corr()['confidence'].sort_values(ascending=False)
    print("\n[Top 5 Positive Correlates with Confidence]")
    print(corrs.iloc[1:6])
    print("\n[Top 5 Negative Correlates with Confidence]")
    print(corrs.tail(5))

if __name__ == "__main__":
    debug_high_confidence()
