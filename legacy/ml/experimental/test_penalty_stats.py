
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time

# Fix paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.backtest_timeframes import TimeframeBacktester, PROCESSED_DIR, BacktestConfig

def test_penalty_strategy():
    start_time = time.time()
    
    config = BacktestConfig(initial_capital=10000)
    config.start_date = '2025-01-01'
    config.end_date = '2026-02-01'
    config.max_bars = 40
    
    print("\n" + "="*80)
    print("BACKTESTING CONFIDENCE PENALTY STRATEGY (8H)")
    print("="*80)
    
    timeframe = '8h'
    bt = TimeframeBacktester(timeframe, config)
    
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

    # --- STEP 2: SIGNAL DETECTION & BATCH PREDICTION ---
    print(f"[{time.time()-start_time:.1f}s] Predicting original signals...")
    crossover_mask = (df['macd_cross_up'] == 1) | (df['macd_cross_down'] == 1)
    df_signals = df[crossover_mask].copy()
    
    # Batch predict
    def get_preds(features, model, scaler, X_batch):
        X = pd.DataFrame(index=X_batch.index)
        for f in features: X[f] = X_batch[f] if f in X_batch.columns else 0
        X_s = scaler.transform(X) if scaler else X
        if hasattr(model, 'predict_proba'): return model.predict_proba(X_s)[:, 1]
        return model.predict(X_s)

    X_full = df_signals.fillna(0).replace([np.inf, -np.inf], 0)
    df_signals['conf_orig'] = get_preds(bt.entry_features, bt.entry_model, bt.entry_scaler, X_full)
    df_signals['sl_pct'] = np.clip(get_preds(bt.sl_features, bt.sl_model, bt.sl_scaler, X_full), 0.005, 0.15)
    tp_raw = get_preds(bt.tp_features, bt.tp_model, bt.tp_scaler, X_full)
    df_signals['tp_pct'] = np.clip(tp_raw * df_signals['sl_pct'] if getattr(bt, 'tp_predict_rr', False) else tp_raw, 0.01, 0.30)

    # --- STEP 3: APPLY PENALTY ---
    print(f"[{time.time()-start_time:.1f}s] Applying Confidence Penalties...")
    
    # Pre-calculate BB Width threshold (top 10%)
    bb_threshold = df['bb_width'].quantile(0.9)
    print(f"  BB Width 90th percentile: {bb_threshold:.4f}")
    
    def adjust_confidence(row):
        c = row['conf_orig']
        # 1. ADX > 40 Penalty
        if row.get('adx', 0) > 40:
            c *= 0.8
        # 2. Price < SMA200 by > 5%
        if row.get('price_to_sma_200', 1.0) < 0.95:
            c *= 0.85
        # 3. BB Width > Top 10%
        if row.get('bb_width', 0) > bb_threshold:
            c *= 0.9
        return c

    df_signals['conf_adj'] = df_signals.apply(adjust_confidence, axis=1)

    # --- STEP 4: SIMULATION ---
    print(f"[{time.time()-start_time:.1f}s] Simulating trades...")
    df_by_sym = {s: g for s, g in df.groupby('symbol')}
    valid_indices = []
    
    # Find valid simulation indices
    for s, g in df_by_sym.items():
        sig_idx_in_sym = df_signals[df_signals['symbol'] == s].index
        g_len = len(g)
        for idx in sig_idx_in_sym:
            pos = g.index.get_loc(idx)
            if pos < g_len - config.max_bars:
                valid_indices.append(idx)
    
    results = []
    for i, idx in enumerate(valid_indices):
        if i % 2000 == 0 and i > 0: print(f"  Simulated {i}/{len(valid_indices)} trades...")
        
        row = df_signals.loc[idx]
        group = df_by_sym[row['symbol']]
        pos = group.index.get_loc(idx)
        future = group.iloc[pos+1 : pos+1+config.max_bars]
        
        entry_price = row['close']
        direction = 'LONG' if row['macd_cross_up'] == 1 else 'SHORT'
        
        # Simulate ONE trade (it's the same regardless of confidence, only confidence changes)
        trade = bt.simulate_trade(row, future, entry_price, row['sl_pct'], row['tp_pct'], direction, 1000)
        
        results.append({
            'conf_orig': row['conf_orig'],
            'conf_adj': row['conf_adj'],
            'win': 1 if trade.pnl > 0 else 0,
            'pnl': trade.pnl_pct * 100
        })

    df_res = pd.DataFrame(results)

    # --- STEP 5: COMPARISON REPORT ---
    def get_bracket(c):
        if 0.70 <= c < 0.75: return '0.70-0.75'
        if 0.75 <= c < 0.80: return '0.75-0.80'
        if 0.80 <= c <= 1.00: return '0.80-1.00'
        return ' < 0.70'

    df_res['bracket_orig'] = df_res['conf_orig'].apply(get_bracket)
    df_res['bracket_adj'] = df_res['conf_adj'].apply(get_bracket)
    
    print("\n" + "="*60)
    print("COMPARISON: ORIGINAL VS ADJUSTED BRACKETS")
    print("="*60)
    
    orig_stats = df_res.groupby('bracket_orig').agg({'win': ['count', 'mean']})
    adj_stats = df_res.groupby('bracket_adj').agg({'win': ['count', 'mean']})
    
    print("\n[ORIGINAL BINS]")
    print(orig_stats)
    
    print("\n[ADJUSTED BINS (After Penalty)]")
    print(adj_stats)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    hi_orig = df_res[df_res['conf_orig'] >= 0.8]
    hi_adj = df_res[df_res['conf_adj'] >= 0.8]
    
    print(f"Original 0.8-1.0: Count={len(hi_orig)}, WR={hi_orig['win'].mean():.1%}")
    print(f"Adjusted 0.8-1.0: Count={len(hi_adj)}, WR={hi_adj['win'].mean():.1%}")
    
    if not hi_adj.empty and len(hi_adj) < len(hi_orig):
        print(f"\nSUCCESS: Penalty correctly pruned {len(hi_orig) - len(hi_adj)} risky signals from the top bracket.")
    elif hi_adj.empty:
        print("\nNOTE: All 0.8+ signals were shifted to lower brackets.")

if __name__ == "__main__":
    test_penalty_strategy()
