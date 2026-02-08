
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time
import argparse

# Fix paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.backtest_timeframes import TimeframeBacktester, PROCESSED_DIR, BacktestConfig

def test_refined_strategy(timeframe='8h'):
    start_time = time.time()
    
    config = BacktestConfig(initial_capital=10000)
    config.start_date = '2020-01-01'
    config.end_date = '2026-02-01'
    config.max_bars = 40
    
    print("\n" + "="*80)
    print(f"REFINED DIRECTIONAL STRATEGY BACKTEST ({timeframe.upper()}, 2020-2026)")
    print("="*80)
    
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
    print(f"[{time.time()-start_time:.1f}s] Predicting signals...")
    crossover_mask = (df['macd_cross_up'] == 1) | (df['macd_cross_down'] == 1)
    df_signals = df[crossover_mask].copy()
    
    def get_preds(features, model, scaler, X_batch):
        X = pd.DataFrame(index=X_batch.index)
        for f in features: X[f] = X_batch[f] if f in X_batch.columns else 0
        X_s = scaler.transform(X) if scaler else X
        if hasattr(model, 'predict_proba'): return model.predict_proba(X_s)[:, 1]
        return model.predict(X_s)

    X_full = df_signals.fillna(0).replace([np.inf, -np.inf], 0)
    df_signals['confidence'] = get_preds(bt.entry_features, bt.entry_model, bt.entry_scaler, X_full)
    df_signals['sl_pct'] = np.clip(get_preds(bt.sl_features, bt.sl_model, bt.sl_scaler, X_full), 0.005, 0.15)
    tp_raw = get_preds(bt.tp_features, bt.tp_model, bt.tp_scaler, X_full)
    df_signals['tp_pct'] = np.clip(tp_raw * df_signals['sl_pct'] if getattr(bt, 'tp_predict_rr', False) else tp_raw, 0.01, 0.30)

    # --- STEP 3: APPLY REFINED FILTERS ---
    print(f"[{time.time()-start_time:.1f}s] Applying Scientific Filters...")
    
    def pass_refined_filters(row):
        is_long = row['macd_cross_up'] == 1
        
        # Global Rule: ADX Fatigue
        if row.get('adx', 0) > 48:
            return False
        
        if is_long:
            # LONG Rules: Volume spike confirmation and SMA floor
            if row.get('volume_spike', 0) < 1.15: # Slightly more inclusive than 1.2
                return False
            if row.get('price_to_sma_100', 1.0) < 0.85:
                return False
        else:
            # SHORT Rules
            # RSI cooling check (avoid shorting absolute parabolic peaks)
            if row.get('rsi_7', 0) > 65:
                return False
                
        return True

    df_signals['pass_refined'] = df_signals.apply(pass_refined_filters, axis=1)

    # --- STEP 4: SIMULATION ---
    print(f"[{time.time()-start_time:.1f}s] Simulating trades...")
    df_by_sym = {s: g for s, g in df.groupby('symbol')}
    

    results = []
    # Test all signals > 0.65 to see the effect
    conf_threshold = 0.65
    print(f"Testing with confidence threshold: {conf_threshold}")
    df_test = df_signals[df_signals['confidence'] >= conf_threshold].copy()
    
    for idx, row in df_test.iterrows():
        symbol = row['symbol']
        group = df_by_sym[symbol]
        pos = group.index.get_loc(idx)
        
        if pos >= len(group) - config.max_bars: continue
        
        future = group.iloc[pos+1 : pos+1+config.max_bars]
        entry_price = row['close']
        direction = 'LONG' if row['macd_cross_up'] == 1 else 'SHORT'
        
        trade = bt.simulate_trade(row, future, entry_price, row['sl_pct'], row['tp_pct'], direction, 1000)
        
        results.append({
            'direction': direction,
            'confidence': row['confidence'],
            'pass_refined': row['pass_refined'],
            'win': 1 if trade.pnl > 0 else 0,
            'pnl': trade.pnl_pct * 100
        })

    df_res = pd.DataFrame(results)

    # --- STEP 5: COMPARISON REPORT ---
    print("\n" + "="*60)
    print(f"COMPARISON: ORIGINAL VS REFINED (Conf >= {conf_threshold})")
    print("="*60)
    
    orig_total = df_res.agg({'win': ['count', 'mean']})
    refined_total = df_res[df_res['pass_refined']].agg({'win': ['count', 'mean']})
    
    print("\n[Total Aggregated]")
    print(f"Original: Count={int(orig_total.loc['count', 'win'])}, WR={orig_total.loc['mean', 'win']:.1%}")
    print(f"Refined:  Count={int(refined_total.loc['count', 'win'])}, WR={refined_total.loc['mean', 'win']:.1%}")
    
    print("\n[By Direction]")
    for d in ['LONG', 'SHORT']:
        d_orig = df_res[df_res['direction'] == d].agg({'win': ['count', 'mean']})
        d_refined = df_res[(df_res['direction'] == d) & (df_res['pass_refined'])].agg({'win': ['count', 'mean']})
        print(f"{d:5} Original: Count={int(d_orig.loc['count', 'win']):4}, WR={d_orig.loc['mean', 'win']:.1%}")
        print(f"{d:5} Refined:  Count={int(d_refined.loc['count', 'win']):4}, WR={d_refined.loc['mean', 'win']:.1%}")

    print("\n" + "="*60)
    print("HIGH CONFIDENCE (0.8+) COMPARISON")
    print("="*60)
    hi_orig = df_res[df_res['confidence'] >= 0.8].agg({'win': ['count', 'mean']})
    hi_refined = df_res[(df_res['confidence'] >= 0.8) & (df_res['pass_refined'])].agg({'win': ['count', 'mean']})
    print(f"0.8+ Original: Count={int(hi_orig.loc['count', 'win']):4}, WR={hi_orig.loc['mean', 'win']:.1%}")
    print(f"0.8+ Refined:  Count={int(hi_refined.loc['count', 'win']):4}, WR={hi_refined.loc['mean', 'win']:.1%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", type=str, default="8h")
    args = parser.parse_args()
    
    test_refined_strategy(args.timeframe)
