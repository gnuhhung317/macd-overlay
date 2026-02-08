
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time
import argparse

# Fix paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.backtest_timeframes import TimeframeBacktester, PROCESSED_DIR, BacktestConfig

def run_dynamic_strategy(timeframe='8h'):
    start_time = time.time()
    config = BacktestConfig(initial_capital=10000)
    config.start_date = '2020-01-01'
    config.end_date = '2026-02-01'
    config.max_bars = 40
    
    print("\n" + "="*80)
    print(f"DYNAMIC FILTERING STRATEGY ({timeframe.upper()})")
    print("="*80)
    
    bt = TimeframeBacktester(timeframe, config)
    
    # 1. LOAD & PREDICT
    print(f"[{time.time()-start_time:.1f}s] Loading and Predicting signals...")
    data_path = PROCESSED_DIR / f'features_{timeframe}_full.parquet'
    df = pd.read_parquet(data_path)
    
    # Date alignment
    ref_ts = df['timestamp'].iloc[0]
    def make_compatible(ts, ref):
        if ref.tzinfo is not None and ts.tzinfo is None: return ts.tz_localize('UTC')
        if ref.tzinfo is None and ts.tzinfo is not None: return ts.tz_localize(None)
        return ts
    df = df[(df['timestamp'] >= make_compatible(pd.Timestamp(config.start_date), ref_ts))]

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

    # 2. SCIENTIFIC ANALYSIS (WIN/LOSS DELTAS)
    # We use historical 0.65+ signals to find the filters
    df_test = df_signals[df_signals['confidence'] >= 0.65].copy()
    df_by_sym = {s: g for s, g in df.groupby('symbol')}
    
    print(f"[{time.time()-start_time:.1f}s] Running Win/Loss analysis to discover filters...")
    outcomes = []
    for idx, row in df_test.iterrows():
        symbol = row['symbol']
        group = df_by_sym[symbol]
        pos = group.index.get_loc(idx)
        if pos >= len(group) - config.max_bars: continue
        future = group.iloc[pos+1 : pos+1+config.max_bars]
        entry_price = row['close']
        direction = 'LONG' if row['macd_cross_up'] == 1 else 'SHORT'
        trade = bt.simulate_trade(row, future, entry_price, row['sl_pct'], row['tp_pct'], direction, 1000)
        outcomes.append({'idx': idx, 'is_win': 1 if trade.pnl > 0 else 0, 'direction': direction})
    
    df_outcomes = pd.DataFrame(outcomes).set_index('idx')
    df_test = df_test.join(df_outcomes)
    
    # 3. DYNAMIC RULE GENERATION
    dynamic_filters = {}
    for d in ['LONG', 'SHORT']:
        subset = df_test[df_test['direction'] == d]
        wins = subset[subset['is_win'] == 1]
        losses = subset[subset['is_win'] == 0]
        
        if wins.empty or losses.empty: continue
        
        feat_stats = []
        for feat in bt.entry_features:
            if feat not in subset.columns: continue
            w_mean, l_mean = wins[feat].mean(), losses[feat].mean()
            std = subset[feat].std()
            shift = (w_mean - l_mean) / std if std > 0 else 0
            feat_stats.append({'feat': feat, 'shift': shift, 'w_mean': w_mean, 'std': std})
        
        df_feat = pd.DataFrame(feat_stats)
        # Top 3 discriminatory features
        top_feats = df_feat.assign(abs_shift=df_feat['shift'].abs()).sort_values('abs_shift', ascending=False).head(3)
        dynamic_filters[d] = top_feats.to_dict('records')
        print(f"\n[{d}] Top Dynamic Filters:")
        for f in dynamic_filters[d]:
            print(f"  - {f['feat']}: shift={f['shift']:.2f} (Win Mean={f['w_mean']:.2f})")

    # 4. APPLY & BACKTEST (SCORING SYSTEM)
    def get_dynamic_score(row):
        d = 'LONG' if row['macd_cross_up'] == 1 else 'SHORT'
        if d not in dynamic_filters: return 1.0 # No filters
        
        score = 0
        total = len(dynamic_filters[d])
        for f in dynamic_filters[d]:
            val = row.get(f['feat'], 0)
            if f['shift'] > 0:
                if val >= f['w_mean'] - 0.2 * f['std']: score += 1
            else:
                if val <= f['w_mean'] + 0.2 * f['std']: score += 1
        return score / total if total > 0 else 1.0

    df_test['dynamic_score'] = df_test.apply(get_dynamic_score, axis=1)
    
    # 5. SAVE TRADE LOG FOR VISUALIZATION (using Balanced 0.6+ score)
    log_path = Path(f"ml/results/dynamic_trades_{timeframe}.csv")
    df_log = df_test[df_test['dynamic_score'] >= 0.6][['timestamp', 'direction', 'is_win']].copy()
    df_log.to_csv(log_path, index=False)
    print(f"[{time.time()-start_time:.1f}s] Trade log saved to {log_path}")

    # 6. FINAL RECALL OPTIMIZATION REPORT
    print("\n" + "="*80)
    print(f"RECALL OPTIMIZATION: COUNT VS WIN RATE ({timeframe.upper()})")
    print("="*80)
    print("Threshold |  Count  | Win Rate | Missed winners recovered")
    print("-" * 55)
    
    base_winners = df_test['is_win'].sum()
    
    for thresh in [1.0, 0.6, 0.3, 0.0]:
        subset = df_test[df_test['dynamic_score'] >= thresh]
        count = len(subset)
        wr = subset['is_win'].mean() if count > 0 else 0
        recovered = subset['is_win'].sum()
        label = "ELITE (3/3)" if thresh == 1.0 else "BALANCED (2/3)" if thresh == 0.6 else "RELAXED (1/3)" if thresh == 0.3 else "RAW (0/0)"
        print(f"{thresh:9} | {count:7} | {wr:8.1%} | {recovered:4} / {base_winners} winners kept ({label})")

    # 7. EXPORT PROFILE FOR API SERVER
    profile_path = Path(f"ml/models/profile_{timeframe}.json")
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with open(profile_path, 'w') as f:
        json.dump(dynamic_filters, f, indent=2)
    print(f"\n[{time.time()-start_time:.1f}s] Profile exported to {profile_path}")

if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", type=str, default="8h")
    args = parser.parse_args()
    run_dynamic_strategy(args.timeframe)
