
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time
import argparse

# Fix paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.backtest_timeframes import TimeframeBacktester, PROCESSED_DIR, BacktestConfig

def analyze_winners_vs_losers(timeframe='8h'):
    start_time = time.time()
    
    config = BacktestConfig(initial_capital=10000)
    config.start_date = '2020-01-01'
    config.end_date = '2026-02-01'
    config.max_bars = 40
    
    print("\n" + "="*80)
    print(f"SCIENTIFIC WINNERS VS LOSERS ANALYSIS ({timeframe.upper()}, CONF > 0.65)")
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
    
    # Batch predict confidence
    def get_preds(features, model, scaler, X_batch):
        X = pd.DataFrame(index=X_batch.index)
        for f in features: X[f] = X_batch[f] if f in X_batch.columns else 0
        X_s = scaler.transform(X) if scaler else X
        if hasattr(model, 'predict_proba'): return model.predict_proba(X_s)[:, 1]
        return model.predict(X_s)

    X_full = df_signals.fillna(0).replace([np.inf, -np.inf], 0)
    df_signals['confidence'] = get_preds(bt.entry_features, bt.entry_model, bt.entry_scaler, X_full)
    
    # Filter for signals > 0.65
    conf_threshold = 0.65
    print(f"Confidence threshold: {conf_threshold}")
    df_high = df_signals[df_signals['confidence'] >= conf_threshold].copy()
    print(f"[{time.time()-start_time:.1f}s] Found {len(df_high)} signals with confidence >= {conf_threshold}")

    # --- STEP 3: SIMULATE OUTCOMES ---
    print(f"[{time.time()-start_time:.1f}s] Simulating trade outcomes...")
    df_by_sym = {s: g for s, g in df.groupby('symbol')}
    
    results = []
    for idx, row in df_high.iterrows():
        symbol = row['symbol']
        group = df_by_sym[symbol]
        pos = group.index.get_loc(idx)
        
        if pos >= len(group) - config.max_bars: continue
        
        future = group.iloc[pos+1 : pos+1+config.max_bars]
        
        # SL/TP for context
        row_df = pd.DataFrame([row]).fillna(0).replace([np.inf, -np.inf], 0)
        sl_pct = np.clip(get_preds(bt.sl_features, bt.sl_model, bt.sl_scaler, row_df)[0], 0.005, 0.15)
        tp_raw = get_preds(bt.tp_features, bt.tp_model, bt.tp_scaler, row_df)[0]
        tp_pct = np.clip(tp_raw * sl_pct if getattr(bt, 'tp_predict_rr', False) else tp_raw, 0.01, 0.30)
        
        entry_price = row['close']
        direction = 'LONG' if row['macd_cross_up'] == 1 else 'SHORT'
        
        trade = bt.simulate_trade(row, future, entry_price, sl_pct, tp_pct, direction, 1000)
        
        res_row = row.copy()
        res_row['is_win'] = 1 if trade.pnl > 0 else 0
        res_row['pnl_pct'] = trade.pnl_pct * 100
        res_row['direction'] = direction
        results.append(res_row)
        

    df_results = pd.DataFrame(results)

    # --- STEP 4: COMPUTE FEATURE DELTAS ---
    print(f"[{time.time()-start_time:.1f}s] Analyzing feature differences and direction bias...")
    
    # Direction breakdown
    dir_stats = df_results.groupby('direction').agg({'is_win': ['count', 'mean']})
    print("\n" + "="*50)
    print("DIRECTIONAL BREAKDOWN")
    print("="*50)
    print(dir_stats)
    
    # Features to analyze
    analyze_features = [f for f in bt.entry_features if f in df_results.columns]
    

    def get_feature_deltas(subset, name):
        w = subset[subset['is_win'] == 1]
        l = subset[subset['is_win'] == 0]
        if w.empty or l.empty: return pd.DataFrame()
        
        s_list = []
        for feat in analyze_features:
            wm = w[feat].mean()
            lm = l[feat].mean()
            std = subset[feat].std()
            shift = (wm - lm) / std if std > 0 else 0
            s_list.append({'feature': feat, 'win_mean': wm, 'loss_mean': lm, 'z_shift': shift})
        
        res = pd.DataFrame(s_list)
        res['abs_shift'] = res['z_shift'].abs()
        return res.sort_values('abs_shift', ascending=False)
        
    # We'll stick to a simpler version for the tool
    wins = df_results[df_results['is_win'] == 1]
    losses = df_results[df_results['is_win'] == 0]
    
    print(f"\nTotal Wins: {len(wins)}, Total Losses: {len(losses)}")
    
    stats = []
    for feat in analyze_features:
        w_mean = wins[feat].mean()
        l_mean = losses[feat].mean()
        combined_std = df_results[feat].std()
        shift = (w_mean - l_mean) / combined_std if combined_std > 0 else 0
        stats.append({'feature': feat, 'win_mean': w_mean, 'loss_mean': l_mean, 'z_shift': shift})
        

    df_stats = pd.DataFrame(stats)
    df_stats['abs_shift'] = df_stats['z_shift'].abs()
    top_stats = df_stats.sort_values('abs_shift', ascending=False).head(20)
    
    print("\n" + "="*80)
    print("TOP FEATURES DIFFERENTIATING WINS FROM LOSSES (TOTAL)")
    print("="*80)
    print(top_stats[['feature', 'win_mean', 'loss_mean', 'z_shift']])

    # --- DIRECTION SPECIFIC ANALYSIS ---
    for d in ['LONG', 'SHORT']:
        print("\n" + "="*50)
        print(f"TOP FEATURES FOR {d}")
        print("="*50)
        d_subset = df_results[df_results['direction'] == d]
        d_stats = get_feature_deltas(d_subset, d)
        if not d_stats.empty:
            d_stats['abs_shift'] = d_stats['z_shift'].abs()
            print(d_stats.sort_values('abs_shift', ascending=False).head(10)[['feature', 'win_mean', 'loss_mean', 'z_shift']])
    

    # Save the full analysis
    output_path = Path(f"ml/results/winners_vs_losers_{timeframe}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_stats.sort_values('abs_shift', ascending=False).to_csv(output_path, index=False)
    print(f"\n✅ Full analysis saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", type=str, default="8h")
    args = parser.parse_args()
    
    analyze_winners_vs_losers(args.timeframe)
