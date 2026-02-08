
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Fix paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.backtest_timeframes import run_timeframe_backtest, BacktestConfig





def analyze_win_loss_8h():
    import time
    start_time = time.time()
    
    config = BacktestConfig(initial_capital=10000)
    config.start_date = '2022-01-01'
    config.end_date = '2026-02-01'
    config.max_bars = 40
    
    print("\n" + "="*80)
    print("RUNNING 8H TIMEFRAME CONFIDENCE ANALYSIS (EXPRESS MODE)")
    print("="*80)
    
    timeframe = '8h'
    from backtesting.backtest_timeframes import TimeframeBacktester, PROCESSED_DIR
    
    # --- STEP 1: LOAD DATA DIRECTLY ---
    print(f"[{time.time()-start_time:.1f}s] Step 1/5: Loading 8h feature data...")
    data_path = PROCESSED_DIR / f'features_{timeframe}_full.parquet'
    if not data_path.exists():
        print(f"❌ Data not found at {data_path}")
        return
    df = pd.read_parquet(data_path)
    

    # Filter by date manually to be fast
    def make_compatible(ts, ref):
        if ref.tzinfo is not None and ts.tzinfo is None:
            return ts.tz_localize('UTC')
        if ref.tzinfo is None and ts.tzinfo is not None:
            return ts.tz_localize(None)
        return ts

    if not df.empty:
        ref_ts = df['timestamp'].iloc[0]
        if config.start_date:
            start_ts = make_compatible(pd.Timestamp(config.start_date), ref_ts)
            df = df[df['timestamp'] >= start_ts]
        if config.end_date:
            end_ts = make_compatible(pd.Timestamp(config.end_date), ref_ts)
            df = df[df['timestamp'] <= end_ts]
    
    print(f"[{time.time()-start_time:.1f}s] Data loaded: {len(df):,} rows.")

    # --- STEP 2: LOAD MODELS ---
    print(f"[{time.time()-start_time:.1f}s] Step 2/5: Loading ML models...")
    bt = TimeframeBacktester(timeframe, config)
    
    # --- STEP 3: SIGNAL DETECTION ---
    print(f"[{time.time()-start_time:.1f}s] Step 3/5: Detecting signals...")
    # Faster crossover detection
    df = df.sort_values(['symbol', 'timestamp'])
    crossover_mask = (df['macd_cross_up'] == 1) | (df['macd_cross_down'] == 1)
    df_signals = df[crossover_mask].copy()
    
    if df_signals.empty:
        print("No signals found.")
        return

    # Efficiently gather valid signals (avoiding O(N) loop if possible)
    # We can use group shift to check if 40 bars exist ahead
    df['bars_ahead'] = df.groupby('symbol')['timestamp'].transform('count') 
    # Actually, a simpler way is to just use indices
    valid_signal_indices = []
    
    print(f"[{time.time()-start_time:.1f}s] Filtering valid signals...")
    for symbol, group in df.groupby('symbol'):
        sig_indices = df_signals[df_signals['symbol'] == symbol].index
        # Get positional indices in the group
        group_len = len(group)
        # This is slightly slow but better than before
        # We'll just do a quick index check
        for idx in sig_indices:
            pos = group.index.get_loc(idx)
            if pos < group_len - config.max_bars:
                valid_signal_indices.append(idx)
                
    num_signals = len(valid_signal_indices)
    print(f"[{time.time()-start_time:.1f}s] Total valid signals: {num_signals}")

    # --- STEP 4: BATCH PREDICTION ---
    print(f"[{time.time()-start_time:.1f}s] Step 4/5: Batch predicting ML values...")
    # FAST SLICING instead of pd.DataFrame(list)
    X_batch_raw = df.loc[valid_signal_indices].copy()
    X_batch_raw = X_batch_raw.fillna(0).replace([np.inf, -np.inf], 0)

    def fast_batch_predict(model, scaler, features):
        if model is None: return np.zeros(num_signals)
        # Prepare feature matrix
        X = pd.DataFrame(index=X_batch_raw.index)
        for f in features:
            if f in X_batch_raw.columns:
                X[f] = X_batch_raw[f]
            else:
                X[f] = 0
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
            
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X_scaled)[:, 1]
        return model.predict(X_scaled)

    conf_preds = fast_batch_predict(bt.entry_model, bt.entry_scaler, bt.entry_features)
    sl_preds = fast_batch_predict(bt.sl_model, bt.sl_scaler, bt.sl_features)
    tp_preds = fast_batch_predict(bt.tp_model, bt.tp_scaler, bt.tp_features)
    
    # Apply logic
    sl_preds = np.clip(sl_preds, 0.005, 0.15)
    if getattr(bt, 'tp_predict_rr', False):
        tp_actual = tp_preds * sl_preds
    else:
        tp_actual = tp_preds
    tp_actual = np.clip(tp_actual, 0.01, 0.30)

    # --- STEP 5: SIMULATION ---
    print(f"[{time.time()-start_time:.1f}s] Step 5/5: Simulating {num_signals} signals...")
    results_data = []
    
    # We still need the future data for simulation
    # To be FAST, we pre-group the whole dataframe
    df_by_symbol = {s: g for s, g in df.groupby('symbol')}
    
    for i, idx in enumerate(valid_signal_indices):
        if i % 1000 == 0 and i > 0:
            print(f"  Simulated {i}/{num_signals} trades...")
            
        row = df.loc[idx]
        group = df_by_symbol[row['symbol']]
        pos = group.index.get_loc(idx)
        future_data = group.iloc[pos+1 : pos+1+config.max_bars]
        
        direction = 'LONG' if row['macd_cross_up'] == 1 else 'SHORT'
        entry_price = row['close']
        
        # Simulation
        t_ml = bt.simulate_trade(row, future_data, entry_price, sl_preds[i], tp_actual[i], direction, 1000)
        t_20 = bt.simulate_trade(row, future_data, entry_price, sl_preds[i], max(tp_actual[i], 0.20), direction, 1000)
        t_rr = bt.simulate_trade(row, future_data, entry_price, sl_preds[i], sl_preds[i]*2.0, direction, 1000)
        
        results_data.append({
            'confidence': conf_preds[i],
            'win_ml': 1 if t_ml.pnl > 0 else 0,
            'pnl_ml': t_ml.pnl_pct * 100,
            'win_20': 1 if t_20.pnl > 0 else 0,
            'pnl_20': t_20.pnl_pct * 100,
            'win_rr': 1 if t_rr.pnl > 0 else 0
        })


    print(f"[{time.time()-start_time:.1f}s] Simulation complete. Generating report...")
    df_res = pd.DataFrame(results_data)
    
    # Define brackets
    def get_bracket(c):
        if 0.5 <= c < 0.6: return '0.50-0.60'
        if 0.6 <= c < 0.65: return '0.60-0.65'
        if 0.65 <= c < 0.7: return '0.65-0.70'
        if 0.7 <= c < 0.75: return '0.70-0.75'
        if 0.75 <= c < 0.8: return '0.75-0.80'
        if 0.8 <= c <= 1.0: return '0.80-1.00'
        return 'Other'

    df_res['bracket'] = df_res['confidence'].apply(get_bracket)
    
    summary = df_res.groupby('bracket').agg({
        'confidence': 'count',
        'win_ml': 'mean',
        'pnl_ml': 'mean',
        'win_20': 'mean',
        'pnl_20': 'mean',
        'win_rr': 'mean'
    }).rename(columns={'confidence': 'count'})
    
    summary['win_ml'] *= 100
    summary['win_20'] *= 100
    summary['win_rr'] *= 100
    
    report_lines = []
    report_lines.append("\n" + "="*100)
    report_lines.append(f"{'Bracket':<12} | {'Count':<6} | {'WR ML %':<8} | {'PnL ML %':<8} | {'WR 20% %':<8} | {'PnL 20% %':<10} | {'WR RR %':<8}")
    report_lines.append("-" * 100)
    
    for bracket, row in summary.sort_index().iterrows():
        line = f"{bracket:<12} | {int(row['count']):<6} | {row['win_ml']:>7.1f}% | {row['pnl_ml']:>7.2f}% | {row['win_20']:>7.1f}% | {row['pnl_20']:>9.2f}% | {row['win_rr']:>7.1f}%"
        report_lines.append(line)

    report_lines.append("="*100)
    
    overall_text = (
        f"\nOverall Stats (Total {len(df_res)} signals):\n"
        f"ML Win Rate: {df_res['win_ml'].mean():.1%}, Avg PnL: {df_res['pnl_ml'].mean():.2f}%\n"
        f"20% Floor WR: {df_res['win_20'].mean():.1%}, Avg PnL: {df_res['pnl_20'].mean():.2f}%\n"
        f"Fixed RR WR: {df_res['win_rr'].mean():.1%}"
    )
    report_lines.append(overall_text)
    
    report_content = "\n".join(report_lines)
    print(report_content)
    
    # Save to file
    from pathlib import Path
    results_dir = Path("ml/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "8h_stats_report.txt"
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"\n✅ Detailed report saved to: {report_path}")

if __name__ == "__main__":
    analyze_win_loss_8h()
