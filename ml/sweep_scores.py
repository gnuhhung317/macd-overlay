#!/usr/bin/env python3
"""
Score Threshold Multi-Window Sweep.

Runs the 4-way score comparison (All, Weak, Balanced, Elite) across multiple
rolling windows to assess the robustness of different score thresholds.

Usage:
    python ml/sweep_scores.py --timeframe 1d --window-days 90 --step-days 30
"""
import os
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')

import argparse
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from backtest_3stage import BacktestConfig, ThreeStageBacktester

DATA_DIR = Path(__file__).parent.parent / 'bitget-data'
PROCESSED_DIR = DATA_DIR / 'processed'
RESULTS_DIR = Path(__file__).parent / 'results'

def load_data(timeframe='1d'):
    """Load full dataset once."""
    path = PROCESSED_DIR / f'features_{timeframe}_full.parquet'
    if not path.exists():
        path = PROCESSED_DIR / f'features_{timeframe}.parquet'
    
    print(f"Loading data from {path.name}...")
    df = pd.read_parquet(path)
    df = df.sort_values('timestamp')
    print(f"Loaded {len(df):,} total rows.")
    return df

def slice_window(df, start, end):
    """Slice dataframe to a date window."""
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    return df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)].copy()

def run_score_comparison_for_window(df_window, base_config):
    """Run 4 backtests with different score thresholds for a single window."""
    thresholds = [
        (0.0, "All Signals"),
        (0.33, "Weak (>=0.33)"),
        (0.66, "Balanced (>=0.66)"),
        (1.0, "Elite (1.0)")
    ]
    
    results = {}
    
    # Pre-instantiate backtester to avoid redundant model loading if possible
    # Note: ThreeStageBacktester loads models in __init__
    
    for score_thr, label in thresholds:
        config = deepcopy(base_config)
        config.min_refined_score = score_thr
        
        bt = ThreeStageBacktester(config)
        result = bt.run_backtest(df_window, verbose=False)
        
        results[label] = {
            'threshold': score_thr,
            'return_pct': result.total_return * 100,
            'max_dd_pct': result.max_drawdown * 100,
            'win_rate_pct': result.win_rate * 100,
            'trades': result.total_trades,
            'profit_factor': result.profit_factor,
            'avg_trade_pnl': np.mean([t.pnl for t in result.trades]) if result.trades else 0
        }
        
    return results

def main():
    parser = argparse.ArgumentParser(description='Score Threshold Multi-Window Sweep')
    parser.add_argument('--start', type=str, default='2025-08-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2026-02-22', help='End date (YYYY-MM-DD)')
    parser.add_argument('--window-days', type=int, default=90, help='Size of each window in days')
    parser.add_argument('--step-days', type=int, default=30, help='Days to step forward for each window')
    
    parser.add_argument('--timeframe', type=str, default='1d', help='Timeframe')
    parser.add_argument('--leverage', type=float, default=20.0, help='Leverage')
    parser.add_argument('--threshold', type=float, default=0.6, help='Entry confidence threshold')
    parser.add_argument('--max-positions', type=int, default=13, help='Max open positions')
    parser.add_argument('--use-scanner', default=True, action='store_true', help='Enable scanner filter')
    
    args = parser.parse_args()
    
    # Generate rolling windows
    start_dt = datetime.strptime(args.start, '%Y-%m-%d')
    end_dt = datetime.strptime(args.end, '%Y-%m-%d')
    step_td = timedelta(days=args.step_days)
    window_td = timedelta(days=args.window_days)
    
    windows = []
    current_start = start_dt
    while current_start + window_td <= end_dt:
        current_end = current_start + window_td
        windows.append((current_start.strftime('%Y-%m-%d'), current_end.strftime('%Y-%m-%d')))
        current_start += step_td
        
    if not windows:
        print("Error: No windows generated. Check start/end/window settings.")
        return

    print("=" * 80)
    print("SCORE THRESHOLD - MULTI-WINDOW SWEEP")
    print("=" * 80)
    print(f"  Timeframe:     {args.timeframe}")
    print(f"  Windows:       {len(windows)}")
    for s, e in windows:
        print(f"                 {s} -> {e}")
    print(f"  Leverage:      {args.leverage}x")
    print(f"  Entry Thr:     {args.threshold}")
    print("=" * 80)

    # Load data once
    df_full = load_data(args.timeframe)
    
    base_config = BacktestConfig(
        leverage=args.leverage,
        use_kelly=True,
        margin_mode='ISOLATED',
        timeframe=args.timeframe,
        initial_capital=100,
        entry_threshold=args.threshold,
        use_scanner_filter=args.use_scanner,
        max_open_trades=args.max_positions,
    )

    all_window_results = []
    t0 = time.time()
    
    for i, (start_s, end_s) in enumerate(windows):
        print(f"\n[{i+1}/{len(windows)}] Window: {start_s} to {end_s}")
        df_window = slice_window(df_full, start_s, end_s)
        
        if len(df_window) < 100:
            print(f"  Skipping window: insufficient data ({len(df_window)} rows)")
            continue
            
        window_results = run_score_comparison_for_window(df_window, base_config)
        all_window_results.append({
            'start': start_s,
            'end': end_s,
            'results': window_results
        })
        
        # Intermediate summary
        print(f"  {'Threshold':<20} | {'Return%':>10} | {'MaxDD%':>8} | {'Trades':>6} | {'PF':>6}")
        for label, m in window_results.items():
            print(f"  {label:<20} | {m['return_pct']:>10.1f} | {m['max_dd_pct']:>8.1f} | {m['trades']:>6} | {m['profit_factor']:>6.2f}")

    # Aggregate Overall Results
    print("\n" + "=" * 80)
    print("AGGREGATED PERFORMANCE ACROSS ALL WINDOWS")
    print("=" * 80)
    
    agg_metrics = defaultdict(lambda: defaultdict(list))
    for wr in all_window_results:
        for label, metrics in wr['results'].items():
            for k, v in metrics.items():
                if k != 'threshold':
                    agg_metrics[label][k].append(v)
    
    summary_table = []
    hdr = '{:<20} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}'.format(
        'Threshold', 'Avg Ret%', 'Med Ret%', 'Max DD%', 'Avg PF', 'Avg Trades')
    print(hdr)
    print("-" * 80)
    
    for label, data in agg_metrics.items():
        row = '{:<20} | {:>10.1f} | {:>10.1f} | {:>10.1f} | {:>10.2f} | {:>10.1f}'.format(
            label, 
            np.mean(data['return_pct']),
            np.median(data['return_pct']),
            np.max(data['max_dd_pct']),
            np.mean(data['profit_factor']),
            np.mean(data['trades'])
        )
        print(row)
        summary_table.append({
            'label': label,
            'avg_return': np.mean(data['return_pct']),
            'median_return': np.median(data['return_pct']),
            'max_drawdown': np.max(data['max_dd_pct']),
            'avg_profit_factor': np.mean(data['profit_factor']),
            'avg_trades': np.mean(data['trades'])
        })

    # Save Results
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f'score_sweep_{args.timeframe}.json'
    with open(out_path, 'w') as f:
        json.dump({
            'config': vars(args),
            'windows': all_window_results,
            'summary': summary_table
        }, f, indent=2)
    print(f"\nFull results saved to: {out_path}")

    # Visualization
    try:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        labels = [s['label'] for s in summary_table]
        avg_rets = [s['avg_return'] for s in summary_table]
        max_dds = [s['max_drawdown'] for s in summary_table]
        
        x = np.arange(len(labels))
        
        axes[0].bar(x, avg_rets, color='skyblue', alpha=0.8)
        axes[0].set_title(f'Average Return % per Threshold ({len(windows)} windows)')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels)
        axes[0].set_ylabel('Return %')
        axes[0].grid(axis='y', alpha=0.3)
        
        axes[1].bar(x, max_dds, color='salmon', alpha=0.8)
        axes[1].set_title(f'Worst-Case Max Drawdown % per Threshold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels)
        axes[1].set_ylabel('Max DD %')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plot_path = RESULTS_DIR / f'score_sweep_{args.timeframe}.png'
        plt.savefig(plot_path, dpi=150)
        print(f"Summary plot saved to: {plot_path}")
    except Exception as e:
        print(f"Warning: Plotting failed: {e}")

    elapsed = time.time() - t0
    print(f"\nTotal execution time: {elapsed/60:.1f} minutes")

if __name__ == '__main__':
    main()
