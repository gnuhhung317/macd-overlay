#!/usr/bin/env python3
"""
Circuit Breaker Multi-Window Sweep Optimization.

Runs Optuna optimization across multiple date windows and entry thresholds,
then aggregates results to find robust CB parameters.

Usage:
    python sweep_breaker.py --trials 50 --leverage 20
    python sweep_breaker.py --trials 30 --leverage 20 --dd-max 0.35
"""
import os
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import pandas as pd
import numpy as np

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("ERROR: pip install optuna")
    exit(1)

from backtest_3stage import BacktestConfig, ThreeStageBacktester

DATA_DIR = Path(__file__).parent.parent / 'bitget-data'
PROCESSED_DIR = DATA_DIR / 'processed'
RESULTS_DIR = Path(__file__).parent / 'results'

# ─── Default sweep grid ──────────────────────────────────────────────
DEFAULT_WINDOWS = [
    ('2025-09-03', '2026-02-22'),   # Full 6-month window
    ('2025-09-03', '2025-12-31'),   # H2 2025
    ('2026-01-01', '2026-02-22'),   # YTD 2026
    ('2025-10-01', '2026-01-31'),   # Mid-range 4 months
]

DEFAULT_THRESHOLDS = [0.60, 0.65]


def load_data(timeframe='1d'):
    """Load full dataset once."""
    path = PROCESSED_DIR / f'features_{timeframe}_full.parquet'
    df = pd.read_parquet(path)
    df = df.sort_values('timestamp')
    print(f"Loaded {len(df):,} total rows from {path.name}")
    return df


def slice_window(df, start, end):
    """Slice dataframe to a date window."""
    return df[(df['timestamp'] >= start) & (df['timestamp'] <= end)].copy()


def create_objective(df_test, base_config, dd_constraint=0.40):
    """Create Optuna objective: maximize return with DD constraint."""
    _bt_template = ThreeStageBacktester(base_config)

    def objective(trial):
        confluence_tf = trial.suggest_categorical('confluence_tf', [ '12h'])
        confluence_threshold = trial.suggest_float('confluence_threshold', 0.15, 0.50, step=0.05)
        velocity_threshold = trial.suggest_float('velocity_threshold', 0.10, 0.40, step=0.05)
        velocity_lookback = trial.suggest_int('velocity_lookback', 1, 3)
        sleep_bars = trial.suggest_int('sleep_bars', 1, 5)

        config = deepcopy(base_config)
        config.use_circuit_breaker = True
        config.cb_confluence_tf = confluence_tf
        config.cb_confluence_threshold = confluence_threshold
        config.cb_velocity_threshold = velocity_threshold
        config.cb_velocity_lookback = velocity_lookback
        config.cb_sleep_hours = sleep_bars

        bt = ThreeStageBacktester(config)
        bt.entry_model = _bt_template.entry_model
        bt.sl_model = _bt_template.sl_model
        bt.tp_model = _bt_template.tp_model
        bt.entry_scaler = _bt_template.entry_scaler
        bt.sl_scaler = _bt_template.sl_scaler
        bt.tp_scaler = _bt_template.tp_scaler
        bt.entry_features = _bt_template.entry_features
        bt.sl_features = _bt_template.sl_features
        bt.tp_features = _bt_template.tp_features
        if hasattr(_bt_template, 'tp_predict_rr'):
            bt.tp_predict_rr = _bt_template.tp_predict_rr

        result = bt.run_backtest(df_test, verbose=False)

        total_return = result.total_return
        max_dd = result.max_drawdown
        n_trades = result.total_trades

        if n_trades < 10:
            return -(1000 + (10 - n_trades) * 10)

        trial.set_user_attr('total_return', total_return)
        trial.set_user_attr('max_drawdown', max_dd)
        trial.set_user_attr('total_trades', n_trades)
        trial.set_user_attr('win_rate', result.win_rate)
        trial.set_user_attr('equity_final', result.equity_curve[-1] if result.equity_curve else 0)
        trial.set_user_attr('cb_exits', sum(1 for t in result.trades if t.exit_reason == 'CIRCUIT_BREAKER'))

        if max_dd > dd_constraint:
            return -(max_dd * 100)
        return total_return

    return objective


def run_single_combo(df_full, threshold, window, args):
    """Run optimization for one (threshold, window) combo."""
    start, end = window
    label = f"thr={threshold}, {start}~{end}"
    df_test = slice_window(df_full, start, end)
    n_symbols = df_test['symbol'].nunique()
    print(f"\n{'='*70}")
    print(f"  {label}  |  {len(df_test):,} rows, {n_symbols} symbols")
    print(f"{'='*70}")

    base_config = BacktestConfig(
        leverage=args.leverage,
        use_kelly=True,
        margin_mode='ISOLATED',
        timeframe=args.timeframe,
        initial_capital=100,
        entry_threshold=threshold,
        use_scanner_filter=args.use_scanner,
        max_open_trades=args.max_positions,
        min_refined_score=args.min_score,
    )

    # Baseline
    bt_base = ThreeStageBacktester(deepcopy(base_config))
    res_base = bt_base.run_backtest(df_test, verbose=False)
    baseline_ret = res_base.total_return * 100
    baseline_dd = res_base.max_drawdown * 100
    baseline_calmar = res_base.total_return / max(res_base.max_drawdown, 0.001)
    print(f"  Baseline: Return={baseline_ret:.0f}%, MaxDD={baseline_dd:.1f}%, Calmar={baseline_calmar:.1f}")

    # Optuna
    objective = create_objective(df_test, base_config, args.dd_max)
    study = optuna.create_study(direction='maximize', study_name=f'cb_{threshold}_{start}')
    study.optimize(objective, n_trials=args.trials, show_progress_bar=(args.n_jobs == 1))

    best = study.best_trial
    best_ret = best.user_attrs.get('total_return', 0) * 100
    best_dd = best.user_attrs.get('max_drawdown', 0) * 100
    best_wr = best.user_attrs.get('win_rate', 0) * 100
    best_trades = best.user_attrs.get('total_trades', 0)
    cb_exits = best.user_attrs.get('cb_exits', 0)

    print(f"  Best CB:  Return={best_ret:.0f}%, MaxDD={best_dd:.1f}%, "
          f"WR={best_wr:.1f}%, Trades={best_trades}, CB_exits={cb_exits}")
    print(f"  Params:   {best.params}")

    return {
        'label': label,
        'threshold': threshold,
        'window': f"{start} ~ {end}",
        'baseline': {
            'return_pct': baseline_ret,
            'max_dd_pct': baseline_dd,
            'calmar': baseline_calmar,
            'trades': res_base.total_trades,
        },
        'best_params': best.params,
        'best_value': best.value,
        'best_metrics': {
            'return_pct': best_ret,
            'max_dd_pct': best_dd,
            'win_rate_pct': best_wr,
            'trades': best_trades,
            'cb_exits': cb_exits,
        },
        'all_trials': [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'return_pct': t.user_attrs.get('total_return', 0) * 100,
                'max_dd_pct': t.user_attrs.get('max_drawdown', 0) * 100,
            }
            for t in study.trials if t.value is not None and t.value > -999
        ],
    }


def aggregate_results(all_results):
    """Find parameters that appear consistently in top results across windows."""
    print("\n" + "=" * 80)
    print("AGGREGATED RESULTS - CROSS-WINDOW ROBUSTNESS")
    print("=" * 80)

    # Summary table
    hdr = '{:<35} {:>10} {:>10} {:>9} {:>10} {:>10} {:>9}'.format(
        'Config', 'BL Ret%', 'CB Ret%', 'BL DD%', 'CB DD%', 'CB WR%', 'CB_Exit')
    print(f"\n{hdr}")
    print('-' * 95)

    for r in all_results:
        bl = r['baseline']
        cb = r['best_metrics']
        label = f"thr={r['threshold']}, {r['window'][:10]}~{r['window'][-10:]}"
        row = '{:<35} {:>10.0f} {:>10.0f} {:>9.1f} {:>10.1f} {:>10.1f} {:>9}'.format(
            label, bl['return_pct'], cb['return_pct'],
            bl['max_dd_pct'], cb['max_dd_pct'],
            cb['win_rate_pct'], cb['cb_exits'])
        print(row)

    # Find most common parameter values
    print("\n--- Parameter Frequency (Best per combo) ---")
    param_counts = defaultdict(lambda: defaultdict(int))
    for r in all_results:
        for k, v in r['best_params'].items():
            param_counts[k][v] += 1

    for param, values in param_counts.items():
        sorted_vals = sorted(values.items(), key=lambda x: -x[1])
        top_vals = ', '.join(f"{v}({c}x)" for v, c in sorted_vals[:3])
        print(f"  {param}: {top_vals}")

    # Find params that are consistently good across ALL windows for each threshold
    print("\n--- Consensus Best Params ---")
    for thr in set(r['threshold'] for r in all_results):
        thr_results = [r for r in all_results if r['threshold'] == thr]
        if not thr_results:
            continue
        # Most common value per param
        consensus = {}
        for param in thr_results[0]['best_params']:
            vals = [r['best_params'][param] for r in thr_results]
            # Mode
            from statistics import mode as stat_mode
            try:
                consensus[param] = stat_mode(vals)
            except Exception:
                consensus[param] = vals[0]
        print(f"  Threshold {thr}: {consensus}")


def main():
    parser = argparse.ArgumentParser(description='CB Multi-Window Sweep Optimization')
    parser.add_argument('--trials', type=int, default=30, help='Trials PER combo (default: 30)')
    parser.add_argument('--leverage', type=float, default=20.0, help='Leverage')
    parser.add_argument('--timeframe', type=str, default='1d', help='Timeframe')
    parser.add_argument('--max-positions', type=int, default=13, help='Max open positions')
    parser.add_argument('--use-scanner', action='store_true', help='Enable scanner filter')
    parser.add_argument('--dd-max', type=float, default=0.40, help='Max DD constraint')
    parser.add_argument('--thresholds', type=str, default='0.60,0.65',
                       help='Comma-separated entry thresholds (default: 0.60,0.65)')
    parser.add_argument('--min-score', type=float, default=0.0, help='Minimum refined score threshold (0.33, 0.66, 1.0)')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Number of parallel processes (-1 for all cores)')
    
    # Rolling window args
    parser.add_argument('--start', type=str, default='2025-08-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2026-02-22', help='End date (YYYY-MM-DD)')
    parser.add_argument('--step-days', type=int, default=30, help='Days to step forward for each window')
    parser.add_argument('--window-days', type=int, default=90, help='Size of each window in days')
    
    args = parser.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(',')]

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
        
    # Add final window if the last one missed the end date
    if windows and windows[-1][1] != end_dt.strftime('%Y-%m-%d'):
        last_start = end_dt - window_td
        if last_start >= start_dt:
             windows.append((last_start.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')))
             
    # Unique windows
    windows = list(dict.fromkeys(windows))

    print("=" * 80)
    print("CIRCUIT BREAKER - MULTI-WINDOW SWEEP OPTIMIZATION")
    print("=" * 80)
    print(f"  Thresholds:    {thresholds}")
    print(f"  Windows:       {len(windows)}")
    for s, e in windows:
        print(f"                 {s} -> {e}")
    print(f"  Trials/combo:  {args.trials}")
    print(f"  DD constraint: {args.dd_max:.0%}")
    print(f"  Parallel Jobs: {args.n_jobs if args.n_jobs != -1 else 'All Cores'}")
    print(f"  Total combos:  {len(thresholds) * len(windows)}")
    total_trials = args.trials * len(thresholds) * len(windows)
    est_mins = total_trials * 30 / 60
    print(f"  Total trials:  {total_trials} (est. ~{est_mins:.0f} min sequentially)")

    # Load data once
    df_full = load_data(args.timeframe)

    # Run sweep
    all_results = []
    t0 = time.time()
    
    combos = [(thr, window) for thr in thresholds for window in windows]
    total = len(combos)
    
    if args.n_jobs == 1:
        for thr, window in combos:
            result = run_single_combo(df_full, thr, window, args)
            all_results.append(result)
            elapsed = time.time() - t0
            done = len(all_results)
            remaining = elapsed / done * (total - done) / 60
            print(f"  [{done}/{total}] elapsed={elapsed/60:.1f}m, remaining~{remaining:.0f}m")
    else:
        print(f"  Running {total} combinations in parallel processes...")
        import joblib
        all_results = joblib.Parallel(n_jobs=args.n_jobs)(
            joblib.delayed(run_single_combo)(df_full, thr, window, args) for thr, window in combos
        )
        elapsed = time.time() - t0
        print(f"  [DONE] {total} combos completed in {elapsed/60:.1f} minutes")

    # Aggregate
    aggregate_results(all_results)

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / 'cb_sweep_results.json'
    # Strip non-serializable
    for r in all_results:
        for t in r.get('all_trials', []):
            if t.get('value') is not None and not np.isfinite(t['value']):
                t['value'] = None

    with open(out_path, 'w') as f:
        json.dump({
            'sweep_config': {
                'thresholds': thresholds,
                'windows': [list(w) for w in windows],
                'trials_per_combo': args.trials,
                'dd_max': args.dd_max,
                'leverage': args.leverage,
                'max_positions': args.max_positions,
                'min_score': args.min_score
            },
            'results': all_results,
        }, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")

    # Visualization
    try:
        import matplotlib.pyplot as plt

        n_combos = len(all_results)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Baseline vs Best CB return per combo
        labels = [f"{r['threshold']}\n{r['window'][:10]}" for r in all_results]
        bl_rets = [r['baseline']['return_pct'] for r in all_results]
        cb_rets = [r['best_metrics']['return_pct'] for r in all_results]

        x = np.arange(n_combos)
        w = 0.35
        axes[0].bar(x - w/2, bl_rets, w, label='Baseline', color='steelblue', alpha=0.8)
        axes[0].bar(x + w/2, cb_rets, w, label='Best CB', color='seagreen', alpha=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, fontsize=8)
        axes[0].set_ylabel('Return %')
        axes[0].set_title('Return: Baseline vs Best CB')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        # Plot 2: Max DD comparison
        bl_dds = [r['baseline']['max_dd_pct'] for r in all_results]
        cb_dds = [r['best_metrics']['max_dd_pct'] for r in all_results]

        axes[1].bar(x - w/2, bl_dds, w, label='Baseline DD', color='indianred', alpha=0.8)
        axes[1].bar(x + w/2, cb_dds, w, label='CB DD', color='seagreen', alpha=0.8)
        axes[1].axhline(y=args.dd_max * 100, color='red', linestyle='--', alpha=0.5,
                        label=f'DD Cap ({args.dd_max:.0%})')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, fontsize=8)
        axes[1].set_ylabel('Max Drawdown %')
        axes[1].set_title('Max Drawdown: Baseline vs Best CB')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plot_path = RESULTS_DIR / 'cb_sweep_plot.png'
        plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {plot_path}")
    except Exception as e:
        print(f"Warning: plot failed: {e}")


if __name__ == '__main__':
    main()
