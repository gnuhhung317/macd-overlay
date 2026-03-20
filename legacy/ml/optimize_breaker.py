#!/usr/bin/env python3
"""
Circuit Breaker Hyperparameter Optimization using Optuna.

Searches for optimal Circuit Breaker parameters that maximize
risk-adjusted returns (Calmar Ratio) or total return with
Max Drawdown constraint.

Usage:
    python optimize_breaker.py --trials 50 --leverage 20
    python optimize_breaker.py --trials 100 --leverage 20 --objective calmar
"""
import os
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')

import argparse
import json
from copy import deepcopy
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("ERROR: Optuna not installed. Run: pip install optuna")
    exit(1)

from backtest_3stage import BacktestConfig, ThreeStageBacktester

DATA_DIR = Path(__file__).parent.parent / 'bitget-data'
PROCESSED_DIR = DATA_DIR / 'processed'
RESULTS_DIR = Path(__file__).parent / 'results'


def load_test_data(timeframe='1d', months=6, start_date=None, end_date=None):
    """Load and prepare test data."""
    path = PROCESSED_DIR / f'features_{timeframe}_full.parquet'
    df = pd.read_parquet(path)
    df = df.sort_values('timestamp')

    if start_date and end_date:
        df_test = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)].copy()
    elif start_date:
        df_test = df[df['timestamp'] >= start_date].copy()
    else:
        latest = df['timestamp'].max()
        cutoff = latest - pd.DateOffset(months=months)
        df_test = df[df['timestamp'] >= cutoff].copy()

    print(f"Loaded {len(df_test):,} rows, "
          f"{df_test['symbol'].nunique()} symbols, "
          f"{df_test['timestamp'].min()} to {df_test['timestamp'].max()}")
    return df_test


def create_objective(df_test, base_config, objective_type='calmar', dd_constraint=0.50):
    """Create Optuna objective function."""

    # Pre-create backtester for model loading (shared across trials)
    _bt_template = ThreeStageBacktester(base_config)

    def objective(trial):
        # Search space
        confluence_tf = trial.suggest_categorical('confluence_tf', ['12h'])
        confluence_threshold = trial.suggest_float('confluence_threshold', 0.15, 0.50, step=0.05)
        velocity_threshold = trial.suggest_float('velocity_threshold', 0.10, 0.40, step=0.05)
        velocity_lookback = trial.suggest_int('velocity_lookback', 1, 3)  # 4-12h in 4H bars
        sleep_bars = trial.suggest_int('sleep_bars', 1, 5)  # Bars in base TF (1d = 1-5 days)

        # Build config
        config = deepcopy(base_config)
        config.use_circuit_breaker = True
        config.cb_confluence_tf = confluence_tf
        config.cb_confluence_threshold = confluence_threshold
        config.cb_velocity_threshold = velocity_threshold
        config.cb_velocity_lookback = velocity_lookback
        config.cb_sleep_hours = sleep_bars  # cb_sleep_hours is actually bars in backtester

        # Run backtest (reuse model weights)
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

        # Extract metrics
        total_return = result.total_return
        max_dd = result.max_drawdown
        n_trades = result.total_trades

        # Minimum trade filter — proportional penalty so TPE can learn direction
        if n_trades < 20:
            return -(1000 + (20 - n_trades) * 10)

        # Store for analysis
        trial.set_user_attr('total_return', total_return)
        trial.set_user_attr('max_drawdown', max_dd)
        trial.set_user_attr('total_trades', n_trades)
        trial.set_user_attr('win_rate', result.win_rate)
        trial.set_user_attr('equity_final', result.equity_curve[-1] if result.equity_curve else 0)
        trial.set_user_attr('cb_exits', sum(1 for t in result.trades if t.exit_reason == 'CIRCUIT_BREAKER'))

        if objective_type == 'calmar':
            if max_dd > dd_constraint:
                # Proportional penalty: guides TPE toward lower DD
                return -(max_dd * 100)
            calmar = total_return / max(max_dd, 0.001)
            return calmar
        elif objective_type == 'return':
            if max_dd > dd_constraint:
                return -(max_dd * 100)
            return total_return
        elif objective_type == 'dd_penalized':
            # Total return with quadratic DD penalty
            dd_penalty = max(0, (max_dd - 0.30)) ** 2 * 100
            return total_return - dd_penalty
        else:
            return total_return / max(max_dd, 0.001)

    return objective


def run_optimization(args):
    """Run the full optimization pipeline."""
    print("=" * 60)
    print("CIRCUIT BREAKER - OPTUNA OPTIMIZATION")
    print("=" * 60)

    # Load data
    df_test = load_test_data(args.timeframe, args.months,
                             start_date=args.start, end_date=args.end)

    # Base config
    base_config = BacktestConfig(
        leverage=args.leverage,
        use_kelly=True,
        margin_mode='ISOLATED',
        timeframe=args.timeframe,
        initial_capital=100,
        entry_threshold=args.threshold,
        use_scanner_filter=args.use_scanner,
        max_open_trades=args.max_positions,
        min_refined_score=args.min_score,
    )

    # Run baseline first
    print("\nRunning baseline (no CB)...")
    bt_base = ThreeStageBacktester(deepcopy(base_config))
    res_base = bt_base.run_backtest(df_test, verbose=False)
    print(f"  Baseline: Return={res_base.total_return*100:.1f}%, "
          f"MaxDD={res_base.max_drawdown*100:.1f}%, "
          f"Calmar={res_base.total_return/max(res_base.max_drawdown, 0.001):.2f}")

    # Create study
    objective = create_objective(df_test, base_config, args.objective, args.dd_max)

    study = optuna.create_study(
        direction='maximize',
        study_name='circuit_breaker_optimization'
    )

    print(f"\nOptimizing {args.trials} trials (objective: {args.objective}, DD cap: {args.dd_max:.0%})...")
    study.optimize(objective, n_trials=args.trials, n_jobs=args.n_jobs, show_progress_bar=(args.n_jobs == 1))

    # Results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)

    best = study.best_trial
    print(f"\nBest Trial #{best.number}:")
    print(f"  Objective Value: {best.value:.4f}")
    print(f"  Parameters:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
    print(f"  Metrics:")
    for k in ['total_return', 'max_drawdown', 'total_trades', 'win_rate', 'equity_final', 'cb_exits']:
        v = best.user_attrs.get(k, 'N/A')
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")

    # Top 5 trials
    print("\nTop 5 Trials:")
    print('{:<6} {:>10} {:>10} {:>10} {:>8} {:>8} {:>6} {:>6} {:>6}'.format(
        '#', 'Objective', 'Return%', 'MaxDD%', 'Trades', 'WinR%', 'ConfTF', 'ConfTh', 'Sleep'))
    print('-' * 95)

    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -9999, reverse=True)
    for t in sorted_trials[:5]:
        if t.value is None or t.value < -999:
            continue
        ret = t.user_attrs.get('total_return', 0) * 100
        dd = t.user_attrs.get('max_drawdown', 0) * 100
        trades = t.user_attrs.get('total_trades', 0)
        wr = t.user_attrs.get('win_rate', 0) * 100
        print('{:<6} {:>10.2f} {:>10.1f} {:>10.1f} {:>8} {:>8.1f} {:>6} {:>6.2f} {:>6}'.format(
            t.number, t.value, ret, dd, trades, wr,
            t.params.get('confluence_tf', '?'),
            t.params.get('confluence_threshold', 0),
            t.params.get('sleep_bars', 0)))

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    results_path = RESULTS_DIR / 'cb_optimization_results.json'
    results_data = {
        'best_params': best.params,
        'best_value': best.value,
        'best_metrics': best.user_attrs,
        'baseline': {
            'total_return': res_base.total_return,
            'max_drawdown': res_base.max_drawdown,
            'total_trades': res_base.total_trades,
            'win_rate': res_base.win_rate,
        },
        'n_trials': args.trials,
        'objective': args.objective,
        'dd_constraint': args.dd_max,
    }
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    # Generate optimization visualization
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Optimization history
        valid_trials = [t for t in study.trials if t.value is not None and t.value > -999]
        trial_nums = [t.number for t in valid_trials]
        trial_vals = [t.value for t in valid_trials]

        axes[0].scatter(trial_nums, trial_vals, alpha=0.5, s=20, c='steelblue')
        axes[0].plot(trial_nums,
                     pd.Series(trial_vals).expanding().max().values,
                     color='red', linewidth=2, label='Best so far')
        axes[0].set_xlabel('Trial #')
        axes[0].set_ylabel('Objective Value')
        axes[0].set_title('Optimization History')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Return vs Drawdown scatter
        returns = [t.user_attrs.get('total_return', 0) * 100 for t in valid_trials]
        drawdowns = [t.user_attrs.get('max_drawdown', 0) * 100 for t in valid_trials]

        axes[1].scatter(drawdowns, returns, alpha=0.5, s=20, c='steelblue')
        axes[1].scatter([res_base.max_drawdown * 100], [res_base.total_return * 100],
                       s=100, c='red', marker='X', label='Baseline', zorder=5)
        axes[1].scatter([best.user_attrs.get('max_drawdown', 0) * 100],
                       [best.user_attrs.get('total_return', 0) * 100],
                       s=100, c='green', marker='*', label='Best CB', zorder=5)
        axes[1].set_xlabel('Max Drawdown (%)')
        axes[1].set_ylabel('Total Return (%)')
        axes[1].set_title('Return vs Drawdown Trade-off')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = RESULTS_DIR / 'cb_optimization_plot.png'
        plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {plot_path}")
    except Exception as e:
        print(f"Warning: Could not create plot: {e}")


def main():
    parser = argparse.ArgumentParser(description='Circuit Breaker Optuna Optimization')
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--leverage', type=float, default=20.0, help='Leverage multiplier')
    parser.add_argument('--timeframe', type=str, default='1d', help='Timeframe')
    parser.add_argument('--months', type=int, default=6, help='Months of test data')
    parser.add_argument('--start', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--threshold', type=float, default=0.65, help='Entry confidence threshold')
    parser.add_argument('--min-score', type=float, default=0.0, help='Minimum refined score threshold (0.33, 0.66, 1.0)')
    parser.add_argument('--use-scanner', action='store_true', help='Enable daily Top Volatility scanner filter')
    parser.add_argument('--max-positions', type=int, default=15, help='Maximum open trades')
    parser.add_argument('--objective', type=str, default='calmar',
                       choices=['calmar', 'return', 'dd_penalized'],
                       help='Optimization objective')
    parser.add_argument('--dd-max', type=float, default=0.50,
                       help='Max drawdown constraint (0.50 = 50%%)')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Number of parallel jobs (-1 for all cores)')
    args = parser.parse_args()
    run_optimization(args)


if __name__ == '__main__':
    main()
