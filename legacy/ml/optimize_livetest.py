#!/usr/bin/env python3
"""
Optimization script for finding the best threshold and timeframe for live testing.
Uses a rolling window approach to evaluate the robustness of configurations across time.

Usage:
    python optimize_livetest.py --start 2024-01-01 --end 2025-02-22 --window-days 180 --step-days 30
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from copy import deepcopy

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None

# Add parent directory to path to allow importing from ml
sys.path.append(str(Path(__file__).parent.parent))

from ml.backtest_3stage import ThreeStageBacktester, BacktestConfig

DATA_DIR = Path(__file__).parent.parent / 'bitget-data'
PROCESSED_DIR = DATA_DIR / 'processed'
OUTPUT_DIR = Path(__file__).parent.parent / 'output'

def generate_rolling_windows(start_date: str, end_date: str, window_days: int, step_days: int):
    """Generate a list of start and end tuples for rolling windows."""
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    windows = []
    current_start = start_dt
    
    while current_start < end_dt:
        current_end = current_start + pd.Timedelta(days=window_days)
        # We can allow the last window to be clipped to the end date, or strictly require full windows.
        # Here we require the window to fit within the overall timeline or at least have data.
        actual_end = min(current_end, end_dt)
        if (actual_end - current_start).days < (window_days / 2):
            # Skip windows that are too small at the end
            break
            
        windows.append((current_start, actual_end))
        current_start += pd.Timedelta(days=step_days)
        
    return windows

def run_rolling_optimization(args):
    print(f"🚀 Starting Rolling Window Optimization for Livetest Settings")
    print(f"   Period: {args.start} to {args.end}")
    print(f"   Rolling Window: {args.window_days} days (Step: {args.step_days} days)")
    print(f"   Leverage: {args.leverage}x, Risk: {args.risk:.1%}, Max Positions: {args.max_positions}")
    print("-" * 60)
    
    windows = generate_rolling_windows(args.start, args.end, args.window_days, args.step_days)
    if not windows:
        print("❌ Generated 0 windows. Check your start/end dates and window size.")
        return
        
    print(f"Generated {len(windows)} rolling windows:")
    for i, (ws, we) in enumerate(windows):
        print(f"  Window {i+1}: {ws.strftime('%Y-%m-%d')} to {we.strftime('%Y-%m-%d')} ({(we-ws).days} days)")
        
    print("-" * 60)
    
    timeframes = ['1d', '12h']
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    
    all_results = []
    
    for tf in timeframes:
        print(f"\n📊 Processing Timeframe: {tf}")
        # Load dataset once for the timeframe to optimize loading
        data_path = PROCESSED_DIR / f'features_{tf}_full.parquet'
        if not data_path.exists():
            print(f"❌ Data not found: {data_path}. Skipping timeframe {tf}.")
            continue
            
        df_full = pd.read_parquet(data_path)
        df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
        
        # Pre-initialize template backtester
        base_config = BacktestConfig(
            initial_capital=args.capital,
            risk_per_trade=args.risk,
            fee_rate=args.fee,
            slippage=args.slippage,
            leverage=args.leverage,
            timeframe=tf,
            margin_mode=args.margin_mode,
            use_kelly=True,  # Kelly is recommended for live
            max_open_trades=args.max_positions,
            use_scanner_filter=args.use_scanner,
            use_circuit_breaker=True,  # Important to enable circuit breaker for robustness
            min_refined_score=args.min_score
        )
        template_bt = ThreeStageBacktester(base_config)
        
        for thresh in thresholds:
            print(f"  Evaluating Threshold: {thresh:.2f} across all windows...")
            window_metrics = []
            
            for win_start, win_end in windows:
                # Filter dataset safely
                mask = (df_full['timestamp'] >= win_start) & (df_full['timestamp'] <= win_end)
                df_window = df_full.loc[mask].copy()
                
                if df_window.empty:
                    continue
                    
                # Modify template config directly to avoid redundant model loading
                template_bt.config.entry_threshold = thresh
                template_bt.config.start_date = win_start.strftime('%Y-%m-%d')
                template_bt.config.end_date = win_end.strftime('%Y-%m-%d')
                
                result = template_bt.run_backtest(df_window, verbose=False)
                
                # Minimum trades filter
                if result.total_trades < 5:
                    window_metrics.append({
                        "return": result.total_return,
                        "drawdown": result.max_drawdown,
                        "trades": result.total_trades,
                        "win_rate": 0,
                        "positive": False
                    })
                else:
                    window_metrics.append({
                        "return": result.total_return,
                        "drawdown": result.max_drawdown,
                        "trades": result.total_trades,
                        "win_rate": result.win_rate,
                        "positive": result.total_return > 0
                    })
            
            if not window_metrics:
                continue
                
            # Aggregate metrics
            returns = [m['return'] for m in window_metrics]
            drawdowns = [m['drawdown'] for m in window_metrics]
            win_rates = [m['win_rate'] for m in window_metrics if m['trades'] >= 5]
            trades = [m['trades'] for m in window_metrics]
            positive_runs = sum([1 for m in window_metrics if m['positive']])
            total_runs = len(window_metrics)
            
            mean_ret = np.mean(returns)
            std_ret = np.std(returns)
            mean_dd = np.mean(drawdowns)
            max_dd = np.max(drawdowns)
            mean_wr = np.mean(win_rates) if win_rates else 0
            mean_trades = np.mean(trades)
            calmar = mean_ret / max(mean_dd, 0.001) if mean_ret > 0 else 0
            
            # Robustness Metric (Return / Variance) - Sort of a modified Sharpe across windows
            robustness = mean_ret / max(std_ret, 0.01) if mean_ret > 0 else 0
            
            all_results.append({
                "Timeframe": tf,
                "Threshold": thresh,
                "TR Mean (%)": f"{mean_ret * 100:.1f}%",
                "TR Std (%)": f"{std_ret * 100:.1f}%",
                "DD Mean (%)": f"{mean_dd * 100:.1f}%",
                "DD Max (%)": f"{max_dd * 100:.1f}%",
                "WinRate (%)": f"{mean_wr * 100:.1f}%",
                "Avg Trades": f"{mean_trades:.1f}",
                "Positive WinRatio": f"{positive_runs}/{total_runs}",
                "Calmar": round(calmar, 2),
                "Robustness": round(robustness, 2),
                "_raw_calmar": calmar,
                "_raw_ret": mean_ret
            })
            
    # Compile and Sort Report
    if not all_results:
        print("❌ No results collected.")
        return
        
    df_report = pd.DataFrame(all_results)
    df_report = df_report.sort_values(by=["_raw_calmar", "_raw_ret"], ascending=[False, False])
    
    # Drop raw columns for clean table
    df_display = df_report.drop(columns=["_raw_calmar", "_raw_ret"])
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f'livetest_optimization.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Live Test Threshold Optimization (Rolling Windows)\n\n")
        f.write("This report evaluates the robustness of Timeframes and Entry Thresholds over sliding historical windows.\n\n")
        f.write("### Configuration\n")
        f.write(f"- **Period**: {args.start} to {args.end}\n")
        f.write(f"- **Rolling Window**: {args.window_days} days, Sliding by {args.step_days} days\n")
        f.write(f"- **Total Windows Tested**: {len(windows)}\n")
        f.write(f"- **Leverage**: {args.leverage}x\n")
        f.write(f"- **Max Positions**: {args.max_positions}\n\n")
        
        f.write("### Aggregated Results\n")
        f.write("> Sorted by highest average Calmar Ratio (Mean Return / Mean Drawdown) across all windows.\n\n")
        f.write(df_display.to_markdown(index=False))
        
    print("\n" + "="*80)
    print("🏆 OPTIMIZATION COMPLETE 🏆")
    print("="*80)
    print(df_display.head(5).to_markdown(index=False))
    print(f"\n✅ Full report saved to: {report_path}")

def run_optuna_optimization(args):
    if optuna is None:
        print("❌ Optuna is not installed. Please run: pip install optuna")
        return
        
    print(f"🚀 Starting Optuna Rolling Window Optimization")
    print(f"   Period: {args.start} to {args.end}")
    print(f"   Rolling Window: {args.window_days} days (Step: {args.step_days} days)")
    print(f"   Objective: {args.objective}, Trials: {args.trials}")
    print("-" * 60)
    
    windows = generate_rolling_windows(args.start, args.end, args.window_days, args.step_days)
    if not windows:
        print("❌ Generated 0 windows. Check your start/end dates and window size.")
        return
        
    print(f"Generated {len(windows)} rolling windows:")
    for i, (ws, we) in enumerate(windows):
        print(f"  Window {i+1}: {ws.strftime('%Y-%m-%d')} to {we.strftime('%Y-%m-%d')}")
    print("-" * 60)
    
    # Load dataset
    tf = args.timeframe
    data_path = PROCESSED_DIR / f'features_{tf}_full.parquet'
    if not data_path.exists():
        print(f"❌ Data not found: {data_path}")
        return
        
    df_full = pd.read_parquet(data_path)
    df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
    
    # Pre-initialize template backtester to load models once
    base_config = BacktestConfig(
        initial_capital=args.capital,
        risk_per_trade=args.risk,
        fee_rate=args.fee,
        slippage=args.slippage,
        leverage=args.leverage, # Will be overridden by optuna if included in search space
        timeframe=tf,
        margin_mode=args.margin_mode,
        use_kelly=True,
        max_open_trades=args.max_positions,
        use_scanner_filter=args.use_scanner,
        use_circuit_breaker=True,
        min_refined_score=args.min_score
    )
    template_bt = ThreeStageBacktester(base_config)
    
    def objective(trial):
        thresh = trial.suggest_float('entry_threshold', 0.60, 0.70, step=0.05)
        leverage = trial.suggest_int('leverage',5, 25,step=2)
        
        template_bt.config.entry_threshold = thresh
        template_bt.config.leverage = leverage
        
        window_metrics = []
        for win_start, win_end in windows:
            mask = (df_full['timestamp'] >= win_start) & (df_full['timestamp'] <= win_end)
            df_window = df_full.loc[mask].copy()
            
            if df_window.empty:
                continue
                
            template_bt.config.start_date = win_start.strftime('%Y-%m-%d')
            template_bt.config.end_date = win_end.strftime('%Y-%m-%d')
            
            result = template_bt.run_backtest(df_window, verbose=False)
            
            if result.total_trades < 5:
                window_metrics.append({"return": result.total_return, "drawdown": result.max_drawdown, "trades": result.total_trades, "win_rate": 0, "positive": False})
            else:
                window_metrics.append({"return": result.total_return, "drawdown": result.max_drawdown, "trades": result.total_trades, "win_rate": result.win_rate, "positive": result.total_return > 0})
                
        if not window_metrics:
            return 0.0 if args.objective != 'drawdown' else 10.0
            
        returns = [m['return'] for m in window_metrics]
        drawdowns = [m['drawdown'] for m in window_metrics]
        win_rates = [m['win_rate'] for m in window_metrics if m['trades'] >= 5]
        
        mean_ret = np.mean(returns)
        mean_dd = np.mean(drawdowns)
        mean_wr = np.mean(win_rates) if win_rates else 0
        
        # Penalties logic
        is_profitable = mean_ret > 0
        has_minimum_winrate = mean_wr >= 0.40
        positive_runs = sum([1 for m in window_metrics if m['positive']])
        has_positive_runs = positive_runs >= (len(windows)/2)
        
        if args.objective == 'drawdown':
            # Minimize drawdown, but strongly penalize unprofitable runs
            if not is_profitable or not has_minimum_winrate or not has_positive_runs:
                return mean_dd + 10.0 + (abs(mean_ret) * 10) # Massive penalty making it worse than losing 100%
            return mean_dd
        elif args.objective == 'return':
            # Maximize return, but penalize bad win rates and negative runs
            if not is_profitable or not has_minimum_winrate or not has_positive_runs:
                return mean_ret - 100.0
            return mean_ret
        else: # calmar
            calmar = mean_ret / max(mean_dd, 0.001) if mean_ret > 0 else 0
            # Strongly penalize losing approaches
            if not is_profitable or not has_minimum_winrate or not has_positive_runs:
                return calmar - 100.0
            return calmar
            
    study_direction = 'minimize' if args.objective == 'drawdown' else 'maximize'
    study = optuna.create_study(direction=study_direction)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)
    
    print("\n" + "=" * 60)
    print("🏆 OPTUNA OPTIMIZATION COMPLETE 🏆")
    print("=" * 60)
    
    print("\nTop 5 Trials:")
    top_trials = sorted(study.trials, key=lambda t: t.value, reverse=(study_direction == 'maximize'))
    
    for i, t in enumerate(top_trials[:5]):
        print(f"#{i+1} [Trial {t.number}] => Objective ({args.objective}): {t.value:.4f}")
        for k, v in t.params.items():
            print(f"    {k}: {v}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Test Threshold Optimization via Rolling Windows")
    parser.add_argument('--start', type=str, default='2024-01-01', help='Start date of overall optimization period')
    parser.add_argument('--end', type=str, default=datetime.now().strftime('%Y-%m-%d'), help='End date of overall optimization period')
    parser.add_argument('--window-days', type=int, default=120, help='Size of each rolling window in days')
    parser.add_argument('--step-days', type=int, default=30, help='How many days to shift the window each step')
    
    parser.add_argument('--capital', type=float, default=100.0)
    parser.add_argument('--min-score', type=float, default=0.0, help='Minimum refined score threshold (0.33, 0.66, 1.0)')
    parser.add_argument('--risk', type=float, default=0.01)
    parser.add_argument('--fee', type=float, default=0.001)
    parser.add_argument('--slippage', type=float, default=0.0005)
    parser.add_argument("--leverage", type=float, default=20)
    parser.add_argument("--max-positions", type=int, default=15)
    parser.add_argument("--margin-mode", type=str, default="ISOLATED", choices=["ISOLATED", "CROSS"])
    parser.add_argument('--use-scanner', action='store_true', default=False, help='Enable daily Top Volatility scanner filter')
    parser.add_argument('--timeframe', type=str, default='1d', help='Timeframe to optimize on (for Optuna mode)')
    
    # Optuna Args
    parser.add_argument('--optuna', action='store_true', help='Use Optuna to optimize threshold and leverage')
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--objective', type=str, default='drawdown', choices=['calmar', 'drawdown', 'return'], help='Optimization objective')
    
    args = parser.parse_args()
    
    if args.optuna:
        run_optuna_optimization(args)
    else:
        run_rolling_optimization(args)
