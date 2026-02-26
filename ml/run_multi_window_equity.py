#!/usr/bin/env python3
"""
Multi-Window Equity Runner
Runs plot_time_equity logic across sliding date windows to build a large equity dataset.
Supports day-level step and window size customization.

Example:
  python ml/run_multi_window_equity.py \
    --start 2021-01-01 --end 2026-02-22 \
    --window-days 90 --step-days 10 \
    --timeframe 1d --leverage 20 --threshold 0.6 \
    --use-scanner --max-positions 10
"""

import pandas as pd
import numpy as np
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# Import backtest modules
from backtest_3stage import ThreeStageBacktester, BacktestConfig
from plot_time_equity import create_time_based_equity_mtm


def run_single_window(args_dict, df_path, window_start, window_end, window_idx, total_windows):
    """
    Run a single backtest window and return the daily equity DataFrame.
    Designed to be called in a subprocess.
    """
    try:
        label = f"[{window_idx+1}/{total_windows}] {window_start} → {window_end}"
        print(f"  🔄 {label} ...", flush=True)

        # Load data inside subprocess
        df = pd.read_parquet(df_path)

        # Filter to window + buffer (warm-up handled by backtester)
        buffer_start = pd.to_datetime(window_start) - pd.DateOffset(months=args_dict.get('warmup', 0))
        buffer_end = pd.to_datetime(window_end)
        
        df_window = df[
            (df['timestamp'] >= buffer_start) & 
            (df['timestamp'] <= buffer_end)
        ].copy()

        if df_window.empty:
            print(f"  ⚠️  {label} — No data, skipping")
            return None

        # Circuit Breaker Profile Mapping
        cb_kwargs = {}
        cb_profile = args_dict.get('cb_profile', 'none')
        if cb_profile == '0.6':
            cb_kwargs = {
                'use_circuit_breaker': True,
                'cb_confluence_tf': '12h',
                'cb_confluence_threshold': 0.2,
                'cb_velocity_lookback': 2,
                'cb_velocity_threshold': 0.1,
                'cb_sleep_hours': 4
            }
        elif cb_profile == '0.65':
            cb_kwargs = {
                'use_circuit_breaker': True,
                'cb_confluence_tf': '12h',
                'cb_confluence_threshold': 0.15,
                'cb_velocity_lookback': 1,
                'cb_velocity_threshold': 0.1,
                'cb_sleep_hours': 5
            }

        # Build config
        config = BacktestConfig(
            initial_capital=args_dict['capital'],
            risk_per_trade=args_dict['risk'],
            entry_threshold=args_dict['threshold'],
            fee_rate=args_dict['fee'],
            slippage=args_dict['slippage'],
            leverage=args_dict['leverage'],
            timeframe=args_dict['timeframe'],
            margin_mode=args_dict['margin_mode'],
            use_kelly=args_dict['kelly'],
            fixed_position_size=args_dict['fixed_size'],
            position_size_usd=args_dict['size_usd'],
            max_open_trades=args_dict['max_positions'],
            require_fresh_crossover_after_exit=True,
            use_trailing_stop=args_dict['trailing'],
            trailing_start_pct=args_dict['trailing_start'],
            trailing_step_pct=args_dict['trailing_step'],
            use_portfolio_trailing=args_dict.get('portfolio_trailing', False),
            # portfolio_trailing_start_pct=args_dict.get('pt_start', 0.30),
            # portfolio_trailing_step_pct=args_dict.get('pt_step', 0.15),
            # portfolio_cooldown_days=args_dict.get('pt_cooldown', 1.0),
            entry_pullback_pct=args_dict['entry_pullback'],
            entry_pullback_timeout=args_dict['entry_timeout'],
            max_bars=args_dict['max_bars'],
            use_scanner_filter=args_dict['use_scanner'],
            scanner_mae=args_dict['scanner_mae'],
            scanner_mfe=args_dict['scanner_mfe'],
            scanner_lookback_days=args_dict['scanner_lookback'],
            **cb_kwargs
        )

        backtester = ThreeStageBacktester(config)
        result = backtester.run_backtest(df_window, verbose=False)

        if not result.trades:
            print(f"  ⚠️  {label} — 0 trades")
            return None

        # Filter trades to only those starting within the analysis window
        analysis_start_ts = pd.to_datetime(window_start)
        result.trades = [
            t for t in result.trades
            if t.entry_time.replace(tzinfo=None) >= analysis_start_ts
        ]

        if not result.trades:
            print(f"  ⚠️  {label} — 0 trades after filtering")
            return None

        # Prepare price data
        price_columns = ['timestamp', 'close']
        if 'symbol' in df_window.columns:
            price_columns.insert(0, 'symbol')
        price_data = df_window[price_columns].copy()

        # Create equity curve
        daily_eq = create_time_based_equity_mtm(
            result.trades, window_start, window_end,
            args_dict['capital'], price_data,
            leverage=args_dict['leverage'],
            actual_daily_positions=result.daily_open_positions
        )

        if daily_eq is None or daily_eq.empty:
            return None

        # Add window metadata columns
        daily_eq['window_start'] = window_start
        daily_eq['window_end'] = window_end
        daily_eq['window_idx'] = window_idx
        daily_eq['num_trades'] = len(result.trades)

        final_eq = daily_eq['equity'].iloc[-1]
        init_eq = daily_eq['equity'].iloc[0]
        ret = (final_eq / init_eq - 1) * 100 if init_eq > 0 else 0

        # Drawdown
        peak = np.maximum.accumulate(daily_eq['equity'].values)
        dd = (peak - daily_eq['equity'].values) / peak * 100
        max_dd = np.max(dd)

        print(f"  ✅ {label} — {len(result.trades)} trades, Return: {ret:+.1f}%, MaxDD: {max_dd:.1f}%")

        return daily_eq

    except Exception as e:
        print(f"  ❌ Window {window_idx+1} error: {e}")
        traceback.print_exc()
        return None


def compute_drawdown_features(combined_df):
    """Add drawdown and time features to the combined DataFrame."""
    df = combined_df.copy()
    
    # Parse date if needed
    df['date'] = pd.to_datetime(df['date'])

    # Per-window drawdown calculation
    dd_list = []
    dd_from_peak_list = []
    for _, group in df.groupby('window_idx'):
        eq = group['equity'].values
        peak = np.maximum.accumulate(eq)
        dd_pct = (peak - eq) / peak * 100
        dd_from_peak = peak - eq
        dd_list.extend(dd_pct.tolist())
        dd_from_peak_list.extend(dd_from_peak.tolist())

    df['drawdown_pct'] = dd_list
    df['drawdown_usd'] = dd_from_peak_list

    # Time features
    df['day_of_week'] = df['date'].dt.dayofweek       # 0=Mon..6=Sun
    df['day_of_week_name'] = df['date'].dt.day_name()
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.month_name()
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    df['day_of_month'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)

    # Rolling volatility (5-day window)
    df['rolling_vol_5d'] = df.groupby('window_idx')['daily_return'].transform(
        lambda x: x.rolling(5, min_periods=2).std()
    )

    # Drawdown streak: consecutive days of increasing drawdown
    def calc_streak(dd_series):
        streak = []
        current = 0
        prev = 0
        for v in dd_series:
            if v > prev and v > 0:
                current += 1
            else:
                current = 0
            streak.append(current)
            prev = v
        return streak

    streak_list = []
    for _, group in df.groupby('window_idx'):
        streak_list.extend(calc_streak(group['drawdown_pct'].values))
    df['drawdown_streak'] = streak_list

    return df


def main():
    parser = argparse.ArgumentParser(description="Multi-Window Equity Runner for Drawdown Analysis")

    # Window params (support both days and months)
    parser.add_argument('--window-days', type=int, default=None, help='Window size in days (e.g. 60)')
    parser.add_argument('--window-months', type=int, default=None, help='Window size in months (alternative to --window-days)')
    parser.add_argument('--step-days', type=int, default=None, help='Step size in days (e.g. 30)')
    parser.add_argument('--step-months', type=int, default=None, help='Step size in months (alternative to --step-days)')

    # Backtest params (same as plot_time_equity.py)
    parser.add_argument('--data', type=str, default=None, help='Path to data file')
    parser.add_argument('--capital', type=float, default=100.0, help='Initial capital')
    parser.add_argument('--risk', type=float, default=0.01, help='Risk per trade')
    parser.add_argument('--threshold', type=float, default=0.65, help='Entry confidence threshold')
    parser.add_argument('--fee', type=float, default=0.001, help='Fee rate')
    parser.add_argument('--slippage', type=float, default=0.0005, help='Slippage')
    parser.add_argument('--kelly', action='store_true', help='Use Kelly Criterion')
    parser.add_argument('--fixed-size', action='store_true', help='Use fixed position size')
    parser.add_argument('--size-usd', type=float, default=1000, help='Fixed position size in USD')
    parser.add_argument('--leverage', type=float, default=20.0, help='Leverage multiplier')
    parser.add_argument('--max-positions', type=int, default=10, help='Max open positions')

    # Trailing Stop
    parser.add_argument('--trailing', action='store_true', help='Enable Trailing Stop')
    parser.add_argument('--trailing-start', type=float, default=0.1, help='Trailing start pct')
    parser.add_argument('--trailing-step', type=float, default=0.05, help='Trailing step pct')
    
    # Portfolio Trailing
    parser.add_argument('--portfolio-trailing', action='store_true', help='Enable Portfolio Trailing')
    parser.add_argument('--pt-start', type=float, default=0.30, help='Portfolio Trailing start pct')
    parser.add_argument('--pt-step', type=float, default=0.15, help='Portfolio Trailing step pct')
    parser.add_argument('--pt-cooldown', type=float, default=1.0, help='Days to cooldown after portfolio trailing stop hits')

    # Pullback
    parser.add_argument('--entry-pullback', type=float, default=0.0, help='Pullback pct for limit entry')
    parser.add_argument('--entry-timeout', type=int, default=3, help='Timeout bars for limit entry')
    parser.add_argument('--max-bars', type=int, default=10, help='Max bars to hold trade')

    # Scanner
    parser.add_argument('--use-scanner', action='store_true', help='Enable SmartScanner filtering')
    parser.add_argument('--scanner-mae', type=float, default=0.04, help='Max Adverse Excursion')
    parser.add_argument('--scanner-mfe', type=float, default=0.12, help='Max Favorable Excursion')
    parser.add_argument('--scanner-lookback', type=int, default=6, help='Lookback days for scanner')

    # Circuit Breaker
    parser.add_argument('--cb-profile', type=str, choices=['0.6', '0.65', 'none'], default='none',
                        help='Circuit Breaker optimization profile to use based on robustness insights')

    # Date range
    parser.add_argument('--start', type=str, default='2021-01-01', help='Overall start date')
    parser.add_argument('--end', type=str, default='2026-02-22', help='Overall end date')
    parser.add_argument('--timeframe', type=str, default='1d', help='Timeframe')
    parser.add_argument('--margin-mode', type=str, default='ISOLATED', choices=['ISOLATED', 'CROSS'], help='Margin mode')
    parser.add_argument('--warmup', type=int, default=0, help='Warm-up months for indicators')

    # Execution
    parser.add_argument('--parallel', type=int, default=1, help='Number of parallel workers (default: sequential)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: ml/equity_windows)')

    args = parser.parse_args()

    # ── Resolve window & step sizes ──────────────────────────────────────
    if args.window_days:
        window_delta = timedelta(days=args.window_days)
        window_label = f"{args.window_days} days"
    elif args.window_months:
        window_delta = timedelta(days=args.window_months * 30)
        window_label = f"{args.window_months} months (~{args.window_months * 30}d)"
    else:
        window_delta = timedelta(days=60)
        window_label = "60 days (default)"

    if args.step_days:
        step_delta = timedelta(days=args.step_days)
        step_label = f"{args.step_days} days"
    elif args.step_months:
        step_delta = timedelta(days=args.step_months * 30)
        step_label = f"{args.step_months} months (~{args.step_months * 30}d)"
    else:
        step_delta = timedelta(days=30)
        step_label = "30 days (default)"

    # ── Generate windows ─────────────────────────────────────────────────
    start_dt = datetime.strptime(args.start, '%Y-%m-%d')
    end_dt = datetime.strptime(args.end, '%Y-%m-%d')

    windows = []
    cursor = start_dt
    while cursor + window_delta <= end_dt:
        w_start = cursor.strftime('%Y-%m-%d')
        w_end = (cursor + window_delta).strftime('%Y-%m-%d')
        windows.append((w_start, w_end))
        cursor += step_delta

    # Include partial last window if significant (>= 50% of window size)
    if cursor < end_dt and (end_dt - cursor) >= window_delta * 0.5:
        windows.append((cursor.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')))

    if not windows:
        print("❌ No windows generated. Check --start, --end, --window-days/--step-days")
        return

    print(f"🚀 Multi-Window Equity Runner")
    print(f"   Period: {args.start} → {args.end}")
    print(f"   Window: {window_label}")
    print(f"   Step: {step_label}")
    print(f"   Windows to run: {len(windows)}")
    print(f"   Timeframe: {args.timeframe}, Leverage: {args.leverage}x")
    print(f"   Workers: {args.parallel}")
    print()

    # ── Resolve data path ────────────────────────────────────────────────
    data_path = Path(__file__).parent.parent / 'bitget-data' / 'processed' / f'features_{args.timeframe}_full.parquet'
    if not data_path.exists():
        data_path = Path(__file__).parent.parent / 'bitget-data' / 'processed' / f'features_{args.timeframe}.parquet'
    if args.data:
        data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return

    data_path_str = str(data_path)

    # ── Output directory ─────────────────────────────────────────────────
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / 'equity_windows'
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Convert args to dict for subprocess ──────────────────────────────
    args_dict = vars(args)

    # ── Run windows ──────────────────────────────────────────────────────
    all_results = []

    if args.parallel > 1:
        print(f"⚡ Running {len(windows)} windows in parallel ({args.parallel} workers)...\n")
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {}
            for idx, (w_start, w_end) in enumerate(windows):
                fut = executor.submit(
                    run_single_window, args_dict, data_path_str,
                    w_start, w_end, idx, len(windows)
                )
                futures[fut] = idx

            for fut in as_completed(futures):
                result = fut.result()
                if result is not None:
                    all_results.append((futures[fut], result))
    else:
        print(f"📊 Running {len(windows)} windows sequentially...\n")
        # Load once for sequential mode to avoid re-reading
        for idx, (w_start, w_end) in enumerate(windows):
            result = run_single_window(args_dict, data_path_str, w_start, w_end, idx, len(windows))
            if result is not None:
                all_results.append((idx, result))

    if not all_results:
        print("\n❌ No results generated!")
        return

    # Sort by window index
    all_results.sort(key=lambda x: x[0])

    # ── Save individual window CSVs ──────────────────────────────────────
    print(f"\n💾 Saving {len(all_results)} window CSVs...")
    for idx, df_eq in all_results:
        w_start = windows[idx][0]
        w_end = windows[idx][1]
        fname = f"window_{w_start}_to_{w_end}.csv"
        df_eq.to_csv(output_dir / fname, index=False)

    # ── Combine and add features ─────────────────────────────────────────
    combined = pd.concat([df_eq for _, df_eq in all_results], ignore_index=True)
    combined = compute_drawdown_features(combined)

    combined_path = output_dir / f'{args.timeframe}_all_windows_combined.csv'
    combined.to_csv(combined_path, index=False)
    print(f"💾 Combined dataset: {combined_path} ({len(combined):,} rows)")

    # ── Summary stats ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"📊 SUMMARY")
    print(f"{'='*70}")
    print(f"   Windows completed: {len(all_results)}/{len(windows)}")
    print(f"   Total rows: {len(combined):,}")
    print(f"   Date range: {combined['date'].min()} → {combined['date'].max()}")
    print()

    # Per-window summary
    window_stats = []
    for idx, df_eq in all_results:
        eq = df_eq['equity'].values
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / peak * 100
        ret = (eq[-1] / eq[0] - 1) * 100 if eq[0] > 0 else 0
        window_stats.append({
            'window': f"{windows[idx][0]} → {windows[idx][1]}",
            'trades': df_eq['num_trades'].iloc[0],
            'return_pct': ret,
            'max_dd_pct': np.max(dd),
        })

    ws_df = pd.DataFrame(window_stats)

    print(f"{'WINDOW':<30} | {'TRADES':>7} | {'RETURN':>9} | {'MAX DD':>9}")
    print("-" * 65)
    for _, row in ws_df.iterrows():
        print(f"{row['window']:<30} | {int(row['trades']):>7} | {row['return_pct']:>+8.1f}% | {row['max_dd_pct']:>8.1f}%")
    print("-" * 65)

    avg_ret = ws_df['return_pct'].mean()
    avg_dd = ws_df['max_dd_pct'].mean()
    worst_dd = ws_df['max_dd_pct'].max()
    losing_windows = len(ws_df[ws_df['return_pct'] < 0])

    print(f"\n   Avg Return per Window: {avg_ret:+.1f}%")
    print(f"   Avg Max DD per Window: {avg_dd:.1f}%")
    print(f"   Worst Max DD: {worst_dd:.1f}%")
    print(f"   Losing Windows: {losing_windows}/{len(ws_df)} ({losing_windows/len(ws_df)*100:.0f}%)")

    # ── Quick drawdown pattern preview ───────────────────────────────────
    print(f"\n{'='*70}")
    print(f"📉 DRAWDOWN PATTERN PREVIEW")
    print(f"{'='*70}")

    deep_dd = combined[combined['drawdown_pct'] > 10]
    if len(deep_dd) > 0:
        print(f"\n   Deep Drawdowns (>10%): {len(deep_dd)} occurrences")
        print(f"\n   By Day of Week:")
        dow_counts = deep_dd['day_of_week_name'].value_counts()
        for day, count in dow_counts.items():
            total_day = len(combined[combined['day_of_week_name'] == day])
            pct = count / total_day * 100 if total_day > 0 else 0
            print(f"     {day:<12}: {count:>5} ({pct:.1f}% of all {day}s)")

        print(f"\n   By Month:")
        month_counts = deep_dd['month_name'].value_counts()
        for month, count in month_counts.items():
            total_month = len(combined[combined['month_name'] == month])
            pct = count / total_month * 100 if total_month > 0 else 0
            print(f"     {month:<12}: {count:>5} ({pct:.1f}% of all {month} days)")

        print(f"\n   Avg Open Positions during deep DD: {deep_dd['open_positions_count'].mean():.1f}")
        print(f"   Avg Rolling Vol (5d) during deep DD: {deep_dd['rolling_vol_5d'].mean():.4f}")
    else:
        print("   No deep drawdowns >10% detected in the dataset.")

    print(f"\n✅ Done! Use analyze_drawdowns.py for detailed analysis:")
    print(f"   python ml/analyze_drawdowns.py --input {combined_path}")

if __name__ == '__main__':
    main()
