#!/usr/bin/env python3
"""
Monthly Backtest Script
Runs the 3-Stage ML Backtester month by month and summarizes the performance.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from dateutil.relativedelta import relativedelta
from datetime import datetime

# Import backtest module
from backtest_3stage import ThreeStageBacktester, BacktestConfig, BacktestResult

def calculate_monthly_metrics(trades, initial_capital):
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'return_pct': 0.0,
            'max_dd': 0.0,
            'profit_factor': 0.0,
            'pnl': 0.0
        }
    
    winning_trades = [t for t in trades if t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl <= 0]
    
    win_rate = len(winning_trades) / len(trades) if trades else 0
    total_pnl = sum(t.pnl for t in trades)
    return_pct = (total_pnl / initial_capital)
    
    gross_profit = sum(t.pnl for t in winning_trades)
    gross_loss = abs(sum(t.pnl for t in losing_trades))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    
    # Calculate Max Drawdown based on closed trades
    sorted_trades = sorted(trades, key=lambda t: t.exit_time if t.exit_time else t.entry_time)
    equity = initial_capital
    equity_curve = [equity]
    for t in sorted_trades:
        equity += t.pnl
        equity_curve.append(equity)
        
    equity_curve = np.array(equity_curve)
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (peaks - equity_curve) / peaks
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0
    
    return {
        'total_trades': len(trades),
        'win_rate': win_rate,
        'return_pct': return_pct,
        'max_dd': max_dd,
        'profit_factor': profit_factor,
        'pnl': total_pnl
    }

def main():
    parser = argparse.ArgumentParser(description="Monthly Backtest Runner")
    
    # Matching arguments from backtest_3stage.py & plot_time_equity.py
    parser.add_argument('--data', type=str, default=None, help='Path to data file')
    parser.add_argument('--capital', type=float, default=100.0, help='Initial capital for each month')
    parser.add_argument('--risk', type=float, default=0.01, help='Risk per trade (0.01 = 1%%)')
    parser.add_argument('--threshold', type=float, default=0.65, help='Entry confidence threshold')
    parser.add_argument('--fee', type=float, default=0.001, help='Fee rate (0.001 = 0.1%%)')
    parser.add_argument('--slippage', type=float, default=0.0005, help='Slippage (0.0005 = 0.05%%)')
    parser.add_argument('--kelly', action='store_true', help='Use Kelly Criterion')
    parser.add_argument('--fixed-size', action='store_true', help='Use fixed position size')
    parser.add_argument('--size-usd', type=float, default=1000, help='Fixed position size in USD')
    parser.add_argument('--leverage', type=float, default=20.0, help='Leverage multiplier')
    parser.add_argument('--max-positions', type=int, default=10, help='Max open positions')
    parser.add_argument('--min-score', type=float, default=0.0, help='Minimum refined score threshold (0.33, 0.66, 1.0)')
    
    # Trailing Stop arguments
    parser.add_argument('--trailing', action='store_true', help='Enable Trailing Stop')
    parser.add_argument('--trailing-start', type=float, default=0.1, help='Trailing start pct')
    parser.add_argument('--trailing-step', type=float, default=0.05, help='Trailing step pct')
    
    # Pullback options
    parser.add_argument('--entry-pullback', type=float, default=0.0, help='Pullback pct for limit entry')
    parser.add_argument('--entry-timeout', type=int, default=3, help='Timeout bars for limit entry')
    parser.add_argument('--max-bars', type=int, default=10, help='Max bars to hold trade')
    
    # Scanner Filter arguments
    parser.add_argument('--use-scanner', action='store_true', help='Enable SmartScanner Entry Zone filtering')
    parser.add_argument('--scanner-mae', type=float, default=0.04, help='Max Adverse Excursion')
    parser.add_argument('--scanner-mfe', type=float, default=0.12, help='Max Favorable Excursion')
    parser.add_argument('--scanner-lookback', type=int, default=6, help='Lookback days for scanner entry')
    
    parser.add_argument("--start", type=str, default='2025-01-01', help="Analysis start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default='2026-02-23', help="Analysis end date (YYYY-MM-DD)")
    parser.add_argument("--timeframe", type=str, default='1d', help="Timeframe (1d, 4h, etc.)")
    parser.add_argument("--margin-mode", type=str, default='ISOLATED', choices=['ISOLATED', 'CROSS'], help="Margin mode")
    parser.add_argument("--warmup", type=int, default=0, help="Warm-up months")
    
    args = parser.parse_args()
    
    print(f"🚀 Monthly Backtest Runner")
    print(f"   Period: {args.start} to {args.end}")
    print(f"   Timeframe: {args.timeframe}")
    print(f"   Leverage: {args.leverage}x")
    print(f"   Initial Capital per Month: ${args.capital:,.2f}")
    
    # Load data
    data_path = Path(__file__).parent.parent / 'bitget-data' / 'processed' / f'features_{args.timeframe}_full.parquet'
    if not data_path.exists():
        data_path = Path(__file__).parent.parent / 'bitget-data' / 'processed' / f'features_{args.timeframe}.parquet'
        
    if args.data:
        data_path = Path(args.data)
        
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
        
    print(f"\n📂 Loading data...")
    df = pd.read_parquet(data_path)
    
    start_dt = pd.to_datetime(args.start)
    end_dt = pd.to_datetime(args.end)
    
    # Generate month intervals
    months = []
    current_dt = start_dt.replace(day=1)
    while current_dt <= end_dt:
        month_end = current_dt + relativedelta(months=1)
        months.append((current_dt, month_end))
        current_dt = month_end
        
    config = BacktestConfig(
        initial_capital=args.capital,
        risk_per_trade=args.risk,
        entry_threshold=args.threshold,
        fee_rate=args.fee,
        slippage=args.slippage,
        leverage=args.leverage,
        timeframe=args.timeframe,
        margin_mode=args.margin_mode,
        use_kelly=args.kelly,
        fixed_position_size=args.fixed_size,
        position_size_usd=args.size_usd,
        max_open_trades=args.max_positions,
        require_fresh_crossover_after_exit=True,
        use_trailing_stop=args.trailing,
        trailing_start_pct=args.trailing_start,
        trailing_step_pct=args.trailing_step,
        entry_pullback_pct=args.entry_pullback,
        entry_pullback_timeout=args.entry_timeout,
        max_bars=args.max_bars,
        use_scanner_filter=args.use_scanner,
        scanner_mae=args.scanner_mae,
        scanner_mfe=args.scanner_mfe,
        scanner_lookback_days=args.scanner_lookback,
        min_refined_score=args.min_score
    )
    
    print(f"\n" + "="*85)
    print(f"{'MONTH':<12} | {'TRADES':>8} | {'WIN%':>8} | {'RETURN':>10} | {'PNL ($)':>10} | {'MAX DD':>8} | {'PF':>8}")
    print("-" * 85)
    
    total_months = 0
    positive_months = 0
    all_monthly_returns = []
    all_trades = []
    
    # Instantiate once to avoid model loading prints on every month loop
    backtester = ThreeStageBacktester(config)
    
    for month_start, month_end in months:
        # Buffer of ~30 days so trades opened at the end of the month can complete
        eval_end_dt = month_end + pd.Timedelta(days=30)
        
        df_month = df[
            (df['timestamp'] >= month_start) & 
            (df['timestamp'] < eval_end_dt)
        ].copy()
        
        if df_month.empty:
            continue
            
        result = backtester.run_backtest(df_month, verbose=False)
        
        # Filter trades that were actually entered in this month
        month_actual_end = min(month_end, end_dt) # clamp to overall end_dt
        valid_trades = [
            t for t in result.trades 
            if pd.to_datetime(t.entry_time).tz_localize(None) >= month_start and pd.to_datetime(t.entry_time).tz_localize(None) < month_actual_end
        ]
        
        metrics = calculate_monthly_metrics(valid_trades, args.capital)
        all_trades.extend(valid_trades)
        
        month_label = month_start.strftime("%Y-%m")
        trd = metrics['total_trades']
        win = metrics['win_rate'] * 100
        ret = metrics['return_pct'] * 100
        pnl = metrics['pnl']
        mdd = metrics['max_dd'] * 100
        pf = metrics['profit_factor']
        
        print(f"{month_label:<12} | {trd:>8} | {win:>7.1f}% | {ret:>9.1f}% | {pnl:>10.2f} | {mdd:>7.1f}% | {pf:>8.2f}")
        
        if trd > 0:
            total_months += 1
            all_monthly_returns.append(metrics['return_pct'])
            if pnl > 0:
                positive_months += 1
                
    print("=" * 85)
    
    if all_trades:
        global_metrics = calculate_monthly_metrics(all_trades, args.capital)
        avg_monthly_return = np.mean(all_monthly_returns) * 100 if all_monthly_returns else 0
        
        print(f"\n📊 GLOBAL SUMMARY:")
        print(f"   Total Months Traded: {total_months}")
        print(f"   Positive Months: {positive_months} ({positive_months/total_months*100:.1f}%)" if total_months > 0 else "")
        print(f"   Avg Monthly Return: {avg_monthly_return:.2f}%")
        print(f"   Total Trades: {global_metrics['total_trades']}")
        print(f"   Global Win Rate: {global_metrics['win_rate']*100:.1f}%")
        print(f"   Global Profit Factor: {global_metrics['profit_factor']:.2f}")
    else:
        print("\n⚠️ No trades were executed in the given period.")

if __name__ == '__main__':
    main()
