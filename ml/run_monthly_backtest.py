#!/usr/bin/env python3
"""
Monthly Backtest Script
Runs the 3-Stage ML Backtester month by month and summarizes the performance.
"""

import joblib
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from dateutil.relativedelta import relativedelta
from datetime import datetime

# Import backtest module
from backtest_sniper import BacktestConfig, Trade, TradeState, run_portfolio_simulation, load_assets, backtest_symbol

def calculate_monthly_metrics(trades, initial_capital, equity_curve=None):
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'return_pct': 0.0,
            'max_dd': 0.0,
            'profit_factor': 0.0,
            'pnl': 0.0,
            'min_balance': initial_capital
        }
    
    winning_trades = [t for t in trades if t.pnl_usd > 0]
    losing_trades = [t for t in trades if t.pnl_usd <= 0]
    
    win_rate = len(winning_trades) / len(trades) if trades else 0
    total_pnl = sum(t.pnl_usd for t in trades)
    return_pct = (total_pnl / initial_capital)
    
    gross_profit = sum(t.pnl_usd for t in winning_trades)
    gross_loss = abs(sum(t.pnl_usd for t in losing_trades))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
    
    # Calculate Max Drawdown and Min Balance
    if equity_curve is not None and len(equity_curve) > 0:
        # Use high-fidelity equity curve if available
        equity_vals = np.array([e[1] for e in equity_curve])
        max_equity_vals = np.maximum.accumulate(equity_vals)
        drawdowns = (max_equity_vals - equity_vals) / max_equity_vals
        max_dd_pct = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
        min_balance = np.min(equity_vals) if len(equity_vals) > 0 else initial_capital
    else:
        # Fallback to closed trades analysis
        def sort_key(t):
            t_time = t.exit_time if t.exit_time is not None else t.entry_time
            if t_time is None:
                return datetime.min
            if hasattr(t_time, 'tz_localize'):
                return t_time.tz_localize(None)
            if isinstance(t_time, str):
                return pd.to_datetime(t_time).tz_localize(None)
            return t_time

        sorted_trades = sorted(trades, key=sort_key)
        equity = initial_capital
        max_equity = initial_capital
        min_balance = initial_capital
        max_dd_pct = 0.0
        
        for t in sorted_trades:
            equity += t.pnl_usd
            max_equity = max(max_equity, equity)
            min_balance = min(min_balance, equity)
            dd = (max_equity - equity) / max_equity if max_equity > 0 else 0
            max_dd_pct = max(max_dd_pct, dd)
    
    return {
        'total_trades': len(trades),
        'win_rate': win_rate,
        'return_pct': return_pct,
        'max_dd': max_dd_pct,
        'profit_factor': profit_factor,
        'pnl': total_pnl,
        'min_balance': min_balance
    }

def main():
    parser = argparse.ArgumentParser(description="Monthly Sniper Backtest Runner")
    parser.add_argument('--capital', type=float, default=100.0, help='Initial capital for each month')
    parser.add_argument('--risk', type=float, default=0.05, help='Risk per trade (0.05 = 5%%)')
    parser.add_argument('--leverage', type=float, default=20.0, help='Leverage multiplier')
    parser.add_argument('--max-positions', type=int, default=10, help='Max open positions')
    parser.add_argument("--start", type=str, default='2025-10-01', help="Analysis start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default='2026-03-01', help="Analysis end date (YYYY-MM-DD)")
    parser.add_argument("--warmup-bars", type=int, default=1000, help="Warm-up bars")
    
    args = parser.parse_args()
    
    print(f"🚀 Monthly Sniper Backtest Runner")
    print(f"   Period: {args.start} to {args.end}")
    print(f"   Leverage: {args.leverage}x | Risk: {args.risk*100}%")
    print(f"   Initial Capital per Month: ${args.capital:,.2f}")
    
    clf, features, threshold = load_assets()
    if clf is None: return

    config = BacktestConfig(
        initial_capital=args.capital,
        risk_per_trade=args.risk,
        leverage=args.leverage,
        max_open_trades=args.max_positions,
        start_date=args.start,
        end_date=args.end
    )
    
    # 1. Scan all symbols for context + signals (with Caching)
    base_dir = Path(__file__).parent.parent
    cache_path = base_dir / "ml" / "potential_signals_cache.joblib"
    symbols_dir = base_dir / "data" / "processed" / "symbols_v3"
    all_files = list(symbols_dir.glob("*.parquet"))
    
    all_potential_signals = []
    full_price_db = {}
    
    use_cache = False
    if cache_path.exists():
        cache_data = joblib.load(cache_path)
        if cache_data.get('start') == args.start and cache_data.get('end') == args.end:
            print(f"✅ Loading {len(cache_data['signals'])} signals from cache...")
            all_potential_signals = cache_data['signals']
            full_price_db = cache_data['price_db']
            use_cache = True
            
    if not use_cache:
        print(f"\n📂 Scanning {len(all_files)} symbols for potential signals...")
        for i, file_path in enumerate(all_files):
            if i % 100 == 0: print(f"Processing: {i}/{len(all_files)}...")
            sigs, ohlcv = backtest_symbol(file_path, features, clf, threshold, config)
            if sigs:
                all_potential_signals.extend(sigs)
                full_price_db[Path(file_path).stem.replace('_USDT','').replace('USDT','')] = ohlcv
        
        # Save cache
        joblib.dump({
            'start': args.start, 'end': args.end,
            'signals': all_potential_signals,
            'price_db': full_price_db
        }, cache_path)
        print(f"💾 Cache saved to {cache_path}")

    start_dt = pd.to_datetime(args.start)
    end_dt = pd.to_datetime(args.end)
    
    # Generate month intervals
    months = []
    current_dt = start_dt.replace(day=1)
    while current_dt < end_dt:
        month_end = current_dt + relativedelta(months=1)
        months.append((current_dt, month_end))
        current_dt = month_end
        
    print(f"\n" + "="*98)
    print(f"{'MONTH':<12} | {'TRADES':>8} | {'WIN%':>8} | {'RETURN':>10} | {'PNL ($)':>10} | {'MAX DD':>8} | {'MIN BAL':>8} | {'PF':>8}")
    print("-" * 98)
    
    total_months = 0
    positive_months = 0
    all_monthly_returns = []
    all_trades = []

    for month_start, month_end in months:
        # Filter signals for this month
        month_signals = [
            s for s in all_potential_signals 
            if s['timestamp'] >= month_start and s['timestamp'] < month_end
        ]
        
        if not month_signals:
            continue
            
        # Run state-machine simulation for this month's signals
        trades, equity_curve, _ = run_portfolio_simulation(month_signals, full_price_db, config)
        
        if not trades:
            continue
            
        metrics = calculate_monthly_metrics(trades, args.capital, equity_curve)
        all_trades.extend(trades)
        
        month_label = month_start.strftime("%Y-%m")
        trd = metrics['total_trades']
        win = metrics['win_rate'] * 100
        ret = metrics['return_pct'] * 100
        pnl = metrics['pnl']
        mdd = metrics['max_dd'] * 100
        mbal = metrics['min_balance']
        pf = metrics['profit_factor']
        
        print(f"{month_label:<12} | {trd:>8} | {win:>7.1f}% | {ret:>9.1f}% | {pnl:>10.2f} | {mdd:>7.1f}% | {mbal:>8.2f} | {pf:>8.2f}")
        
        if trd > 0:
            total_months += 1
            all_monthly_returns.append(metrics['return_pct'])
            if pnl > 0:
                positive_months += 1
                
    print("=" * 98)
    
    if all_trades:
        global_metrics = calculate_monthly_metrics(all_trades, args.capital)
        avg_monthly_return = np.mean(all_monthly_returns) * 100 if all_monthly_returns else 0
        
        print(f"\n📊 GLOBAL SUMMARY:")
        print(f"   Total Months Traded: {total_months}")
        print(f"   Positive Months: {positive_months} ({positive_months/total_months*100:.1f}%)" if total_months > 0 else "")
        print(f"   Avg Monthly Return: {avg_monthly_return:.2f}%")
        print(f"   Absolute Min Balance: ${global_metrics['min_balance']:.2f}")
        print(f"   Total Trades: {global_metrics['total_trades']}")
        print(f"   Global Win Rate: {global_metrics['win_rate']*100:.1f}%")
        print(f"   Global Profit Factor: {global_metrics['profit_factor']:.2f}")
    else:
        print("\n⚠️ No trades were executed in the given period.")

if __name__ == '__main__':
    main()
