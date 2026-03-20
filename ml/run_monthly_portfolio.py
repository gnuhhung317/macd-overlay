#!/usr/bin/env python3
"""
Monthly Batch Backtester for Multi-Portfolio.
Splits a date range into monthly windows and runs the multi-portfolio backtest for each.
Generates a summary report of performance per month.
"""
import json
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
import sys

# Add root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from ml.run_multi_portfolio import run_portfolio, run_multi_portfolio_aggregation

def main():
    parser = argparse.ArgumentParser(description="Run Monthly Batch Backtest for Multi-Portfolio")
    parser.add_argument('--config', type=str, default="ml/test_portfolios.json", help='Path to JSON configuration file')
    parser.add_argument('--start', type=str, required=True, help='Global Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='Global End date (YYYY-MM-DD)')
    
    # Global overrides
    parser.add_argument('--leverage', type=float, help='Override leverage')
    parser.add_argument('--threshold', type=float, help='Override threshold')
    
    args = parser.parse_args()
    
    if not Path(args.config).exists():
        print(f"Config file {args.config} not found.")
        return
        
    with open(args.config, 'r') as f:
        portfolios = json.load(f)
        
    # Apply global overrides to portfolios
    for port in portfolios:
        if args.threshold is not None: port['threshold'] = args.threshold

    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    
    # Generate monthly windows
    windows = []
    curr = start_dt
    while curr < end_dt:
        win_start = curr
        win_end = curr + relativedelta(months=1)
        if win_end > end_dt:
            win_end = end_dt
            
        windows.append((win_start, win_end))
        curr = win_end
        
    print(f"🚀 Starting Monthly Batch Backtest with {len(windows)} months...")
    print("=" * 100)
    
    monthly_results = []
    
    for ws, we in windows:
        ws_str = ws.strftime("%Y-%m-%d")
        we_str = we.strftime("%Y-%m-%d")
        print(f"\n📅 TESTING MONTH: {ws_str} to {we_str}")
        print("-" * 100)
        
        results = {}
        total_capital = 0.0
        
        for i, port in enumerate(portfolios):
            name = port.get('name', f"Strategy_{i+1}")
            total_capital += port.get('capital', 100)
            
            res = run_portfolio(port, ws_str, we_str)
            if res:
                results[name] = res
                
        if results:
            agg = run_multi_portfolio_aggregation(results, total_capital)
            if agg:
                report = {
                    'month': ws.strftime("%Y-%m"),
                    'return': agg['total_return'],
                    'max_dd': agg['max_dd'],
                    'trades': agg['total_trades'],
                    'win_rate': agg['win_rate'],
                    'pf': agg['profit_factor'],
                    'final_equity': agg['final_equity']
                }
                monthly_results.append(report)
                
                print(f"\n✅ Month Result: Final Eq: ${agg['final_equity']:,.2f} | Ret: {agg['total_return']:.2%} | MaxDD: {agg['max_dd']:.2%}")
            else:
                print(f"\n⚠️ No aggregation data for this month.")
        else:
            print(f"\n⚠️ No trades executed in this period.")

    # Print Final Summary Table
    print("\n" + "=" * 100)
    print("📋 MONTHLY PERFORMANCE SUMMARY REPORT")
    print("=" * 100)
    print(f"{'Month':<10} | {'Return':<10} | {'Max DD':<10} | {'Trades':<8} | {'Win%':<8} | {'PF':<8} | {'Final Equity':<15}")
    print("-" * 100)
    
    total_ret = 0
    total_trades = 0
    
    for r in monthly_results:
        print(f"{r['month']:<10} | {r['return']:>9.2%} | {r['max_dd']:>9.2%} | {r['trades']:>8} | {r['win_rate']:>7.1%} | {r['pf']:>7.2f} | ${r['final_equity']:>13,.2f}")
        total_ret += r['return']
        total_trades += r['trades']
        
    print("-" * 100)
    if monthly_results:
        avg_ret = total_ret / len(monthly_results)
        print(f"{'AVERAGE':<10} | {avg_ret:>9.2%} | {'-':>10} | {total_trades/len(monthly_results):>8.1f} | {'-':>8} | {'-':>8} |")
    print("=" * 100)

if __name__ == '__main__':
    main()
