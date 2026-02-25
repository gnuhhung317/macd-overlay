#!/usr/bin/env python3
"""
Monthly Rebalanced Multi-Portfolio Backtester.
Simulates a strategy where all portfolios share PnL and rebalance capital every month.
All open positions are effectively closed at the end of each month.
"""
import json
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
import sys
import copy

# Add root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from ml.run_multi_portfolio import run_portfolio, run_multi_portfolio_aggregation

def main():
    parser = argparse.ArgumentParser(description="Monthly Rebalanced Multi-Portfolio Backtest")
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
        
    # Initial setup
    num_strategies = len(portfolios)
    initial_total_capital = sum(p.get('capital', 100) for p in portfolios)
    
    # Apply overrides
    for port in portfolios:
        if args.leverage is not None: port['leverage'] = args.leverage
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
        
    print(f"🔄 Starting REBALANCED Multi-Portfolio Backtest")
    print(f"📊 {num_strategies} strategies | Initial Total: ${initial_total_capital:,.2f}")
    print("=" * 100)
    
    # Variables to track rebalanced state
    current_capitals = [p.get('capital', 100) for p in portfolios]
    all_combined_trades = []
    monthly_snapshots = []
    
    total_rebalanced_equity = initial_total_capital

    for idx, (ws, we) in enumerate(windows):
        ws_str = ws.strftime("%Y-%m-%d")
        we_str = we.strftime("%Y-%m-%d")
        print(f"\n📅 MONTH {idx+1}: {ws_str} to {we_str}")
        print(f"💰 Starting Month with Total Equity: ${total_rebalanced_equity:,.2f} (${total_rebalanced_equity/num_strategies:,.2f} each)")
        print("-" * 100)
        
        current_month_results = {}
        
        # Run each portfolio with CURRENT rebalanced capital
        for i, port in enumerate(portfolios):
            # Create a localized config with rebalanced capital
            local_port = copy.deepcopy(port)
            local_port['capital'] = current_capitals[i]
            
            res = run_portfolio(local_port, ws_str, we_str)
            if res:
                current_month_results[local_port.get('name', f"Strat_{i}")] = res
                for t in res.trades:
                    t_copy = copy.deepcopy(t)
                    t_copy.strategy = local_port.get('name', f"Strat_{i}")
                    all_combined_trades.append(t_copy)
        
        # Calculate End of Month Equity
        if current_month_results:
            eom_total_equity = 0
            for name, res in current_month_results.items():
                if len(res.equity_curve) > 0:
                    eom_total_equity += res.equity_curve[-1]
                else:
                    # No trades, keep the allocated capital
                    # Actually we need to know WHICH portfolio it was to get its start cap
                    # Find capital from current_capitals using index i
                    # Let's map it better
                    pass
            
            # Since some might not have results, we need to handle them
            # A more robust way is to iterate strategies
            eom_total_equity = 0
            for i, port in enumerate(portfolios):
                name = port.get('name', f"Strategy_{i+1}")
                if name in current_month_results:
                    res = current_month_results[name]
                    eom_total_equity += res.equity_curve[-1] if len(res.equity_curve) > 0 else current_capitals[i]
                else:
                    eom_total_equity += current_capitals[i]

            prev_equity = total_rebalanced_equity
            total_rebalanced_equity = eom_total_equity
            month_ret = (total_rebalanced_equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            
            # REBALANCE FOR NEXT MONTH
            new_allocation = total_rebalanced_equity / num_strategies
            current_capitals = [new_allocation] * num_strategies
            
            monthly_snapshots.append({
                'month': ws.strftime("%Y-%m"),
                'equity': total_rebalanced_equity,
                'return': month_ret
            })
            
            print(f"\n⚖️ REBALANCE COMPLETE: End Eq: ${total_rebalanced_equity:,.2f} | Month Ret: {month_ret:.2%}")
        else:
            print("\n⚠️ No trades this month. Capital remains unchanged.")

    # Final Summary
    print("\n" + "=" * 100)
    print("🏆 FINAL REBALANCED PORTFOLIO SUMMARY")
    print("=" * 100)
    print(f"   Initial Capital: ${initial_total_capital:,.2f}")
    print(f"   Final Equity:    ${total_rebalanced_equity:,.2f}")
    print(f"   Total Return:    {(total_rebalanced_equity - initial_total_capital) / initial_total_capital:.2%}")
    print(f"   Total Trades:    {len(all_combined_trades)}")
    
    if all_combined_trades:
        win_rate = len([t for t in all_combined_trades if t.pnl > 0]) / len(all_combined_trades)
        print(f"   Win Rate:        {win_rate:.1%}")
    
    print("\n   [Monthly Snapshot]")
    for s in monthly_snapshots:
        print(f"   - {s['month']}: Equity: ${s['equity']:>12,.2f} | Return: {s['return']:>8.2%}")
    
    print("\n⚖️ This simulation assumes that at the end of each month, all profits/losses are pooled")
    print("   and redistributed equally among all strategy 'slots' to reset the risk exposure.")
    print("=" * 100)

if __name__ == '__main__':
    main()
