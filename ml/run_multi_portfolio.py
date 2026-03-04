#!/usr/bin/env python3
"""
Multi-Portfolio Backtester
Runs multiple isolated configurations and aggregates their equity curves.
"""
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import copy

from config import get_timeframe_config
from backtest_3stage import ThreeStageBacktester, BacktestConfig, DATA_DIR, PROCESSED_DIR, BacktestResult

def run_portfolio(port_config: dict, start_date=None, end_date=None) -> BacktestResult:
    """Runs a single portfolio configuration."""
    tf_val = port_config.get('timeframe', '4h')
    
    tf_config = get_timeframe_config(tf_val)
    config = BacktestConfig(
        initial_capital=port_config.get('capital', 100),
        risk_per_trade=port_config.get('risk', 0.01),
        leverage=port_config.get('leverage', 1.0),
        margin_mode=port_config.get('margin_mode', 'ISOLATED'),
        timeframe=tf_val,
        max_bars=8,
        max_open_trades=port_config.get('max_positions', 13),
        entry_threshold=port_config.get('threshold', 0.6),
        min_refined_score=port_config.get('minscore', 0.0),
        use_scanner_filter=port_config.get('use_scanner', True),
        start_date=start_date,
        end_date=end_date,
        slippage=0.01
    )
    
    # Enable circuit breaker if configured
    if 'use_circuit_breaker' in port_config:
        config.use_circuit_breaker = port_config['use_circuit_breaker']
    
    # Load dataset
    data_path = PROCESSED_DIR / f'features_{tf_val}_full.parquet'
    if not data_path.exists():
        print(f"Error: Data file {data_path} not found for strategy {port_config.get('name', 'Unknown')}.")
        return None
        
    df = pd.read_parquet(data_path)
    df = df.sort_values('timestamp')
    
    if start_date:
        df = df[df['timestamp'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['timestamp'] <= pd.to_datetime(end_date)]
        
    if df.empty:
        print(f"Error: No data in specified date range for strategy {port_config.get('name', 'Unknown')}.")
        return None
        
    print(f"⏳ Running strategy: {port_config.get('name', 'Unnamed')} (TF: {tf_val}, Capital: ${config.initial_capital:,.2f})")
    backtester = ThreeStageBacktester(config)
    result = backtester.run_backtest(df, verbose=False)
    result.config = config  # Injected config to make it available during aggregation
    
    if len(result.equity_curve) > 0:
        print(f"✅ Completed '{port_config.get('name', 'Unnamed')}': {len(result.trades)} trades, Final Equity: ${result.equity_curve[-1]:,.2f}")
    else:
        print(f"✅ Completed '{port_config.get('name', 'Unnamed')}': 0 trades, Final Equity: ${config.initial_capital:,.2f}")
    return result

def run_multi_portfolio_aggregation(results, initial_total_capital):
    """Aggregates multiple backtest results and calculates portfolio metrics."""
    if not results:
        return None
        
    all_trades = []
    all_timestamps = set()
    
    for name, res in results.items():
        if res.timestamps:
            all_timestamps.update(res.timestamps)
        
        # Tag trades
        for t in res.trades:
            trade_copy = copy.deepcopy(t)
            trade_copy.strategy = name
            all_trades.append(trade_copy)
            
    all_timestamps = sorted(list(all_timestamps))
    if not all_timestamps:
        return None
        
    df_equity = pd.DataFrame(index=all_timestamps)
    
    for name, res in results.items():
        if len(res.timestamps) == 0:
            df_equity[name] = res.config.initial_capital
            continue
            
        s = pd.Series(res.equity_curve, index=res.timestamps)
        s = s[~s.index.duplicated(keep='last')]
        s = s.reindex(all_timestamps).ffill().bfill()
        df_equity[name] = s
        
    df_equity['Total'] = df_equity.sum(axis=1)
    
    # Metrics
    peak = df_equity['Total'].iloc[0]
    max_dd = 0
    for eq in df_equity['Total']:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        max_dd = max(max_dd, dd)
        
    final_equity = df_equity['Total'].iloc[-1]
    total_return = (final_equity - initial_total_capital) / initial_total_capital
    
    all_trades.sort(key=lambda t: t.entry_time)
    winning_trades = [t for t in all_trades if t.pnl > 0]
    win_rate = len(winning_trades) / len(all_trades) if all_trades else 0
    
    gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
    losing_trades = [t for t in all_trades if t.pnl <= 0]
    gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    return {
        'df_equity': df_equity,
        'final_equity': final_equity,
        'total_return': total_return,
        'max_dd': max_dd,
        'total_trades': len(all_trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'all_trades': all_trades
    }

def main():
    parser = argparse.ArgumentParser(description="Run Multi-Portfolio Backtest")
    parser.add_argument('--config', type=str,default="ml/test_portfolios.json", help='Path to JSON configuration file')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, help='Optional output CSV for combined portfolio equity')
    args = parser.parse_args()
    
    if not Path(args.config).exists():
        print(f"Config file {args.config} not found.")
        return
        
    with open(args.config, 'r') as f:
        portfolios = json.load(f)
        
    if not isinstance(portfolios, list):
        print("Error: Config file must contain a JSON array of configuration objects.")
        return
        
    print(f"🚀 Starting Multi-Portfolio Backtest with {len(portfolios)} strategies...")
    print("=" * 80)
    
    results = {}
    total_capital = 0.0
    
    for i, port in enumerate(portfolios):
        name = port.get('name', f"Strategy_{i+1}")
        total_capital += port.get('capital', 100)
        
        res = run_portfolio(port, args.start, args.end)
        if res:
            results[name] = res
                
    if not results:
        print("No successful runs. Exiting.")
        return
        
    print("\n" + "=" * 80)
    print("🔄 Aggregating Portfolio Data...")
    
    agg = run_multi_portfolio_aggregation(results, total_capital)
    if not agg:
        print("Aggregation failed.")
        return

    print("=" * 80)
    print("🏆 MULTI-PORTFOLIO BACKTEST RESULTS")
    print("=" * 80)
    print(f"   Initial Total Capital: ${total_capital:,.2f}")
    print(f"   Final Total Equity:    ${agg['final_equity']:,.2f}")
    print(f"   Portfolio Return:      {agg['total_return']:.2%}")
    print(f"   Portfolio Max DD:      {agg['max_dd']:.2%}")
    print(f"   Combined Total Trades: {agg['total_trades']}")
    print(f"   Combined Win Rate:     {agg['win_rate']:.1%}")
    print(f"   Combined Profit Fact:  {agg['profit_factor']:.2f}")
    print("-" * 80)
    
    print("\n   [Sub-Portfolio Breakdown]")
    for name, res in results.items():
        cap = res.config.initial_capital
        eq_final = res.equity_curve[-1] if len(res.equity_curve) > 0 else cap
        ret = (eq_final - cap) / cap
        min_score = res.config.min_refined_score
        print(f"   - {name:<20} | capital: ${cap:<7.2f} | score: {min_score:>4.1f} | trades: {len(res.trades):<4} | return: {ret:>8.2%} | maxDD: {res.max_drawdown:>6.2%} | final eq: ${eq_final:,.2f}")
        
    if args.output:
        df_equity.to_csv(args.output, index_label='timestamp')
        print(f"\n💾 Saved portfolio equity curve to {args.output}")

if __name__ == '__main__':
    main()
