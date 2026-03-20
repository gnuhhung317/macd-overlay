#!/usr/bin/env python3
"""
Multi-Portfolio Backtester (Sniper Version)
Runs multiple isolated configurations and aggregates their equity curves.
"""
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import copy
import sys
import joblib

# Add root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from ml.backtest_sniper import (
    BacktestConfig, Trade, TradeState, 
    run_portfolio_simulation, load_assets, backtest_symbol,
    SYMBOLS_DIR
)

class BacktestResult:
    def __init__(self, trades, equity_curve, timestamps, config):
        self.trades = trades
        self.equity_curve = equity_curve
        self.timestamps = timestamps
        self.config = config
        
        # Calculate max drawdown for report
        if len(equity_curve) > 0:
            equity_vals = np.array(equity_curve)
            max_equity_vals = np.maximum.accumulate(equity_vals)
            drawdowns = (max_equity_vals - equity_vals) / max_equity_vals
            self.max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
        else:
            self.max_drawdown = 0.0

def run_portfolio(port_config: dict, assets: tuple, start_date=None, end_date=None) -> BacktestResult:
    """Runs a single portfolio configuration using Sniper logic."""
    clf, features, threshold = assets
    
    # Override threshold if provided in config
    port_threshold = port_config.get('threshold', threshold)
    
    config = BacktestConfig(
        initial_capital=port_config.get('capital', 100.0),
        risk_per_trade=port_config.get('risk', 0.05),
        leverage=port_config.get('leverage', 1.0),
        max_open_trades=port_config.get('max_positions', 5),
        start_date=start_date if start_date else '2025-01-01',
        end_date=end_date,
        max_bars_hold=port_config.get('max_bars_hold', 24)
    )
    
    # Load and process symbols for this portfolio
    # (In a real scenario, we might want to pre-load or cache these)
    all_potential_signals = []
    full_price_db = {}
    
    # Filter symbols if specified in port_config
    target_symbols = port_config.get('symbols', None)
    
    all_files = list(SYMBOLS_DIR.glob("*.parquet"))
    if target_symbols:
        all_files = [f for f in all_files if any(s in f.name for s in target_symbols)]

    print(f"⏳ Running strategy: {port_config.get('name', 'Unnamed')} (Capital: ${config.initial_capital:,.2f})")
    
    for i, file_path in enumerate(all_files):
        sigs, ohlcv = backtest_symbol(file_path, features, clf, port_threshold, config)
        if sigs:
            all_potential_signals.extend(sigs)
            symbol_name = Path(file_path).stem.replace('_USDT','').replace('USDT','')
            full_price_db[symbol_name] = ohlcv
            
    if not all_potential_signals:
        print(f"⚠️ No signals found for strategy '{port_config.get('name', 'Unnamed')}'")
        return BacktestResult([], [], [], config)
        
    # Run state-machine simulation
    trades, full_equity_curve, _ = run_portfolio_simulation(all_potential_signals, full_price_db, config)
    
    # Extract timestamps and equity values from [(ts, val), ...]
    # Sniper returns a list of (timestamp, value) tuples
    timestamps = [x[0] for x in full_equity_curve] if full_equity_curve else []
    equity_values = [x[1] for x in full_equity_curve] if full_equity_curve else []
    
    result = BacktestResult(trades, equity_values, timestamps, config)
    
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
    equity_vals = df_equity['Total'].values
    peak = equity_vals[0]
    max_dd = 0
    for eq in equity_vals:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
        
    final_equity = equity_vals[-1]
    total_return = (final_equity - initial_total_capital) / initial_total_capital if initial_total_capital > 0 else 0
    
    all_trades.sort(key=lambda t: t.entry_time if t.entry_time else t.signal_time)
    winning_trades = [t for t in all_trades if t.pnl_usd > 0]
    win_rate = len(winning_trades) / len(all_trades) if all_trades else 0
    
    gross_profit = sum(t.pnl_usd for t in winning_trades) if winning_trades else 0
    losing_trades = [t for t in all_trades if t.pnl_usd <= 0]
    gross_loss = abs(sum(t.pnl_usd for t in losing_trades)) if losing_trades else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
    
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
    parser = argparse.ArgumentParser(description="Run Multi-Portfolio Sniper Backtest")
    parser.add_argument('--config', type=str, default="ml/test_portfolios.json", help='Path to JSON configuration file')
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
        
    print(f"🚀 Starting Multi-Portfolio Sniper Backtest with {len(portfolios)} strategies...")
    print("=" * 80)
    
    assets = load_assets()
    if assets[0] is None:
        print("❌ Failed to load Sniper models.")
        return
        
    results = {}
    total_capital = 0.0
    
    for i, port in enumerate(portfolios):
        name = port.get('name', f"Strategy_{i+1}")
        total_capital += port.get('capital', 100.0)
        
        res = run_portfolio(port, assets, args.start, args.end)
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
    print("🏆 MULTI-PORTFOLIO SNIPER BACKTEST RESULTS")
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
        ret = (eq_final - cap) / cap if cap > 0 else 0
        trds = len(res.trades)
        mdd = res.max_drawdown
        print(f"   - {name:<20} | capital: ${cap:<7.2f} | trades: {trds:<4} | return: {ret:>8.2%} | maxDD: {mdd:>6.2%} | final eq: ${eq_final:,.2f}")
        
    if args.output:
        agg['df_equity'].to_csv(args.output, index_label='timestamp')
        print(f"\n💾 Saved portfolio equity curve to {args.output}")

if __name__ == '__main__':
    main()
