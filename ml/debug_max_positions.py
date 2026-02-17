#!/usr/bin/env python3
"""Debug max positions test - why are all results identical?"""
# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from backtest_3stage import ThreeStageBacktester, BacktestConfig

# Load data  
data_path = Path('..') / 'data' / 'processed' / 'features_1d_full.parquet'
df = pd.read_parquet(data_path)

# Filter to recent period
df_recent = df[(df['timestamp'] >= '2025-11-01') & (df['timestamp'] <= '2026-01-31')].copy()
print(f'Dataset: {len(df_recent):,} rows')
print(f'Symbols: {df_recent["symbol"].nunique()}')

# Count crossover signals per day
df_recent['date'] = df_recent['timestamp'].dt.date
cross_up = df_recent[df_recent['macd_cross_up'] == 1].groupby('date').size()
cross_down = df_recent[df_recent['macd_cross_down'] == 1].groupby('date').size()
total_per_day = (cross_up.add(cross_down, fill_value=0)).astype(int)

print(f'\n📊 Crossover signals per day:')
print(f'  Max: {total_per_day.max()}')
print(f'  Mean: {total_per_day.mean():.1f}')
print(f'  Days with >5 signals: {(total_per_day > 5).sum()}')
print(f'  Days with >10 signals: {(total_per_day > 10).sum()}')
print(f'\nTop 10 days with most signals:')
print(total_per_day.sort_values(ascending=False).head(10))

# Now test backtest with verbose mode
print("\n" + "="*80)
print("🔍 DEBUG: Running backtest with max_positions=7 and VERBOSE mode")
print("="*80)

config = BacktestConfig(
    initial_capital=100,
    risk_per_trade=0.01,
    entry_threshold=0.65,
    fee_rate=0.001,
    slippage=0.0005,
    leverage=5.0,
    max_open_trades=7,
    timeframe='1d'
)

backtester = ThreeStageBacktester(config)

# Patch to track open positions at each signal
original_run = backtester.run_backtest
def debug_run(df, verbose=True):
    """Debug wrapper to track position state"""
    result = original_run(df, verbose=verbose)
    
    # Analyze trades
    print(f"\n📈 Analysis of {len(result.trades)} trades:")
    
    # Group trades by entry time to see concurrent entries
    from collections import defaultdict
    trades_by_date = defaultdict(list)
    for t in result.trades:
        trades_by_date[t.entry_time.date()].append(t)
    
    # Find days with multiple entries
    multi_entry_days = [(d, len(ts)) for d, ts in trades_by_date.items() if len(ts) > 1]
    multi_entry_days.sort(key=lambda x: -x[1])
    
    print(f"\n📅 Days with multiple trade entries:")
    for d, count in multi_entry_days[:10]:
        print(f"  {d}: {count} trades")
    
    # Check max concurrent positions
    from datetime import timedelta
    all_positions = []
    for t in result.trades:
        all_positions.append((t.entry_time, 'OPEN', t.symbol))
        if t.exit_time:
            all_positions.append((t.exit_time, 'CLOSE', t.symbol))
    
    all_positions.sort()
    
    concurrent = 0
    max_concurrent = 0
    max_concurrent_time = None
    for time, action, symbol in all_positions:
        if action == 'OPEN':
            concurrent += 1
        else:
            concurrent -= 1
        if concurrent > max_concurrent:
            max_concurrent = concurrent
            max_concurrent_time = time
    
    print(f"\n🔢 Max concurrent positions: {max_concurrent}")
    print(f"   At time: {max_concurrent_time}")
    
    return result

result = debug_run(df_recent)

print("\n" + "="*80)
print("🔍 Now testing with max_positions=20")
print("="*80)

config20 = BacktestConfig(
    initial_capital=100,
    risk_per_trade=0.01,
    entry_threshold=0.65,
    fee_rate=0.001,
    slippage=0.0005,
    leverage=5.0,
    max_open_trades=20,
    timeframe='1d'
)

backtester20 = ThreeStageBacktester(config20)
result20 = debug_run(df_recent)

print("\n" + "="*80)
print("🤔 COMPARISON")
print("="*80)
print(f"Max 7 positions: {result.total_trades} trades, {result.total_return:.1%} return")
print(f"Max 20 positions: {result20.total_trades} trades, {result20.total_return:.1%} return")

if result.total_trades == result20.total_trades:
    print("\n⚠️ Same number of trades! Possible causes:")
    print("1. Not enough concurrent signals (entry_threshold=0.65 filters most)")
    print("2. Capital constraint kicks in before max positions")
    print("3. Same-symbol rule prevents multiple entries")
