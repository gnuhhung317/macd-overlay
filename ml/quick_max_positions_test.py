#!/usr/bin/env python3
"""
Quick Max Positions Test
Test different max positions with a smaller dataset for faster results
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from backtest_3stage import ThreeStageBacktester, BacktestConfig

def quick_max_positions_test():
    """Quick test of max positions with smaller dataset."""
    
    # Load data
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'features_1d_full.parquet'
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    print("📈 Quick Max Positions Test")
    df = pd.read_parquet(data_path)
    
    # Use only recent 3 months for speed
    df_recent = df.tail(10000).copy()  # Last 10K rows
    print(f"✓ Using {len(df_recent):,} rows for testing")
    print(f"Date range: {df_recent['timestamp'].min()} to {df_recent['timestamp'].max()}")
    
    # Test scenarios
    max_positions = [7, 10, 15, 20]
    results = {}
    
    base_config = BacktestConfig(
        initial_capital=100,
        risk_per_trade=0.01,
        entry_threshold=0.65,
        fee_rate=0.001,
        slippage=0.0005,
        leverage=5.0,
        timeframe='1d'
    )
    
    print(f"\n🧪 Testing scenarios...")
    print(f"Configuration: {base_config.leverage}x leverage, {base_config.risk_per_trade:.1%} risk")
    
    for max_pos in max_positions:
        config = BacktestConfig(
            initial_capital=base_config.initial_capital,
            risk_per_trade=base_config.risk_per_trade,
            entry_threshold=base_config.entry_threshold,
            fee_rate=base_config.fee_rate,
            slippage=base_config.slippage,
            leverage=base_config.leverage,
            max_open_trades=max_pos,
            timeframe=base_config.timeframe
        )
        
        print(f"\n🔄 Testing {max_pos} max positions...")
        backtester = ThreeStageBacktester(config)
        result = backtester.run_backtest(df_recent, verbose=False)
        results[max_pos] = result
        
        # Count problematic trades
        liquidations = sum(1 for t in result.trades if t.exit_reason == 'LIQUIDATED')
        extreme_trades = sum(1 for t in result.trades if abs(t.pnl_pct) > 5)  # >500%
        
        print(f"   ✅ Trades: {result.total_trades:,}")
        print(f"   📈 Return: {result.total_return:.1%}")
        print(f"   📉 Max DD: {result.max_drawdown:.1%}")
        print(f"   ⚡ Liquidations: {liquidations}")
        print(f"   🚨 Extreme PnL: {extreme_trades}")
    
    # Summary table
    print("\n" + "="*80)
    print(" 📊 QUICK COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Max Pos':<8} {'Trades':>8} {'Return':>10} {'Max DD':>8} {'Sharpe':>8} {'Liq':>5}")
    print("-"*60)
    
    for max_pos in max_positions:
        res = results[max_pos]
        liquidations = sum(1 for t in res.trades if t.exit_reason == 'LIQUIDATED')
        print(f"{max_pos:<8} {res.total_trades:>8} {res.total_return:>9.1%} {res.max_drawdown:>7.1%} {res.sharpe_ratio:>7.2f} {liquidations:>5}")
    
    # Best performer analysis
    print("\n💡 Analysis:")
    best_return = max(results.values(), key=lambda r: r.total_return)
    best_sharpe = max(results.values(), key=lambda r: r.sharpe_ratio)
    lowest_dd = min(results.values(), key=lambda r: r.max_drawdown)
    
    best_return_pos = [k for k, v in results.items() if v == best_return][0]
    best_sharpe_pos = [k for k, v in results.items() if v == best_sharpe][0]
    lowest_dd_pos = [k for k, v in results.items() if v == lowest_dd][0]
    
    print(f"   🏆 Best Return: {best_return_pos} positions ({best_return.total_return:.1%})")
    print(f"   📈 Best Sharpe: {best_sharpe_pos} positions ({best_sharpe.sharpe_ratio:.2f})")
    print(f"   🛡️ Lowest DD: {lowest_dd_pos} positions ({lowest_dd.max_drawdown:.1%})")
    
    # Risk-Return analysis
    print(f"\n🎯 Risk-Return Ratio (Return/DD):")
    for max_pos in max_positions:
        res = results[max_pos]
        risk_return = res.total_return / res.max_drawdown if res.max_drawdown > 0 else float('inf')
        print(f"   {max_pos} positions: {risk_return:.2f}x")

if __name__ == '__main__':
    quick_max_positions_test()