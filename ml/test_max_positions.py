#!/usr/bin/env python3
"""
Test Max Open Positions Comparison
Compare impact of different max_open_trades settings: 7, 10, 15, 20
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import List, Dict, Optional

# Import backtest module
from backtest_3stage import ThreeStageBacktester, BacktestConfig
from plot_time_equity import create_time_based_equity_mtm, create_benchmark_data

def run_max_positions_comparison():
    """Compare different max_open_trades settings."""
    
    # Configuration
    backtest_start = '2025-11-01'  # Analysis period
    backtest_end = '2026-01-31'
    warm_up_months = 6  # Months before analysis start for indicators
    timeframe = '1d'
    leverage = 5
    initial_capital = 100
    
    # Test scenarios
    max_positions_tests = [7, 10, 15, 20]
    
    print(f"🧪 Testing Max Open Positions Impact")
    print(f"   Analysis Period: {backtest_start} to {backtest_end}")
    print(f"   Warm-up Period: {warm_up_months} months") 
    print(f"   Timeframe: {timeframe}")
    print(f"   Leverage: {leverage}x")
    print(f"   Capital: ${initial_capital:,.2f}")
    print(f"   Testing Max Positions: {max_positions_tests}")
    
    # Load data
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'features_1d_full.parquet'
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    print(f"\n📖 Loading data from: {data_path}")
    df = pd.read_parquet(data_path)
    
    # Convert timestamp and filter
    if 'timestamp' not in df.columns:
        print("❌ No 'timestamp' column found!")
        return
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    # Create date objects
    backtest_start_dt = pd.to_datetime(backtest_start, utc=True)
    backtest_end_dt = pd.to_datetime(backtest_end, utc=True)
    warm_up_start_dt = backtest_start_dt - pd.DateOffset(months=warm_up_months)
    
    # Filter data with warm-up
    df_with_warmup = df[
        (df['timestamp'] >= warm_up_start_dt) & 
        (df['timestamp'] <= backtest_end_dt)
    ].copy()
    
    print(f"   Total data with warm-up: {len(df_with_warmup):,} rows")
    
    if df_with_warmup.empty:
        print("❌ No data found for the specified period!")
        return
    
    # Prepare price data for mark-to-market
    price_columns = ['timestamp', 'close']
    if 'open' in df_with_warmup.columns:
        price_columns.extend(['open', 'high', 'low'])
    price_data = df_with_warmup[price_columns].copy()
    
    # Run tests for different max_open_trades
    results = {}
    
    for max_pos in max_positions_tests:
        print(f"\n🔄 Testing Max Positions = {max_pos}...")
        
        # Configure backtest
        config = BacktestConfig(
            initial_capital=initial_capital,
            risk_per_trade=0.01,  # 1% risk per trade
            entry_threshold=0.65,
            leverage=leverage,
            timeframe=timeframe,
            max_open_trades=max_pos,  # Key parameter we're testing
            require_fresh_crossover_after_exit=False
        )
        
        # Create backtester
        backtester = ThreeStageBacktester(config)
        
        # Run backtest
        result = backtester.run_backtest(df_with_warmup, verbose=False)
        
        if not result.trades:
            print(f"   ❌ No trades found for max_positions = {max_pos}")
            continue
        
        # Create time-based equity curve with mark-to-market
        daily_equity_df = create_time_based_equity_mtm(
            result.trades, backtest_start_dt, backtest_end_dt, initial_capital, price_data
        )
        
        # Save results
        results[f"{max_pos}_pos"] = {
            'config': config,
            'result': result,
            'daily_equity': daily_equity_df
        }
        
        # Print summary
        final_equity = daily_equity_df['equity'].iloc[-1]
        realized_equity = daily_equity_df['realized_equity'].iloc[-1]
        max_floating_loss = daily_equity_df['floating_pnl'].min()
        max_floating_gain = daily_equity_df['floating_pnl'].max()
        max_open_positions = daily_equity_df['open_positions_count'].max()
        
        print(f"   ✅ Completed:")
        print(f"      Total Trades: {len(result.trades)}")
        print(f"      Final Equity: ${final_equity:,.2f}")
        print(f"      Final Realized: ${realized_equity:,.2f}")
        print(f"      Total Return: {(final_equity/initial_capital - 1)*100:.1f}%")
        print(f"      Max Floating Loss: ${max_floating_loss:,.2f}")
        print(f"      Max Floating Gain: ${max_floating_gain:,.2f}")
        print(f"      Max Open Positions: {max_open_positions}")
    
    # Create benchmark
    print(f"\n📊 Creating benchmark data...")
    benchmark_df = create_benchmark_data(price_data, backtest_start_dt, backtest_end_dt, initial_capital)
    
    # Plot comparison
    print(f"\n📈 Creating comparison plots...")
    plot_max_positions_comparison(results, benchmark_df)
    
    # Print final comparison table
    print_comparison_summary(results)
    
    # Save detailed results
    save_path = Path(__file__).parent / 'results'
    save_path.mkdir(exist_ok=True)
    
    for name, data in results.items():
        csv_path = save_path / f"max_positions_{name}_time_equity.csv"
        data['daily_equity'].to_csv(csv_path, index=False)
        print(f"📁 Saved: {csv_path}")

def plot_max_positions_comparison(results: Dict, benchmark_df: pd.DataFrame = None):
    """Plot comprehensive comparison of max positions impact."""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    colors = {
        '7_pos': '#E74C3C',   # Red
        '10_pos': '#3498DB',  # Blue  
        '15_pos': '#2ECC71',  # Green
        '20_pos': '#F39C12'   # Orange
    }
    
    # 1. Daily Equity Curves
    ax1 = axes[0, 0]
    for name, data in results.items():
        daily_df = data['daily_equity']
        ax1.plot(daily_df['date'], daily_df['equity'], 
                label=f"Max {name.replace('_pos', '')} positions", 
                linewidth=2.5, color=colors.get(name, '#333'))
    
    if benchmark_df is not None and not benchmark_df.empty:
        ax1.plot(benchmark_df['date'], benchmark_df['benchmark_equity'],
                label='Buy & Hold', linewidth=2, color='gray', linestyle='--', alpha=0.7)
    
    ax1.set_title('Equity Curves by Max Positions', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Equity ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    # 2. Floating PnL Range
    ax2 = axes[0, 1]
    for name, data in results.items():
        daily_df = data['daily_equity']
        if 'floating_pnl' in daily_df.columns:
            ax2.plot(daily_df['date'], daily_df['floating_pnl'], 
                    label=f"Max {name.replace('_pos', '')}", 
                    linewidth=2, color=colors.get(name, '#333'))
    
    ax2.set_title('Floating PnL by Max Positions', fontweight='bold', fontsize=14) 
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Floating PnL ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    # 3. Drawdown Comparison
    ax3 = axes[0, 2]
    for name, data in results.items():
        daily_df = data['daily_equity']
        equity_values = daily_df['equity'].values
        peak = np.maximum.accumulate(equity_values)
        drawdown = (peak - equity_values) / peak * 100
        ax3.plot(daily_df['date'], drawdown, 
                label=f"Max {name.replace('_pos', '')}", 
                linewidth=2, color=colors.get(name, '#333'))
    
    ax3.set_title('Drawdown % by Max Positions', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Drawdown (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    # 4. Active Positions Count
    ax4 = axes[1, 0]
    for name, data in results.items():
        daily_df = data['daily_equity']
        if 'open_positions_count' in daily_df.columns:
            ax4.plot(daily_df['date'], daily_df['open_positions_count'],
                    label=f"Max {name.replace('_pos', '')}", 
                    linewidth=2, color=colors.get(name, '#333'))
    
    ax4.set_title('Active Positions Count', fontweight='bold', fontsize=14)
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Open Positions')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    # 5. Performance Metrics Bar Charts
    ax5 = axes[1, 1]
    metrics = {}
    for name, data in results.items():
        daily_df = data['daily_equity']
        initial_equity = daily_df['equity'].iloc[0]
        final_equity = daily_df['equity'].iloc[-1]
        total_return = (final_equity / initial_equity - 1) * 100
        
        max_floating_loss = daily_df['floating_pnl'].min()
        max_floating_gain = daily_df['floating_pnl'].max()
        
        metrics[name] = {
            'Return %': total_return,
            'Max Float Loss': max_floating_loss,
            'Max Float Gain': max_floating_gain
        }
    
    positions = list(results.keys())
    returns = [metrics[p]['Return %'] for p in positions]
    
    bars = ax5.bar([p.replace('_pos', '') for p in positions], returns, 
                  color=[colors.get(p, '#333') for p in positions], alpha=0.8)
    ax5.set_title('Total Return % by Max Positions', fontweight='bold', fontsize=14)
    ax5.set_xlabel('Max Positions')
    ax5.set_ylabel('Total Return (%)')
    ax5.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, ret in zip(bars, returns):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                f'{ret:,.0f}%', ha='center', fontweight='bold')
    
    # 6. Risk Metrics 
    ax6 = axes[1, 2]
    max_losses = [metrics[p]['Max Float Loss'] for p in positions]
    
    bars = ax6.bar([p.replace('_pos', '') for p in positions], max_losses,
                  color=[colors.get(p, '#333') for p in positions], alpha=0.8)
    ax6.set_title('Max Floating Loss by Max Positions', fontweight='bold', fontsize=14)
    ax6.set_xlabel('Max Positions')  
    ax6.set_ylabel('Max Floating Loss ($)')
    ax6.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, loss in zip(bars, max_losses):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 10000,
                f'${loss:,.0f}', ha='center', fontweight='bold', color='white')
    
    plt.tight_layout()
    
    # Save plot
    save_path = Path(__file__).parent / 'results' / 'max_positions_comparison.png'
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Plot saved: {save_path}")
    
    plt.show()

def print_comparison_summary(results: Dict):
    """Print detailed comparison summary."""
    
    print("\n" + "="*100)
    print("📊 MAX OPEN POSITIONS COMPARISON SUMMARY")
    print("="*100)
    
    print(f"\n{'Max Pos':<8} {'Trades':<8} {'Final $':<12} {'Return %':<10} {'Max Float Loss':<15} {'Max Float Gain':<15} {'Max Open':<8}")
    print("-" * 100)
    
    for name, data in results.items():
        result = data['result']
        daily_df = data['daily_equity']
        
        max_pos = name.replace('_pos', '')
        trades_count = len(result.trades)
        final_equity = daily_df['equity'].iloc[-1]
        initial_equity = daily_df['equity'].iloc[0]
        total_return = (final_equity / initial_equity - 1) * 100
        
        max_floating_loss = daily_df['floating_pnl'].min()
        max_floating_gain = daily_df['floating_pnl'].max()
        max_open_positions = daily_df['open_positions_count'].max()
        
        print(f"{max_pos:<8} {trades_count:<8} ${final_equity:<11,.0f} {total_return:<9.1f}% "
              f"${max_floating_loss:<14,.0f} ${max_floating_gain:<14,.0f} {max_open_positions:<8}")
    
    print("\n" + "="*100)
    print("💡 KEY INSIGHTS:")
    
    # Find best performer
    best_return = 0
    best_name = ""
    lowest_risk = float('inf')
    lowest_risk_name = ""
    
    for name, data in results.items():
        daily_df = data['daily_equity']
        final_equity = daily_df['equity'].iloc[-1]
        initial_equity = daily_df['equity'].iloc[0]
        total_return = (final_equity / initial_equity - 1) * 100
        max_floating_loss = abs(daily_df['floating_pnl'].min())
        
        if total_return > best_return:
            best_return = total_return
            best_name = name.replace('_pos', '')
            
        if max_floating_loss < lowest_risk:
            lowest_risk = max_floating_loss
            lowest_risk_name = name.replace('_pos', '')
    
    print(f"   🏆 Best Return: Max {best_name} positions with {best_return:.1f}%")
    print(f"   🛡️ Lowest Risk: Max {lowest_risk_name} positions with ${lowest_risk:,.0f} max floating loss")
    
    # Risk-Return Analysis
    print(f"\n📈 RISK-RETURN ANALYSIS:")
    for name, data in results.items():
        daily_df = data['daily_equity']
        final_equity = daily_df['equity'].iloc[-1]
        initial_equity = daily_df['equity'].iloc[0]
        total_return = (final_equity / initial_equity - 1) * 100
        max_floating_loss = abs(daily_df['floating_pnl'].min())
        
        risk_return_ratio = total_return / (max_floating_loss / initial_equity) if max_floating_loss > 0 else float('inf')
        
        print(f"   Max {name.replace('_pos', ''):<2} positions: Return/Risk ratio = {risk_return_ratio:.2f}")
    
    print("="*100)

if __name__ == "__main__":
    run_max_positions_comparison()