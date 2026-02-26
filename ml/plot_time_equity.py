#!/usr/bin/env python3
"""
Time-based Equity Curve Plotter
Create daily equity curve from backtest results
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
from analyze_peaks import analyze_peaks, filter_significant_peaks

def create_daily_equity_curve(df, backtester, start_date, end_date, price_data):
    """
    Create daily equity curve from backtest results with mark-to-market.
    
    Args:
        df: Full dataset with warm-up period
        backtester: Configured backtester
        start_date: Start date for equity curve (after warm-up)
        end_date: End date for equity curve
        price_data: Price data for mark-to-market calculation
    """
    print("Starting backtest...")
    result = backtester.run_backtest(df, verbose=True)
    
    if not result.trades:
        print("❌ No trades found!")
        return None, None, None
    
    # NEW: Filter trades to only include those starting after start_date
    # This matches backtest_3stage.py's behavior when --start is used.
    original_trade_count = len(result.trades)
    analysis_start_ts = pd.to_datetime(start_date).tz_localize(None)
    result.trades = [t for t in result.trades if t.entry_time.replace(tzinfo=None) >= analysis_start_ts]
    
    if len(result.trades) != original_trade_count:
        print(f"   💡 Filtered to {len(result.trades)} trades starting after {start_date} (from {original_trade_count} total)")

    print(f"\n📊 Backtest Complete:")
    print(f"   Total Trades: {len(result.trades)}")
    # Note: result.total_return and final_capital might still reflect original run, 
    # but create_time_based_equity_mtm will use the filtered list.
    
    # Create time-based equity curve with mark-to-market
    daily_equity = create_time_based_equity_mtm(
        result.trades, start_date, end_date, 
        backtester.config.initial_capital, price_data,
        leverage=backtester.config.leverage,
        actual_daily_positions=result.daily_open_positions
    )
    
    # Create benchmark data
    benchmark_data = create_benchmark_data(price_data, start_date, end_date, backtester.config.initial_capital)
    
    # Extract CB events if Circuit Breaker is active
    cb_events = []
    if getattr(backtester.config, 'use_circuit_breaker', False):
        for trade in result.trades:
            if trade.exit_reason == 'CIRCUIT_BREAKER' and trade.exit_time:
                ts = trade.exit_time
                if ts not in cb_events:
                    cb_events.append(ts)
    
    return result, daily_equity, benchmark_data, cb_events

def create_time_based_equity_mtm(trades, start_date, end_date, initial_capital, price_data, leverage=20.0, actual_daily_positions=None):
    """
    Create daily equity curve with mark-to-market (floating PnL).
    This properly tracks unrealized gains/losses for open positions.
    In ISOLATED margin, each position's max loss is capped at its margin.
    """
    # Convert trades to DataFrame for efficient processing
    trade_events = []
    
    skipped_trades = 0
    for trade in trades:
        # ⚠️ CRITICAL: Validate trade data before processing
        if trade.position_size <= 0 or trade.position_size > 1_000_000_000:  # 1 billion USD max
            skipped_trades += 1
            continue
        if trade.entry_price <= 0 or trade.entry_price > 1_000_000:
            skipped_trades += 1
            continue
            
        # Entry event
        trade_events.append({
            'date': trade.entry_time.date(),
            'type': 'entry',
            'trade_id': id(trade),
            'position_size': trade.position_size,
            'direction': trade.direction,
            'entry_price': trade.entry_price,
            'symbol': getattr(trade, 'symbol', 'BTCUSDT')  # Default symbol
        })
        
        # Exit event (if closed)
        if trade.exit_time:
            trade_events.append({
                'date': trade.exit_time.date(),
                'type': 'exit',
                'trade_id': id(trade),
                'realized_pnl': trade.pnl
            })
    
    if skipped_trades > 0:
        print(f"⚠️ Skipped {skipped_trades} corrupted trades with invalid data")
    
    trade_df = pd.DataFrame(trade_events)
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Prepare price data for mark-to-market (per symbol)
    price_df = price_data.copy()
    price_df['date'] = pd.to_datetime(price_df['timestamp']).dt.date
    
    # Group by BOTH symbol and date to get correct price for each coin
    if 'symbol' in price_df.columns:
        price_daily = price_df.groupby(['symbol', 'date'])['close'].last().reset_index()
    else:
        # Fallback: if no symbol column, assume single asset
        price_daily = price_df.groupby('date')['close'].last().reset_index()
        price_daily['symbol'] = 'BTCUSDT'  # Default symbol
    
    # Initialize tracking
    daily_equity = []
    open_positions = {}  # trade_id -> position info
    realized_equity = initial_capital
    
    for date in date_range:
        current_date = date.date()
        
        # Process trade events for this date
        # ⚠️ CRITICAL: Process EXITS before ENTRIES to keep position count accurate
        day_events = trade_df[trade_df['date'] == current_date] if not trade_df.empty else pd.DataFrame()
        
        daily_realized_pnl = 0
        
        # First: process all exits
        if not day_events.empty:
            for _, event in day_events[day_events['type'] == 'exit'].iterrows():
                if event['trade_id'] in open_positions:
                    del open_positions[event['trade_id']]
                daily_realized_pnl += event['realized_pnl']
        
        # Then: process all entries
        if not day_events.empty:
            for _, event in day_events[day_events['type'] == 'entry'].iterrows():
                open_positions[event['trade_id']] = {
                    'position_size': event['position_size'],
                    'direction': event['direction'],
                    'entry_price': event['entry_price'],
                    'symbol': event['symbol']
                }
        
        # Update realized equity
        realized_equity += daily_realized_pnl
        
        # Calculate mark-to-market for open positions with enhanced validation
        floating_pnl = 0
        
        if open_positions:
            for trade_id, pos in open_positions.items():
                # ⚠️ CRITICAL: Validate entry price
                if pos['entry_price'] <= 0 or pos['entry_price'] > 1_000_000:
                    print(f"⚠️ Invalid entry price ${pos['entry_price']:.2f} for position {trade_id} - skipping")
                    continue
                
                # ⚠️ CRITICAL: Validate position size
                if pos['position_size'] <= 0 or pos['position_size'] > 1_000_000:
                    print(f"⚠️ Invalid position size ${pos['position_size']:.2f} for position {trade_id} - skipping")
                    continue
                
                # 🔧 FIX: Get price for THIS SPECIFIC symbol
                symbol_price_data = price_daily[
                    (price_daily['symbol'] == pos['symbol']) & 
                    (price_daily['date'] == current_date)
                ]
                
                if symbol_price_data.empty:
                    # No price data for this symbol on this date - skip
                    continue
                
                current_price = symbol_price_data['close'].iloc[0]
                
                # Validate current price
                if current_price <= 0 or current_price > 1_000_000:
                    continue
                
                # Calculate percentage change
                price_change_pct = (current_price / pos['entry_price']) - 1
                
                # ⚠️ Cap extreme price changes (data errors)
                price_change_pct = np.clip(price_change_pct, -0.99, 100)  # -99% to +10,000% max
                
                # ⚠️ SIMPLIFIED & SAFER floating PnL calculation
                if pos['direction'] == 'LONG':
                    pos_pnl = pos['position_size'] * price_change_pct
                else:  # SHORT 
                    pos_pnl = pos['position_size'] * (-price_change_pct)
                
                # ⚠️ ISOLATED margin cap: max loss = margin = position_size / leverage
                margin = pos['position_size'] / leverage
                pos_pnl = max(pos_pnl, -margin)  # Can't lose more than margin
                
                floating_pnl += pos_pnl
        
        # Debug extreme floating PnL (optional - can be removed after testing)
        if abs(floating_pnl) > 100000:  # $100K threshold (lowered)
            print(f"⚠️ Extreme floating PnL on {current_date}: ${floating_pnl:,.2f}")
            print(f"   Open positions: {len(open_positions)}")
            for i, (tid, pos) in enumerate(list(open_positions.items())[:3]):
                symbol_price = price_daily[
                    (price_daily['symbol'] == pos['symbol']) & 
                    (price_daily['date'] == current_date)
                ]
                if not symbol_price.empty:
                    curr_px = symbol_price['close'].iloc[0]
                    if pos['direction'] == 'LONG':
                        pnl_calc = (curr_px - pos['entry_price']) * pos['position_size'] / pos['entry_price']
                    else:
                        pnl_calc = (pos['entry_price'] - curr_px) * pos['position_size'] / pos['entry_price']
                    print(f"   {pos['symbol']}: ${pos['position_size']:,.2f} @ ${pos['entry_price']:,.2f} ({pos['direction']}) → ${pnl_calc:,.2f}")
                else:
                    print(f"   {pos['symbol']}: No price data for {current_date}")
        
        # Total equity = realized + floating (can't go below 0 in real trading)
        total_equity = max(realized_equity + floating_pnl, 0)
        
        # Calculate daily return using log returns (more stable)
        if len(daily_equity) > 0:
            prev_equity = daily_equity[-1]['equity']
            daily_return = np.log(total_equity / prev_equity) if prev_equity > 0 and total_equity > 0 else 0
        else:
            daily_return = 0
        
        daily_equity.append({
            'date': date,
            'equity': total_equity,
            'realized_equity': realized_equity,
            'floating_pnl': floating_pnl,
            'daily_realized_pnl': daily_realized_pnl,
            'daily_return': daily_return,
            'open_positions_count': actual_daily_positions.get(date, len(open_positions)) if actual_daily_positions else len(open_positions)
        })
        
        # 🔥 Account blown — ONLY when REALIZED equity (closed trades) <= 0
        # In ISOLATED margin, floating losses don't blow account — each position
        # is independent and can only lose its own margin. The account is only 
        # truly blown when all margin has been lost through actual liquidations.
        if realized_equity <= 0:
            for remaining_date in date_range[date_range > date]:
                daily_equity.append({
                    'date': remaining_date,
                    'equity': 0,
                    'realized_equity': 0,
                    'floating_pnl': 0,
                    'daily_realized_pnl': 0,
                    'daily_return': 0,
                    'open_positions_count': 0
                })
            break
    
    return pd.DataFrame(daily_equity)

def create_benchmark_data(price_data, start_date, end_date, initial_capital):
    """
    Create realistic buy & hold benchmark.
    """
    # Prepare price data for mark-to-market (single asset benchmark)
    price_df = price_data.copy()
    price_df['date'] = pd.to_datetime(price_df['timestamp']).dt.date
    
    # For benchmark, typically use BTC or a single representative asset
    if 'symbol' in price_df.columns:
        # Use BTCUSDT as benchmark if available
        btc_data = price_df[price_df['symbol'] == 'BTCUSDT']
        if not btc_data.empty:
            price_df = btc_data
        else:
            # Fallback to first available symbol
            first_symbol = price_df['symbol'].iloc[0]
            price_df = price_df[price_df['symbol'] == first_symbol]
    
    # Filter to date range
    start_dt = pd.to_datetime(start_date).date()
    end_dt = pd.to_datetime(end_date).date()
    
    price_range = price_df[
        (price_df['date'] >= start_dt) & 
        (price_df['date'] <= end_dt)
    ]
    
    if price_range.empty:
        return pd.DataFrame()
    
    # Get daily prices
    daily_prices = price_range.groupby('date')['close'].last().reset_index()
    
    # Calculate buy & hold returns
    initial_price = daily_prices['close'].iloc[0]
    daily_prices['benchmark_equity'] = (daily_prices['close'] / initial_price) * initial_capital
    
    # Create date range and merge
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    benchmark_df = pd.DataFrame({'date': date_range})
    benchmark_df['date_only'] = benchmark_df['date'].dt.date
    
    # Merge and forward fill
    benchmark_df = benchmark_df.merge(
        daily_prices[['date', 'benchmark_equity']], 
        left_on='date_only', right_on='date', how='left'
    )
    benchmark_df['benchmark_equity'] = benchmark_df['benchmark_equity'].ffill()
    
    return benchmark_df[['date_x', 'benchmark_equity']].rename(columns={'date_x': 'date'})

def plot_time_based_equity(daily_equity_df, trades, benchmark_df=None, title="Time-Based Equity Curve", cb_events=None):
    """
    Plot comprehensive time-based equity analysis.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 14))
    
    # 1. Daily Equity Curve
    ax1.plot(daily_equity_df['date'], daily_equity_df['equity'], 
             linewidth=2.5, color='#1f77b4', alpha=0.8)
    ax1.fill_between(daily_equity_df['date'], daily_equity_df['equity'], 
                     alpha=0.2, color='#1f77b4')
    
    # Add trade markers with error handling
    trade_dates = [t.exit_time for t in trades if t.exit_time]
    trade_pnls = [t.pnl for t in trades if t.exit_time]
    
    win_trades = []
    lose_trades = []
    
    for t in trades:
        if t.exit_time:
            # Find corresponding equity value safely
            matching_dates = daily_equity_df[daily_equity_df['date'].dt.date == t.exit_time.date()]
            if not matching_dates.empty:
                equity_value = matching_dates['equity'].iloc[0]
                if t.pnl > 0:
                    win_trades.append((t.exit_time, equity_value))
                else:
                    lose_trades.append((t.exit_time, equity_value))
    
    if win_trades:
        win_dates, win_equities = zip(*win_trades)
        ax1.scatter(win_dates, win_equities, color='green', s=30, alpha=0.7, label='Winning Trades', zorder=5)
    
    if lose_trades:
        lose_dates, lose_equities = zip(*lose_trades)
        ax1.scatter(lose_dates, lose_equities, color='red', s=30, alpha=0.7, label='Losing Trades', zorder=5)
    
    # 🆕 Plot Circuit Breaker exit markers
    if cb_events:
        for cb_ts in cb_events:
            cb_date = pd.to_datetime(cb_ts).date()
            
            # Find closest equity point for the marker by comparing dates
            # daily_equity_df['date'] contains datetime/timestamp objects
            # Convert both sides to pure date components or normalize
            mask = pd.to_datetime(daily_equity_df['date']).dt.date == cb_date
            
            if mask.any():
                idx = mask.idxmax()
                x_val = daily_equity_df.loc[idx, 'date']
                
                # Annotate the chart with a vertical dotted line and text
                ax1.axvline(x=x_val, color='darkorange', linestyle=':', alpha=0.8, zorder=1)
                ax1.text(x_val, ax1.get_ylim()[1]*0.95, 'CB', color='darkorange', 
                         fontsize=9, fontweight='bold', ha='center', va='top', 
                         bbox=dict(facecolor='white', alpha=0.6, pad=2, edgecolor='darkorange'))
                
        # Just to add to legend
        ax1.plot([], [], color='darkorange', linestyle=':', label='Circuit Breaker')
    
    ax1.set_title('Daily Equity Curve', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Equity ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Format y-axis
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # 2. Daily Returns (convert log returns to %)  
    returns = daily_equity_df['daily_return'] * 100
    colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in returns]
    
    ax2.bar(daily_equity_df['date'], returns, color=colors, alpha=0.7, width=0.8)
    ax2.set_title('Daily Returns (Log %) + Mark-to-Market', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Daily Return (%)')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Add floating vs realized annotation
    if 'floating_pnl' in daily_equity_df.columns:
        ax2_text = f"Floating PnL Range: ${daily_equity_df['floating_pnl'].min():,.0f} to ${daily_equity_df['floating_pnl'].max():,.0f}"
        ax2.text(0.02, 0.98, ax2_text, transform=ax2.transAxes, 
                verticalalignment='top', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 3. Drawdown
    equity_values = daily_equity_df['equity'].values
    peak = np.maximum.accumulate(equity_values)
    drawdown = (peak - equity_values) / peak * 100
    
    ax3.fill_between(daily_equity_df['date'], drawdown, alpha=0.3, color='red')
    ax3.plot(daily_equity_df['date'], drawdown, color='red', linewidth=1.5)
    ax3.set_title('Drawdown (%)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Drawdown (%)')
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax3.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # 4. Cumulative vs Realistic Benchmark
    initial_equity = daily_equity_df['equity'].iloc[0]
    cumulative_return = (daily_equity_df['equity'] / initial_equity - 1) * 100
    
    ax4.plot(daily_equity_df['date'], cumulative_return, 
             label='3-Stage Strategy (MTM)', linewidth=2.5, color='#1f77b4')
    
    # Add realistic benchmark if available
    if benchmark_df is not None and not benchmark_df.empty:
        benchmark_initial = benchmark_df['benchmark_equity'].iloc[0]
        benchmark_return = (benchmark_df['benchmark_equity'] / benchmark_initial - 1) * 100
        ax4.plot(benchmark_df['date'], benchmark_return, 
                 label='Buy & Hold BTC', linewidth=2, color='orange', linestyle='--')
    else:
        # Fallback to simple benchmark
        benchmark_daily = 0.013  # ~5% annual
        benchmark_return = np.cumsum([benchmark_daily] * len(daily_equity_df))
        ax4.plot(daily_equity_df['date'], benchmark_return, 
                 label='5% Annual Benchmark', linewidth=2, color='gray', linestyle='--')
    
    ax4.set_title('Strategy vs Buy & Hold (Mark-to-Market)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Cumulative Return (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax4.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig

def print_time_based_stats(daily_equity_df, trades, benchmark_df=None, target_initial_capital=100.0):
    """Print comprehensive time-based performance statistics with mark-to-market."""
    
    # Calculate metrics
    total_days = len(daily_equity_df)
    trading_days = len([d for d in daily_equity_df.get('daily_realized_pnl', [0]) if d != 0])
    
    # Use specified initial capital for more accurate return stats
    initial_equity = target_initial_capital
    final_equity = daily_equity_df['equity'].iloc[-1]
    total_return = (final_equity / initial_equity - 1) * 100
    # Daily returns (log returns)
    daily_returns = daily_equity_df['daily_return']
    positive_days = len([r for r in daily_returns if r > 0])
    negative_days = len([r for r in daily_returns if r < 0])
    
    # Best/worst days
    if 'daily_realized_pnl' in daily_equity_df.columns:
        best_day = daily_equity_df.loc[daily_equity_df['daily_realized_pnl'].idxmax()]
        worst_day = daily_equity_df.loc[daily_equity_df['daily_realized_pnl'].idxmin()]
    else:
        # Fallback for old format
        best_day = daily_equity_df.loc[daily_equity_df['equity'].diff().idxmax()]
        worst_day = daily_equity_df.loc[daily_equity_df['equity'].diff().idxmin()]
    
    # Max drawdown (now properly calculated with mark-to-market)
    equity_values = daily_equity_df['equity'].values
    peak = np.maximum.accumulate(equity_values)
    drawdown = (peak - equity_values) / peak
    max_drawdown = np.max(drawdown) * 100
    max_dd_date = daily_equity_df.iloc[np.argmax(drawdown)]['date']
    
    # Enhanced volatility metrics
    daily_vol = np.std(daily_returns) * np.sqrt(252)
    
    # Sharpe ratio (log returns, 3% risk-free rate)
    risk_free_rate = 0.03 / 252  # Daily risk-free rate
    excess_returns = daily_returns - risk_free_rate
    sharpe = np.mean(excess_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
    
    # Sortino ratio (downside deviation)
    negative_returns = daily_returns[daily_returns < 0]
    downside_deviation = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
    sortino = np.mean(excess_returns) * np.sqrt(252) / downside_deviation if downside_deviation > 0 else 0
    
    # Calmar ratio (return/max drawdown)
    annualized_return = np.mean(daily_returns) * 252
    calmar = annualized_return / (max_drawdown / 100) if max_drawdown > 0 else 0
    
    # Mark-to-market specific metrics with validation
    if 'floating_pnl' in daily_equity_df.columns:
        max_floating_loss = daily_equity_df['floating_pnl'].min()
        max_floating_gain = daily_equity_df['floating_pnl'].max()
        avg_positions = daily_equity_df['open_positions_count'].mean()
        
        # Validate for extreme outliers
        if max_floating_gain > initial_equity * 1000:  # More than 1000x initial capital
            print(f"\n⚠️ WARNING: Extreme floating gain detected: ${max_floating_gain:,.2f}")
            print(f"   This suggests an error in floating PnL calculation.")
        
        if abs(max_floating_loss) > initial_equity * 10:  # More than 10x initial capital loss
            print(f"\n⚠️ WARNING: Extreme floating loss detected: ${max_floating_loss:,.2f}")
            print(f"   This suggests an error in floating PnL calculation.")
    else:
        max_floating_loss = max_floating_gain = avg_positions = 0
    
    print("\n" + "="*80)
    print("📈 TIME-BASED PERFORMANCE ANALYSIS (MARK-TO-MARKET) 📈")
    print("="*80)
    
    print(f"\n📅 Time Period:")
    print(f"   Start Date: {daily_equity_df['date'].iloc[0].strftime('%Y-%m-%d')}")
    print(f"   End Date: {daily_equity_df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"   Total Days: {total_days}")
    print(f"   Active Trading Days: {trading_days}")
    
    print(f"\n💰 Equity Performance (MTM):")
    print(f"   Initial Equity: ${initial_equity:,.2f}")
    print(f"   Final Equity: ${final_equity:,.2f}")
    print(f"   Total Return: {total_return:+.2f}%")
    print(f"   Annualized Return: {annualized_return*100:.2f}%")
    
    # Mark-to-market specifics
    if 'floating_pnl' in daily_equity_df.columns:
        print(f"\n💫 Mark-to-Market Analysis:")
        print(f"   Max Floating Loss: ${max_floating_loss:,.2f}")
        print(f"   Max Floating Gain: ${max_floating_gain:,.2f}")
        print(f"   Average Open Positions: {avg_positions:.1f}")
        print(f"   Final Realized Equity: ${daily_equity_df['realized_equity'].iloc[-1]:,.2f}")
        print(f"   Final Floating PnL: ${daily_equity_df['floating_pnl'].iloc[-1]:,.2f}")
    
    print(f"\n📊 Daily Statistics:")
    print(f"   Positive Days: {positive_days} ({positive_days/total_days*100:.1f}%)")
    print(f"   Negative Days: {negative_days} ({negative_days/total_days*100:.1f}%)")
    if 'daily_realized_pnl' in daily_equity_df.columns:
        print(f"   Best Day: {best_day['date'].strftime('%Y-%m-%d')} (+${best_day['daily_realized_pnl']:,.2f})")
        print(f"   Worst Day: {worst_day['date'].strftime('%Y-%m-%d')} (${worst_day['daily_realized_pnl']:,.2f})")
    
    print(f"\n📉 Enhanced Risk Metrics:")
    print(f"   Max Drawdown: {max_drawdown:.2f}% (on {max_dd_date.strftime('%Y-%m-%d')})")
    print(f"   Annualized Volatility: {daily_vol*100:.2f}%")
    print(f"   Sharpe Ratio: {sharpe:.3f}")
    print(f"   Sortino Ratio: {sortino:.3f}")
    print(f"   Calmar Ratio: {calmar:.3f}")
    
    # Benchmark comparison
    if benchmark_df is not None and not benchmark_df.empty:
        bench_initial = benchmark_df['benchmark_equity'].iloc[0]
        bench_final = benchmark_df['benchmark_equity'].iloc[-1]
        bench_return = (bench_final / bench_initial - 1) * 100
        alpha = total_return - bench_return
        print(f"\n🎯 vs Buy & Hold Benchmark:")
        print(f"   Benchmark Return: {bench_return:+.2f}%")
        print(f"   Alpha (Excess Return): {alpha:+.2f}%")
    
    print(f"\n🎯 Trade Summary:")
    print(f"   Total Trades: {len(trades)}")
    print(f"   Avg Trades/Day: {len(trades)/trading_days:.2f}" if trading_days > 0 else "   Avg Trades/Day: 0")
    
    print("\n⚠️  Note: This analysis includes mark-to-market (floating PnL)")
    print("   which provides accurate drawdown and risk metrics.")
    print("="*80)
    
    # ── Peak-to-Drawdown Analysis ────────────────────────────────────
    try:
        all_peaks = analyze_peaks(daily_equity_df)
        sig_peaks = filter_significant_peaks(all_peaks, min_gain_pct=10.0)
        
        if len(sig_peaks) >= 3:
            dd_vals = sig_peaks['dd_depth']
            dur_vals = sig_peaks['dd_duration_days']
            rec_vals = sig_peaks['recovery_days'].dropna()
            deep_pct = (dd_vals > 20).mean() * 100
            
            print(f"\n{'='*80}")
            print(f"🏔️  PEAK-TO-DRAWDOWN ANALYSIS ({len(sig_peaks)} significant peaks)")
            print(f"{'='*80}")
            print(f"\n   After each new ATH (>10% gain):")
            print(f"     Median DD:      {dd_vals.median():.1f}%")
            print(f"     Mean DD:        {dd_vals.mean():.1f}%")
            print(f"     Worst DD:       {dd_vals.max():.1f}%")
            print(f"     DD > 20%:       {deep_pct:.0f}% of peaks")
            print(f"     Time to trough: {dur_vals.median():.0f} days (median)")
            if len(rec_vals) > 0:
                print(f"     Recovery time:  {rec_vals.median():.0f} days (median)")
            never = sig_peaks['recovery_days'].isna().sum()
            if never > 0:
                print(f"     Never recovered: {never} peaks")
            
            print(f"\n   💡 After ATH: expect ~{dd_vals.median():.0f}% DD in ~{dur_vals.median():.0f}d, "
                  f"recover ~{rec_vals.median():.0f}d" if len(rec_vals) > 0 else "")
            print(f"{'='*80}")
    except Exception:
        pass  # Peak analysis is optional, don't break main output

def main():
    import argparse
    """Main function to create time-based equity curve with mark-to-market."""
    
    parser = argparse.ArgumentParser(description="Enhanced Time-Based Equity Curve Plotter")
    
    # Matching arguments from backtest_3stage.py
    parser.add_argument('--data', type=str, default=None, help='Path to data file')
    parser.add_argument('--capital', type=float, default=100.0, help='Initial capital')
    parser.add_argument('--risk', type=float, default=0.01, help='Risk per trade (0.01 = 1%)')
    parser.add_argument('--threshold', type=float, default=0.65, help='Entry confidence threshold')
    parser.add_argument('--fee', type=float, default=0.001, help='Fee rate (0.001 = 0.1%)')
    parser.add_argument('--slippage', type=float, default=0.0005, help='Slippage (0.0005 = 0.05%)')
    parser.add_argument('--kelly', action='store_true', help='Use Kelly Criterion')
    parser.add_argument('--fixed-size', action='store_true', help='Use fixed position size')
    parser.add_argument('--size-usd', type=float, default=1000, help='Fixed position size in USD')
    parser.add_argument('--leverage', type=float, default=20.0, help='Leverage multiplier (e.g. 1, 3, 5, 7, 10, 20)')
    parser.add_argument('--max-positions', type=int, default=10, help='Max open positions (default: 10)')
    
    # Trailing Stop arguments
    parser.add_argument('--trailing', action='store_true', help='Enable Trailing Stop')
    parser.add_argument('--trailing-start', type=float, default=0.1, help='Trailing start pct (e.g. 0.02 for 2%)')
    parser.add_argument('--trailing-step', type=float, default=0.05, help='Trailing step pct (e.g. 0.01 for 1%)')
    
    # Portfolio Trailing Stop arguments
    parser.add_argument('--portfolio-trailing', action='store_true', help='Enable Portfolio-level Trailing Stop')
    parser.add_argument('--pt-start', type=float, default=0.30, help='Portfolio Trailing start pct (default 30%)')
    parser.add_argument('--pt-step', type=float, default=0.15, help='Portfolio Trailing step pct (default 15%)')
    parser.add_argument('--pt-cooldown', type=float, default=1.0, help='Days to cooldown after portfolio trailing stop hits')
    
    # Pullback options
    parser.add_argument('--entry-pullback', type=float, default=0.0, help='Pullback pct for limit entry (e.g. 0.005 for 0.5%)')
    parser.add_argument('--entry-timeout', type=int, default=3, help='Timeout bars for limit entry')
    parser.add_argument('--max-bars', type=int, default=10, help='Max bars to hold trade (timeout)')
    
    # Scanner Filter arguments
    parser.add_argument('--use-scanner', action='store_true', help='Enable SmartScanner Entry Zone filtering')
    parser.add_argument('--scanner-mae', type=float, default=0.04, help='Max Adverse Excursion for zone (default: 0.04)')
    parser.add_argument('--scanner-mfe', type=float, default=0.12, help='Max Favorable Excursion for zone (default: 0.12)')
    parser.add_argument('--scanner-lookback', type=int, default=6, help='Lookback days for scanner entry (default: 6)')
    
    # Circuit Breaker
    parser.add_argument('--cb-profile', type=str, choices=['0.6', '0.65', 'none'], default='none',
                        help='Circuit Breaker optimization profile to use based on robustness insights')
    
    parser.add_argument("--start", type=str, default='2026-01-01', help="Analysis start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default='2026-02-23', help="Analysis end date (YYYY-MM-DD)")
    parser.add_argument("--timeframe", type=str, default='1d', help="Timeframe (1d, 4h, etc.)")
    parser.add_argument("--margin-mode", type=str, default='ISOLATED', choices=['ISOLATED', 'CROSS'], help="Margin mode")
    parser.add_argument("--warmup", type=int, default=0, help="Warm-up months for indicators")
    parser.add_argument("--reset-capital", action="store_true", help="Reset capital to initial on start date (for direct comparison)")
    
    args = parser.parse_args()

    # Configuration
    backtest_start = args.start
    backtest_end = args.end
    warm_up_months = args.warmup
    timeframe = args.timeframe
    leverage = args.leverage
    initial_capital = args.capital
    margin_mode = args.margin_mode
    use_kelly = args.kelly
    
    print(f"🚀 Creating Enhanced Time-Based Equity Curve (Mark-to-Market)")
    print(f"   Analysis Period: {backtest_start} to {backtest_end}")
    print(f"   Warm-up Period: {warm_up_months} months")
    print(f"   Timeframe: {timeframe}")
    print(f"   Leverage: {leverage}x")
    print(f"   Margin Mode: {margin_mode}")
    print(f"   Capital: ${initial_capital:,.2f}")
    if args.reset_capital:
        print(f"   ⚠️  Capital WILL BE RESET TO ${initial_capital:,.2f} on {backtest_start}")
    
    # Load data
    data_path = Path(__file__).parent.parent / 'bitget-data' / 'processed' / f'features_{timeframe}_full.parquet'
    if not data_path.exists():
        # Fallback to standard path if features_tf_full doesn't exist
        data_path = Path(__file__).parent.parent / 'bitget-data' / 'processed' / f'features_{timeframe}.parquet'
        
    if args.data:
        data_path = Path(args.data)
        
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print(f"Run 'python ml/sync_and_rebuild.py --timeframe {timeframe}' first.")
        return
    
    print(f"\n📂 Loading data with warm-up period...")
    df = pd.read_parquet(data_path)
    print(f"   Loaded {len(df):,} rows")
    
    # Calculate warm-up start date
    backtest_start_dt = pd.to_datetime(backtest_start)
    warm_up_start_dt = backtest_start_dt - pd.DateOffset(months=warm_up_months)
    backtest_end_dt = pd.to_datetime(backtest_end)
    
    # Filter with warm-up period
    df_with_warmup = df[
        (df['timestamp'] >= warm_up_start_dt) & 
        (df['timestamp'] <= backtest_end_dt)
    ].copy()
    
    print(f"   Warm-up starts: {warm_up_start_dt.strftime('%Y-%m-%d')}")
    print(f"   Analysis starts: {backtest_start}")
    print(f"   Total data with warm-up: {len(df_with_warmup):,} rows")
    
    if df_with_warmup.empty:
        print("❌ No data found for the specified period!")
        return
    
    # Configure backtest
    config = BacktestConfig(
        initial_capital=initial_capital,
        risk_per_trade=args.risk,
        entry_threshold=args.threshold,
        fee_rate=args.fee,
        slippage=args.slippage,
        leverage=leverage,
        timeframe=timeframe,
        margin_mode=margin_mode,
        use_kelly=use_kelly,
        fixed_position_size=args.fixed_size,
        position_size_usd=args.size_usd,
        max_open_trades=args.max_positions,
        require_fresh_crossover_after_exit=True,
        # Trailing Stop arguments
        use_trailing_stop=args.trailing,
        trailing_start_pct=args.trailing_start,
        trailing_step_pct=args.trailing_step,
        # Portfolio Trailing arguments
        use_portfolio_trailing=args.portfolio_trailing,
        portfolio_trailing_start_pct=args.pt_start,
        portfolio_trailing_step_pct=args.pt_step,
        # Pullback options
        entry_pullback_pct=args.entry_pullback,
        entry_pullback_timeout=args.entry_timeout,
        max_bars=args.max_bars,
        # Scanner options
        use_scanner_filter=args.use_scanner,
        scanner_mae=args.scanner_mae,
        scanner_mfe=args.scanner_mfe,
        scanner_lookback_days=args.scanner_lookback
    )
    
    # Attach CB properties if requested
    if args.cb_profile == '0.6':
        config.use_circuit_breaker = True
        config.cb_confluence_tf = '12h'
        config.cb_confluence_threshold = 0.2
        config.cb_velocity_lookback = 2
        config.cb_velocity_threshold = 0.1
        config.cb_sleep_hours = 4
        print(f"   🛡️ Using Circuit Breaker Profile: 0.6")
    elif args.cb_profile == '0.65':
        config.use_circuit_breaker = True
        config.cb_confluence_tf = '12h'
        config.cb_confluence_threshold = 0.15
        config.cb_velocity_lookback = 1
        config.cb_velocity_threshold = 0.1
        config.cb_sleep_hours = 5
        print(f"   🛡️ Using Circuit Breaker Profile: 0.65")
    
    # Create backtester
    backtester = ThreeStageBacktester(config)
    
    # Prepare price data for mark-to-market (must include symbol column)
    price_columns = ['timestamp', 'close']
    if 'symbol' in df_with_warmup.columns:
        price_columns.insert(0, 'symbol')
    if 'open' in df_with_warmup.columns:
        price_columns.extend(['open', 'high', 'low'])
    price_data = df_with_warmup[price_columns].copy()
    
    # Run enhanced backtest with mark-to-market
    result, daily_equity_df, benchmark_df, cb_events = create_daily_equity_curve(
        df_with_warmup, backtester, backtest_start, backtest_end, price_data
    )
    
    if daily_equity_df is not None:
        # If reset-capital is enabled, normalize the daily_equity
        if args.reset_capital:
            # Find the capital on start_date
            analysis_start_ts = pd.to_datetime(backtest_start).tz_localize(None)
            
            # Use normalize() to compare only dates if they are Timestamps
            daily_equity_df['date_dt'] = pd.to_datetime(daily_equity_df['date']).dt.tz_localize(None).dt.normalize()
            mask = daily_equity_df['date_dt'] == analysis_start_ts.normalize()
            
            if mask.any():
                actual_capital_at_start = daily_equity_df.loc[mask, 'equity'].values[0]
                ratio = initial_capital / actual_capital_at_start
                daily_equity_df['equity'] *= ratio
                daily_equity_df['realized_equity'] *= ratio
                print(f"   🔄 Data normalized: Resetting capital to ${initial_capital} on {backtest_start}")
            
            # Clean up temp column
            daily_equity_df = daily_equity_df.drop(columns=['date_dt'])

        # Print enhanced statistics
        print_time_based_stats(daily_equity_df, result.trades, benchmark_df, initial_capital)
        
        # Plot enhanced equity curve
        print(f"\n📈 Creating enhanced time-based equity curve plot...")
        fig = plot_time_based_equity(
            daily_equity_df, 
            result.trades,
            benchmark_df,
            title=f'3-Stage ML Strategy ({margin_mode}, {leverage}x leverage)',
            cb_events=cb_events
        )
        
        # Save enhanced results
        filename_suffix = f"{timeframe}_{leverage}x_{margin_mode.lower()}"
        if args.reset_capital: filename_suffix += "_reset"
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(exist_ok=True)
        save_path = results_dir / f'time_equity_{filename_suffix}.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"💾 Enhanced chart saved: {save_path.name}")
        
        # Save comprehensive data
        csv_path = save_path.with_suffix('.csv')
        daily_equity_df.to_csv(csv_path, index=False)
        print(f"📊 Mark-to-market data saved: {csv_path.name}")
    
    if benchmark_df is not None and not benchmark_df.empty:
        benchmark_path = results_dir / f'benchmark_data_{timeframe}.csv'
        benchmark_df.to_csv(benchmark_path, index=False)
        print(f"📊 Benchmark data saved: {benchmark_path.name}")
    
    print(f"\n✅ Analysis complete!")
    plt.show()

if __name__ == '__main__':
    main()
if __name__ == '__main__':
    main()