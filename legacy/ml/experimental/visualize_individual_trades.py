
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Fix paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest_3stage import plot_backtest_trades
from backtesting.backtest_timeframes import run_timeframe_backtest, BacktestConfig

def main():
    print("running 12h backtest to find trades...")
    config = BacktestConfig(initial_capital=10000)
    config.start_date = '2024-01-01'
    config.end_date = '2024-03-01'
    
    # 1. Get trades
    result, df = run_timeframe_backtest('12h', config)
    
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    
    if not result or not result.trades:
        print("No trades found.")
        return
        
    trades = result.trades
    print(f"Found {len(trades)} trades.")
    
    # 2. Select representative trades
    # Sort by PnL to find best win and worst loss
    trades_sorted = sorted(trades, key=lambda t: t.pnl_pct, reverse=True)
    
    selected_trades = []
    
    # Best Win
    if trades_sorted:
        selected_trades.append(('Best Win', trades_sorted[0]))
        
    # Worst Loss (last one)
    if len(trades_sorted) > 1 and trades_sorted[-1].pnl < 0:
        selected_trades.append(('Worst Loss', trades_sorted[-1]))
        
    # Random others (middle)
    if len(trades_sorted) > 5:
        selected_trades.append(('Average Trade', trades_sorted[len(trades_sorted)//2]))
        
    # 3. Plot each
    results_dir = Path('ml/results/individual_trades')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for label, trade in selected_trades:
        print(f"Plotting {label}: {trade.symbol} {trade.direction} PnL: {trade.pnl_pct:.1%}")
        
        # Filter DF for this symbol first
        df_symbol = df[df['symbol'] == trade.symbol].copy()
        
        # Define Window
        if trade.entry_time not in df_symbol.index:
            print(f"Skipping {label}: Entry time not found in symbol data")
            continue
            
        start_idx = df_symbol.index.get_loc(trade.entry_time) - 20
        # handle slice if duplicate index (shouldn't happen with filtered symbol but just in case)
        if isinstance(start_idx, slice):
            start_idx = start_idx.start - 20
        elif hasattr(start_idx, '__iter__'): 
             start_idx = start_idx[0] - 20
             
        start_idx = max(int(0), int(start_idx))
        
        # approximate exit index since plot_backtest_trades handles slicing visually
        # just grab a chunk
        chunk_size = 50 
        end_idx = min(len(df_symbol), start_idx + chunk_size)
        
        # Slice DataFrame
        df_trade = df_symbol.iloc[start_idx:end_idx].copy()
        
        # Filename
        safe_label = label.replace(' ', '_').lower()
        save_path = results_dir / f"{safe_label}_{trade.symbol}.png"
        
        # Plot just this trade
        plot_backtest_trades(df_trade, [trade], title=f"{label} - {trade.symbol} ({trade.direction})", save_path=str(save_path))
        
    print(f"\nDone. Images saved to {results_dir}")

if __name__ == "__main__":
    main()
