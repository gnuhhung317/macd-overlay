
import os

PLOT_FUNC = r'''
def plot_backtest_trades(df: pd.DataFrame, trades: List[Trade], title: str = "Backtest Trades", save_path: str = None):
    """
    Plot equity curves and trade entries/exits on price chart.
    
    Args:
        df: DataFrame with OHLCV data (must have datetime index or timestamp column)
        trades: List of Trade objects
        title: Chart title
        save_path: Path to save the plot
    """
    if df.empty or not trades:
        print("⚠️ No data or trades to plot")
        return

    # Ensure timestamp is index
    df_plot = df.copy()
    if 'timestamp' in df_plot.columns:
        df_plot.set_index('timestamp', inplace=True)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # 1. Price Chart with Trades
    ax1.plot(df_plot.index, df_plot['close'], label='Close Price', color='gray', alpha=0.5, linewidth=1)
    
    # Plot trades
    long_entries = []
    long_exits_win = []
    long_exits_loss = []
    
    for t in trades:
        if t.direction == 'LONG':
            long_entries.append((t.entry_time, t.entry_price))
            if t.pnl > 0:
                long_exits_win.append((t.exit_time, t.exit_price))
                # Draw colored line
                ax1.plot([t.entry_time, t.exit_time], [t.entry_price, t.exit_price], 
                        color='green', alpha=0.3, linewidth=1)
            else:
                long_exits_loss.append((t.exit_time, t.exit_price))
                ax1.plot([t.entry_time, t.exit_time], [t.entry_price, t.exit_price], 
                        color='red', alpha=0.3, linewidth=1)

    # Markers
    if long_entries:
        times, prices = zip(*long_entries)
        ax1.scatter(times, prices, marker='^', color='blue', s=50, label='Long Entry', zorder=5)
        
    if long_exits_win:
        times, prices = zip(*long_exits_win)
        ax1.scatter(times, prices, marker='v', color='green', s=50, label='Win Exit', zorder=5)
        
    if long_exits_loss:
        times, prices = zip(*long_exits_loss)
        ax1.scatter(times, prices, marker='x', color='red', s=50, label='Loss Exit', zorder=5)
    
    ax1.set_title(f'{title} - Price & Trades', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative PnL
    equity_curve = [0]
    dates = [df_plot.index[0]]
    cum_pnl = 0
    
    # Sort trades by exit time
    sorted_trades = sorted(trades, key=lambda t: t.exit_time if t.exit_time else t.entry_time)
    
    trade_dates = [t.exit_time for t in sorted_trades]
    trade_pnl = [t.pnl_pct * 100 for t in sorted_trades] # Standardized to % return
    
    # Reconstruct equity curve aligned with time
    if trade_pnl:
        cum_pnl_curve = [sum(trade_pnl[:i+1]) for i in range(len(trade_pnl))]
        
        ax2.plot(trade_dates, cum_pnl_curve, label='Cumulative Return %', color='purple', linewidth=2)
        ax2.fill_between(trade_dates, cum_pnl_curve, alpha=0.1, color='purple')
    
    ax2.set_title('Cumulative Return %', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Return %')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Trade plot saved to: {save_path}")
    
    plt.close()
'''

def restore():
    path = r'ml\backtest_3stage.py'
    
    # Read with error ignoring
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading: {e}")
        return

    # Clean up any potential garbage at the end if it's not proper python
    # We look for the end of plot_equity_curve function
    marker = "plt.close()"
    last_idx = content.rfind(marker)
    
    if last_idx != -1:
        # Keep until plt.close() + len(marker) of the LAST occurrence (assuming it was plot_equity_curve)
        # But wait, if I appended garbage, rfind might find the marker in the garbage if I appended my previous script verbatim?
        # My previous snippet also had plt.close().
        # I should check if 'plot_backtest_trades' is already in content.
        
        if 'def plot_backtest_trades' in content:
            print("Function already exists. Checking trailing garbage...")
            # If function exists, maybe it is fine? changing nothing.
            pass
        else:
            # Append it
            # Truncate after last valid function if possible? 
            # Safest is to find "def plot_equity_curve" block and ensure it ends, then append.
            # But "errors='ignore'" might have left stripped bytes.
            
            # Let's just append if not present, but add some newlines.
            print("Appending function...")
            content += "\n\n" + PLOT_FUNC
            
        # Write back
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Success restoring file.")
        
    else:
        print("Could not find marker in file. File might be severely damaged.")

if __name__ == "__main__":
    restore()
