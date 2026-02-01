
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add ml path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest_3stage import Trade, plot_backtest_trades

def test_plot():
    print("Generating dummy data...")
    dates = pd.date_range(start='2024-01-01', periods=100, freq='4h')
    prices = 100 + np.random.randn(100).cumsum()
    df = pd.DataFrame({'close': prices, 'timestamp': dates})
    
    print("Creating dummy trades...")
    trades = []
    # Win trade
    trades.append(Trade(
        symbol='BTC',
        entry_time=dates[10],
        entry_price=prices[10],
        exit_time=dates[20],
        exit_price=prices[20],
        direction='LONG',
        pnl=10,
        pnl_pct=0.1
    ))
    
    # Loss trade
    trades.append(Trade(
        symbol='BTC',
        entry_time=dates[50],
        entry_price=prices[50],
        exit_time=dates[60],
        exit_price=prices[60],
        direction='LONG',
        pnl=-5,
        pnl_pct=-0.05
    ))
    
    print("Plotting...")
    save_path = "ml/results/backtest_trades_4h_dummy.png"
    plot_backtest_trades(df, trades, title="Test Plot", save_path=save_path)
    
    if Path(save_path).exists():
        print(f"✅ Success! Plot saved at {save_path}")
    else:
        print("❌ Failed to save plot")

if __name__ == "__main__":
    test_plot()
