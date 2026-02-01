
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Fix paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.backtest_timeframes import run_timeframe_backtest, BacktestConfig

def analyze_win_loss_conf():
    config = BacktestConfig(initial_capital=10000)
    config.start_date = '2024-01-01'
    config.end_date = '2024-06-01'
    
    print("Running 12h analysis (Jan-Jun 2024)...")
    result, _ = run_timeframe_backtest('12h', config)
    
    if not result or not result.trades:
        print("No trades found.")
        return
        
    print(f"Analyzing {len(result.trades)} trades...")
    data = []
    for t in result.trades:
        print(f"Trade: {t.symbol} {t.direction} Conf: {t.confidence:.3f} Win: {1 if t.pnl > 0 else 0}")
        data.append({
            'confidence': t.confidence,
            'pnl': t.pnl,
            'is_win': 1 if t.pnl > 0 else 0
        })
        
    df = pd.DataFrame(data)
    
    win_stats = df[df['is_win'] == 1]['confidence'].describe()
    loss_stats = df[df['is_win'] == 0]['confidence'].describe()
    
    print("\nWINNING TRADES Confidence:")
    print(win_stats)
    
    print("\nLOSING TRADES Confidence:")
    print(loss_stats)
    
    # Check correlation
    corr = df['confidence'].corr(df['is_win'])
    print(f"\nCorrelation (Confidence vs Win): {corr:.4f}")

if __name__ == "__main__":
    analyze_win_loss_conf()
