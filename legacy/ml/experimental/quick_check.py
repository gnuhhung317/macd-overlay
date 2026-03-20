
import sys
import pandas as pd
from pathlib import Path

# Fix paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.backtest_timeframes import run_timeframe_backtest, BacktestConfig

def quick_check():
    config = BacktestConfig(initial_capital=10000)
    config.start_date = '2024-01-01'
    config.end_date = '2024-02-01'
    
    result, _ = run_timeframe_backtest('12h', config)
    
    if not result: return
    
    wins = [t.confidence for t in result.trades if t.pnl > 0]
    losses = [t.confidence for t in result.trades if t.pnl <= 0]
    
    print(f"Wins: {len(wins)}, Mean Conf: {sum(wins)/len(wins) if wins else 0:.3f}")
    if losses:
        print(f"Losses: {len(losses)}, Mean Conf: {sum(losses)/len(losses):.3f}")
        print(f"Max Loss Conf: {max(losses):.3f}")

if __name__ == "__main__":
    quick_check()
