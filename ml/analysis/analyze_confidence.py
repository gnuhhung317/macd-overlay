
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Fix paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest_3stage import ThreeStageBacktester, BacktestConfig

def analyze_confidence_correlation(timeframe='4h', start_date='2024-01-01', end_date=None):
    print(f"Running Analysis for {timeframe} from {start_date}...")
    
    # Run backtest
    from backtesting.backtest_timeframes import run_timeframe_backtest, BacktestConfig
    
    config = BacktestConfig(initial_capital=10000)
    config.start_date = start_date
    config.end_date = end_date
    result, df = run_timeframe_backtest(timeframe, config)
    
    if not result or not result.trades:
        print("No trades found.")
        return

    # Create DataFrame from trades
    trades_data = []
    for t in result.trades:
        trades_data.append({
            'confidence': t.confidence,
            'pnl_pct': t.pnl_pct,
            'is_win': 1 if t.pnl > 0 else 0,
            'direction': t.direction,
            'bars_held': t.bars_held
        })
    
    df_trades = pd.DataFrame(trades_data)
    
    # 1. Bucketize Confidence
    # We expect confidence > 0.65 (threshold)
    bins = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    labels = ['60-65%', '65-70%', '70-75%', '75-80%', '80-85%', '85-90%', '90-95%', '95-100%']
    df_trades['conf_bucket'] = pd.cut(df_trades['confidence'], bins=bins, labels=labels)
    
    # 2. Group Stats
    stats = df_trades.groupby('conf_bucket').agg({
        'is_win': ['count', 'mean'],
        'pnl_pct': 'mean'
    })
    stats.columns = ['Count', 'Win_Rate', 'Avg_Return']
    
    print("\nConfidence Analysis:")
    print(stats)
    
    # 3. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Win Rate by Bucket
    sns.barplot(x=stats.index, y=stats['Win_Rate'], ax=ax1, palette='viridis')
    ax1.set_title('Win Rate by Confidence Level')
    ax1.set_ylabel('Win Rate')
    ax1.set_ylim(0, 1.0)
    ax1.axhline(0.5, color='red', linestyle='--')
    
    # Return by Bucket
    sns.barplot(x=stats.index, y=stats['Avg_Return'], ax=ax2, palette='magma')
    ax2.set_title('Avg Return by Confidence Level')
    ax2.set_ylabel('Avg Return')
    
    plt.tight_layout()
    save_path = f'ml/results/confidence_analysis_{timeframe}.png'
    plt.savefig(save_path)
    print(f"Chart saved to {save_path}")
    
    # Correlation
    corr_win = df_trades['confidence'].corr(df_trades['is_win'])
    corr_ret = df_trades['confidence'].corr(df_trades['pnl_pct'])
    
    print(f"\nCorrelation (Confidence vs Win): {corr_win:.4f}")
    print(f"Correlation (Confidence vs Return): {corr_ret:.4f}")
    
    return stats

if __name__ == "__main__":
    timeframes = ['4h', '8h', '12h', '1d']
    for tf in timeframes:
        try:
            analyze_confidence_correlation(timeframe=tf, start_date='2024-01-01', end_date='2024-06-01')
        except Exception as e:
            print(f"Error analyzing {tf}: {e}")
