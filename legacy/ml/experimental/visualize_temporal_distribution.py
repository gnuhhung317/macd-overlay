
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob

def visualize_distribution():
    print("Aggregating trade logs for temporal analysis...")
    files = glob.glob("ml/results/dynamic_trades_*.csv")
    if not files:
        print("No trade logs found. Run test_dynamic_filters.py first.")
        return

    all_trades = []
    for f in files:
        tf = Path(f).stem.split('_')[-1]
        df = pd.read_csv(f)
        df['timeframe'] = tf
        all_trades.append(df)
    
    df = pd.concat(all_trades)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp', 'is_win'])
    df['is_win'] = df['is_win'].astype(int)
    df['month'] = df['timestamp'].dt.to_period('M').astype(str)
    
    # 1. TRADE COUNTS BY MONTH (Win vs Loss)
    stats = df.groupby(['month', 'is_win']).size().unstack(fill_value=0)
    stats.columns = ['Loss', 'Win']
    
    plt.figure(figsize=(15, 7))
    stats.plot(kind='bar', stacked=True, color=['#e74c3c', '#2ecc71'], ax=plt.gca())
    plt.title("Distribution of Trades over Time (All Timeframes Combined)", fontsize=14, fontweight='bold')
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Number of Trades", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    
    plot_path = Path("ml/results/temporal_distribution.png")
    plt.savefig(plot_path)
    print(f"Chart saved to {plot_path}")

    # 2. WIN RATE BY MONTH
    wr_stats = df.groupby('month')['is_win'].mean()
    plt.figure(figsize=(15, 5))
    wr_stats.plot(kind='line', marker='o', color='#3498db', linewidth=2)
    plt.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='70% Threshold')
    plt.title("Win Rate Stability over Time", fontsize=14, fontweight='bold')
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Win Rate", fontsize=12)
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(range(len(wr_stats.index)), wr_stats.index, rotation=45, fontsize=8)
    plt.tight_layout()
    
    wr_path = Path("ml/results/win_rate_stability.png")
    plt.savefig(wr_path)
    print(f"Win rate chart saved to {wr_path}")

    # 3. STATISTICAL SUMMARY
    print("\n" + "="*40)
    print("MONTHLY STABILITY REPORT")
    print("="*40)
    monthly_summary = df.groupby('month').agg(
        total_trades=('is_win', 'count'),
        wins=('is_win', 'sum'),
        win_rate=('is_win', 'mean')
    )
    print(monthly_summary.tail(12).to_string())
    
    # Detect weak months
    weak_months = monthly_summary[monthly_summary['win_rate'] < 0.5]
    if not weak_months.empty:
        print("\nAlert: Months with Win Rate < 50%:")
        print(weak_months.to_string())
    else:
        print("\nStability Verified: No months identified with Win Rate < 50%.")

if __name__ == "__main__":
    visualize_distribution()
