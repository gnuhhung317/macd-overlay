import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================
# CONFIG & PATHS
# ============================================================
BASE_DIR = Path(r"d:\Code\Projects\self-projects\macd-overlay - Copy")
RESULTS_FILE = BASE_DIR / "ml" / "outlier_results_full.csv"
OUTPUT_DIR = BASE_DIR / "ml" / "analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

def visualize_strike_zone():
    if not RESULTS_FILE.exists():
        print(f"❌ Results file not found: {RESULTS_FILE}")
        return

    df = pd.read_csv(RESULTS_FILE)
    if df.empty:
        print("❌ Dataset is empty.")
        return

    # Filter for signals that hit MFE within horizon (already done in scanner but for safety)
    df = df.dropna(subset=['mfe_atr', 'bars_to_mfe', 'mae_atr'])
    
    # 1. Scatter Plot: MFE vs Bars_to_MFE
    plt.figure(figsize=(12, 8))
    sns.set_style("darkgrid")
    
    # Cap MFE for better visualization (outliers can be huge)
    mfe_cap = 20
    plot_df = df.copy()
    plot_df['mfe_display'] = plot_df['mfe_atr'].clip(upper=mfe_cap)
    
    scatter = sns.scatterplot(
        data=plot_df, 
        x='bars_to_mfe', 
        y='mfe_display', 
        hue='type', 
        palette={'LONG': '#2ebd85', 'SHORT': '#f6465d'},
        alpha=0.5,
        s=40
    )
    
    plt.axvline(x=24, color='orange', linestyle='--', label='Flash Zone (24h)')
    plt.axvline(x=90, color='purple', linestyle='--', label='Trend Zone (90h)')
    plt.axhline(y=6.0, color='red', linestyle=':', label='Elite Threshold (6.0 ATR)')
    
    plt.title("The Strike Zone: MFE vs Time (Bars)", fontsize=15)
    plt.xlabel("Bars to reach peak (Time)", fontsize=12)
    plt.ylabel("Maximum Favorable Excursion (ATR)", fontsize=12)
    plt.legend()
    
    scatter_path = OUTPUT_DIR / "strike_zone_scatter.png"
    plt.savefig(scatter_path, dpi=300)
    print(f"✅ Saved Scatter Plot: {scatter_path}")
    plt.close()

    # 2. Histogram: MAE Distribution
    plt.figure(figsize=(12, 8))
    
    # Focus on "Elite" candidates (MFE >= 6.0) to see their MAE profile
    elite_df = df[df['mfe_atr'] >= 6.0]
    
    # MAE is negative, we want to see how "deep" it goes
    sns.histplot(
        elite_df['mae_atr'], 
        bins=50, 
        kde=True, 
        color='#3498db'
    )
    
    # Calculate Percentiles for Safe SL
    p95 = elite_df['mae_atr'].quantile(0.05) # 95% of elite trades have MAE > this
    p90 = elite_df['mae_atr'].quantile(0.10)
    
    plt.axvline(x=p95, color='red', linestyle='--', label=f'95th Percentile ({p95:.2f} ATR)')
    plt.axvline(x=-1.2, color='green', linestyle=':', label='Suggested SL (-1.2 ATR)')
    
    plt.title("The Safety Net: MAE Distribution of Elite Outliers", fontsize=15)
    plt.xlabel("Maximum Adverse Excursion (ATR)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    
    hist_path = OUTPUT_DIR / "mae_distribution.png"
    plt.savefig(hist_path, dpi=300)
    print(f"✅ Saved Histogram: {hist_path}")
    plt.close()
    
    # 3. Stats Summary
    print("\n--- Strike Zone Analysis ---")
    print(f"95% of Elite trades never drop below: {p95:.2f} ATR")
    print(f"90% of Elite trades never drop below: {p90:.2f} ATR")
    
    flash_trades = elite_df[elite_df['bars_to_mfe'] <= 24]
    trend_trades = elite_df[elite_df['bars_to_mfe'] > 90]
    
    print(f"\nFlash Trades (<= 24h): {len(flash_trades)}")
    print(f"Trend Trades (> 90h):  {len(trend_trades)}")

if __name__ == "__main__":
    visualize_strike_zone()
