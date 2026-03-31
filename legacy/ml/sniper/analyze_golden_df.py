import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import gc

# Add project root to sys.path
BASE_DIR = Path(r"d:\Code\Projects\self-projects\macd-overlay - Copy")
sys.path.append(str(BASE_DIR))

# Import from train_sniper (mocking imports or copying logic if needed)
# For reliability, I will redefine relevant parts or import if possible.
try:
    from ml.training.train_sniper import prepare_cascade_data_optimized, MODEL_FEATURES, INPUT_FILE
except ImportError:
    print("Could not import from train_sniper, please check paths.")
    sys.exit(1)

def analyze_distributions(df, features, target_col='label'):
    """Plot feature distributions for Long (1) vs Skip (0)"""
    print("📊 Generating feature distributions...")
    os.makedirs(BASE_DIR / "ml" / "analysis" / "plots", exist_ok=True)
    
    # Filter for Long and Skip only
    plot_df = df[df[target_col].isin([0, 1])].copy()
    plot_df[target_col] = plot_df[target_col].map({0: 'Skip', 1: 'Long'})
    
    num_features = len(features)
    cols = 4
    rows = (num_features // cols) + (1 if num_features % cols != 0 else 0)
    
    plt.figure(figsize=(20, 5 * rows))
    for i, feat in enumerate(features):
        plt.subplot(rows, cols, i + 1)
        sns.kdeplot(data=plot_df, x=feat, hue=target_col, fill=True, common_norm=False)
        plt.title(f"Distribution of {feat}")
    
    plt.tight_layout()
    plt.savefig(BASE_DIR / "ml" / "analysis" / "plots" / "feature_distributions.png")
    plt.close()

def analyze_correlations(df, features, target_col='mfe_atr'):
    """Calculate correlation between features and MFE"""
    print("🔗 Calculating correlations with MFE...")
    corr_series = df[features + [target_col]].corr(method='spearman')[target_col].drop(target_col)
    corr_df = corr_series.sort_values(ascending=False).to_frame(name='correlation_with_mfe')
    
    print("\n🔍 Top Correlations with MFE (Bay Mạnh):")
    print(corr_df.head(10))
    print("\n🔍 Bottom Correlations (Negative impact):")
    print(corr_df.tail(10))
    
    corr_df.to_csv(BASE_DIR / "ml" / "analysis" / "correlations_mfe.csv")
    return corr_df

def analyze_winrate_heatmaps(df, feat1, feat2, target_col='label'):
    """Heatmap of Long win rate vs two features"""
    print(f"🌡️ Generating heatmap for {feat1} vs {feat2}...")
    
    # Bin the features
    df['feat1_bin'] = pd.qcut(df[feat1], q=5, duplicates='drop')
    df['feat2_bin'] = pd.qcut(df[feat2], q=5, duplicates='drop')
    
    # Calculate win rate (Long / (Long + Skip))
    pivot_df = df[df[target_col].isin([0, 1])].pivot_table(
        index='feat1_bin', 
        columns='feat2_bin', 
        values=target_col,
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', fmt=".2f")
    plt.title(f"Win Rate Heatmap: {feat1} vs {feat2}")
    plt.xlabel(feat2)
    plt.ylabel(feat1)
    
    plt.savefig(BASE_DIR / "ml" / "analysis" / "plots" / f"heatmap_{feat1}_{feat2}.png")
    plt.close()

def analyze_top_performers(df, features, top_n=50):
    """Profile the top N performers by mfe_atr"""
    print(f"🚀 Profiling top {top_n} performers...")
    top_df = df.sort_values('mfe_atr', ascending=False).head(top_n)
    
    stats = pd.DataFrame({
        'feature': features,
        'top_mean': top_df[features].mean(),
        'overall_mean': df[features].mean(),
        'top_std': top_df[features].std(),
        'overall_std': df[features].std()
    })
    
    stats['diff_pct'] = (stats['top_mean'] - stats['overall_mean']) / (stats['overall_mean'] + 1e-9) * 100
    stats = stats.sort_values('diff_pct', key=abs, ascending=False)
    
    print("\n💡 Key differences in top performers vs average:")
    print(stats.head(10)[['top_mean', 'overall_mean', 'diff_pct']])
    
    stats.to_csv(BASE_DIR / "ml" / "analysis" / "top_performers_profile.csv")

if __name__ == "__main__":
    print("⏳ Loading golden_df...")
    golden_df = prepare_cascade_data_optimized(INPUT_FILE)
    
    if golden_df.empty:
        print("❌ No data found.")
        sys.exit(0)
    
    # Data cleaning
    golden_df = golden_df.dropna(subset=MODEL_FEATURES + ['label', 'mfe_atr']).copy()
    
    # 1. Distributions
    analyze_distributions(golden_df, MODEL_FEATURES)
    
    # 2. Correlations
    analyze_correlations(golden_df, MODEL_FEATURES)
    
    # 3. Heatmaps (Example: RSI vs Volume Ratio)
    analyze_winrate_heatmaps(golden_df, 'rsi_14', 'volume_ratio')
    analyze_winrate_heatmaps(golden_df, 'stoch_k', 'vol_compression')
    
    # 4. Top Performers
    analyze_top_performers(golden_df, MODEL_FEATURES)
    
    print(f"\n✅ Analysis complete. Results saved in {BASE_DIR / 'ml' / 'analysis'}")
