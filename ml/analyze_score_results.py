#!/usr/bin/env python3
"""
Advanced Quantitative Analysis of Score Sweep Results.
Parses window-level data to evaluate distributions, risk-adjusted returns, 
and strategy robustness across 8h, 12h, and 1d timeframes.
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / 'results'

def load_window_data():
    """Extract granular window-by-window data for distribution analysis."""
    files = {
        '8h': RESULTS_DIR / 'score_sweep_8h.json',
        '12h': RESULTS_DIR / 'score_sweep_12h.json',
        '1d': RESULTS_DIR / 'score_sweep_1d.json'
    }
    
    records = []
    for tf, path in files.items():
        if not path.exists():
            print(f"Warning: {path} not found.")
            continue
            
        with open(path, 'r') as f:
            data = json.load(f)
            # Parse each rolling window instead of just the summary
            for w in data.get('windows', []):
                w_start = w['start']
                for label, metrics in w['results'].items():
                    records.append({
                        'Timeframe': tf,
                        'Window': w_start,
                        'Signal_Tier': label,
                        'Return_Pct': metrics.get('return_pct', 0),
                        'Max_DD_Pct': metrics.get('max_dd_pct', 0),
                        'Win_Rate': metrics.get('win_rate_pct', 0),
                        'Trades': metrics.get('trades', 0),
                        'Profit_Factor': metrics.get('profit_factor', 0)
                    })
                    
    return pd.DataFrame(records)

def calculate_quant_metrics(df):
    """Calculate advanced quantitative metrics per timeframe and tier."""
    # Group by timeframe and signal tier
    grouped = df.groupby(['Timeframe', 'Signal_Tier'])
    
    quant_stats = grouped.agg(
        Total_Windows=('Window', 'count'),
        Avg_Return=('Return_Pct', 'mean'),
        Median_Return=('Return_Pct', 'median'),
        Return_Std=('Return_Pct', 'std'),  # Volatility across windows
        Worst_DD=('Max_DD_Pct', 'max'),
        Median_DD=('Max_DD_Pct', 'median'),
        Avg_DD=('Max_DD_Pct', 'mean'),
        Avg_WinRate=('Win_Rate', 'mean'),
        Avg_Trades=('Trades', 'mean')
    ).reset_index()

    # Probability of a profitable window (Consistency)
    profitable_windows = df[df['Return_Pct'] > 0].groupby(['Timeframe', 'Signal_Tier']).size().reset_index(name='Profitable_Count')
    quant_stats = quant_stats.merge(profitable_windows, on=['Timeframe', 'Signal_Tier'], how='left')
    quant_stats['Profit_Probability'] = (quant_stats['Profitable_Count'] / quant_stats['Total_Windows'] * 100).fillna(0)
    
    # Return to Drawdown Ratio (Calmar Proxy)
    # Handle division by zero if Worst_DD is 0
    quant_stats['Calmar_Proxy'] = np.where(
        quant_stats['Worst_DD'] == 0, 
        np.inf, 
        quant_stats['Avg_Return'] / quant_stats['Worst_DD']
    )
    
    return quant_stats

def generate_quant_report(quant_df):
    """Generate a Markdown report focusing on risk-adjusted performance."""
    if quant_df is None or quant_df.empty:
        return
        
    report_path = RESULTS_DIR / 'quant_analysis_report.md'
    
    # Format numeric columns for clean markdown output
    format_cols = ['Avg_Return', 'Median_Return', 'Return_Std', 'Worst_DD', 'Median_DD', 'Avg_DD', 'Profit_Probability', 'Calmar_Proxy']
    for col in format_cols:
        quant_df[col] = quant_df[col].round(2)

    pivot_calmar = quant_df.pivot(index='Signal_Tier', columns='Timeframe', values='Calmar_Proxy')
    pivot_prob = quant_df.pivot(index='Signal_Tier', columns='Timeframe', values='Profit_Probability')
    pivot_std = quant_df.pivot(index='Signal_Tier', columns='Timeframe', values='Return_Std')
    pivot_med_dd = quant_df.pivot(index='Signal_Tier', columns='Timeframe', values='Median_DD')
    
    with open(report_path, 'w') as f:
        f.write("# Quantitative Strategy Analysis\n\n")
        
        f.write("## 1. Return/Risk Ratio (Calmar Proxy)\n")
        f.write("> Ratio of Average Return to Maximum Drawdown. Higher is better.\n\n")
        f.write(pivot_calmar.to_markdown() + "\n\n")
        
        f.write("## 2. Profit Probability (% of Profitable Windows)\n")
        f.write("> Consistency metric: How often does this config survive a 90-day period with >0% return?\n\n")
        f.write(pivot_prob.to_markdown() + "\n\n")
        
        f.write("## 3. Return Volatility (Standard Deviation)\n")
        f.write("> Represents the variance in returns across different market regimes. Lower means more stable.\n\n")
        f.write(pivot_std.to_markdown() + "\n\n")

        f.write("## 4. Median Drawdown (%) Across Timeframes\n")
        f.write("> Typical risk profile experienced in a window.\n\n")
        f.write(pivot_med_dd.to_markdown() + "\n\n")
        
        f.write("## 5. Full Master Data Table\n\n")
        f.write(quant_df.to_markdown(index=False) + "\n\n")
        
    print(f"Quant report generated at: {report_path}")

def plot_distributions(df):
    """Create advanced distribution and scatter plots for quant evaluation."""
    if df is None or df.empty:
        return
        
    sns.set_theme(style="whitegrid")
    
    # Create a 2x2 grid of plots
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Quantitative Distribution & Risk Analysis', fontsize=18, fontweight='bold', y=0.98)
    
    # Plot 1: Return Distribution (Boxplot with strip overlay for outlier visibility)
    sns.boxplot(data=df, x='Signal_Tier', y='Return_Pct', hue='Timeframe', ax=axes[0, 0], showfliers=False, palette='viridis')
    axes[0, 0].set_title('Distribution of Returns Across Windows (Excl. Extreme Outliers)')
    axes[0, 0].set_ylabel('Return (%)')
    
    # Plot 2: Drawdown Distribution (Violin Plot for density estimation)
    sns.violinplot(data=df, x='Signal_Tier', y='Max_DD_Pct', hue='Timeframe', ax=axes[0, 1], split=False, inner='quartile', palette='flare')
    axes[0, 1].set_title('Density of Maximum Drawdowns')
    axes[0, 1].set_ylabel('Max Drawdown (%)')
    
    # Plot 3: Risk vs Reward Scatter (Median Return vs Worst DD)
    # We aggregate this specific view to avoid plotting every single window
    agg_df = df.groupby(['Timeframe', 'Signal_Tier']).agg({'Return_Pct': 'median', 'Max_DD_Pct': 'max'}).reset_index()
    # CLIP Max_DD at 100% for the scatter plot visual
    agg_df['Max_DD_Pct'] = agg_df['Max_DD_Pct'].clip(upper=100)
    
    sns.scatterplot(data=agg_df, x='Max_DD_Pct', y='Return_Pct', hue='Timeframe', style='Signal_Tier', s=200, ax=axes[1, 0], palette='deep')
    axes[1, 0].set_title('Risk vs Reward (Worst Drawdown vs Median Return)')
    axes[1, 0].set_xlabel('Worst Case Drawdown (%)')
    axes[1, 0].set_ylabel('Median Return (%)')
    axes[1, 0].set_xlim(0, 105) # Cap at 100% (+ padding)
    
    # Draw a line representing a 1:1 risk/reward ratio for visual reference
    lims = [0, 100]
    axes[1, 0].plot(lims, lims, 'k--', alpha=0.3, zorder=0)
    
    # Plot 4: Win Rate Consistency
    sns.boxplot(data=df, x='Signal_Tier', y='Win_Rate', hue='Timeframe', ax=axes[1, 1], palette='crest')
    axes[1, 1].set_title('Win Rate Consistency Across Windows')
    axes[1, 1].set_ylabel('Win Rate (%)')
    axes[1, 1].axhline(50, color='r', linestyle='--', alpha=0.5) # Break-even visual line

    plt.tight_layout()
    plot_path = RESULTS_DIR / 'quant_distribution_analysis.png'
    plt.savefig(plot_path, dpi=150)
    print(f"Distribution plots saved to: {plot_path}")

def main():
    print("Extracting window-level data for Quant Analysis...")
    df_windows = load_window_data()
    
    if df_windows.empty:
        print("No valid window data found. Check your JSON files.")
        return
        
    quant_df = calculate_quant_metrics(df_windows)
    
    print("Generating statistical distributions and reports...")
    generate_quant_report(quant_df)
    plot_distributions(df_windows)
    
    print("\nQuant Analysis Complete.")

if __name__ == "__main__":
    main()