#!/usr/bin/env python3
"""
Peak-to-Drawdown Analysis
=========================
Analyzes equity curve peaks and the drawdown patterns that follow.
Key questions:
  - After hitting a new all-time high, how deep/long is the subsequent drawdown?
  - Is there a predictable pattern (e.g., always DD 30% before resuming)?
  - When should you cash out / reduce exposure after a peak?

Usage:
  python ml/analyze_peaks.py --input ml/time_equity_1d_10.0x_isolated.csv
  python ml/analyze_peaks.py --backtest --start 2025-01-02 --end 2026-02-22 --leverage 10 --max-positions 13
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime


def load_equity_from_csv(csv_path: str) -> pd.DataFrame:
    """Load equity curve from a saved CSV file."""
    df = pd.read_csv(csv_path, parse_dates=['date'])
    return df


def load_equity_from_backtest(args) -> pd.DataFrame:
    """Run backtest and extract MtM equity curve."""
    from backtest_3stage import ThreeStageBacktester, BacktestConfig
    
    DATA_DIR = Path(__file__).parent.parent / 'bitget-data'
    features_path = DATA_DIR / 'processed' / f'features_{args.timeframe}_full.parquet'
    
    df = pd.read_parquet(features_path)
    mask = (df['timestamp'] >= args.start) & (df['timestamp'] <= args.end)
    df = df[mask]
    
    config = BacktestConfig(
        leverage=args.leverage,
        initial_capital=100.0,
        max_open_trades=args.max_positions,
        entry_threshold=args.threshold,
        use_scanner_filter=args.use_scanner,
        timeframe=args.timeframe,
    )
    
    bt = ThreeStageBacktester(config)
    result = bt.run_backtest(df, verbose=False)
    
    # Extract equity timeline
    if result.equity_curve and result.timestamps:
        return pd.DataFrame({
            'date': result.timestamps[:len(result.equity_curve)],
            'equity': result.equity_curve
        })
    
    return pd.DataFrame()


def analyze_peaks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify all equity peaks (new all-time highs) and measure subsequent drawdowns.
    
    Returns DataFrame with one row per peak:
      - peak_date, peak_equity
      - dd_depth (max % drawdown after this peak)
      - dd_trough_date (when the trough occurred)
      - dd_duration_days (days from peak to trough)
      - recovery_date (when equity recovered back to peak level, or NaT)
      - recovery_days (days from peak to recovery)
      - peak_to_peak_days (days since last peak)
      - gain_since_last_peak_pct (% gain from last peak to this peak)
    """
    equity = df['equity'].values
    dates = df['date'].values
    
    peaks = []
    running_max = -np.inf
    last_peak_idx = None
    last_peak_equity = None
    
    for i in range(len(equity)):
        if equity[i] > running_max:
            running_max = equity[i]
            
            # New ATH found
            peak = {
                'peak_date': dates[i],
                'peak_equity': equity[i],
                'peak_idx': i,
            }
            
            # Distance from last peak
            if last_peak_idx is not None:
                peak['peak_to_peak_days'] = (pd.Timestamp(dates[i]) - pd.Timestamp(dates[last_peak_idx])).days
                peak['gain_since_last_peak_pct'] = (equity[i] - last_peak_equity) / last_peak_equity * 100
            else:
                peak['peak_to_peak_days'] = 0
                peak['gain_since_last_peak_pct'] = 0
            
            last_peak_idx = i
            last_peak_equity = equity[i]
            peaks.append(peak)
    
    # For each peak, find the subsequent drawdown
    for p in peaks:
        pidx = p['peak_idx']
        peak_eq = p['peak_equity']
        
        # Look forward from this peak to the NEXT peak (or end of data)
        future_eq = equity[pidx:]
        future_dates = dates[pidx:]
        
        # Calculate drawdown from this peak
        dd_pct = (peak_eq - future_eq) / peak_eq * 100
        
        # Find maximum drawdown BEFORE next recovery
        max_dd_idx = np.argmax(dd_pct)
        p['dd_depth'] = dd_pct[max_dd_idx]
        p['dd_trough_date'] = future_dates[max_dd_idx]
        p['dd_duration_days'] = (pd.Timestamp(future_dates[max_dd_idx]) - pd.Timestamp(dates[pidx])).days
        
        # Find recovery (when equity returns to peak level)
        recovery_mask = future_eq[max_dd_idx:] >= peak_eq
        if recovery_mask.any():
            rec_idx = max_dd_idx + np.argmax(recovery_mask)
            p['recovery_date'] = future_dates[rec_idx]
            p['recovery_days'] = (pd.Timestamp(future_dates[rec_idx]) - pd.Timestamp(dates[pidx])).days
        else:
            p['recovery_date'] = pd.NaT
            p['recovery_days'] = np.nan
    
    result = pd.DataFrame(peaks)
    # Drop the helper column
    result = result.drop(columns=['peak_idx'])
    
    return result


def filter_significant_peaks(peaks_df: pd.DataFrame, min_gain_pct: float = 5.0) -> pd.DataFrame:
    """
    Filter to only significant peaks (where the gain from last peak was meaningful).
    Tiny peaks (0.1% gain) create noise — focus on real ATH breakouts.
    """
    # First peak is always significant
    mask = (peaks_df['gain_since_last_peak_pct'] >= min_gain_pct) | (peaks_df.index == 0)
    return peaks_df[mask].reset_index(drop=True)


def print_peak_report(peaks_df: pd.DataFrame, sig_peaks: pd.DataFrame):
    """Print analysis report."""
    print(f"\n{'='*90}")
    print(f"📊 PEAK-TO-DRAWDOWN ANALYSIS")
    print(f"{'='*90}")
    
    print(f"\n  Total peaks (new ATH): {len(peaks_df)}")
    print(f"  Significant peaks (>{5}% gain): {len(sig_peaks)}")
    
    if len(sig_peaks) == 0:
        print("  No significant peaks found.")
        return
    
    # Overall statistics
    print(f"\n{'─'*90}")
    print(f"  📈 SIGNIFICANT PEAK STATISTICS")
    print(f"{'─'*90}")
    
    dd = sig_peaks['dd_depth']
    print(f"\n  Drawdown after peak:")
    print(f"    Mean:   {dd.mean():.1f}%")
    print(f"    Median: {dd.median():.1f}%")
    print(f"    Min:    {dd.min():.1f}%")
    print(f"    Max:    {dd.max():.1f}%")
    print(f"    Std:    {dd.std():.1f}%")
    
    dur = sig_peaks['dd_duration_days']
    print(f"\n  Time to trough (days):")
    print(f"    Mean:   {dur.mean():.0f}")
    print(f"    Median: {dur.median():.0f}")
    print(f"    Min:    {dur.min():.0f}")
    print(f"    Max:    {dur.max():.0f}")
    
    rec = sig_peaks['recovery_days'].dropna()
    if len(rec) > 0:
        print(f"\n  Recovery time (days):")
        print(f"    Mean:   {rec.mean():.0f}")
        print(f"    Median: {rec.median():.0f}")
        print(f"    Min:    {rec.min():.0f}")
        print(f"    Max:    {rec.max():.0f}")
        print(f"    Never recovered: {sig_peaks['recovery_days'].isna().sum()} peaks")
    
    p2p = sig_peaks['peak_to_peak_days']
    p2p_valid = p2p[p2p > 0]
    if len(p2p_valid) > 0:
        print(f"\n  Peak-to-peak interval (days):")
        print(f"    Mean:   {p2p_valid.mean():.0f}")
        print(f"    Median: {p2p_valid.median():.0f}")
    
    gain = sig_peaks['gain_since_last_peak_pct']
    gain_valid = gain[gain > 0]
    if len(gain_valid) > 0:
        print(f"\n  Gain at each new peak:")
        print(f"    Mean:   {gain_valid.mean():.1f}%")
        print(f"    Median: {gain_valid.median():.1f}%")
    
    # Detailed table
    print(f"\n{'─'*90}")
    print(f"  📋 PEAK DETAIL TABLE")
    print(f"{'─'*90}")
    print(f"  {'DATE':<12} {'EQUITY':>12} {'GAIN%':>8} {'DD DEPTH':>10} {'DD DAYS':>8} {'RECOVERY':>10} {'REC DAYS':>10}")
    print(f"  {'─'*12} {'─'*12} {'─'*8} {'─'*10} {'─'*8} {'─'*10} {'─'*10}")
    
    for _, row in sig_peaks.iterrows():
        dt = pd.Timestamp(row['peak_date']).strftime('%Y-%m-%d')
        eq = f"${row['peak_equity']:,.0f}"
        gain = f"+{row['gain_since_last_peak_pct']:.1f}%" if row['gain_since_last_peak_pct'] > 0 else "—"
        dd = f"-{row['dd_depth']:.1f}%"
        dd_days = f"{row['dd_duration_days']:.0f}d"
        rec_dt = pd.Timestamp(row['recovery_date']).strftime('%Y-%m-%d') if pd.notna(row['recovery_date']) else "NEVER"
        rec_days = f"{row['recovery_days']:.0f}d" if pd.notna(row['recovery_days']) else "∞"
        print(f"  {dt:<12} {eq:>12} {gain:>8} {dd:>10} {dd_days:>8} {rec_dt:>10} {rec_days:>10}")
    
    # Actionable insights
    print(f"\n{'─'*90}")
    print(f"  💡 ACTIONABLE INSIGHTS")
    print(f"{'─'*90}")
    
    median_dd_val = sig_peaks['dd_depth'].median()
    mean_dd_days = dur.mean() if len(dur) > 0 else 0
    mean_rec_days = rec.mean() if len(rec) > 0 else 0
    
    # Check if DD > 20% happens consistently
    deep_dd_pct = (sig_peaks['dd_depth'] > 20).mean() * 100
    print(f"\n  After hitting a new ATH:")
    print(f"    • {deep_dd_pct:.0f}% of the time, a >20% drawdown follows")
    print(f"    • Median drawdown: {median_dd_val:.1f}%")
    print(f"    • Average duration: {mean_dd_days:.0f} days to trough, {mean_rec_days:.0f} days to recover")
    
    if median_dd_val > 15:
        print(f"\n  🎯 RECOMMENDATION: After new ATH, consider withdrawing profits.")
        print(f"     Expected DD: ~{median_dd_val:.0f}% within ~{mean_dd_days:.0f} days.")
        print(f"     Wait ~{mean_rec_days:.0f} days for recovery before re-entering fully.")
    
    print(f"\n{'='*90}")


def plot_peak_analysis(df: pd.DataFrame, sig_peaks: pd.DataFrame, output_dir: Path):
    """Create visualization of peak-to-drawdown patterns."""
    fig, axes = plt.subplots(3, 2, figsize=(20, 16))
    fig.suptitle('Peak-to-Drawdown Analysis', fontsize=16, fontweight='bold')
    
    # ── 1. Equity curve with peaks and troughs ──────────────────────────
    ax1 = axes[0, 0]
    ax1.plot(df['date'], df['equity'], 'b-', alpha=0.7, linewidth=1, label='Equity')
    
    # Mark significant peaks
    for _, p in sig_peaks.iterrows():
        ax1.axvline(x=pd.Timestamp(p['peak_date']), color='green', alpha=0.3, linestyle='--')
        ax1.plot(pd.Timestamp(p['peak_date']), p['peak_equity'], 'g^', markersize=8)
        if pd.notna(p['dd_trough_date']):
            trough_eq = p['peak_equity'] * (1 - p['dd_depth']/100)
            ax1.plot(pd.Timestamp(p['dd_trough_date']), trough_eq, 'rv', markersize=8)
    
    ax1.set_ylabel('Equity ($)')
    ax1.set_title('Equity Curve with Peaks (▲) and Troughs (▼)')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # ── 2. Drawdown depth distribution ──────────────────────────────────
    ax2 = axes[0, 1]
    if len(sig_peaks) > 0:
        dd_vals = sig_peaks['dd_depth'].values
        ax2.hist(dd_vals, bins=min(20, len(dd_vals)), color='#e74c3c', alpha=0.7, edgecolor='black')
        ax2.axvline(x=np.median(dd_vals), color='blue', linestyle='--', label=f'Median: {np.median(dd_vals):.1f}%')
        ax2.axvline(x=np.mean(dd_vals), color='orange', linestyle='--', label=f'Mean: {np.mean(dd_vals):.1f}%')
        ax2.set_xlabel('Drawdown Depth (%)')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Drawdown After Each Peak')
        ax2.legend()
    
    # ── 3. DD duration vs depth scatter ─────────────────────────────────
    ax3 = axes[1, 0]
    if len(sig_peaks) > 0:
        ax3.scatter(sig_peaks['dd_duration_days'], sig_peaks['dd_depth'], 
                   c=sig_peaks['peak_equity'], cmap='viridis', s=80, alpha=0.7, edgecolors='black')
        ax3.set_xlabel('Days to Trough')
        ax3.set_ylabel('Drawdown Depth (%)')
        ax3.set_title('DD Duration vs Depth (color = equity size)')
        ax3.grid(True, alpha=0.3)
        
        # Add colorbar
        cb = plt.colorbar(ax3.collections[0], ax=ax3)
        cb.set_label('Peak Equity ($)')
    
    # ── 4. Recovery time distribution ───────────────────────────────────
    ax4 = axes[1, 1]
    rec_days = sig_peaks['recovery_days'].dropna().values
    if len(rec_days) > 0:
        ax4.hist(rec_days, bins=min(15, len(rec_days)), color='#2ecc71', alpha=0.7, edgecolor='black')
        ax4.axvline(x=np.median(rec_days), color='blue', linestyle='--', label=f'Median: {np.median(rec_days):.0f}d')
        ax4.set_xlabel('Recovery Time (days)')
        ax4.set_ylabel('Count')
        ax4.set_title('How Long to Recover After Drawdown?')
        ax4.legend()
    
    never = sig_peaks['recovery_days'].isna().sum()
    if never > 0:
        ax4.text(0.95, 0.95, f'{never} peaks never recovered', transform=ax4.transAxes,
                ha='right', va='top', fontsize=10, color='red',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ── 5. Peak-to-peak gain vs subsequent DD ───────────────────────────
    ax5 = axes[2, 0]
    if len(sig_peaks) > 1:
        valid = sig_peaks[sig_peaks['gain_since_last_peak_pct'] > 0]
        if len(valid) > 0:
            ax5.scatter(valid['gain_since_last_peak_pct'], valid['dd_depth'],
                       s=80, alpha=0.7, color='#e67e22', edgecolors='black')
            ax5.set_xlabel('Gain Since Last Peak (%)')
            ax5.set_ylabel('Subsequent Drawdown (%)')
            ax5.set_title('Bigger Gain → Bigger Drawdown?')
            ax5.grid(True, alpha=0.3)
            
            # Add correlation line
            if len(valid) >= 3:
                z = np.polyfit(valid['gain_since_last_peak_pct'], valid['dd_depth'], 1)
                x_line = np.linspace(valid['gain_since_last_peak_pct'].min(), valid['gain_since_last_peak_pct'].max(), 50)
                ax5.plot(x_line, np.polyval(z, x_line), 'r--', alpha=0.5, label=f'Trend (slope={z[0]:.3f})')
                corr = valid['gain_since_last_peak_pct'].corr(valid['dd_depth'])
                ax5.legend(title=f'Correlation: {corr:.2f}')
    
    # ── 6. Equity cycle visualization ───────────────────────────────────
    ax6 = axes[2, 1]
    if len(sig_peaks) > 0:
        # Overlay normalized drawdown curves from each peak
        max_days = int(sig_peaks['dd_duration_days'].max()) + 10 if sig_peaks['dd_duration_days'].max() > 0 else 30
        max_days = min(max_days, 90)  # Cap at 90 days
        
        for _, p in sig_peaks.iterrows():
            peak_date = pd.Timestamp(p['peak_date'])
            peak_eq = p['peak_equity']
            
            # Get equity data after this peak
            mask = (df['date'] >= peak_date) & (df['date'] <= peak_date + pd.Timedelta(days=max_days))
            future = df[mask].copy()
            
            if len(future) > 1:
                days_from_peak = [(pd.Timestamp(d) - peak_date).days for d in future['date']]
                dd_from_peak = (peak_eq - future['equity'].values) / peak_eq * 100
                ax6.plot(days_from_peak, dd_from_peak, alpha=0.3, linewidth=1)
        
        # Add average line
        ax6.axhline(y=0, color='green', linestyle='-', alpha=0.5, label='Peak level')
        if len(sig_peaks) > 0:
            median_dd = sig_peaks['dd_depth'].median()
            ax6.axhline(y=median_dd, color='red', linestyle='--', alpha=0.7, label=f'Median DD: {median_dd:.1f}%')
        
        ax6.set_xlabel('Days After Peak')
        ax6.set_ylabel('Drawdown from Peak (%)')
        ax6.set_title('Overlay: All Drawdown Paths After Each Peak')
        ax6.invert_yaxis()
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'peak_drawdown_analysis.png'
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\n💾 Plot saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Peak-to-Drawdown Analysis')
    
    # Input source
    parser.add_argument('--input', type=str, help='Path to equity CSV file')
    parser.add_argument('--backtest', action='store_true', help='Run backtest to generate equity')
    
    # Backtest params (if --backtest)
    parser.add_argument('--start', type=str, default='2025-01-02')
    parser.add_argument('--end', type=str, default='2026-02-22')
    parser.add_argument('--timeframe', type=str, default='1d')
    parser.add_argument('--leverage', type=float, default=10)
    parser.add_argument('--threshold', type=float, default=0.6)
    parser.add_argument('--max-positions', type=int, default=13)
    parser.add_argument('--use-scanner', action='store_true', default=True)
    
    # Analysis params
    parser.add_argument('--min-gain', type=float, default=5.0, help='Min %% gain to count as significant peak')
    parser.add_argument('--no-plot', action='store_true')
    
    args = parser.parse_args()
    
    # Load equity data
    if args.input:
        print(f"📂 Loading equity from: {args.input}")
        df = load_equity_from_csv(args.input)
    elif args.backtest:
        print(f"🚀 Running backtest: {args.start} → {args.end}")
        df = load_equity_from_backtest(args)
    else:
        # Try default CSV
        default_csv = Path(__file__).parent / 'results' / 'time_equity_1d_10.0x_isolated.csv'
        if default_csv.exists():
            print(f"📂 Loading default: {default_csv}")
            df = load_equity_from_csv(str(default_csv))
        else:
            print("❌ No input specified. Use --input or --backtest")
            return
    
    if len(df) == 0:
        print("❌ No equity data")
        return
    
    # Ensure column names
    if 'equity' not in df.columns:
        # Try common alternatives
        for col in ['mtm_equity', 'realized_equity', 'Equity']:
            if col in df.columns:
                df['equity'] = df[col]
                break
    
    print(f"   Loaded {len(df)} data points: {df['date'].min()} → {df['date'].max()}")
    
    # Run analysis
    print(f"\n🔍 Analyzing equity peaks...")
    all_peaks = analyze_peaks(df)
    sig_peaks = filter_significant_peaks(all_peaks, min_gain_pct=args.min_gain)
    
    # Print report
    print_peak_report(all_peaks, sig_peaks)
    
    # Save peaks data
    output_dir = Path(args.input).parent if args.input else Path(__file__).parent
    peaks_path = output_dir / 'peak_analysis.csv'
    sig_peaks.to_csv(peaks_path, index=False)
    print(f"\n💾 Peak data saved: {peaks_path}")
    
    # Plot
    if not args.no_plot:
        print(f"\n📊 Generating peak analysis plots...")
        plot_peak_analysis(df, sig_peaks, output_dir)
    
    print(f"\n✅ Peak analysis complete!")


if __name__ == '__main__':
    main()
