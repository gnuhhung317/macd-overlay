#!/usr/bin/env python3
"""
Drawdown Condition Analyzer (v2)
Analyzes WHAT MARKET CONDITIONS precede strong drawdowns.
Focuses on actionable signals: volatility, BTC trend, position count, equity momentum.

Example:
  python ml/analyze_drawdowns.py --input ml/equity_windows/all_windows_combined.csv --threshold 35
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from scipy import stats


# ─── DATA LOADING ────────────────────────────────────────────────────────────

def load_btc_prices(timeframe='1d'):
    """Load BTC price data for trend analysis."""
    data_dir = Path(__file__).parent.parent / 'bitget-data' / 'processed'
    
    for filename in [f'features_{timeframe}_full.parquet', f'features_{timeframe}.parquet']:
        path = data_dir / filename
        if path.exists():
            df = pd.read_parquet(path, columns=['timestamp', 'symbol', 'close', 'high', 'low', 'volume'])
            btc = df[df['symbol'] == 'BTCUSDT'][['timestamp', 'close', 'high', 'low', 'volume']].copy()
            if not btc.empty:
                btc = btc.sort_values('timestamp').reset_index(drop=True)
                btc['date'] = btc['timestamp'].dt.date
                
                # BTC features
                btc['btc_sma20'] = btc['close'].rolling(20).mean()
                btc['btc_sma50'] = btc['close'].rolling(50).mean()
                btc['btc_return_1d'] = btc['close'].pct_change()
                btc['btc_return_7d'] = btc['close'].pct_change(7)
                btc['btc_vol_14d'] = btc['btc_return_1d'].rolling(14).std()
                btc['btc_trend'] = np.where(btc['btc_sma20'] > btc['btc_sma50'], 'UPTREND', 'DOWNTREND')
                btc['btc_rsi'] = compute_rsi(btc['close'], 14)
                
                # Leading regime indicators
                btc['btc_adx'] = compute_adx(btc['high'], btc['low'], btc['close'], 14)
                btc['btc_chop'] = compute_choppiness(btc['high'], btc['low'], btc['close'], 14)
                btc['btc_atr'] = (btc['high'] - btc['low']).rolling(14).mean()
                btc['btc_atr_ratio'] = btc['btc_atr'] / btc['close']  # Normalized ATR
                
                return btc
    
    print("⚠️  BTC price data not found, skipping BTC analysis")
    return None


def compute_rsi(prices, period=14):
    """Compute RSI from price series."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_adx(high, low, close, period=14):
    """Compute Average Directional Index (ADX). High ADX = strong trend, Low ADX = ranging."""
    # True Range
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    # Directional Movement
    up_move = high - high.shift()
    down_move = low.shift() - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di = 100 * pd.Series(plus_dm, index=high.index).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=high.index).rolling(period).mean() / atr
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    
    return adx


def compute_choppiness(high, low, close, period=14):
    """Compute Choppiness Index (CHOP). High = choppy/ranging, Low = trending."""
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(period).sum()
    
    high_max = high.rolling(period).max()
    low_min = low.rolling(period).min()
    hl_range = high_max - low_min
    
    chop = 100 * np.log10(atr_sum / hl_range) / np.log10(period)
    return chop


# ─── FEATURE ENGINEERING ────────────────────────────────────────────────────

def compute_condition_features(df, btc_data=None):
    """Compute market condition features for each row."""
    result = df.copy()
    result['date'] = pd.to_datetime(result['date'])
    
    if 'window_idx' not in result.columns:
        result['window_idx'] = 0
    
    # ── Per-window: drawdown ────────────────────────────────────────────
    dd_pct_list = []
    for _, group in result.groupby('window_idx'):
        eq = group['equity'].values
        peak = np.maximum.accumulate(eq)
        dd = np.where(peak > 0, (peak - eq) / peak * 100, 0)
        dd_pct_list.extend(dd.tolist())
    result['drawdown_pct'] = dd_pct_list
    
    # ── Per-window: rolling features ────────────────────────────────────
    for col_name, window in [('vol_5d', 5), ('vol_10d', 10), ('vol_20d', 20)]:
        result[f'rolling_{col_name}'] = result.groupby('window_idx')['daily_return'].transform(
            lambda x: x.rolling(window, min_periods=max(2, window//2)).std()
        )
    
    # Equity momentum: rolling return over last N days
    for n in [3, 7, 14]:
        result[f'equity_momentum_{n}d'] = result.groupby('window_idx')['equity'].transform(
            lambda x: x.pct_change(n)
        )
    
    # Drawdown streak: consecutive days of drawdown > 0
    streak_list = []
    for _, group in result.groupby('window_idx'):
        streak = []
        current = 0
        for dd in group['drawdown_pct'].values:
            if dd > 0:
                current += 1
            else:
                current = 0
            streak.append(current)
        streak_list.extend(streak)
    result['dd_streak_days'] = streak_list
    
    # Daily return streak (consecutive losing days)
    lose_streak_list = []
    for _, group in result.groupby('window_idx'):
        streak = []
        current = 0
        for ret in group['daily_return'].values:
            if ret < 0:
                current += 1
            else:
                current = 0
            streak.append(current)
        lose_streak_list.extend(streak)
    result['losing_streak'] = lose_streak_list
    
    # ── Account state features ──────────────────────────────────────────
    # Floating PnL / Equity ratio (how much is unrealized)
    result['floating_ratio'] = np.where(
        result['equity'] > 0,
        result['floating_pnl'] / result['equity'],
        0
    )
    
    # Unrealized gain ratio (positive floating / equity — profits at risk)
    result['unrealized_gain_ratio'] = np.where(
        (result['equity'] > 0) & (result['floating_pnl'] > 0),
        result['floating_pnl'] / result['equity'],
        0
    )
    
    # Realized vs Equity gap (how much equity depends on floating)
    result['realized_equity_gap'] = np.where(
        result['equity'] > 0,
        (result['equity'] - result['realized_equity']) / result['equity'],
        0
    )
    
    # Floating / Realized equity ratio (how much floating exposure vs locked-in capital)
    result['floating_vs_realized'] = np.where(
        result['realized_equity'] > 0,
        result['floating_pnl'] / result['realized_equity'],
        0
    )
    
    # Realized equity momentum (based on closed trades only)
    for n in [3, 7]:
        result[f'realized_momentum_{n}d'] = result.groupby('window_idx')['realized_equity'].transform(
            lambda x: x.pct_change(n)
        )
    
    # Daily realized PnL streak (consecutive days with negative realized PnL)
    rpnl_streak_list = []
    for _, group in result.groupby('window_idx'):
        streak = []
        current = 0
        for pnl in group['daily_realized_pnl'].values:
            if pnl < 0:
                current += 1
            else:
                current = 0
            streak.append(current)
        rpnl_streak_list.extend(streak)
    result['realized_pnl_losing_streak'] = rpnl_streak_list
    
    # Position exposure: positions * floating risk
    result['position_exposure'] = result['open_positions_count'] * result['floating_ratio'].abs()
    
    # ── Merge BTC data ──────────────────────────────────────────────────
    if btc_data is not None:
        btc_daily = btc_data[['date', 'btc_sma20', 'btc_sma50', 'btc_return_1d', 
                              'btc_return_7d', 'btc_vol_14d', 'btc_trend', 'btc_rsi', 'close',
                              'btc_adx', 'btc_chop', 'btc_atr_ratio']].copy()
        btc_daily = btc_daily.rename(columns={'close': 'btc_price'})
        
        result['date_key'] = result['date'].dt.date
        btc_daily['date_key'] = btc_daily['date']
        
        result = result.merge(btc_daily.drop(columns=['date']), on='date_key', how='left')
        result = result.drop(columns=['date_key'])
        
        # Forward fill BTC data for gaps
        btc_cols = ['btc_sma20', 'btc_sma50', 'btc_return_1d', 'btc_return_7d', 
                    'btc_vol_14d', 'btc_trend', 'btc_rsi', 'btc_price',
                    'btc_adx', 'btc_chop', 'btc_atr_ratio']
        for col in btc_cols:
            if col in result.columns:
                result[col] = result[col].ffill()
    
    return result


# ─── ANALYSIS ────────────────────────────────────────────────────────────────

def analyze_condition_impact(df, condition_name, condition_col, threshold_dd):
    """
    Compare drawdown probability and severity when a condition is True vs False.
    Returns a dict of metrics.
    """
    is_dd = df['drawdown_pct'] > threshold_dd
    
    cond_true = df[df[condition_col] == True]
    cond_false = df[df[condition_col] == False]
    
    if len(cond_true) == 0 or len(cond_false) == 0:
        return None
    
    dd_rate_true = (cond_true['drawdown_pct'] > threshold_dd).mean() * 100
    dd_rate_false = (cond_false['drawdown_pct'] > threshold_dd).mean() * 100
    
    avg_dd_true = cond_true['drawdown_pct'].mean()
    avg_dd_false = cond_false['drawdown_pct'].mean()
    
    # Statistical significance (chi-square test)
    contingency = pd.crosstab(df[condition_col], is_dd)
    if contingency.shape == (2, 2):
        chi2, p_value, _, _ = stats.chi2_contingency(contingency)
    else:
        p_value = 1.0
    
    lift = dd_rate_true / dd_rate_false if dd_rate_false > 0 else 0
    
    return {
        'condition': condition_name,
        'n_true': len(cond_true),
        'n_false': len(cond_false),
        'dd_rate_when_true': dd_rate_true,
        'dd_rate_when_false': dd_rate_false,
        'lift': lift,
        'avg_dd_true': avg_dd_true,
        'avg_dd_false': avg_dd_false,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def analyze_numeric_condition(df, feature_name, threshold_dd, percentile=75):
    """
    For numeric features, compare top quantile vs rest.
    Returns analysis dict.
    """
    valid = df[feature_name].dropna()
    if len(valid) < 10:
        return None
    
    cutoff = valid.quantile(percentile / 100)
    df_temp = df.copy()
    df_temp['_cond'] = df_temp[feature_name] >= cutoff
    
    result = analyze_condition_impact(
        df_temp, 
        f"{feature_name} >= {cutoff:.4f} (top {100-percentile}%)",
        '_cond',
        threshold_dd
    )
    
    if result:
        result['cutoff_value'] = cutoff
        result['feature'] = feature_name
    
    return result


def run_full_analysis(df, threshold_dd, btc_available=False):
    """Run comprehensive condition analysis and return sorted results."""
    leading_results = []   # Market regime (BEFORE trade entry)
    lagging_results = []   # Equity-based (AFTER damage done)
    
    # ══════════════════════════════════════════════════════════════════════
    # LEADING INDICATORS — Market regime (predictive, computed before entry)
    # ══════════════════════════════════════════════════════════════════════
    
    if btc_available:
        # ADX low = weak trend = bad for trend-following
        for threshold_val in [20, 25]:
            if 'btc_adx' in df.columns:
                df_temp = df.copy()
                df_temp['_cond'] = df_temp['btc_adx'] < threshold_val
                r = analyze_condition_impact(df_temp, f"BTC ADX < {threshold_val} (weak trend)", '_cond', threshold_dd)
                if r:
                    r['label'] = f'BTC ADX < {threshold_val} (weak trend)'
                    r['feature'] = 'btc_adx'
                    r['category'] = 'LEADING'
                    leading_results.append(r)
        
        # CHOP high = choppy market = bad for trend-following
        for threshold_val in [50, 60]:
            if 'btc_chop' in df.columns:
                df_temp = df.copy()
                df_temp['_cond'] = df_temp['btc_chop'] > threshold_val
                r = analyze_condition_impact(df_temp, f"BTC CHOP > {threshold_val} (choppy market)", '_cond', threshold_dd)
                if r:
                    r['label'] = f'BTC CHOP > {threshold_val} (choppy)'
                    r['feature'] = 'btc_chop'
                    r['category'] = 'LEADING'
                    leading_results.append(r)
        
        # BTC Downtrend
        if 'btc_trend' in df.columns:
            df_temp = df.copy()
            df_temp['_cond'] = df_temp['btc_trend'] == 'DOWNTREND'
            r = analyze_condition_impact(df_temp, "BTC in downtrend (SMA20 < SMA50)", '_cond', threshold_dd)
            if r:
                r['label'] = 'BTC Downtrend (SMA20<50)'
                r['feature'] = 'btc_trend'
                r['category'] = 'LEADING'
                leading_results.append(r)
        
        # BTC RSI extremes
        for rsi_thresh, label_str in [(70, 'BTC RSI > 70 (overbought)'), (30, 'BTC RSI < 30 (oversold)')]:
            if 'btc_rsi' in df.columns:
                df_temp = df.copy()
                if rsi_thresh > 50:
                    df_temp['_cond'] = df_temp['btc_rsi'] > rsi_thresh
                else:
                    df_temp['_cond'] = df_temp['btc_rsi'] < rsi_thresh
                r = analyze_condition_impact(df_temp, label_str, '_cond', threshold_dd)
                if r:
                    r['label'] = label_str
                    r['feature'] = 'btc_rsi'
                    r['category'] = 'LEADING'
                    leading_results.append(r)
        
        # BTC volatility (ATR ratio)
        if 'btc_atr_ratio' in df.columns:
            r = analyze_numeric_condition(df, 'btc_atr_ratio', threshold_dd, 75)
            if r:
                r['label'] = 'High BTC ATR ratio (top 25%)'
                r['category'] = 'LEADING'
                leading_results.append(r)
        
        # BTC volatility (14d std)
        if 'btc_vol_14d' in df.columns:
            r = analyze_numeric_condition(df, 'btc_vol_14d', threshold_dd, 75)
            if r:
                r['label'] = 'High BTC vol (14d, top 25%)'
                r['category'] = 'LEADING'
                leading_results.append(r)
        
        # BTC 7d return bottom 25%
        if 'btc_return_7d' in df.columns:
            valid = df['btc_return_7d'].dropna()
            if len(valid) >= 10:
                cutoff = valid.quantile(0.25)
                df_temp = df.copy()
                df_temp['_cond'] = df_temp['btc_return_7d'] <= cutoff
                r = analyze_condition_impact(df_temp, f"BTC 7d return <= {cutoff:.4f}", '_cond', threshold_dd)
                if r:
                    r['cutoff_value'] = cutoff
                    r['feature'] = 'btc_return_7d'
                    r['label'] = 'BTC 7d return bottom 25%'
                    r['category'] = 'LEADING'
                    leading_results.append(r)
    
    # Equity rolling volatility (market regime proxy)
    for col_name, label in [('rolling_vol_5d', 'High 5d equity vol'), ('rolling_vol_10d', 'High 10d equity vol')]:
        if col_name in df.columns:
            r = analyze_numeric_condition(df, col_name, threshold_dd, 75)
            if r:
                r['label'] = label
                r['category'] = 'LEADING'
                leading_results.append(r)
    
    # Position count (can be controlled)
    for pos_thresh in [5, 8]:
        df_temp = df.copy()
        df_temp['_cond'] = df_temp['open_positions_count'] >= pos_thresh
        r = analyze_condition_impact(df_temp, f"Open positions >= {pos_thresh}", '_cond', threshold_dd)
        if r:
            r['label'] = f'Positions >= {pos_thresh}'
            r['feature'] = 'open_positions_count'
            r['category'] = 'LEADING'
            leading_results.append(r)
    
    # ══════════════════════════════════════════════════════════════════════
    # LAGGING INDICATORS — Equity-based (descriptive, confirms damage done)
    # ══════════════════════════════════════════════════════════════════════
    
    lagging_features = [
        ('dd_streak_days', 75, 'Long drawdown streak'),
        ('losing_streak', 75, 'Long losing streak'),
        ('floating_ratio', 25, 'Large negative floating ratio'),
        ('floating_vs_realized', 25, 'Large negative float/realized'),
        ('equity_momentum_7d', 25, 'Negative 7d equity momentum'),
        ('equity_momentum_14d', 25, 'Negative 14d equity momentum'),
        ('realized_momentum_7d', 25, 'Negative 7d realized momentum'),
    ]
    
    for feature, pct, label in lagging_features:
        if feature not in df.columns:
            continue
        valid = df[feature].dropna()
        if len(valid) < 10:
            continue
        cutoff = valid.quantile(pct / 100)
        df_temp = df.copy()
        if pct <= 50:
            df_temp['_cond'] = df_temp[feature] <= cutoff
        else:
            df_temp['_cond'] = df_temp[feature] >= cutoff
        r = analyze_condition_impact(df_temp, f"{feature} (p{pct})", '_cond', threshold_dd)
        if r:
            r['cutoff_value'] = cutoff
            r['feature'] = feature
            r['label'] = label
            r['category'] = 'LAGGING'
            lagging_results.append(r)
    
    # Sort each group by lift
    leading_results.sort(key=lambda x: x['lift'], reverse=True)
    lagging_results.sort(key=lambda x: x['lift'], reverse=True)
    
    return leading_results, lagging_results


# ─── REPORTING ───────────────────────────────────────────────────────────────

def print_condition_report(results_tuple, threshold_dd):
    """Print the condition analysis report with LEADING vs LAGGING separation."""
    leading, lagging = results_tuple
    
    print(f"\n{'='*80}")
    print(f"📊 DRAWDOWN CONDITION ANALYSIS (threshold: {threshold_dd}%)")
    print(f"{'='*80}")
    
    # ── LEADING INDICATORS ──────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"🔮 LEADING INDICATORS — Market Regime (BEFORE trade entry)")
    print(f"{'─'*80}")
    print(f"\n  {'CONDITION':<45} {'DD RATE':>8} {'vs BASE':>8} {'LIFT':>6} {'p-val':>8} {'SIG':>4}")
    print(f"  {'─'*80}")
    
    for r in leading:
        sig_marker = '✅' if r['significant'] else '  '
        lift_marker = '🔥' if r['lift'] > 1.3 and r['significant'] else '  '
        print(f"  {r['label']:<45} {r['dd_rate_when_true']:>7.1f}% {r['dd_rate_when_false']:>7.1f}% {r['lift']:>5.2f}x {r['p_value']:>8.4f} {sig_marker}{lift_marker}")
    
    if not leading:
        print("   No leading indicators available (need BTC data).")
    
    # Actionable leading signals
    actionable = [r for r in leading if r['significant'] and r['lift'] > 1.3]
    if actionable:
        print(f"\n  💡 ACTIONABLE LEADING SIGNALS:")
        for i, r in enumerate(actionable, 1):
            print(f"    {i}. {r['label']} → DD {r['dd_rate_when_true']:.0f}% (lift {r['lift']:.2f}x, p={r['p_value']:.4f})")
    else:
        # Check for protective (lift < 0.7)
        protective = [r for r in leading if r['significant'] and r['lift'] < 0.7]
        if protective:
            print(f"\n  🛡️ PROTECTIVE CONDITIONS (reduce DD):")
            for r in protective:
                print(f"    • {r['label']} → DD only {r['dd_rate_when_true']:.0f}% vs {r['dd_rate_when_false']:.0f}% baseline")
    
    # ── LAGGING INDICATORS ──────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"📉 LAGGING INDICATORS — Equity-Based (describes damage already done)")
    print(f"{'─'*80}")
    print(f"\n  {'CONDITION':<45} {'DD RATE':>8} {'vs BASE':>8} {'LIFT':>6} {'p-val':>8} {'SIG':>4}")
    print(f"  {'─'*80}")
    
    for r in lagging:
        sig_marker = '✅' if r['significant'] else '  '
        print(f"  {r['label']:<45} {r['dd_rate_when_true']:>7.1f}% {r['dd_rate_when_false']:>7.1f}% {r['lift']:>5.2f}x {r['p_value']:>8.4f} {sig_marker}")
    
    print(f"\n  ⚠️  Lagging indicators confirm DD but cannot predict it.")
    print(f"     They are useful for monitoring, not for entry decisions.")
    
    print(f"\n{'='*80}")


def plot_condition_analysis(df, results_tuple, threshold_dd, output_dir):
    """Create condition analysis visualizations."""
    leading, lagging = results_tuple
    all_results = leading + lagging
    significant = [r for r in all_results if r['significant']]
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'Drawdown Condition Analysis (>{threshold_dd}%)', fontsize=16, fontweight='bold')
    
    # ── 1. Lift chart — LEADING only ─────────────────────────────────────
    ax1 = axes[0, 0]
    if leading:
        labels = [r['label'][:30] for r in leading]
        lifts = [r['lift'] for r in leading]
        colors = ['#e74c3c' if r['lift'] > 1.3 and r['significant'] else 
                  '#2ecc71' if r['lift'] < 0.7 and r['significant'] else 
                  '#95a5a6' for r in leading]
        
        y_pos = range(len(labels))
        ax1.barh(y_pos, lifts, color=colors, edgecolor='black', alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(labels, fontsize=8)
        ax1.axvline(x=1.0, color='black', linestyle='--', alpha=0.5, label='Baseline (1.0x)')
        ax1.set_xlabel('Lift (DD probability ratio)')
        ax1.set_title('🔮 LEADING Indicators — Predictive Power')
        ax1.legend()
        ax1.invert_yaxis()
    
    # ── 2. ADX vs Drawdown (if available) ────────────────────────────────
    ax2 = axes[0, 1]
    if 'btc_adx' in df.columns and df['btc_adx'].notna().any():
        valid_adx = df.dropna(subset=['btc_adx'])
        ax2.scatter(valid_adx['btc_adx'], valid_adx['drawdown_pct'],
                   alpha=0.1, s=10, color='#3498db')
        
        bins = pd.qcut(valid_adx['btc_adx'], q=10, duplicates='drop')
        binned = valid_adx.groupby(bins)['drawdown_pct'].mean()
        bin_centers = [(b.left + b.right)/2 for b in binned.index]
        ax2.plot(bin_centers, binned.values, 'r-o', linewidth=2, markersize=4, label='Binned mean')
        
        ax2.axvline(x=20, color='orange', linestyle='--', alpha=0.7, label='ADX=20 (weak trend)')
        ax2.axvline(x=25, color='red', linestyle='--', alpha=0.7, label='ADX=25')
        ax2.set_xlabel('BTC ADX (14)')
        ax2.set_ylabel('Drawdown %')
        ax2.set_title('BTC ADX vs Drawdown — Does Weak Trend Cause DD?')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        # Fallback: Volatility vs Drawdown
        valid_vol = df.dropna(subset=['rolling_vol_10d'])
        if len(valid_vol) > 0:
            ax2.scatter(valid_vol['rolling_vol_10d'], valid_vol['drawdown_pct'],
                       alpha=0.1, s=10, color='#3498db')
            ax2.set_xlabel('Rolling Volatility (10d)')
            ax2.set_ylabel('Drawdown %')
            ax2.set_title('Volatility vs Drawdown Severity')
            ax2.grid(True, alpha=0.3)
    
    # ── 3. CHOP vs DD probability (if available) ─────────────────────────
    ax3 = axes[1, 0]
    if 'btc_chop' in df.columns and df['btc_chop'].notna().any():
        valid_chop = df.dropna(subset=['btc_chop'])
        
        bins = pd.qcut(valid_chop['btc_chop'], q=10, duplicates='drop')
        chop_dd = valid_chop.groupby(bins).agg(
            dd_rate=('drawdown_pct', lambda x: (x > threshold_dd).mean() * 100),
            count=('drawdown_pct', 'count')
        ).reset_index()
        
        chop_labels = [f"{b.left:.0f}-{b.right:.0f}" for b in chop_dd['btc_chop']]
        colors = ['#e74c3c' if b.right > 60 else '#3498db' for b in chop_dd['btc_chop']]
        ax3.bar(range(len(chop_labels)), chop_dd['dd_rate'], color=colors, alpha=0.8, edgecolor='black')
        ax3.set_xticks(range(len(chop_labels)))
        ax3.set_xticklabels(chop_labels, rotation=45, fontsize=7)
        ax3.set_xlabel('BTC Choppiness Index (14)')
        ax3.set_ylabel(f'DD Rate (>{threshold_dd}%)')
        ax3.set_title('BTC CHOP vs Drawdown — Choppy = More DD?')
        ax3.grid(True, alpha=0.3, axis='y')
    else:
        # Fallback: Position count
        pos_groups = df.groupby('open_positions_count').agg(
            dd_rate=('drawdown_pct', lambda x: (x > threshold_dd).mean() * 100),
            count=('drawdown_pct', 'count')
        ).reset_index()
        pos_groups = pos_groups[pos_groups['count'] >= 10]
        if len(pos_groups) > 0:
            ax3.bar(pos_groups['open_positions_count'], pos_groups['dd_rate'],
                   color='#e74c3c', alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Open Positions Count')
            ax3.set_ylabel(f'DD Rate (>{threshold_dd}%)')
            ax3.set_title('Position Count vs DD')
            ax3.grid(True, alpha=0.3, axis='y')
    
    # ── 4. BTC trend impact ──────────────────────────────────────────────
    ax4 = axes[1, 1]
    if 'btc_trend' in df.columns and df['btc_trend'].notna().any():
        trend_impact = df.groupby('btc_trend').agg(
            dd_rate=('drawdown_pct', lambda x: (x > threshold_dd).mean() * 100),
            avg_dd=('drawdown_pct', 'mean'),
            count=('drawdown_pct', 'count')
        ).reset_index()
        
        colors_map = {'UPTREND': '#2ecc71', 'DOWNTREND': '#e74c3c'}
        bar_colors = [colors_map.get(t, '#95a5a6') for t in trend_impact['btc_trend']]
        
        bars = ax4.bar(trend_impact['btc_trend'], trend_impact['dd_rate'],
                      color=bar_colors, alpha=0.8, edgecolor='black')
        ax4.set_xlabel('BTC Trend')
        ax4.set_ylabel(f'DD Rate (>{threshold_dd}%)')
        ax4.set_title('BTC Trend vs Drawdown Probability')
        ax4.grid(True, alpha=0.3, axis='y')
        
        for bar, (_, row) in zip(bars, trend_impact.iterrows()):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"n={int(row['count'])}\navg DD={row['avg_dd']:.1f}%",
                    ha='center', fontsize=9)
    else:
        ax4.text(0.5, 0.5, 'BTC data not available', transform=ax4.transAxes,
                ha='center', va='center', fontsize=14, color='gray')
        ax4.set_title('BTC Trend vs Drawdown')
    
    plt.tight_layout()
    save_path = output_dir / 'drawdown_conditions.png'
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"   💾 Condition analysis plot saved: {save_path}")
    plt.close()


# ─── RULE GENERATOR ─────────────────────────────────────────────────────────

def generate_risk_rules(results_tuple, threshold_dd):
    """Generate actionable risk management rules from LEADING indicators only."""
    leading, _ = results_tuple
    
    # Only use leading indicators with lift > 1.3 for rules
    significant = [r for r in leading if r['significant'] and r['lift'] > 1.3]
    
    if not significant:
        return []
    
    rules = []
    for r in significant:
        feature = r.get('feature', '')
        cutoff = r.get('cutoff_value', None)
        
        if 'adx' in feature:
            c_val = cutoff if cutoff is not None else 20
            rules.append({
                'name': 'WEAK_TREND_FILTER',
                'condition': f"btc_adx < {c_val:.1f}",
                'action': 'REDUCE_POSITION_SIZE',
                'detail': f"When BTC ADX < {c_val:.1f}, reduce position size 50% (weak trend = bad for trend-following)",
                'lift': r['lift'],
                'p_value': r['p_value']
            })
        elif 'chop' in feature:
            c_val = cutoff if cutoff is not None else 60
            rules.append({
                'name': 'CHOPPY_MARKET_FILTER',
                'condition': f"btc_chop > {c_val:.1f}",
                'action': 'SKIP_NEW_ENTRIES',
                'detail': f"When BTC CHOP > {c_val:.1f}, skip new entries (choppy market = whipsaws)",
                'lift': r['lift'],
                'p_value': r['p_value']
            })
        elif 'vol' in feature and 'btc' not in feature:
            c_val = cutoff if cutoff is not None else 0.05
            rules.append({
                'name': 'HIGH_VOLATILITY_FILTER',
                'condition': f"rolling_vol > {c_val:.5f}",
                'action': 'REDUCE_LEVERAGE',
                'detail': f"When equity vol > {c_val:.5f}, reduce leverage by 50%",
                'lift': r['lift'],
                'p_value': r['p_value']
            })
        elif 'position' in feature:
            rules.append({
                'name': 'MAX_POSITION_FILTER',
                'condition': f"open_positions >= {int(cutoff) if cutoff else 8}",
                'action': 'STOP_NEW_ENTRIES',
                'detail': f"Stop new entries when positions >= {int(cutoff) if cutoff else 8}",
                'lift': r['lift'],
                'p_value': r['p_value']
            })
        elif 'btc' in feature and 'trend' in feature:
            rules.append({
                'name': 'BTC_TREND_FILTER',
                'condition': 'btc_sma20 < btc_sma50',
                'action': 'REDUCE_LONG_EXPOSURE',
                'detail': 'Reduce LONG position size by 50% when BTC is in downtrend',
                'lift': r['lift'],
                'p_value': r['p_value']
            })
        elif 'btc' in feature:
            rules.append({
                'name': 'BTC_MARKET_FILTER',
                'condition': r['condition'],
                'action': 'REDUCE_EXPOSURE',
                'detail': f"{r['label']} → reduce exposure",
                'lift': r['lift'],
                'p_value': r['p_value']
            })
    
    return rules

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split

def run_ml_decision_tree(df, threshold_dd):
    print(f"\n{'='*80}")
    print(f"🧠 MACHINE LEARNING: MULTI-CONDITION INSIGHTS (Decision Tree)")
    print(f"{'='*80}")
    
    # 1. Chọn các features (chỉ dùng Leading Indicators)
    features = ['btc_rsi', 'btc_adx', 'btc_chop', 'btc_atr_ratio', 
                'rolling_vol_5d', 'rolling_vol_10d', 'open_positions_count']
    
    # Thêm encode cho btc_trend (1 là DOWNTREND, 0 là UPTREND)
    if 'btc_trend' in df.columns:
        df['btc_trend_encoded'] = np.where(df['btc_trend'] == 'DOWNTREND', 1, 0)
        features.append('btc_trend_encoded')

    # Lọc data hợp lệ
    ml_df = df.dropna(subset=features + ['drawdown_pct']).copy()
    if len(ml_df) < 50:
        print(" ⚠️ Không đủ dữ liệu để chạy Machine Learning.")
        return

    X = ml_df[features]
    y = np.where(ml_df['drawdown_pct'] > threshold_dd, 1, 0) # 1 = Bị Drawdown nặng

    # 2. Train mô hình (Giới hạn depth=3 để chống Overfitting và dễ đọc)
    clf = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
    clf.fit(X, y)

    # 3. Xuất luật ra dạng text dễ hiểu
    tree_rules = export_text(clf, feature_names=list(X.columns))
    print("\n🔍 Các nhánh ra quyết định (Class 1 = Drawdown > 35%, Class 0 = An toàn):")
    print(tree_rules)
    
    # 4. In ra Feature Importance (Trọng số quan trọng nhất)
    importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': clf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n🏆 Top các chỉ số quyết định sự sống còn:")
    for _, row in importances.head(5).iterrows():
        print(f"   • {row['Feature']:<20}: {row['Importance']:.2f}")

    return clf


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Drawdown Condition Analyzer")
    parser.add_argument('--input', type=str, required=True, help='Path to all_windows_combined.csv')
    parser.add_argument('--threshold', type=float, default=35.0, help='Drawdown threshold %% (default: 35)')
    parser.add_argument('--no-plot', action='store_true', help='Skip plot generation')
    parser.add_argument('--timeframe', type=str, default='1d', help='Timeframe for BTC data lookup')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return
    
    output_dir = input_path.parent
    
    # Load data
    print(f"📂 Loading: {input_path}")
    df = pd.read_csv(input_path, parse_dates=['date'])
    if 'window_idx' not in df.columns:
        df['window_idx'] = 0
    print(f"   Loaded {len(df):,} rows, {df['window_idx'].nunique()} windows")
    
    # Load BTC prices
    print(f"📈 Loading BTC price data...")
    btc_data = load_btc_prices(args.timeframe)
    btc_available = btc_data is not None
    if btc_available:
        print(f"   BTC data: {btc_data['date'].min()} → {btc_data['date'].max()}")
    
    # Compute features
    print(f"⚙️  Computing condition features...")
    df = compute_condition_features(df, btc_data)
    print(f"   Features computed: {len(df.columns)} columns")
    
    # Run analysis
    print(f"🔍 Analyzing conditions for drawdowns > {args.threshold}%...")
    results = run_full_analysis(df, args.threshold, btc_available)
    
    # Print report
    print_condition_report(results, args.threshold)
    
    # Generate rules
    rules = generate_risk_rules(results, args.threshold)
    if rules:
        print(f"\n{'─'*80}")
        print(f"🤖 GENERATED RISK MANAGEMENT RULES (for bot integration)")
        print(f"{'─'*80}\n")
        for i, rule in enumerate(rules, 1):
            print(f"  Rule {i}: {rule['name']}")
            print(f"    IF: {rule['condition']}")
            print(f"    THEN: {rule['action']}")
            print(f"    Detail: {rule['detail']}")
            print(f"    Evidence: lift={rule['lift']:.2f}x, p={rule['p_value']:.4f}")
            print()
        
        # Save rules as JSON
        import json
        rules_path = output_dir / 'risk_rules.json'
        with open(rules_path, 'w') as f:
            json.dump(rules, f, indent=2)
        print(f"💾 Risk rules saved: {rules_path}")
    
    # Save enriched dataset
    analysis_path = output_dir / 'drawdown_conditions_analysis.csv'
    df.to_csv(analysis_path, index=False)
    print(f"💾 Enriched analysis saved: {analysis_path}")
    
    # Plots
    # if not args.no_plot:
    #     print(f"\n📊 Generating condition analysis plots...")
    #     plot_condition_analysis(df, results, args.threshold, output_dir)
    run_ml_decision_tree(df, args.threshold)
    print(f"\n✅ Analysis complete!")


if __name__ == '__main__':
    main()
