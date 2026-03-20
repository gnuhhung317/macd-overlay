#!/usr/bin/env python3
"""
Grid Search: Find dangerous conditions that predict large drawdowns.
Works with both:
  - Single-window CSV from plot_time_equity.py
  - Multi-window combined CSV from run_multi_window_equity.py (all_windows_combined.csv)

Usage:
  python ml/check_predict.py                                           # default single-window
  python ml/check_predict.py ml/equity_windows/all_windows_combined.csv  # multi-window
"""
import pandas as pd
import numpy as np
import sys
import itertools
from pathlib import Path

# =====================================================================
# 1. LOAD & DETECT DATA FORMAT
# =====================================================================
csv_path = sys.argv[1] if len(sys.argv) > 1 else 'ml/results/time_equity_1d_20.0x_isolated.csv'
df = pd.read_csv(csv_path, parse_dates=['date'])

is_multi_window = 'window_idx' in df.columns
n_windows = df['window_idx'].nunique() if is_multi_window else 1
total_rows = len(df)

print("=" * 80)
if is_multi_window:
    print(f"📊 MULTI-WINDOW GRID SEARCH ({n_windows} windows, {total_rows:,} rows)")
else:
    print(f"📊 SINGLE-WINDOW GRID SEARCH ({total_rows} rows)")
print(f"   Source: {csv_path}")
print("=" * 80)

# =====================================================================
# 2. PREPARE FEATURES
# =====================================================================
if 'floating_pnl' in df.columns:
    df['float_pct'] = (df['floating_pnl'] / df['equity'].replace(0, np.nan)) * 100
else:
    df['float_pct'] = 0

df['pos'] = df['open_positions_count'] if 'open_positions_count' in df.columns else 0
df['daily_pct'] = df['equity'].pct_change() * 100

# =====================================================================
# 3. COMPUTE PER-WINDOW FUTURE DRAWDOWN
# For multi-window: compute within each window to avoid cross-window leakage
# =====================================================================
forward_windows = [1, 2, 3, 5]
crash_thresholds = [-20, -30, -50, -70]
min_samples = max(5, n_windows)  # Scale min_samples with data size

print(f"\n   Forward windows: {forward_windows}")
print(f"   Crash thresholds: {crash_thresholds}")
print(f"   Min samples required: {min_samples}")

# Pre-compute future DD for each forward window, respecting window boundaries
future_dd_cache = {}

if is_multi_window:
    for fw in forward_windows:
        future_dd = pd.Series(np.nan, index=df.index)
        for widx, group in df.groupby('window_idx'):
            idx = group.index
            eq = group['equity'].values
            # For each row, find min equity in next fw days (within same window)
            for i in range(len(eq)):
                forward_slice = eq[i+1 : i+1+fw]
                if len(forward_slice) > 0:
                    min_eq = forward_slice.min()
                    future_dd.iloc[idx[i]] = (min_eq / eq[i] - 1) * 100
        future_dd_cache[fw] = future_dd
else:
    for fw in forward_windows:
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=fw)
        future_min = df['equity'].shift(-1).rolling(window=indexer).min()
        future_dd_cache[fw] = (future_min / df['equity'] - 1) * 100

# =====================================================================
# 4. GRID SEARCH
# =====================================================================
print(f"\n{'='*80}")
print("🔍 ĐANG CHẠY GRID SEARCH TỐI ƯU HÓA...")
print(f"{'='*80}")

# Auto-detect feature ranges from data
float_percentiles = np.percentile(df['float_pct'].dropna(), np.arange(10, 100, 10))
max_pos = int(df['pos'].max()) if df['pos'].max() > 0 else 1
pos_range = range(1, max_pos + 1)

total_combos = len(forward_windows) * len(crash_thresholds) * len(pos_range) * len(float_percentiles)
print(f"   Tổng tổ hợp cần quét: {total_combos:,}")

results = []

for fw in forward_windows:
    future_dd = future_dd_cache[fw]
    
    for crash_th in crash_thresholds:
        is_crash = (future_dd <= crash_th).astype(float)
        is_crash_valid = is_crash.dropna()
        
        if is_crash_valid.sum() < min_samples:
            continue
        
        for p, f in itertools.product(pos_range, float_percentiles):
            subset_mask = (df['pos'] >= p) & (df['float_pct'] >= f)
            subset_idx = df.index[subset_mask]
            
            # Only count rows where future DD is available
            valid_idx = subset_idx.intersection(is_crash_valid.index)
            n_samples = len(valid_idx)
            
            if n_samples >= min_samples:
                crash_prob = is_crash[valid_idx].mean() * 100
                crashes_count = int(is_crash[valid_idx].sum())
                
                if crash_prob > 0:
                    # Count unique windows where this rule triggers
                    if is_multi_window:
                        windows_triggered = df.loc[valid_idx, 'window_idx'].nunique()
                    else:
                        windows_triggered = 1
                    
                    results.append({
                        'fw': fw,
                        'crash_th': crash_th,
                        'pos_min': p,
                        'float_min': f,
                        'prob': crash_prob,
                        'n': n_samples,
                        'crashes': crashes_count,
                        'windows': windows_triggered
                    })

# =====================================================================
# 5. FILTER & DISPLAY RESULTS
# =====================================================================
res_df = pd.DataFrame(results)

if res_df.empty:
    print("\n❌ Không tìm thấy tổ hợp rủi ro nào đạt đủ số lượng mẫu.")
else:
    # For multi-window: require rule to trigger in at least 3 different windows
    if is_multi_window and n_windows >= 5:
        min_windows = max(3, n_windows // 3)
        res_df = res_df[res_df['windows'] >= min_windows]
        print(f"\n   Lọc: chỉ giữ rules xuất hiện trong >= {min_windows} windows khác nhau")
    
    if res_df.empty:
        print("\n❌ Không tìm thấy rules đủ robust (xuất hiện trong nhiều windows).")
    else:
        # Sort by probability, then crashes count
        res_df = res_df.sort_values(by=['prob', 'crashes'], ascending=[False, False])
        
        # Deduplicate: keep best rule per (forward_window, crash_threshold) combo
        best_rules = res_df.drop_duplicates(subset=['fw', 'crash_th']).head(10)
        
        print(f"\n{'='*80}")
        print(f"[TOP {len(best_rules)} TỔ HỢP ĐIỀU KIỆN NGUY HIỂM NHẤT]")
        print(f"{'='*80}\n")
        
        for _, row in best_rules.iterrows():
            severity = f"Sập > {abs(row['crash_th'])}%"
            timeframe = f"{row['fw']} ngày"
            win_str = f", {row['windows']} windows" if is_multi_window else ""
            
            print(f"⚠️  NGUY CƠ: {severity} trong {timeframe} tới")
            print(f"   Xác suất : {row['prob']:.1f}% ({row['crashes']}/{row['n']} lần{win_str})")
            print(f"   ĐIỀU KIỆN: Vị thế >= {row['pos_min']} VÀ Float >= {row['float_min']:.1f}%")
            print("-" * 70)
        
        # ── Summary statistics ───────────────────────────────────────
        print(f"\n{'='*80}")
        print(f"📊 TỔNG KẾT")
        print(f"{'='*80}")
        
        # Base rate comparison
        for fw in forward_windows:
            future_dd = future_dd_cache[fw]
            valid = future_dd.dropna()
            for crash_th in crash_thresholds:
                base_rate = (valid <= crash_th).mean() * 100
                if base_rate > 0:
                    best = best_rules[(best_rules['fw'] == fw) & (best_rules['crash_th'] == crash_th)]
                    if not best.empty:
                        lift = best.iloc[0]['prob'] / base_rate
                        print(f"   Sập >{abs(crash_th)}% trong {fw}d: "
                              f"Base rate={base_rate:.1f}% → Rule={best.iloc[0]['prob']:.1f}% "
                              f"(Lift={lift:.1f}x)")

print(f"\n{'='*80}")