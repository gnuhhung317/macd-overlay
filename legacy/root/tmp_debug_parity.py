import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))
from ml.data_pipeline import calculate_features as calc_offline, apply_global_feature_shift as off_shift
from sniper_bot.feature import calculate_features as calc_live, apply_feature_shift as live_shift

def debug_parity():
    parquet_path = r'd:\Code\Projects\self-projects\macd-overlay - Copy\data\processed\features_1h_btc_context.parquet'
    gt_df = pd.read_parquet(parquet_path).head(200)
    raw_ohlcv = gt_df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']].copy()
    
    # Run Offline Engine
    off_df_raw = calc_offline(raw_ohlcv.copy())
    # Trace the shift
    non_shift_cols = {'timestamp','symbol','open','high','low','close','volume','macd_cross_up','macd_cross_down','macd_crossover','date','fundingTime'}
    already_shifted = {c for c in off_df_raw.columns if any(x in c for x in ['dist_to', 'price_to', 'price_vs'])}
    shift_cols = [c for c in off_df_raw.columns if c not in non_shift_cols and c not in already_shifted]
    
    print("Columns considered 'Already Shifted':", already_shifted)
    if 'price_to_sma_30' in already_shifted: print("✅ price_to_sma_30 is correctly guarded.")
    else: print("❌ price_to_sma_30 is NOT guarded!")
    
    off_df = off_shift(off_df_raw.copy())
    live_df = live_shift(calc_live(raw_ohlcv.copy()))
    
    print("GT Columns:", [c for c in gt_df.columns if 'rsi' in c or 'sma' in c or 'ema' in c][:15])
    print("Offline Columns:", [c for c in off_df.columns if 'rsi' in c or 'sma' in c or 'ema' in c][:15])
    print("Live Columns:", [c for c in live_df.columns if 'rsi' in c or 'sma' in c or 'ema' in c][:15])
    
    target_idx = 50
    ts = gt_df.iloc[target_idx]['timestamp']
    print(f"\nComparing at {ts}")
    cols_to_check = ['rsi_14', 'rsi_slope', 'ema_21', 'sma_30', 'price_vs_sma_30', 'dist_to_ema_21_pct', 'dist_to_high_30d', 'dist_to_low_30d']
    
    for c in cols_to_check:
        v_gt = gt_df.iloc[target_idx].get(c, np.nan)
        c_curr = gt_df.iloc[target_idx]['close']
        c_prev = gt_df.iloc[target_idx-1]['close']
        
        # Manual find to avoid column name issues
        l_col = c if c in live_df.columns else ('price_to_sma_30' if c == 'price_vs_sma_30' else None)
        o_col = c if c in off_df.columns else ('price_to_sma_30' if c == 'price_vs_sma_30' else None)
        
        v_off = off_df[off_df['timestamp'] == ts][o_col].values[0] if o_col and not off_df[off_df['timestamp'] == ts].empty else "MISSING"
        v_live = live_df[live_df['timestamp'] == ts][l_col].values[0] if l_col and not live_df[live_df['timestamp'] == ts].empty else "MISSING"
        
        print(f"{c:<20} | GT: {v_gt:10.5f} | OFF: {v_off} | LIVE: {v_live}")
    
    print(f"\nPrice Check:")
    print(f"Close[curr]: {c_curr}")
    print(f"Close[prev]: {c_prev}")
    print(f"GT SMA30:    {gt_df.iloc[target_idx]['sma_30']}")
    print(f"Calc Ratio (curr): {c_curr / gt_df.iloc[target_idx]['sma_30']}")
    print(f"Calc Ratio (prev): {c_prev / gt_df.iloc[target_idx]['sma_30']}")

if __name__ == "__main__":
    debug_parity()
