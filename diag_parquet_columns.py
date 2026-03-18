import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Mocking or importing calculation functions
sys.path.append(str(Path.cwd()))
from ml.data_pipeline import calculate_features as calc_offline, apply_global_feature_shift as off_shift
from sniper_bot.feature import calculate_features as calc_live, apply_feature_shift as live_shift

def debug():
    parquet_path = r'd:\Code\Projects\self-projects\macd-overlay - Copy\data\processed\features_1h_btc_context.parquet'
    gt_df = pd.read_parquet(parquet_path).head(100)
    
    raw_ohlcv = gt_df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']].copy()
    
    off_df = calc_offline(raw_ohlcv.copy(), df_1d=None, btc_df=None)
    # off_df = off_shift(off_df) # Don't shift yet to see raw calc
    
    live_df = calc_live(raw_ohlcv.copy(), df_1d=None, btc_df=None)
    # live_df = live_shift(live_df) # Don't shift yet to see raw calc
    
    print("\n--- Columns in Ground Truth ---")
    print(sorted(gt_df.columns))
    
    print("\n--- Columns in Offline DF ---")
    print(sorted(off_df.columns))
    
    print("\n--- Columns in Live DF ---")
    print(sorted(live_df.columns))
    
    # Check a few specific values
    idx = 50
    ts = gt_df.iloc[idx]['timestamp']
    print(f"\n--- Checking values at index {idx} ({ts}) ---")
    
    cols_to_check = ['price_vs_sma_30', 'price_to_sma_30', 'dist_to_ema_21_pct', 'rsi_14', 'rsi_slope']
    
    for c in cols_to_check:
        v_gt = gt_df.iloc[idx].get(c, "N/A")
        v_off = off_df[off_df['timestamp'] == ts][c].values[0] if c in off_df.columns else "N/A"
        v_live = live_df[live_df['timestamp'] == ts][c].values[0] if c in live_df.columns else "N/A"
        print(f"{c:<20} | GT: {v_gt} | OFF: {v_off} | LIVE: {v_live}")

if __name__ == "__main__":
    debug()
