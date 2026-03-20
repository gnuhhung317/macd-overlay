import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Mocking or importing calculation functions
sys.path.append(str(Path.cwd()))
from ml.data_pipeline import calculate_features as calc_offline, apply_global_feature_shift as off_shift
from sniper_bot.feature import calculate_features as calc_live, apply_feature_shift as live_shift

def verify():
    print("🧪 Starting FINAL Feature Parity Test...")
    parquet_path = r'd:\Code\Projects\self-projects\macd-overlay - Copy\data\processed\features_1h_btc_context.parquet'
    
    # Load Ground Truth (Need enough for warm-up)
    gt_df = pd.read_parquet(parquet_path).head(500)
    
    # We need RAW OHLCV to re-calculate. 
    raw_ohlcv = gt_df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']].copy()
    
    # Run Offline Engine
    off_df = calc_offline(raw_ohlcv.copy(), df_1d=None, btc_df=None)
    off_df = off_shift(off_df)
    
    # Run Live Engine
    live_df = calc_live(raw_ohlcv.copy(), df_1d=None, btc_df=None)
    live_df = live_shift(live_df)
    
    # Mapping
    # Parquet Name -> (Offline Name, Live Name)
    mapping = {
        'price_vs_sma_30': ('price_vs_sma_30', 'price_vs_sma_30'),
        'dist_to_ema_21_pct': ('dist_to_ema_21_pct', 'dist_to_ema_21_pct'),
        'rsi_slope': ('rsi_slope', 'rsi_slope'),
        'log_returns': ('log_returns', 'log_returns')
    }
    
    print(f"\n{'Column':<25} | {'GT':<12} | {'Offline':<12} | {'Live':<12}")
    print("-" * 75)
    
    # Skip first 300 bars for warm-up
    for i in range(350, 360):
        ts = gt_df.iloc[i]['timestamp']
        print(f"Index {i} ({ts}):")
        for p_col, (off_col, l_col) in mapping.items():
            v_gt = gt_df.iloc[i].get(p_col, np.nan)
            
            # Offline match
            v_off = off_df[off_df['timestamp'] == ts][off_col].values[0] if off_col in off_df.columns and not off_df[off_df['timestamp'] == ts].empty else np.nan
            
            # Live match
            v_live = live_df[live_df['timestamp'] == ts][l_col].values[0] if l_col in live_df.columns and not live_df[live_df['timestamp'] == ts].empty else np.nan
            
            status = "✅" if (np.isclose(v_gt, v_off, atol=1e-5) if not np.isnan(v_off) else False) else "❌"
            status_l = "✅" if (np.isclose(v_gt, v_live, atol=1e-5) if not np.isnan(v_live) else False) else "❌"
            
            print(f"  {p_col:<22} | {v_gt:10.4f} | {v_off:10.4f} {status} | {v_live:10.4f} {status_l}")
        print("-" * 75)

if __name__ == "__main__":
    verify()
