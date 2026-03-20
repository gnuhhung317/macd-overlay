
import pandas as pd
from pathlib import Path
import sys
import os

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.data_pipeline import generate_labels
from ml.config import SUPPORTED_TIMEFRAMES

PROCESSED_DIR = Path("data/processed")

def fast_relabel():
    timeframes = ['4h', '8h', '12h', '1d']
    
    # Columns needed for labeling
    NECESSARY_COLS = [
        'close', 'high', 'low', 'timestamp', 'symbol', 
        'macd_cross_up', 'macd_cross_down', 'atr_14'
    ]
    
    for tf in timeframes:
        file_path = PROCESSED_DIR / f"features_{tf}_full.parquet"
        if not file_path.exists():
            continue
            
        print(f"\n🔄 Relabeling {tf} (Memory Efficient)...")
        
        # Load only what we need to calculate labels
        df_minimal = pd.read_parquet(file_path, columns=NECESSARY_COLS)
        
        # Original params
        atr_tp_mult, atr_sl_mult = 3.0, 1.5
        tf_scale = {'1h': 0.5, '4h': 7, '8h': 7, '12h': 7.0, '1d': 7.0}
        scale = tf_scale.get(tf, 1.0)
        tp_pct, sl_pct = 0.03 * scale, 0.015 * scale
        
        # Generate new labels
        df_labeled = generate_labels(
            df_minimal,
            tp_pct=tp_pct, sl_pct=sl_pct, max_bars=10,
            use_atr=True, atr_tp_mult=atr_tp_mult, atr_sl_mult=atr_sl_mult,
            min_tp_pct=0.20, max_tp_pct=1.00
        )
        
        # Now we need to update the original file's SL/Label columns
        # To avoid loading everything at once again, we'll use a merge or just save the update
        print(f"   Merging labels back to full dataset...")
        
        # Load the FULL dataset (this is the risky part, maybe process symbols 1 by 1?)
        # Let's try loading full but dropping the columns we're about to replace
        cols_to_replace = ['label', 'max_profit', 'max_drawdown', 'bars_to_tp', 'bars_to_sl', 'trade_result', 'tp_pct_used', 'sl_pct_used']
        
        full_df = pd.read_parquet(file_path)
        for col in cols_to_replace:
            if col in full_df.columns:
                full_df[col] = df_labeled[col]
        
        full_df.to_parquet(file_path)
        print(f"✅ Success updated {tf}")

if __name__ == "__main__":
    fast_relabel()
