
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.data_pipeline import generate_labels
from ml.config import SUPPORTED_TIMEFRAMES

PROCESSED_DIR = Path("data/processed")

def export_sl_training_data(timeframe: str):
    file_path = PROCESSED_DIR / f"features_{timeframe}_full.parquet"
    if not file_path.exists():
        print(f"⚠️ File not found: {file_path}")
        return
        
    print(f"\n🚀 Exporting SL training data for {timeframe}...")
    
    # Load in chunks or by symbol to be safe
    # We first need the symbols
    metadata = pd.read_parquet(file_path, columns=['symbol'])
    symbols = metadata['symbol'].unique()
    del metadata
    
    # ATR multipliers
    atr_tp_mult, atr_sl_mult = 3.0, 1.5
    tf_scale = {'1h': 0.5, '4h': 7, '8h': 7, '12h': 7.0, '1d': 7.0}
    scale = tf_scale.get(timeframe, 1.0)
    tp_pct, sl_pct = 0.03 * scale, 0.015 * scale
    
    all_crossovers = []
    
    for i, symbol in enumerate(symbols):
        # Load only this symbol
        df_symbol = pd.read_parquet(file_path, filters=[('symbol', '==', symbol)])
        
        # Calculate new robust labels using the updated logic in data_pipeline.py
        df_labeled = generate_labels(
            df_symbol,
            tp_pct=tp_pct, sl_pct=sl_pct, max_bars=10,
            use_atr=True, atr_tp_mult=atr_tp_mult, atr_sl_mult=atr_sl_mult,
            min_tp_pct=0.20, max_tp_pct=1.00
        )
        
        # Keep only identified crossovers with labels
        df_cross = df_labeled.dropna(subset=['sl_pct_used'])
        df_cross = df_cross[(df_cross['macd_cross_up'] == 1) | (df_cross['macd_cross_down'] == 1)]
        
        # Add to list
        if not df_cross.empty:
            all_crossovers.append(df_cross)
            
        if i % 100 == 0:
            print(f"   Processed {i}/{len(symbols)} symbols...")
            
    if all_crossovers:
        df_train = pd.concat(all_crossovers, ignore_index=True)
        out_path = PROCESSED_DIR / f"sl_train_{timeframe}.parquet"
        df_train.to_parquet(out_path)
        print(f"✅ Exported {len(df_train)} training samples to {out_path}")
    else:
        print("❌ No crossovers found!")

if __name__ == "__main__":
    tfs = ['4h', '8h', '12h', '1d']
    for tf in tfs:
        export_sl_training_data(tf)
