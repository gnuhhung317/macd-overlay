import pandas as pd
import os
from pathlib import Path

def split_parquet(input_file, output_dir):
    print(f"Reading {input_file}...")
    df = pd.read_parquet(input_file)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    symbols = df['symbol'].unique()
    print(f"Found {len(symbols)} symbols. Splitting...")
    
    for i, symbol in enumerate(symbols):
        symbol_df = df[df['symbol'] == symbol].copy()
        # Clean up timestamp just in case
        symbol_df['timestamp'] = pd.to_datetime(symbol_df['timestamp']).dt.tz_localize(None)
        symbol_df = symbol_df.sort_values('timestamp')
        
        target_file = output_path / f"{symbol}.parquet"
        symbol_df.to_parquet(target_file, index=False)
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(symbols)} symbols...")

    print("✅ Successfully split data into symbols_v3")

if __name__ == "__main__":
    split_parquet(
        r"data/processed/features_1h_btc_context.parquet",
        r"data/processed/symbols_v3"
    )
