import pandas as pd
from pathlib import Path
import glob
import os
import gc

# Config
BASE_DIR = Path(__file__).resolve().parent.parent
SOURCES = [
    # BASE_DIR / "bitget-data" / "symbols_v3",
    BASE_DIR / "data" / "processed" / "symbols_v3"
]
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "features_1h_honest_dataset.parquet"

def merge_datasets():
    print(f"🚀 Starting Batch-Merge into {OUTPUT_FILE}")
    
    batch_dir = BASE_DIR / "data" / "temp_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    current_batch = []
    batch_count = 0
    total_symbols = 0
    
    for source_dir in SOURCES:
        if not source_dir.exists(): continue
        files = glob.glob(str(source_dir / "*.parquet"))
        
        for f in files:
            try:
                df = pd.read_parquet(f)
                if df.empty: continue
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                current_batch.append(df)
                total_symbols += 1
                
                if len(current_batch) >= 50:
                    batch_count += 1
                    batch_path = batch_dir / f"batch_{batch_count}.parquet"
                    pd.concat(current_batch, ignore_index=True).to_parquet(batch_path)
                    print(f"   💾 Saved Batch {batch_count} ({total_symbols} symbols total)")
                    current_batch = []
                    gc.collect()
            except Exception as e:
                print(f"   ❌ Error {f}: {e}")

    # Final batch
    if current_batch:
        batch_count += 1
        pd.concat(current_batch, ignore_index=True).to_parquet(batch_dir / f"batch_{batch_count}.parquet")

    # Final Merge of Batches
    batch_files = glob.glob(str(batch_dir / "*.parquet"))
    all_final = []
    print(f"🔗 Merging {len(batch_files)} batches...")
    for bf in batch_files:
        all_final.append(pd.read_parquet(bf))
    
    final_df = pd.concat(all_final, ignore_index=True)
    print("🧹 cleaning up...")
    final_df = final_df.drop_duplicates(subset=['symbol', 'timestamp'], keep='last')
    final_df = final_df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    
    final_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"✅ Saved {len(final_df)} rows to {OUTPUT_FILE}")
    
    # Cleanup temp
    for bf in batch_files: os.remove(bf)
    os.rmdir(batch_dir)

if __name__ == "__main__":
    merge_datasets()
