#!/usr/bin/env python3
"""
Unified Flow: Fetch latest Bitget data and rebuild ML dataset/features.
This script combines bitget_fetcher.py and ml/data_pipeline.py.
"""
import sys
from pathlib import Path
import argparse
from datetime import datetime

# Add root folder to sys.path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from bitget_fetcher import BitgetFetcher
    from binance_fetcher import BinanceFetcher
    import ml.data_pipeline as data_pipeline
    from ml.multi_timeframe_pipeline import build_timeframe_dataset, build_all_timeframes
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Ensure you are running this from the project root or ml folder.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Unified Sync & Dataset Rebuild")
    parser.add_argument("--exchange", type=str, choices=['bitget', 'binance'], default='bitget', help="Exchange to fetch from")
    parser.add_argument("--limit", type=int, help="Limit number of coins to fetch")
    parser.add_argument("--min-days", type=int, default=180, help="Minimum days for feature engineering")
    parser.add_argument("--timeframe", type=str, default="1d", help="Target timeframe (1h, 4h, 1d, etc.)")
    parser.add_argument("--all", action="store_true", help="Rebuild ALL timeframes (1h, 4h, 8h, 12h, 1d, 1w)")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip fetching, only rebuild dataset")
    args = parser.parse_args()

    start_time = datetime.now()
    
    print("="*80)
    print(f"🚀 UNIFIED SYNC & REBUILD FLOW - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌍 Exchange: {args.exchange.upper()}")
    print("="*80)

    # Set data directory based on exchange
    if args.exchange == 'binance':
        data_dir = ROOT_DIR / 'data'
    else:
        data_dir = ROOT_DIR / 'bitget-data'
    
    data_pipeline.set_data_directory(data_dir)

    # 1. Fetch Latest Data
    if not args.skip_fetch:
        print(f"\n[STEP 1/2] 📥 Fetching latest data from {args.exchange.capitalize()}...")
        try:
            if args.exchange == 'bitget':
                fetcher = BitgetFetcher()
            else:
                fetcher = BinanceFetcher()
                
            fetcher.run(limit_coins=args.limit)
            print(f"\n✅ Data fetching from {args.exchange.capitalize()} complete.")
        except Exception as e:
            print(f"\n❌ Error during fetching: {e}")
            sys.exit(1)
    else:
        print("\n[STEP 1/2] ⏭️ Skipping data fetch as requested.")

    # 2. Rebuild Dataset
    print("\n" + "="*80)
    if args.all:
        print(f"[STEP 2/2] 🔨 Rebuilding ALL timeframes...")
    else:
        print(f"[STEP 2/2] 🔨 Rebuilding {args.timeframe} dataset and features...")
    print("="*80)
    
    try:
        if args.all:
            # Rebuild all timeframes using multi_timeframe_pipeline
            build_all_timeframes()
        else:
            # Build specific timeframe using multi_timeframe_pipeline for better flexibility
            build_timeframe_dataset(timeframe=args.timeframe)
        
        print(f"\n✅ Dataset rebuild complete.")
        
    except Exception as e:
        print(f"\n❌ Error during dataset rebuild: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    end_time = datetime.now()
    duration = end_time - start_time
    print("\n" + "="*80)
    print(f"✨ ALL STEPS COMPLETE! Total duration: {duration}")
    print("="*80)

if __name__ == "__main__":
    main()
