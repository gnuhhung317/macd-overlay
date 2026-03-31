#!/usr/bin/env python3
"""
🔄 Sync & Rebuild (Bitget Local Edition)
Reads raw OHLCV Parquet files from bitget-data/ohlcv/,
computes ML features with BTC context, and saves enriched
data to bitget-data/symbols_v3/.
"""
import sys, os, time, gc, argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Root path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from sniper_bot.feature import calculate_features, apply_feature_shift

# Directories
OHLCV_DIR = BASE_DIR / "bitget-data" / "ohlcv"
SYMBOLS_DIR = BASE_DIR / "bitget-data" / "symbols_v3"
SYMBOLS_DIR.mkdir(parents=True, exist_ok=True)

def build_btc_context():
    """Fetch and prepare BTC 1h data for context features from local parquet."""
    print("📊 Loading BTC context from local OHLCV...")
    btc_path = OHLCV_DIR / "BTCUSDT_USDT.parquet"
    if not btc_path.exists():
        print("⚠️ BTCUSDT_USDT.parquet not found!")
        return None
    
    btc = pd.read_parquet(btc_path)
    btc['timestamp'] = pd.to_datetime(btc['timestamp']).dt.tz_localize(None)
    btc = btc.sort_values('timestamp').reset_index(drop=True)
    
    btc['log_returns'] = np.log(btc['close'] / (btc['close'].shift(1) + 1e-9))
    btc['sma_200'] = btc['close'].rolling(200).mean()
    
    # ADX
    tr = pd.concat([btc['high'] - btc['low'], abs(btc['high'] - btc['close'].shift(1)), abs(btc['low'] - btc['close'].shift(1))], axis=1).max(axis=1)
    pdm = btc['high'].diff(); mdm = -btc['low'].diff()
    pdm = pdm.where((pdm > mdm) & (pdm > 0), 0); mdm = mdm.where((mdm > pdm) & (mdm > 0), 0)
    atr_s = tr.rolling(14).mean()
    pdi = 100 * (pdm.rolling(14).mean() / atr_s.replace(0, np.nan))
    mdi = 100 * (mdm.rolling(14).mean() / atr_s.replace(0, np.nan))
    btc['adx'] = (100 * abs(pdi - mdi) / (pdi + mdi).replace(0, np.nan)).rolling(14).mean()
    
    print(f"   ✅ BTC: {len(btc)} bars ({btc['timestamp'].min()} → {btc['timestamp'].max()})")
    return btc

def process_symbol(parquet_file, clean_name, btc_df, min_date=None):
    """
    Load raw 1h OHLCV from local, compute features, and save.
    """
    df_merged = pd.read_parquet(parquet_file)
    df_merged['timestamp'] = pd.to_datetime(df_merged['timestamp']).dt.tz_localize(None)
    df_merged = df_merged.sort_values('timestamp').reset_index(drop=True)
    
    if min_date:
        df_merged = df_merged[df_merged['timestamp'] >= pd.to_datetime(min_date)]
        
    if len(df_merged) < 200:
        return 'empty'
        
    # Resample 1h → 1d for daily MTF context
    df_1d = df_merged.set_index('timestamp').resample('1D').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    ).dropna().reset_index()
    
    # Compute ALL features with BTC + daily context
    df_enriched = calculate_features(df_merged, df_1d=df_1d, btc_df=btc_df)
    
    # Apply feature shift
    df_enriched = apply_feature_shift(df_enriched)
    
    out_path = SYMBOLS_DIR / f"{clean_name}.parquet"
    df_enriched.to_parquet(out_path, index=False)
    return len(df_enriched)

import subprocess

def run_rebuild(min_date=None, filter_symbols=None, no_fetch=False):
    start = datetime.now()
    print(f"\n{'='*70}")
    print(f"🔄 Rebuild Features from Local Bitget OHLCV — {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Target: {SYMBOLS_DIR}")
    print(f"{'='*70}\n")
    
    if not no_fetch:
        print("📥 Step 1: Syncing latest raw data from Bitget API...")
        fetcher_script = BASE_DIR / "bitget_fetcher.py"
        if fetcher_script.exists():
            try:
                subprocess.run([sys.executable, str(fetcher_script)], check=True)
                print("✅ Raw data sync complete.\n")
            except subprocess.CalledProcessError as e:
                print(f"❌ Error syncing raw data: {e}")
                return
        else:
            print("⚠️ bitget_fetcher.py not found! Skipping API sync.")
            
    print("🧠 Step 2: Extracting ML Features & merging BTC context...")
    
    btc_df = build_btc_context()
    if btc_df is None: return
    
    all_files = list(OHLCV_DIR.glob("*.parquet"))
    if filter_symbols:
        filter_set = {f"{s.upper()}USDT_USDT.parquet" for s in filter_symbols}
        all_files = [f for f in all_files if f.name in filter_set]
        
    total = len(all_files)
    synced = 0
    errors = 0
    
    for i, file_path in enumerate(all_files):
        # MASKUSDT_USDT.parquet -> MASKUSDT
        clean_name = file_path.name.replace('_USDT.parquet', '').replace('.parquet', '')
        # Special catch for those without _USDT suffix
        if not clean_name.endswith('USDT'):
            clean_name += 'USDT'
            
        try:
            result = process_symbol(file_path, clean_name, btc_df, min_date)
            if result == 'empty':
                errors += 1
            else:
                synced += 1
                if synced <= 10 or synced % 20 == 0:
                    print(f"   ✅ {clean_name}: {result} bars (enriched)")
                    
            if total >= 10 and (i + 1) % (total // 10) == 0:
                pct = (i + 1) / total * 100
                elapsed = (datetime.now() - start).total_seconds()
                est_rem = elapsed / (i + 1) * (total - i - 1)
                print(f"   ⏳ Progress: {pct:.0f}% ({i+1}/{total}) | "
                      f"Synced: {synced} | ETA: {est_rem/60:.1f}m")
            gc.collect()
        except Exception as e:
            print(f"   ❌ {clean_name}: {e}")
            errors += 1
            
    duration = datetime.now() - start
    print(f"\n{'='*70}")
    print(f"✅ Rebuild Complete in {duration}")
    print(f"   Processed: {synced} | Errors (insufficient data): {errors}")
    print(f"   Output: {SYMBOLS_DIR}")
    print(f"{'='*70}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-date", type=str, default="2023-01-01", help="Minimum date to load")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols (e.g. BTC ETH)")
    parser.add_argument("--no-fetch", action="store_true", help="Skip fetching from API, only rebuild from local data")
    args = parser.parse_args()
    
    run_rebuild(min_date=args.min_date, filter_symbols=args.symbols, no_fetch=args.no_fetch)
