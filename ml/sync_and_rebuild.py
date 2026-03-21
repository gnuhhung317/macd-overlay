#!/usr/bin/env python3
"""
🔄 Sync & Rebuild - Fetch raw OHLCV from Binance Futures, compute features
with BTC context, and save enriched data to symbols_v3/ for backtest.

Usage:
    python ml/sync_and_rebuild.py                    # Incremental update (append new bars)
    python ml/sync_and_rebuild.py --days 600         # Fetch last 600 days
    python ml/sync_and_rebuild.py --full             # Full re-download (~2000 days)
    python ml/sync_and_rebuild.py --symbols BTC ETH  # Specific symbols only
"""
import sys, os, time, gc, argparse
import pandas as pd
import numpy as np
import ccxt
from pathlib import Path
from datetime import datetime, timedelta

# Root path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from sniper_bot.feature import calculate_features, apply_feature_shift

# ============================================================
# CONFIG
# ============================================================
SYMBOLS_DIR = BASE_DIR / "data" / "processed" / "symbols_v3"
SYMBOLS_DIR.mkdir(parents=True, exist_ok=True)

exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'},
    'timeout': 15000
})

# ============================================================
# FETCH HELPERS
# ============================================================
def fetch_ohlcv_with_retry(symbol, timeframe, limit=1000, since=None, max_retries=3):
    for attempt in range(max_retries):
        try:
            return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)
        except ccxt.RateLimitExceeded:
            sleep_time = 5 * (attempt + 1)
            print(f"  ⏳ Rate limit {symbol}. Sleep {sleep_time}s...")
            time.sleep(sleep_time)
        except ccxt.NetworkError as e:
            print(f"  🌐 Network error {symbol}: {e}. Retry {attempt+1}/{max_retries}...")
            time.sleep(2)
        except (ccxt.BadSymbol, ccxt.ExchangeError) as e:
            return []
        except Exception as e:
            print(f"  ❌ Fatal error {symbol}: {e}")
            return []
    return []

def fetch_ohlcv_paginated(symbol, timeframe, since_ms, until_ms=None):
    """Fetch OHLCV from 'since_ms' to now via pagination."""
    all_ohlcv = []
    if until_ms is None:
        until_ms = int(time.time() * 1000)
    
    cursor = since_ms
    while cursor < until_ms:
        batch = fetch_ohlcv_with_retry(symbol, timeframe, limit=1000, since=cursor)
        if not batch:
            break
        all_ohlcv.extend(batch)
        cursor = batch[-1][0] + 1
        if len(batch) < 1000:
            break
        time.sleep(0.05)
    
    return all_ohlcv

def raw_to_df(ohlcv):
    """Convert raw OHLCV list to DataFrame."""
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df.sort_values('timestamp').reset_index(drop=True)

# ============================================================
# CORE SYNC LOGIC
# ============================================================
def get_all_usdt_perpetuals():
    """Get all active USDT-M perpetual symbols from Binance."""
    print("📋 Loading Binance markets...")
    markets = exchange.load_markets()
    symbol_map = []
    for m in markets.values():
        if m['quote'] == 'USDT' and m['active'] and m.get('type') == 'swap' and m.get('linear'):
            api_sym = m['symbol']
            clean_name = api_sym.split(':')[0].replace('/', '')
            symbol_map.append((api_sym, clean_name))
    symbol_map.sort(key=lambda x: x[1])
    print(f"   Found {len(symbol_map)} USDT perpetuals.")
    return symbol_map

def fetch_btc_context(days):
    """Fetch and prepare BTC 1h data for context features."""
    print("📊 Loading BTC context...")
    since_ms = int((time.time() - days * 86400) * 1000)
    raw = fetch_ohlcv_paginated('BTC/USDT:USDT', '1h', since_ms)
    if not raw:
        print("⚠️ Failed to fetch BTC data!")
        return None
    
    btc = raw_to_df(raw)
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

def sync_symbol(api_sym, clean_name, days, btc_df, force_full=False):
    """
    Fetch 1h OHLCV, compute features with BTC + daily context,
    and save enriched data to symbols_v3/.
    """
    parquet_path = SYMBOLS_DIR / f"{clean_name}.parquet"
    
    # Determine fetch start (incremental vs full)
    if not force_full and parquet_path.exists():
        try:
            existing = pd.read_parquet(parquet_path, columns=['timestamp'])
            if 'timestamp' in existing.columns and len(existing) > 0:
                existing['timestamp'] = pd.to_datetime(existing['timestamp'])
                last_ts = existing['timestamp'].max()
                since_ms = int(last_ts.timestamp() * 1000) - (200 * 3600000)  # 200 bar overlap for indicator warmup
                bars_behind = (time.time() * 1000 - int(last_ts.timestamp() * 1000)) / 3600000
                if bars_behind < 2:
                    return 'skip'
            else:
                since_ms = int((time.time() - days * 86400) * 1000)
        except:
            since_ms = int((time.time() - days * 86400) * 1000)
    else:
        since_ms = int((time.time() - days * 86400) * 1000)
    
    # Fetch raw 1h OHLCV
    raw = fetch_ohlcv_paginated(api_sym, '1h', since_ms)
    if not raw:
        return 'empty'
    
    df_new = raw_to_df(raw)
    
    # Merge with existing raw data (use only OHLCV columns from existing)
    if parquet_path.exists() and not force_full:
        try:
            existing = pd.read_parquet(parquet_path)
            ohlcv_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            existing_cols = [c for c in ohlcv_cols if c in existing.columns]
            existing = existing[existing_cols].copy()
            existing['timestamp'] = pd.to_datetime(existing['timestamp']).dt.tz_localize(None)
            
            df_merged = pd.concat([existing, df_new], ignore_index=True)
            df_merged = df_merged.drop_duplicates(subset='timestamp', keep='last')
            df_merged = df_merged.sort_values('timestamp').reset_index(drop=True)
        except:
            df_merged = df_new
    else:
        df_merged = df_new
    
    # Resample 1h → 1d for daily MTF context
    df_1d = df_merged.set_index('timestamp').resample('1D').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    ).dropna().reset_index()
    
    # Compute ALL features with BTC + daily context (matches training pipeline)
    df_enriched = calculate_features(df_merged, df_1d=df_1d, btc_df=btc_df)
    
    # Apply feature shift (shift computed features by 1 bar) - matches training pipeline
    df_enriched = apply_feature_shift(df_enriched)
    
    # Save enriched data
    df_enriched.to_parquet(parquet_path, index=False)
    return len(df_enriched)

def run_sync(days=500, force_full=False, filter_symbols=None):
    """Main sync loop."""
    start = datetime.now()
    print(f"\n{'='*70}")
    print(f"🔄 Sync & Rebuild — {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Target: {SYMBOLS_DIR}")
    print(f"   Days: {days} | Full: {force_full}")
    print(f"{'='*70}\n")
    
    symbol_map = get_all_usdt_perpetuals()
    
    if filter_symbols:
        filter_set = {s.upper() + 'USDT' for s in filter_symbols}
        symbol_map = [(api, clean) for api, clean in symbol_map if clean in filter_set]
        print(f"   Filtered to {len(symbol_map)} symbols: {[s[1] for s in symbol_map]}")
    
    # Fetch BTC context once
    btc_df = fetch_btc_context(days)
    
    total = len(symbol_map)
    synced = 0
    skipped = 0
    errors = 0
    
    for i, (api_sym, clean_name) in enumerate(symbol_map):
        try:
            result = sync_symbol(api_sym, clean_name, days, btc_df, force_full)
            
            if result == 'skip':
                skipped += 1
            elif result == 'empty':
                errors += 1
            else:
                synced += 1
                if synced <= 5 or synced % 20 == 0:
                    print(f"   ✅ {clean_name}: {result} bars (enriched)")
            
            # Progress every 10%
            if total >= 10 and (i + 1) % (total // 10) == 0:
                pct = (i + 1) / total * 100
                elapsed = (datetime.now() - start).total_seconds()
                est_rem = elapsed / (i + 1) * (total - i - 1)
                print(f"   ⏳ Progress: {pct:.0f}% ({i+1}/{total}) | "
                      f"Synced: {synced} | Skipped: {skipped} | "
                      f"ETA: {est_rem/60:.1f}m")
            
            gc.collect()
            
        except Exception as e:
            print(f"   ❌ {clean_name}: {e}")
            errors += 1
            continue
    
    duration = datetime.now() - start
    print(f"\n{'='*70}")
    print(f"✅ Sync Complete in {duration}")
    print(f"   Synced: {synced} | Skipped (up-to-date): {skipped} | Errors: {errors}")
    print(f"   Output: {SYMBOLS_DIR}")
    print(f"{'='*70}")

# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch & rebuild enriched data for backtest_sniper.py")
    parser.add_argument("--days", type=int, default=500, help="Days of history to fetch (default: 500)")
    parser.add_argument("--full", action="store_true", help="Force full re-download (~2000 days)")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols (e.g. BTC ETH SOL)")
    args = parser.parse_args()
    
    if args.full:
        args.days = 2000
    
    run_sync(days=args.days, force_full=args.full, filter_symbols=args.symbols)
