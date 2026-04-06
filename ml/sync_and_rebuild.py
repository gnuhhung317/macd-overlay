#!/usr/bin/env python3
"""Sync raw OHLCV and update processed symbols_v3 for backtest_sniper.

Usage:
    python ml/sync_and_rebuild.py                    # Fetch OHLCV + update symbols_v3
    python ml/sync_and_rebuild.py --days 1200        # Fetch/maintain last 1200 days
    python ml/sync_and_rebuild.py --full             # Full re-download (~2000 days)
    python ml/sync_and_rebuild.py --symbols BTC ETH  # Specific symbols only
    python ml/sync_and_rebuild.py --check-only       # Skip API and validate local OHLCV only
    python ml/sync_and_rebuild.py --no-v3            # Fetch OHLCV only, skip symbols_v3 update
"""

import argparse
import gc
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import ccxt
import numpy as np
import pandas as pd

# Root path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from sniper_bot.feature import calculate_features, apply_feature_shift

# ============================================================
# CONFIG
# ============================================================
OHLCV_DIR = BASE_DIR / "data" / "ohlcv"
SYMBOLS_V3_DIR = BASE_DIR / "data" / "processed" / "symbols_v3"
OHLCV_DIR.mkdir(parents=True, exist_ok=True)
SYMBOLS_V3_DIR.mkdir(parents=True, exist_ok=True)

exchange = ccxt.binance({
    "enableRateLimit": True,
    "options": {"defaultType": "swap"},
    "timeout": 15000,
})

TIMEFRAME = "1h"
OVERLAP_BARS = 200
DEFAULT_HISTORY_DAYS = 1200


# ============================================================
# FETCH HELPERS
# ============================================================
def fetch_ohlcv_with_retry(symbol, timeframe, limit=1000, since=None, max_retries=3):
    for attempt in range(max_retries):
        try:
            return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)
        except ccxt.RateLimitExceeded:
            sleep_time = 5 * (attempt + 1)
            print(f"  Rate limit {symbol}. Sleep {sleep_time}s...")
            time.sleep(sleep_time)
        except ccxt.NetworkError as e:
            print(f"  Network error {symbol}: {e}. Retry {attempt + 1}/{max_retries}...")
            time.sleep(2)
        except (ccxt.BadSymbol, ccxt.ExchangeError):
            return []
        except Exception as e:
            print(f"  Fatal error {symbol}: {e}")
            return []
    return []


def fetch_ohlcv_paginated(symbol, timeframe, since_ms, until_ms=None):
    """Fetch OHLCV from since_ms to now via pagination."""
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
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.sort_values("timestamp").reset_index(drop=True)


def resolve_ohlcv_path(clean_name: str) -> Path:
    preferred = OHLCV_DIR / f"{clean_name}_USDT.parquet"
    legacy = OHLCV_DIR / f"{clean_name}.parquet"
    if preferred.exists():
        return preferred
    if legacy.exists():
        return legacy
    return preferred


def load_raw_df(file_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


# ============================================================
# CORE SYNC LOGIC
# ============================================================
def get_all_usdt_perpetuals():
    """Get all active USDT-M perpetual symbols from Binance."""
    print("Loading Binance markets...")
    markets = exchange.load_markets()
    symbol_map = []
    for m in markets.values():
        if m["quote"] == "USDT" and m["active"] and m.get("type") == "swap" and m.get("linear"):
            api_sym = m["symbol"]
            clean_name = api_sym.split(":")[0].replace("/", "")
            symbol_map.append((api_sym, clean_name))
    symbol_map.sort(key=lambda x: x[1])
    print(f"   Found {len(symbol_map)} USDT perpetuals.")
    return symbol_map


def sync_raw_symbol(
    api_sym: str,
    clean_name: str,
    days: int,
    force_full: bool = False,
    check_only: bool = False,
) -> Tuple[str, Optional[pd.DataFrame], Path]:
    """Fetch 1h OHLCV and save raw bars to data/ohlcv."""
    ohlcv_file = resolve_ohlcv_path(clean_name)
    ohlcv_file.parent.mkdir(parents=True, exist_ok=True)

    if check_only:
        if not ohlcv_file.exists():
            if clean_name.endswith("USDT"):
                print(f"  Warning: {clean_name} OHLCV not found at {ohlcv_file.name}. Skipping.")
            return "empty", None, ohlcv_file
        return "ok", load_raw_df(ohlcv_file), ohlcv_file

    target_since_ms = int((time.time() - days * 86400) * 1000)
    since_ms = target_since_ms

    df_local = pd.DataFrame()
    if ohlcv_file.exists():
        df_local = load_raw_df(ohlcv_file)

    fetch_needed = force_full or len(df_local) == 0
    if not fetch_needed and len(df_local) > 0:
        first_ts = df_local["timestamp"].min()
        last_ts = df_local["timestamp"].max()
        first_ms = int(first_ts.timestamp() * 1000)
        last_ms = int(last_ts.timestamp() * 1000)
        needs_backfill = first_ms > target_since_ms
        bars_behind = (time.time() * 1000 - last_ms) / 3600000

        if needs_backfill:
            fetch_needed = True
            since_ms = target_since_ms
        elif bars_behind >= 2:
            fetch_needed = True
            since_ms = last_ms - (OVERLAP_BARS * 3600000)

    if not fetch_needed:
        return "skip", df_local, ohlcv_file

    raw = fetch_ohlcv_paginated(api_sym, TIMEFRAME, since_ms)
    if not raw:
        if len(df_local) == 0:
            return "empty", None, ohlcv_file
        return "skip", df_local, ohlcv_file

    df_new = raw_to_df(raw)
    if len(df_local) > 0:
        df_merged = pd.concat([df_local, df_new], ignore_index=True)
        df_merged = df_merged.drop_duplicates(subset="timestamp", keep="last")
        df_merged = df_merged.sort_values("timestamp").reset_index(drop=True)
    else:
        df_merged = df_new

    df_merged.to_parquet(ohlcv_file, index=False)
    return "synced", df_merged, ohlcv_file


def build_btc_context_features(btc_df: pd.DataFrame) -> pd.DataFrame:
    btc = btc_df.copy()
    btc["log_returns"] = np.log(btc["close"] / (btc["close"].shift(1) + 1e-9))
    btc["sma_200"] = btc["close"].rolling(200).mean()

    tr = pd.concat(
        [
            btc["high"] - btc["low"],
            abs(btc["high"] - btc["close"].shift(1)),
            abs(btc["low"] - btc["close"].shift(1)),
        ],
        axis=1,
    ).max(axis=1)
    pdm = btc["high"].diff()
    mdm = -btc["low"].diff()
    pdm = pdm.where((pdm > mdm) & (pdm > 0), 0)
    mdm = mdm.where((mdm > pdm) & (mdm > 0), 0)
    atr_s = tr.rolling(14).mean()
    pdi = 100 * (pdm.rolling(14).mean() / atr_s.replace(0, np.nan))
    mdi = 100 * (mdm.rolling(14).mean() / atr_s.replace(0, np.nan))
    btc["adx"] = (100 * abs(pdi - mdi) / (pdi + mdi).replace(0, np.nan)).rolling(14).mean()
    return btc


def needs_v3_rebuild(raw_df: pd.DataFrame, v3_path: Path) -> bool:
    if not v3_path.exists():
        return True
    try:
        v3_ts = pd.read_parquet(v3_path, columns=["timestamp"])
        v3_ts["timestamp"] = pd.to_datetime(v3_ts["timestamp"], errors="coerce")
        raw_max = pd.to_datetime(raw_df["timestamp"], errors="coerce").max()
        v3_max = v3_ts["timestamp"].max()
        if pd.isna(raw_max) or pd.isna(v3_max):
            return True
        return bool(v3_max < raw_max)
    except Exception:
        return True


def update_symbol_v3(clean_name: str, raw_df: pd.DataFrame, btc_df: pd.DataFrame) -> str:
    if raw_df is None or raw_df.empty:
        return "empty"

    parquet_path = SYMBOLS_V3_DIR / f"{clean_name}.parquet"
    if not needs_v3_rebuild(raw_df, parquet_path):
        return "skip"

    df_1d = (
        raw_df.set_index("timestamp")
        .resample("1D")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )

    df_enriched = calculate_features(raw_df.copy(), df_1d=df_1d, btc_df=btc_df)
    df_enriched = apply_feature_shift(df_enriched)
    df_enriched.to_parquet(parquet_path, index=False)
    return str(len(df_enriched))


def run_sync(days=DEFAULT_HISTORY_DAYS, force_full=False, filter_symbols=None, check_only=False, update_v3_enabled=True):
    """Main sync loop."""
    start = datetime.now()
    print("\n" + "=" * 70)
    print(f"OHLCV Sync - {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Raw target: {OHLCV_DIR}")
    print(f"   V3 target: {SYMBOLS_V3_DIR}")
    print(
        f"   Days: {days if not check_only else 'N/A (Check Only)'} | "
        f"Full: {force_full} | Check Only: {check_only} | Update v3: {update_v3_enabled and not check_only}"
    )
    print("=" * 70 + "\n")

    symbol_map = get_all_usdt_perpetuals()

    if filter_symbols:
        filter_set = {s.upper() + "USDT" for s in filter_symbols}
        symbol_map = [(api, clean) for api, clean in symbol_map if clean in filter_set]
        print(f"   Filtered to {len(symbol_map)} symbols: {[s[1] for s in symbol_map]}")

    btc_ctx = None
    if update_v3_enabled and not check_only:
        btc_api = next((api for api, clean in symbol_map if clean == "BTCUSDT"), "BTC/USDT:USDT")
        btc_status, btc_raw_df, _ = sync_raw_symbol(
            api_sym=btc_api,
            clean_name="BTCUSDT",
            days=days,
            force_full=force_full,
            check_only=False,
        )
        if btc_raw_df is None or btc_raw_df.empty:
            print("Warning: Cannot build BTC context, disabling symbols_v3 updates.")
            update_v3_enabled = False
        else:
            btc_ctx = build_btc_context_features(btc_raw_df)
            print(
                f"   BTC context ready ({btc_status}): "
                f"{btc_ctx['timestamp'].min()} -> {btc_ctx['timestamp'].max()} | rows={len(btc_ctx)}"
            )

    total = len(symbol_map)
    raw_synced = 0
    raw_skipped = 0
    v3_synced = 0
    v3_skipped = 0
    errors = 0

    for i, (api_sym, clean_name) in enumerate(symbol_map):
        try:
            raw_status, raw_df, _ = sync_raw_symbol(api_sym, clean_name, days, force_full, check_only)

            if raw_status in {"synced", "ok"}:
                raw_synced += 1
            elif raw_status == "skip":
                raw_skipped += 1
            else:
                errors += 1

            v3_status = "disabled"
            if update_v3_enabled and not check_only and btc_ctx is not None and raw_df is not None:
                v3_status = update_symbol_v3(clean_name, raw_df, btc_ctx)
                if v3_status == "skip":
                    v3_skipped += 1
                elif v3_status == "empty":
                    errors += 1
                else:
                    v3_synced += 1

            if (raw_synced + raw_skipped) <= 5 or (i + 1) % 20 == 0:
                if update_v3_enabled and not check_only:
                    print(f"   {clean_name}: raw={raw_status}, v3={v3_status}")
                else:
                    print(f"   {clean_name}: raw={raw_status}")

            if total >= 10 and (i + 1) % max(1, total // 10) == 0:
                pct = (i + 1) / total * 100
                elapsed = (datetime.now() - start).total_seconds()
                est_rem = elapsed / (i + 1) * (total - i - 1)
                print(
                    f"   Progress: {pct:.0f}% ({i + 1}/{total}) | "
                    f"Raw synced={raw_synced} skipped={raw_skipped} | "
                    f"V3 synced={v3_synced} skipped={v3_skipped} | "
                    f"Errors={errors} | ETA: {est_rem / 60:.1f}m"
                )

            gc.collect()
        except Exception as e:
            print(f"   Error {clean_name}: {e}")
            errors += 1
            continue

    duration = datetime.now() - start
    print("\n" + "=" * 70)
    print(f"Sync Complete in {duration}")
    print(f"   Raw: synced={raw_synced}, skipped={raw_skipped}")
    if update_v3_enabled and not check_only:
        print(f"   V3:  synced={v3_synced}, skipped={v3_skipped}")
    print(f"   Errors: {errors}")
    print(f"   Raw output: {OHLCV_DIR}")
    if update_v3_enabled and not check_only:
        print(f"   V3 output: {SYMBOLS_V3_DIR}")
    print("=" * 70)


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch OHLCV and optionally update processed symbols_v3")
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_HISTORY_DAYS,
        help=f"Days of history to fetch/maintain (default: {DEFAULT_HISTORY_DAYS})",
    )
    parser.add_argument("--full", action="store_true", help="Force full re-download (~2000 days)")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols (e.g. BTC ETH SOL)")
    parser.add_argument("--check-only", dest="check_only", action="store_true", help="Skip API calls and only check local OHLCV files")
    parser.add_argument("--no-fetch", dest="check_only", action="store_true", help="Deprecated alias for --check-only")
    parser.add_argument("--no-v3", action="store_true", help="Fetch raw OHLCV only, skip symbols_v3 update")
    args = parser.parse_args()

    if args.full:
        args.days = 2000

    run_sync(
        days=args.days,
        force_full=args.full,
        filter_symbols=args.symbols,
        check_only=args.check_only,
        update_v3_enabled=not bool(args.no_v3),
    )
