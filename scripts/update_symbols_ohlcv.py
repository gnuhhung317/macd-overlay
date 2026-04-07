#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from binance_fetcher import BinanceFetcher, OHLCV_DIR


def to_ccxt(sym: str) -> str:
    s = sym.upper()
    if s.endswith("USDT"):
        return f"{s[:-4]}/USDT"
    return f"{s}/USDT"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="")
    args = parser.parse_args()

    fetcher = BinanceFetcher()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        print("No symbols provided. Exiting.")
        return

    for sym in symbols:
        ccxt_sym = to_ccxt(sym)
        clean = fetcher.get_clean_symbol(ccxt_sym)
        target_path = OHLCV_DIR / f"{clean}.parquet"
        existing = None
        if target_path.exists():
            try:
                existing = pd.read_parquet(target_path)
            except Exception:
                existing = None

        print(f"Syncing {ccxt_sym} -> {target_path.name}...")
        df = fetcher.sync_ohlcv(ccxt_sym, existing)
        if df is None or df.empty:
            print(f"  No data fetched for {ccxt_sym}")
            continue
        df = fetcher.fill_gaps(ccxt_sym, df)
        df.to_parquet(target_path, index=False)
        print(f"  Saved {len(df)} rows ({df['timestamp'].min()} -> {df['timestamp'].max()})")


if __name__ == '__main__':
    main()
