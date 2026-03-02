#!/usr/bin/env python3
import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import os
import argparse

# Config
DATA_ROOT = Path("kraken-data")
OHLCV_DIR = DATA_ROOT / "ohlcv"
FUNDING_DIR = DATA_ROOT / "funding"

# Create directories
OHLCV_DIR.mkdir(parents=True, exist_ok=True)
FUNDING_DIR.mkdir(parents=True, exist_ok=True)

class KrakenFetcher:
    def __init__(self):
        self.exchange = ccxt.krakenfutures({
            'enableRateLimit': True,
        })
        self.start_ts = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

    def find_start_date(self, symbol):
        """Find earliest available data timestamp (probing yearly backwards)"""
        current_year = datetime.now(timezone.utc).year
        # Probing backwards is much faster for new coins
        start_year = 2020
        years = list(range(current_year, start_year - 1, -1))
        
        found_earliest = None
        for year in years:
            test_ts = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', since=test_ts, limit=1)
                if ohlcv:
                    found_earliest = ohlcv[0][0]
                    # Continue searching backwards
                else:
                    # No data this year, earliest must be in the next year (found_earliest)
                    if found_earliest:
                        return found_earliest
            except Exception:
                if found_earliest: return found_earliest
                continue
        return found_earliest

    def get_all_symbols(self):
        """Get all USDT-M perpetual symbols from Kraken Futures"""
        print("Fetching Kraken Futures markets...")
        try:
            markets = self.exchange.load_markets()
            symbols = [
                m['symbol'] for m in markets.values() 
                if m['active'] and (m['quote'] == 'USDT' or m['quote'] == 'USD') and m['type'] == 'swap'
            ]
            return symbols
        except Exception as e:
            print(f"Error fetching symbols: {e}")
            return []

    def get_clean_symbol(self, symbol):
        """Standardize symbol to Pair_Quote format"""
        # PF_BTCUSD:USD -> BTCUSDT_USDT (or keep USD if preferred, but system usually expects USDT label)
        # Let's normalize everything to _USDT for the pipeline
        clean = symbol.replace('PF_', '').replace('/', '').split(':')[0] # Get Pair
        if clean.endswith('USD'):
            clean = clean.replace('USD', 'USDT')
        if '_' not in clean:
            clean = f"{clean}_USDT"
        return clean

    def fetch_ohlcv_range(self, symbol, start_ts, end_ts):
        all_ohlcv = []
        since = start_ts
        limit = 500 # Kraken limit
        
        while since < end_ts:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', since=since, limit=limit)
                if not ohlcv: break
                
                ohlcv = [x for x in ohlcv if x[0] <= end_ts]
                if not ohlcv: break
                
                all_ohlcv.extend(ohlcv)
                last_ts = ohlcv[-1][0]
                if last_ts == since:
                    since += 3600000
                    continue
                
                since = last_ts + 1
                time.sleep(self.exchange.rateLimit / 1000)
            except Exception as e:
                time.sleep(5)
                continue
        return all_ohlcv

    def sync_ohlcv(self, symbol, existing_df=None):
        current_time = int(time.time() * 1000)
        if existing_df is not None and not existing_df.empty:
            start_fetch_ts = existing_df['timestamp'].max().value // 10**6 + 1
        else:
            # Dynamically find start date
            print(f"  Probing start date for {symbol}...")
            start_fetch_ts = self.find_start_date(symbol)
            if start_fetch_ts is None:
                start_fetch_ts = self.start_ts

        new_data = self.fetch_ohlcv_range(symbol, start_fetch_ts, current_time)
        if not new_data: return existing_df
        
        new_df = pd.DataFrame(new_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
        
        if existing_df is not None:
            combined = pd.concat([existing_df, new_df]).drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
            return combined
        return new_df

    def fetch_funding_rates(self, symbol):
        """Fetch funding rates from Kraken"""
        print(f"  Fetching Funding Rates for {symbol}...")
        try:
            rates = self.exchange.fetch_funding_rate_history(symbol, since=self.start_ts, limit=1000)
            if not rates: return pd.DataFrame()
            df = pd.DataFrame(rates)
            df = df[['timestamp', 'fundingRate']].copy()
            df.rename(columns={'fundingRate': 'funding_rate'}, inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df.sort_values('timestamp').reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def run(self, limit_coins=None):
        symbols = self.get_all_symbols()
        if not symbols: return
        if limit_coins: symbols = symbols[:limit_coins]
        
        for i, symbol in enumerate(symbols):
            clean_name = self.get_clean_symbol(symbol)
            ohlcv_path = OHLCV_DIR / f"{clean_name}.parquet"
            funding_path = FUNDING_DIR / f"{clean_name}.parquet"
            
            print(f"[{i+1}/{len(symbols)}] Kraken: {symbol}...")
            
            existing_ohlcv = pd.read_parquet(ohlcv_path) if ohlcv_path.exists() else None
            df_ohlcv = self.sync_ohlcv(symbol, existing_ohlcv)
            
            if df_ohlcv is not None and not df_ohlcv.empty:
                df_ohlcv.to_parquet(ohlcv_path, index=False)
            
            if not funding_path.exists():
                df_funding = self.fetch_funding_rates(symbol)
                if not df_funding.empty:
                    df_funding.to_parquet(funding_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    KrakenFetcher().run(limit_coins=args.limit)
