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
DATA_ROOT = Path("bybit-data")
OHLCV_DIR = DATA_ROOT / "ohlcv"
FUNDING_DIR = DATA_ROOT / "funding"

# Create directories
OHLCV_DIR.mkdir(parents=True, exist_ok=True)
FUNDING_DIR.mkdir(parents=True, exist_ok=True)

class BybitFetcher:
    def __init__(self):
        self.exchange = ccxt.bybit({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear',  # USDT-M Perpetual
            }
        })
        self.start_ts = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

    def get_all_symbols(self):
        """Get all USDT-M perpetual symbols from Bybit"""
        print("Fetching Bybit markets...")
        try:
            markets = self.exchange.load_markets()
            symbols = [
                m['symbol'] for m in markets.values() 
                if m['linear'] and m['active'] and m['quote'] == 'USDT' and m['type'] == 'swap'
            ]
            return symbols
        except Exception as e:
            print(f"Error fetching symbols: {e}")
            return []

    def find_start_date(self, symbol):
        """Find the earliest available data timestamp (probing yearly)"""
        current_year = datetime.now(timezone.utc).year
        # Bybit history usually starts around 2020-2021
        years = list(range(2020, current_year + 1))
        
        for year in years:
            test_ts = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', since=test_ts, limit=1)
                if ohlcv:
                    return ohlcv[0][0]
            except Exception:
                time.sleep(0.5)
                continue
        return None

    def get_clean_symbol(self, symbol):
        """Standardize symbol to Pair_Quote format"""
        # BTC/USDT:USDT -> BTCUSDT_USDT
        base = symbol.replace('/', '').split(':')[0].split('_')[0]
        if base.endswith('USDC'):
            base = base.replace('USDC', 'USDT')
        elif base.endswith('USD'):
            base = base.replace('USD', 'USDT')
            
        if not base.endswith('USDT'):
            base = f"{base}USDT"
            
        return f"{base}_USDT"

    def fetch_ohlcv_range(self, symbol, start_ts, end_ts):
        """Fetch range using forward pagination"""
        all_ohlcv = []
        since = start_ts
        limit = 1000 # Bybit max
        
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
        """Fetch funding rates from Bybit"""
        print(f"  Fetching Funding Rates for {symbol}...")
        try:
            rates = self.exchange.fetch_funding_rate_history(symbol, since=self.start_ts, limit=1000)
            if not rates: return pd.DataFrame()
            df = pd.DataFrame(rates)
            df = df[['timestamp', 'fundingRate']].copy()
            df.rename(columns={'fundingRate': 'funding_rate'}, inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df.sort_values('timestamp').reset_index(drop=True)
        except Exception as e:
            return pd.DataFrame()

    def run(self, limit_coins=None):
        symbols = self.get_all_symbols()
        if not symbols: return
        if limit_coins: symbols = symbols[:limit_coins]
        
        for i, symbol in enumerate(symbols):
            clean_name = self.get_clean_symbol(symbol)
            ohlcv_path = OHLCV_DIR / f"{clean_name}.parquet"
            funding_path = FUNDING_DIR / f"{clean_name}.parquet"
            
            print(f"[{i+1}/{len(symbols)}] Bybit: {symbol}...")
            
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
    BybitFetcher().run(limit_coins=args.limit)
