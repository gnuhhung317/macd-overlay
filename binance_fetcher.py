#!/usr/bin/env python3
import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import os
import argparse

# Config - Using existing data directory
DATA_ROOT = Path("data")
OHLCV_DIR = DATA_ROOT / "ohlcv"
FUNDING_DIR = DATA_ROOT / "funding"

# Create directories
OHLCV_DIR.mkdir(parents=True, exist_ok=True)
FUNDING_DIR.mkdir(parents=True, exist_ok=True)

class BinanceFetcher:
    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',  # USDT-M Perpetual
            }
        })
        self.start_ts = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

    def get_all_symbols(self):
        """Get all USDT-M perpetual symbols from Binance"""
        print("Fetching markets...")
        try:
            markets = self.exchange.load_markets()
            symbols = [
                m['symbol'] for m in markets.values() 
                if m['linear'] and m['active'] and m['quote'] == 'USDT' and m['type'] == 'swap'
            ]
            return symbols
        except Exception as e:
            print(f"Error fetching markets: {e}")
            return []

    def get_clean_symbol(self, symbol):
        """
        Convert symbol from CCXT format (e.g., BTC/USDT:USDT) to stored format (BTCUSDT_USDT).
        Existing files follow the pattern BTCUSDT_USDT.parquet.
        """
        clean_symbol = symbol.replace('/', '').replace(':', '_')
        return clean_symbol

    def find_start_date(self, symbol):
        """Find the earliest available data timestamp (probing yearly then monthly)"""
        current_year = datetime.now(timezone.utc).year
        # Binance has history for many coins starting 2019-2020
        years = list(range(2019, current_year + 2)) 
        
        first_found_ts = None
        last_empty_year = None
        
        for year in years:
            test_ts = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
            if test_ts < self.start_ts:
                test_ts = self.start_ts

            # YEARLY PROBE
            try:
                # Try to fetch 1 candle
                ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', since=test_ts, limit=1)
                
                if ohlcv:
                    first_found_ts = ohlcv[0][0]
                    
                    if last_empty_year is not None and last_empty_year == year - 1:
                         # MONTHLY PROBE for last_empty_year
                         print(f"    Probing months in {last_empty_year}...")
                         for month in range(1, 13):
                             month_ts = int(datetime(last_empty_year, month, 1, tzinfo=timezone.utc).timestamp() * 1000)
                             if month_ts < self.start_ts: continue
                             
                             try:
                                 m_ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', since=month_ts, limit=1)
                                 if m_ohlcv:
                                     print(f"    Found start in {last_empty_year}-{month:02d}")
                                     return m_ohlcv[0][0]
                             except:
                                 pass
                    
                    return first_found_ts
                else:
                    last_empty_year = year
            except Exception as e:
                last_empty_year = year
                time.sleep(0.5)
                continue

        # If probing failed, try without since (latest)
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=1)
            if ohlcv:
                return None 
        except:
            pass
        return None

    def fetch_ohlcv_range(self, symbol, start_ts, end_ts):
        """Fetch OHLCV data for a specific range using forward pagination"""
        all_ohlcv = []
        since = start_ts
        limit = 1000 # Binance supports up to 1000
        
        while since < end_ts:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', since=since, limit=limit)
                
                if not ohlcv:
                    break
                
                # Filter out candles beyond end_ts
                ohlcv = [x for x in ohlcv if x[0] <= end_ts]
                if not ohlcv:
                    break
                    
                all_ohlcv.extend(ohlcv)
                
                last_ts = ohlcv[-1][0]
                if last_ts == since: # Check progress
                    if len(ohlcv) == 1:
                        since += 3600000 
                        continue
                    break
                
                since = last_ts + 1
                time.sleep(self.exchange.rateLimit / 1000)
                
            except Exception as e:
                # print(f"    Error fetching range {symbol}: {e}")
                time.sleep(5)
                continue
                
        return all_ohlcv

    def sync_ohlcv(self, symbol, existing_df=None):
        """Sync OHLCV data: append new data or fetch from scratch"""
        current_time = int(time.time() * 1000)
        
        if existing_df is not None and not existing_df.empty:
            last_ts = existing_df['timestamp'].max().value // 10**6 # Convert ns to ms
            start_fetch_ts = last_ts + 1
            print(f"  {symbol}: Updating from {datetime.fromtimestamp(start_fetch_ts/1000, tz=timezone.utc)}...")
        else:
            print(f"  {symbol}: Finding start date...")
            start_fetch_ts = self.find_start_date(symbol)
            if not start_fetch_ts:
                start_fetch_ts = self.start_ts
                print(f"    Could not discover start date, using {datetime.fromtimestamp(start_fetch_ts/1000, tz=timezone.utc)}")
            else:
                print(f"    Start date found: {datetime.fromtimestamp(start_fetch_ts/1000, tz=timezone.utc)}")

        # Fetch forward until now
        new_data = self.fetch_ohlcv_range(symbol, start_fetch_ts, current_time)
        
        if not new_data:
            print(f"    No new data found.")
            return existing_df
            
        new_df = pd.DataFrame(new_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
        
        if existing_df is not None:
            combined = pd.concat([existing_df, new_df])
            combined = combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
            return combined
        else:
            return new_df

    def fill_gaps(self, symbol, df):
        """Check for gaps > 1h and fill them"""
        if df is None or df.empty:
            return df
            
        df = df.sort_values('timestamp')
        df['diff'] = df['timestamp'].diff()
        
        gap_mask = df['diff'] > pd.Timedelta(minutes=90)
        gaps = df[gap_mask]
        
        if gaps.empty:
            return df
            
        print(f"  {symbol}: Found {len(gaps)} gaps. Filling...")
        
        new_rows = []
        for idx, row in gaps.iterrows():
            gap_end = row['timestamp']
            prev_idx = df.index.get_loc(idx) - 1
            gap_start = df.iloc[prev_idx]['timestamp']
            
            start_ms = int(gap_start.timestamp() * 1000)
            end_ms = int(gap_end.timestamp() * 1000)
            
            print(f"    Filling gap: {gap_start} -> {gap_end}")
            
            fill_data = self.fetch_ohlcv_range(symbol, start_ms + 1, end_ms - 1)
            if fill_data:
                new_rows.extend(fill_data)
        
        if new_rows:
            print(f"    Filled {len(new_rows)} missing candles.")
            fill_df = pd.DataFrame(new_rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            fill_df['timestamp'] = pd.to_datetime(fill_df['timestamp'], unit='ms')
            
            df = pd.concat([df, fill_df])
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        
        return df.drop(columns=['diff'], errors='ignore')

    def fetch_funding_rates(self, symbol):
        """Fetch historical funding rate data using CCXT's fetch_funding_rate_history"""
        print(f"  Fetching Funding Rates for {symbol}...")
        all_funding = []
        since = self.start_ts
        limit = 1000 # Binance limit
        
        while True:
            try:
                rates = self.exchange.fetch_funding_rate_history(symbol, since=since, limit=limit)
                
                if not rates: break
                
                all_funding.extend(rates)
                
                last_ts = rates[-1]['timestamp']
                if last_ts == since: break
                
                since = last_ts + 1
                time.sleep(self.exchange.rateLimit / 1000)
                
                if len(rates) < limit: break
            except Exception as e:
                print(f"    Error fetching funding for {symbol}: {e}")
                break
                
        if not all_funding: return pd.DataFrame()
        
        df = pd.DataFrame(all_funding)
        df = df.drop_duplicates(subset=['timestamp'])
        # CCXT formats funding rate history consistently across exchanges
        df = df[['timestamp', 'fundingRate']].copy()
        df.rename(columns={'fundingRate': 'funding_rate'}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.sort_values('timestamp').reset_index(drop=True)

    def run(self, limit_coins=None):
        symbols = self.get_all_symbols()
        if not symbols: return

        if limit_coins:
            symbols = symbols[:limit_coins]
            print(f"Limiting to first {limit_coins} symbols...")
            
        print(f"Total symbols to process: {len(symbols)}")
        
        for i, symbol in enumerate(symbols):
            clean_name = self.get_clean_symbol(symbol)
            ohlcv_path = OHLCV_DIR / f"{clean_name}.parquet"
            funding_path = FUNDING_DIR / f"{clean_name}.parquet"
            
            print(f"[{i+1}/{len(symbols)}] Processing {symbol} ({clean_name})...")
            
            # --- OHLCV Sync & Gap Fill ---
            existing_ohlcv = None
            if ohlcv_path.exists():
                try:
                    existing_ohlcv = pd.read_parquet(ohlcv_path)
                    
                    if not existing_ohlcv.empty:
                        min_date = existing_ohlcv['timestamp'].min()
                        # Skip proactive history probing for existing files to save API weight
                        # Binance has much older data, but probing it sequentially for 500+ coins 
                        # triggers Rate Limit -1003. 
                        print("    Existing data found, skipping start date probe.")
                except:
                    print("    Corrupt parquet, re-fetching.")
                    existing_ohlcv = None
            
            # 1. Sync
            df_ohlcv = self.sync_ohlcv(symbol, existing_ohlcv)
            
            # 2. Gap Fill
            if df_ohlcv is not None and not df_ohlcv.empty:
                df_ohlcv = self.fill_gaps(symbol, df_ohlcv)
                df_ohlcv.to_parquet(ohlcv_path, index=False)
                print(f"    Saved OHLCV: {len(df_ohlcv)} rows ({df_ohlcv.timestamp.min()} -> {df_ohlcv.timestamp.max()})")
            
            # --- Funding ---
            if not funding_path.exists():
                df_funding = self.fetch_funding_rates(symbol)
                if not df_funding.empty:
                    df_funding.to_parquet(funding_path, index=False)
                    print(f"    Saved Funding ({len(df_funding)} rows)")
            
            print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binance Historical Data Fetcher")
    parser.add_argument("--limit", type=int, help="Limit number of coins to fetch")
    args = parser.parse_args()
    
    fetcher = BinanceFetcher()
    fetcher.run(limit_coins=args.limit)
