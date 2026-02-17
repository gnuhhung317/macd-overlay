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
DATA_ROOT = Path("bitget-data")
OHLCV_DIR = DATA_ROOT / "ohlcv"
FUNDING_DIR = DATA_ROOT / "funding"

# Create directories
OHLCV_DIR.mkdir(parents=True, exist_ok=True)
FUNDING_DIR.mkdir(parents=True, exist_ok=True)

class BitgetFetcher:
    def __init__(self):
        self.exchange = ccxt.bitget({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',  # USDT-M Perpetual
            }
        })
        self.start_ts = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

    def get_all_symbols(self):
        """Get all USDT-M perpetual symbols from Bitget"""
        print("Fetching markets...")
        try:
            markets = self.exchange.load_markets()
            symbols = [
                m['symbol'] for m in markets.values() 
                if m['linear'] and m['active'] and m['quote'] == 'USDT'
            ]
            return symbols
        except Exception as e:
            print(f"Error fetching markets: {e}")
            return []

    def get_clean_symbol(self, symbol):
        clean_symbol = symbol.replace('/', '').replace(':', '_')
        if not clean_symbol.endswith('_USDT'):
            if '_' in clean_symbol:
                base = clean_symbol.split('_')[0]
            else:
                base = clean_symbol.replace('USDT', '') + 'USDT'
            clean_symbol = f"{base}_USDT"
        return clean_symbol

    def find_start_date(self, symbol):
        """Find the earliest available data timestamp (probing yearly then monthly)"""
        current_year = datetime.now(timezone.utc).year
        # Probe up to next year to cover very recent listings or future dates issues
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
                    
                    # If we found data in year X, and X-1 was empty, 
                    # we must probe year X-1 monthly to find the exact start month.
                    # (Unless X is the very first year we checked)
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
                    
                    # If no monthly data found in previous year (or we didn't check), 
                    # then this year's start is likely the true start (or close enough).
                    # But we should also check if the data found in Jan 1st of `year` 
                    # is actually from Jan 1st.
                    # Usage of `since` ensures we get the first candle AFTER that date.
                    return first_found_ts
                else:
                    last_empty_year = year
            except Exception as e:
                # Treat validation errors (like symbol not found in that year) as empty
                last_empty_year = year
                time.sleep(0.5)
                continue

        # If probing failed, try without since (latest)
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=1)
            if ohlcv:
                # We have latest data but couldn't find history start.
                return None 
        except:
            pass
        return None

    def fetch_ohlcv_range(self, symbol, start_ts, end_ts):
        """Fetch OHLCV data for a specific range using forward pagination"""
        all_ohlcv = []
        since = start_ts
        limit = 200 # Safe limit
        
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
                # Fallback to hardcoded start if discovery fails
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
        
        # Find gaps > 1h + buffer (e.g. 1h 10m to be safe)
        gap_mask = df['diff'] > pd.Timedelta(minutes=90) # Standard is 60m
        gaps = df[gap_mask]
        
        if gaps.empty:
            # print(f"    No gaps found for {symbol}.")
            return df
            
        print(f"  {symbol}: Found {len(gaps)} gaps. Filling...")
        
        new_rows = []
        for idx, row in gaps.iterrows():
            gap_end = row['timestamp']
            # Previous row timestamp is gap start
            prev_idx = df.index.get_loc(idx) - 1
            gap_start = df.iloc[prev_idx]['timestamp']
            
            start_ms = int(gap_start.timestamp() * 1000)
            end_ms = int(gap_end.timestamp() * 1000)
            
            # Don't fill if gap is massive (might be delisted period or maintenance?)
            # But we try anyway.
            print(f"    Filling gap: {gap_start} -> {gap_end}")
            
            # Fetch specifically this range
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
        """Fetch historical funding rate data with simple pagination"""
        # (Preserving previous successful logic for Funding)
        print(f"  Fetching Funding Rates for {symbol}...")
        all_funding = []
        page_no = 1
        page_size = 100
        since_ms = self.start_ts
        
        while True:
            try:
                params = {'pageNo': page_no, 'pageSize': page_size}
                rates = self.exchange.fetch_funding_rate_history(symbol, since=None, limit=page_size, params=params)
                
                if not rates: break
                
                newest = max(r['timestamp'] for r in rates)
                oldest = min(r['timestamp'] for r in rates)
                
                if newest < since_ms: break
                valid = [r for r in rates if r['timestamp'] >= since_ms]
                all_funding.extend(valid)
                
                if oldest <= since_ms: break
                if len(rates) < page_size: break
                
                page_no += 1
                time.sleep(self.exchange.rateLimit / 1000)
                if page_no > 500: break
            except Exception as e:
                # print(f"    Error funding {symbol}: {e}")
                break
                
        if not all_funding: return pd.DataFrame()
        df = pd.DataFrame(all_funding)
        df = df.drop_duplicates(subset=['timestamp'])
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
                    
                    # VALIDATION: Check if existing data is suspiciously recent (likely result of previous bug)
                    if not existing_ohlcv.empty:
                        min_date = existing_ohlcv['timestamp'].min()
                        # If data starts after Jan 1, 2025, it's suspicious for most coins. 
                        # We'll probe for earlier data.
                        if min_date.year >= 2025:
                            print(f"    Existing data starts late ({min_date}), checking for earlier history...")
                            found_start_ts = self.find_start_date(symbol)
                            if found_start_ts:
                                found_start_dt = datetime.fromtimestamp(found_start_ts/1000, tz=timezone.utc)
                                # If we found data > 90 days older than current start, assume file is incomplete
                                current_start_ms = existing_ohlcv['timestamp'].min().timestamp() * 1000
                                if found_start_ts < current_start_ms - (90 * 24 * 3600 * 1000):
                                    print(f"    Found older data starting {found_start_dt}. Discarding incomplete file.")
                                    existing_ohlcv = None
                                else:
                                    print("    Existing start seems correct.")
                            else:
                                print("    No earlier data found.")
                                
                except:
                    print("    Corrupt parquet, re-fetching.")
                    existing_ohlcv = None
            
            # 1. Sync (Update or Initial)
            df_ohlcv = self.sync_ohlcv(symbol, existing_ohlcv)
            
            # 2. Gap Fill
            if df_ohlcv is not None and not df_ohlcv.empty:
                df_ohlcv = self.fill_gaps(symbol, df_ohlcv)
                
                # Save
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
    parser = argparse.ArgumentParser(description="Bitget Historical Data Fetcher")
    parser.add_argument("--limit", type=int, help="Limit number of coins to fetch")
    args = parser.parse_args()
    
    fetcher = BitgetFetcher()
    fetcher.run(limit_coins=args.limit)
