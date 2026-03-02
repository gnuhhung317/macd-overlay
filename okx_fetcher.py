import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone
from pathlib import Path
import os

class OKXFetcher:
    def __init__(self):
        self.exchange = ccxt.okx({
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'}
        })
        # OKX history for swaps generally starts around 2021
        self.start_ts = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        self.data_dir = Path("okx-data")
        self.ohlcv_dir = self.data_dir / "ohlcv"
        self.funding_dir = self.data_dir / "funding"
        
        for d in [self.ohlcv_dir, self.funding_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def find_start_date(self, symbol):
        """Find earliest available data timestamp (probing yearly backwards)"""
        current_year = datetime.now(timezone.utc).year
        start_year = 2020
        years = list(range(current_year, start_year - 1, -1))
        
        found_earliest = None
        for year in years:
            test_ts = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', since=test_ts, limit=1)
                if ohlcv:
                    found_earliest = ohlcv[0][0]
                else:
                    if found_earliest: return found_earliest
            except Exception:
                if found_earliest: return found_earliest
                time.sleep(0.1)
                continue
        return found_earliest

    def get_all_symbols(self):
        """Get all USDT-M swap symbols from OKX"""
        print("Fetching OKX markets...")
        try:
            markets = self.exchange.load_markets()
            symbols = [m['symbol'] for m in markets.values() 
                      if m['swap'] and m['linear'] and m['quote'] == 'USDT']
            return symbols
        except Exception as e:
            print(f"Error fetching symbols: {e}")
            return []

    def get_clean_symbol(self, symbol):
        """Standardize symbol to PairUSDT_USDT format"""
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
        all_ohlcv = []
        since = start_ts
        limit = 100
        
        while since < end_ts:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', since=since, limit=limit)
                if not ohlcv: break
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                if len(ohlcv) < limit: break
                time.sleep(self.exchange.rateLimit / 1000)
            except Exception as e:
                print(f"  Error fetching {symbol} at {since}: {e}")
                time.sleep(5)
        return all_ohlcv

    def sync_ohlcv(self, symbol, existing_df=None):
        current_time = int(time.time() * 1000)
        if existing_df is not None and not existing_df.empty:
            start_fetch_ts = existing_df['timestamp'].max().value // 10**6 + 1
        else:
            print(f"  Probing start date for {symbol}...")
            start_fetch_ts = self.find_start_date(symbol)
            if start_fetch_ts is None:
                start_fetch_ts = self.start_ts

        new_data = self.fetch_ohlcv_range(symbol, start_fetch_ts, current_time)
        if not new_data: return existing_df
        
        new_df = pd.DataFrame(new_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
        
        if existing_df is not None:
            return pd.concat([existing_df, new_df]).drop_duplicates('timestamp').sort_values('timestamp')
        return new_df

    def fetch_funding_rates(self, symbol):
        """OKX fetch_funding_rate_history support varies, using a simple fetch for current"""
        try:
            # fetch_funding_rate_history is not always available or consistent across exchanges in ccxt
            # We'll try it if available, otherwise return empty
            if self.exchange.has['fetchFundingRateHistory']:
                rates = self.exchange.fetch_funding_rate_history(symbol)
                if rates:
                    df = pd.DataFrame(rates)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    return df
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def run(self, limit_coins=None):
        symbols = self.get_all_symbols()
        if limit_coins:
            symbols = symbols[:limit_coins]
            
        for i, symbol in enumerate(symbols, 1):
            clean_name = self.get_clean_symbol(symbol)
            print(f"[{i}/{len(symbols)}] OKX: {symbol} -> {clean_name}...")
            
            # OHLCV
            file_path = self.ohlcv_dir / f"{clean_name}.parquet"
            existing_df = pd.read_parquet(file_path) if file_path.exists() else None
            df = self.sync_ohlcv(symbol, existing_df)
            if df is not None and not df.empty:
                df.to_parquet(file_path)
            
            # Funding (Optional/Best effort)
            funding_path = self.funding_dir / f"{clean_name}.parquet"
            f_df = self.fetch_funding_rates(symbol)
            if not f_df.empty:
                f_df.to_parquet(funding_path)
                
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit number of coins")
    args = parser.parse_args()
    
    fetcher = OKXFetcher()
    fetcher.run(limit_coins=args.limit)
