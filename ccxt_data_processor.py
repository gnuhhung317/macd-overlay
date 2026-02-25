import ccxt
import pandas as pd
from datetime import datetime, timezone
import time

try:
    from data_processor import BinanceDataProcessor
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_processor import BinanceDataProcessor

class CCXTDataProcessor(BinanceDataProcessor):
    def __init__(self, exchange_id, api_key="", api_secret="", password="", fast_period=12, slow_period=26, signal_period=9, use_futures=True):
        # We do NOT call super().__init__() because we don't want to init binance.Client
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.use_futures = use_futures
        
        # Init CCXT Exchange
        exchange_class = getattr(ccxt, exchange_id.lower())
        
        exchange_args = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap' if use_futures else 'spot'
            }
        }
        
        if password:
            exchange_args['password'] = password
            
        self.client = exchange_class(exchange_args)
        
        try:
            self.client.load_markets()
        except Exception as e:
            print(f"⚠️ Error loading markets for {exchange_id}: {e}")
            
    def _get_ccxt_symbol(self, symbol: str) -> str:
        if not hasattr(self.client, 'symbols') or not self.client.symbols:
            return symbol.replace("USDT", "/USDT:USDT") if self.use_futures else symbol.replace("USDT", "/USDT")
            
        for s in self.client.symbols:
            if s.replace('/', '').replace(':', '') == symbol or s.replace('/', '').split(':')[0] == symbol:
                if self.use_futures and self.client.markets[s]['swap']:
                    return s
                elif not self.use_futures and self.client.markets[s]['spot']:
                    return s
        
        return symbol.replace("USDT", "/USDT:USDT") if self.use_futures else symbol.replace("USDT", "/USDT")

    def _parse_time_string(self, time_str):
        # basic parser for "30 days ago UTC"
        if "days ago" in time_str:
            days = int(time_str.split()[0])
            ms = int(time.time() * 1000) - (days * 24 * 60 * 60 * 1000)
            return ms
        elif "now" in time_str:
            return int(time.time() * 1000)
        return None

    def get_historical_data(self, symbol='BTCUSDT', interval='1h', start_date=None, end_date=None):
        if not self.client.has['fetchOHLCV']:
            print(f"✗ {self.client.id} doesn't support fetchOHLCV")
            return pd.DataFrame()
        
        # Format symbol for CCXT (e.g. BTC/USDT:USDT for futures on some exchanges, but CCXT often handles BTC/USDT)
        ccxt_symbol = self._get_ccxt_symbol(symbol)
        
        # Fallback mapping for unsupported timeframes
        fallback_map = {
            'bitget': {
                '8h': '4h'
            }
        }
        
        target_interval = interval
        fetch_interval = interval
        resample_needed = False
        
        if self.client.id in fallback_map and interval in fallback_map[self.client.id]:
            fetch_interval = fallback_map[self.client.id][interval]
            resample_needed = True
        
        since = None
        if start_date:
            since = self._parse_time_string(start_date)
            
        limit = 1000
        all_klines = []
        batch_count = 0
        
        current_since = since
        
        while True:
            batch_count += 1
            max_retries = 3
            success = False
            klines = []
            
            for attempt in range(max_retries):
                try:
                    # fetchohlcv returns [timestamp, open, high, low, close, volume]
                    klines = self.client.fetch_ohlcv(ccxt_symbol, timeframe=fetch_interval, since=current_since, limit=limit)
                    # print(klines)
                    success = True
                    break
                except Exception as e:
                    print(f"⚠️ CCXT Error fetching data: {e} | Attempt {attempt+1}")
                    time.sleep(2)
                    
            if not success or not klines:
                break
                
            all_klines.extend(klines)
            if len(klines) < limit:
                break
                
            current_since = klines[-1][0] + 1
            
        if not all_klines:
            return pd.DataFrame()
            
        unique_klines = []
        seen_timestamps = set()
        for kline in all_klines:
            ts = kline[0]
            if ts not in seen_timestamps:
                seen_timestamps.add(ts)
                # Pad for binance format compatibility
                padded = kline + [kline[0], 0, 0, 0, 0, 0] # timestamp, open, high, low, close, vol, (close_time, qv, trades, tbvb, tbqv, ignore)
                unique_klines.append(padded)
                
        df = pd.DataFrame(unique_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        if resample_needed and not df.empty:
            df = self._resample_ohlcv(df, target_interval)
            
        return df

    def _resample_ohlcv(self, df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
        """Resamples OHLCV dataframe to a higher timeframe"""
        freq = target_interval.replace('m', 'min').replace('d', 'D')
        
        df_resampled = df.set_index('timestamp').resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'close_time': 'last',
            'quote_volume': 'sum',
            'trades': 'sum',
            'taker_buy_base': 'sum',
            'taker_buy_quote': 'sum',
            'ignore': 'last'
        }).dropna().reset_index()
        
        return df_resampled

    def get_current_funding_rate(self, symbol):
        if not self.use_futures:
            return 0.0
        
        ccxt_symbol = self._get_ccxt_symbol(symbol)
        try:
            if getattr(self.client, 'has', {}).get('fetchFundingRate'):
                rate = self.client.fetch_funding_rate(ccxt_symbol)
                return float(rate.get('fundingRate', 0.0))
            return 0.0
        except Exception as e:
            print(f"Error fetching funding rate for {symbol}: {e}")
            return 0.0

    def get_top_symbols(self, limit: int = 0, min_volume: float = 0) -> list:
        try:
            tickers = self.client.fetch_tickers()
            symbols = []
            for ccxt_sym, t in tickers.items():
                if '/USDT' not in ccxt_sym:
                    continue
                # depending on CCXT parsing, use baseVolume or quoteVolume. 'quoteVolume' is usually in USDT
                vol = t.get('quoteVolume', 0)
                if vol is None:
                    continue
                if vol < min_volume:
                    continue
                raw_symbol = ccxt_sym.split(':')[0].replace("/", "") # Handle BTC/USDT:USDT -> BTCUSDT
                symbols.append((raw_symbol, float(vol)))
                
            symbols.sort(key=lambda x: x[1], reverse=True)
            result = [s[0] for s in symbols]
            if limit > 0:
                return result[:limit]
            return result
        except Exception as e:
            print(f"Error fetching top symbols via CCXT: {e}")
            return []
