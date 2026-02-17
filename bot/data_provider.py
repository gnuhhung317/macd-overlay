import pandas as pd
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from .config import BotConfig

# Assuming we reuse the tested BinanceDataProcessor logic
# In a real scenario, this should be adaptable or use ccxt for multi-exchange support
try:
    from data_processor import BinanceDataProcessor
except ImportError:
    import sys
    sys.path.append('..') # Add parent directory to path
    from data_processor import BinanceDataProcessor

class DataProvider:
    def __init__(self, config: BotConfig):
        self.config = config
        self.processor = BinanceDataProcessor()
        self.interval_map = {
            "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
            "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "8h": 28800,
            "12h": 43200, "1d": 86400
        }

    def fetch_closed_candles(self, symbol: str, interval: str, lookback_days: int = 5) -> pd.DataFrame:
        """
        Fetch OHLCV data guaranteed to be CLOSED candles only.
        Crucial for avoiding repaint.
        """
        
        # Calculate start time
        start_str = f"{lookback_days} days ago UTC"
        
        # Fetch from processor (REST API)
        df = self.processor.get_historical_data(symbol, interval, start_str, "now UTC")
        
        if df.empty:
            return pd.DataFrame()

        # VALIDATION: Check the last candle's timestamp
        last_ts = df.iloc[-1]['timestamp']
        
        # Ensure last_ts is timezone-aware (UTC)
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
            
        now_ts = datetime.now(timezone.utc)
        
        # Calculate when the last candle SHOULD have closed
        # e.g. for 4h candle starting at 12:00, it closes at 16:00.
        # If current time is 15:59, we must drop it.
        
        interval_seconds = self.interval_map.get(interval, 3600)
        
        # The 'timestamp' column usually denotes the OPEN time of the candle
        # So candle is closed if: current_time >= candle_open_time + interval
        candle_close_time = last_ts + timedelta(seconds=interval_seconds)
        
        if now_ts < candle_close_time:
            # Drop the last candle as it is still forming
            df = df.iloc[:-1]
            
        return df

    def get_current_price(self, symbol: str) -> float:
        """Get realtime price (ticker)"""
        try:
            ticker = self.processor.client.futures_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            print(f"Error fetching price for {symbol}: {e}")
            return 0.0

    def get_funding_rate(self, symbol: str) -> float:
        """Get current funding rate"""
        try:
            # client.futures_funding_rate returns list, we want the latest
            funding = self.processor.client.futures_funding_rate(symbol=symbol, limit=1)
            if funding:
                return float(funding[-1]['fundingRate'])
            return 0.0
        except Exception as e:
            # Fallback or silent error
            return 0.0

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD and other indicators needed for ML"""
        if df.empty:
            return df
            
        # Use the robust calculation from processor
        df = self.processor.calculate_macd(df)
        return df

    def get_top_symbols(self, limit: int = 0, min_volume: float = 0) -> list:
        """Get top symbols from processor"""
        return self.processor.get_top_symbols(limit, min_volume)
