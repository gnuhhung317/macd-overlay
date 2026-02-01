import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta
import time


class BinanceDataProcessor:
    """
    Module xử lý dữ liệu từ Binance và tính toán MACD
    """
    
    def __init__(self, api_key="", api_secret="", fast_period=12, slow_period=26, signal_period=9, use_futures=True):
        """
        Khởi tạo data processor
        
        Args:
            api_key (str): Binance API key (để trống cho public data)
            api_secret (str): Binance API secret (để trống cho public data)
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            signal_period (int): Signal SMA period
            use_futures (bool): Sử dụng Futures API (True) hoặc Spot API (False)
        """
        self.client = Client(api_key, api_secret)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.use_futures = use_futures
        
    def _get_interval_ms(self, interval):
        """
        Chuyển interval string sang milliseconds
        
        Args:
            interval (str): Interval (1m, 5m, 15m, 30m, 1h, 4h, 1d, etc.)
            
        Returns:
            int: Milliseconds
        """
        interval_map = {
            '1m': 60 * 1000,
            '3m': 3 * 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '8h': 8 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '3d': 3 * 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
            '1M': 30 * 24 * 60 * 60 * 1000,
        }
        return interval_map.get(interval, 60 * 60 * 1000)  # default 1h
    
    def get_historical_data(self, symbol='BTCUSDT', interval='1h', start_date=None, end_date=None):
        """
        Lấy dữ liệu lịch sử từ Binance với batch processing
        Tự động fetch nhiều lần nếu vượt quá giới hạn 1500 nến/request
        
        Args:
            symbol (str): Trading pair (e.g., BTCUSDT)
            interval (str): Kline interval (1m, 5m, 15m, 30m, 1h, 4h, 1d)
            start_date (str): Start date (e.g., '2024-01-01', '1 year ago UTC')
            end_date (str): End date (e.g., '2024-12-31', 'now UTC') - None = now
            
        Returns:
            pd.DataFrame: DataFrame chứa dữ liệu OHLCV
        """
        # Mặc định: 30 ngày gần nhất nếu không có tham số
        if start_date is None:
            start_date = '30 days ago UTC'
        if end_date is None:
            end_date = 'now UTC'
            
        # print(f"Đang lấy dữ liệu {symbol} với khung thời gian {interval}...")
        # print(f"Từ: {start_date} → Đến: {end_date}")
        
        all_klines = []
        batch_count = 0
        limit = 1000  # Binance actual limit
        
        # Fetch data in batches
        current_start = start_date
        while True:
            batch_count += 1
            # print(f"  Batch {batch_count}: Đang tải...", end='', flush=True)
            
            try:
                # Sử dụng Futures hoặc Spot API
                if self.use_futures:
                    klines = self.client.futures_historical_klines(
                        symbol, 
                        interval, 
                        current_start,
                        end_date,
                        limit=limit
                    )
                else:
                    klines = self.client.get_historical_klines(
                        symbol, 
                        interval, 
                        current_start,
                        end_date,
                        limit=limit
                    )
            except Exception as e:
                print(f" Lỗi: {e}")
                break
            
            if not klines:
                print(" Không có dữ liệu")
                break
            
            # print(f" {len(klines)} nến")
            all_klines.extend(klines)
            
            # Nếu số nến < limit, đã hết dữ liệu trong khoảng thời gian
            if len(klines) < limit:
                # print(f"      (Đã lấy hết dữ liệu - nhận được {len(klines)} < {limit} nến)")
                break
            
            # Cập nhật start_date cho batch tiếp theo
            # Lấy timestamp của nến cuối cùng + 1ms (để tránh duplicate)
            last_timestamp = klines[-1][0]  # timestamp của nến cuối (ms)
            
            # Convert sang datetime để kiểm tra
            last_dt = pd.to_datetime(last_timestamp, unit='ms')
            # print(f"      (Timestamp cuối: {last_dt})")
            
            # Start của batch tiếp theo = timestamp cuối + 1ms
            current_start = last_timestamp + 1
            
            
        
        if not all_klines:
            print("✗ Không lấy được dữ liệu")
            return pd.DataFrame()
        
        # Remove duplicates (có thể có overlap giữa các batch)
        unique_klines = []
        seen_timestamps = set()
        
        for kline in all_klines:
            ts = kline[0]
            if ts not in seen_timestamps:
                seen_timestamps.add(ts)
                unique_klines.append(kline)
        
        # print(f"\n  Tổng dữ liệu gốc: {len(all_klines)} nến")
        # print(f"  Sau khi loại bỏ duplicate: {len(unique_klines)} nến")
        
        # Chuyển đổi sang DataFrame
        df = pd.DataFrame(unique_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Chuyển đổi kiểu dữ liệu
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # print(f"\n✓ Đã lấy tổng cộng {len(df)} nến ({batch_count} batch)")
        # print(f"  Từ: {df['timestamp'].iloc[0]}")
        # print(f"  Đến: {df['timestamp'].iloc[-1]}")
        
        return df
    
    def calculate_ema(self, data, period):
        """
        Tính EMA (Exponential Moving Average)
        
        Args:
            data (pd.Series): Dữ liệu input
            period (int): EMA period
            
        Returns:
            pd.Series: EMA values
        """
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_sma(self, data, period):
        """
        Tính SMA (Simple Moving Average)
        
        Args:
            data (pd.Series): Dữ liệu input
            period (int): SMA period
            
        Returns:
            pd.Series: SMA values
        """
        return data.rolling(window=period).mean()
    
    def calculate_macd(self, df):
        """
        Tính MACD và Signal line theo Pine Script logic
        
        Pine Script code:
            macd = ema(close, sa-fa)  // ema(close, 14) với fast=12, slow=26
            signal = sma(macd, sig)
        
        Args:
            df (pd.DataFrame): DataFrame chứa dữ liệu giá
            
        Returns:
            pd.DataFrame: DataFrame với MACD indicators
        """
        # Standard MACD: EMA_fast - EMA_slow
        # Formula matches ml/data_pipeline.py used for training
        ema_fast = self.calculate_ema(df['close'], self.fast_period)
        ema_slow = self.calculate_ema(df['close'], self.slow_period)
        df['macd'] = ema_fast - ema_slow
        
        # Signal line = EMA(MACD) (Standard uses EMA, not SMA)
        df['signal'] = self.calculate_ema(df['macd'], self.signal_period)
        
        # Histogram = MACD - Signal
        df['histogram'] = df['macd'] - df['signal']
        
        return df
    
    def calculate_rsi(self, data, period=14):
        """
        Tính RSI (Relative Strength Index)
        
        Args:
            data (pd.Series): Price data
            period (int): RSI period
            
        Returns:
            pd.Series: RSI values
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_bollinger_bands(self, data, period=20, std_dev=2):
        """
        Tính Bollinger Bands
        
        Args:
            data (pd.Series): Price data
            period (int): BB period
            std_dev (float): Standard deviation multiplier
            
        Returns:
            tuple: (upper, middle, lower, width)
        """
        middle = self.calculate_sma(data, period)
        std = data.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        width = (upper - lower) / middle  # Normalized width
        
        return upper, middle, lower, width
    
    def calculate_atr(self, df, period=14):
        """
        Tính ATR (Average True Range)
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC data
            period (int): ATR period
            
        Returns:
            pd.Series: ATR values
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR = EMA of TR
        atr = tr.ewm(span=period, adjust=False).mean()
        
        return atr
    
    def add_indicators(self, df, ema_period=200, rsi_period=14, bb_period=20, atr_period=14):
        """
        Thêm tất cả các indicators cần thiết
        
        Args:
            df (pd.DataFrame): DataFrame với MACD data
            ema_period (int): EMA period for trend filter
            rsi_period (int): RSI period
            bb_period (int): Bollinger Bands period
            atr_period (int): ATR period
            
        Returns:
            pd.DataFrame: DataFrame with all indicators
        """
        # EMA 200 for trend
        df['ema_200'] = self.calculate_ema(df['close'], ema_period)
        
        # RSI
        df['rsi'] = self.calculate_rsi(df['close'], rsi_period)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'], df['bb_width'] = \
            self.calculate_bollinger_bands(df['close'], bb_period)
        
        # ATR
        df['atr'] = self.calculate_atr(df, atr_period)
        
        return df
    
    def detect_crossovers(self, df):
        """
        Xác định các điểm giao cắt MACD
        
        Args:
            df (pd.DataFrame): DataFrame với MACD data
            
        Returns:
            list: Danh sách các crossover points
        """
        crossovers = []
        
        for i in range(1, len(df)):
            prev_macd = df['macd'].iloc[i-1]
            prev_signal = df['signal'].iloc[i-1]
            curr_macd = df['macd'].iloc[i]
            curr_signal = df['signal'].iloc[i]
            
            # Kiểm tra NaN
            if pd.isna(prev_macd) or pd.isna(prev_signal) or pd.isna(curr_macd) or pd.isna(curr_signal):
                continue
            
            # Bullish crossover: MACD cắt lên trên Signal
            if prev_macd <= prev_signal and curr_macd > curr_signal:
                crossovers.append({
                    'timestamp': df['timestamp'].iloc[i],
                    'type': 'BULLISH',
                    'price': df['close'].iloc[i],
                    'macd': curr_macd,
                    'signal': curr_signal,
                    'histogram': curr_macd - curr_signal,
                    'index': i
                })
            
            # Bearish crossover: MACD cắt xuống dưới Signal
            elif prev_macd >= prev_signal and curr_macd < curr_signal:
                crossovers.append({
                    'timestamp': df['timestamp'].iloc[i],
                    'type': 'BEARISH',
                    'price': df['close'].iloc[i],
                    'macd': curr_macd,
                    'signal': curr_signal,
                    'histogram': curr_macd - curr_signal,
                    'index': i
                })
        
        return crossovers
    
    def save_to_csv(self, df, filename):
        """
        Lưu DataFrame ra file CSV
        
        Args:
            df (pd.DataFrame): DataFrame cần lưu
            filename (str): Tên file
        """
        df.to_csv(filename, index=False)
        print(f"✓ Đã lưu dữ liệu vào {filename}")
    
    def get_latest_crossover(self, df):
        """
        Lấy crossover mới nhất
        
        Args:
            df (pd.DataFrame): DataFrame với MACD data
            
        Returns:
            dict or None: Crossover mới nhất hoặc None
        """
        crossovers = self.detect_crossovers(df)
        return crossovers[-1] if crossovers else None
    
    def analyze_crossovers(self, crossovers):
        """
        Phân tích thống kê các crossovers
        
        Args:
            crossovers (list): Danh sách crossovers
            
        Returns:
            dict: Thống kê
        """
        if not crossovers:
            return {
                'total': 0,
                'bullish': 0,
                'bearish': 0,
                'avg_interval_hours': 0
            }
        
        bullish_count = sum(1 for c in crossovers if c['type'] == 'BULLISH')
        bearish_count = sum(1 for c in crossovers if c['type'] == 'BEARISH')
        
        # Tính khoảng cách trung bình giữa các crossovers
        if len(crossovers) > 1:
            intervals = []
            for i in range(1, len(crossovers)):
                delta = crossovers[i]['timestamp'] - crossovers[i-1]['timestamp']
                intervals.append(delta.total_seconds() / 3600)  # Convert to hours
            avg_interval = sum(intervals) / len(intervals)
        else:
            avg_interval = 0
        
    def get_current_funding_rate(self, symbol):
        """
        Lấy funding rate hiện tại cho symbol
        
        Args:
            symbol (str): Trading pair (e.g., BTCUSDT)
            
        Returns:
            float: Funding rate (e.g., 0.0001) or 0.0 if failed
        """
        if not self.use_futures:
            return 0.0
            
        try:
            # Get funding rate (returns list of funding rates, we need latest or current)
            # client.futures_premium_index returns dict with lastFundingRate
            info = self.client.futures_premium_index(symbol=symbol)
            if isinstance(info, dict) and 'lastFundingRate' in info:
                return float(info['lastFundingRate'])
            return 0.0
        except Exception as e:
            print(f"Error fetching funding rate for {symbol}: {e}")
            return 0.0
