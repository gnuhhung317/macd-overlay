"""
Script monitor real-time
Chạy liên tục để theo dõi crossovers và gửi thông báo
"""

import time
from datetime import datetime
from data_processor import BinanceDataProcessor
from telegram_notifier import TelegramNotifier
import sys
import os
from zoneinfo import ZoneInfo

# ML imports
try:
    import pandas as pd
    import numpy as np
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


def print_separator(char='=', length=70):
    """In dòng phân cách"""
    print(char * length)


class MLPredictor:
    """Lightweight ML predictor for realtime monitor"""
    
    def __init__(self, models_dir='ml/models'):
        self.models_dir = models_dir
        self.entry_filter = None
        self.sl_predictor = None
        self.tp_predictor = None
        self.bars_predictor = None
        self.feature_columns = None
        self.loaded = False
    
    def load_models(self):
        """Load all trained models"""
        if not ML_AVAILABLE:
            print("⚠️ ML libraries not available")
            return False
        
        try:
            entry_path = os.path.join(self.models_dir, 'entry_filter.joblib')
            sl_path = os.path.join(self.models_dir, 'sl_predictor.joblib')
            tp_path = os.path.join(self.models_dir, 'tp_predictor.joblib')
            bars_path = os.path.join(self.models_dir, 'bars_predictor.joblib')
            
            if not all(os.path.exists(p) for p in [entry_path, sl_path, tp_path, bars_path]):
                print("⚠️ ML models not found. Run training scripts first.")
                return False
            
            print("📦 Loading ML models...")
            self.entry_filter = joblib.load(entry_path)
            self.sl_predictor = joblib.load(sl_path)
            self.tp_predictor = joblib.load(tp_path)
            self.bars_predictor = joblib.load(bars_path)
            
            # Get feature columns from entry filter (all models use same features)
            if hasattr(self.entry_filter, 'feature_names_in_'):
                self.feature_columns = list(self.entry_filter.feature_names_in_)
            else:
                # Default feature list
                self.feature_columns = self._get_default_features()
            
            self.loaded = True
            print(f"✅ ML models loaded ({len(self.feature_columns)} features)")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def _get_default_features(self):
        """Get default feature columns"""
        return [
            'crossover_type', 'macd', 'signal', 'histogram', 'histogram_slope',
            'macd_above_zero', 'histogram_positive', 'macd_change', 'signal_change',
            'histogram_change', 'macd_acceleration', 'histogram_acceleration',
            'close', 'volume', 'returns', 'volatility', 'atr', 'atr_pct',
            'rsi', 'rsi_oversold', 'rsi_overbought', 'bb_position', 'bb_width',
            'bb_squeeze', 'sma_20', 'sma_50', 'ema_20', 'price_vs_sma20',
            'price_vs_sma50', 'sma20_vs_sma50', 'adx', 'plus_di', 'minus_di',
            'trend_strength', 'trend_direction', 'obv', 'obv_sma', 'obv_signal',
            'mfi', 'cmf', 'high_low_range', 'body_size', 'upper_shadow', 
            'lower_shadow', 'candle_type', 'consecutive_green', 'consecutive_red',
            'volume_sma', 'volume_ratio', 'volume_trend'
        ]
    
    def calculate_features(self, df):
        """Calculate features from OHLCV dataframe for the last row"""
        if not ML_AVAILABLE or len(df) < 50:
            return None
        
        try:
            # Make a copy
            df = df.copy()
            
            # Basic price features
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(20).std()
            df['high_low_range'] = (df['high'] - df['low']) / df['close']
            
            # ATR
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift(1)).abs()
            low_close = (df['low'] - df['close'].shift(1)).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = tr.rolling(14).mean()
            df['atr_pct'] = df['atr'] / df['close']
            
            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            df['rsi'] = 100 - (100 / (1 + rs))
            df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
            df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
            
            # Bollinger Bands
            df['sma_20'] = df['close'].rolling(20).mean()
            df['bb_std'] = df['close'].rolling(20).std()
            df['bb_upper'] = df['sma_20'] + 2 * df['bb_std']
            df['bb_lower'] = df['sma_20'] - 2 * df['bb_std']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma_20']
            df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean()).astype(int)
            
            # Moving averages
            df['sma_50'] = df['close'].rolling(50).mean()
            df['ema_20'] = df['close'].ewm(span=20).mean()
            df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
            df['price_vs_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
            df['sma20_vs_sma50'] = (df['sma_20'] - df['sma_50']) / df['sma_50']
            
            # ADX
            plus_dm = df['high'].diff()
            minus_dm = df['low'].diff().mul(-1)
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
            
            atr14 = tr.rolling(14).mean()
            df['plus_di'] = 100 * (plus_dm.rolling(14).mean() / (atr14 + 1e-10))
            df['minus_di'] = 100 * (minus_dm.rolling(14).mean() / (atr14 + 1e-10))
            dx = 100 * (df['plus_di'] - df['minus_di']).abs() / (df['plus_di'] + df['minus_di'] + 1e-10)
            df['adx'] = dx.rolling(14).mean()
            df['trend_strength'] = (df['adx'] > 25).astype(int)
            df['trend_direction'] = (df['plus_di'] > df['minus_di']).astype(int)
            
            # Volume features
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-10)
            df['volume_trend'] = (df['volume'] > df['volume'].rolling(10).mean()).astype(int)
            
            # OBV
            obv = (df['volume'] * np.sign(df['close'].diff())).fillna(0).cumsum()
            df['obv'] = obv
            df['obv_sma'] = obv.rolling(20).mean()
            df['obv_signal'] = (obv > df['obv_sma']).astype(int)
            
            # MFI
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            mf = typical_price * df['volume']
            positive_mf = mf.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
            negative_mf = mf.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
            df['mfi'] = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-10)))
            
            # CMF
            mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
            mf_volume = mf_multiplier * df['volume']
            df['cmf'] = mf_volume.rolling(20).sum() / (df['volume'].rolling(20).sum() + 1e-10)
            
            # Candle patterns
            df['body_size'] = (df['close'] - df['open']).abs() / (df['high'] - df['low'] + 1e-10)
            df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-10)
            df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-10)
            df['candle_type'] = (df['close'] > df['open']).astype(int)
            
            # Consecutive candles
            green = (df['close'] > df['open']).astype(int)
            red = (df['close'] < df['open']).astype(int)
            df['consecutive_green'] = green.groupby((green != green.shift()).cumsum()).cumsum() * green
            df['consecutive_red'] = red.groupby((red != red.shift()).cumsum()).cumsum() * red
            
            # MACD features
            df['histogram_slope'] = df['histogram'].diff()
            df['macd_above_zero'] = (df['macd'] > 0).astype(int)
            df['histogram_positive'] = (df['histogram'] > 0).astype(int)
            df['macd_change'] = df['macd'].diff()
            df['signal_change'] = df['signal'].diff()
            df['histogram_change'] = df['histogram'].diff()
            df['macd_acceleration'] = df['macd_change'].diff()
            df['histogram_acceleration'] = df['histogram_change'].diff()
            
            # Crossover type (1 for bullish, 0 for bearish)
            df['crossover_type'] = (df['macd'] > df['signal']).astype(int)
            
            return df
            
        except Exception as e:
            print(f"⚠️ Feature calculation error: {e}")
            return None
    
    def predict(self, features_df):
        """Make predictions for the last row of features"""
        if not self.loaded or features_df is None:
            return None
        
        try:
            # Get last row
            last_row = features_df.iloc[[-1]]
            
            # Select only the features used by models
            available_features = [f for f in self.feature_columns if f in last_row.columns]
            missing_features = [f for f in self.feature_columns if f not in last_row.columns]
            
            if missing_features:
                # Fill missing features with 0
                for f in missing_features:
                    last_row[f] = 0
            
            X = last_row[self.feature_columns].fillna(0)
            
            # Predictions
            entry_prob = self.entry_filter.predict_proba(X)[0][1]
            sl_pct = self.sl_predictor.predict(X)[0]
            tp_pct = self.tp_predictor.predict(X)[0]
            bars = self.bars_predictor.predict(X)[0]
            
            # Ensure reasonable values
            sl_pct = max(0.005, min(0.20, sl_pct))  # 0.5% to 20%
            tp_pct = max(0.005, tp_pct)  # At least 0.5%
            bars = max(1, min(50, int(round(bars))))
            
            rr = tp_pct / sl_pct if sl_pct > 0 else 2.0
            
            return {
                'entry_confidence': entry_prob,
                'sl_pct': sl_pct,
                'tp_pct': tp_pct,
                'bars_to_peak': bars,
                'risk_reward': rr
            }
            
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            return None


def monitor_realtime(symbol='BTCUSDT', interval='1h', check_interval=60, 
                     telegram_token=None, telegram_chat_id=None, quiet=False,
                     use_ml=True, ml_threshold=0.5):
    """
    Monitor real-time và gửi thông báo khi có crossover
    
    Args:
        symbol (str): Trading pair
        interval (str): Time interval
        check_interval (int): Số giây giữa mỗi lần check
        telegram_token (str): Telegram bot token
        telegram_chat_id (str): Telegram chat ID
        quiet (bool): Chỉ hiển thị crossover alerts, ẩn status updates
        use_ml (bool): Sử dụng ML để filter và predict
        ml_threshold (float): Ngưỡng confidence tối thiểu (0-1)
    """
    # Calculate lookback period based on interval (need enough data for MACD calculation)
    # MACD needs at least slow_period (26) + signal_period (9) = 35 candles minimum
    # Add extra buffer for better accuracy
    interval_to_lookback = {
        '1m': '6 hours ago UTC',
        '3m': '12 hours ago UTC', 
        '5m': '1 day ago UTC',
        '15m': '2 days ago UTC',
        '30m': '3 days ago UTC',
        '1h': '5 days ago UTC',
        '2h': '10 days ago UTC',
        '4h': '20 days ago UTC',
        '6h': '30 days ago UTC',
        '8h': '40 days ago UTC',
        '12h': '60 days ago UTC',
        '1d': '90 days ago UTC',  # 90 days for daily chart
        '3d': '270 days ago UTC',
        '1w': '1 year ago UTC',
    }
    lookback_period = interval_to_lookback.get(interval, '5 days ago UTC')
    
    if not quiet:
        print_separator()
        print("BẮT ĐẦU MONITOR REAL-TIME")
        print_separator()
        print(f"Symbol:          {symbol}")
        print(f"Interval:        {interval}")
        print(f"Lookback:        {lookback_period}")
        print(f"Check every:     {check_interval} seconds")
        print(f"Telegram:        {'Enabled ✓' if telegram_token else 'Disabled ✗'}")
        print(f"ML Filter:       {'Enabled ✓' if use_ml else 'Disabled ✗'}")
        if use_ml:
            print(f"ML Threshold:    {ml_threshold:.0%}")
        print("Nhấn Ctrl+C để dừng...")
        print_separator()
        print()
    
    # Khởi tạo processor
    processor = BinanceDataProcessor()
    
    # Khởi tạo ML predictor
    ml_predictor = None
    if use_ml:
        ml_predictor = MLPredictor()
        if not ml_predictor.load_models():
            print("⚠️ ML disabled - models not loaded")
            ml_predictor = None
        print()
    
    # Khởi tạo Telegram (nếu có)
    telegram = None
    if telegram_token and telegram_chat_id:
        telegram = TelegramNotifier(
            telegram_token, 
            telegram_chat_id,
            ml_system=ml_predictor,
            entry_threshold=ml_threshold
        )
        if not quiet:
            print("🔔 Test Telegram...")
        if telegram.test_connection():
            if not quiet:
                print("✓ Telegram sẵn sàng!")
        else:
            if not quiet:
                print("⚠️  Telegram không hoạt động, chỉ hiển thị console")
            telegram = None
        if not quiet:
            print()
    
    last_crossover_time = None
    check_count = 0
    
    try:
        while True:
            check_count += 1
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            try:
                # Lấy dữ liệu gần đây với lookback period phù hợp với interval
                df = processor.get_historical_data(symbol, interval, lookback_period, 'now UTC')
                
                # Skip if not enough data for MACD calculation
                if len(df) < 50:  # Need at least 50 candles for reliable MACD
                    if not quiet:
                        print(f"[{current_time}] ⚠️  Không đủ dữ liệu: {len(df)} nến (cần ít nhất 50)")
                    time.sleep(check_interval)
                    continue
                    
                df = processor.calculate_macd(df)
                
                # Kiểm tra crossover ở các nến gần nhất
                recent_crossovers = processor.detect_crossovers(df.tail(20))
                
                if recent_crossovers:
                    latest_cross = recent_crossovers[-1]
                    
                    # Chỉ thông báo nếu là crossover mới
                    if last_crossover_time != latest_cross['timestamp']:
                        emoji = "🟢" if latest_cross['type'] == 'BULLISH' else "🔴"
                        
                        # Handle timestamp - could be string or datetime object
                        timestamp = latest_cross['timestamp']
                        if isinstance(timestamp, str):
                            timestamp = datetime.fromisoformat(timestamp)
                        if timestamp.tzinfo is None:
                            timestamp = timestamp.replace(tzinfo=ZoneInfo("UTC"))
                        latest_cross['timestamp'] = timestamp.astimezone(ZoneInfo("Asia/Ho_Chi_Minh"))
                        
                        # Calculate features for ML prediction
                        features_df = None
                        ml_prediction = None
                        if ml_predictor is not None:
                            features_df = ml_predictor.calculate_features(df)
                            if features_df is not None:
                                ml_prediction = ml_predictor.predict(features_df)
                        
                        print()
                        print_separator('-')
                        print(f"{emoji} PHÁT HIỆN {latest_cross['type']} CROSSOVER!")
                        print_separator('-')
                        print(f"Thời gian:  {latest_cross['timestamp']}")
                        print(f"Giá:        ${latest_cross['price']:.2f}")
                        print(f"MACD:       {latest_cross['macd']:.4f}")
                        print(f"Signal:     {latest_cross['signal']:.4f}")
                        print(f"Histogram:  {latest_cross['histogram']:.4f}")
                        
                        if ml_prediction:
                            print_separator('-')
                            print("🤖 ML PREDICTION:")
                            print(f"  Confidence: {ml_prediction['entry_confidence']:.1%}")
                            print(f"  SL:         {ml_prediction['sl_pct']*100:.1f}%")
                            print(f"  TP:         {ml_prediction['tp_pct']*100:.1f}%")
                            print(f"  R/R:        1:{ml_prediction['risk_reward']:.1f}")
                            print(f"  Bars:       ~{ml_prediction['bars_to_peak']}")
                            
                            if ml_prediction['entry_confidence'] < ml_threshold:
                                print(f"  ⚠️ FILTERED (conf < {ml_threshold:.0%})")
                        
                        print_separator('-')
                        
                        # Gửi Telegram (with ML filtering inside)
                        if telegram:
                            sent = telegram.send_crossover_alert(
                                latest_cross, 
                                symbol, 
                                interval,
                                features_df
                            )
                            if not sent and ml_prediction and ml_prediction['entry_confidence'] >= ml_threshold:
                                print("⚠️ Telegram send failed")
                            elif not sent:
                                print("📭 Signal filtered by ML")
                        
                        last_crossover_time = latest_cross['timestamp']
                        print()
                
                # Hiển thị trạng thái
                if not quiet:
                    current_price = df['close'].iloc[-1]
                    current_macd = df['macd'].iloc[-1]
                    current_signal = df['signal'].iloc[-1]
                    
                    trend = "↑" if current_macd > current_signal else "↓"
                    distance = abs(current_macd - current_signal)
                    
                    print(f"[{current_time}] Check #{check_count} | "
                          f"Giá: ${current_price:.2f} | "
                          f"MACD {trend} Signal | "
                          f"Distance: {distance:.4f}")
                
            except Exception as e:
                print(f"[{current_time}] ✗ Lỗi: {e}")
            
            # Chờ trước khi check tiếp
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        if not quiet:
            print()
            print()
            print_separator()
            print("✓ ĐÃ DỪNG MONITOR")
            print_separator()
            print(f"Tổng số lần check: {check_count}")
            if last_crossover_time:
                print(f"Crossover cuối:    {last_crossover_time}")
            print()


def main():
    """Main function"""
    print("\n📡 CÔNG CỤ MONITOR REAL-TIME\n")
    
    # Cấu hình
    SYMBOL = 'BTCUSDT'
    INTERVAL = '30m'
    CHECK_INTERVAL = 60  # giây
    USE_ML = True  # Sử dụng ML để filter và predict
    ML_THRESHOLD = 0.5  # Ngưỡng confidence tối thiểu
    
    # Telegram credentials
    TELEGRAM_BOT_TOKEN = "8484997609:AAHb_L8wO0WjtKRioas0USfhqHOXW_zlFQ0"  # Thay bằng token của bạn
    TELEGRAM_CHAT_ID = "6465176588"    # Thay bằng chat ID của bạn
    
    # Cho phép override từ command line
    quiet_mode = '--quiet' in sys.argv or '-q' in sys.argv
    no_ml = '--no-ml' in sys.argv
    
    # Parse ML threshold from args
    for arg in sys.argv[1:]:
        if arg.startswith('--threshold='):
            try:
                ML_THRESHOLD = float(arg.split('=')[1])
            except:
                pass
    
    args = [arg for arg in sys.argv[1:] if not arg.startswith('-')]
    
    if len(args) > 0:
        SYMBOL = args[0]
    if len(args) > 1:
        INTERVAL = args[1]
    
    if no_ml:
        USE_ML = False
    
    # Kiểm tra Telegram config
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️  CẢNH BÁO: Chưa cấu hình Telegram!")
        print("Chỉnh sửa file realtime_monitor.py để thêm:")
        print("  - TELEGRAM_BOT_TOKEN")
        print("  - TELEGRAM_CHAT_ID")
        print()
        choice = input("Tiếp tục mà không có Telegram? (y/n): ").strip().lower()
        if choice != 'y':
            print("Đã hủy.")
            return
        TELEGRAM_BOT_TOKEN = None
        TELEGRAM_CHAT_ID = None
    
    # Chạy monitor
    monitor_realtime(
        symbol=SYMBOL,
        interval=INTERVAL,
        check_interval=CHECK_INTERVAL,
        telegram_token=TELEGRAM_BOT_TOKEN,
        telegram_chat_id=TELEGRAM_CHAT_ID,
        quiet=quiet_mode,
        use_ml=USE_ML,
        ml_threshold=ML_THRESHOLD
    )


if __name__ == "__main__":
    main()
