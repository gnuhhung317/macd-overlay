"""
Script monitor real-time
Chạy liên tục để theo dõi crossovers và gửi thông báo
"""

import time
from datetime import datetime
from zoneinfo import ZoneInfo
import sys
import os

from data_processor import BinanceDataProcessor
from telegram_notifier import TelegramNotifier
from ml.inference import InferenceEngine

ML_AVAILABLE = True

def print_separator(char='=', length=70):
    """In dòng phân cách"""
    print(char * length)

def monitor_realtime(symbol='BTCUSDT', interval='1h', check_interval=60, 
                     telegram_token=None, telegram_chat_id=None, quiet=False,
                     use_ml=True, ml_threshold=0.65):
    """
    Monitor real-time và gửi thông báo khi có crossover
    """
    print_separator()
    print(f"🚀 BẮT ĐẦU MONITOR REAL-TIME MACD")
    print(f"📌 Symbol: {symbol}")
    print(f"⏱️ Interval: {interval}")
    print(f"🤖 ML Filter: {'ON' if use_ml else 'OFF'} (Threshold: {ml_threshold:.0%})")
    print(f"🔄 Check Interval: {check_interval}s")
    
    if telegram_token and telegram_chat_id:
        print(f"📱 Telegram: ENABLED")
    else:
        print(f"📱 Telegram: DISABLED (Check logs)")
    print_separator()
    
    # Initialize processor
    # Always fetch enough data for indicators (EMA 200)
    # 4h: 365 days ~2190 bars
    # 1d: 365 days ~365 bars
    # This is safe for all supported timeframes
    fetch_start = "400 days ago UTC" 
        
    processor = BinanceDataProcessor(use_futures=True)
    
    # Initialize ML Engine
    inference_engine = None
    if use_ml:
        try:
            print("📦 Loading ML models...")
            inference_engine = InferenceEngine(interval)
            print("✅ ML models loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load ML models: {e}")
            print("Running without ML filter...")
            use_ml = False
    
    # Initialize notifier
    notifier = TelegramNotifier(telegram_token, telegram_chat_id)
    if telegram_token and telegram_chat_id:
        if not quiet:
            print("🔔 Test Telegram...")
        if notifier.test_connection():
            if not quiet:
                print("✓ Telegram sẵn sàng!")
        else:
            if not quiet:
                print("⚠️  Telegram không hoạt động, chỉ hiển thị console")
            notifier = None
    else:
        notifier = None
        
    last_processed_time = None
    check_count = 0
    
    print(f"⏳ Fetching history from: {fetch_start}")
    
    try:
        while True:
            check_count += 1
            now = datetime.now()
            current_time_str = now.strftime('%Y-%m-%d %H:%M:%S')
            
            if not quiet:
                print(f"\n[{now.strftime('%H:%M:%S')}] Checking...", end='\r')
            
            try:
                # 1. Fetch data
                df = processor.get_historical_data(symbol, interval, fetch_start, 'now UTC')
                
                if len(df) < 200:
                    print(f"⚠️ Not enough data ({len(df)} bars). Waiting...")
                    time.sleep(check_interval)
                    continue
                    
                # 2. Calculate MACD and Features
                # Note: InferenceEngine.predict calls calculate_features internally on a copy
                # But we need basic MACD here for crossover detection first
                df = processor.calculate_macd(df)
                
                # 3. Detect recent crossover
                crossovers = processor.detect_crossovers(df.tail(10)) 
                
                if crossovers:
                    latest_cross = crossovers[-1]
                    cross_time = latest_cross['timestamp']
                    
                    # Handle timezone for display
                    if isinstance(cross_time, str):
                        cross_time = datetime.fromisoformat(cross_time)
                    if cross_time.tzinfo is None:
                        cross_time = cross_time.replace(tzinfo=ZoneInfo("UTC"))
                    display_time = cross_time.astimezone(ZoneInfo("Asia/Ho_Chi_Minh"))
                    
                    # Determine if new signal
                    is_new = False
                    if last_processed_time is None:
                        # On startup, finding a signal that just happened or is very recent
                        # For simplicity, if it's the latest bar, we might consider it.
                        # But to avoid spam on restart, usually we skip old ones.
                        # Let's just track it and wait for next one.
                        last_processed_time = cross_time
                        print(f"\nFound existing signal at {display_time}. Waiting for new ones.")
                    elif cross_time > last_processed_time:
                        is_new = True
                        last_processed_time = cross_time
                        
                    if is_new:
                        price = latest_cross['price']
                        cross_type = latest_cross['type']
                        emoji = "🟢" if cross_type == 'BULLISH' else "🔴"
                        
                        print()
                        print_separator('-')
                        print(f"{emoji} PHÁT HIỆN {cross_type} CROSSOVER!")
                        print_separator('-')
                        print(f"Thời gian:  {display_time}")
                        print(f"Giá:        ${price:.4f}")
                        
                        # 4. ML Prediction
                        ml_confidence = 0.0
                        is_valid = True
                        sl_info = ""
                        tp_info = ""
                        
                        if use_ml and inference_engine:
                            # Pass data up to crossover
                            # Find index in original df
                            # crossovers return 'index' which is index in the tail slice? No, data_processor returns row index usually
                            # Let's rely on timestamp match or pass latest data
                            
                            prediction = inference_engine.predict(symbol, df)
                            
                            if not prediction.get('error'):
                                ml_confidence = prediction['confidence']
                                print(f"🤖 ML Confidence: {ml_confidence:.1%}")
                                
                                sl_p = prediction.get('sl_price', 0)
                                tp_p = prediction.get('tp_price', 0)
                                sl_pct = prediction.get('sl_pct', 0)
                                tp_pct = prediction.get('tp_pct', 0)
                                
                                sl_info = f"{sl_p:.4f} ({sl_pct:.1%})"
                                tp_info = f"{tp_p:.4f} ({tp_pct:.1%})"
                                
                                print(f"  SL: {sl_info} | TP: {tp_info}")
                                
                                if ml_confidence < ml_threshold:
                                    print(f"⚠️ FILTERED (Low Confidence < {ml_threshold:.0%})")
                                    is_valid = False
                                else:
                                    print(f"✅ ML APPROVED")
                            else:
                                print(f"⚠️ ML Error: {prediction.get('error')}")
                        
                        # 5. Send Notification
                        if is_valid and notifier:
                            message = f"🚀 **MACD SIGNAL: {symbol}** 🚀\n"
                            message += f"Type: **{cross_type}** {emoji}\n"
                            message += f"Time: {display_time.strftime('%H:%M %d/%m')}\n"
                            message += f"Price: {price}\n"
                            message += f"Timeframe: {interval}\n"
                            
                            if use_ml:
                                message += f"🤖 Conf: **{ml_confidence:.1%}**\n"
                                if sl_info:
                                    message += f"🛑 SL: {sl_info}\n"
                                if tp_info:
                                    message += f"🎯 TP: {tp_info}\n"
                            
                            notifier.send_message(message)
                            print("📨 Notification sent!")
                        
                        print_separator('-')
                        print()

                # Status update
                if not quiet:
                    current_price = df['close'].iloc[-1]
                    macd = df['macd'].iloc[-1]
                    sig = df['signal'].iloc[-1]
                    trend = "↑" if macd > sig else "↓"
                    print(f"[{current_time_str}] #{check_count} | ${current_price:.2f} | MACD {trend} Sig", end='\r')
                    
            except Exception as e:
                print(f"\n❌ Loop Error: {e}")
            
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n\n✓ STOPPED MONITOR")

def main():
    """Main function"""
    print("\n📡 CÔNG CỤ MONITOR REAL-TIME (UPDATED)\n")
    
    # Cấu hình
    SYMBOL = 'BTCUSDT'
    INTERVAL = '30m'
    CHECK_INTERVAL = 60  # giây
    USE_ML = True  # Sử dụng ML để filter và predict
    ML_THRESHOLD = 0.5  # Ngưỡng confidence tối thiểu
    
    # Telegram credentials
    TELEGRAM_BOT_TOKEN = "8484997609:AAHb_L8wO0WjtKRioas0USfhqHOXW_zlFQ0" 
    TELEGRAM_CHAT_ID = "6465176588"
    
    # Cho phép override từ command line
    quiet_mode = '--quiet' in sys.argv or '-q' in sys.argv
    no_ml = '--no-ml' in sys.argv
    
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
    
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️  CẢNH BÁO: Chưa cấu hình Telegram!")
        
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
