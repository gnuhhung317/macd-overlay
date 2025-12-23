"""
Script monitor real-time
Chạy liên tục để theo dõi crossovers và gửi thông báo
"""

import time
from datetime import datetime
from data_processor import BinanceDataProcessor
from telegram_notifier import TelegramNotifier
import sys
from zoneinfo import ZoneInfo


def print_separator(char='=', length=70):
    """In dòng phân cách"""
    print(char * length)


def monitor_realtime(symbol='BTCUSDT', interval='1h', check_interval=60, 
                     telegram_token=None, telegram_chat_id=None, quiet=False):
    """
    Monitor real-time và gửi thông báo khi có crossover
    
    Args:
        symbol (str): Trading pair
        interval (str): Time interval
        check_interval (int): Số giây giữa mỗi lần check
        telegram_token (str): Telegram bot token
        telegram_chat_id (str): Telegram chat ID
        quiet (bool): Chỉ hiển thị crossover alerts, ẩn status updates
    """
    if not quiet:
        print_separator()
        print("BẮT ĐẦU MONITOR REAL-TIME")
        print_separator()
        print(f"Symbol:          {symbol}")
        print(f"Interval:        {interval}")
        print(f"Check every:     {check_interval} seconds")
        print(f"Telegram:        {'Enabled ✓' if telegram_token else 'Disabled ✗'}")
        print("Nhấn Ctrl+C để dừng...")
        print_separator()
        print()
    
    # Khởi tạo processor
    processor = BinanceDataProcessor()
    
    # Khởi tạo Telegram (nếu có)
    telegram = None
    if telegram_token and telegram_chat_id:
        telegram = TelegramNotifier(telegram_token, telegram_chat_id)
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
                # Lấy dữ liệu gần đây
                df = processor.get_historical_data(symbol, interval, '2 days ago UTC', 'now UTC')
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
                        print()
                        print_separator('-')
                        print(f"{emoji} PHÁT HIỆN {latest_cross['type']} CROSSOVER!")
                        print_separator('-')
                        print(f"Thời gian:  {latest_cross['timestamp']}")
                        print(f"Giá:        ${latest_cross['price']:.2f}")
                        print(f"MACD:       {latest_cross['macd']:.4f}")
                        print(f"Signal:     {latest_cross['signal']:.4f}")
                        print(f"Histogram:  {latest_cross['histogram']:.4f}")
                        print_separator('-')
                        
                        # Gửi Telegram
                        if telegram:
                            telegram.send_crossover_alert(latest_cross, symbol)
                        
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
    
    # Telegram credentials
    TELEGRAM_BOT_TOKEN = "8484997609:AAHb_L8wO0WjtKRioas0USfhqHOXW_zlFQ0"  # Thay bằng token của bạn
    TELEGRAM_CHAT_ID = "6465176588"    # Thay bằng chat ID của bạn
    
    # Cho phép override từ command line
    quiet_mode = '--quiet' in sys.argv or '-q' in sys.argv
    args = [arg for arg in sys.argv[1:] if arg not in ['--quiet', '-q']]
    
    if len(args) > 0:
        SYMBOL = args[0]
    if len(args) > 1:
        INTERVAL = args[1]
    
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
        quiet=quiet_mode
    )


if __name__ == "__main__":
    main()
