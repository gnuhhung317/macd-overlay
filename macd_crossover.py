"""
MACD Crossover Detector - Main Program
Chương trình chính tích hợp tất cả modules
"""

from data_processor import BinanceDataProcessor
from telegram_notifier import TelegramNotifier
from datetime import datetime
import time


class MACDCrossoverDetector:
    def __init__(self, api_key="", api_secret="", telegram_token=None, telegram_chat_id=None):
        """
        Khởi tạo detector với Binance API và Telegram credentials
        """
        # Khởi tạo data processor
        self.processor = BinanceDataProcessor(api_key, api_secret)
        
        # Khởi tạo Telegram notifier (nếu có)
        self.telegram = None
        if telegram_token and telegram_chat_id:
            self.telegram = TelegramNotifier(telegram_token, telegram_chat_id)
        

    
    def test_historical_crossovers(self, symbol='BTCUSDT', interval='1h', start_date='30 days ago UTC', end_date='now UTC'):
        """
        Test với dữ liệu lịch sử và hiển thị các điểm giao cắt
        """
        print("="*60)
        print("TEST VỚI DỮ LIỆU LỊCH SỬ")
        print("="*60)
        
        # Lấy và xử lý dữ liệu
        df = self.processor.get_historical_data(symbol, interval, start_date, end_date)
        df = self.processor.calculate_macd(df)
        
        # Phát hiện crossovers
        crossovers = self.processor.detect_crossovers(df)
        
        # Thống kê
        stats = self.processor.analyze_crossovers(crossovers)
        
        print(f"\n✓ Tìm thấy {stats['total']} điểm giao cắt:")
        print(f"  • Bullish: {stats['bullish']}")
        print(f"  • Bearish: {stats['bearish']}")
        print("-"*60)
        
        # Hiển thị chi tiết
        for i, cross in enumerate(crossovers, 1):
            emoji = "🟢" if cross['type'] == 'BULLISH' else "🔴"
            print(f"\n{i}. {emoji} {cross['type']} CROSSOVER")
            print(f"   Thời gian: {cross['timestamp']}")
            print(f"   Giá: ${cross['price']:.2f}")
            print(f"   MACD: {cross['macd']:.4f}")
            print(f"   Signal: {cross['signal']:.4f}")
            print(f"   Histogram: {cross['histogram']:.4f}")
        
        # Lưu kết quả
        if crossovers:
            import pandas as pd
            output_file = f"crossovers_{symbol}_{interval}.csv"
            pd.DataFrame(crossovers).to_csv(output_file, index=False)
            print(f"\n✓ Đã lưu kết quả vào {output_file}")
            
            macd_output = f"macd_data_{symbol}_{interval}.csv"
            self.processor.save_to_csv(
                df[['timestamp', 'close', 'macd', 'signal', 'histogram']], 
                macd_output
            )
        
        return df, crossovers
    
    def monitor_realtime(self, symbol='BTCUSDT', interval='1h', check_interval=60):
        """
        Monitor real-time và gửi thông báo khi có crossover
        """
        print("="*60)
        print("MONITOR REAL-TIME")
        print("="*60)
        print(f"Symbol: {symbol}")
        print(f"Interval: {interval}")
        print(f"Telegram: {'Enabled ✓' if self.telegram else 'Disabled ✗'}")
        print("Nhấn Ctrl+C để dừng...")
        print("-"*60)
        
        last_crossover_time = None
        
        try:
            while True:
                # Lấy và xử lý dữ liệu
                df = self.processor.get_historical_data(symbol, interval, '2 days ago UTC', 'now UTC')
                df = self.processor.calculate_macd(df)
                
                # Kiểm tra crossover
                latest_cross = self.processor.get_latest_crossover(df.tail(20))
                
                if latest_cross and last_crossover_time != latest_cross['timestamp']:
                    emoji = "🟢" if latest_cross['type'] == 'BULLISH' else "🔴"
                    print(f"\n{emoji} Phát hiện {latest_cross['type']} crossover!")
                    print(f"   Giá: ${latest_cross['price']:.2f}")
                    
                    # Gửi Telegram
                    if self.telegram:
                        self.telegram.send_crossover_alert(latest_cross, symbol)
                    
                    last_crossover_time = latest_cross['timestamp']
                
                # Hiển thị trạng thái
                current_price = df['close'].iloc[-1]
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Giá: ${current_price:.2f}")
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n\n✓ Đã dừng monitor.")


def main(): 
 
    print("\n" + "="*70)
    print(" "*20 + "MACD CROSSOVER DETECTOR")
    print("="*70)
    print("\nChọn chức năng:")
    print("  1. Phân tích dữ liệu lịch sử (Historical Analysis)")
    print("  2. Monitor real-time")
    print("  3. Cả hai (Test trước, sau đó monitor)")
    print()
    
    choice = input("Lựa chọn (1/2/3): ").strip()
    
    # Cấu hình
    SYMBOL = 'BTCUSDT'
    INTERVAL = '1h'
    START_DATE = '30 days ago UTC'
    END_DATE = 'now UTC'
    
    # Telegram (tùy chọn)
    TELEGRAM_BOT_TOKEN = ""  # Thay bằng token của bạn
    TELEGRAM_CHAT_ID = ""    # Thay bằng chat ID của bạn
    
    # Khởi tạo detector
    detector = MACDCrossoverDetector(
        telegram_token=TELEGRAM_BOT_TOKEN if TELEGRAM_BOT_TOKEN else None,
        telegram_chat_id=TELEGRAM_CHAT_ID if TELEGRAM_CHAT_ID else None
    )
    
    if choice == '1':
        # Chỉ phân tích lịch sử
        print("\n📊 PHÂN TÍCH DỮ LIỆU LỊCH SỬ\n")START_DATE, END_DATE
        detector.test_historical_crossovers(SYMBOL, INTERVAL, LOOKBACK)
        print("\n✓ Hoàn thành!")
        
    elif choice == '2':
        # Chỉ monitor real-time
        print("\n📡 MONITOR REAL-TIME\n")
        if not TELEGRAM_BOT_TOKEN:
            print("⚠️  Telegram chưa được cấu hình. Chỉ hiển thị trên console.")
            input("Nhấn Enter để tiếp tục...")
        detector.monitor_realtime(SYMBOL, INTERVAL, check_interval=60)
        
    elif choice == '3':
        # Cả hai
        print("\n📊 BƯỚC 1: PHÂN TÍCH DỮ LIỆU LỊCH SỬ\n")
        detector.test_historical_crossovers(SYMBOL, INTERVAL, START_DATE, END_DATE)
        
        print("\n" + "="*70)
        cont = input("\nTiếp tục với monitor real-time? (y/n): ").strip().lower()
        
        if cont == 'y':
            print("\n📡 BƯỚC 2: MONITOR REAL-TIME\n")
            detector.monitor_realtime(SYMBOL, INTERVAL, check_interval=60)
        else:
            print("\n✓ Hoàn thành!")
    else:
        print("\n✗ Lựa chọn không hợp lệ")


if __name__ == "__main__":
    main()
