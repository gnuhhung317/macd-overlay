"""
Script riêng để phân tích dữ liệu lịch sử
Chạy độc lập để test và xem các điểm giao cắt
"""

import pandas as pd
from data_processor import BinanceDataProcessor
from datetime import datetime
import sys


def print_separator(char='=', length=70):
    """In dòng phân cách"""
    print(char * length)


def analyze_historical_data(symbol='BTCUSDT', interval='1h', start_date='1 year ago UTC', end_date='now UTC'):
    """
    Phân tích dữ liệu lịch sử và hiển thị các crossover
    
    Args:
        symbol (str): Trading pair
        interval (str): Time interval
        start_date (str): Start date (e.g., '2024-01-01', '1 year ago UTC')
        end_date (str): End date (e.g., 'now UTC')
    """
    print_separator()
    print(f"PHÂN TÍCH DỮ LIỆU LỊCH SỬ - {symbol}")
    print_separator()
    print(f"Khung thời gian: {interval}")
    print(f"Từ: {start_date} → Đến: {end_date}")
    print()
    
    # Khởi tạo processor
    processor = BinanceDataProcessor()
    
    # Lấy dữ liệu
    for symbol in ["BTCUSDT", "ETHUSDT"]:

        df = processor.get_historical_data(symbol, interval, start_date, end_date)
        
        # Tính MACD
        print("Đang tính toán MACD...")
        df = processor.calculate_macd(df)
        
        # Phát hiện crossovers
        print("Đang phát hiện các điểm giao cắt...")
        crossovers = processor.detect_crossovers(df)
        
        # Thống kê
        stats = processor.analyze_crossovers(crossovers)
        
        print_separator()
        print("THỐNG KÊ CROSSOVERS")
        print_separator()
        print(f"Tổng số crossovers: {stats['total']}")
        print(f"  • Bullish (Mua):  {stats['bullish']} ({stats['bullish']/stats['total']*100:.1f}%)" if stats['total'] > 0 else "  • Bullish (Mua):  0")
        print(f"  • Bearish (Bán):  {stats['bearish']} ({stats['bearish']/stats['total']*100:.1f}%)" if stats['total'] > 0 else "  • Bearish (Bán):  0")
        print(f"Khoảng cách TB:    {stats['avg_interval_hours']:.1f} giờ")
        print()
        
    # Tính P&L với chiến lược trung tính (LONG & SHORT)
    trades = []
    in_position = False
    entry_price = 0
    entry_time = None
    entry_type = None
    
    for i, cross in enumerate(crossovers):
        if cross['type'] == 'BULLISH':
            # Đóng vị thế hiện tại (nếu có) trước khi vào LONG
            if in_position:
                exit_price = cross['price']
                exit_time = cross['timestamp']
                
                if entry_type == 'LONG':
                    # Đóng LONG: profit khi giá tăng
                    pnl_percent = ((exit_price - entry_price) / entry_price) * 100
                else:  # SHORT
                    # Đóng SHORT: profit khi giá giảm
                    pnl_percent = ((entry_price - exit_price) / entry_price) * 100
                
                duration = (exit_time - entry_time).total_seconds() / 3600
                
                trades.append({
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': exit_time,
                    'exit_price': exit_price,
                    'type': entry_type,
                    'pnl_percent': pnl_percent,
                    'duration_hours': duration,
                    'result': 'WIN' if pnl_percent > 0 else 'LOSS'
                })
            
            # Vào lệnh LONG
            in_position = True
            entry_price = cross['price']
            entry_time = cross['timestamp']
            entry_type = 'LONG'
            
        elif cross['type'] == 'BEARISH':
            # Đóng vị thế hiện tại (nếu có) trước khi vào SHORT
            if in_position:
                exit_price = cross['price']
                exit_time = cross['timestamp']
                
                if entry_type == 'LONG':
                    # Đóng LONG: profit khi giá tăng
                    pnl_percent = ((exit_price - entry_price) / entry_price) * 100
                else:  # SHORT
                    # Đóng SHORT: profit khi giá giảm
                    pnl_percent = ((entry_price - exit_price) / entry_price) * 100
                
                duration = (exit_time - entry_time).total_seconds() / 3600
                
                trades.append({
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': exit_time,
                    'exit_price': exit_price,
                    'type': entry_type,
                    'pnl_percent': pnl_percent,
                    'duration_hours': duration,
                    'result': 'WIN' if pnl_percent > 0 else 'LOSS'
                })
            
            # Vào lệnh SHORT
            in_position = True
            entry_price = cross['price']
            entry_time = cross['timestamp']
            entry_type = 'SHORT'
    
    # Bỏ qua hiển thị chi tiết crossovers - chỉ lưu file
    
    # Hiển thị phân tích giao dịch
    if trades:
        print()
        print_separator()
        print("PHÂN TÍCH GIAO DỊCH (LONG & SHORT)")
        print_separator()
        
        total_trades = len(trades)
        winning_trades = [t for t in trades if t['result'] == 'WIN']
        losing_trades = [t for t in trades if t['result'] == 'LOSS']
        long_trades = [t for t in trades if t['type'] == 'LONG']
        short_trades = [t for t in trades if t['type'] == 'SHORT']
        
        total_pnl = sum(t['pnl_percent'] for t in trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = sum(t['pnl_percent'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['pnl_percent'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        avg_duration = sum(t['duration_hours'] for t in trades) / total_trades if total_trades > 0 else 0
        
        # Tính max drawdown
        max_win = max((t['pnl_percent'] for t in winning_trades), default=0)
        max_loss = min((t['pnl_percent'] for t in losing_trades), default=0)
        
        print(f"\n📊 TỔNG QUAN:")
        print(f"   Tổng số giao dịch:     {total_trades}")
        print(f"   • LONG:                {len(long_trades)} lệnh")
        print(f"   • SHORT:               {len(short_trades)} lệnh")
        print(f"   Giao dịch thắng:       {len(winning_trades)} ({win_rate:.1f}%)")
        print(f"   Giao dịch thua:        {len(losing_trades)} ({100-win_rate:.1f}%)")
        print(f"   Thời gian giữ TB:      {avg_duration:.1f} giờ ({avg_duration/24:.1f} ngày)")
        print(f"\n💰 P&L:")
        print(f"   Tổng P&L:              {'+' if total_pnl >= 0 else ''}{total_pnl:.2f}%")
        print(f"   P&L trung bình:        {'+' if avg_pnl >= 0 else ''}{avg_pnl:.2f}%")
        print(f"   Thắng TB:              +{avg_win:.2f}%")
        print(f"   Thua TB:               {avg_loss:.2f}%")
        print(f"   Thắng lớn nhất:        +{max_win:.2f}%")
        print(f"   Thua lớn nhất:         {max_loss:.2f}%")
        
        # Thêm vào danh sách để lưu CSV
        trades_df = pd.DataFrame(trades)
        trades_file = f"trades_{symbol}_{interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        processor.save_to_csv(trades_df, trades_file)
    else:
        print()
        print("⚠️  Không có giao dịch hoàn chỉnh (cần ít nhất 1 crossover)")
    
    # Lưu kết quả crossovers
    print()
    print_separator()
    print("LƯU KẾT QUẢ")
    print_separator()
    
    # Lưu crossovers
    if crossovers:
        crossover_df = pd.DataFrame(crossovers)
        crossover_file = f"crossovers_{symbol}_{interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        processor.save_to_csv(crossover_df, crossover_file)
    
    # Lưu dữ liệu MACD đầy đủ
    macd_data = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'macd', 'signal', 'histogram']].copy()
    macd_file = f"macd_data_{symbol}_{interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    processor.save_to_csv(macd_data, macd_file)
    
    print()
    print_separator()
    print("✓ HOÀN THÀNH!")
    print_separator()
    
    return df, crossovers, stats


def main():
    """Main function"""
    print("\n📊 CÔNG CỤ PHÂN TÍCH MACD LỊCH SỬ\n")
    
    # Cấu hình mặc định
    SYMBOL = 'BTCUSDT'
    INTERVAL = '1h'
    START_DATE = '1 year ago UTC'  # hoặc '2024-01-01'
    END_DATE = 'now UTC'           # hoặc '2024-12-31'
    
    # Cho phép override từ command line
    if len(sys.argv) > 1:
        SYMBOL = sys.argv[1]
    if len(sys.argv) > 2:
        INTERVAL = sys.argv[2]
    if len(sys.argv) > 3:
        START_DATE = sys.argv[3]
    if len(sys.argv) > 4:
        END_DATE = sys.argv[4]
    
    # Chạy phân tích
    df, crossovers, stats = analyze_historical_data(SYMBOL, INTERVAL, START_DATE, END_DATE)
    
    print(f"\nSử dụng: python historical_analysis.py [SYMBOL] [INTERVAL] [START_DATE] [END_DATE]")
    print(f"Ví dụ:   python historical_analysis.py ETHUSDT 4h '2024-01-01' '2024-12-31'")
    print(f"         python historical_analysis.py BTCUSDT 1h '1 year ago UTC' 'now UTC'")


if __name__ == "__main__":
    main()
