# MACD Crossover Detector

Chương trình Python để phát hiện điểm giao cắt MACD từ dữ liệu Binance và gửi thông báo qua Telegram.

## 🏗️ Cấu trúc dự án

```
macd-overlay/
├── config.py                   # File cấu hình chính
├── data_processor.py          # Module xử lý dữ liệu Binance & MACD
├── telegram_notifier.py       # Module gửi thông báo Telegram
├── historical_analysis.py     # Script phân tích dữ liệu lịch sử
├── realtime_monitor.py        # Script monitor real-time
├── macd_crossover.py          # Chương trình chính (all-in-one)
├── requirements.txt           # Dependencies
└── README.md                  # Tài liệu này
```

## ✨ Tính năng

- ✅ **Modular design**: Tách riêng từng chức năng
- ✅ Lấy dữ liệu lịch sử từ Binance
- ✅ Tính toán MACD và Signal line
- ✅ Phát hiện điểm giao cắt (Bullish/Bearish)
- ✅ Phân tích dữ liệu lịch sử chi tiết
- ✅ Monitor real-time với Telegram
- ✅ Lưu kết quả ra file CSV
- ✅ Thống kê và báo cáo

## Cài đặt

### 1. Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```

### 2. Cấu hình Telegram Bot

#### Tạo Telegram Bot:
1. Mở Telegram và tìm `@BotFather`
2. Gửi lệnh `/newbot`
3. Đặt tên và username cho bot
4. Lưu lại **Bot Token** mà BotFather cung cấp

#### Lấy Chat ID:
1. Gửi tin nhắn bất kỳ cho bot của bạn
2. Truy cập URL sau (thay `<YOUR_BOT_TOKEN>`):
   ```
   https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
   ```
3. Tìm `"chat":{"id":` trong kết quả JSON
4. Lưu lại **Chat ID**

### 3. Cấu hình

Mở file **`config.py`** và cập nhật:

```python
# Telegram
TELEGRAM_BOT_TOKEN = "your_bot_token_here"
TELEGRAM_CHAT_ID = "your_chat_id_here"

# Trading parameters
SYMBOL = 'BTCUSDT'
INTERVAL = '1h'
LOOKBACK = '30 days ago UTC'
```

## 🚀 Sử dụng

### Cách 1: Chương trình chính (All-in-one)

```bash
python macd_crossover.py
```

Menu sẽ hiện ra:
```
1. Phân tích dữ liệu lịch sử
2. Monitor real-time
3. Cả hai
```

### Cách 2: Chạy riêng từng module

**Chỉ phân tích lịch sử:**
```bash
python historical_analysis.py
python historical_analysis.py ETHUSDT 4h "60 days ago UTC"
```

**Chỉ monitor real-time:**
```bash
python realtime_monitor.py
```

## Output

### Console Output:
```
==============================================================
BẮT ĐẦU TEST VỚI DỮ LIỆU LỊCH SỬ
==============================================================
Đang lấy dữ liệu BTCUSDT với khung thời gian 1h...
Đã lấy 720 nến

✓ Tìm thấy 12 điểm giao cắt:
------------------------------------------------------------

1. BULLISH CROSSOVER
   Thời gian: 2024-12-15 08:00:00
   Giá: $42150.50
   MACD: 125.4523
   Signal: 118.3421
   Histogram: 7.1102
```

### File CSV:
- `crossovers_BTCUSDT_1h.csv`: Danh sách các điểm giao cắt
- `macd_data_BTCUSDT_1h.csv`: Dữ liệu MACD đầy đủ

### Telegram notification:
```
🟢 MACD Crossover - BTCUSDT

📊 Loại: BULLISH
💰 Giá: $42150.50
📅 Thời gian: 2024-12-15 08:00:00

📈 MACD: 125.4523
📉 Signal: 118.3421
📊 Histogram: 7.1102
```

## Tham số MACD

Mặc định (theo Pine Script):
- Fast EMA: 12
- Slow EMA: 26
- Signal SMA: 9

Có thể điều chỉnh trong class `MACDCrossoverDetector`:
```python
self.fast_period = 12
self.slow_period = 26
self.signal_period = 9
```

## Lưu ý

- **Binance API**: Không cần API key/secret cho public data
- **Rate limit**: Binance giới hạn số request, nên đặt `check_interval` >= 60 giây
- **Historical data**: Binance giới hạn 1000 nến mỗi request
- **Telegram**: Cần cấu hình bot token và chat ID để nhận thông báo

## Troubleshooting

### Lỗi connection:
```bash
pip install --upgrade python-binance
```

### Không nhận được Telegram notification:
- Kiểm tra bot token và chat ID
- Đảm bảo đã gửi tin nhắn cho bot trước
- Kiểm tra bot đã được start chưa

### Lỗi pandas/numpy:
```bash
pip install --upgrade pandas numpy
```
