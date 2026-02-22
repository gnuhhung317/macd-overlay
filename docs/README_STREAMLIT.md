# 📊 MACD Monitor Dashboard - Hướng dẫn sử dụng

## 🚀 Khởi chạy

```bash
streamlit run streamlit_monitor.py
```

Dashboard sẽ mở tại: `http://localhost:8501`

## ✨ Tính năng

### 1. **Theo dõi nhiều coins**
- Thêm/xóa coins dễ dàng qua sidebar
- Mỗi coin có thể có khung thời gian riêng (1m, 5m, 15m, 30m, 1h, 4h, 1d...)
- Bật/tắt từng coin theo ý muốn

### 2. **Cấu hình linh hoạt**
- **Tần suất quét**: Điều chỉnh từ 10-300 giây
- **Telegram alerts**: Tùy chọn bật/tắt thông báo
- **Lưu cấu hình**: Tự động lưu vào file `monitor_config.json`

### 3. **Giao diện trực quan**
- **Tab Trạng thái**: Xem real-time status của tất cả coins
  - Giá hiện tại
  - MACD & Signal values
  - Trend (Bullish/Bearish)
  - Khoảng cách MACD-Signal
  
- **Tab Alerts**: Lịch sử các crossover signals
  - Loại crossover (Bullish/Bearish)
  - Thời gian chính xác
  - Thông tin chi tiết (MACD, Signal, Histogram)

### 4. **Telegram Integration**
- Nhận thông báo ngay khi có crossover
- Cấu hình Bot Token và Chat ID qua sidebar
- Test connection trước khi bắt đầu

## 🎯 Cách sử dụng

### Bước 1: Thêm coins
1. Mở sidebar (bên trái)
2. Kéo xuống phần "➕ Thêm Coin"
3. Nhập symbol (VD: BTCUSDT, ETHUSDT)
4. Chọn interval
5. Click "➕ Thêm"

### Bước 2: Cấu hình
1. Điều chỉnh "Tần suất quét" theo nhu cầu
   - Quét nhanh (10-30s): Theo dõi sát
   - Quét chậm (60-300s): Tiết kiệm tài nguyên

2. (Tùy chọn) Cấu hình Telegram
   - Bật checkbox "Bật thông báo Telegram"
   - Nhập Bot Token và Chat ID
   - Click "💾 Lưu Telegram"

### Bước 3: Bắt đầu monitor
1. Click nút "▶️ Bắt đầu" ở sidebar
2. Dashboard sẽ tự động refresh và cập nhật
3. Xem trạng thái real-time ở tab "📊 Trạng thái"
4. Kiểm tra alerts ở tab "🔔 Alerts"

### Bước 4: Dừng monitor
- Click nút "⏸️ Dừng" khi muốn ngừng theo dõi

## 🔧 Cấu hình mẫu

File `monitor_config.json` sẽ tự động tạo:

```json
{
  "coins": [
    {"symbol": "BTCUSDT", "interval": "30m", "enabled": true},
    {"symbol": "ETHUSDT", "interval": "1h", "enabled": true},
    {"symbol": "BNBUSDT", "interval": "15m", "enabled": false}
  ],
  "scan_interval": 60,
  "telegram_enabled": true,
  "telegram_token": "YOUR_BOT_TOKEN",
  "telegram_chat_id": "YOUR_CHAT_ID"
}
```

## 📱 Setup Telegram Bot

1. Tạo bot với [@BotFather](https://t.me/BotFather)
   - Gửi `/newbot`
   - Đặt tên cho bot
   - Lấy Bot Token

2. Lấy Chat ID
   - Chat với bot của bạn
   - Truy cập: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
   - Tìm `"chat":{"id":123456789}`

3. Nhập vào dashboard và test

## 💡 Tips

- **Multi-timeframe**: Theo dõi cùng coin ở nhiều khung giờ khác nhau
- **Selective monitoring**: Tắt coins không quan tâm thay vì xóa
- **Alert history**: Giữ lại 50 alerts gần nhất để review
- **Auto-save**: Tất cả cấu hình được lưu tự động

## ⚠️ Lưu ý

- Dashboard tự refresh mỗi 2 giây khi đang chạy
- Scan interval ảnh hưởng đến tốc độ phát hiện crossover
- Quá nhiều coins có thể làm chậm dashboard
- Khuyến nghị: 5-10 coins cho performance tốt nhất

## 🎨 Giao diện

```
┌─────────────────────────────────────────────────────────┐
│  Sidebar                │  Main Dashboard               │
│  ─────────              │  ─────────────                │
│  ⚙️ Cấu hình            │  📊 Header Stats              │
│  🔄 Tần suất quét       │  ┌──────┬──────┬──────┐      │
│  📱 Telegram            │  │Coins │Freq  │Status│      │
│  💰 Danh sách Coins     │  └──────┴──────┴──────┘      │
│    ✓ BTCUSDT  [30m]    │                                │
│    ✓ ETHUSDT  [1h]     │  Tab: Trạng thái              │
│  ➕ Thêm Coin           │  ┌─────────────────────────┐  │
│  ▶️ Bắt đầu             │  │ Symbol │ Price │ Trend │  │
│                         │  ├─────────────────────────┤  │
│  Stats:                 │  │ BTC... │$43.2K │ 🟢   │  │
│  Số lần quét: 42        │  └─────────────────────────┘  │
│  Alerts: 3              │                                │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Performance Tips

1. **Tần suất quét hợp lý**
   - 1m chart → 30-60s scan
   - 15m chart → 60-120s scan
   - 1h+ chart → 120-300s scan

2. **Số lượng coins**
   - Ít coins (1-5): 30s scan OK
   - Nhiều coins (10+): 60s+ scan recommended

3. **Telegram**
   - Chỉ bật khi cần thiết
   - Giảm load khi không dùng
