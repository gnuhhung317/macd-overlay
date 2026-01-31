# Hướng Dẫn Sử Dụng - Multi-Timeframe Monitor

## 🚀 Khởi Động Nhanh

### Bước 1: Chạy Test Kiểm Tra
```bash
python test_integration.py
```

Kết quả mong đợi:
```
✅ PASS - Imports
✅ PASS - Configuration  
✅ PASS - Monitor Init
✅ PASS - API Endpoints

Total: 4/4 tests passed 🎉
```

### Bước 2: Khởi Động API Server
```bash
python api_server.py
```

Server sẽ chạy tại: `http://localhost:8000`

### Bước 3: Mở Web UI
1. Mở trình duyệt
2. Truy cập: `http://localhost:8000`
3. Nhấn nút **Start** để bắt đầu monitoring

---

## 📱 Cấu Hình Telegram

### Lấy Chat ID Cho Các Kênh

1. **Tạo kênh mới** (nếu chưa có):
   - Mở Telegram → Tạo kênh mới
   - Đặt tên: "MACD 4h", "MACD 8h", v.v.
   - Thêm bot vào kênh

2. **Lấy Chat ID**:
   - Gửi tin nhắn trong kênh
   - Truy cập: `https://api.telegram.org/bot<TOKEN>/getUpdates`
   - Tìm `"chat":{"id":-123456789}`

3. **Cập nhật config**:
```bash
# Mở file config
notepad monitor_config.json

# Sửa phần này:
"timeframes": {
  "4h": {
    "telegram_chat_id": "-YOUR_4H_CHAT_ID",  # Thay đổi
    ...
  },
  "8h": {
    "telegram_chat_id": "-YOUR_8H_CHAT_ID",  # Thay đổi
    ...
  }
}
```

---

## 🎯 Sử Dụng Web UI

### Màn Hình Chính

#### 1. Hero Stats (Phía trên)
```
┌────────────────────────────────────────┐
│ Multi-Timeframe MACD Monitor           │
├─────────┬─────────┬─────────┬──────────┤
│ TFrames │ Coins   │ Alerts  │ Memory   │
│    4    │   422   │   12    │ 512 MB   │
└─────────┴─────────┴─────────┴──────────┘
```

#### 2. Timeframe Tabs
- **4h** - Khung 4 giờ (15 phút/lần scan)
- **8h** - Khung 8 giờ (30 phút/lần scan)
- **12h** - Khung 12 giờ (45 phút/lần scan)
- **1d** - Khung 1 ngày (60 phút/lần scan)

Nhấn vào tab để xem dữ liệu riêng cho mỗi khung.

#### 3. Live Status Table
Mỗi tab hiển thị:
- **Symbol**: Tên coin (BTCUSDT, ETHUSDT, ...)
- **Price**: Giá hiện tại
- **MACD**: Giá trị MACD
- **Signal**: Đường signal
- **Trend**: 🟢 BULLISH hoặc 🔴 BEARISH
- **Last Update**: Thời gian cập nhật cuối

#### 4. Alerts - Cảnh Báo
Hiển thị các tín hiệu crossover:

```
┌────────────────────────────────────────────┐
│ 🟢 BTCUSDT - BULLISH @ 43250.00           │
│    ML: SL 1.8% | TP 3.2% | Confidence 78% │
│    2026-01-27 10:15:00                     │
└────────────────────────────────────────────┘
```

- **SL**: Stop Loss % (nên cắt lỗ)
- **TP**: Take Profit % (nên chốt lời)  
- **Confidence**: Độ tin cậy của ML model

---

## ⚙️ Cài Đặt Nâng Cao

### 1. Điều Chỉnh Memory Limit
```json
{
  "global_settings": {
    "max_memory_mb": 1000,        // Giới hạn bộ nhớ
    "model_cache_ttl": 3600,      // Giữ model trong 1 giờ
    "base_scan_interval": 60      // Kiểm tra mỗi 60 giây
  }
}
```

### 2. Thay Đổi Scan Interval
```json
{
  "timeframes": {
    "4h": {
      "scan_interval": 900,   // 15 phút (900 giây)
      ...
    },
    "8h": {
      "scan_interval": 1800,  // 30 phút
      ...
    }
  }
}
```

### 3. Tắt/Bật Timeframe
```json
{
  "4h": {
    "enabled": true,   // true = bật, false = tắt
    ...
  }
}
```

---

## 📊 Theo Dõi Hiệu Suất

### Kiểm Tra Memory Usage
- Xem trong Hero Stats: "Memory: 512 MB"
- Hoặc gọi API: `http://localhost:8000/api/status`
  ```json
  {
    "memory_usage_mb": 512.5
  }
  ```

### Xem Statistics Theo Timeframe
```bash
# Gọi API
curl http://localhost:8000/api/timeframes/4h/status

# Kết quả:
{
  "interval": "4h",
  "stats": {
    "last_scan": "2026-01-27T10:30:00",
    "scan_count": 15,
    "alerts_found": 3
  }
}
```

---

## 🔧 Xử Lý Sự Cố

### Lỗi: "Monitor not initialized"
**Nguyên nhân**: Chưa nhấn Start hoặc có lỗi khởi tạo

**Giải pháp**:
1. Kiểm tra console log
2. Nhấn nút Start lại
3. Xem file log để biết lỗi chi tiết

### Lỗi: "Memory usage too high"
**Nguyên nhân**: Vượt quá giới hạn memory

**Giải pháp**:
```json
// Giảm max_memory_mb hoặc model_cache_ttl
{
  "global_settings": {
    "max_memory_mb": 500,      // Giảm xuống 500MB
    "model_cache_ttl": 1800    // Cache 30 phút thay vì 1 giờ
  }
}
```

### Lỗi: "Telegram not sending"
**Nguyên nhân**: Chat ID sai hoặc bot chưa vào kênh

**Giải pháp**:
1. Kiểm tra chat ID trong config
2. Đảm bảo bot đã được thêm vào kênh
3. Test kết nối:
```python
python -c "from telegram_notifier import TelegramNotifier; \
    t = TelegramNotifier('TOKEN', 'CHAT_ID'); \
    print(t.test_connection())"
```

### Lỗi: "No timeframe tabs showing"
**Nguyên nhân**: Không có timeframe nào enabled

**Giải pháp**:
```json
// Đảm bảo có ít nhất 1 timeframe enabled = true
{
  "timeframes": {
    "4h": {
      "enabled": true  // <-- Phải là true
    }
  }
}
```

---

## 📈 Tips Sử Dụng

### 1. Chọn Timeframe Phù Hợp
- **4h**: Scalping/Day trading (tín hiệu nhiều)
- **8h**: Swing trading (tín hiệu ít hơn, chất lượng hơn)
- **12h**: Medium-term trading
- **1d**: Long-term trading (tín hiệu ít nhất, reliable nhất)

### 2. Sử Dụng ML Predictions
```
ML: SL 1.8% | TP 3.2% | Confidence 78%
```
- **Confidence > 70%**: Tín hiệu mạnh, đáng tin cậy
- **Confidence < 50%**: Tín hiệu yếu, cân nhắc bỏ qua
- **SL/TP**: Tham khảo để đặt lệnh

### 3. Theo Dõi Memory
- Nếu Memory > 800MB: Hệ thống sẽ tự cleanup
- Nếu thường xuyên > 1000MB: Giảm cache_ttl

### 4. Kiểm Tra Định Kỳ
- Xem Last Scan time để biết timeframe nào đang hoạt động
- Kiểm tra Alerts count để biết có tín hiệu mới không
- Theo dõi Scan Count để biết số lần đã quét

---

## 🎯 Workflow Đề Xuất

### Khi Bắt Đầu Ngày
1. Mở web UI
2. Nhấn **Start** nếu chưa chạy
3. Kiểm tra Hero Stats
4. Xem alerts mới nhất ở mỗi tab

### Trong Ngày
1. Nhận thông báo từ Telegram
2. Mở web UI để xem chi tiết
3. Kiểm tra ML predictions
4. Quyết định entry/exit

### Cuối Ngày
1. Xem tổng số alerts
2. Review các tín hiệu đã bỏ lỡ
3. Đánh giá hiệu suất của từng timeframe

---

## 📱 Cấu Hình Telegram Channels

### Đề Xuất Setup
```
@macd_4h_alerts   → Khung 4h  (nhiều tín hiệu)
@macd_8h_alerts   → Khung 8h  (vừa phải)
@macd_12h_alerts  → Khung 12h (ít tín hiệu)
@macd_1d_alerts   → Khung 1d  (rất ít, nhưng quan trọng)
```

### Ưu Điểm
- ✅ Tách biệt rõ ràng theo timeframe
- ✅ Có thể mute/unmute riêng từng kênh
- ✅ Dễ quản lý, không bị spam
- ✅ ML predictions hiển thị ngay trong alert

---

## 🚀 Bắt Đầu Ngay!

```bash
# Bước 1: Test
python test_integration.py

# Bước 2: Start server
python api_server.py

# Bước 3: Mở browser
# http://localhost:8000

# Bước 4: Nhấn Start và enjoy! 🎉
```

---

**Chúc bạn trade thành công! 📈💰**
