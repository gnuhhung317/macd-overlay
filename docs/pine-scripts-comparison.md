# So sánh Pine Scripts

## 1. MACD Overlay (Bản gốc - Custom)
**File:** `macd-overlay.pine`

### Công thức:
```pine
macd = ema(close, sa-fa)  // ema(close, 14) với fast=12, slow=26
signal = sma(macd, sig)   // sma(macd, 9)
```

### Đặc điểm:
- ❌ **KHÔNG phải MACD chuẩn**
- Chỉ là EMA(14) của giá close
- Đơn giản hơn nhưng khác với MACD truyền thống

---

## 2. MACD Overlay (Standard)
**File:** `macd-overlay-standard.pine`

### Công thức:
```pine
fastEMA = ema(close, 12)
slowEMA = ema(close, 26)
macdLine = fastEMA - slowEMA  // MACD = EMA(12) - EMA(26)
signalLine = sma(macdLine, 9) // Signal = SMA(9) của MACD
```

### Đặc điểm:
- ✅ **MACD chuẩn** (standard)
- Được sử dụng rộng rãi trong technical analysis
- Khớp với code Python sau khi sửa

---

## So sánh trực quan

| Tính năng | Bản Gốc (Custom) | Bản Standard |
|-----------|------------------|--------------|
| MACD Line | `ema(close, 14)` | `ema(12) - ema(26)` |
| Signal Line | `sma(macd, 9)` | `sma(macd, 9)` |
| Histogram | ✅ Có | ✅ Có |
| Crossover Markers | ❌ Không | ✅ Có |
| Alerts | ❌ Không | ✅ Có |
| Version | v2 | v5 |

---

## Cách test

1. **Thêm cả 2 indicators vào TradingView:**
   - Add `macd-overlay.pine` (custom)
   - Add `macd-overlay-standard.pine` (standard)

2. **So sánh kết quả:**
   - Bạn sẽ thấy MACD line khác nhau hoàn toàn
   - Crossover points sẽ khác nhau

3. **Test với Python:**
   - Code Python hiện tại khớp với **bản gốc (custom)**
   - Để khớp với **bản standard**, cần sửa lại `data_processor.py`

---

## Khuyến nghị

- 🎯 **Dùng bản Standard** nếu bạn muốn MACD chuẩn được công nhận
- 🔧 **Giữ bản Custom** nếu đã backtest và có kết quả tốt
- 📊 **Test cả 2** để xem bản nào cho tín hiệu tốt hơn với chiến lược của bạn
