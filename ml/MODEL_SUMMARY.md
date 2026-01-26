# 📊 ML Model Summary - MACD Crossover Strategy

> Generated: 2026-01-26 22:22

---

## 🎯 Tổng Quan

Hệ thống sử dụng 3-Stage ML để quyết định giao dịch:

1. **Entry Filter**: Quyết định có nên vào lệnh không (Classification)
2. **SL Predictor**: Dự đoán Stop Loss tối ưu (Regression)
3. **TP Predictor**: Dự đoán Take Profit tối ưu (Regression)


---
## ⏱️ Timeframe: 4H

### 📈 Thống Kê Data

| Metric | Value |
|--------|-------|
| Tổng rows | 2,252,387 |
| Crossover signals | 175,525 |
| Số symbols | 531 |
| Thời gian | 2020-01-01 → 2026-01-17 |
| Win rate (raw) | 44.1% |

### 🎯 Entry Filter Accuracy

**AUC Score: 0.612**

| Threshold | Accuracy | Precision | Signals |
|-----------|----------|-----------|---------|
| 0.50 | 59.0% | 58.7% | 6,218 |
| 0.60 | 56.9% | 77.1% | 647 |
| 0.65 | 56.4% | 82.3% | 271 |
| 0.70 | 56.2% | 84.5% | 174 |
| 0.75 | 56.2% | 86.5% | 126 |

### 📉 SL Predictor Accuracy

| Metric | Value |
|--------|-------|
| SL thực tế (trung bình) | 5.21% |
| SL thực tế (std dev) | ±2.46% |
| SL range | 0.75% - 15.00% |
| **MAE (sai số tuyệt đối)** | **0.59%** |
| Sai số tương đối | 14.7% |

### 📈 TP Predictor Accuracy

| Metric | Value |
|--------|-------|
| TP thực tế (trung bình) | 10.80% |
| TP thực tế (std dev) | ±5.58% |
| TP range | 1.50% - 30.00% |
| **MAE (sai số tuyệt đối)** | **1.21%** |
| Sai số tương đối | 14.9% |


---
## ⏱️ Timeframe: 8H

### 📈 Thống Kê Data

| Metric | Value |
|--------|-------|
| Tổng rows | 1,124,323 |
| Crossover signals | 86,296 |
| Số symbols | 517 |
| Thời gian | 2020-01-01 → 2026-01-17 |
| Win rate (raw) | 46.0% |

### 🎯 Entry Filter Accuracy

**AUC Score: 0.643**

| Threshold | Accuracy | Precision | Signals |
|-----------|----------|-----------|---------|
| 0.50 | 60.2% | 59.5% | 5,635 |
| 0.60 | 57.4% | 73.2% | 1,282 |
| 0.65 | 56.1% | 80.1% | 598 |
| 0.70 | 55.5% | 86.4% | 360 |
| 0.75 | 54.9% | 91.1% | 202 |

### 📉 SL Predictor Accuracy

| Metric | Value |
|--------|-------|
| SL thực tế (trung bình) | 7.09% |
| SL thực tế (std dev) | ±2.80% |
| SL range | 0.75% - 15.00% |
| **MAE (sai số tuyệt đối)** | **0.78%** |
| Sai số tương đối | 16.6% |

### 📈 TP Predictor Accuracy

| Metric | Value |
|--------|-------|
| TP thực tế (trung bình) | 15.10% |
| TP thực tế (std dev) | ±6.57% |
| TP range | 1.50% - 30.00% |
| **MAE (sai số tuyệt đối)** | **1.60%** |
| Sai số tương đối | 16.2% |


---
## ⏱️ Timeframe: 12H

### 📈 Thống Kê Data

| Metric | Value |
|--------|-------|
| Tổng rows | 745,284 |
| Crossover signals | 56,835 |
| Số symbols | 490 |
| Thời gian | 2020-01-01 → 2026-01-17 |
| Win rate (raw) | 46.4% |

### 🎯 Entry Filter Accuracy

**AUC Score: 0.654**

| Threshold | Accuracy | Precision | Signals |
|-----------|----------|-----------|---------|
| 0.50 | 61.4% | 60.8% | 3,916 |
| 0.60 | 59.2% | 73.8% | 1,239 |
| 0.65 | 57.6% | 79.7% | 689 |
| 0.70 | 56.2% | 84.8% | 361 |
| 0.75 | 55.4% | 90.5% | 189 |

### 📉 SL Predictor Accuracy

| Metric | Value |
|--------|-------|
| SL thực tế (trung bình) | 8.38% |
| SL thực tế (std dev) | ±3.00% |
| SL range | 0.75% - 15.00% |
| **MAE (sai số tuyệt đối)** | **0.95%** |
| Sai số tương đối | 20.8% |

### 📈 TP Predictor Accuracy

| Metric | Value |
|--------|-------|
| TP thực tế (trung bình) | 18.27% |
| TP thực tế (std dev) | ±7.04% |
| TP range | 1.50% - 30.00% |
| **MAE (sai số tuyệt đối)** | **1.89%** |
| Sai số tương đối | 19.3% |


---
## ⏱️ Timeframe: 1D

### 📈 Thống Kê Data

| Metric | Value |
|--------|-------|
| Tổng rows | 360,864 |
| Crossover signals | 27,603 |
| Số symbols | 408 |
| Thời gian | 2020-01-01 → 2026-01-17 |
| Win rate (raw) | 45.3% |

### 🎯 Entry Filter Accuracy

**AUC Score: 0.690**

| Threshold | Accuracy | Precision | Signals |
|-----------|----------|-----------|---------|
| 0.50 | 64.4% | 61.8% | 2,073 |
| 0.60 | 63.0% | 74.5% | 831 |
| 0.65 | 60.8% | 78.9% | 497 |
| 0.70 | 59.1% | 83.4% | 289 |
| 0.75 | 57.5% | 85.1% | 148 |

### 📉 SL Predictor Accuracy

| Metric | Value |
|--------|-------|
| SL thực tế (trung bình) | 10.11% |
| SL thực tế (std dev) | ±3.27% |
| SL range | 0.75% - 15.00% |
| **MAE (sai số tuyệt đối)** | **1.14%** |
| Sai số tương đối | 30.6% |

### 📈 TP Predictor Accuracy

| Metric | Value |
|--------|-------|
| TP thực tế (trung bình) | 23.53% |
| TP thực tế (std dev) | ±7.05% |
| TP range | 1.50% - 30.00% |
| **MAE (sai số tuyệt đối)** | **2.09%** |
| Sai số tương đối | 23.9% |


---
## 💡 Khuyến Nghị Khi Vào Lệnh

### Entry Filter
- Chỉ vào lệnh khi **confidence ≥ 0.65** (tối thiểu)
- Confidence **0.70-0.75** cho trades an toàn hơn
- Confidence càng cao → Win rate càng cao, nhưng ít signals hơn

### Stop Loss
- Model dự đoán SL với sai số khoảng **0.5-1%**
- Nên đặt SL = **SL dự đoán + buffer 0.3-0.5%** để an toàn
- Hoặc dùng trailing SL sau khi profit

### Take Profit
- Model dự đoán TP với sai số khoảng **1-2%**
- Có thể partial TP: **50% tại TP/2, 50% tại TP**
- Hoặc trailing TP nếu momentum mạnh

### Leverage
- **1x**: An toàn nhất, drawdown thấp
- **5x**: Cân bằng risk/reward, recommended
- **7x**: Aggressive, chỉ dùng với confidence cao (≥0.70)

### Position Sizing
- **Fixed $1000**: Đơn giản, MaxDD thấp hơn
- **% Equity**: Return cao hơn, nhưng DD cũng cao hơn
- Max **10 positions** cùng lúc
- Không mở 2 lệnh cùng 1 symbol


---
## 📋 Bảng Tổng Hợp Nhanh

| Timeframe | Entry AUC | SL MAE | TP MAE | Recommend Threshold |
|-----------|-----------|--------|--------|---------------------|
| 4h | 0.612 | 0.59% | 1.21% | 0.65 |
| 8h | 0.643 | 0.78% | 1.60% | 0.65 |
| 12h | 0.654 | 0.95% | 1.89% | 0.65 |
| 1d | 0.690 | 1.14% | 2.09% | 0.65 |
