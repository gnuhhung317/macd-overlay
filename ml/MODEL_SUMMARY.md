# 📊 ML Model Summary - MACD Crossover Strategy

> Generated: 2026-02-21 12:30

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
| Tổng rows | 163,100 |
| Crossover signals | 12,673 |
| Số symbols | 49 |
| Thời gian | 2021-01-01 → 2026-02-16 |
| Win rate (raw) | 42.3% |

### 🎯 Entry Filter Accuracy

**AUC Score: 0.675**

| Threshold | Accuracy | Precision | Signals |
|-----------|----------|-----------|---------|
| 0.50 | 64.1% | 67.5% | 418 |
| 0.60 | 59.9% | 93.5% | 46 |
| 0.65 | 58.8% | 100.0% | 13 |
| 0.70 | 58.4% | 100.0% | 3 |
| 0.75 | 58.3% | 100.0% | 1 |

### 📉 SL Predictor Accuracy

| Metric | Value |
|--------|-------|
| SL thực tế (trung bình) | 4.44% |
| SL thực tế (std dev) | ±2.54% |
| SL range | 1.00% - 14.98% |
| **MAE (sai số tuyệt đối)** | **1.45%** |
| Sai số tương đối | 65.5% |

### 📈 TP Predictor Accuracy

| Metric | Value |
|--------|-------|
| TP thực tế (trung bình) | 20.30% |
| TP thực tế (std dev) | ±1.46% |
| TP range | 20.00% - 30.00% |
| **MAE (sai số tuyệt đối)** | **0.23%** |
| Sai số tương đối | 1.0% |


---
## ⏱️ Timeframe: 8H

### 📈 Thống Kê Data

| Metric | Value |
|--------|-------|
| Tổng rows | 817,161 |
| Crossover signals | 63,096 |
| Số symbols | 455 |
| Thời gian | 2020-01-01 → 2026-02-16 |
| Win rate (raw) | 44.0% |

### 🎯 Entry Filter Accuracy

**AUC Score: 0.647**

| Threshold | Accuracy | Precision | Signals |
|-----------|----------|-----------|---------|
| 0.50 | 61.0% | 59.4% | 3,320 |
| 0.60 | 58.3% | 70.8% | 675 |
| 0.65 | 57.3% | 74.9% | 303 |
| 0.70 | 56.8% | 83.8% | 136 |
| 0.75 | 56.6% | 89.7% | 78 |

### 📉 SL Predictor Accuracy

| Metric | Value |
|--------|-------|
| SL thực tế (trung bình) | 5.79% |
| SL thực tế (std dev) | ±3.10% |
| SL range | 1.00% - 15.00% |
| **MAE (sai số tuyệt đối)** | **1.99%** |
| Sai số tương đối | 81.9% |

### 📈 TP Predictor Accuracy

| Metric | Value |
|--------|-------|
| TP thực tế (trung bình) | 20.82% |
| TP thực tế (std dev) | ±2.33% |
| TP range | 20.00% - 30.00% |
| **MAE (sai số tuyệt đối)** | **0.56%** |
| Sai số tương đối | 2.5% |


---
## ⏱️ Timeframe: 12H

### 📈 Thống Kê Data

| Metric | Value |
|--------|-------|
| Tổng rows | 544,865 |
| Crossover signals | 41,140 |
| Số symbols | 455 |
| Thời gian | 2020-01-01 → 2026-02-16 |
| Win rate (raw) | 45.7% |

### 🎯 Entry Filter Accuracy

**AUC Score: 0.656**

| Threshold | Accuracy | Precision | Signals |
|-----------|----------|-----------|---------|
| 0.50 | 61.3% | 59.2% | 2,738 |
| 0.60 | 60.1% | 75.8% | 794 |
| 0.65 | 58.5% | 83.7% | 405 |
| 0.70 | 56.8% | 87.9% | 173 |
| 0.75 | 55.7% | 91.2% | 57 |

### 📉 SL Predictor Accuracy

| Metric | Value |
|--------|-------|
| SL thực tế (trung bình) | 6.85% |
| SL thực tế (std dev) | ±3.50% |
| SL range | 1.00% - 15.00% |
| **MAE (sai số tuyệt đối)** | **2.44%** |
| Sai số tương đối | 92.8% |

### 📈 TP Predictor Accuracy

| Metric | Value |
|--------|-------|
| TP thực tế (trung bình) | 21.80% |
| TP thực tế (std dev) | ±3.24% |
| TP range | 20.00% - 30.00% |
| **MAE (sai số tuyệt đối)** | **0.96%** |
| Sai số tương đối | 4.2% |


---
## ⏱️ Timeframe: 1D

### 📈 Thống Kê Data

| Metric | Value |
|--------|-------|
| Tổng rows | 263,723 |
| Crossover signals | 20,606 |
| Số symbols | 396 |
| Thời gian | 2020-01-01 → 2026-02-16 |
| Win rate (raw) | 44.4% |

### 🎯 Entry Filter Accuracy

**AUC Score: 0.699**

| Threshold | Accuracy | Precision | Signals |
|-----------|----------|-----------|---------|
| 0.50 | 65.6% | 62.8% | 1,331 |
| 0.60 | 63.5% | 74.2% | 524 |
| 0.65 | 61.6% | 81.0% | 284 |
| 0.70 | 59.8% | 87.1% | 139 |
| 0.75 | 58.3% | 87.0% | 54 |

### 📉 SL Predictor Accuracy

| Metric | Value |
|--------|-------|
| SL thực tế (trung bình) | 8.43% |
| SL thực tế (std dev) | ±3.89% |
| SL range | 1.00% - 15.00% |
| **MAE (sai số tuyệt đối)** | **3.15%** |
| Sai số tương đối | 112.1% |

### 📈 TP Predictor Accuracy

| Metric | Value |
|--------|-------|
| TP thực tế (trung bình) | 24.79% |
| TP thực tế (std dev) | ±4.24% |
| TP range | 20.00% - 30.00% |
| **MAE (sai số tuyệt đối)** | **1.44%** |
| Sai số tương đối | 5.9% |


---
## ⏱️ Timeframe: 1W

### 📈 Thống Kê Data

| Metric | Value |
|--------|-------|
| Tổng rows | 8,574 |
| Crossover signals | 575 |
| Số symbols | 34 |
| Thời gian | 2020-01-05 → 2026-02-22 |
| Win rate (raw) | 43.4% |

### 🎯 Entry Filter Accuracy

**AUC Score: 0.712**

| Threshold | Accuracy | Precision | Signals |
|-----------|----------|-----------|---------|
| 0.50 | 65.2% | 54.4% | 57 |
| 0.60 | 69.6% | 67.9% | 28 |
| 0.65 | 62.6% | 55.6% | 18 |
| 0.70 | 61.7% | 55.6% | 9 |
| 0.75 | 61.7% | 60.0% | 5 |

### 📉 SL Predictor Accuracy

| Metric | Value |
|--------|-------|
| SL thực tế (trung bình) | 7.55% |
| SL thực tế (std dev) | ±4.21% |
| SL range | 1.00% - 14.89% |
| **MAE (sai số tuyệt đối)** | **3.14%** |
| Sai số tương đối | 94.8% |

### 📈 TP Predictor Accuracy

| Metric | Value |
|--------|-------|
| TP thực tế (trung bình) | 30.74% |
| TP thực tế (std dev) | ±3.53% |
| TP range | 20.00% - 42.00% |
| **MAE (sai số tuyệt đối)** | **0.34%** |
| Sai số tương đối | 1.5% |


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
| 4h | 0.675 | 1.45% | 0.23% | 0.65 |
| 8h | 0.647 | 1.99% | 0.56% | 0.65 |
| 12h | 0.656 | 2.44% | 0.96% | 0.65 |
| 1d | 0.699 | 3.15% | 1.44% | 0.65 |
| 1w | 0.712 | 3.14% | 0.34% | 0.65 |
