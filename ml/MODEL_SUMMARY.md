# 📊 ML Model Summary - MACD Crossover Strategy

> Generated: 2026-03-02 21:47

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
| Tổng rows | 1,690,069 |
| Crossover signals | 131,331 |
| Số symbols | 513 |
| Thời gian | 2020-01-01 → 2026-03-02 |
| Win rate (raw) | 42.8% |

### 🎯 Entry Filter Accuracy

**AUC Score: 0.624**

| Threshold | Accuracy | Precision | Signals |
|-----------|----------|-----------|---------|
| 0.50 | 60.5% | 60.2% | 3,837 |
| 0.60 | 58.2% | 77.8% | 334 |
| 0.65 | 57.7% | 87.5% | 72 |
| 0.70 | 57.6% | 93.3% | 15 |
| 0.75 | 57.5% | 100.0% | 4 |

### 📉 SL Predictor Accuracy

| Metric | Value |
|--------|-------|
| SL thực tế (trung bình) | 4.22% |
| SL thực tế (std dev) | ±2.46% |
| SL range | 1.00% - 14.99% |
| **MAE (sai số tuyệt đối)** | **1.39%** |
| Sai số tương đối | 63.8% |

### 📈 TP Predictor Accuracy

| Metric | Value |
|--------|-------|
| TP thực tế (trung bình) | 20.25% |
| TP thực tế (std dev) | ±1.35% |
| TP range | 20.00% - 30.00% |
| **MAE (sai số tuyệt đối)** | **0.19%** |
| Sai số tương đối | 0.9% |


---
## ⏱️ Timeframe: 8H

### 📈 Thống Kê Data

| Metric | Value |
|--------|-------|
| Tổng rows | 834,943 |
| Crossover signals | 64,523 |
| Số symbols | 457 |
| Thời gian | 2020-01-01 → 2026-03-02 |
| Win rate (raw) | 43.5% |

### 🎯 Entry Filter Accuracy

**AUC Score: 0.829**

| Threshold | Accuracy | Precision | Signals |
|-----------|----------|-----------|---------|
| 0.50 | 75.1% | 75.4% | 4,627 |
| 0.60 | 71.8% | 84.8% | 2,757 |
| 0.65 | 68.9% | 87.8% | 2,040 |
| 0.70 | 65.8% | 90.7% | 1,405 |
| 0.75 | 62.5% | 92.4% | 852 |

### 📉 SL Predictor Accuracy

| Metric | Value |
|--------|-------|
| SL thực tế (trung bình) | 5.71% |
| SL thực tế (std dev) | ±3.05% |
| SL range | 1.00% - 15.00% |
| **MAE (sai số tuyệt đối)** | **1.75%** |
| Sai số tương đối | 62.6% |

### 📈 TP Predictor Accuracy

| Metric | Value |
|--------|-------|
| TP thực tế (trung bình) | 20.78% |
| TP thực tế (std dev) | ±2.31% |
| TP range | 20.00% - 30.00% |
| **MAE (sai số tuyệt đối)** | **0.54%** |
| Sai số tương đối | 2.4% |


---
## ⏱️ Timeframe: 12H

### 📈 Thống Kê Data

| Metric | Value |
|--------|-------|
| Tổng rows | 556,262 |
| Crossover signals | 41,866 |
| Số symbols | 455 |
| Thời gian | 2020-01-02 → 2026-03-02 |
| Win rate (raw) | 45.4% |

### 🎯 Entry Filter Accuracy

**AUC Score: 0.823**

| Threshold | Accuracy | Precision | Signals |
|-----------|----------|-----------|---------|
| 0.50 | 75.1% | 74.4% | 3,272 |
| 0.60 | 72.3% | 83.3% | 2,047 |
| 0.65 | 69.5% | 87.1% | 1,516 |
| 0.70 | 66.5% | 90.1% | 1,095 |
| 0.75 | 63.2% | 92.6% | 704 |

### 📉 SL Predictor Accuracy

| Metric | Value |
|--------|-------|
| SL thực tế (trung bình) | 6.77% |
| SL thực tế (std dev) | ±3.46% |
| SL range | 1.00% - 15.00% |
| **MAE (sai số tuyệt đối)** | **2.09%** |
| Sai số tương đối | 67.2% |

### 📈 TP Predictor Accuracy

| Metric | Value |
|--------|-------|
| TP thực tế (trung bình) | 21.72% |
| TP thực tế (std dev) | ±3.22% |
| TP range | 20.00% - 30.00% |
| **MAE (sai số tuyệt đối)** | **0.96%** |
| Sai số tương đối | 4.2% |


---
## ⏱️ Timeframe: 1D

### 📈 Thống Kê Data

| Metric | Value |
|--------|-------|
| Tổng rows | 268,334 |
| Crossover signals | 20,432 |
| Số symbols | 396 |
| Thời gian | 2020-01-03 → 2026-03-02 |
| Win rate (raw) | 43.8% |

### 🎯 Entry Filter Accuracy

**AUC Score: 0.823**

| Threshold | Accuracy | Precision | Signals |
|-----------|----------|-----------|---------|
| 0.50 | 75.6% | 73.3% | 1,587 |
| 0.60 | 73.9% | 83.0% | 1,020 |
| 0.65 | 71.1% | 86.2% | 773 |
| 0.70 | 68.1% | 90.1% | 543 |
| 0.75 | 65.1% | 94.3% | 351 |

### 📉 SL Predictor Accuracy

| Metric | Value |
|--------|-------|
| SL thực tế (trung bình) | 8.40% |
| SL thực tế (std dev) | ±3.89% |
| SL range | 1.00% - 15.00% |
| **MAE (sai số tuyệt đối)** | **2.62%** |
| Sai số tương đối | 77.1% |

### 📈 TP Predictor Accuracy

| Metric | Value |
|--------|-------|
| TP thực tế (trung bình) | 24.81% |
| TP thực tế (std dev) | ±4.28% |
| TP range | 20.00% - 30.00% |
| **MAE (sai số tuyệt đối)** | **1.51%** |
| Sai số tương đối | 6.2% |


---
## ⏱️ Timeframe: 1W

### 📈 Thống Kê Data

| Metric | Value |
|--------|-------|
| Tổng rows | 8,574 |
| Crossover signals | 545 |
| Số symbols | 34 |
| Thời gian | 2020-01-19 → 2026-03-08 |
| Win rate (raw) | 41.1% |

### 🎯 Entry Filter Accuracy

**AUC Score: 0.786**

| Threshold | Accuracy | Precision | Signals |
|-----------|----------|-----------|---------|
| 0.50 | 75.2% | 67.5% | 40 |
| 0.60 | 70.6% | 69.6% | 23 |
| 0.65 | 65.1% | 58.8% | 17 |
| 0.70 | 64.2% | 58.3% | 12 |
| 0.75 | 63.3% | 60.0% | 5 |

### 📉 SL Predictor Accuracy

| Metric | Value |
|--------|-------|
| SL thực tế (trung bình) | 7.91% |
| SL thực tế (std dev) | ±4.13% |
| SL range | 1.00% - 14.91% |
| **MAE (sai số tuyệt đối)** | **3.80%** |
| Sai số tương đối | 97.8% |

### 📈 TP Predictor Accuracy

| Metric | Value |
|--------|-------|
| TP thực tế (trung bình) | 29.35% |
| TP thực tế (std dev) | ±2.31% |
| TP range | 20.00% - 30.00% |
| **MAE (sai số tuyệt đối)** | **0.70%** |
| Sai số tương đối | 3.0% |


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
| 4h | 0.624 | 1.39% | 0.19% | 0.65 |
| 8h | 0.829 | 1.75% | 0.54% | 0.65 |
| 12h | 0.823 | 2.09% | 0.96% | 0.65 |
| 1d | 0.823 | 2.62% | 1.51% | 0.65 |
| 1w | 0.786 | 3.80% | 0.70% | 0.65 |
