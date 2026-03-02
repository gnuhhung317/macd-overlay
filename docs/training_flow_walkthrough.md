# 🧠 Quy trình Train Mô hình ML (MACD Overlay)

Hệ thống hiện tại sử dụng kiến trúc **3-Stage ML (Decision Triad)** để tối ưu hóa việc vào lệnh và quản lý rủi ro trên nhiều timeframe khác nhau.

## 1. Chuẩn bị Dữ liệu (Data Pipeline)

Quy trình bắt đầu từ dữ liệu thô và biến đổi thành bộ Features/Labels hoàn chỉnh:

*   **Dữ liệu gốc**: OHLCV 1 giờ (1h) được tải từ Binance/Bitget.
*   **Resampling**: Dữ liệu 1h được gom lại thành các timeframe mục tiêu: **4h, 8h, 12h, 1d, 1w**. (Hiện tại đang bật: 4h, 8h, 12h, 1d).
*   **Feature Engineering (Cập nhật)**: Tính toán ~84 features cho mỗi nến:
    *   **Macro Market Regime**: Tính toán xu hướng BTCUSDT trước (nằm trên/dưới SMA200, độ mạnh trend theo ADX). Sau đó broadcast (phân phối) context này sang toàn bộ các Altcoin khác. Đồng thời tính *Relative Strength* (`rs_vs_btc`) để xem Altcoin đang mạnh hay yếu hơn so với BTC.
    *   **Advanced Regime**: `dist_to_ema` (khoảng cách giá đến EMA 21/200), `trend_state` (trạng thái xu hướng kết hợp ADX & SMA), `liquidity_regime` (lọc nhiễu thanh khoản).
    *   **Xu hướng & Động lượng**: SMA/EMA, MACD, RSI, RSI Slope, Stochastic, ADX.
    *   **Biến động & Khối lượng**: ATR, Bollinger Bands, OBV, Volume Spikes.
    *   **Sentiment**: Funding Rates trung bình theo ngày.
*   **Gán nhãn (Labeling)**: Sử dụng phương pháp **Triple Barrier Method** với mục tiêu động dựa trên **ATR**:
    *   **Take Profit (TP)** = 3.0x ATR.
    *   **Stop Loss (SL)** = 1.5x ATR.
    *   **Thời gian (Time Barrier)** = 10 nến (nếu không chạm TP/SL thì đóng lệnh).
    *   **Nhãn (Label)**: 1 (Thắng - chạm TP trước) hoặc 0 (Thua - chạm SL trước hoặc timeout).

---

## 2. Quy trình Train 3 Giai đoạn (3-Stage ML)

Mô hình được chia làm 3 bước quyết định độc lập nhưng phối hợp với nhau:

### Giai đoạn 1: Entry Filter (Phân loại)
*   **File**: `ml/train_entry_filter.py`
*   **Mục tiêu**: Dự đoán xác suất một tín hiệu Crossover là "ngon" (Label 1).
*   **Xử lý**: Sử dụng XGBoost/RandomForest để lọc bỏ các nhiễu thị trường.
*   **Ngưỡng (Threshold)**: Thường đặt ở mức **0.63 - 0.65** (chỉ vào lệnh khi mô hình tự tin trên 63%).

### Giai đoạn 2: SL Predictor (Hồi quy)
*   **File**: `ml/training/train_sl.py`
*   **Mục tiêu**: Dự đoán mức Stop Loss tối ưu cho từng lệnh cụ thể.
*   **Thước đo đánh giá**: MAE (Mean Absolute Error) và **IC (Information Coefficient - Spearman Rank Correlation)** để đo độ ổn định của dự đoán so với mức SL thực tế.

### Giai đoạn 3: TP Predictor (Hồi quy)
*   **File**: `ml/training/train_tp.py`
*   **Mục tiêu**: Dự đoán mức Take Profit hoặc tỷ lệ Risk/Reward (RR) tiềm năng.
*   **Thước đo đánh giá**: MAE và **IC (Information Coefficient)**. Đảm bảo mô hình bắt đúng tương quan rank của các lệnh có lợi nhuận cao.

---

## 3. Tích hợp và Thực thi

Tất cả các mô hình được hợp nhất trong lớp `ThreeStageMLSystem` (`ml/three_stage_ml.py`):

1.  **Lọc**: Kiểm tra `Entry Filter` (Confidence >= Threshold).
2.  **Tính toán rủi ro**: Dự đoán `SL %`.
3.  **Tính toán lợi nhuận**: Dự đoán `TP %`.
4.  **Kiểm tra tỷ lệ RR**: Lệnh chỉ được thực hiện nếu `TP / SL >= 1.0` (tỷ lệ lợi nhuận/rủi ro tối thiểu).
5.  **Position Sizing**: Có thể tùy chỉnh size lệnh dựa trên độ tự tin của mô hình và tỷ lệ RR.

## 4. Các Script Điều phối chính

*   `ml/sync_and_rebuild.py`: Tải dữ liệu mới nhất và xây dựng lại dataset cho tất cả timeframe.
*   `ml/multi_timeframe_pipeline.py`: Xây dựng tập features/labels cho từng timeframe riêng biệt. (Dùng ATR theo mặc định).
*   `ml/run_pipeline.py`: Script chạy end-to-end cho 1 symbol, sử dụng các giá trị TP/SL cố định (3% / 1.5%) làm cơ sở hoặc fallback.
*   `ml/cli.py`: Giao diện dòng lệnh chính để quản lý toàn bộ vòng đời của mô hình.
*   `ml/MODEL_SUMMARY.md`: Báo cáo kết quả độ chính xác (AUC, MAE) của các mô hình đã train.

---

## 5. Cơ chế Fallback và CLI

Theo trí nhớ của bạn, hệ thống đúng là có các cơ chế fallback quan trọng giữa việc dùng **ATR (động)** và **TP/SL (cố định)**:

### Cách chạy CLI để Train
Bạn có thể sử dụng `ml/cli.py` để thực hiện các tác vụ:
```bash
# Prepare data (tạo nhãn ATR mặc định)
python ml/cli.py prepare all

# Train models (tự động nhận diện nhãn có sẵn)
python ml/cli.py train all

# Chạy pipeline đầy đủ
python ml/cli.py full 4h
```

### Cơ chế Fallback TP/SL
Hệ thống được thiết kế để linh hoạt trong việc huấn luyện:

1.  **Khi Gán nhãn (Labeling)**: Trong `ml/data_pipeline.py`, nếu tham số `use_atr=False` được truyền vào hoặc dữ liệu ATR bị thiếu, hệ thống sẽ tự động sử dụng các giá trị cố định (**20% TP, 10% SL**). Điều này đảm bảo nhãn huấn luyện luôn nhất quán với mục tiêu lợi nhuận cao.
2.  **Khi Huấn luyện (Training)**: Trong `ml/training/train_sl.py` và `train_tp.py`, hệ thống sẽ ưu tiên cột `sl_pct_used` (nhãn ATR động). Nếu không tìm thấy, nó sẽ fallback về cột `actual_sl` (nhãn cố định).
3.  **Khi Dự đoán (Inference)**: Trong `ml/three_stage_ml.py`, nếu các mô hình SL/TP chưa được train hoặc không load được, `ThreeStageMLSystem` sẽ sử dụng các giá trị `default_sl` (10%) và `default_tp` (20%) để đảm bảo bot vẫn hoạt động đúng kỳ vọng.

---

## 6. Đánh giá Đa sàn (Cross-Exchange Evaluation)

Để đảm bảo mô hình không bị "overfitting" (học vẹt) với dữ liệu của duy nhất một sàn, pipeline huấn luyện tích hợp cơ chế tự động **Cross-Exchange Robustness Test**:

1. **Huấn luyện (Training)**: Mô hình thực hiện quá trình train và grid/random search (nếu gõ `tune`) trên bộ dữ liệu chính của **Binance** (lấy từ thư mục `data/`).
2. **Kiểm định (Evaluation)**: Ngay sau khi tìm ra mô hình tốt nhất (Best Model) cho Binance, hàm `evaluate_on_exchanges()` (`ml/training/training_utils.py`) sẽ được gọi để tự động đem mô hình đó sang test trên tập dữ liệu hoàn toàn chưa từng thấy của **Bitget** (`bitget-data/`) và **Bybit** (`bybit-data/`).
3. **Chỉ số ghi nhận**: Nó xuất ra đầy đủ số điểm AUC (đối với Entry Filter) và MAE / IC (đối với SL/TP Predictors) của các tín hiệu thuộc Bitget và Bybit. Nếu chênh lệch (degradation) chỉ trong khoảng 5%, bạn có một mô hình thực sự thấu hiểu quy luật thị trường, sẵn sàng triển khai trên bất kỳ sàn nào.
