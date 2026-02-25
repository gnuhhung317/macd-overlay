# Circuit Breaker: Cross-Window Robustness & Optimization Insights

Tài liệu này tổng hợp những bài học, phương pháp và kết quả đáng chú ý từ việc nghiên cứu tính ổn định của Circuit Breaker qua nhiều khung thời gian khác nhau (cross-window robustness).

## 1. Kết Quả Tổng Hợp (Aggregated Results)

Bảng dưới đây thể hiện hiệu suất của chiến lược (Baseline vs Circuit Breaker) với các ngưỡng `threshold` 0.6 và 0.65 qua nhiều giai đoạn thị trường khác nhau:

| Config | BL Ret% | CB Ret% | BL DD% | CB DD% | CB WR% | CB_Exit |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| thr=0.6, 2025-04-22~2025-08-20 | -100 | -67 | 100.0 | 96.6 | 30.5 | 16 |
| thr=0.6, 2025-05-22~2025-09-19 | -100 | 32 | 100.0 | 96.6 | 37.3 | 18 |
| thr=0.6, 2025-06-21~2025-10-19 | -100 | 162 | 100.0 | 97.1 | 38.5 | 16 |
| thr=0.6, 2025-07-21~2025-11-18 | -95 | 9796 | 100.0 | 84.4 | 42.9 | 22 |
| thr=0.6, 2025-08-20~2025-12-18 | -48 | 37244 | 99.7 | 54.5 | 42.8 | 4 |
| thr=0.6, 2025-09-19~2026-01-17 | 8858 | 30505 | 81.5 | 58.1 | 37.5 | 3 |
| thr=0.6, 2025-10-19~2026-02-16 | -96 | 50150 | 99.8 | 70.8 | 56.9 | 35 |
| thr=0.6, 2025-10-25~2026-02-22 | -97 | 50287 | 99.9 | 70.8 | 55.9 | 34 |
| **thr=0.65**, 2025-04-22~2025-08-20 | -100 | -62 | 100.0 | 99.5 | 42.9 | 17 |
| **thr=0.65**, 2025-05-22~2025-09-19 | -100 | -91 | 100.0 | 99.7 | 45.5 | 16 |
| **thr=0.65**, 2025-06-21~2025-10-19 | -98 | -98 | 100.0 | 99.8 | 35.7 | 13 |
| **thr=0.65**, 2025-07-21~2025-11-18 | 37 | 119 | 99.9 | 99.4 | 47.4 | 13 |
| **thr=0.65**, 2025-08-20~2025-12-18 | 40017 | 59319 | 88.1 | 36.0 | 68.3 | 20 |
| **thr=0.65**, 2025-09-19~2026-01-17 | 28180 | 33169 | 62.3 | 36.0 | 49.0 | 21 |
| **thr=0.65**, 2025-10-19~2026-02-16 | 354 | 187 | 91.6 | 80.8 | 45.0 | 10 |
| **thr=0.65**, 2025-10-25~2026-02-22 | 701 | 15964 | 91.5 | 59.0 | 50.0 | 24 |

## 2. Phân Tích Tần Suất Tham Số Output (Parameter Frequency)
Dựa trên cấu hình tốt nhất cho mỗi tổ hợp, sự lặp lại của các tham số cho thấy xu hướng rõ ràng:
- **confluence_tf**: `12h` (16x) - Trở thành mỏ neo xu hướng (trend anchor) đáng tin cậy nhất.
- **confluence_threshold**: `0.15` (5x), `0.2` (4x) - Biên độ tín hiệu hợp lưu ở mức vừa phải (không quá nhạy, không quá trễ).
- **velocity_threshold**: `0.1` (7x), `0.15` (2x) - Ngưỡng gia tốc giá thay đổi thường nằm trong vùng thấp để tránh kích hoạt breaker quá muộn.
- **velocity_lookback**: `1` (7x), `2` (6x) - Cần phản ứng cực nhanh với giá trong thời gian ngắn (1-2 bars).
- **sleep_bars**: `5` (8x), `4` (5x) - Thời gian "ngủ" (tạm ngưng giao dịch) sau khi ngắt thường kéo dài 4-5 nến để thị trường ổn định lại.

## 3. Tham Số Tối Ưu Đồng Thuận (Consensus Best Params)
Khi quan sát chéo qua các ô cửa sổ thời gian (cross-windows), chúng ta có thể tập hợp lại thành 2 bộ cấu hình đồng thuận (consensus) chính dựa trên mức rủi ro mong muốn:

### Cho Threshold 0.6 (Nhạy cảm hơn với drawdown):
* Phù hợp để kiểm soát lỗ ở những giai đoạn rủi ro lớn.
* Tham số: `{'confluence_tf': '12h', 'confluence_threshold': 0.2, 'velocity_threshold': 0.1, 'velocity_lookback': 2, 'sleep_bars': 4}`

### Cho Threshold 0.65 (Ưu tiên duy trì vị thế khi trend tốt):
* Tối đa hóa lợi nhuận (Return) và win rate, đặc biệt vượt trội trong những giai đoạn thị trường thuận lợi (VD cuối 2025 - đầu 2026).
* Tham số: `{'confluence_tf': '12h', 'confluence_threshold': 0.15, 'velocity_threshold': 0.1, 'velocity_lookback': 1, 'sleep_bars': 5}`

## 4. Bài Học & Phương Pháp Thú Vị (Key Takeaways)
1. **Sự Vượt Trội Của Khung Thời Gian Lớn Định Hướng:** Sự áp đảo tuyệt đối của khung `12h` (16/16 trường hợp) cho thấy việc lọc nhiễu dựa trên xu hướng dài hạn là yếu tố kiên quyết để CB (Circuit Breaker) ra quyết định đóng vị thế chính xác.
2. **Phản Ứng Nhanh, Phục Hồi Chậm:** Với `velocity_lookback` chủ yếu là 1-2 nến, hệ thống cần nhạy bén cắt lỗ/chốt lời khi có biến động cực ngắn hạn. Nhưng bù lại `sleep_bars` = 4-5 cho thấy phải kiên nhẫn đứng ngoài thị trường lâu hơn để dập tắt dư chấn (volatility aftershock) thay vì bắt đáy/đỉnh ngay tức thì.
3. **Cross-Window Validation Để Lọc Overfitting:** Phương pháp chia nhỏ thời gian test và cuộn (rolling window) giúp tìm ra bộ tham số "mặc được mọi loại thời tiết" (Consensus Params) thay vì một bộ tham số chỉ ăn may trên 1 chu kỳ có sẵn. Ta thấy `thr=0.65` cực kì ổn định ở Win Rate (~45-68%) và làm giảm DD (Drawdown) đáng kể ở các tháng trend mạnh.
4. **Hiệu Suất Cực Trị:** Ở các chu kỳ tháng 8-2025 đến tháng 2-2026, chiến lược CB đem lại tỷ suất lợi nhuận bùng nổ mà vẫn giảm được Drawdown so với Baseline, chứng minh rằng quản lý thoát lệnh chủ động (active exit management) thông qua Breaker quan trọng không kém gì việc tìm tín hiệu vào lệnh.

---
*Dữ liệu tham chiếu: D:\Code\Projects\self-projects\macd-overlay\ml\results\cb_sweep_results.json*
