import pandas as pd
import numpy as np

# 1. Tạo giả lập dữ liệu Daily (d1d_feat)
d1d = pd.DataFrame({
    'timestamp': pd.to_datetime(['2026-03-21', '2026-03-22', '2026-03-23']),
    'ema_val': [100, 110, 120] # Giả sử giá đóng cửa/EMA của ngày đó
})
d1d_feat = d1d.copy()
d1d_feat['date'] = d1d_feat['timestamp'].dt.date
print("--- Dữ liệu Daily gốc ---")
print(d1d_feat[['date', 'ema_val']])

# 2. Chạy logic của bạn: drop timestamp và shift(1)
d1d_feat_shifted = d1d_feat.drop(columns='timestamp').shift(1)
print("\n--- Sau khi .shift(1) (Logic của bạn) ---")
print(d1d_feat_shifted)

# 3. Tạo giả lập dữ liệu 1H (dt) của ngày 2026-03-22
dt = pd.DataFrame({
    'timestamp': pd.to_datetime(['2026-03-22 08:00:00', '2026-03-22 15:00:00']),
    'price': [108, 109]
})
dt['date'] = dt['timestamp'].dt.date

# 4. Merge
merged = dt.merge(d1d_feat_shifted, on='date', how='left')

print("\n--- KẾT QUẢ MERGE (Checking for Leak) ---")
print(merged)

print("\nPHÂN TÍCH:")
print("Giá trị 'ema_val' mà nến 1H ngày 2026-03-22 nhận được là:", merged['ema_val'].iloc[0])
print("Giá trị EMA của ngày 2026-03-22 (tương lai) trong bảng gốc là:", d1d[d1d['timestamp'] == '2026-03-22']['ema_val'].values[0])

if merged['ema_val'].iloc[0] == d1d[d1d['timestamp'] == '2026-03-22']['ema_val'].values[0]:
    print("\n=> KẾT LUẬN: BỊ LEAK! Nến 8:00 sáng đã thấy giá trị Close của cuối ngày hôm đó.")
else:
    print("\n=> KẾT LUẬN: KHÔNG BỊ LEAK.")
