import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def prove_lookahead():
    print("🧪 [PROOF OF LOOK-AHEAD BIAS]")
    
    # 1. Tạo dữ liệu 1H giả lập cho 1 ngày (Thứ 2 16/03)
    # Buổi sáng giá thấp (10.0), Buổi chiều giá bay lên (20.0)
    times = [datetime(2026, 3, 16, h) for h in range(24)]
    prices = [10.0] * 12 + [20.0] * 12 
    
    df_1h = pd.DataFrame({'timestamp': times, 'close_1h': prices})
    
    # 2. Resample sang Daily
    # Giá đóng cửa ngày Thứ 2 cuối cùng sẽ là 20.0
    df_1d = df_1h.set_index('timestamp').resample('1D').agg({'close_1h': 'last'}).reset_index()
    df_1d.rename(columns={'close_1h': 'daily_close'}, inplace=True)
    
    print(f"\n📅 Dữ liệu Daily trong database (Thứ 2): Close = {df_1d.loc[0, 'daily_close']}")
    
    # 3. Cách cũ (BỊ LEAK): Merge trực tiếp
    df_merged_biased = pd.merge_asof(
        df_1h.sort_values('timestamp'),
        df_1d.sort_values('timestamp'),
        on='timestamp',
        direction='backward'
    )
    
    row_10am = df_merged_biased[df_merged_biased['timestamp'].dt.hour == 10].iloc[0]
    print(f"\n🔍 [CÁCH CŨ - BỊ LEAK]")
    print(f"   Thời điểm: {row_10am['timestamp']}")
    print(f"   Giá hiện tại khung 1H: {row_10am['close_1h']}")
    print(f"   Giá Daily 'biết trước': {row_10am['daily_close']}  <-- SAI! Mới 10 sáng mà đã biết tối đóng nến 20.0")
    
    # 4. Cách mới (HONEST): Shift Daily lùi 1 nến
    df_1d_honest = df_1d.copy()
    df_1d_honest['daily_close'] = df_1d_honest['daily_close'].shift(1)
    
    df_merged_honest = pd.merge_asof(
        df_1h.sort_values('timestamp'),
        df_1d_honest.sort_values('timestamp'),
        on='timestamp',
        direction='backward'
    )
    
    row_10am_honest = df_merged_honest[df_merged_honest['timestamp'].dt.hour == 10].iloc[0]
    print(f"\n✅ [CÁCH MỚI - HONEST]")
    print(f"   Thời điểm: {row_10am_honest['timestamp']}")
    print(f"   Giá hiện tại khung 1H: {row_10am_honest['close_1h']}")
    print(f"   Giá Daily nhận được: {row_10am_honest['daily_close']} (NaN vì chưa có ngày Chủ Nhật giả lập)")
    print("   => Rất an toàn, chỉ dùng dữ liệu của ngày hôm TRƯỚC đã đóng nến.")

if __name__ == "__main__":
    prove_lookahead()
