import pandas as pd
import numpy as np
from pathlib import Path
import os

def scan_historical_confluence(df, horizon=12):
    """
    Quét toàn bộ lịch sử để tìm các điểm Hợp Lưu Breakout kinh điển.
    Kèm theo 'max_pump' để ông tự Backtest bằng mắt xem sau đó nó bay bao nhiêu %.
    """
    print(f"🔍 Đang rà soát {len(df):,} dòng dữ liệu...")
    df = df.copy()
    
    # 1. TÌM CẢN (Resistance): Mức giá cao nhất trong 50 nến trước đó
    df['resistance_50'] = df.groupby('symbol')['high'].transform(lambda x: x.rolling(50).max().shift(1))
    
    # 2. XÁC NHẬN "ĐÁY TĂNG DẦN" (Higher Lows / Uptrend Structure): 
    # Thay vì đếm đáy phức tạp, ta dùng EMA: EMA20 > EMA50 và Giá đang bám trên EMA20
    df['ema_20'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=20).mean())
    df['ema_50'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=50).mean())
    cond_uptrend = (df['ema_20'] > df['ema_50']) & (df['low'] > df['ema_50'])
    
    # 3. RSI ĐỘNG LƯỢNG (Momentum):
    if 'rsi_14' not in df.columns:
        # Nếu chưa có RSI thì tính nhanh
        delta = df.groupby('symbol')['close'].diff()
        gain = (delta.where(delta > 0, 0)).groupby(df['symbol']).rolling(window=14).mean().reset_index(0, drop=True)
        loss = (-delta.where(delta < 0, 0)).groupby(df['symbol']).rolling(window=14).mean().reset_index(0, drop=True)
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
    cond_rsi = df['rsi_14'] > 60
    
    # 4. VOLUME BÙNG NỔ (Cú đấm quyết định):
    vol_sma_20 = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).mean().shift(1))
    cond_volume = df['volume'] > (vol_sma_20 * 2.5)
    
    # 5. BREAKOUT (Khoảnh khắc bóp cò):
    cond_breakout = df['close'] > df['resistance_50']
    
    # TỔNG HỢP HỢP LƯU: Cả 4 điều kiện phải ĐÚNG cùng 1 lúc
    df['is_golden_setup'] = cond_uptrend & cond_rsi & cond_volume & cond_breakout
    
    # --- PHẦN PHỤ TRỢ CHO VIỆC "BACKTEST BẰNG MẮT" ---
    # Tìm xem 12h sau đó giá bay được tối đa bao nhiêu %
    # Dùng groupby để tránh rò rỉ giữa các symbol
    df['future_max_high'] = df.groupby('symbol')['high'].transform(lambda x: x.shift(-horizon).rolling(horizon, min_periods=1).max())
    df['actual_pump_pct'] = (df['future_max_high'] - df['close']) / df['close']
    
    # Lọc ra CHỈ NHỮNG DÒNG CÓ TÍN HIỆU
    signals = df[df['is_golden_setup'] == True].copy()
    
    # Chọn các cột cần thiết
    cols_to_show = ['timestamp', 'symbol', 'close', 'resistance_50', 'volume', 'actual_pump_pct', 'rsi_14']
    return signals[cols_to_show]

if __name__ == "__main__":
    # Đường dẫn data
    INPUT_FILE = Path(r"d:\Code\Projects\self-projects\macd-overlay - Copy\ml\training\models\features_1h_full.parquet")
    
    if not INPUT_FILE.exists():
        print(f"❌ Không tìm thấy file dữ liệu tại {INPUT_FILE}")
        print("Vui lòng chạy train_ranker_model.txt trước để gộp feature.")
    else:
        df_all = pd.read_parquet(INPUT_FILE)
        df_signals = scan_historical_confluence(df_all)
        
        print(f"\n✅ Tìm thấy {len(df_signals)} kèo Hợp lưu (Golden Setup) trong lịch sử!")
        if not df_signals.empty:
            print("\n🔥 TOP 20 KÈO CHIẾN NHẤT (Sắp xếp theo Lợi nhuận thực tế sau 12h):")
            print(df_signals.sort_values('actual_pump_pct', ascending=False).head(20).to_string(index=False))
            
            # Thống kê nhanh
            win_rate = (df_signals['actual_pump_pct'] > 0.05).mean() * 100
            avg_pump = df_signals['actual_pump_pct'].mean() * 100
            print(f"\n--- THỐNG KÊ NHANH ---")
            print(f"Tỷ lệ chạm >5% trong 12h: {win_rate:.2f}%")
            print(f"Lợi nhuận Max trung bình: {avg_pump:.2f}%")
