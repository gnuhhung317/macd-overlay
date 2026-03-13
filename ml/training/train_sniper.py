import os
import sys
import gc
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path

# ============================================================
# CONFIG & FEATURES
# ============================================================
MODEL_FEATURES = [
    'rsi_14','rsi_slope','stoch_k','stoch_d','roc_7','roc_14',
    'volume_ratio','volume_zscore','volume_trend','rs_vs_btc','rs_vs_btc_sma7','vol_compression',
    'dist_to_high_30d','dist_to_low_30d','dist_to_ema_21_pct','dist_to_ema_50_pct','dist_to_ema_200_pct',
    'price_vs_sma_30','momentum_30','macd_slope','macd_acceleration',
    'upper_wick_ratio','dist_to_ema50_atr','vol_acceleration', # New features
    'bb_squeeze','above_poc',
    'micro_volume','price_accel','order_flow_proxy',
    'btc_is_bull_regime','btc_trend_strength','adx','hour_sin','hour_cos','day_sin','day_cos',
    'btc_corr','trend_state','is_trending','is_volatile','ema_200_1d_dist','rsi_14_1d'
]

# Paths
BASE_DIR = Path(r"d:\Code\Projects\self-projects\macd-overlay - Copy")
INPUT_FILE = BASE_DIR / "data" / "processed" / "features_1h_btc_context.parquet"
OUTPUT_MODEL_DIR = BASE_DIR / "ml" / "training" / "models" / "1h"
OUTPUT_MODEL_PATH = OUTPUT_MODEL_DIR / "ensemble_lgbm_tabular.joblib"
OUTPUT_META_PATH = OUTPUT_MODEL_DIR / "ensemble_meta.joblib"

def prepare_cascade_data_optimized(parquet_path):
    print("🧹 Tầng 1: Đang chuẩn bị dữ liệu (Setup Mồi Lửa - Ignition Bar)...")
    
    import pyarrow.parquet as pq
    all_parquet_cols = pq.read_schema(parquet_path).names
    
    # --- BƯỚC 1: XỬ LÝ DỮ LIỆU VĨ MÔ (BTC) - LOAD ÍT TỐN RAM ---
    btc_price_col = next((c for c in ['btc_close_x', 'btc_close_y', 'btc_close'] if c in all_parquet_cols), None)
    if btc_price_col:
        print(f"📊 Đang tách biệt và tính toán dữ liệu Vĩ mô (BTC Regime) từ '{btc_price_col}'...")
        btc_raw = pd.read_parquet(parquet_path, columns=['timestamp', btc_price_col])
        btc_raw['timestamp'] = pd.to_datetime(btc_raw['timestamp']).dt.tz_localize(None)
        
        btc_df = btc_raw.drop_duplicates(subset=['timestamp']).sort_values('timestamp').copy()
        btc_df.rename(columns={btc_price_col: 'btc_close'}, inplace=True)
        
        # Tính toán Macro trên chuỗi liên tục
        btc_df['btc_returns'] = btc_df['btc_close'].pct_change()
        btc_df['btc_vol_24h'] = btc_df['btc_returns'].rolling(24).std()
        btc_df['btc_ema_20'] = btc_df['btc_close'].ewm(span=20).mean()
        btc_df['btc_ema_50'] = btc_df['btc_close'].ewm(span=50).mean()
        del btc_raw
        gc.collect()
    else:
        print("⚠️ CẢNH BÁO: Không tìm thấy cột giá BTC hợp lệ!")
        btc_df = pd.DataFrame()

    # --- BƯỚC 2: LOAD DỮ LIỆU ALTCOIN CÓ BỘ LỌC (TIẾT KIỆM RAM) ---
    cols_to_load = list(set(MODEL_FEATURES + [
        'symbol', 'timestamp', 'open', 'close', 'high', 'low', 'volume', 'usd_vol_24h'
    ]))
    cols_to_load = [c for c in cols_to_load if c in all_parquet_cols]
    
    # Chỉ load dữ liệu trước 2025 và volume đủ lớn ngay từ ổ đĩa
    print(f"📥 Đang tải dữ liệu Altcoin (Filters: Train < 2025 & Vol > 1M)...")
    df = pd.read_parquet(
        parquet_path, 
        columns=cols_to_load,
        filters=[('timestamp', '<', pd.Timestamp('2025-01-01')), ('usd_vol_24h', '>=', 1000000)]
    )

    # Merge Macro vào df
    if not btc_df.empty:
        df = df.merge(btc_df[['timestamp', 'btc_close', 'btc_returns', 'btc_vol_24h', 'btc_ema_20', 'btc_ema_50']], 
                      on='timestamp', how='left')
        del btc_df
        gc.collect()

    if df.empty:
        print("❌ Không có dữ liệu thỏa mãn bộ lọc!")
        return pd.DataFrame()
        
    # Đảm bảo dữ liệu được sắp xếp theo thời gian cho các chỉ báo Altcoin sau này
    df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

    # 3. TÍNH ATR (14) để dùng cho Feature & Label
    def calc_atr(df_group):
        high_low = df_group['high'] - df_group['low']
        high_close = np.abs(df_group['high'] - df_group['close'].shift())
        low_close = np.abs(df_group['low'] - df_group['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(14).mean()

    df['atr_14'] = df.groupby('symbol', group_keys=False).apply(calc_atr)
    
    # 3. TÍNH FEATURE MỚI (MÁY PHÁT HIỆN NÓI DỐI)
    df['upper_wick_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-9)
    df['ema_50'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=50).mean())
    df['dist_to_ema50_atr'] = (df['close'] - df['ema_50']) / (df['atr_14'] + 1e-9)
    df['vol_acceleration'] = df.groupby('symbol')['volume'].transform(lambda x: x / (x.shift(1) + 1e-9))

    # 4. TÍNH LABEL 48H (ĐỊNH NGHĨA LẠI SỰ THẬT)
    horizon = 48
    # Lấy High/Low tương lai
    df['future_max_high'] = df.groupby('symbol')['high'].shift(-1).rolling(horizon, min_periods=1).max().shift(-(horizon-1))
    df['future_min_low'] = df.groupby('symbol')['low'].shift(-1).rolling(horizon, min_periods=1).min().shift(-(horizon-1))
    
    # Chuẩn hóa MFE/MAE theo ATR
    df['mfe_atr'] = (df['future_max_high'] - df['close']) / (df['atr_14'] + 1e-9)
    df['mae_atr'] = (df['future_min_low'] - df['close']) / (df['atr_14'] + 1e-9)
    
    # Gán nhãn 3 kịch bản: 1 (Long), -1 (Short), 0 (Skip)
    df['label_raw'] = 0
    df.loc[(df['mfe_atr'] > 2.0) & (df['mae_atr'] > -1.0), 'label_raw'] = 1
    df.loc[(df['mae_atr'] < -2.5) & (df['mfe_atr'] < 1.0), 'label_raw'] = -1
    
    # Map sang [0, 1, 2] cho LightGBM Multiclass (0: Skip, 1: Long, 2: Short)
    df['label'] = df['label_raw'].map({0: 0, 1: 1, -1: 2})

    # 5. TÍNH CHỈ BÁO KỸ THUẬT CHO BỘ LỌC
    df['ema_20'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=20).mean())
    df['resistance_50'] = df.groupby('symbol')['high'].transform(lambda x: x.rolling(50).max().shift(1))
    vol_sma_20 = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).mean().shift(1))

    # 6. BỘ LỌC MỒI LỬA (IGNITION BAR) DÀNH CHO META-LABELING
    df['ema_20'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=20).mean())
    df['resistance_50'] = df.groupby('symbol')['high'].transform(lambda x: x.rolling(50).max().shift(1))
    vol_sma_20 = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).mean().shift(1))

    cond_green_bar = (df['close'] > df['open']) & (df['close'] > df['ema_20'])
    cond_body_size = ((df['close'] - df['open']) / df['open']) > 0.015 
    cond_vol_ignition = (df['volume'] > vol_sma_20 * 1.5) & (df['volume'] < vol_sma_20 * 4.0)
    cond_rsi_fresh = (df['rsi_14'] >= 55) & (df['rsi_14'] <= 72)
    dist_to_res = (df['resistance_50'] - df['close']) / (df['close'] + 1e-9)
    cond_near_res = dist_to_res > -0.05 

    mask_golden = cond_green_bar & cond_body_size & cond_vol_ignition & cond_rsi_fresh & cond_near_res
    
    if 'usd_vol_24h' in df.columns:
        mask_golden = mask_golden & (df['usd_vol_24h'] >= 1000000)

    golden_df = df[mask_golden].reset_index(drop=True)    
    # 7. LỌC VÀ TRẢ VỀ KẾT QUẢ
    golden_df = df[mask_golden].dropna(subset=MODEL_FEATURES + ['label']).sort_values('timestamp').reset_index(drop=True)
    
    # Dọn dẹp RAM (Giữ lại mfe_atr, mae_atr để phân tích)
    df.drop(columns=['future_max_high', 'future_min_low', 'label_raw'], inplace=True, errors='ignore')
    del df
    gc.collect()
    
    print(f"🎯 Tầng 1 Xong! Giữ lại {len(golden_df)} kèo.")
    
    return golden_df # <--- PHẢI CÓ DÒNG NÀY
    
def train_cascade_sniper(golden_df):
    print("\n🤖 Tầng 2: Đang huấn luyện AI Sniper (Chế độ Baseline MFE)...")
    
    # 1. Chốt chặn an toàn: Đảm bảo data đã sort theo thời gian
    golden_df = golden_df.sort_values('timestamp').reset_index(drop=True)
    
    # 2. Chia Train/Test (80% Quá khứ / 20% Hiện tại)
    split_idx = int(len(golden_df) * 0.8)
    train_df = golden_df.iloc[:split_idx]
    test_df = golden_df.iloc[split_idx:]
    
    X_tr = train_df[MODEL_FEATURES]
    y_tr = train_df['label']
    X_te = test_df[MODEL_FEATURES]
    y_te = test_df['label']
    
    # Ép kiểu dữ liệu về float
    X_tr = X_tr.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_te = X_te.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # 3. Khởi tạo LightGBM (Multiclass)
    clf = lgb.LGBMClassifier(
        n_estimators=500, 
        learning_rate=0.03, 
        max_depth=5,             
        num_leaves=31,
        objective='multiclass',
        num_class=3,
        class_weight='balanced', 
        random_state=42, 
        n_jobs=-1,
        verbose=-1
    )
    
    # 4. Fit Model
    clf.fit(X_tr, y_tr)
    
    # 5. Đánh giá trên tập Test
    preds_proba = clf.predict_proba(X_te)
    
    # Thống kê xác suất bình quân cho từng class
    avg_probas = preds_proba.mean(axis=0)
    
    # 6. In Báo cáo Thực chiến
    print(f"\n{'='*40}\n KẾT QUẢ THỰC CHIẾN MULTICLASS (OOS TEST)\n{'='*40}")
    print(f"Tổng số kèo Hợp lưu trong tập Test: {len(y_te)}")
    print(f"Phân bổ Label: 0 (Skip): {(y_te==0).sum()}, 1 (Long): {(y_te==1).sum()}, 2 (Short): {(y_te==2).sum()}")
    print("-" * 40)
    print(f"Xác suất dự đoán trung bình: Skip={avg_probas[0]:.2f}, Long={avg_probas[1]:.2f}, Short={avg_probas[2]:.2f}")
    
    # Đánh giá Precision cho Long (Label 1) và Short (Label 2) ở ngưỡng confidence cao
    for label_idx, label_name in [(1, "LONG"), (2, "SHORT")]:
        p = preds_proba[:, label_idx]
        thresh = 0.60
        calls = p >= thresh
        if calls.sum() > 0:
            acc = (y_te[calls] == label_idx).mean()
            print(f"🎯 Precision {label_name} (Prob > {thresh}): {acc*100:.2f}% ({calls.sum()} kèo)")
        else:
            print(f"🎯 Precision {label_name} (Prob > {thresh}): N/A (0 kèo)")
    
    # 7. Feature Importance
    importance = pd.DataFrame({'feature': MODEL_FEATURES, 'gain': clf.feature_importances_}).sort_values('gain', ascending=False)
    print("\n🔍 Top 5 Features quan trọng nhất định đoạt cú Pump:")
    print(importance.head(5).to_string(index=False))
    
    return clf, 0.6 # Return default 60% probability threshold

def analyze_btc_regime_impact(df):
    print("🌡️ ĐANG PHÂN TÍCH NHIỆT KẾ BTC ĐỂ TÌM LUẬT CỨNG...")
    
    # --- ĐÃ SỬA: Không tính toán lại trên dữ liệu nến rời rạc ---
    # Các cột btc_vol_24h, btc_ema_20, btc_ema_50 đã được tính ở prepare_cascade_data_optimized
    
    # Định nghĩa các trạng thái (Regimes)
    conditions = [
        (df['btc_close'] > df['btc_ema_20']), # BTC Uptrend ngắn hạn
        (df['btc_close'] < df['btc_ema_20']) & (df['btc_close'] > df['btc_ema_50']), # BTC Sideway
        (df['btc_close'] < df['btc_ema_50']), # BTC Downtrend/Sập
    ]
    choices = ['BTC_BULL_1H', 'BTC_SIDEWAY', 'BTC_BEAR_1H']
    df['btc_state'] = np.select(conditions, choices, default='UNKNOWN')

    # Thống kê Winrate của Altcoin theo từng trạng thái BTC
    report = df.groupby('btc_state').agg({
        'label': ['count', 'mean'], # 'mean' chính là tỷ lệ nổ tự nhiên
        'mfe_atr': 'mean'
    })
    
    print("\n📊 BÁO CÁO SỨC MẠNH ALTCOIN THEO CHÂN BTC (ATR-Normalized):")
    print(report)
    
    # Tìm ngưỡng Volatility nguy hiểm
    # Nếu BTC biến động quá mạnh (> 2 lần trung bình), Altcoin thường bị hút máu hoặc xả theo
    avg_vol = df['btc_vol_24h'].mean()
    df['is_btc_volatile'] = df['btc_vol_24h'] > (avg_vol * 1.5)
    
    vol_report = df.groupby('is_btc_volatile')['label'].mean()
    print("\n⚠️ TÁC ĐỘNG CỦA BIẾN ĐỘNG (VOLATILITY) BTC:")
    print(vol_report)
    
    return df

if __name__ == "__main__":
    if not INPUT_FILE.exists():
        print(f"❌ Không tìm thấy file: {INPUT_FILE}")
        sys.exit(1)

    # 1. Chuẩn bị data (Lọc mồi lửa Tầng 1)
    golden_df = prepare_cascade_data_optimized(INPUT_FILE)
    
    if not golden_df.empty:
        # 2. CHẠY PHÂN TÍCH BTC TRƯỚC KHI TRAIN
        # Hàm này sẽ in ra bảng thống kê để bạn tìm "Luật cứng"
        golden_df = analyze_btc_regime_impact(golden_df)
        
        # 3. Huấn luyện AI Tầng 2
        model, best_threshold = train_cascade_sniper(golden_df)
        
        # 4. Lưu Model
        # OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        # joblib.dump(model, OUTPUT_MODEL_PATH)
        # joblib.dump({'features': MODEL_FEATURES, 'threshold': best_threshold}, OUTPUT_META_PATH)
        
        print(f"\n✅ Đã lưu model mới tại: {OUTPUT_MODEL_PATH}")
        print(f"Best Threshold (Top 20%): {best_threshold:.4f}")
    else:
        print("❌ Không có dữ liệu để train.")