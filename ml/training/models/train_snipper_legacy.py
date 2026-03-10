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
    'lower_wick_ratio_current','upper_wick_ratio_current',
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
    btc_cols_in_file = [c for c in all_parquet_cols if c.startswith('btc_')]
    
    # 1. Load data với các cột cần thiết
    cols_to_load = list(set(MODEL_FEATURES + btc_cols_in_file + [
        'symbol', 'timestamp', 'open', 'close', 'high', 'low', 'volume', 'usd_vol_24h'
    ]))
    
    df = pd.read_parquet(parquet_path, columns=cols_to_load)
    
    # Đảm bảo có btc_returns để phân tích sau này
    if 'btc_returns' not in df.columns and 'btc_close' in df.columns:
        df['btc_returns'] = df.groupby('symbol')['btc_close'].pct_change()
    
    # 2. TÍNH LABEL 48H
    horizon = 48
    min_pump = 0.10
    df['future_max_high'] = df.groupby('symbol')['high'].shift(-1).rolling(horizon, min_periods=1).max().shift(-(horizon-1))
    df['actual_pump_pct'] = (df['future_max_high'] - df['close']) / df['close']
    df['label'] = (df['actual_pump_pct'] >= min_pump).astype(int)

    # 3. TÍNH CHỈ BÁO KỸ THUẬT CHO BỘ LỌC
    df['ema_20'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=20).mean())
    df['ema_50'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=50).mean())
    df['resistance_50'] = df.groupby('symbol')['high'].transform(lambda x: x.rolling(50).max().shift(1))
    vol_sma_20 = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).mean().shift(1))

    # 4. BỘ LỌC MỒI LỬA (IGNITION BAR)
    cond_green_bar = (df['close'] > df['open']) & (df['close'] > df['ema_20'])
    cond_body_size = ((df['close'] - df['open']) / df['open']) > 0.015 
    cond_vol_ignition = (df['volume'] > vol_sma_20 * 1.5) & (df['volume'] < vol_sma_20 * 4.0)
    cond_rsi_fresh = (df['rsi_14'] >= 55) & (df['rsi_14'] <= 72)
    dist_to_res = (df['resistance_50'] - df['close']) / df['close']
    cond_near_res = dist_to_res > -0.05 

    mask_golden = cond_green_bar & cond_body_size & cond_vol_ignition & cond_rsi_fresh & cond_near_res
    
    if 'usd_vol_24h' in df.columns:
        mask_golden = mask_golden & (df['usd_vol_24h'] >= 1000000)

    # 5. LỌC VÀ TRẢ VỀ KẾT QUẢ (QUAN TRỌNG NHẤT)
    golden_df = df[mask_golden].dropna(subset=MODEL_FEATURES + ['label']).sort_values('timestamp').reset_index(drop=True)
    
    # Dọn dẹp RAM
    df.drop(columns=['future_max_high', 'actual_pump_pct'], inplace=True, errors='ignore')
    del df
    gc.collect()
    
    print(f"🎯 Tầng 1 Xong! Giữ lại {len(golden_df)} kèo.")
    
    return golden_df # <--- PHẢI CÓ DÒNG NÀY
    
def train_cascade_sniper(golden_df):
    print("\n🤖 Tầng 2: Đang huấn luyện AI Sniper (Chế độ Baseline MFE)...")
    
    # 1. Chốt chặn an toàn: Đảm bảo data đã sort theo thời gian
    golden_df = golden_df.sort_values('timestamp').reset_index(drop=True)
    
    # 2. Chia Train/Test (80% Quá khứ / 20% Hiện tại)
    split_idx = int(len(golden_df) * 0.9)
    train_df = golden_df.iloc[:split_idx]
    test_df = golden_df.iloc[split_idx:]
    
    X_tr = train_df[MODEL_FEATURES]
    y_tr = train_df['label']
    X_te = test_df[MODEL_FEATURES]
    y_te = test_df['label']
    
    # Ép kiểu dữ liệu về float
    X_tr = X_tr.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_te = X_te.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # 3. Khởi tạo LightGBM
    clf = lgb.LGBMClassifier(
        n_estimators=300, 
        learning_rate=0.05, 
        max_depth=4,             
        class_weight='balanced', 
        random_state=42, 
        n_jobs=-1,
        verbose=-1
    )
    
    # 4. Fit Model
    clf.fit(X_tr, y_tr)
    
    # 5. Đánh giá trên tập Test
    preds_proba = clf.predict_proba(X_te)[:, 1]
    
    # Tính Precision ở ngưỡng tự tin cao (Top 20%)
    threshold = np.percentile(preds_proba, 80)
    bot_calls = preds_proba >= threshold
    
    true_wins = y_te[bot_calls].sum()
    total_calls = bot_calls.sum()
    precision = true_wins / total_calls if total_calls > 0 else 0
    
    # 6. In Báo cáo Thực chiến
    print(f"\n{'='*40}\n KẾT QUẢ THỰC CHIẾN (OOS TEST)\n{'='*40}")
    print(f"Tổng số kèo Hợp lưu trong tập Test: {len(y_te)}")
    print(f"Tỷ lệ nổ Tự nhiên (Nếu đánh mù mờ): {y_te.mean()*100:.2f}%")
    print("-" * 40)
    print(f"Ngưỡng bóp cò (Threshold Top 20%): {threshold:.4f}")
    print(f"AI đã bóp cò: {total_calls} kèo")
    print(f"Số kèo Win (>10%): {true_wins}")
    print(f"🏆 AI PRECISION (Tỷ lệ Thắng thực tế): {precision*100:.2f}%")
    
    # 7. Feature Importance
    importance = pd.DataFrame({'feature': MODEL_FEATURES, 'gain': clf.feature_importances_}).sort_values('gain', ascending=False)
    print("\n🔍 Top 5 Features quan trọng nhất định đoạt cú Pump:")
    print(importance.head(5).to_string(index=False))
    
    return clf, threshold

def analyze_btc_regime_impact(df):
    print("🌡️ ĐANG PHÂN TÍCH NHIỆT KẾ BTC ĐỂ TÌM LUẬT CỨNG...")
    
    # Kiểm tra nếu thiếu btc_returns thì tính nhanh (dựa trên btc_close đã map)
    if 'btc_returns' not in df.columns:
         df['btc_returns'] = df.groupby('symbol')['btc_close'].pct_change()
    
    # Ép kiểu và xử lý NaN để tránh lỗi rolling
    df['btc_returns'] = df['btc_returns'].fillna(0)
    
    # Giả sử df đã có các cột: btc_close, btc_sma_200 (hoặc ema), btc_returns
    # Ta tạo thêm các biến động (Volatility) của BTC
    df['btc_vol_24h'] = df['btc_returns'].rolling(24).std()
    df['btc_ema_20'] = df['btc_close'].ewm(span=20).mean()
    df['btc_ema_50'] = df['btc_close'].ewm(span=50).mean()

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
        'actual_pump_pct': 'mean'
    })
    
    print("\n📊 BÁO CÁO SỨC MẠNH ALTCOIN THEO CHÂN BTC:")
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
        OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, OUTPUT_MODEL_PATH)
        joblib.dump({'features': MODEL_FEATURES, 'threshold': best_threshold}, OUTPUT_META_PATH)
        
        print(f"\n✅ Đã lưu model mới tại: {OUTPUT_MODEL_PATH}")
        print(f"Best Threshold (Top 20%): {best_threshold:.4f}")
    else:
        print("❌ Không có dữ liệu để train.")