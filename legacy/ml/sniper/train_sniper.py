import os
import sys
import gc
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path

# ============================================================
# CONFIG & FEATURES (ĐÃ THANH TRỪNG BIẾN THỜI GIAN)
# ============================================================
MODEL_FEATURES = [
    'rsi_14', 'rsi_slope', 'stoch_k', 'stoch_d', 'roc_7', 'roc_14',
    'volume_ratio', 'volume_zscore', 'volume_trend', 'rs_vs_btc', 'rs_vs_btc_sma7', 'vol_compression',
    'dist_to_high_30d', 'dist_to_low_30d', 'dist_to_ema_21_pct', 'dist_to_ema_50_pct', 'dist_to_ema_200_pct',
    'price_vs_sma_30', 'momentum_30', 'macd_slope', 'macd_acceleration',
    'upper_wick_ratio', 'dist_to_ema50_atr', 'vol_acceleration', 
    'bb_squeeze', 'above_poc',
    'micro_volume', 'price_accel', 'order_flow_proxy',
    'btc_is_bull_regime', 'btc_trend_strength', 'adx',
    'btc_corr', 'trend_state', 'is_trending', 'is_volatile', 'ema_200_1d_dist', 'rsi_14_1d'
]

# Paths
BASE_DIR = Path(r"d:\Code\Projects\self-projects\macd-overlay - Copy")
INPUT_FILE = BASE_DIR / "ml" / "features_1h_full (1).parquet"
OUTPUT_MODEL_DIR = BASE_DIR / "ml" / "training" / "models" / "honest"
OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

def prepare_cascade_data_optimized(parquet_path):
    print("🧹 Tầng 1: Lọc nhiễu cơ bản & Tính toán Vi mô/Vĩ mô...")
    import pyarrow.parquet as pq
    all_parquet_cols = pq.read_schema(parquet_path).names
    
    # --- BƯỚC 1: XỬ LÝ DỮ LIỆU VĨ MÔ (BTC) ---
    btc_price_col = next((c for c in ['btc_close_x', 'btc_close_y', 'btc_close'] if c in all_parquet_cols), None)
    if btc_price_col:
        btc_raw = pd.read_parquet(parquet_path, columns=['timestamp', btc_price_col])
        btc_raw['timestamp'] = pd.to_datetime(btc_raw['timestamp']).dt.tz_localize(None)
        btc_df = btc_raw.drop_duplicates(subset=['timestamp']).sort_values('timestamp').copy()
        btc_df.rename(columns={btc_price_col: 'btc_close'}, inplace=True)
        btc_df['btc_returns'] = btc_df['btc_close'].pct_change()
        btc_df['btc_vol_24h'] = btc_df['btc_returns'].rolling(24).std()
        del btc_raw; gc.collect()
    else:
        btc_df = pd.DataFrame()

    # --- BƯỚC 2: LOAD DỮ LIỆU ALTCOIN ---
    cols_to_load = [c for c in set(MODEL_FEATURES + ['symbol', 'timestamp', 'open', 'close', 'high', 'low', 'volume', 'usd_vol_24h']) if c in all_parquet_cols]
    df = pd.read_parquet(
        parquet_path, columns=cols_to_load,
        filters=[('timestamp', '<', pd.Timestamp('2025-01-01')), ('usd_vol_24h', '>=', 1000000)]
    )

    if not btc_df.empty:
        df = df.merge(btc_df[['timestamp', 'btc_close', 'btc_returns', 'btc_vol_24h']], on='timestamp', how='left')
        del btc_df; gc.collect()

    df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

    # --- BƯỚC 3: TÍNH ATR & MÁY PHÁT HIỆN NÓI DỐI (VI MÔ) ---
    def calc_atr(df_group):
        high_low = df_group['high'] - df_group['low']
        high_close = np.abs(df_group['high'] - df_group['close'].shift())
        low_close = np.abs(df_group['low'] - df_group['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(14).mean()

    # Tắt cảnh báo GroupBy của Pandas mới
    df['atr_14'] = df.groupby('symbol', group_keys=False).apply(calc_atr, include_groups=False)
    
    # Vũ khí quét nhiễu vi mô
    df['upper_wick_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-9)
    df['lower_wick_ratio'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-9)
    df['ema_50'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=50).mean())
    df['dist_to_ema50_atr'] = (df['close'] - df['ema_50']) / (df['atr_14'] + 1e-9)
    df['vol_acceleration'] = df.groupby('symbol')['volume'].transform(lambda x: x / (x.shift(1) + 1e-9))

    # --- BƯỚC 4: BỘ LỌC MỞ (DYNAMIC IGNITION) ---
    vol_sma_20 = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).mean().shift(1))
    
    # Mồi lửa: Nến xanh, body > 1%, volume > 1.2x trung bình
    cond_green_bar = df['close'] > df['open']
    cond_body_size = ((df['close'] - df['open']) / df['open']) > 0.01 
    cond_vol_ignition = df['volume'] > (vol_sma_20 * 1.2)
    
    mask_base = cond_green_bar & cond_body_size & cond_vol_ignition
    if 'usd_vol_24h' in df.columns:
        mask_base = mask_base & (df['usd_vol_24h'] >= 1000000)

    # Lọc data thô trước để giảm tải tính toán MFE/MAE
    df = df[mask_base].reset_index(drop=True)

    # --- BƯỚC 5: TÍNH LABEL THỰC CHIẾN MỚI (CHỈ LONG) ---
    horizon = 48
    df['future_max_high'] = df.groupby('symbol')['high'].shift(-1).rolling(horizon, min_periods=1).max().shift(-(horizon-1))
    df['future_min_low'] = df.groupby('symbol')['low'].shift(-1).rolling(horizon, min_periods=1).min().shift(-(horizon-1))
    
    df['mfe_atr'] = (df['future_max_high'] - df['close']) / (df['atr_14'] + 1e-9)
    df['mae_atr'] = (df['future_min_low'] - df['close']) / (df['atr_14'] + 1e-9)
    
    # --- BƯỚC 6: TÁCH REGIME & GÁN NHÃN BINARY ---
    cond_reversal = df['ema_200_1d_dist'] < -0.15 
    
    # [TỐI ƯU HÓA BREAKOUT] Ép điều kiện Nén (Squeeze/Compression)
    # Không nén thì không có bùng nổ thực sự.
    cond_compression = (df['bb_squeeze'] > 0.5) | (df['vol_compression'] > 1.2)
    cond_breakout = (df['dist_to_high_30d'] > -0.15) & (df['ema_200_1d_dist'] >= -0.15) & cond_compression

    df_reversal = df[cond_reversal].dropna(subset=MODEL_FEATURES).copy()
    df_breakout = df[cond_breakout].dropna(subset=MODEL_FEATURES).copy()

    # Reversal: Target to (x2), Stoploss nới rộng (-1.5)
    df_reversal['label'] = 0
    df_reversal.loc[(df_reversal['mfe_atr'] >= 2.0) & (df_reversal['mae_atr'] >= -1.5), 'label'] = 1

    # [TỐI ƯU HÓA BREAKOUT] Trả lại sự thật cho thị trường
    # Chấp nhận râu nến quét Stoploss (-1.5) để ăn sóng đẩy MFE (>= 2.0)
    df_breakout['label'] = 0
    df_breakout.loc[(df_breakout['mfe_atr'] >= 2.0) & (df_breakout['mae_atr'] >= -1.5), 'label'] = 1

    print(f"🎯 Tầng 1 Xong! Tách thành công: {len(df_reversal)} kèo Reversal, {len(df_breakout)} kèo Breakout.")
    return df_reversal, df_breakout

def train_binary_regime_model(df_regime, regime_name):
    print(f"\n🤖 Đang huấn luyện AI Binary Sniper - Chế độ: {regime_name.upper()}...")
    
    if len(df_regime) < 1000:
        print(f"⚠️ Không đủ data cho {regime_name} ({len(df_regime)} dòng). Skip.")
        return None

    df_regime = df_regime.sort_values('timestamp').reset_index(drop=True)
    split_idx = int(len(df_regime) * 0.8)
    
    X_tr = df_regime.iloc[:split_idx][MODEL_FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
    y_tr = df_regime.iloc[:split_idx]['label']
    X_te = df_regime.iloc[split_idx:][MODEL_FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
    y_te = df_regime.iloc[split_idx:]['label']
    
    # Đổi sang bài toán Binary (Nhị phân)
    clf = lgb.LGBMClassifier(
        n_estimators=600, 
        learning_rate=0.015,  # Giảm learning rate để học mượt hơn
        max_depth=5,          # Giảm depth chống Overfit        
        num_leaves=25,
        objective='binary',   # Lõi thuật toán thay đổi hoàn toàn tại đây
        class_weight='balanced', 
        random_state=42, 
        n_jobs=-1,
        verbose=-1
    )
    
    clf.fit(X_tr, y_tr)
    
    # Xác suất trả về giờ chỉ có cột [0] (Skip) và [1] (Long)
    preds_proba = clf.predict_proba(X_te)[:, 1] 
    
    print(f"\n{'='*40}\n KẾT QUẢ THỰC CHIẾN BINARY - {regime_name.upper()}\n{'='*40}")
    print(f"Tổng số kèo Out-of-Sample: {len(y_te)}")
    print(f"Tỷ lệ Kèo Tốt (Win Rate tự nhiên): {y_te.mean()*100:.2f}%")
    print("-" * 40)
    
    # Đánh giá các ngưỡng tự tin (Confidence Thresholds)
    for thresh in [0.60, 0.70, 0.80]:
        calls = preds_proba >= thresh
        num_calls = calls.sum()
        if num_calls > 0:
            precision = y_te[calls].mean() * 100
            print(f"🎯 Threshold > {thresh:.2f} | Precision: {precision:.2f}% | Gọi lệnh: {num_calls} kèo")
        else:
            print(f"🎯 Threshold > {thresh:.2f} | N/A (0 kèo)")
            
    importance = pd.DataFrame({'feature': MODEL_FEATURES, 'gain': clf.feature_importances_}).sort_values('gain', ascending=False)
    print(f"\n🔍 Top 5 Features định đoạt mô hình {regime_name}:")
    print(importance.head(5).to_string(index=False))
    
    return clf

if __name__ == "__main__":
    if not INPUT_FILE.exists():
        print(f"❌ Không tìm thấy file: {INPUT_FILE}"); sys.exit(1)

    df_reversal, df_breakout = prepare_cascade_data_optimized(INPUT_FILE)
    
    model_reversal = train_binary_regime_model(df_reversal, "Reversal_DipSniper")
    model_breakout = train_binary_regime_model(df_breakout, "Breakout_Momentum")
    
    if model_reversal:
        path_rev = OUTPUT_MODEL_DIR / "model_reversal.joblib"
        joblib.dump(model_reversal, path_rev)
        print(f"\n✅ Đã lưu Binary Reversal tại: {path_rev}")
        
    if model_breakout:
        path_brk = OUTPUT_MODEL_DIR / "model_breakout.joblib"
        joblib.dump(model_breakout, path_brk)
        print(f"✅ Đã lưu Binary Breakout tại: {path_brk}")
        
    joblib.dump({'features': MODEL_FEATURES, 'threshold': 0.70}, OUTPUT_MODEL_DIR / "ensemble_meta.joblib")
    print("\n🏁 Hoàn tất Binary Pipeline!")