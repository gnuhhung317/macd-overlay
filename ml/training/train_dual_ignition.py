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
    'upper_wick_ratio','dist_to_ema50_atr','vol_acceleration',
    'bb_squeeze','above_poc',
    'micro_volume','price_accel','order_flow_proxy',
    'btc_is_bull_regime','btc_trend_strength','adx','hour_sin','hour_cos','day_sin','day_cos',
    'btc_corr','trend_state','is_trending','is_volatile','ema_200_1d_dist','rsi_14_1d'
]

# THƯ MỤC MỚI ĐỂ TRÁNH GHI ĐÈ MODEL CŨ
BASE_DIR = Path(r"d:\Code\Projects\self-projects\macd-overlay - Copy")
INPUT_FILE = BASE_DIR / "data" / "processed" / "features_1h_btc_context.parquet"
OUTPUT_MODEL_DIR = BASE_DIR / "ml" / "training" / "models" / "1h_dual_ignition"
OUTPUT_MODEL_PATH = OUTPUT_MODEL_DIR / "dual_ignition_models.joblib"

# PARAMS CHO SIÊU CỔ PHIẾU NĂM 2025
HORIZON = 120
TARGET_MFE = 1.0
SAFE_MAE = -6

def prepare_dual_data(parquet_path):
    print("🧹 Tầng 1: Đang chuẩn bị dữ liệu (Lọc Nến Xanh Mồi Lửa)...")
    
    import pyarrow.parquet as pq
    all_parquet_cols = pq.read_schema(parquet_path).names
    
    # 1. XỬ LÝ BTC MACRO
    btc_price_col = next((c for c in ['btc_close_x', 'btc_close_y', 'btc_close'] if c in all_parquet_cols), None)
    if btc_price_col:
        btc_raw = pd.read_parquet(parquet_path, columns=['timestamp', btc_price_col])
        btc_raw['timestamp'] = pd.to_datetime(btc_raw['timestamp']).dt.tz_localize(None)
        btc_df = btc_raw.drop_duplicates(subset=['timestamp']).sort_values('timestamp').copy()
        btc_df.rename(columns={btc_price_col: 'btc_close'}, inplace=True)
        btc_df['btc_returns'] = btc_df['btc_close'].pct_change()
        btc_df['btc_vol_24h'] = btc_df['btc_returns'].rolling(24).std()
        btc_df['btc_ema_20'] = btc_df['btc_close'].ewm(span=20).mean()
        btc_df['btc_ema_50'] = btc_df['btc_close'].ewm(span=50).mean()
        del btc_raw; gc.collect()
    else:
        btc_df = pd.DataFrame()

    # 2. LOAD DATA ALTCOIN
    cols_to_load = list(set(MODEL_FEATURES + ['symbol', 'timestamp', 'open', 'close', 'high', 'low', 'volume', 'usd_vol_24h']))
    cols_to_load = [c for c in cols_to_load if c in all_parquet_cols]
    
    print("📥 Đang tải dữ liệu Altcoin...")
    df = pd.read_parquet(parquet_path, columns=cols_to_load)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)

    if not btc_df.empty:
        df = df.merge(btc_df[['timestamp', 'btc_close', 'btc_returns', 'btc_vol_24h', 'btc_ema_20', 'btc_ema_50']], on='timestamp', how='left')
        del btc_df; gc.collect()

    df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

    # 3. TÍNH ATR & FEATURES
    print("⚙️ Đang tính toán ATR và các Features nền tảng...")
    def calc_atr(df_group):
        hl = df_group['high'] - df_group['low']
        hc = np.abs(df_group['high'] - df_group['close'].shift())
        lc = np.abs(df_group['low'] - df_group['close'].shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(14).mean()

    # Thêm include_groups=False để dập cái warning vàng của Pandas
    df['atr_14'] = df.groupby('symbol', group_keys=False).apply(calc_atr, include_groups=False)
    
    if 'vol_compression' not in df.columns:
        df['vol_compression'] = df.groupby('symbol')['atr_14'].transform(lambda x: x / (x.rolling(100).mean() + 1e-9))
    
    df['ema_20'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=20).mean())
    df['ema_50'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=50).mean())
    vol_sma_20 = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).mean().shift(1))

    # ==============================================================
    # FIX LỖI Ở ĐÂY: Thêm lại 3 features bị thiếu cho Model
    # ==============================================================
    if 'upper_wick_ratio' not in df.columns:
        df['upper_wick_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-9)
    if 'dist_to_ema50_atr' not in df.columns:
        df['dist_to_ema50_atr'] = (df['close'] - df['ema_50']) / (df['atr_14'] + 1e-9)
    if 'vol_acceleration' not in df.columns:
        df['vol_acceleration'] = df.groupby('symbol')['volume'].transform(lambda x: x / (x.shift(1) + 1e-9))
    # 4. LỌC ĐIỂM NỔ (BULLISH IGNITION BAR)
    # Nến xanh, body > 1.5%, Volume đột biến, cắt lên hoặc nằm trên EMA20
    df['resistance_50'] = df.groupby('symbol')['high'].transform(lambda x: x.rolling(50).max().shift(1))

    cond_green_bar = (df['close'] > df['open']) & (df['close'] > df['ema_20'])
    cond_body_size = ((df['close'] - df['open']) / df['open']) > 0.015 
    cond_vol_ignition = (df['volume'] > vol_sma_20 * 1.5) & (df['volume'] < vol_sma_20 * 4.0)
    cond_rsi_fresh = (df['rsi_14'] >= 55) & (df['rsi_14'] <= 72)
    dist_to_res = (df['resistance_50'] - df['close']) / (df['close'] + 1e-9)
    cond_near_res = dist_to_res > -0.05 

    mask_ignition = cond_green_bar & cond_body_size & cond_vol_ignition & cond_rsi_fresh & cond_near_res
    
    
    # mask_ignition = cond_green & cond_body & cond_vol & cond_trend
    if 'usd_vol_24h' in df.columns:
        mask_ignition = mask_ignition & (df['usd_vol_24h'] >= 1000000)

    # Lấy index của các điểm nổ để quét tương lai cho chính xác (Tránh Lookahead)
    ignition_indices = df[mask_ignition].index
    print(f"🔍 Tìm thấy {len(ignition_indices)} nến Bullish Ignition. Đang trích xuất DNA tương lai...")

    # 5. TẠO NHÃN BẰNG FORWARD SCAN
    outliers = []
    
    # Ép kiểu numpy để quét cực nhanh
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    atrs = df['atr_14'].values
    symbols = df['symbol'].values

    for idx in ignition_indices:
        if idx + HORIZON >= len(df) or symbols[idx] != symbols[idx + HORIZON - 1]:
            continue # Bỏ qua nếu không đủ data hoặc bị lẹm sang symbol khác
            
        entry_price = closes[idx]
        atr = atrs[idx]
        
        # Quét mảng tương lai
        window_highs = highs[idx + 1 : idx + 1 + HORIZON]
        window_lows = lows[idx + 1 : idx + 1 + HORIZON]
        
        max_idx = np.argmax(window_highs)
        min_idx = np.argmin(window_lows)
        max_p = window_highs[max_idx]
        min_p = window_lows[min_idx]
        
        # Chỉ số cho BREAKOUT (LONG)
        long_mfe = (max_p - entry_price) / (atr + 1e-9)
        long_mae = (min_p - entry_price) / (atr + 1e-9)
        is_breakout = 1 if (long_mfe >= TARGET_MFE and long_mae >= SAFE_MAE and max_idx < min_idx) else 0
        
        # Chỉ số cho BULL TRAP (SHORT)
        short_mfe = (entry_price - min_p) / (atr + 1e-9)
        short_mae = (entry_price - max_p) / (atr + 1e-9)
        is_trap = 1 if (short_mfe >= TARGET_MFE and short_mae >= SAFE_MAE and min_idx < max_idx) else 0
        
        # Lọc bỏ các case Whipsaw/Nhiễu loạn (Cả 2 đều True là vô lý nhưng đề phòng)
        if is_breakout and is_trap:
            is_breakout, is_trap = 0, 0

        outliers.append({
            'index': idx,
            'is_breakout': is_breakout,
            'is_trap': is_trap
        })

    labels_df = pd.DataFrame(outliers).set_index('index')
    
    # Merge lại vào tập dữ liệu gốc
    final_df = df.loc[labels_df.index].copy()
    final_df['is_breakout'] = labels_df['is_breakout'].values
    final_df['is_trap'] = labels_df['is_trap'].values
    
    # Dọn RAM
    del df; gc.collect()
    
    final_df = final_df.dropna(subset=MODEL_FEATURES).reset_index(drop=True)
    print(f"🎯 Dataset hoàn tất! Tổng số mẫu: {len(final_df)}")
    print(f"   -> Số ca True Breakout: {final_df['is_breakout'].sum()}")
    print(f"   -> Số ca Bull Trap: {final_df['is_trap'].sum()}")
    
    return final_df

def train_dual_models(df):
    print("\n🤖 Đang huấn luyện Cặp Song Kiếm (Dual-Model Architecture)...")
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_tr = train_df[MODEL_FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
    X_te = test_df[MODEL_FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # -----------------------------------------
    # 1. TRAIN MODEL: THE BREAKOUT HUNTER (LONG)
    # -----------------------------------------
    print("   [1/2] Đang rèn kiếm Breakout Hunter...")
    y_tr_breakout = train_df['is_breakout']
    y_te_breakout = test_df['is_breakout']
    
    clf_breakout = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.02, max_depth=6, num_leaves=31,
        objective='binary', class_weight='balanced', # Quan trọng để học class thiểu số
        random_state=42, n_jobs=-1, verbose=-1
    )
    clf_breakout.fit(X_tr, y_tr_breakout)
    
    # Đánh giá Breakout
    pred_breakout = clf_breakout.predict_proba(X_te)[:, 1]
    calls_b = pred_breakout > 0.75
    if calls_b.sum() > 0:
        prec_b = y_te_breakout[calls_b].mean()
        print(f"   🎯 Precision Breakout (>0.75): {prec_b*100:.2f}% (Bắt được {calls_b.sum()} nến)")
    
    # -----------------------------------------
    # 2. TRAIN MODEL: THE TRAP HUNTER (SHORT)
    # -----------------------------------------
    print("\n   [2/2] Đang rèn kiếm Trap Hunter...")
    y_tr_trap = train_df['is_trap']
    y_te_trap = test_df['is_trap']
    
    clf_trap = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.02, max_depth=6, num_leaves=31,
        objective='binary', class_weight='balanced',
        random_state=84, n_jobs=-1, verbose=-1
    )
    clf_trap.fit(X_tr, y_tr_trap)
    
    # Đánh giá Trap
    pred_trap = clf_trap.predict_proba(X_te)[:, 1]
    calls_t = pred_trap > 0.75
    if calls_t.sum() > 0:
        prec_t = y_te_trap[calls_t].mean()
        print(f"   🎯 Precision Bull Trap (>0.75): {prec_t*100:.2f}% (Phát hiện {calls_t.sum()} bẫy)")

    # In ra DNA (Feature Importance) của mô hình bắt Bẫy
    importance = pd.DataFrame({'feature': MODEL_FEATURES, 'gain': clf_trap.feature_importances_}).sort_values('gain', ascending=False)
    print("\n🔍 Top 5 Dấu hiệu nhận biết Bull Trap (Theo mô hình):")
    print(importance.head(5).to_string(index=False))

    return clf_breakout, clf_trap

if __name__ == "__main__":
    if not INPUT_FILE.exists():
        print(f"❌ Không tìm thấy file: {INPUT_FILE}")
        sys.exit(1)

    golden_df = prepare_dual_data(INPUT_FILE)
    
    if not golden_df.empty:
        clf_breakout, clf_trap = train_dual_models(golden_df)
        
        OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        # Lưu cả 2 model vào 1 file dictionary để tiện load
        joblib.dump({
            'breakout_model': clf_breakout,
            'trap_model': clf_trap,
            'features': MODEL_FEATURES,
            'threshold_breakout': 0.75,
            'threshold_trap': 0.75
        }, OUTPUT_MODEL_PATH)
        
        print(f"\n✅ Đã lưu Cặp Song Kiếm tại: {OUTPUT_MODEL_PATH}")
    else:
        print("❌ Không đủ dữ liệu mồi lửa để train.")