import os, gc, joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(r"d:\Code\Projects\self-projects\macd-overlay - Copy")
SYMBOLS_DIR = BASE_DIR / "data" / "processed" / "symbols_v2"
MODEL_PATH = BASE_DIR / "ml" / "training" / "models" / "1h" / "ensemble_lgbm_tabular.joblib"
META_PATH = BASE_DIR / "ml" / "training" / "models" / "1h" / "ensemble_meta.joblib"

def load_assets():
    if not META_PATH.exists() or not MODEL_PATH.exists():
        print("❌ Thiếu model hoặc meta!")
        return None, [], 0.6146
    meta = joblib.load(META_PATH)
    clf = joblib.load(MODEL_PATH)
    features = meta.get('features', []) if isinstance(meta, dict) else meta
    threshold = meta.get('threshold', 0.6146)
    return clf, features, threshold

def process_single_file(file_path, features, clf, threshold, horizon=48):
    try:
        df = pd.read_parquet(file_path)
        if df.empty: return None
        
        # --- SỬA TẠI ĐÂY: Đảm bảo luôn có cột symbol ---
        if 'symbol' not in df.columns:
            df['symbol'] = Path(file_path).stem.replace('_USDT', '').replace('USDT', '')

        # --- Chặn Data Leakage: Chỉ test trên dữ liệu CHƯA TỪNG NHÌN THẤY ---
        cutoff_date = pd.to_datetime('2025-01-01')
        # Đảm bảo cột timestamp là datetime (tz-naive)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        df = df[df['timestamp'] > cutoff_date]
        
        if df.empty: return None

        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 1. Tính MFE/MAE (Chỉ dành cho validation dữ liệu cũ)
        df['f_high'] = df['high'].shift(-1).rolling(horizon, min_periods=1).max().shift(-(horizon-1))
        df['f_low'] = df['low'].shift(-1).rolling(horizon, min_periods=1).min().shift(-(horizon-1))
        df['mfe'] = ((df['f_high'] - df['close']) / df['close']) * 100
        df['mae'] = ((df['f_low'] - df['close']) / df['close']) * 100

        # 2. Indicators & Filter Tầng 1
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        vol_sma = df['volume'].rolling(20).mean().shift(1)
        
        # New Feature: Volume Ratio
        df['vol_ratio'] = df['volume'] / (vol_sma + 1e-9)
        
        # Calculated ATR (14) & ATR Pct
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['tr'] = np.max(ranges, axis=1)
        df['atr_14'] = df['tr'].rolling(14).mean()
        df['atr_pct'] = (df['atr_14'] / df['close']) * 100
        
        # New Features (Machine Detector of Lies)
        df['upper_wick_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-9)
        df['dist_to_ema50_atr'] = (df['close'] - df['ema_50']) / (df['atr_14'] + 1e-9)
        df['vol_acceleration'] = df['volume'] / (df['volume'].shift(1) + 1e-9)
        
        # Áp dụng Tầng 1: Filter cứng (Ignition Bar)
        c1 = (df['close'] > df['open']) & (df['close'] > df['ema_20'])
        c2 = ((df['close'] - df['open']) / df['open']) > 0.015
        c3 = (df['volume'] > vol_sma * 1.5) & (df['volume'] < vol_sma * 4.0)
        c4 = (df['rsi_14'] >= 55) & (df['rsi_14'] <= 72)
        
        hits = df[c1 & c2 & c3 & c4].copy()
        if hits.empty: return None
        
        # 3. AI Sniper Chấm điểm (Multiclass)
        X = hits[features].apply(pd.to_numeric, errors='coerce').fillna(0)
        probas = clf.predict_proba(X)
        
        hits['prob_long'] = probas[:, 1]
        hits['prob_short'] = probas[:, 2]
        
        # Logic bóp cò: 
        # Nếu Prob(Long) > 0.6 -> LONG
        # Nếu Prob(Short) > 0.6 -> SHORT
        hits['final_signal'] = 'WAIT'
        hits.loc[hits['prob_long'] > 0.6, 'final_signal'] = '🚀 LONG'
        hits.loc[hits['prob_short'] > 0.6, 'final_signal'] = '💀 SHORT'
        
        return hits[hits['final_signal'] != 'WAIT']
    except:
        return None


# Remove obsolete classification functions for lean logic

if __name__ == "__main__":
    print(f"\n{'='*60}\n🚀 AI SNIPER SCANNER v2.0 - CHẾ ĐỘ CHIA ĐỂ TRỊ\n{'='*60}")
    clf, feat, thresh = load_assets()
    all_files = list(SYMBOLS_DIR.glob("*.parquet"))
    
    final_list = []
    print(f"🔍 Đang quét {len(all_files)} mã tài sản...")

    for f in all_files:
        res = process_single_file(f, feat, clf, thresh)
        
        # KIỂM TRA: Chỉ xử lý nếu res không None và KHÔNG RỖNG
        if res is not None and not res.empty:
            final_list.append(res)
            
            # Lấy kèo mới nhất của Symbol này
            latest = res.iloc[-1]
            
            # Kiểm tra thời gian: Chỉ in ra nếu kèo xuất hiện trong vòng 7 ngày qua
            # (Giả sử nến mới nhất của bạn là quanh tháng 3/2026)
            latest_ts = pd.to_datetime(latest['timestamp'])
            threshold_date = pd.Timestamp.now() - pd.Timedelta(days=7)
            
            if latest_ts > threshold_date:
                  print(f"🔥 Kèo: {latest['symbol']} | Signal: {latest['final_signal']} | {latest['timestamp']} | {latest['close']:.2f} | ProbL: {latest['prob_long']:.2f} | ProbS: {latest['prob_short']:.2f}")
        else:
            # Skip những con không có kèo mà không gây crash
            continue

    if final_list:
        report_df = pd.concat(final_list, ignore_index=True)
        report_df = report_df.sort_values(['timestamp', 'prob_long', 'prob_short'], ascending=[False, False, False])
        
        print(f"\n✅ TỔNG HỢP: Tìm thấy {len(report_df)} điểm vào lệnh Sniper.")
        cols = ['timestamp', 'symbol', 'final_signal','close', 'prob_long', 'prob_short', 'mfe', 'mae']
        print(report_df[cols].head(30).to_string(index=False))
        
        # Thống kê hiệu suất - TÁCH BẠCH ĐA CHIỀU
        longs = report_df[report_df['final_signal'] == '🚀 LONG'].copy()
        shorts = report_df[report_df['final_signal'] == '💀 SHORT'].copy()
        
        print(f"\n📊 BÁO CÁO HIỆU SUẤT TỔNG QUAN:")
        print(f"Tổng số lệnh kích hoạt: {len(report_df)} (Trong đó: {len(longs)} LONG, {len(shorts)} SHORT)")

        if not longs.empty:
            # Normalize MFE/MAE theo ATR cho phe LONG
            longs['mfe_atr'] = longs['mfe'] / (longs['atr_pct'] + 1e-9)
            longs['mae_atr'] = longs['mae'] / (longs['atr_pct'] + 1e-9)
            win_longs = (longs['mfe_atr'] >= 2.0).mean() * 100
            
            print(f"\n� HỆ LONG (Breakout Thật):")
            print(f"   - Tỷ lệ cắn TP (> 2x ATR): {win_longs:.2f}%")
            print(f"   - Sức rướn (MFE Median): {longs['mfe'].median():.2f}% ({longs['mfe_atr'].median():.2f}x ATR)")
            print(f"   - Gồng lỗ (MAE Median): {longs['mae'].median():.2f}% ({longs['mae_atr'].median():.2f}x ATR)")

        if not shorts.empty:
            # Normalize MFE/MAE theo ATR cho phe SHORT
            shorts['mfe_atr'] = shorts['mfe'] / (shorts['atr_pct'] + 1e-9)
            shorts['mae_atr'] = shorts['mae'] / (shorts['atr_pct'] + 1e-9)
            # Short win khi giá sập sâu (MAE âm nặng)
            win_shorts = (shorts['mae_atr'] <= -2.0).mean() * 100
            
            print(f"\n🔴 HỆ SHORT (Săn Bull Trap):")
            print(f"   - Tỷ lệ sập sâu (Lãi > 2x ATR): {win_shorts:.2f}%")
            print(f"   - Độ sập/Lãi (MAE Median): {shorts['mae'].median():.2f}% ({abs(shorts['mae_atr'].median()):.2f}x ATR)")
            print(f"   - Gồng rướn/Lỗ (MFE Median): {shorts['mfe'].median():.2f}% ({shorts['mfe_atr'].median():.2f}x ATR)")
    else:
        print("\nℹ️ Đang rình mồi... AI chưa thấy cơ hội nào thực sự tinh quái.")