#!/usr/bin/env python3
"""
🚀 AI Sniper Sync Worker - Production Grade (Standalone & 100% Synced)
Nhiệm vụ: 
1. Đồng bộ dữ liệu Binance OHLCV tuần tự với cơ chế Auto-Retry.
2. Feature Engineering đồng bộ tuyệt đối với file Training (Không có điểm mù).
3. Cơ chế Fail-Fast: Khuyết data -> Khóa cò máy Sniper.
4. Chế độ Loop 24/7: Tự động quét khi đóng nến 1h.
"""
import os, sys, time, json, joblib, gc, traceback
import pandas as pd
import numpy as np
import ccxt
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG & PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
SYMBOLS_DIR_1H = DATA_DIR / "symbols_v2"
SYMBOLS_DIR_1D = DATA_DIR / "symbols_1d"
MODEL_PATH  = BASE_DIR / "ml" / "training" / "models" / "1h" / "ensemble_lgbm_tabular.joblib"
META_PATH   = BASE_DIR / "ml" / "training" / "models" / "1h" / "ensemble_meta.joblib"

SIGNALS_JSON = Path(__file__).parent / "signals_live.json"
STATS_JSON   = Path(__file__).parent / "market_stats.json"

for d in [SYMBOLS_DIR_1H, SYMBOLS_DIR_1D]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# T1 FILTER RULES (INTERNAL CONFIG)
# Lưu ý: Các thông số này đóng vai trò màng lọc thô trước khi ném cho AI.
# ============================================================
SYNC_RULES = {
    'BODY_SIZE_MIN': 0.015,     # Thân nến >= 1.5%
    'VOL_IGNITION_MIN': 1.5,    # Volume đột biến >= 1.5x SMA20
    'VOL_IGNITION_MAX': 4.0,    # Chặn trần Volume <= 4.0x
    'RSI_MIN': 55,              # RSI vào đà
    'RSI_MAX': 72,              # Tránh mua đuổi
    'DIST_TO_RES_MIN': -0.05    # Đừng mua sát cản (cách ít nhất 5%)
}

# Khởi tạo Binance API với Timeout cứng tránh treo process
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'},
    'timeout': 10000 
})

# ============================================================
# UTILS: ROBUST NETWORK MATRICE
# ============================================================
def fetch_ohlcv_with_retry(symbol, timeframe, limit=1000, since=None, max_retries=3):
    """Cơ chế sinh tồn: Chống sập khi dính Rate Limit hoặc đứt mạng."""
    for attempt in range(max_retries):
        try:
            return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)
        except ccxt.RateLimitExceeded:
            sleep_time = 5 * (attempt + 1)
            print(f" ⏳ [API Lìmit] {symbol} bị chặn. Ngủ {sleep_time}s...")
            time.sleep(sleep_time)
        except ccxt.NetworkError as e:
            print(f" 🌐 [Mạng Lỗi] {symbol} ({e}). Thử lại {attempt+1}/{max_retries}...")
            time.sleep(2)
        except (ccxt.BadSymbol, ccxt.ExchangeError) as e:
            print(f" ❌ [LỖI SÀN] {symbol}: {str(e)}")
            return []
        except Exception as e:
            print(f" ❌ [LỖI NGHIÊM TRỌNG] {symbol}: {str(e)}")
            raise e
    return []

def fetch_ohlcv_paginated(symbol, timeframe, lookback_total):
    """
    Fetch OHLCV dữ liệu lớn bằng cách phân trang.
    Tự động tính toán 'since' và lặp lại cho đến khi đủ số lượng nến yêu cầu.
    """
    # Buffer để tính toán EMA/RSI chính xác (cần khoảng 200 nến mồi)
    buffer = 200
    target_count = lookback_total + buffer
    all_ohlcv = []
    
    # Ước tính thời gian bắt đầu (miliseconds)
    # 1h = 3600000ms
    ms_per_bar = 3600000 if timeframe == '1h' else 86400000
    since = int(time.time() * 1000) - (target_count * ms_per_bar)
    
    print(f"   [Sync] Deep Fetch {symbol}: Target {target_count} bars...", end="", flush=True)
    
    while len(all_ohlcv) < target_count:
        batch = fetch_ohlcv_with_retry(symbol, timeframe, limit=1000, since=since)
        if not batch:
            break
            
        all_ohlcv.extend(batch)
        # Cập nhật since = timestamp của nến cuối + 1ms để lấy batch tiếp theo
        since = batch[-1][0] + 1
        
        # Nếu batch cuối cùng không đủ 1000, nghĩa là đã lấy hết dữ liệu mới nhất
        if len(batch) < 1000:
            break
            
        # Tránh spam API quá nhanh
        time.sleep(0.1)
    
    print(f" Got {len(all_ohlcv)} bars.", flush=True)
    return all_ohlcv

# ============================================================
# FEATURE ENGINEERING (REANIMATED BRAIN - 100% SYNCED)
# ============================================================
def calculate_rsi(prices, period=14):
    d = prices.diff()
    g = d.where(d > 0, 0).rolling(period).mean()
    l = (-d.where(d < 0, 0)).rolling(period).mean()
    return 100 - (100 / (1 + g / (l.replace(0, np.nan) + 1e-9)))

def calculate_macd_features(df, fast=12, slow=26, signal=9):
    ef = df['close'].ewm(span=fast).mean()
    es = df['close'].ewm(span=slow).mean()
    df['macd'] = ef - es
    df['macd_signal'] = df['macd'].ewm(span=signal).mean()
    df['macd_slope'] = df['macd'].diff()
    df['macd_acceleration'] = df['macd_slope'].diff()
    df['is_bullish_cross'] = ((df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
    return df

def extract_features_live(df_1h, df_1d, btc_df):
    """Tính toán Features chính xác từng dấu phẩy so với tập Train."""
    df = df_1h.copy()
    
    # 1. Price & Range
    df['log_returns'] = np.log(df['close'] / (df['close'].shift(1) + 1e-9))
    df['high_low_range'] = (df['high'] - df['low']) / (df['close'] + 1e-9)
    df['body_size'] = abs(df['close'] - df['open']) / (df['close'] + 1e-9)
    df['candle_range'] = df['high'] - df['low'] + 1e-9
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['upper_wick_ratio'] = df['upper_wick'] / df['candle_range'] 
    
    # 2. Moving Averages (Đã thêm 20 cho T1 Filter)
    for p in [7, 14, 20, 21, 50, 100, 200]: 
        df[f'ema_{p}'] = df['close'].ewm(span=p).mean()
    for p in [10, 20, 30, 50, 200]: 
        df[f'sma_{p}'] = df['close'].rolling(p).mean()
        
    df['price_vs_sma_30'] = df['close'] / (df['sma_30'] + 1e-9)
    df['momentum_30'] = df['close'].pct_change(30)
    
    # 3. Volatility
    tr = pd.concat([df['high'] - df['low'], abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['volatility_14'] = df['log_returns'].rolling(14).std()
    df['vol_sma_14'] = df['volatility_14'].rolling(14).mean()
    df['vol_compression'] = df['volatility_14'] / (df['vol_sma_14'] + 1e-9)
    
    # 4. Volume
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_std_20'] = df['volume'].rolling(20).std()
    df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-9)
    df['volume_zscore'] = (df['volume'] - df['volume_sma_20']) / (df['volume_std_20'] + 1e-9)
    df['volume_trend'] = df['volume'].rolling(7).mean() / (df['volume'].rolling(21).mean() + 1e-9)
    
    # 5. Indicators
    df['rsi_14'] = calculate_rsi(df['close'], 14)
    df['rsi_slope'] = df['rsi_14'].diff(3)
    
    l14 = df['low'].rolling(14).min(); h14 = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * (df['close'] - l14) / (h14 - l14).replace(0, np.nan)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    
    df = calculate_macd_features(df)
    
    # ADX
    pdm = df['high'].diff(); mdm = -df['low'].diff()
    pdm = pdm.where((pdm > mdm) & (pdm > 0), 0); mdm = mdm.where((mdm > pdm) & (mdm > 0), 0)
    pdi = 100 * (pdm.rolling(14).mean() / df['atr_14'].replace(0, np.nan))
    mdi = 100 * (mdm.rolling(14).mean() / df['atr_14'].replace(0, np.nan))
    df['adx'] = (100 * abs(pdi - mdi) / (pdi + mdi).replace(0, np.nan)).rolling(14).mean()
    
    # 6. Pulse & Distances
    df['roc_7'] = df['close'].pct_change(7)
    df['roc_14'] = df['close'].pct_change(14)
    df['dist_to_high_30d'] = (df['close'] - df['high'].rolling(30).max()) / df['close']
    df['dist_to_low_30d'] = (df['close'] - df['low'].rolling(30).min()) / df['close']
    
    for e in [21, 50, 200]: 
        df[f'dist_to_ema_{e}_pct'] = (df['close'] - df[f'ema_{e}']) / df['close']
        
    df['trend_state'] = np.where(df['close'] > df['sma_50'], 1, np.where(df['close'] < df['sma_50'], -1, 0))
    df['is_trending'] = (df['adx'] > 25).astype(int)
    df['is_volatile'] = (df['vol_compression'] > 1.5).astype(int)
    
    bb_mid = df['close'].rolling(20).mean(); bb_std = df['close'].rolling(20).std()
    bb_wd = (bb_mid + 2*bb_std - (bb_mid - 2*bb_std)) / (bb_mid + 1e-9)
    df['bb_squeeze'] = (bb_wd < bb_wd.rolling(20).quantile(0.2)).astype(int)
    
    # 7. Time
    df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.dayofweek / 7)

    # 8. Hit & Run / Proxies
    df['vwap_30d'] = (df['close'] * df['volume']).rolling(30).sum() / (df['volume'].rolling(30).sum() + 1e-9)
    df['above_poc'] = (df['close'] > df['vwap_30d']).astype(int)
    df['micro_volume'] = df['volume'] / (df['volume'].rolling(5).mean() + 1e-9)
    df['price_accel'] = df['close'].pct_change(1) / (df['close'].pct_change(4).replace(0, np.nan) + 1e-9)
    df['order_flow_proxy'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)
    
    df['dist_to_ema50_atr'] = (df['close'] - df['ema_50']) / (df['atr_14'] + 1e-9)
    df['vol_acceleration'] = df['volume'] / (df['volume'].shift(1) + 1e-9)
    df['resistance_50'] = df['high'].rolling(50).max().shift(1)
    df['dist_to_res'] = (df['resistance_50'] - df['close']) / (df['close'] + 1e-9)

    # 9. 1D MTF (Anti-Lookahead)
    if df_1d is not None and not df_1d.empty:
        d1d_ema = df_1d['close'].ewm(span=200).mean().shift(1).iloc[-1]
        df['ema_200_1d_dist'] = (df['close'] - d1d_ema) / df['close']
        df['rsi_14_1d'] = calculate_rsi(df_1d['close'], 14).shift(1).iloc[-1]
    else:
        df['ema_200_1d_dist'] = np.nan
        df['rsi_14_1d'] = np.nan

    # 10. BTC Context
    if btc_df is not None and not btc_df.empty:
        # Ánh xạ (Map) dữ liệu BTC sang Altcoin khớp chính xác từng nến 1H
        btc_map = btc_df.set_index('timestamp')
        df['btc_close'] = df['timestamp'].map(btc_map['close']).ffill()
        df['btc_ema_200'] = df['timestamp'].map(btc_map['ema_200']).ffill()
        df['btc_adx'] = df['timestamp'].map(btc_map['adx']).ffill()
        
        # Đây là cột btc_returns thật của từng nến trong quá khứ, không phải là mảng object
        df['btc_returns_hist'] = df['timestamp'].map(btc_map['log_returns']).fillna(0)
        
        # Tính toán chuẩn như file Train
        df['btc_is_bull_regime'] = (df['btc_close'] > df['btc_ema_200']).astype(int)
        df['btc_trend_strength'] = (df['btc_adx'] > 25).astype(int)
        df['rs_vs_btc'] = df['log_returns'] - df['btc_returns_hist']
        df['rs_vs_btc_sma7'] = df['rs_vs_btc'].rolling(7).mean()
        
        # Dứt điểm lỗi Object và tính Corr chuẩn xác:
        df['btc_corr'] = df['log_returns'].rolling(14).corr(df['btc_returns_hist']).fillna(0)
        
        # Dọn dẹp RAM các cột mượn tạm
        df.drop(columns=['btc_close', 'btc_ema_200', 'btc_adx', 'btc_returns_hist'], inplace=True, errors='ignore')
    else:
        for c in ['btc_is_bull_regime', 'btc_trend_strength', 'btc_corr', 'rs_vs_btc', 'rs_vs_btc_sma7']:
            df[c] = np.nan

    return df

# ============================================================
# SYNC WORKER CORE
# ============================================================
def sync_all(lookback=1):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 Bắt đầu chu kỳ Sync (Lookback: {lookback} bars)...")
    
    # Fetch All USDT Markets from Binance
    print(f"   [Sync] Đang lấy danh sách Market từ Binance...")
    try:
        markets = exchange.load_markets()
        # Filter for active USD-M (Linear) Perpetual Futures
        # Store as (original_api_symbol, clean_internal_name)
        symbol_map = []
        for m in markets.values():
            if m['quote'] == 'USDT' and m['active'] and m.get('type') == 'swap' and m.get('linear'):
                api_sym = m['symbol']
                # Clean name for local storage (BTC/USDT:USDT -> BTCUSDT)
                clean_name = api_sym.split(':')[0].replace('/', '')
                symbol_map.append((api_sym, clean_name))
        
        # Sort by clean name for consistency
        symbol_map.sort(key=lambda x: x[1])
    except Exception as e:
        print(f"❌ Không thể lấy danh sách symbols từ Binance: {e}")
        return
    
    # Sinh Tử Môn: Lấy Context BTC
    print(f"   [Sync] BTC Context...")
    try:
        # Sử dụng paginated fetch để lấy đủ dữ liệu mapping với Altcoin
        ohlcv_btc_1h = fetch_ohlcv_paginated('BTC/USDT', '1h', lookback)
        btc_1h = pd.DataFrame(ohlcv_btc_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        # BỔ SUNG DÒNG NÀY: Bóp cổ bắt ép về Float
        btc_1h[['open', 'high', 'low', 'close', 'volume']] = btc_1h[['open', 'high', 'low', 'close', 'volume']].astype(float)

        btc_1h['timestamp'] = pd.to_datetime(btc_1h['timestamp'], unit='ms')
        btc_1h['ema_200'] = btc_1h['close'].ewm(span=200).mean()
        # ADX nội bộ cho BTC
        pdm = btc_1h['high'].diff(); mdm = -btc_1h['low'].diff()
        pdm = pdm.where((pdm > mdm) & (pdm > 0), 0); mdm = mdm.where((mdm > pdm) & (mdm > 0), 0)
        tr = pd.concat([btc_1h['high'] - btc_1h['low'], abs(btc_1h['high'] - btc_1h['close'].shift(1)), abs(btc_1h['low'] - btc_1h['close'].shift(1))], axis=1).max(axis=1)
        pdi = 100 * (pdm.rolling(14).mean() / tr.rolling(14).mean().replace(0, np.nan))
        mdi = 100 * (mdm.rolling(14).mean() / tr.rolling(14).mean().replace(0, np.nan))
        btc_1h['adx'] = (100 * abs(pdi - mdi) / (pdi + mdi).replace(0, np.nan)).rolling(14).mean()
        btc_1h['log_returns'] = np.log(btc_1h['close'] / btc_1h['close'].shift(1))
    except Exception as e:
        print(f"\n🚨 [TỬ HUYỆT] Mất kết nối dữ liệu BTC. Dừng Worker để bảo toàn vốn! Lỗi: {e}")
        return 

    # Load Model
    try:
        clf = joblib.load(MODEL_PATH)
        meta = joblib.load(META_PATH)
        features  = meta.get('features', [])
        threshold = meta.get('threshold', 0.6)
    except Exception as e:
        print(f"🚨 Không thể load Model AI: {e}")
        return

    all_signals = []
    existing_ids = set()
    
    # LOAD EXISTING SIGNALS & REPAIR IF NEEDED
    if os.path.exists(SIGNALS_JSON):
        try:
            with open(SIGNALS_JSON, 'r') as f:
                all_signals = json.load(f)
                
            # Migration/Repair: Fill missing keys for dashboard compatibility
            for s in all_signals:
                if 'tp_pct' not in s:
                    s['tp_pct'] = round(abs(s['tp_price'] - s['price']) / s['price'] * 100, 2)
                if 'sl_pct' not in s:
                    s['sl_pct'] = round(abs(s['sl_price'] - s['price']) / s['price'] * 100, 2)
                if 'atr_pct' not in s:
                    s['atr_pct'] = 2.0 # Default fallback
                if 'xai' not in s:
                    s['xai'] = {}
                if 'probas' not in s:
                    s['probas'] = [0.0, 0.5, 0.5] if s.get('side') == 'LONG' else [0.0, 0.5, 0.5]
            
            existing_ids = {s['id'] for s in all_signals}
        except:
            all_signals = []

    print(f"   [Sync] Đang quét {len(symbol_map)} symbols...")
    
    new_detected_count = 0
    updated_status_count = 0

    for i, (api_sym, sym) in enumerate(symbol_map):
        try:
            # Fetch Data (Paginated if lookback > 1)
            if lookback > 1:
                ohlcv_1h = fetch_ohlcv_paginated(api_sym, '1h', lookback)
            else:
                # Normal fast fetch for loop
                ohlcv_1h = fetch_ohlcv_with_retry(api_sym, '1h', 50) # fetch 50 to have context for updates
            
            if not ohlcv_1h: continue
                
            df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'], unit='ms')
            
            # 1D data for MTF (Keep it fast)
            ohlcv_1d = fetch_ohlcv_with_retry(api_sym, '1d', 50)
            df_1d = pd.DataFrame(ohlcv_1d, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'], unit='ms')
            df_1d.to_parquet(SYMBOLS_DIR_1D / f"{sym}.parquet", index=False)
            
            # Extract Features
            full_df = extract_features_live(df_1h, df_1d, btc_1h)
            full_df.to_parquet(SYMBOLS_DIR_1H / f"{sym}.parquet", index=False)
            
            # --- UPDATE STATUS FOR EXISTING SIGNALS FOR THIS SYMBOL ---
            for s in all_signals:
                if s['symbol'] == sym and s.get('is_win') is None:
                    # Check if Price hit TP/SL in the latest bars
                    # Use the last few bars from full_df
                    pending_bars = full_df[full_df['timestamp'] > pd.to_datetime(s['timestamp'])]
                    if not pending_bars.empty:
                        for _, f_row in pending_bars.iterrows():
                            if s['side'] == "LONG":
                                if f_row['low'] <= s['sl_price']:
                                    s['is_win'] = False; updated_status_count += 1; break
                                if f_row['high'] >= s['tp_price']:
                                    s['is_win'] = True; updated_status_count += 1; break
                            else: # SHORT
                                if f_row['high'] >= s['sl_price']:
                                    s['is_win'] = False; updated_status_count += 1; break
                                if f_row['low'] <= s['tp_price']:
                                    s['is_win'] = True; updated_status_count += 1; break

                # Stage 1: Closed Candle Check
                completed_df = full_df.iloc[:-1] # Always use COMPLETED candle
                if completed_df.empty: continue
                
                curr_row = completed_df.iloc[-1:] 
                ts = curr_row['timestamp'].iloc[0]
                sig_id = f"{sym}_{ts.strftime('%Y%m%d%H%M')}"
                
                if sig_id in existing_ids:
                    continue

                # Stage 1: Ignition Filter (EXACTLY as in train_sniper.py)
                # Note: df['volume_sma_20'] in extract_features_live is rolling(20).mean()
                # We need the value BEFORE this bar: .shift(1)
                vol_sma_series = completed_df['volume'].rolling(20).mean().shift(1)
                vol_sma_val = vol_sma_series.iloc[-1]
                
                close_val = curr_row['close'].iloc[0]
                open_val = curr_row['open'].iloc[0]
                rsi_val = curr_row['rsi_14'].iloc[0]
                ema20_val = curr_row['ema_20'].iloc[0]
                res50_val = curr_row['resistance_50'].iloc[0]
                
                cond_green_bar = (close_val > open_val) and (close_val > ema20_val)
                cond_body_size = ((close_val - open_val) / open_val) > 0.015
                cond_vol_ignition = (vol_sma_val * 1.5 < curr_row['volume'].iloc[0] < vol_sma_val * 4.0)
                cond_rsi_fresh = (55 <= rsi_val <= 72)
                
                dist_to_res = (res50_val - close_val) / (close_val + 1e-9)
                cond_near_res = dist_to_res > -0.05

                if not (cond_green_bar and cond_body_size and cond_vol_ignition and cond_rsi_fresh and cond_near_res):
                    continue
                
                # Tầng 2: AI Check
                X = curr_row[features].apply(pd.to_numeric, errors='coerce')
                        
                probas = clf.predict_proba(X)
                prob_long, prob_short = probas[0, 1], probas[0, 2]
                final_prob = max(prob_long, prob_short)
                
                if final_prob >= threshold:
                    side = "LONG" if prob_long > prob_short else "SHORT"
                    atr_pct = (curr_row['atr_14'].values[0] / curr_row['close'].values[0]) * 100
                    tp_pct_val = (atr_pct * 4.0)
                    sl_pct_val = (atr_pct * 1.5) if side == "LONG" else (atr_pct * 2.0)
                    
                    entry_price = float(curr_row['close'].iloc[0])
                    tp_price = entry_price * (1 + (tp_pct_val/100)) if side == "LONG" else entry_price * (1 - (tp_pct_val/100))
                    sl_price = entry_price * (1 - (sl_pct_val/100)) if side == "LONG" else entry_price * (1 + (sl_pct_val/100))

                    # Evaluate historical excursion if not live
                    hist_mfe, hist_mae = 0.0, 0.0
                    is_win = None
                    future_bars = full_df.iloc[lookback_df.index[idx]+1:]
                    if not future_bars.empty:
                        if side == "LONG":
                            hist_mfe = (future_bars['high'].max() - entry_price) / entry_price * 100
                            hist_mae = (future_bars['low'].min() - entry_price) / entry_price * 100
                            for _, f_row in future_bars.iterrows():
                                if f_row['low'] <= sl_price: is_win = False; break
                                if f_row['high'] >= tp_price: is_win = True; break
                        else:
                            hist_mfe = (entry_price - future_bars['low'].min()) / entry_price * 100
                            hist_mae = (entry_price - future_bars['high'].max()) / entry_price * 100
                            for _, f_row in future_bars.iterrows():
                                if f_row['high'] >= sl_price: is_win = False; break
                                if f_row['low'] <= tp_price: is_win = True; break

                    all_signals.append({
                        'id': sig_id,
                        'symbol': sym,
                        'timestamp': ts.isoformat(),
                        'side': side,
                        'price': entry_price,
                        'prob': float(final_prob),
                        'tp_price': float(tp_price),
                        'sl_price': float(sl_price),
                        'tp_pct': round(float(tp_pct_val), 2),
                        'sl_pct': round(float(sl_pct_val), 2),
                        'atr_pct': round(float(atr_pct), 2),
                        'is_win': is_win,
                        'mfe': round(hist_mfe, 2),
                        'mae': round(hist_mae, 2),
                        'xai': {
                            'upper_wick_ratio': float(curr_row['upper_wick_ratio'].iloc[0]),
                            'dist_to_ema50_atr': float(curr_row['dist_to_ema50_atr'].iloc[0]),
                            'volume_ratio': float(curr_row['volume_ratio'].iloc[0]),
                            'rsi_14': float(curr_row['rsi_14'].iloc[0]),
                            'vol_compression': float(curr_row['vol_compression'].iloc[0]),
                            'volume_zscore': float(curr_row['volume_zscore'].iloc[0]),
                            'btc_is_bull_regime': int(curr_row['btc_is_bull_regime'].iloc[0]) if 'btc_is_bull_regime' in curr_row.columns else 0,
                            'adx': float(curr_row['adx'].iloc[0]) if 'adx' in curr_row.columns else 20.0,
                            'macd_slope': float(curr_row['macd_slope'].iloc[0]) if 'macd_slope' in curr_row.columns else 0.0,
                            'stoch_k': float(curr_row['stoch_k'].iloc[0]) if 'stoch_k' in curr_row.columns else 50.0
                        },
                        'probas': [float(p) for p in probas[0]]
                    })
                    existing_ids.add(sig_id)
                    new_detected_count += 1
                    print(f"   [AI] 🔥 KÈO DETECTED: {sym} {side} ({final_prob:.2%})", flush=True)

            if i > 0 and i % 50 == 0:
                print(f"   [Sync] Tiến độ quét: {i}/{len(symbol_map)}", flush=True)
            
            gc.collect()

        except Exception as e:
            # print(f" ❌ [LỖI PIPELINE] {sym} gặp sự cố: {str(e)}", flush=True)
            continue

    # SORT BY TIMESTAMP NEWEST FIRST
    all_signals.sort(key=lambda x: x['timestamp'], reverse=True)

    # SAVE RESULTS
    with open(SIGNALS_JSON, 'w') as f:
        json.dump(all_signals, f, indent=4)
    
    stats = {
        'last_scan': datetime.now().strftime('%H:%M:%S'),
        'btc_regime': 'Uptrend' if (btc_1h is not None and btc_1h['close'].iloc[-1] > btc_1h['ema_200'].iloc[-1]) else 'Downtrend',
        'market_atr': float(full_df['dist_to_ema50_atr'].abs().mean() if 'full_df' in locals() else 0.02)
    }
    with open(STATS_JSON, 'w') as f:
        json.dump(stats, f, indent=4)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Sync Xong. Mới: +{new_detected_count} | Cập nhật Win/Loss: {updated_status_count} | Tổng Database: {len(all_signals)}")

def sleep_until_next_hour():
    """Calculates time until the next 1h candle and sleeps."""
    now = datetime.now()
    # Next hour
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    # Add 45s buffer for exchange data stability
    target_time = next_hour + timedelta(seconds=45)
    
    sleep_seconds = (target_time - now).total_seconds()
    if sleep_seconds > 0:
        print(f"[{now.strftime('%H:%M:%S')}] 💤 Đang nghỉ {sleep_seconds/60:.1f} phút cho đến nến tiếp theo ({target_time.strftime('%H:%M:%S')})...", flush=True)
        time.sleep(sleep_seconds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Sniper Sync Worker")
    parser.add_argument("--lookback", type=int, default=1, help="Số lượng nến cũ cần quét (default: 1)")
    args = parser.parse_args()

    print("🚀 Sniper Sync Worker - Chế độ Loop 24/7 đã kích hoạt.", flush=True)
    
    # First run immediately
    sync_all(lookback=args.lookback)
    
    # Then loop forever
    while True:
        sleep_until_next_hour()
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🔔 Bắt đầu chu kỳ quét mới...", flush=True)
        sync_all(lookback=1) # Only scan the latest candle in loop