#!/usr/bin/env python3
import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import concurrent.futures
from zoneinfo import ZoneInfo
import warnings
warnings.filterwarnings('ignore')

# Thêm root path vào sys.path để import được các module khác
BASE_DIR = Path(r"d:\Code\Projects\self-projects\macd-overlay - Copy")
sys.path.append(str(BASE_DIR))

try:
    from ccxt_data_processor import CCXTDataProcessor
except ImportError:
    print("❌ Lỗi: Không tìm thấy CCXTDataProcessor. Hãy đảm bảo bạn đang chạy từ đúng thư mục.")
    sys.exit(1)

# ============================================================
# CONFIG & PATHS
# ============================================================
MODEL_PATH = BASE_DIR / "ml" / "training" / "models" / "1h" / "ensemble_lgbm_tabular.joblib"
META_PATH = BASE_DIR / "ml" / "training" / "models" / "1h" / "ensemble_meta.joblib"

MODEL_FEATURES_DEFAULT = [
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

# ============================================================
# HELPER FUNCTIONS (LOCAL TO AVOID IMPORT ISSUES)
# ============================================================

def calculate_rsi_local(prices, period=14):
    d = prices.diff()
    g = d.where(d > 0, 0).rolling(period).mean()
    l = (-d.where(d < 0, 0)).rolling(period).mean()
    return 100 - (100 / (1 + g / (l.replace(0, np.nan) + 1e-9)))

def calculate_macd_local(df, fast=12, slow=26, signal=9):
    df = df.copy()
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    return df

def calculate_adx_local(df, period=14):
    df = df.copy()
    tr = pd.concat([df['high'] - df['low'], abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))], axis=1).max(axis=1)
    pdm = df['high'].diff(); mdm = -df['low'].diff()
    pdm = pdm.where((pdm > mdm) & (pdm > 0), 0)
    mdm = mdm.where((mdm > pdm) & (mdm > 0), 0)
    atr_s = tr.rolling(period).mean()
    pdi = 100 * (pdm.rolling(period).mean() / atr_s.replace(0, np.nan))
    mdi = 100 * (mdm.rolling(period).mean() / atr_s.replace(0, np.nan))
    adx = (100 * abs(pdi - mdi) / (pdi + mdi).replace(0, np.nan)).rolling(period).mean()
    return adx

def calculate_all_features_live(df_1h, df_1d, btc_df):
    """Tính toán đầy đủ MODEL_FEATURES cho nến cuối cùng."""
    df = df_1h.copy()
    
    # 1. Price Basics
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['candle_range'] = df['high'] - df['low'] + 1e-9
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick_ratio_current'] = df['lower_wick'] / df['candle_range']
    df['upper_wick_ratio_current'] = df['upper_wick'] / df['candle_range']
    
    # 2. Indicators
    for p in [21, 50, 200]:
        df[f'ema_{p}'] = df['close'].ewm(span=p).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    
    # 3. Volatility & Volume
    df['volatility_14'] = df['log_returns'].rolling(14).std()
    df['vol_sma_14'] = df['volatility_14'].rolling(14).mean()
    df['vol_compression'] = df['volatility_14'] / (df['vol_sma_14'] + 1e-9)
    
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_std_20'] = df['volume'].rolling(20).std()
    df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-9)
    df['volume_zscore'] = (df['volume'] - df['volume_sma_20']) / (df['volume_std_20'] + 1e-9)
    df['volume_trend'] = df['volume'].rolling(7).mean() / (df['volume'].rolling(21).mean() + 1e-9)
    
    # 4. RSI & Stoch
    df['rsi_14'] = calculate_rsi_local(df['close'], 14)
    df['rsi_slope'] = df['rsi_14'].diff(3)
    l14 = df['low'].rolling(14).min(); h14 = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * (df['close'] - l14) / (h14 - l14).replace(0, np.nan)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    
    # 5. Momentum
    df['roc_7'] = df['close'].pct_change(7)
    df['roc_14'] = df['close'].pct_change(14)
    df['momentum_30'] = df['close'].pct_change(30)
    df['price_vs_sma_30'] = df['close'] / (df['sma_30'] + 1e-9)
    
    # 6. ADX
    df['adx'] = calculate_adx_local(df, 14)
    
    # 7. Distances
    df['dist_to_high_30d'] = (df['close'] - df['high'].rolling(30).max()) / df['close']
    df['dist_to_low_30d'] = (df['close'] - df['low'].rolling(30).min()) / df['close']
    for e in [21, 50, 200]:
        df[f'dist_to_ema_{e}_pct'] = (df['close'] - df[f'ema_{e}']) / df['close']
        
    # 8. MACD
    df = calculate_macd_local(df)
    df['macd_slope'] = df['macd'].diff(); df['macd_acceleration'] = df['macd_slope'].diff()
    
    # 9. Structure
    bb_mid = df['close'].rolling(20).mean(); bb_std = df['close'].rolling(20).std()
    bb_wd = (bb_mid + 2 * bb_std - (bb_mid - 2 * bb_std)) / (bb_mid + 1e-9)
    df['bb_squeeze'] = (bb_wd < bb_wd.rolling(20).quantile(0.2)).astype(int)
    df['vwap_30d'] = (df['close'] * df['volume']).rolling(30).sum() / (df['volume'].rolling(30).sum() + 1e-9)
    df['above_poc'] = (df['close'] > df['vwap_30d']).astype(int)
    df['micro_volume'] = df['volume'] / (df['volume'].rolling(5).mean() + 1e-9)
    df['price_accel'] = df['close'].pct_change(1) / (df['close'].pct_change(4).replace(0, np.nan) + 1e-9)
    df['order_flow_proxy'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)
    
    # 10. Time features
    df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
    
    # 11. BTC & Trend State
    df['trend_state'] = np.where(df['close'] > df['sma_50'], 1, np.where(df['close'] < df['sma_50'], -1, 0))
    df['is_trending'] = (df['adx'] > 25).astype(int)
    df['is_volatile'] = (df['vol_compression'] > 1.5).astype(int)
    
    # Merging BTC context
    if btc_df is not None and not btc_df.empty:
        btc_latest = btc_df.iloc[-1]
        df['btc_is_bull_regime'] = 1 if btc_latest['close'] > btc_latest['sma_200'] else 0
        df['btc_trend_strength'] = 1 if btc_latest['adx'] > 25 else 0
        df['btc_returns'] = btc_latest['log_returns']
        df['rs_vs_btc'] = df['log_returns'] - df['btc_returns']
        df['rs_vs_btc_sma7'] = df['rs_vs_btc'].rolling(7).mean()
        df['btc_corr'] = df['log_returns'].rolling(14).corr(pd.Series([btc_latest['log_returns']]*len(df))).fillna(0)
    else:
        df['btc_is_bull_regime'] = 0
        df['btc_trend_strength'] = 0
        df['rs_vs_btc'] = 0
        df['rs_vs_btc_sma7'] = 0
        df['btc_corr'] = 0

    # 12. MTF (1D)
    if df_1d is not None and not df_1d.empty:
        ema_200_1d = df_1d['close'].ewm(span=200).mean().iloc[-1]
        df['ema_200_1d_dist'] = (df['close'] - ema_200_1d) / df['close']
        df['rsi_14_1d'] = calculate_rsi_local(df_1d['close'], 14).iloc[-1]
    else:
        df['ema_200_1d_dist'] = 0
        df['rsi_14_1d'] = 50

    return df


# ============================================================
# MAIN SCANNER CLASS
# ============================================================

class SniperRealtimeScanner:
    def __init__(self):
        print("📦 Khởi tạo Scanner...")
        self.processor = CCXTDataProcessor('binance', use_futures=True)
        self.load_assets()
        
    def load_assets(self):
        try:
            self.clf = joblib.load(MODEL_PATH)
            meta = joblib.load(META_PATH)
            if isinstance(meta, dict):
                self.features = meta.get('features', MODEL_FEATURES_DEFAULT)
                self.threshold = meta.get('threshold', 0.7057)
            else:
                self.features = meta
                self.threshold = 0.7057
            print(f"✅ Đã load Model & Meta. Threshold: {self.threshold:.4f}")
        except Exception as e:
            print(f"❌ Lỗi load model: {e}")
            self.clf = None
            self.features = MODEL_FEATURES_DEFAULT
            self.threshold = 0.7057

    def get_all_usdt_symbols(self):
        try:
            markets = self.processor.client.load_markets()
            symbols = [
                m['id'] for m in markets.values()
                if m['linear'] and m['active'] and m['quote'] == 'USDT' and m['type'] == 'swap'
            ]
            # Standardizing to unified format
            clean_symbols = [s.replace('USDT', '/USDT:USDT') for s in symbols]
            return clean_symbols
        except Exception as e:
            print(f"❌ Lỗi lấy symbols: {e}")
            return []

    def scan_single_symbol(self, symbol, btc_df, lookback=1):
        try:
            # 1. Fetch 1h data
            clean_sym = symbol.split(':')[0].replace('/', '')
            df_1h = self.processor.get_historical_data(clean_sym, '1h', start_date="15 days ago UTC")
            if df_1h is None or len(df_1h) < 200: return None
            
            # Pre-calculate basics for the whole dataframe
            df_1h['rsi_14'] = calculate_rsi_local(df_1h['close'], 14)
            df_1h['ema_20'] = df_1h['close'].ewm(span=20).mean()
            df_1h['ema_50'] = df_1h['close'].ewm(span=50).mean()
            df_1h['res_50'] = df_1h['high'].rolling(50).max().shift(1)
            df_1h['vol_sma_20'] = df_1h['volume'].rolling(20).mean().shift(1)
            
            # Fetch 1d data once
            df_1d = self.processor.get_historical_data(clean_sym, '1d', start_date="250 days ago UTC")
            
            # Calculate all AI features for the whole dataframe
            df_feat = calculate_all_features_live(df_1h, df_1d, btc_df)
            
            found = []
            # Duyệt ngược từ nến mới nhất về trước theo lookback
            for i in range(1, lookback + 1):
                idx = -i
                curr = df_feat.iloc[idx]
                
                # Tầng 1: Rule-based Filter
                cond_uptrend = (curr['ema_20'] > curr['ema_50']) and (curr['low'] > curr['ema_50'])
                cond_rsi = curr['rsi_14'] > 60
                cond_volume = curr['volume'] > (curr['vol_sma_20'] * 2.5)
                cond_breakout = curr['close'] > curr['res_50']
                
                if cond_uptrend and cond_rsi and cond_volume and cond_breakout:
                    # Tầng 2: AI Sniper
                    X = pd.DataFrame([curr[self.features]]).apply(pd.to_numeric, errors='coerce').fillna(0)
                    score = self.clf.predict_proba(X)[:, 1][0]
                    
                    if score >= self.threshold:
                        found.append({
                            'symbol': clean_sym,
                            'price': curr['close'],
                            'ai_score': score,
                            'rsi': curr['rsi_14'],
                            'vol_ratio': curr['volume'] / curr['vol_sma_20'] if curr['vol_sma_20'] > 0 else 0,
                            'last_update': curr['timestamp'].strftime('%H:%M')
                        })
            
            return found if found else None
            
        except Exception:
            pass
        return None


    def run_loop(self, limit=None, workers=10):
        print(f"🚀 Bắt đầu quét {'FULL' if not limit else f'TOP {limit}'} symbols (Binance USDT-M)...")
        while True:
            try:
                start_scan = time.time()
                symbols = self.get_all_usdt_symbols()
                if limit:
                    symbols = symbols[:limit]
                
                print(f"\n🔍 [{datetime.now().strftime('%H:%M:%S')}] Đang quét {len(symbols)} symbols...")
                
                # Fetch BTC context
                btc_df = self.processor.get_historical_data('BTCUSDT', '1h', start_date="15 days ago UTC")
                if btc_df is not None and len(btc_df) >= 200:
                    btc_df = calculate_macd_local(btc_df)
                    btc_df['log_returns'] = np.log(btc_df['close'] / btc_df['close'].shift(1))
                    btc_df['adx'] = calculate_adx_local(btc_df, 14)
                    btc_df['sma_200'] = btc_df['close'].rolling(200).mean()
                else:
                    btc_df = None
                
                found_signals = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [executor.submit(self.scan_single_symbol, s, btc_df, getattr(self, 'lookback', 1)) for s in symbols]
                    for future in concurrent.futures.as_completed(futures):
                        res = future.result()
                        if res:
                            if isinstance(res, list):
                                found_signals.extend(res)
                            else:
                                found_signals.append(res)

                
                print(f"⏱️ Quét xong {len(symbols)} symbols trong {time.time() - start_scan:.1f}s")
                if found_signals:
                    df_sig = pd.DataFrame(found_signals).sort_values('ai_score', ascending=False)
                    print("\n🔥 TÍN HIỆU AI SNIPER REAL-TIME:")
                    print(df_sig.to_string(index=False))
                else:
                    print("😴 Không tìm thấy tín hiệu nào thỏa mãn Rule-based + AI.")
                
                time.sleep(120) # Quét mỗi 2 phút
                
            except KeyboardInterrupt:
                print("\n🛑 Đã dừng Scanner.")
                break
            except Exception as e:
                print(f"❌ Lỗi hệ thống: {e}")
                time.sleep(10)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Sniper Real-time Scanner")
    parser.add_argument("--limit", type=int, help="Limit number of symbols to scan")
    parser.add_argument("--workers", type=int, default=15, help="Number of parallel workers")
    parser.add_argument("--lookback", type=int, default=1, help="Number of previous bars to scan")
    args = parser.parse_args()

    scanner = SniperRealtimeScanner()
    scanner.lookback = args.lookback
    scanner.run_loop(limit=args.limit, workers=args.workers)

