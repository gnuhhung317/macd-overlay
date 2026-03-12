import pandas as pd
import numpy as np
import time
import joblib
from typing import List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Ensure root is in path
sys.path.append(str(Path(__file__).parent.parent))
from data_processor import BinanceDataProcessor

class SniperScanner:
    def __init__(self, config=None, data_processor=None):
        self.config = config
        self.processor = data_processor if data_processor else BinanceDataProcessor(use_futures=True)
        
        # Absolute root path detection
        self.base_dir = Path(__file__).resolve().parent.parent
        self.model_path = self.base_dir / "ml" / "training" / "models" / "1h" / "ensemble_lgbm_tabular.joblib"
        self.meta_path = self.base_dir / "ml" / "training" / "models" / "1h" / "ensemble_meta.joblib"
        
        print(f"🔍 [SniperScanner] Checking models at: {self.model_path}")
        
        self.clf = None
        self.features = []
        self.threshold = 0.6
        self._load_model()

    def _load_model(self):
        try:
            if not self.model_path.exists() or not self.meta_path.exists():
                print(f"❌ [SniperScanner] Missing model or meta file at {self.model_path.parent}")
                return
            
            meta = joblib.load(self.meta_path)
            self.clf = joblib.load(self.model_path)
            self.features = meta.get('features', []) if isinstance(meta, dict) else meta
            self.threshold = meta.get('threshold', 0.6)
            
            # Allow config threshold to override model default if provided
            if self.config and hasattr(self.config.strategy, 'entry_threshold'):
                if self.config.strategy.entry_threshold > 0:
                    self.threshold = self.config.strategy.entry_threshold
                    
            print(f"✅ [SniperScanner] Model loaded. Features: {len(self.features)}, Threshold: {self.threshold:.2f}")
        except Exception as e:
            print(f"❌ [SniperScanner] Error loading model: {e}")

    def calculate_features_sniper(self, df_1h: pd.DataFrame, df_1d: pd.DataFrame = None, btc_df: pd.DataFrame = None) -> pd.DataFrame:
        """Calculate missing features for backtesting logic if not present."""
        df = df_1h.copy()
        
        # 1. Price Basics & Simple Returns
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # 2. Indicators (EMA, SMA)
        for p in [20, 21, 50, 200]:
            df[f'ema_{p}'] = df['close'].ewm(span=p).mean()
        df['sma_30'] = df['close'].rolling(30).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # 3. ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()
        if 'atr_pct' not in df.columns:
            df['atr_pct'] = (df['atr_14'] / df['close']) * 100
        
        # 4. Volatility
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility_14'] = df['log_returns'].rolling(14).std()
        
        # Volatility Compression (Match original feature.py: StdDev ratio)
        df['vol_sma_14'] = df['volatility_14'].rolling(14).mean()
        df['vol_compression'] = df['volatility_14'] / (df['vol_sma_14'] + 1e-9)
        
        # 5. Volume
        # Match data_pipeline.py: volume / volume_sma_14
        df['volume_sma_14'] = df['volume'].rolling(14).mean()
        df['volume_std_14'] = df['volume'].rolling(14).std()
        df['volume_ratio'] = df['volume'] / (df['volume_sma_14'] + 1e-9)
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_std_20'] = df['volume'].rolling(20).std()
        df['volume_zscore'] = (df['volume'] - df['volume_sma_20']) / (df['volume_std_20'] + 1e-9)
        df['volume_trend'] = df['volume'].rolling(7).mean() / (df['volume'].rolling(21).mean() + 1e-9)
        
        # 6. RSI & Stoch
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        df['rsi_14'] = df['rsi_14'].fillna(50)
        # Match original feature.py: diff(3)
        df['rsi_slope'] = df['rsi_14'].diff(3)
        
        l14 = df['low'].rolling(14).min(); h14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - l14) / (h14 - l14).replace(0, np.nan)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # 7. Momentum & Distance
        df['roc_7'] = df['close'].pct_change(7)
        df['roc_14'] = df['close'].pct_change(14)
        df['momentum_30'] = df['close'].pct_change(30)
        df['price_vs_sma_30'] = df['close'] / (df['sma_30'] + 1e-9)
        
        df['dist_to_high_30d'] = (df['close'] - df['high'].rolling(30).max()) / (df['close'] + 1e-9)
        df['dist_to_low_30d'] = (df['close'] - df['low'].rolling(30).min()) / (df['close'] + 1e-9)
        for e in [21, 50, 200]:
            # Match original feature.py: (P - EMA) / P
            df[f'dist_to_ema_{e}_pct'] = (df['close'] - df[f'ema_{e}']) / (df['close'] + 1e-9)
            
        # 8. MACD
        ema_fast = df['close'].ewm(span=12, adjust=False).mean()
        ema_slow = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_slope'] = df['macd'].diff()
        df['macd_acceleration'] = df['macd_slope'].diff()
        
        # 9. Sniper specialized features
        df['upper_wick_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-9)
        df['dist_to_ema50_atr'] = (df['close'] - df['ema_50']) / (df['atr_14'] + 1e-9)
        df['vol_acceleration'] = df['volume'] / (df['volume'].shift(1) + 1e-9)
        df['vol_ratio_alpha'] = df['volume_ratio'] * df['volatility_14']
        
        # 10. ADX
        pdm = df['high'].diff(); mdm = -df['low'].diff()
        pdm = pdm.where((pdm > mdm) & (pdm > 0), 0)
        mdm = mdm.where((mdm > pdm) & (mdm > 0), 0)
        atr_s = tr.rolling(14).mean()
        pdi = 100 * (pdm.rolling(14).mean() / atr_s.replace(0, np.nan))
        mdi = 100 * (mdm.rolling(14).mean() / atr_s.replace(0, np.nan))
        df['adx'] = (100 * abs(pdi - mdi) / (pdi + mdi).replace(0, np.nan)).rolling(14).mean()
        
        # 11. Time features
        df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
        
        # 12. Structure & Flow
        bb_mid = df['close'].rolling(20).mean(); bb_std = df['close'].rolling(20).std()
        bb_wd = (bb_mid + 2 * bb_std - (bb_mid - 2 * bb_std)) / (bb_mid + 1e-9)
        df['bb_squeeze'] = (bb_wd < bb_wd.rolling(20).quantile(0.2)).astype(int)
        df['vwap_30d'] = (df['close'] * df['volume']).rolling(30).sum() / (df['volume'].rolling(30).sum() + 1e-9)
        df['above_poc'] = (df['close'] > df['vwap_30d']).astype(int)
        df['micro_volume'] = df['volume'] / (df['volume'].rolling(5).mean() + 1e-9)
        df['price_accel'] = df['close'].pct_change(1) / (df['close'].pct_change(4).replace(0, np.nan) + 1e-9)
        df['order_flow_proxy'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)
        
        # 13. Regime
        df['trend_state'] = np.where(df['close'] > df['sma_50'], 1, np.where(df['close'] < df['sma_50'], -1, 0))
        df['is_trending'] = (df['adx'] > 25).astype(int)
        # Match original feature.py logic
        df['is_volatile'] = (df['vol_compression'] > 1.5).astype(int)
        
        # 14. BTC Context Integration
        if btc_df is not None and not btc_df.empty:
            # We must MERGE btc_df with df on timestamp to properly calculate rolling correlation and relative strength
            # Create a temporary btc subset
            btc_sub = btc_df[['timestamp', 'close', 'sma_200', 'adx', 'log_returns']].copy()
            btc_sub.rename(columns={'close': 'btc_close', 'sma_200': 'btc_sma_200', 'adx': 'btc_adx', 'log_returns': 'btc_log_returns'}, inplace=True)
            
            # Merge
            df = df.merge(btc_sub, on='timestamp', how='left')
            
            # Forward fill simply in case of missing 1h BTC candles
            df['btc_close'] = df['btc_close'].ffill()
            df['btc_sma_200'] = df['btc_sma_200'].ffill()
            df['btc_adx'] = df['btc_adx'].ffill()
            df['btc_log_returns'] = df['btc_log_returns'].ffill().fillna(0)
            
            # Now calculate accurate rolling correlations on the merged series
            df['btc_is_bull_regime'] = (df['btc_close'] > df['btc_sma_200']).astype(int)
            df['btc_trend_strength'] = (df['btc_adx'] > 25).astype(int)
            df['btc_returns'] = df['btc_log_returns']
            df['rs_vs_btc'] = df['log_returns'] - df['btc_returns']
            df['rs_vs_btc_sma7'] = df['rs_vs_btc'].rolling(7).mean()
            df['btc_corr'] = df['log_returns'].rolling(14).corr(df['btc_returns']).fillna(0)
            
            # Drop temporary merge columns
            df.drop(columns=['btc_close', 'btc_sma_200', 'btc_adx', 'btc_log_returns'], inplace=True)
        else:
            df['btc_is_bull_regime'] = 0
            df['btc_trend_strength'] = 0
            df['btc_returns'] = 0
            df['rs_vs_btc'] = 0
            df['rs_vs_btc_sma7'] = 0
            df['btc_corr'] = 0
            
        # 15. MTF (1D) Integration (Match train_sniper.py shift(1) logic)
        if df_1d is not None and not df_1d.empty:
            # Shift(1) to get the last fully closed Day context
            ema_200_1d_series = df_1d['close'].ewm(span=200).mean().shift(1)
            ema_200_1d = ema_200_1d_series.iloc[-1]
            df['ema_200_1d_dist'] = (df['close'] - ema_200_1d) / df['close']
            
            # Use the same SMA-based RSI calculation as sync_worker/train_sniper
            delta = df_1d['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-9)
            rsi_14_1d_series = (100 - (100 / (1 + rs))).shift(1)
            
            df['rsi_14_1d'] = rsi_14_1d_series.iloc[-1]
        else:
            df['ema_200_1d_dist'] = 0
            df['rsi_14_1d'] = 50
            
        return df

    def scan(self, symbols: List[str], timeframe: str, lookback_days: int = 4) -> List[Dict[str, Any]]:
        """
        Scan symbols for Sniper Signals using the latest completed candle.
        """
        if not self.clf:
            print("[SniperScanner] Model missing, cannot scan.")
            return []

        signals = []
        # Fetch ~200 candles to ensure EMA200 and long-term MAs are stable
        tf_hours = 1 if timeframe == '1h' else 4
        buffer_days = int((200 * tf_hours) / 24) + 1
        fetch_start = f"{lookback_days + buffer_days} days ago UTC"
        
        # 1. Fetch BTC Macro Context once
        btc_df = None
        try:
            btc_df = self.processor.get_historical_data('BTCUSDT', timeframe, fetch_start, 'now UTC')
            if btc_df is not None and not btc_df.empty:
                btc_df['log_returns'] = np.log(btc_df['close'] / btc_df['close'].shift(1))
                btc_df['sma_200'] = btc_df['close'].rolling(200).mean()
                
                # Setup ADX for BTC 
                tr = pd.concat([btc_df['high'] - btc_df['low'], abs(btc_df['high'] - btc_df['close'].shift(1)), abs(btc_df['low'] - btc_df['close'].shift(1))], axis=1).max(axis=1)
                pdm = btc_df['high'].diff(); mdm = -btc_df['low'].diff()
                pdm = pdm.where((pdm > mdm) & (pdm > 0), 0)
                mdm = mdm.where((mdm > pdm) & (mdm > 0), 0)
                atr_s = tr.rolling(14).mean()
                pdi = 100 * (pdm.rolling(14).mean() / atr_s.replace(0, np.nan))
                mdi = 100 * (mdm.rolling(14).mean() / atr_s.replace(0, np.nan))
                btc_df['adx'] = (100 * abs(pdi - mdi) / (pdi + mdi).replace(0, np.nan)).rolling(14).mean()
        except Exception as e:
            print(f"[SniperScanner] Failed to fetch BTC context: {e}")
        print(f"[SniperScanner] Fetched BTC context: {btc_df}")
        total_symbols = len(symbols)
        start_time = time.time()
        
        for i, symbol in enumerate(symbols):
            try:
                # Progress logging (every 10%)
                if total_symbols >= 10 and (i + 1) % (total_symbols // 10) == 0:
                    percent = ((i + 1) / total_symbols) * 100
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (i + 1)
                    rem_time = avg_time * (total_symbols - (i + 1))
                    print(f"⏳ [Scanner] Progress: {percent:.0f}% ({i+1}/{total_symbols}) | Elapsed: {elapsed:.1f}s | Est. Rem: {rem_time:.1f}s")

                time.sleep(0.05) # Reduced Rate Limit Protection (slightly faster)
                
                # Fetch 1H base data
                df = self.processor.get_historical_data(symbol, timeframe, fetch_start, 'now UTC')
                if df.empty or len(df) < 50: continue
                
                # PRELIMINARY FILTER (Stage 0): Ignition Bar check on 1H ONLY
                # This avoids unnecessary 1D fetches for most coins
                last_idx = df.index[-1]
                # We check the PREVIOUS candle (closed)
                closed_candle_idx = df.index[-2] if len(df) > 1 else last_idx
                row = df.loc[closed_candle_idx]
                
                vol_sma = df['volume'].rolling(20).mean().shift(1).loc[closed_candle_idx]
                
                cond_green_bar = (row['close'] > row['open']) and (row['close'] > row['close'] * 0.98) # Basic green check
                cond_vol_ignition = (vol_sma * 1.5 < row['volume']) # Basic volume check
                
                # If it doesn't even look like an ignition bar, skip the 1D fetch
                if not (cond_green_bar and cond_vol_ignition):
                    continue

                # Fetch 1D target data for Context (LAZY FETCH)
                df_1d = None
                try:
                    df_1d = self.processor.get_historical_data(symbol, '1d', "250 days ago UTC", 'now UTC')
                except:
                    pass
                
                live_price = df['close'].iloc[-1]
                
                # Use only completed candles for robust signal generation
                df_calc = df.iloc[:-1].copy()
                
                df_calc = self.calculate_features_sniper(df_calc, df_1d, btc_df)
                
                # Focus only on the VERY LAST completed candle
                last_calc_idx = df_calc.index[-1]
                row = df_calc.loc[last_calc_idx]
                
                # Stage 1 Filter: Ignition Bar (Full Check)
                vol_sma = df_calc['volume'].rolling(20).mean().shift(1).loc[last_calc_idx]
                resistance_50 = df_calc['high'].rolling(50).max().shift(1).loc[last_calc_idx]
                
                cond_green_bar = (row['close'] > row['open']) and (row['close'] > row['ema_20'])
                cond_body_size = ((row['close'] - row['open']) / row['open']) > 0.015
                cond_vol_ignition = (vol_sma * 1.5 < row['volume'] < vol_sma * 4.0)
                cond_rsi_fresh = (55 <= row['rsi_14'] <= 72)
                
                dist_to_res = (resistance_50 - row['close']) / (row['close'] + 1e-9)
                # cond_near_res = dist_to_res > -0.05 (Removed: Backtest shows this kills super-winners)
                cond_near_res = True 
                
                if not (cond_green_bar and cond_body_size and cond_vol_ignition and cond_rsi_fresh and cond_near_res):
                    continue 

                # Stage 2: ML Scoring
                x_input = df_calc.loc[[last_calc_idx], self.features].apply(pd.to_numeric, errors='coerce').fillna(0)
                probas = self.clf.predict_proba(x_input)[0]
                
                prob_long = probas[1]
                prob_short = probas[2]
                
                if prob_long > self.threshold or prob_short > self.threshold:
                    trade_type = 'LONG' if prob_long > self.threshold else 'SHORT'
                    confidence = prob_long if trade_type == 'LONG' else prob_short
                    
                    atr_val = row['atr_14']
                    close_price = row['close']
                    
                    strat_cfg = self.config.strategy if self.config else None
                    if strat_cfg:
                        sl_mul_long = getattr(strat_cfg, 'sl_atr_multiplier_long', 1.0)
                        tp_mul_long = getattr(strat_cfg, 'tp_atr_multiplier_long', 2.0)
                        sl_mul_short = getattr(strat_cfg, 'sl_atr_multiplier_short', 1.5)
                        tp_mul_short = getattr(strat_cfg, 'tp_atr_multiplier_short', 2.5)
                    else:
                        sl_mul_long, tp_mul_long = 1.0, 2.0
                        sl_mul_short, tp_mul_short = 1.5, 2.5
                    
                    if trade_type == 'LONG':
                        limit_price = close_price + (getattr(strat_cfg, 'long_atr_offset', -0.1) * atr_val)
                        sl_price = limit_price - (sl_mul_long * atr_val)
                        tp_price = limit_price + (tp_mul_long * atr_val)
                    else: # SHORT
                        limit_price = close_price + (getattr(strat_cfg, 'short_atr_offset', 0.5) * atr_val)
                        sl_price = limit_price + (sl_mul_short * atr_val)
                        tp_price = limit_price - (tp_mul_short * atr_val)
                        
                    sl_pct = abs(limit_price - sl_price) / limit_price
                    tp_pct = abs(tp_price - limit_price) / limit_price

                    risk_reward = tp_pct / sl_pct if sl_pct > 0 else 0
                    
                    signal_data = {
                        'symbol': symbol,
                        'type': trade_type,
                        'timestamp': str(row['timestamp']),
                        'confidence': float(confidence),
                        'status': "SNIPER",
                        'signal_price': float(close_price),
                        'limit_price': float(limit_price),
                        'current_price': float(live_price),
                        'refined_score': 1.0,
                        'sl_pct': float(sl_pct),
                        'tp_pct': float(tp_pct),
                        'risk_reward': float(risk_reward),
                        'meta': {
                            'origin': 'sniper_scanner',
                            'atr_val': float(atr_val),
                            'ignition_volume': float(row['volume']),
                            'ema_20': float(row['ema_20'])
                        }
                    }
                    
                    signals.append(signal_data)
                    
            except Exception as e:
                err_msg = str(e)
                if "-1003" in err_msg:
                    print(f"⚠️ RATE LIMIT HIT! Sleeping for 60s... ({err_msg})")
                    time.sleep(60)
                else:
                    print(f"Error scanning {symbol}: {e}")
                continue
                
        return signals
