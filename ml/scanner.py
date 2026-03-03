import pandas as pd
import numpy as np
import time
from typing import List, Dict, Any
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add root to path to find data_processor
sys.path.append(str(Path(__file__).parent.parent)) 
from ml.inference import InferenceEngine
from data_processor import BinanceDataProcessor

class SmartScanner:
    def __init__(self, config=None, data_processor=None):
        self.config = config
        # Reuse processor if provided, else create new
        self.processor = data_processor if data_processor else BinanceDataProcessor(use_futures=True)
        # Cache engines to avoid reloading
        self._engines = {} 
        # Load success profiles for refined score
        self.profiles = {}
        self._load_profiles()

    def _load_profiles(self):
        """Load timeframe-specific success profiles for refined filtering."""
        model_dir = Path(__file__).parent / 'models'
        timeframes = ['4h', '8h', '12h', '1d']
        import json
        for tf in timeframes:
            p = model_dir / f'profile_{tf}.json'
            if p.exists():
                try:
                    with open(p, 'r') as f:
                        self.profiles[tf] = json.load(f)
                except Exception as e:
                    print(f"[Scanner] Failed to load profile for {tf}: {e}")

    def get_engine(self, timeframe: str) -> InferenceEngine:
        """Get or lazy-load InferenceEngine for timeframe"""
        if timeframe not in self._engines:
            try:
                self._engines[timeframe] = InferenceEngine(timeframe)
                print(f"[Scanner] Loaded Engine for {timeframe}")
            except Exception as e:
                print(f"[Scanner] Error loading engine for {timeframe}: {e}")
                return None
        return self._engines[timeframe]

    def scan(self, symbols: List[str], timeframe: str, lookback_days: int = 6) -> List[Dict[str, Any]]:
        """
        Scan a list of symbols for valid signals within the lookback period.
        """
        engine = self.get_engine(timeframe)
        if not engine:
            print(f"[Scanner] No engine for {timeframe}, skipping scan.")
            return []

        signals = []
        # We need at least 100 candles for InferenceEngine + lookback
        # 1d: 100 days. 12h: 50 days. 4h: 17 days.
        tf_days = {
            '15m': 2, '30m': 3, '1h': 5, '2h': 10,
            '4h': 20, '6h': 30, '8h': 40, '12h': 60, '1d': 120
        }
        buffer_days = tf_days.get(timeframe, 100)
        fetch_start = f"{lookback_days + buffer_days} days ago UTC" # Dynamic buffer based on timeframe
        
        # --- NEW: Fetch BTC Macro Context for ML Inference ---
        btc_context = pd.DataFrame()
        try:
            # We fetch a bit more for BTC to ensure 200 SMA is valid
            btc_fetch_start = f"{lookback_days + buffer_days + 200} days ago UTC"
            df_btc = self.processor.get_historical_data("BTCUSDT", timeframe, btc_fetch_start, "now UTC")
            if not df_btc.empty:
                # Calculate features on BTC using the exact same multi_timeframe_pipeline logic
                from ml.multi_timeframe_pipeline import calculate_features_for_timeframe
                df_btc = calculate_features_for_timeframe(df_btc, timeframe)
                
                if not df_btc.empty:
                    btc_context = df_btc[['timestamp', 'close', 'sma_200', 'adx', 'log_returns']].copy()
                    btc_context.columns = ['timestamp', 'btc_close', 'btc_sma_200', 'btc_adx', 'btc_returns']
                    btc_context['btc_is_bull_regime'] = (btc_context['btc_close'] > btc_context['btc_sma_200']).astype(int)
                    btc_context['btc_trend_strength'] = np.where(btc_context['btc_adx'] > 25, 1, 0)
        except Exception as e:
            print(f"[Scanner] Failed to fetch BTC context: {e}")
        # -----------------------------------------------------

        # Get current prices for all symbols efficiently if possible, 
        # but processor might do it one by one.
        # For now, we fetch candles, which contain the close price (approx current).
        
        for symbol in symbols:
            try:
                # 0. Rate Limit Protection
                time.sleep(0.08) # 50ms delay (~20 requests/sec max)

                # 1. Fetch Data
                # print(f"DEBUG: Fetching {symbol} {timeframe}...")
                df = self.processor.get_historical_data(symbol, timeframe, fetch_start, 'now UTC')
                
                if df.empty or len(df) < 50: continue
                
                # Capture live price before dropping the incomplete candle
                live_price = df['close'].iloc[-1]
                
                # Drop incomplete candle
                df = df.iloc[:-1].copy()
                current_price = df['close'].iloc[-1]
                
                # if len(signals) == 0 and symbol == symbols[0]:
                #     print(f"[Scanner Debug] First symbol {symbol} fetched {len(df)} rows. Price: {current_price}")
                
                # 2. Calculate Indicators
                df = self.processor.calculate_macd(df)
                
                # 3. Detect Crossover (Vectorized)
                df['macd_cross_up'] = ((df['macd'] > df['signal']) & (df['macd'].shift(1) <= df['signal'].shift(1))).astype(int)
                df['macd_cross_down'] = ((df['macd'] < df['signal']) & (df['macd'].shift(1) >= df['signal'].shift(1))).astype(int)
                
                # 4. Filter for Recent Crossovers
                cutoff_date = pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)
                
                # Handle tz-naive index if necessary (using timestamp column like Dashboard)
                if df['timestamp'].dt.tz is None: 
                     cutoff_date = cutoff_date.tz_localize(None)
                
                recent = df[df['timestamp'] >= cutoff_date]
                if recent.empty: 
                    # print(f"[Scanner Debug] {symbol}: No recent data after {cutoff_date}")
                    continue
                
                cross_up = recent[recent['macd_cross_up'] == 1]
                cross_down = recent[recent['macd_cross_down'] == 1]
                
                if cross_up.empty and cross_down.empty: 
                    # print(f"[Scanner Debug] {symbol}: No crossover in lookback")
                    continue
                
                # 5. Determine Latest Signal
                is_up = False
                row = None
                
                # Find the absolute latest crossover event
                last_up_idx = cross_up.index[-1] if not cross_up.empty else -1
                last_down_idx = cross_down.index[-1] if not cross_down.empty else -1
                
                if last_up_idx > last_down_idx:
                    is_up = True
                    row = cross_up.iloc[-1]
                else:
                    is_up = False
                    row = cross_down.iloc[-1]
                    
                # 6. ML Prediction
                
                # --- NEW: Merge BTC Context Before Prediction ---
                predict_df = df.loc[:row.name].copy()
                if not btc_context.empty:
                    # Merge on timestamp
                    predict_df = predict_df.merge(btc_context, on='timestamp', how='left')
                    
                    # Forward fill and fillna with 0
                    fill_cols = ['btc_is_bull_regime', 'btc_trend_strength', 'btc_returns']
                    for col in fill_cols:
                        if col in predict_df.columns:
                            predict_df[col] = predict_df[col].ffill().fillna(0)
                        else:
                            predict_df[col] = 0
                            
                    # Calculate Relative Strength
                    if 'log_returns' in predict_df.columns and 'btc_returns' in predict_df.columns:
                        predict_df['rs_vs_btc'] = predict_df['log_returns'] - predict_df['btc_returns']
                        predict_df['rs_vs_btc_sma7'] = predict_df['rs_vs_btc'].rolling(7).mean()
                    else:
                        predict_df['rs_vs_btc'] = 0.0
                        predict_df['rs_vs_btc_sma7'] = 0.0
                # ------------------------------------------------
                
                prediction = engine.predict(symbol, predict_df)
                if not prediction: continue
                
                confidence = prediction.get('entry_confidence', 0.0)
                # Filter by confidence (basic pre-filter)
                if confidence < 0.6: 
                    # print(f"[Scanner Debug] {symbol}: Skipped (Conf {confidence:.2f} < 0.6)")
                    continue 
                
                # 7. Entry Zone Analysis
                cross_price = row['close']
                status = self.analyze_entry_zone(is_up, cross_price, live_price, timeframe)
                
                # Check for "Already Pumped" (Max Favorable Excursion)
                # Ensure we don't enter if price already went > 3% in favorable direction
                # Get data since signal
                since_signal = df.loc[row.name:] 
                
                if is_up:
                    max_price = since_signal['high'].max()
                    mfe_pct = (max_price - cross_price) / cross_price
                else:
                    min_price = since_signal['low'].min()
                    mfe_pct = (cross_price - min_price) / cross_price

                # Check if price already hit the predicted TP or 20%
                tp_threshold = float(prediction.get('tp_pct', 0.20))
                
                if mfe_pct > tp_threshold: 
                    status = f"❌ TOO LATE (Hit TP {tp_threshold:.1%})"

                # Filter bad entry zones
                if "❌ TOO LATE" in status:
                    # print(f"[Scanner Debug] {symbol}: Skipped (Price moved too far)")
                    continue
                
                # 8. Refined Score Calculation
                score_data = self.get_refined_score(row, timeframe)
                refined_score = score_data['score']

                # 9. Compile Result
                meta_cleaned = self.clean_for_msgpack(prediction)
                
                signal_data = {
                    'symbol': symbol,
                    'type': 'LONG' if is_up else 'SHORT',
                    'timestamp': str(row['timestamp']), # Ensure string
                    'confidence': float(confidence), # Ensure float
                    'status': status,
                    'signal_price': float(cross_price), # Ensure float
                    'current_price': float(live_price), # Use live price for display
                    'refined_score': float(refined_score),
                    'sl_pct': float(prediction.get('sl_pct', 0.02)),
                    'tp_pct': float(prediction.get('tp_pct', 0.04)),
                    'risk_reward': float(prediction.get('risk_reward', 0.0)),
                    'meta': meta_cleaned
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

    def clean_for_msgpack(self, obj):
        """Recursively convert numpy/pandas types to native python types for serialization"""
        if isinstance(obj, dict):
            return {k: self.clean_for_msgpack(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.clean_for_msgpack(v) for v in obj]
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return str(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self.clean_for_msgpack(obj.tolist())
        elif pd.isna(obj):
            return None
        return obj

    def analyze_entry_zone(self, is_long: bool, signal_price: float, current_price: float, timeframe: str) -> str:
        """
        Determine if the current price is in a good entry zone relative to the signal price.
        """
        AVG_MAE_STATS = {
            '4h': 0.035,
            '8h': 0.045,
            '12h': 0.055,
            '1d': 0.065
        }
        AVG_MFE_STATS = {
            '4h': 0.11,
            '8h': 0.14,
            '12h': 0.16,
            '1d': 0.22
        }
        
        MAE = AVG_MAE_STATS.get(timeframe, 0.04) # Max Adverse Excursion (Stop Loss proxy)
        MFE = AVG_MFE_STATS.get(timeframe, 0.12) # Max Favorable Excursion (Take Profit proxy)
        
        status = "UNKNOWN"
        
        if is_long:
            limit_price = signal_price * (1 - MAE)
            profit_limit = signal_price * (1 + MFE * 0.5)
            
            if current_price < limit_price: status = "⚠️ DEEP MERGE" # Price dropped too much below signal
            elif limit_price <= current_price <= signal_price: status = "💎 DISCOUNT" # Better price than signal
            elif signal_price < current_price <= signal_price * 1.01: status = "✅ GOOD ENTRY" # Close to signal
            elif current_price > profit_limit: status = "❌ TOO LATE" # Moved too far
            else: status = "⚠️ CHASING" # Between good and too late
        else:
            limit_price = signal_price * (1 + MAE)
            profit_limit = signal_price * (1 - MFE * 0.5)
            
            if current_price > limit_price: status = "⚠️ DEEP MERGE"
            elif signal_price <= current_price <= limit_price: status = "💎 DISCOUNT" # Higher price for short is better
            elif signal_price * 0.99 <= current_price < signal_price: status = "✅ GOOD ENTRY"
            elif current_price < profit_limit: status = "❌ TOO LATE"
            else: status = "⚠️ CHASING"
            
        return status

    def get_refined_score(self, row, timeframe: str) -> Dict:
        """Calculate refined score (0-1) and return details."""
        result = {'score': 1.0, 'total': 0, 'factors': [], 'pass_count': 0}
        
        if not timeframe or timeframe not in self.profiles:
            return result
            
        direction = 'LONG' if row.get('macd_cross_up', 0) == 1 else 'SHORT'
        
        tf_profile = self.profiles[timeframe]
        if direction not in tf_profile:
            return result
            
        score = 0
        filters = tf_profile[direction]
        total = len(filters)
        factors = []
        
        for f in filters:
            val = row.get(f['feat'], 0)
            passed = False
            if f['shift'] > 0:
                if val >= f['w_mean'] - 0.2 * f['std']: 
                    score += 1
                    passed = True
            else:
                if val <= f['w_mean'] + 0.2 * f['std']: 
                    score += 1
                    passed = True
            
            factors.append({
                'feature': f['feat'],
                'value': float(val),
                'target': float(f['w_mean']),
                'passed': passed
            })
                
        return {
            'score': score / total if total > 0 else 1.0,
            'total': total,
            'pass_count': score,
            'factors': factors
        }
