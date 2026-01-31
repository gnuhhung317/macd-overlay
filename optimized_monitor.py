"""
Optimized Multi-Timeframe Monitor
Single thread with lazy loading, model caching, and smart memory management
"""

import time
import os
import psutil
import gc
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional
from timeframe_config import MultiTimeframeConfig
from data_processor import BinanceDataProcessor
from telegram_notifier import TelegramNotifier

# Import ML features calculator
try:
    from ml.data_pipeline import calculate_features
    ML_FEATURES_AVAILABLE = True
except ImportError:
    ML_FEATURES_AVAILABLE = False
    print("⚠️  ML features not available")


class MLPredictor:
    """ML Predictor wrapper with lazy loading"""
    
    def __init__(self, models_dir: str, entry_threshold: float = 0.5):
        self.models_dir = models_dir
        self.entry_threshold = entry_threshold
        self._models = None
        self._scalers = None  # Scalers for each model
        self._loaded_time = None
        self._feature_names = None  # Feature names expected by each model
    
    def load_models(self):
        """Load ML models from disk"""
        try:
            import joblib
            
            entry_path = os.path.join(self.models_dir, 'entry_filter.joblib')
            sl_path = os.path.join(self.models_dir, 'sl_predictor.joblib')
            tp_path = os.path.join(self.models_dir, 'tp_predictor.joblib')
            
            if not all(os.path.exists(p) for p in [entry_path, sl_path, tp_path]):
                print(f"⚠️  Models not found in {self.models_dir}")
                return False
            
            # Load model files (they contain dicts with 'model', 'scaler', 'feature_names')
            entry_data = joblib.load(entry_path)
            sl_data = joblib.load(sl_path)
            tp_data = joblib.load(tp_path)
            
            # Extract actual models from dict structure
            self._models = {
                'entry_filter': entry_data['model'] if isinstance(entry_data, dict) else entry_data,
                'sl_predictor': sl_data['model'] if isinstance(sl_data, dict) else sl_data,
                'tp_predictor': tp_data['model'] if isinstance(tp_data, dict) else tp_data
            }
            
            # Extract scalers (CRITICAL for correct predictions!)
            self._scalers = {
                'entry_filter': entry_data.get('scaler') if isinstance(entry_data, dict) else None,
                'sl_predictor': sl_data.get('scaler') if isinstance(sl_data, dict) else None,
                'tp_predictor': tp_data.get('scaler') if isinstance(tp_data, dict) else None
            }
            
            # Store feature names for EACH model (they may differ!)
            self._feature_names = {
                'entry_filter': entry_data.get('feature_names') if isinstance(entry_data, dict) else None,
                'sl_predictor': sl_data.get('feature_names') if isinstance(sl_data, dict) else None,
                'tp_predictor': tp_data.get('feature_names') if isinstance(tp_data, dict) else None
            }
            
            # Log feature counts
            for name, features in self._feature_names.items():
                if features:
                    print(f"  📋 {name} expects {len(features)} features")
            
            self._loaded_time = time.time()
            print(f"✅ Loaded models from {self.models_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def predict(self, features_df, is_bullish: bool = True):
        """Make predictions using correct features for each model
        
        Args:
            features_df: DataFrame with calculated features
            is_bullish: True for bullish crossover, False for bearish
        """
        if self._models is None:
            if not self.load_models():
                return None
        
        try:
            import pandas as pd
            
            # Get feature sets for each model
            entry_features = self._feature_names.get('entry_filter')
            sl_features = self._feature_names.get('sl_predictor')
            tp_features = self._feature_names.get('tp_predictor')
            
            # ===== ENTRY FILTER =====
            if entry_features:
                # Check if is_bullish_cross is needed (it's added during training)
                features_copy = features_df.copy()
                if 'is_bullish_cross' in entry_features and 'is_bullish_cross' not in features_copy.columns:
                    features_copy['is_bullish_cross'] = 1 if is_bullish else 0
                
                missing = [f for f in entry_features if f not in features_copy.columns]
                if missing:
                    print(f"  ⚠️  Missing features for entry: {missing[:5]}...")
                    return None
                entry_input = features_copy[entry_features].values
            else:
                entry_input = features_df.values
            
            # Apply scaler if available (CRITICAL!)
            if self._scalers.get('entry_filter') is not None:
                entry_input = self._scalers['entry_filter'].transform(entry_input)
            
            # Entry filter prediction
            entry_prob = self._models['entry_filter'].predict_proba(entry_input)[0][1]
            if entry_prob < self.entry_threshold:
                return {'rejected': True, 'reason': 'low_confidence', 'entry_confidence': entry_prob, 'threshold': self.entry_threshold}
            
            # ===== SL PREDICTOR =====
            if sl_features:
                features_copy = features_df.copy()
                if 'is_bullish_cross' in sl_features and 'is_bullish_cross' not in features_copy.columns:
                    features_copy['is_bullish_cross'] = 1 if is_bullish else 0
                sl_input = features_copy[sl_features].values
            else:
                sl_input = features_df.values
            
            if self._scalers.get('sl_predictor') is not None:
                sl_input = self._scalers['sl_predictor'].transform(sl_input)
                
            # ===== TP PREDICTOR =====
            if tp_features:
                features_copy = features_df.copy()
                if 'is_bullish_cross' in tp_features and 'is_bullish_cross' not in features_copy.columns:
                    features_copy['is_bullish_cross'] = 1 if is_bullish else 0
                tp_input = features_copy[tp_features].values
            else:
                tp_input = features_df.values
            
            if self._scalers.get('tp_predictor') is not None:
                tp_input = self._scalers['tp_predictor'].transform(tp_input)
            
            # Predictions
            sl_pred = self._models['sl_predictor'].predict(sl_input)[0]
            tp_pred = self._models['tp_predictor'].predict(tp_input)[0]
            
            return {
                'entry_confidence': entry_prob,
                'sl_percent': float(sl_pred),
                'tp_percent': float(tp_pred)
            }
        except Exception as e:
            print(f"❌ ML prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def unload(self):
        """Unload models to free memory"""
        if self._models is not None:
            del self._models
            self._models = None
            self._loaded_time = None
            gc.collect()
    
    @property
    def is_loaded(self):
        """Check if models are loaded"""
        return self._models is not None
    
    @property
    def age_seconds(self):
        """Get age of loaded models in seconds"""
        if self._loaded_time is None:
            return float('inf')
        return time.time() - self._loaded_time


class OptimizedMonitor:
    """
    Optimized multi-timeframe monitor with:
    - Single thread sequential scanning
    - Lazy model loading with caching
    - Smart memory management
    - Per-timeframe telegram notifications
    """
    
    def __init__(self, config_path='monitor_config.json', stop_event=None, shared_data=None, data_lock=None):
        """
        Initialize monitor with config
        
        Args:
            config_path: Path to monitor config JSON
            stop_event: Threading event to stop monitor (optional)
            shared_data: Shared data dictionary for API integration (optional)
            data_lock: Lock for shared_data access (optional)
        """
        self.config = MultiTimeframeConfig(config_path)
        self.processor = BinanceDataProcessor()
        
        # API integration
        self.stop_event = stop_event
        self.shared_data = shared_data
        self.data_lock = data_lock
        
        # Model cache: {interval: MLPredictor}
        self.model_cache: Dict[str, MLPredictor] = {}
        
        # Telegram notifiers per timeframe
        self.notifiers: Dict[str, TelegramNotifier] = {}
        self._init_notifiers()
        
        # Scan tracking
        self.last_scan: Dict[str, float] = {}
        self.scan_count: Dict[str, int] = {}
        
        # Statistics
        self.total_scans = 0
        self.total_alerts = 0
        self.start_time = time.time()
        
        print(f"🚀 OptimizedMonitor initialized")
        print(f"   Timeframes: {self.config.get_enabled_timeframes()}")
        print(f"   Coins: {len(self.config.get_enabled_coins())}")
    
    def _init_notifiers(self):
        """Initialize telegram notifiers for each timeframe"""
        if not self.config.is_telegram_enabled():
            print("⚠️  Telegram notifications disabled")
            return
        
        token = self.config.get_telegram_token()
        for interval in self.config.get_enabled_timeframes():
            chat_id = self.config.get_telegram_chat_id(interval)
            if chat_id:
                self.notifiers[interval] = TelegramNotifier(token, chat_id)
                print(f"📱 Telegram notifier ready for {interval} → {chat_id}")
    
    def get_model(self, interval: str) -> Optional[MLPredictor]:
        """
        Get ML predictor for timeframe with caching
        
        Args:
            interval: Timeframe interval
            
        Returns:
            MLPredictor instance or None
        """
        # Check cache
        if interval in self.model_cache:
            predictor = self.model_cache[interval]
            
            # Check if cache is still valid
            cache_ttl = self.config.get_model_cache_ttl()
            if predictor.age_seconds < cache_ttl:
                return predictor
            else:
                # Cache expired, unload
                print(f"♻️  Model cache expired for {interval}, reloading...")
                predictor.unload()
                del self.model_cache[interval]
        
        # Load new predictor
        tf_config = self.config.get_timeframe_config(interval)
        if not tf_config:
            return None
        
        predictor = MLPredictor(
            models_dir=tf_config['models_dir'],
            entry_threshold=tf_config['entry_threshold']
        )
        
        # Load models
        if predictor.load_models():
            # Cache if enabled
            if self.config.is_model_caching_enabled():
                self.model_cache[interval] = predictor
            return predictor
        
        return None
    
    def should_scan_now(self, interval: str) -> bool:
        """
        Check if it's time to scan this timeframe
        
        Args:
            interval: Timeframe interval
            
        Returns:
            True if should scan now
        """
        tf_config = self.config.get_timeframe_config(interval)
        if not tf_config:
            return False
        
        scan_interval = tf_config['scan_interval']
        last_scan_time = self.last_scan.get(interval, 0)
        
        return (time.time() - last_scan_time) >= scan_interval
    
    def _get_lookback_period(self, interval: str) -> str:
        """
        Get appropriate lookback period based on interval
        Ensures enough candles for MACD calculation and pattern detection
        
        Args:
            interval: Timeframe interval
            
        Returns:
            Lookback period string (e.g., '20 days ago UTC')
        """
        lookback_map = {
            '1m': '6 hours ago UTC',
            '3m': '12 hours ago UTC',
            '5m': '1 day ago UTC',
            '15m': '2 days ago UTC',
            '30m': '3 days ago UTC',
            '1h': '10 days ago UTC',      # 240 candles
            '2h': '20 days ago UTC',      # 240 candles
            '4h': '40 days ago UTC',      # 240 candles (need 200+ for ML features)
            '6h': '60 days ago UTC',      # 240 candles
            '8h': '80 days ago UTC',      # 240 candles
            '12h': '120 days ago UTC',    # 240 candles
            '1d': '250 days ago UTC',     # 250 candles (need 200+ for SMA200)
            '3d': '270 days ago UTC',
            '1w': '1 year ago UTC',
        }
        return lookback_map.get(interval, '5 days ago UTC')
    
    def scan_timeframe(self, interval: str):
        """
        Scan all symbols for specific timeframe
        
        Args:
            interval: Timeframe interval
        """
        print(f"\n{'='*70}")
        print(f"🔍 Scanning {interval} @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        
        # Get models
        predictor = self.get_model(interval)
        if not predictor:
            print(f"⚠️  No models available for {interval}")
            return
        
        # Get enabled coins
        symbols = self.config.get_enabled_coins()
        lookback_period = self._get_lookback_period(interval)
        print(f"📊 Checking {len(symbols)} symbols (lookback: {lookback_period})...")
        
        alerts = []
        for i, symbol in enumerate(symbols, 1):
            try:
                # Fetch data with interval-aware lookback period
                df = self.processor.get_historical_data(
                    symbol=symbol, 
                    interval=interval,
                    start_date=lookback_period,
                    end_date='now UTC'
                )
                if df is None or len(df) < 50:
                    continue
                
                # Calculate MACD indicators
                df = self.processor.calculate_macd(df)
                
                # Check for MACD crossover
                last_row = df.iloc[-1]
                prev_row = df.iloc[-2]
                
                # Detect crossover
                crossover = None
                if (prev_row['macd'] < prev_row['signal'] and 
                    last_row['macd'] > last_row['signal']):
                    crossover = 'bullish'
                elif (prev_row['macd'] > prev_row['signal'] and 
                      last_row['macd'] < last_row['signal']):
                    crossover = 'bearish'
                
                if crossover:
                    # Create basic alert first
                    alert = {
                        'symbol': symbol,
                        'interval': interval,
                        'type': crossover.upper(),
                        'crossover_type': crossover,
                        'price': float(last_row['close']),
                        'macd': float(last_row['macd']),
                        'signal': float(last_row['signal']),
                        'histogram': float(last_row.get('histogram', 0)),
                        'timestamp': last_row['timestamp'],
                        'entry_confidence': None,  # Will be set by ML or remain None
                        'ml_prediction': None,
                        'predictions': None
                    }
                    
                    # Try to get ML predictions - SKIP alert if ML fails
                    skip_alert = False
                    skip_reason = None
                    is_bullish = (crossover == 'bullish')
                    
                    try:
                        # Prepare features (returns all 72+ features)
                        features = self._prepare_features(df, interval, crossover)
                        if features is not None:
                            # predict() will select correct features for each model
                            predictions = predictor.predict(features, is_bullish=is_bullish)
                            
                            if predictions is None:
                                # Unexpected None (should not happen with new code)
                                skip_alert = True
                                skip_reason = "ML returned None unexpectedly"
                            elif predictions.get('rejected'):
                                # Confidence below threshold
                                skip_alert = True
                                conf = predictions['entry_confidence']
                                thresh = predictions['threshold']
                                skip_reason = f"confidence {conf:.1%} < threshold {thresh:.0%}"
                            else:
                                # Valid prediction
                                alert['ml_prediction'] = predictions
                                alert['predictions'] = predictions
                                alert['entry_confidence'] = predictions['entry_confidence']
                        else:
                            # Features prep failed
                            skip_alert = True
                            skip_reason = "không đủ dữ liệu để tính features"
                    except Exception as e:
                        skip_alert = True
                        skip_reason = f"lỗi: {e}"
                    
                    # Skip alert if ML failed
                    if skip_alert:
                        print(f"  ⏭️  SKIP {symbol} {crossover}: {skip_reason}")
                        continue
                    
                    alerts.append(alert)
                    
                    # Update shared_data if available
                    if self.shared_data is not None and self.data_lock is not None:
                        with self.data_lock:
                            self.shared_data['alerts'].insert(0, alert)
                            key = f"{symbol}_{interval}"
                            self.shared_data['last_check'][key] = last_row['timestamp']
                    
                    conf_str = f"{alert['entry_confidence']:.2%}" if alert['entry_confidence'] else "N/A"
                    print(f"  ✅ {symbol} {crossover} @ ${last_row['close']:.2f} (confidence: {conf_str})")
                
                # Update current_data in shared_data
                if self.shared_data is not None and self.data_lock is not None:
                    current = {
                        'price': float(last_row['close']),
                        'macd': float(last_row['macd']),
                        'signal': float(last_row['signal']),
                        'histogram': float(last_row.get('histogram', 0)),
                        'timestamp': last_row['timestamp'],
                        'trend': 'BULLISH' if last_row['macd'] > last_row['signal'] else 'BEARISH',
                        'has_new_alert': crossover is not None
                    }
                    with self.data_lock:
                        self.shared_data['current_data'][f"{symbol}_{interval}"] = current
                
                # Progress indicator
                if i % 50 == 0:
                    print(f"  ... {i}/{len(symbols)} checked")
                    
            except Exception as e:
                print(f"  ❌ Error with {symbol}: {e}")
                continue
        
        # Send alerts
        if alerts:
            self._send_alerts(interval, alerts)
            self.total_alerts += len(alerts)
        
        # Update stats
        self.last_scan[interval] = time.time()
        self.scan_count[interval] = self.scan_count.get(interval, 0) + 1
        self.total_scans += 1
        
        # Update shared_data stats
        if self.shared_data is not None and self.data_lock is not None:
            with self.data_lock:
                self.shared_data['check_count'] += 1
                self.shared_data['last_scan_time'] = datetime.now(ZoneInfo('Asia/Ho_Chi_Minh'))
                self.shared_data['timeframe_stats'][interval] = {
                    'last_scan': datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')),
                    'scan_count': self.scan_count[interval],
                    'alerts_found': len(alerts)
                }
                # Update memory usage
                process = psutil.Process()
                self.shared_data['memory_usage_mb'] = process.memory_info().rss / 1024 / 1024
        
        print(f"\n✅ Scan complete: {len(alerts)} alerts")
        print(f"{'='*70}\n")
    
    def _prepare_features(self, df, interval: str, crossover_type: str):
        """Prepare all features for ML prediction.
        Returns DataFrame with ALL features. predict() will select correct ones for each model.
        
        Args:
            df: DataFrame with OHLCV and MACD data
            interval: Timeframe interval ('4h', '8h', '12h', '1d')
            crossover_type: 'bullish' or 'bearish'
        """
        if not ML_FEATURES_AVAILABLE:
            print(f"  ⚠️  ML features not available (import failed)")
            return None
        
        try:
            # Need at least 200 rows for all features (SMA200, etc.)
            if len(df) < 200:
                print(f"  ⚠️  Not enough data for ML: {len(df)} rows (need 200+)")
                return None
            
            # Calculate all ML features using data_pipeline
            features_df = calculate_features(df)
            
            # Return last row as features for prediction
            if features_df is None or features_df.empty:
                print(f"  ⚠️  calculate_features returned empty")
                return None
            
            # Add missing features that were used during training
            # 1. timeframe_hours
            tf_map = {'1h': 1, '4h': 4, '8h': 8, '12h': 12, '1d': 24}
            features_df['timeframe_hours'] = tf_map.get(interval, 24)
            
            # 2. volatility_7_scaled and volatility_14_scaled
            import numpy as np
            if interval in ['1h', '4h']:
                scale = 24 / tf_map[interval]
                features_df['volatility_7_scaled'] = features_df['volatility_7'] * np.sqrt(scale)
                features_df['volatility_14_scaled'] = features_df['volatility_14'] * np.sqrt(scale)
            else:
                features_df['volatility_7_scaled'] = features_df['volatility_7']
                features_df['volatility_14_scaled'] = features_df['volatility_14']
            
            # 3. funding_rate (default 0 for realtime - we don't have funding data)
            features_df['funding_rate'] = 0
            
            # 4. is_bullish_cross (from crossover_type parameter)
            features_df['is_bullish_cross'] = 1 if crossover_type == 'bullish' else 0
            
            # Get the last row
            last_row = features_df.iloc[[-1]].copy()
            
            # Check for NaN values in critical features
            nan_cols = last_row.columns[last_row.isna().any()].tolist()
            if nan_cols:
                # Fill NaN with 0 for less critical features, log warning
                last_row = last_row.fillna(0)
            
            return last_row
            
        except Exception as e:
            print(f"  ⚠️  Feature preparation error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _send_alerts(self, interval: str, alerts: List[Dict]):
        """Send alerts to telegram for this timeframe"""
        if interval not in self.notifiers:
            print(f"  ⚠️  No notifier for {interval}")
            return
        
        notifier = self.notifiers[interval]
        
        # Set entry threshold from config
        tf_config = self.config.get_timeframe_config(interval)
        if tf_config:
            notifier.entry_threshold = tf_config.get('entry_threshold', 0.4)
        
        for alert in alerts:
            try:
                # Build crossover dict for send_crossover_alert
                crossover = {
                    'type': 'BULLISH' if alert['crossover_type'] == 'bullish' else 'BEARISH',
                    'price': alert['price'],
                    'timestamp': alert['timestamp'],
                    'macd': alert['macd'],
                    'signal': alert['signal'],
                    'ml_prediction': alert.get('ml_prediction')
                }
                
                # Use send_crossover_alert for proper formatting
                success = notifier.send_crossover_alert(
                    crossover=crossover,
                    symbol=alert['symbol'],
                    interval=interval
                )
                
                if success:
                    print(f"  📨 Telegram sent for {alert['symbol']}")
                else:
                    print(f"  ⚠️  Telegram skipped for {alert['symbol']} (filtered or error)")
                    
            except Exception as e:
                print(f"❌ Failed to send alert: {e}")
                import traceback
                traceback.print_exc()
    
    def _format_alert_message(self, alert: Dict) -> str:
        """Format alert message for telegram"""
        predictions = alert.get('predictions')
        
        # Format timestamp
        if isinstance(alert['timestamp'], str):
            timestamp_str = alert['timestamp']
        else:
            timestamp_str = alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        # Base message
        message = f"""
🚨 <b>{alert['symbol']} - {alert['interval']}</b>

📊 Signal: {alert['crossover_type'].upper()}
💰 Price: ${alert['price']:.4f}
📈 MACD: {alert['macd']:.6f}
📉 Signal: {alert['signal']:.6f}
"""
        
        # Add ML predictions if available
        if predictions:
            message += f"""
🤖 ML Predictions:
  • Confidence: {predictions['entry_confidence']:.1%}
  • Stop Loss: {predictions['sl_percent']:.2f}%
  • Take Profit: {predictions['tp_percent']:.2f}%
"""
        else:
            message += "\n⚠️ ML predictions not available\n"
        
        message += f"\n🕐 {timestamp_str}\n"
        
        return message.strip()
    
    def cleanup_old_cache(self):
        """Remove old cached models to free memory"""
        cache_ttl = self.config.get_model_cache_ttl()
        intervals_to_remove = []
        
        for interval, predictor in self.model_cache.items():
            if predictor.age_seconds > cache_ttl:
                intervals_to_remove.append(interval)
        
        for interval in intervals_to_remove:
            print(f"♻️  Cleaning up cache for {interval}")
            self.model_cache[interval].unload()
            del self.model_cache[interval]
        
        if intervals_to_remove:
            gc.collect()
    
    def check_memory_usage(self) -> float:
        """
        Check current memory usage
        
        Returns:
            Memory usage in MB
        """
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def run_scan_cycle(self):
        """Run one scan cycle across all timeframes"""
        priority_order = self.config.get_priority_order()
        
        for interval in priority_order:
            # Check stop event if provided
            if self.stop_event is not None and self.stop_event.is_set():
                print("🛑 Stop event detected, exiting scan cycle")
                return
            
            if self.should_scan_now(interval):
                self.scan_timeframe(interval)
                
                # Check memory after scan
                memory_mb = self.check_memory_usage()
                max_memory = self.config.get_max_memory_mb()
                
                if memory_mb > max_memory:
                    print(f"⚠️  Memory usage {memory_mb:.0f}MB exceeds {max_memory}MB, cleaning up...")
                    self.cleanup_old_cache()
                else:
                    print(f"💾 Memory usage: {memory_mb:.0f}MB")
    
    def run(self):
        """Main monitoring loop"""
        base_interval = self.config.get_base_scan_interval()
        
        print(f"\n{'='*70}")
        print(f"🚀 Starting OptimizedMonitor")
        print(f"   Base scan interval: {base_interval}s")
        print(f"   Timeframes: {self.config.get_priority_order()}")
        print(f"{'='*70}\n")
        
        try:
            while True:
                # Check stop event if provided
                if self.stop_event is not None and self.stop_event.is_set():
                    print("🛑 Stop event detected, shutting down...")
                    break
                
                self.run_scan_cycle()
                
                # Print statistics
                uptime = time.time() - self.start_time
                print(f"\n📊 Statistics:")
                print(f"   Uptime: {uptime/3600:.1f}h")
                print(f"   Total scans: {self.total_scans}")
                print(f"   Total alerts: {self.total_alerts}")
                print(f"   Cached models: {len(self.model_cache)}")
                for interval, count in self.scan_count.items():
                    print(f"   {interval}: {count} scans")
                
                # Sleep with stop event check
                print(f"\n💤 Sleeping {base_interval}s...\n")
                if self.stop_event is not None:
                    if self.stop_event.wait(timeout=base_interval):
                        print("🛑 Stop event detected during sleep, shutting down...")
                        break
                else:
                    time.sleep(base_interval)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping monitor...")
        finally:
            self.cleanup_old_cache()
            print("✅ Shutdown complete")


if __name__ == '__main__':
    monitor = OptimizedMonitor()
    monitor.run()
