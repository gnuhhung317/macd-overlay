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
from ml.inference import InferenceEngine

# MLPredictor class removed - using ml.inference.InferenceEngine instead

class CachedEngine:
    """Wrapper for InferenceEngine to track age for caching"""
    def __init__(self, interval: str):
        self.engine = InferenceEngine(interval)
        self.created_at = time.time()
        
    @property
    def age_seconds(self):
        return time.time() - self.created_at


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
        
        # Model cache: {interval: CachedEngine}
        self.model_cache: Dict[str, CachedEngine] = {}
        
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
    
    def get_model(self, interval: str) -> Optional[CachedEngine]:
        """
        Get ML predictor for timeframe with caching
        
        Args:
            interval: Timeframe interval
            
        Returns:
            CachedEngine instance or None
        """
        # Check cache
        if interval in self.model_cache:
            wrapper = self.model_cache[interval]
            
            # Check if cache is still valid
            cache_ttl = self.config.get_model_cache_ttl()
            if wrapper.age_seconds < cache_ttl:
                return wrapper
            else:
                # Cache expired, unload
                print(f"♻️  Model cache expired for {interval}, reloading...")
                del self.model_cache[interval]
                gc.collect()
        
        # Load new predictor
        # InferenceEngine verifies models exist in __init__
        try:
            wrapper = CachedEngine(interval)
            
            # Cache if enabled
            if self.config.is_model_caching_enabled():
                self.model_cache[interval] = wrapper
            return wrapper
        except Exception as e:
            print(f"⚠️  Could not load models for {interval}: {e}")
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
        """
        # Always safe safe buffer
        return "400 days ago UTC"
    
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
        wrapper = self.get_model(interval)
        engine = wrapper.engine if wrapper else None
        
        if not engine:
            print(f"⚠️  No ML engine available for {interval}")
            # We can still continue without ML if desired, but let's be strict
            # Or just continue and skip ML predictions
        
        # Get enabled coins
        symbols = self.config.get_enabled_coins()
        lookback_period = self._get_lookback_period(interval)
        print(f"📊 Checking {len(symbols)} symbols (lookback: {lookback_period})...")
        
        # Get threshold from config
        tf_config = self.config.get_timeframe_config(interval)
        entry_threshold = tf_config.get('entry_threshold', 0.65)
        
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
                
                # ENFORCE CLOSED CANDLE LOGIC:
                # Drop the last (forming) candle to prevent repaint/unstable signals
                df = df.iloc[:-1].copy()
                if len(df) < 50:
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
                        'entry_confidence': None,
                        'ml_prediction': None,
                        'predictions': None
                    }
                    
                    # Try to get ML predictions
                    if engine:
                        try:
                            # InferenceEngine.predict now handles feature calculation
                            # It expects symbol and dataframe
                            pred_result = engine.predict(symbol, df)
                            
                            if not pred_result.get('error'):
                                alert['ml_prediction'] = pred_result
                                alert['entry_confidence'] = pred_result['entry_confidence']
                                
                                # Compatibility with old alert format if needed
                                alert['predictions'] = {
                                    'entry_confidence': pred_result['entry_confidence'],
                                    'sl_percent': pred_result['sl_pct'] * 100,
                                    'tp_percent': pred_result['tp_pct'] * 100,
                                    'action': pred_result['action']
                                }
                                
                                # Filter by threshold
                                if pred_result['entry_confidence'] < entry_threshold:
                                    print(f"  ⏭️  SKIP {symbol} {crossover}: confidence {pred_result['entry_confidence']:.1%} < {entry_threshold:.0%}")
                                    continue
                            else:
                                print(f"  ⚠️  ML Error: {pred_result.get('error')}")
                                # Assume filtered or allow? Let's skip if ML broken to be safe
                                continue
                                
                        except Exception as e:
                            print(f"  ❌ ML Prediction exception: {e}")
                            continue
                    
                    # If we got here, it's a valid alert
                    alerts.append(alert)
                    
                    # Update shared_data if available
                    if self.shared_data is not None and self.data_lock is not None:
                        with self.data_lock:
                            self.shared_data['alerts'].insert(0, alert)
                            key = f"{symbol}_{interval}"
                            self.shared_data['last_check'][key] = last_row['timestamp']
                    
                    conf_str = f"{alert['entry_confidence']:.1%}" if alert['entry_confidence'] else "N/A"
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
        
        for interval, wrapper in self.model_cache.items():
            if wrapper.age_seconds > cache_ttl:
                intervals_to_remove.append(interval)
        
        for interval in intervals_to_remove:
            print(f"♻️  Cleaning up cache for {interval}")
            # Just delete the reference, GC will handle the rest
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
