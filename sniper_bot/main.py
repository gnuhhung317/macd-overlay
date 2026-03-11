import time
import signal
import sys
from datetime import datetime
from pathlib import Path

# Add root to Path to allow importing from bot and ml packages
sys.path.append(str(Path(__file__).parent.parent))

from telegram_notifier import TelegramNotifier
from sniper_bot.config import SniperBotConfig
from bot.db import DatabaseManager
from bot.data_provider import DataProvider
from bot.executor import get_executor
from .position_manager import PositionManager

from sniper_bot.sniper_scanner import SniperScanner

class SniperBot:
    def __init__(self):
        self.running = True
        self.config = SniperBotConfig.load()
        self.db = DatabaseManager()
        self.data_provider = DataProvider(self.config)
        
        # SignalEngine is bypassed for SniperBot. We pass None to PositionManager.
        self.signal_engine = None
            
        self.executor = get_executor(self.config)
        self.scanner = SniperScanner(self.config, self.data_provider.processor)
        
        # Initialize Telegram Notifier
        self.notifier = None
        if self.config.telegram.enabled and self.config.telegram.token:
            print(f"📱 Telegram Enabled. Chat ID: {self.config.telegram.chat_id}")
            self.notifier = TelegramNotifier(
                token=self.config.telegram.token, 
                chat_id=self.config.telegram.chat_id
            )
            self.notifier.send_message(f"🏹 <b>Sniper Bot Started</b>\nMode: {'DRY RUN' if self.config.exchange.dry_run else 'LIVE'}")
        
        # Reuse PositionManager from bot module
        self.position_manager = PositionManager(
            self.config,
            self.db,
            self.executor,
            self.signal_engine,
            self.data_provider,
            self.notifier
        )
        
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

    def stop(self, signum, frame):
        print("\n\n🛑 Stopping Sniper Bot...")
        self.running = False

    def run(self):
        print(f"🏹 Starting ML Sniper Bot [{datetime.now()}]")
        print(f"🔧 Mode: {'DRY RUN' if self.config.exchange.dry_run else 'LIVE TRADING'}")
        print(f"📊 TF: {self.config.strategy.timeframes}")
        
    def _fetch_top_symbols(self):
        print("🔄 Fetching top symbols by volume...")
        try:
            limit = getattr(self.config, 'max_symbols', 0)
            min_vol = self.config.strategy.min_volume_usdt
            fetched_coins = self.data_provider.get_top_symbols(limit=limit, min_volume=min_vol)
            if fetched_coins:
               self.config.coins = fetched_coins
               print(f"✅ Found {len(fetched_coins)} coins with Volume > ${min_vol:,.0f}")
            else:
               print("⚠️ No coins found matching criteria! Retaining current list.")
        except Exception as e:
            print(f"❌ Error fetching coins: {e}")
            
    def run(self):
        print(f"🏹 Starting ML Sniper Bot [{datetime.now()}]")
        print(f"🔧 Mode: {'DRY RUN' if self.config.exchange.dry_run else 'LIVE TRADING'}")
        print(f"📊 TF: {self.config.strategy.timeframes}")
        
        if getattr(self.config, 'use_all_symbols', False):
            self._fetch_top_symbols()
            
        # Timestamp to track next symbol fetch
        self.last_fetch_time = time.time()
        
        while self.running:
            try:
                if not hasattr(self, 'last_sync'): self.last_sync = 0
                if time.time() - self.last_sync > 300: # 5 mins
                    print("🔄 Syncing positions with Exchange...")
                    self.position_manager.sync_positions()
                    self.last_sync = time.time()
                
                # 0. Periodically refresh top symbols (e.g., every 6 hours)
                if getattr(self.config, 'use_all_symbols', False) and (time.time() - self.last_fetch_time > 21600):
                    self._fetch_top_symbols()
                    self.last_fetch_time = time.time()
                
                # 1. Manage Active Positions
                # Copy keys to a list to avoid dictionary changed size during iteration error
                active_symbols = list(self.position_manager.active_positions.keys())
                for symbol in active_symbols:
                    if not self.running: break
                    for tf in self.config.strategy.timeframes:
                        try:
                            self.position_manager.process_symbol(symbol, tf)
                        except Exception as e:
                            print(f"⚠️ Error processing {symbol} {tf}: {e}")

                # 2. Scan for NEW Entries using Sniper Scanner
                if len(self.position_manager.active_positions) < self.config.risk.max_open_positions:
                    for tf in self.config.strategy.timeframes:
                        print(f"📡 Scanning {len(self.config.coins)} coins on {tf} (Sniper Model)...")
                        try:
                            signals = self.scanner.scan(self.config.coins, tf)
                            
                            # PRIORITIZATION: Sort signals by confidence (descending)
                            signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                            
                            for sig in signals:
                                if not self.running: break
                                self.position_manager.execute_calculated_signal(sig, tf)
                        except Exception as e:
                            print(f"❌ Scanner Error: {e}")
                            
                # Sleep logic
                min_tf_minutes = float('inf')
                for tf in self.config.strategy.timeframes:
                    if tf.endswith('m'): minutes = int(tf[:-1])
                    elif tf.endswith('h'): minutes = int(tf[:-1]) * 60
                    elif tf.endswith('d'): minutes = int(tf[:-1]) * 1440
                    else: minutes = 60 # Default 1h for sniper
                    if minutes < min_tf_minutes: min_tf_minutes = minutes
                
                if min_tf_minutes == float('inf'): min_tf_minutes = 60 
                
                current_ts = int(time.time())
                minutes_since_epoch = current_ts // 60
                next_candle_minutes = (minutes_since_epoch // min_tf_minutes + 1) * min_tf_minutes
                seconds_until_close = (next_candle_minutes * 60) - current_ts
                
                sleep_time = max(10, seconds_until_close + 5)
                
                print(f"💤 Sleeping for {sleep_time:.1f}s (until next {min_tf_minutes}m candle)...")
                start_sleep = time.time()
                while time.time() - start_sleep < sleep_time and self.running:
                    time.sleep(1) 
            
            except KeyboardInterrupt:
                self.stop(None, None)
            except Exception as e:
                print(f"❌ Critical Loop Error: {e}")
                time.sleep(60)

if __name__ == "__main__":
    bot = SniperBot()
    bot.run()
