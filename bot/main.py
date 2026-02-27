import time
import signal
import sys
from datetime import datetime
from telegram_notifier import TelegramNotifier
from .config import BotConfig
from .db import DatabaseManager
from .data_provider import DataProvider
from .signal_engine import SignalEngine
from .executor import get_executor
from .position_manager import PositionManager
# Add Scanner Import
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from ml.scanner import SmartScanner

class Bot:
    def __init__(self):
        self.running = True
        self.config = BotConfig.load()
        self.db = DatabaseManager()
        self.data_provider = DataProvider(self.config)
        self.signal_engine = SignalEngine(self.config)
        self.executor = get_executor(self.config)
        
        # Initialize Smart Scanner
        self.scanner = SmartScanner(self.config, self.data_provider.processor)
        
        # Initialize Telegram Notifier
        self.notifier = None
        if self.config.telegram.enabled and self.config.telegram.token:
            print(f"📱 Telegram Enabled. Chat ID: {self.config.telegram.chat_id}")
            self.notifier = TelegramNotifier(
                token=self.config.telegram.token, 
                chat_id=self.config.telegram.chat_id
            )
            self.notifier.send_message(f"🚀 <b>Bot Started</b>\nMode: {'DRY RUN' if self.config.exchange.dry_run else 'LIVE'}")
        
        self.position_manager = PositionManager(
            self.config,
            self.db,
            self.executor,
            self.signal_engine,
            self.data_provider,
            self.notifier
        )
        
        # Setup cleanup on exit
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

    def stop(self, signum, frame):
        print("\n\n🛑 Stopping Bot...")
        self.running = False

    def run(self):
        print(f"🚀 Starting 3-Stage ML Bot [{datetime.now()}]")
        print(f"🔧 Mode: {'DRY RUN' if self.config.exchange.dry_run else 'LIVE TRADING'}")
        print(f"📊 Strategy: {self.config.strategy}")
        
        # Dynamic Coin Fetching
        if getattr(self.config, 'use_all_symbols', False):
            print("🔄 Fetching all symbols matching criteria...")
            try:
                limit = getattr(self.config, 'max_symbols', 0)
                min_vol = self.config.strategy.min_volume_usdt
                
                fetched_coins = self.data_provider.get_top_symbols(limit=limit, min_volume=min_vol)
                
                if fetched_coins:
                    self.config.coins = fetched_coins
                    print(f"✅ Found {len(fetched_coins)} coins with Volume > ${min_vol:,.0f}")
                    # print(f"Top 5: {fetched_coins[:5]}")
                else:
                    print("⚠️ No coins found matching criteria! Using default list.")
            except Exception as e:
                print(f"❌ Error fetching coins: {e}")
        
        while self.running:
            try:
                cycle_start = time.time()
                
                # 0. Sync Positions (Every 5 mins)
                # We can straightforwardly check every loop if 5 mins passed, 
                # or just use a simple counter/timer.
                if not hasattr(self, 'last_sync'): self.last_sync = 0
                
                if time.time() - self.last_sync > 300: # 300s = 5 mins
                    print("🔄 Syncing positions with Exchange...")
                    self.position_manager.sync_positions()
                    self.last_sync = time.time()
                
                # 1. Manage Active Positions & Signals
                # Iterate over configured coins
                for symbol in self.config.coins:
                    if not self.running: break
                    
                    # Iterate over timeframes (e.g. check 4h signal)
                    for tf in self.config.strategy.timeframes:
                        try:
                            # This handles both management AND legacy separate analysis
                            # If we use scanner, we might want to disable the internal analysis in process_symbol
                            # But for safety, we keep process_symbol as is for now, 
                            # OR we rely on scanner for NEW entries.
                            
                            # Let's use process_symbol ONLY for active management if we use scanner?
                            # No, process_symbol does both. 
                            
                            # Hybrid Approach:
                            # 1. If active, process_symbol manages it.
                            # 2. If not active, we use SCANNER to find entries for ALL coins efficiently.
                            
                            if symbol in self.position_manager.active_positions:
                                self.position_manager.process_symbol(symbol, tf)
                                
                        except Exception as e:
                            print(f"⚠️ Error processing {symbol} {tf}: {e}")

                # 2. Scan for NEW Entries (Smart Scanner)
                if len(self.position_manager.active_positions) < self.config.risk.max_open_positions:
                    for tf in self.config.strategy.timeframes:
                        print(f"📡 Scanning {len(self.config.coins)} coins on {tf}...")
                        try:
                            signals = self.scanner.scan(self.config.coins, tf)
                            
                            # PRIORITIZATION: Sort signals by confidence (descending)
                            # This ensures we take the "Elite" signals first if slot limited.
                            signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                            
                            for sig in signals:
                                if not self.running: break
                                self.position_manager.execute_calculated_signal(sig, tf)
                        except Exception as e:
                            print(f"❌ Scanner Error: {e}")
                            
                # Sleep logic: Wait for next candle close of the smallest timeframe
                # This prevents continuous scanning and API waste
                min_tf_minutes = float('inf')
                for tf in self.config.strategy.timeframes:
                    if tf.endswith('m'):
                        minutes = int(tf[:-1])
                    elif tf.endswith('h'):
                        minutes = int(tf[:-1]) * 60
                    elif tf.endswith('d'):
                        minutes = int(tf[:-1]) * 1440
                    else:
                        minutes = 240 # Default 4h if unknown
                    
                    if minutes < min_tf_minutes:
                        min_tf_minutes = minutes
                
                if min_tf_minutes == float('inf'): min_tf_minutes = 240 # Default 4h
                
                # Calculate seconds until next candle
                current_ts = int(time.time())
                minutes_since_epoch = current_ts // 60
                next_candle_minutes = (minutes_since_epoch // min_tf_minutes + 1) * min_tf_minutes
                seconds_until_close = (next_candle_minutes * 60) - current_ts
                
                # Add a small buffer (e.g. 15s) to ensure data is ready
                sleep_time = max(10, seconds_until_close + 15)
                
                print(f"💤 Sleeping for {sleep_time:.1f}s (until next {min_tf_minutes}m candle)...")
                start_sleep = time.time()
                while time.time() - start_sleep < sleep_time and self.running:
                    time.sleep(1) # Sleep in 1s chunks to check stop signal
            
            except KeyboardInterrupt:
                self.stop(None, None)
            except Exception as e:
                print(f"❌ Critical Loop Error: {e}")
                time.sleep(60) # Prevent tight loop on crash

if __name__ == "__main__":
    import subprocess
    import sys
    import os

    # Auto-launch Dashboard
    # print("🚀 Launching Dashboard on port 8888...")
    dashboard_process = None
    # try:
    #     # Use sys.executable to ensure we use the same python environment
    #     dashboard_process = subprocess.Popen(
    #         [sys.executable, "-m", "streamlit", "run", "dashboard.py", "--server.port", "8888"],
    #         cwd=os.getcwd(), # Ensure we run from root
    #         # stdout=subprocess.DEVNULL, # Optional: Hide dashboard logs
    #         # stderr=subprocess.DEVNULL
    #     )
    # except Exception as e:
    #     print(f"⚠️ Failed to launch dashboard: {e}")

    bot = Bot()
    try:
        bot.run()
    finally:
        if dashboard_process:
            print("🛑 Terminating Dashboard...")
            dashboard_process.terminate()
            dashboard_process.wait()
