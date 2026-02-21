import pandas as pd
from pathlib import Path
from bot.signal_engine import SignalEngine
from bot.position_manager import PositionManager
from bot.executor import DryRunExecutor
from bot.db import DatabaseManager
from bot.config import BotConfig
from bot.data_provider import DataProvider
import os
import traceback

# Create dummy config
config = BotConfig()
config.exchange.dry_run = True

# Mock DataProvider to return a perfect signal
class MockDataProvider(DataProvider):
    def fetch_closed_candles(self, symbol, interval, lookback_days=5):
        # Create a DataFrame with a Perfect Bullish Cross
        data = {
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='4h'),
            'open': [50000.0] * 100,
            'high': [51000.0] * 100,
            'low': [49000.0] * 100,
            'close': [50500.0] * 100,
            'volume': [1000.0] * 100
        }
        df = pd.DataFrame(data)
        return df

    def calculate_indicators(self, df):
        # Manually force MACD cross
        df['macd'] = 0.0
        df['signal'] = 0.0
        
        # Penultimate candle: MACD < Signal
        df.loc[98, 'macd'] = 100
        df.loc[98, 'signal'] = 110 
        
        # Last candle: MACD > Signal (CROSS UP)
        df.loc[99, 'macd'] = 120
        df.loc[99, 'signal'] = 115 
        
        return df
        
    def get_current_price(self, symbol):
        return 50000.0

def test_dry_run():
    print("🧪 Starting Dry Run Test...", flush=True)
    
    # Simple setup
    config = BotConfig()
    config.exchange.dry_run = True
    
    if os.path.exists("test_bot.db"):
        os.remove("test_bot.db")
    db = DatabaseManager(Path("test_bot.db"))
    
    executor = DryRunExecutor(config)
    data_provider = None # Not needed for execute_calculated_signal
    signal_engine = None # Not needed for execute_calculated_signal
    
    pm = PositionManager(config, db, executor, signal_engine, data_provider)
    
    # Execution
    print("👉 Executing Calculated Signal (Expect Entry)...", flush=True)
    signal_data = {
        'symbol': 'BTCUSDT',
        'type': 'LONG',
        'timestamp': pd.Timestamp.now(),
        'confidence': 0.85,
        'status': '✅ GOOD ENTRY',
        'signal_price': 50000.0,
        'current_price': 50500.0,
        'sl_pct': 0.02,
        'tp_pct': 0.04,
        'risk_reward': 2.0
    }
    
    try:
        pm.execute_calculated_signal(signal_data, "4h")
        
        # Verification
        if "BTCUSDT" in pm.active_positions:
            trade = pm.active_positions["BTCUSDT"]
            print(f"✅ Trade Successfully Created: {trade['symbol']} | Size: ${trade['size']:.2f}", flush=True)
            
            # Test CHASING filter
            print("\n👉 Executing CHASING Signal (Expect Filter)...", flush=True)
            del pm.active_positions["BTCUSDT"] # Reset
            signal_data['status'] = '⚠️ CHASING'
            pm.execute_calculated_signal(signal_data, "4h")
            if "BTCUSDT" not in pm.active_positions:
                print("✅ CHASING successfully filtered!", flush=True)
            else:
                print("❌ CHASING was NOT filtered!", flush=True)
        else:
            print("❌ No Trade Created! Check logs above.", flush=True)
            
    except Exception as e:
        print(f"\n❌ Error during execution: {e}", flush=True)
        import traceback
        traceback.print_exc()

    # Clean up
    if os.path.exists("test_bot.db"):
        os.remove("test_bot.db")

if __name__ == "__main__":
    test_dry_run()
