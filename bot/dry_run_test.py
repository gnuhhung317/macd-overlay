import pandas as pd
from bot.signal_engine import SignalEngine
from bot.position_manager import PositionManager
from bot.executor import DryRunExecutor
from bot.db import DatabaseManager
from bot.config import BotConfig
from bot.data_provider import DataProvider
import os

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
    print("🧪 Starting Dry Run Test...")
    
    # Setup
    if os.path.exists("test_bot.db"):
        os.remove("test_bot.db")
        
    db = DatabaseManager(pd.Path("test_bot.db"))
    executor = DryRunExecutor(config)
    data_provider = MockDataProvider(config)
    signal_engine = SignalEngine(config) # Will try to load real ML, might fail but logic should hold
    
    pm = PositionManager(config, db, executor, signal_engine, data_provider)
    
    # Execution
    print("👉 Processing Symbol with Mock Data (Expect Entry)...")
    pm.process_symbol("BTCUSDT", "4h")
    
    # Verification
    active_trades = pm.active_positions
    if "BTCUSDT" in active_trades:
        trade = active_trades["BTCUSDT"]
        print(f"✅ Trade Successfully Created: {trade}")
        print(f"   Direction: {trade['direction']}")
        print(f"   Entry: {trade['entry_price']}")
        print(f"   SL: {trade['sl_price']}")
        print(f"   TP: {trade['tp_price']}")
    else:
        print("❌ No Trade Created! Check logic.")

    # Clean up
    if os.path.exists("test_bot.db"):
        os.remove("test_bot.db")

if __name__ == "__main__":
    test_dry_run()
