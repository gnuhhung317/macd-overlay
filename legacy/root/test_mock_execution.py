import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add the project root to sys.path
sys.path.append(os.getcwd())

from sniper_bot.config import SniperBotConfig
from bot.db import DatabaseManager
from bot.data_provider import DataProvider
from bot.executor import get_executor
from sniper_bot.position_manager import PositionManager

def test_mock_signal_execution():
    print("🎯 Starting Mock Signal Execution Test...")
    
    # 1. Load Testnet Config
    config_path = Path("ansible/bots-configs/sniper_testnet/sniper_bot_config.json")
    if not config_path.exists():
        print(f"❌ Config not found at {config_path}")
        return
        
    config = SniperBotConfig.load(config_path)
    # Ensure we are on testnet and NOT in dry run
    config.exchange.dry_run = False
    config.exchange.use_testnet = True 
    
    # 2. Initialize Components
    db = DatabaseManager()
    executor = get_executor(config)
    data_provider = DataProvider(config)
    
    pm = PositionManager(
        config=config,
        db=db,
        executor=executor,
        data_provider=data_provider
    )
    
    # 3. Create Mock Signal (Analysis Dict)
    # We'll use DOGEUSDT as in your error log
    symbol = "DOGEUSDT"
    
    # Get current price to set a LIMIT price that won't fill immediately
    current_price = data_provider.get_current_price(symbol)
    if current_price <= 0:
        print(f"❌ Could not get price for {symbol}")
        return
        
    print(f"ℹ️ Current {symbol} Price: {current_price}")
    
    # Mocking a SHORT signal (as per your log)
    # Set limit price 5% higher to ensure it's a pending LIMIT order
    limit_price = current_price * 1.05 
    
    mock_analysis = {
        "action": "ENTRY",
        "signal": "BEARISH", # SHORT
        "confidence": 0.65,
        "sl": 0.03, # 3% SL
        "tp": 0.06, # 6% TP
        "limit_price": limit_price,
        "risk_reward": 2.0,
        "metadata": {"test": True}
    }
    
    print(f"🚀 Executing Mock SHORT Entry for {symbol}")
    print(f"Target Limit Price: {limit_price:.6f}")
    
    # 4. Run Execution
    try:
        # This will call pm._execute_entry -> executor.place_order
        print("--- STARTING EXECUTION ---")
        pm._execute_entry(symbol, mock_analysis)
        print("--- EXECUTION FINISHED ---")
        
        # Check if the trade was added to active_positions
        if symbol in pm.active_positions:
            print(f"🎉 SUCCESS: {symbol} is now in active_positions.")
            print(f"Order Details: {pm.active_positions[symbol]['raw_data']}")
        else:
            print(f"⚠️ WARNING: {symbol} not found in active_positions. Order might have failed silently.")
            
        print("\n✅ Test completed successfully.")
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")

if __name__ == "__main__":
    test_mock_signal_execution()
