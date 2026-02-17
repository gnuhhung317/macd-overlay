import sys
import os
from pathlib import Path
from datetime import datetime

# Add current directory to path so we can import bot modules
sys.path.append(str(Path(__file__).parent))

from bot.config import BotConfig
from bot.executor import BinanceExecutor
from bot.data_provider import DataProvider
from bot.db import DatabaseManager
from bot.signal_engine import SignalEngine

def test_bot_integrity():
    print(f"🔍 Starting Bot Integrity Test [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
    print("-" * 50)

    # 1. Test Config
    print("1. Checking Configuration...")
    try:
        config = BotConfig.load()
        print(f"   ✅ Config loaded. Exchange: {config.exchange.name}")
        print(f"   ✅ Mode: {'DRY RUN' if config.exchange.dry_run else 'LIVE TRADING'}")
        if not config.exchange.api_key or not config.exchange.api_secret:
            print("   ❌ Error: API Key or Secret missing in bot_config.json")
            return
        print("   ✅ API Keys present.")
    except Exception as e:
        print(f"   ❌ Config Error: {e}")
        return

    # 2. Test Database
    print("\n2. Checking Database...")
    try:
        db = DatabaseManager()
        print(f"   ✅ Database connected at: {db.db_path}")
        active = db.get_active_trades()
        print(f"   ✅ Active trades in DB: {len(active)}")
    except Exception as e:
        print(f"   ❌ Database Error: {e}")
        return

    # 3. Test Binance Connection
    print("\n3. Testing Binance Connection (Futures API)...")
    try:
        executor = BinanceExecutor(config)
        balance = executor.get_balance()
        print(f"   ✅ Connection Successful.")
        print(f"   ✅ Available USDT Balance: ${balance:,.2f}")
    except Exception as e:
        print(f"   ❌ Binance API Error: {e}")
        print("      Check if your API keys have 'Futures' permissions enabled.")
        return

    # 4. Test Data Provider & Symbols
    print("\n4. Testing Data Provider...")
    try:
        dp = DataProvider(config)
        test_symbol = config.coins[0] if config.coins else "BTCUSDT"
        print(f"   ⏳ Fetching candles for {test_symbol}...")
        df = dp.get_historical_data(test_symbol, "1d", limit=100)
        if not df.empty:
            print(f"   ✅ Successfully fetched {len(df)} candles for {test_symbol}")
            print(f"   ✅ Current {test_symbol} Price: ${df['close'].iloc[-1]:,.2f}")
        else:
            print(f"   ❌ Failed to fetch data for {test_symbol}")
    except Exception as e:
        print(f"   ❌ Data Error: {e}")

    # 5. Test Signal Engine (ML Components)
    print("\n5. Testing Signal Engine (ML Inference)...")
    try:
        se = SignalEngine(config)
        # Re-use the dataframe from step 4
        if 'df' in locals() and not df.empty:
            df = dp.calculate_indicators(df)
            result = se.analyze(df, test_symbol, "1d")
            print(f"   ✅ Analysis completed for {test_symbol}")
            print(f"   ✅ Action: {result['action']}")
            if result['action'] != "WAIT":
                print(f"   ✅ Confidence: {result['confidence']:.2%}")
                print(f"   ✅ SL: {result['sl_price']} | TP: {result['tp_price']}")
        else:
            print("   ⚠️ Skipping Signal test (No data)")
    except Exception as e:
        print(f"   ❌ Signal Engine Error: {e}")
        print("      Check if your ML models are in the /ml/models directory.")

    print("\n" + "="*50)
    print("🎉 Bot Integrity Test Finished!")
    if not config.exchange.dry_run:
        print("⚠️  REMINDER: You are in LIVE mode. The bot is ready to trade.")
    else:
        print("ℹ️  The bot is in DRY RUN mode.")
    print("="*50)

if __name__ == "__main__":
    test_bot_integrity()
