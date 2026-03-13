import sys
from pathlib import Path
import logging

# Add root to Path
sys.path.append(str(Path(__file__).parent))

from sniper_bot.config import SniperBotConfig
from bot.executor import BinanceExecutor

def test_order_logic():
    # 1. Load Config
    config_path = Path("ansible/bots-configs/sniper_testnet/sniper_bot_config.json")
    config = SniperBotConfig.load(config_path)
    
    # 2. Initialize Executor
    executor = BinanceExecutor(config)
    
    # 3. Define Test Parameters (Using a real symbol on testnet)
    symbol = "DOGEUSDT"
    side = "short"
    size = 100.0  # $100
    leverage = 20
    
    # Get current price to set a LIMIT price that won't fill immediately
    try:
        ticker = executor.client.futures_symbol_ticker(symbol=symbol)
        current_price = float(ticker['price'])
        # Set limit price 5% higher for a short to ensure it doesn't fill
        limit_price = executor.format_price(symbol, current_price * 1.05)
        
        # Calculate SL/TP based on typical ATR (simulate it)
        sl_price = executor.format_price(symbol, limit_price * 1.02)
        tp_price = executor.format_price(symbol, limit_price * 0.95)
        
        print(f"🚀 Testing Order Placement for {symbol}")
        print(f"Target Limit Price: {limit_price} (Current: {current_price})")
        print(f"SL: {sl_price} | TP: {tp_price}")
        print(f"Margin Mode: {config.exchange.margin_mode}")
        
        # 4. Attempt Order Placement
        # This calls our newly fixed logic in BinanceExecutor.place_order
        result = executor.place_order(
            symbol=symbol,
            side=side,
            size=size,
            leverage=leverage,
            sl_price=sl_price,
            tp_price=tp_price,
            order_type='LIMIT',
            price=limit_price
        )
        
        if result and 'order_id' in result:
            print(f"✅ SUCCESS: Primary LIMIT order placed: {result['order_id']}")
            print("Check Binance Testnet Dashboard to confirm SL/TP orders are also visible.")
        else:
            print("❌ FAILED: Primary order placement returned no result.")
            
    except Exception as e:
        print(f"❌ CRITICAL ERROR during test: {e}")

if __name__ == "__main__":
    test_order_logic()
