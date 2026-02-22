import sys
from pathlib import Path
from datetime import datetime

# Add root to sys.path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from bot.config import BotConfig
from binance.client import Client
from binance.enums import *

def test_trailing_stop_placement():
    print("🔍 Loading Configuration...")
    config = BotConfig.load()
    
    if not config.exchange.api_key or not config.exchange.api_secret:
        print("❌ API Key/Secret missing!")
        return

    print("🔌 Connecting to Binance...")
    client = Client(config.exchange.api_key, config.exchange.api_secret)
    
    # 1. Fetch Open Positions
    print("📊 Fetching Open Positions...")
    try:
        positions = client.futures_position_information()
        active_positions = [p for p in positions if float(p['positionAmt']) != 0]
        
        if not active_positions:
            print("⚠️ No open positions found on Binance Futures.")
            return
            
        print(f"✅ Found {len(active_positions)} active positions.")
        
        for p in active_positions:
            symbol = p['symbol']
            amt = float(p['positionAmt'])
            side = "LONG" if amt > 0 else "SHORT"
            entry_price = float(p['entryPrice'])
            
            print(f"\n--- Processing {symbol} ({side}) ---")
            print(f"Entry Price: {entry_price}")
            
            # 2. Calculate Parameters
            # User request: Trigger 10%, Trailing 5%
            callback_rate = 5.0 # 5%
            
            if side == "LONG":
                activation_price = entry_price * 1.10
                order_side = SIDE_SELL
            else:
                activation_price = entry_price * 0.90
                order_side = SIDE_BUY
                
            qty = abs(amt)
            
            # Price Precision
            info = client.futures_exchange_info()
            symbol_info = next(s for s in info['symbols'] if s['symbol'] == symbol)
            price_prec = symbol_info['pricePrecision']
            
            activation_price = round(activation_price, price_prec)
            
            print(f"Target Activation Price (+10%): {activation_price}")
            print(f"Callback Rate: {callback_rate}%")
            
            confirm = input(f"❓ Place Trailing Stop for {symbol}? (y/n): ")
            if confirm.lower() != 'y':
                continue
                
            # 3. Place Order
            print(f"🚀 Placing TRAILING_STOP_MARKET for {symbol}...")
            try:
                order = client.futures_create_order(
                    symbol=symbol,
                    side=order_side,
                    type='TRAILING_STOP_MARKET',
                    quantity=qty,
                    callbackRate=callback_rate,
                    activationPrice=activation_price,
                    reduceOnly=True
                )
                print(f"✅ Order Placed Successfully!")
                print(f"Raw Response: {order}")
                if 'orderId' in order:
                    print(f"Order ID: {order['orderId']}")
            except Exception as e:
                print(f"❌ Error placing order: {e}")
                
    except Exception as e:
        print(f"❌ API Error: {e}")

if __name__ == "__main__":
    test_trailing_stop_placement()
