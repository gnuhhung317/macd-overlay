import ccxt
import json
import time

# Use the same config as the bot
try:
    with open('bot_config.json', 'r') as f:
        config = json.load(f)
    api_key = config['exchange']['api_key']
    api_secret = config['exchange']['api_secret']
    passphrase = config['exchange'].get('passphrase', config['exchange'].get('password', ''))
except:
    print("Error loading bot_config.json. Please ensure this script is in the bot root folder.")
    exit(1)

def run_test():
    client = ccxt.bitget({
        'apiKey': api_key,
        'secret': api_secret,
        'password': passphrase,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap',
            'positionMode': False # Force CCXT to Unilateral
        }
    })

    # symbol = 'ZROUSDT' # User's failing symbol
    symbol = 'XRPUSDT' 
    ccxt_symbol = 'XRP/USDT:USDT'
    
    # Try to set margin mode to ISOLATED as in user's config
    try:
        client.set_margin_mode('isolated', ccxt_symbol)
        print("✅ Set to ISOLATED margin mode")
    except Exception as e:
        print(f"⚠️ Could not set isolated margin (maybe already set or has positions): {e}")

    tests = [
        ("1. Bare (No tradeSide, No posSide)", {}),
        ("2. tradeSide='open' only", {'tradeSide': 'open'}),
        ("3. posSide='long' only", {'posSide': 'long'}),
        ("4. tradeSide='open' AND posSide='long'", {'tradeSide': 'open', 'posSide': 'long'}),
    ]

    print("\n" + "="*50)
    print("RUNNING BITGET V2 DIAGNOSTICS")
    print("="*50)

    for name, extra_params in tests:
        print(f"\nTEST {name}...")
        try:
            # We use a tiny amount
            order = client.create_order(
                symbol=ccxt_symbol,
                type='market',
                side='buy',
                amount=5,
                params=extra_params
            )
            print(f"  ✅ SUCCESS! Order ID: {order['id']}")
            # Cancel if it's a limit or just close if market? 
            # We don't want to leave positions. But market is easier to test.
            # print("  (Closing position immediately...)")
            # client.create_order(ccxt_symbol, 'market', 'sell', 5, params={'reduceOnly': True})
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
        time.sleep(1)

if __name__ == "__main__":
    run_test()
