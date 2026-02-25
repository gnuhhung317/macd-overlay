import ccxt
import json

def run_test():
    client = ccxt.bitget({
        'apiKey': 'bg_632fb4eef66cbb9267b4b4fc88f643d3',
        'secret': 'f14f6ec8f10e6fb845d9ef1ca272adec51fb03e23913e42e6dd84759eac0d5ca',
        'password': 'daylamacd8h',
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap'
        }
    })

    symbol = 'XRP/USDT:USDT'
    
    print("\n" + "="*50)
    print("Setting Margin to ISOLATED")
    print("="*50)
    try:
        client.set_margin_mode('isolated', symbol)
        print("Success: Set to Isolated")
    except Exception as e:
        print(f"Failed to set margin mode: {e}")

    print("\n--- Test Buy (No tradeSide) ---")
    try:
        # ccxt should pick up isolated from market if set, but we can pass it
        res = client.create_order(symbol, 'market', 'buy', 5)
        print(f"  SUCCESS: {res['id']}")
    except Exception as e:
        print(f"  FAILED: {e}")

    print("\n--- Test Buy (tradeSide='open') ---")
    try:
        res = client.create_order(symbol, 'market', 'buy', 5, params={'tradeSide': 'open'})
        print(f"  SUCCESS: {res['id']}")
    except Exception as e:
        print(f"  FAILED: {e}")

if __name__ == "__main__":
    run_test()
