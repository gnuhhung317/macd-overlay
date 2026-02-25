import ccxt
import json

def run_test():
    client = ccxt.bitget({
        'apiKey': 'bg_632fb4eef66cbb9267b4b4fc88f643d3',
        'secret': 'f14f6ec8f10e6fb845d9ef1ca272adec51fb03e23913e42e6dd84759eac0d5ca',
        'password': 'daylamacd8h',
        'options': {
            'defaultType': 'swap'
        }
    })

    symbol = 'XRP/USDT:USDT'
    
    print("\n" + "="*50)
    print("1. Setting Mode to ONE_WAY (Unilateral)")
    print("="*50)
    try:
        res = client.privateMixPostV2MixAccountSetPositionMode({
            'productType': 'usdt-futures',
            'posMode': 'one_way_mode'
        })
        print(f"Set Unilateral Success: {res['msg']}")
    except Exception as e:
        print(f"Set Unilateral Failed (Maybe already set): {e}")

    print("\n--- Test Buy in Unilateral (No tradeSide) ---")
    try:
        res = client.create_order(symbol, 'market', 'buy', 5, params={'positionMode': False})
        print(f"  SUCCESS: {res['id']}")
    except Exception as e:
        print(f"  FAILED: {e}")

    print("\n" + "="*50)
    print("2. Setting Mode to HEDGE (Two-Way)")
    print("="*50)
    try:
        res = client.privateMixPostV2MixAccountSetPositionMode({
            'productType': 'usdt-futures',
            'posMode': 'hedge_mode'
        })
        print(f"Set Hedge Success: {res['msg']}")
    except Exception as e:
        print(f"Set Hedge Failed: {e}")

    print("\n--- Test Buy in Hedge (No tradeSide) ---")
    try:
        res = client.create_order(symbol, 'market', 'buy', 5)
        print(f"  SUCCESS: {res['id']}")
    except Exception as e:
        print(f"  FAILED: {e}")

    print("\n--- Test Buy in Hedge (tradeSide='open') ---")
    try:
        res = client.create_order(symbol, 'market', 'buy', 5, params={'tradeSide': 'open'})
        print(f"  SUCCESS: {res['id']}")
    except Exception as e:
        print(f"  FAILED: {e}")

if __name__ == "__main__":
    run_test()
