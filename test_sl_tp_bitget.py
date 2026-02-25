import ccxt
import json

def run_test():
    client = ccxt.bitget({
        'apiKey': 'bg_632fb4eef66cbb9267b4b4fc88f643d3',
        'secret': 'f14f6ec8f10e6fb845d9ef1ca272adec51fb03e23913e42e6dd84759eac0d5ca',
        'password': 'daylamacd8h',
        'options': {
            'defaultType': 'swap',
            'positionMode': False 
        }
    })

    symbol = 'XRP/USDT:USDT'
    
    print("\nTest A: Bare Buy (No tradeSide)")
    try:
        # We use a private method if we want to see exactly what CCXT constructs
        # but create_order is better. 
        # To see the request, we'll use verbose=True but filter the output.
        client.verbose = True
        res = client.create_order(symbol, 'market', 'buy', 5)
        print(f"  A SUCCESS: {res['id']}")
    except Exception as e:
        print(f"  A FAILED")
        # Verbose mode will have printed the request to stdout already.

if __name__ == "__main__":
    run_test()
