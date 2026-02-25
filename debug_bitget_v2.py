import ccxt
import json

def run_test():
    client = ccxt.bitget({
        'apiKey': 'bg_632fb4eef66cbb9267b4b4fc88f643d3',
        'secret': 'f14f6ec8f10e6fb845d9ef1ca272adec51fb03e23913e42e6dd84759eac0d5ca',
        'password': 'daylamacd8h',
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap',
            'positionMode': False # Force CCXT to Unilateral
        }
    })

    symbol = 'XRP/USDT:USDT'
    
    results = []
    
    # Test cases: (side, tradeSideValue)
    tests = [
        ('buy', None),
        ('buy', 'open'),
        ('sell', None),
        ('sell', 'open'),
    ]

    for side, tradeSide in tests:
        params = {}
        if tradeSide:
            params['tradeSide'] = tradeSide
            
        case_name = f"SIDE={side} | tradeSide={tradeSide}"
        print(f"Running: {case_name}")
        try:
            res = client.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=5,
                params=params
            )
            print(f"  ✅ SUCCESS: {res['id']}")
            results.append((case_name, "SUCCESS"))
        except Exception as e:
            print(f"  ❌ FAILED: {str(e)[:100]}")
            results.append((case_name, f"FAILED: {str(e)[:100]}"))

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for res in results:
        print(f"{res[0]} -> {res[1]}")

if __name__ == "__main__":
    run_test()
