import ccxt
import json

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

def test_combination(side, params):
    print(f"Testing SIDE={side} | PARAMS={params}")
    try:
        res = client.create_order(symbol, 'market', side, 5, None, params)
        print(f"  ✅ SUCCESS: {res['id']}")
        return True
    except Exception as e:
        msg = str(e)
        if '40774' in msg:
            print("  ❌ 40774: Mismatch")
        else:
            print(f"  ❌ OTHER: {msg[:100]}")
        return False

# We know the account is in Hedge mode locally, so let's see which ones PASS in Hedge mode.
# Then we can infer what might work in Unilateral.
combinations = [
    # Baseline
    {},
    
    # tradeSide variations
    {'tradeSide': 'open'},
    {'tradeSide': 'close'},
    {'tradeSide': 'buy'},
    {'tradeSide': 'sell'},
    
    # posSide variations
    {'posSide': 'long'},
    {'posSide': 'short'},
    
    # Combined (For Hedge mode)
    {'tradeSide': 'open', 'posSide': 'long'},
    {'tradeSide': 'open', 'posSide': 'short'},
]

print("--- TESTING FOR 'buy' ---")
for p in combinations:
    test_combination('buy', p)

print("\n--- TESTING FOR 'sell' ---")
for p in combinations:
    test_combination('sell', p)
