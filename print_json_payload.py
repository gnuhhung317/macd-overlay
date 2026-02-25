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

def inspect(label, params={}):
    print(f"\n--- {label} ---")
    try:
        # Pass parameters to create_order
        client.create_order(symbol, 'market', 'buy', 5.0, None, params)
    except Exception as e:
        pass
    
    body = client.last_request_body
    if body:
        try:
            print(json.dumps(json.loads(body), indent=2))
        except:
            print(body)

inspect("Buy with params={'side': 'buy_single'}")
inspect("Sell with params={'side': 'sell_single'}")
