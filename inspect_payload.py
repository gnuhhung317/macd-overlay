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
print("--- Investigating Bare Buy ---")
try:
    client.create_order(symbol, 'market', 'buy', 5)
except Exception as e:
    print(f"Error: {e}")

print("\nLast Request Body:")
print(client.last_request_body)

print("\n--- Investigating Buy with tradeSide='open' ---")
try:
    client.create_order(symbol, 'market', 'buy', 5, {'tradeSide': 'open'})
except Exception as e:
    print(f"Error: {e}")

print("\nLast Request Body:")
print(client.last_request_body)
