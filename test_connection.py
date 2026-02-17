import ccxt
import time

def test_load_markets():
    print("Initializing exchange...")
    exchange = ccxt.bitget({
        'timeout': 10000,
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}
    })
    
    print("Loading markets...")
    try:
        start = time.time()
        markets = exchange.load_markets()
        print(f"Loaded {len(markets)} markets in {time.time() - start:.2f}s")
        
        # Print a few symbols to confirm correctness
        symbols = [m for m in markets.keys() if 'USDT' in m][:5]
        print(f"Sample symbols: {symbols}")
        
    except Exception as e:
        print(f"Error loading markets: {e}")

if __name__ == "__main__":
    test_load_markets()
