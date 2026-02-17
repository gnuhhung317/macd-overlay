import ccxt

def probe_markets():
    exchange = ccxt.bitget({
        'options': {'defaultType': 'swap'}
    })
    markets = exchange.load_markets()
    
    # Check a few random markets
    print(f"Total markets: {len(markets)}")
    
    count_linear = 0
    count_active = 0
    count_quote_usdt = 0
    
    for symbol, m in list(markets.items())[:5]:
        print(f"Symbol: {symbol}")
        print(f"  Linear: {m.get('linear')}")
        print(f"  Active: {m.get('active')}")
        print(f"  Quote: {m.get('quote')}")
        print(f"  Type: {m.get('type')}")
        print("-" * 20)

    for m in markets.values():
        if m.get('linear'): count_linear += 1
        if m.get('active'): count_active += 1
        if m.get('quote') == 'USDT': count_quote_usdt += 1
        
    print(f"Stats:")
    print(f"  Linear: {count_linear}")
    print(f"  Active: {count_active}")
    print(f"  Quote=USDT: {count_quote_usdt}")

if __name__ == "__main__":
    probe_markets()
