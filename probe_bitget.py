import ccxt
from datetime import datetime, timezone
import time

def test_bitget_fetch():
    exchange = ccxt.bitget({
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}
    })
    
    symbol = 'BTC/USDT:USDT'
    # Test dates: 2020, 2021, 2022, 2023, 2024
    years = [2020, 2021, 2022, 2023, 2024]
    
    print(f"Probing {symbol} history...")
    for year in years:
        since = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, '1h', since=since, limit=5)
            if ohlcv:
                print(f"Year {year}: Success! First candle: {datetime.fromtimestamp(ohlcv[0][0]/1000, tz=timezone.utc)}")
            else:
                print(f"Year {year}: Empty response.")
        except Exception as e:
            print(f"Year {year}: Error - {e}")

    # Test batching forward
    print("\nTesting batching forward from 2023-01-01...")
    since = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    all_ohlcv = []
    for i in range(3):
        ohlcv = exchange.fetch_ohlcv(symbol, '1h', since=since, limit=1000)
        if not ohlcv:
            print("  No more data.")
            break
        all_ohlcv.extend(ohlcv)
        print(f"  Batch {i+1}: {len(ohlcv)} rows. Last: {datetime.fromtimestamp(ohlcv[-1][0]/1000, tz=timezone.utc)}")
        since = ohlcv[-1][0] + (3600 * 1000) # Next hour
        time.sleep(exchange.rateLimit / 1000)

if __name__ == "__main__":
    test_bitget_fetch()
