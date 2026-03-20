from data_processor import BinanceDataProcessor
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

def check_last_candle():
    processor = BinanceDataProcessor()
    symbol = 'BTCUSDT'
    interval = '4h'
    print(f"Checking last candle for {symbol} {interval}...")
    
    # Raw fetch
    df_raw = processor.get_historical_data(symbol, interval, '1 day ago UTC', 'now UTC')
    if df_raw.empty:
        print("No data fetched.")
        return
    
    # Processed fetch (simulating the fix)
    df_processed = df_raw.iloc[:-1].copy()
    
    last_row = df_processed.iloc[-1]
    last_ts = last_row['timestamp']
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=ZoneInfo('UTC'))
    
    now = datetime.now(ZoneInfo('UTC'))
    
    interval_ms = processor._get_interval_ms(interval)
    candle_end = last_ts + pd.Timedelta(milliseconds=interval_ms)
    
    print(f"Now (UTC): {now}")
    print(f"Candle used for prediction starts (UTC): {last_ts}")
    print(f"Candle used for prediction ends (UTC): {candle_end}")
    
    if now >= candle_end:
        print("\n✅ SUCCESS: The candle used for prediction is CLOSED.")
    else:
        print("\n🚩 FAILURE: Still using an unclosed candle.")

if __name__ == "__main__":
    check_last_candle()

if __name__ == "__main__":
    check_last_candle()
