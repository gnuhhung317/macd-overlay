import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ccxt_data_processor import CCXTDataProcessor
import json

with open("bot_config.json", "r") as f:
    config = json.load(f)

processor = CCXTDataProcessor(
    exchange_id="bitget",
    api_key=config['exchange'].get('api_key', ''),
    api_secret=config['exchange'].get('api_secret', ''),
    password=config['exchange'].get('passphrase', ''),
    use_futures=True
)

print("Fetching BTCUSDT 12h ...")
df = processor.get_historical_data("BTCUSDT", "12h", start_date="40 days ago UTC", end_date="now UTC")

print(f"Returned {len(df)} candles")
if not df.empty:
    print("HEAD:")
    print(df[['timestamp', 'open', 'close', 'volume']].head())
    print("TAIL:")
    print(df[['timestamp', 'open', 'close', 'volume']].tail())
