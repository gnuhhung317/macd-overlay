import pandas as pd
import numpy as np

# Create mock 4h data
timestamps = pd.date_range('2024-01-01 00:00:00', periods=6, freq='4h')
df = pd.DataFrame({
    'timestamp': timestamps,
    'open': [100, 105, 110, 108, 115, 120],
    'high': [106, 112, 115, 110, 122, 125],
    'low': [95, 102, 105, 105, 112, 118],
    'close': [105, 110, 108, 107, 120, 122],
    'volume': [10, 20, 15, 25, 30, 10],
    'close_time': [1, 2, 3, 4, 5, 6],
    'quote_volume': [0,0,0,0,0,0],
    'trades': [0,0,0,0,0,0],
    'taker_buy_base': [0,0,0,0,0,0],
    'taker_buy_quote': [0,0,0,0,0,0],
    'ignore': [0,0,0,0,0,0]
})

print("Original 4h data:")
print(df[['timestamp', 'open', 'high', 'low', 'close', 'volume']])

freq = '8h'.replace('m', 'min').replace('h', 'h').replace('d', 'D')
print(f"Resampling to {freq}...")

df_resampled = df.set_index('timestamp').resample(freq).agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum',
    'close_time': 'last',
    'quote_volume': 'sum',
    'trades': 'sum',
    'taker_buy_base': 'sum',
    'taker_buy_quote': 'sum',
    'ignore': 'last'
}).dropna().reset_index()

print("Resampled 8h data:")
print(df_resampled[['timestamp', 'open', 'high', 'low', 'close', 'volume']])
