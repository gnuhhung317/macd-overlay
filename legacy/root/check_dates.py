import pandas as pd
import glob

files = glob.glob('data/processed/*.parquet')
for f in files:
    try:
        df = pd.read_parquet(f)
        min_ts = df['timestamp'].min()
        max_ts = df['timestamp'].max()
        print(f'{f}: {len(df)} rows | From {min_ts} to {max_ts}')
    except Exception as e:
        print(f"Error reading {f}: {e}")
