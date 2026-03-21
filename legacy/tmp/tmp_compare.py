import pandas as pd
import numpy as np

orig = pd.read_parquet('data/processed/features_1h_btc_context.parquet', filters=[('symbol', '==', 'ETHUSDT')])
orig['timestamp'] = pd.to_datetime(orig['timestamp']).dt.tz_localize(None)
sync = pd.read_parquet('data/processed/symbols_v3/ETHUSDT.parquet')
sync['timestamp'] = pd.to_datetime(sync['timestamp']).dt.tz_localize(None)

ts = pd.Timestamp('2025-06-01 12:00:00')
o = orig[orig['timestamp'] == ts].iloc[0]
s = sync[sync['timestamp'] == ts].iloc[0]

model_feats = ['rsi_14', 'atr_14', 'volume_ratio', 'adx', 'btc_is_bull_regime', 'btc_trend_strength', 
               'btc_corr', 'rs_vs_btc', 'ema_200_1d_dist', 'rsi_14_1d', 'close', 'open', 'volume']
print(f"{'Feature':30s} {'Original':>15s} {'Sync':>15s} {'Diff':>12s}")
print('-'*75)
for f in model_feats:
    ov = o.get(f, float('nan'))
    sv = s.get(f, float('nan'))
    try:
        diff = abs(float(ov) - float(sv))
        print(f"{f:30s} {float(ov):15.6f} {float(sv):15.6f} {diff:12.6f}")
    except:
        print(f"{f:30s} {str(ov):>15s} {str(sv):>15s}")

# Check feature shift
print("\n--- Feature Shift Check ---")
ts1 = pd.Timestamp('2025-06-01 11:00:00')
ts2 = pd.Timestamp('2025-06-01 12:00:00')
o1 = orig[orig['timestamp'] == ts1].iloc[0]
o2 = orig[orig['timestamp'] == ts2].iloc[0]
s1 = sync[sync['timestamp'] == ts1].iloc[0]
s2 = sync[sync['timestamp'] == ts2].iloc[0]
print(f"Orig: rsi@T={o2['rsi_14']:.4f}  rsi@T-1={o1['rsi_14']:.4f}")
print(f"Sync: rsi@T={s2['rsi_14']:.4f}  rsi@T-1={s1['rsi_14']:.4f}")
print(f"Orig rsi@T == Sync rsi@T-1? {abs(o2['rsi_14'] - s1['rsi_14']) < 0.01}")
print(f"Orig close@T={o2['close']:.2f}  Sync close@T={s2['close']:.2f}  Same? {abs(o2['close'] - s2['close']) < 0.01}")
