import sys
sys.path.append('ml')
import pandas as pd
from bot.data_provider import BinanceDataProcessor
from ml.multi_timeframe_pipeline import calculate_features_for_timeframe
from ml.inference import InferenceEngine
import numpy as np

processor = BinanceDataProcessor()
df = processor.get_historical_data('ETHUSDT', '1d', '300 days ago UTC', 'now UTC')
engine = InferenceEngine('1d')
df_calc = calculate_features_for_timeframe(df.copy(), '1d')

# Manually trigger signal
df_calc.loc[df_calc.index[-1], 'macd_cross_up'] = 1

# Force btc dummy context
df_calc.loc[df_calc.index[-1], 'btc_is_bull_regime'] = 1
df_calc.loc[df_calc.index[-1], 'btc_trend_strength'] = 1
df_calc.loc[df_calc.index[-1], 'rs_vs_btc'] = 0.5
df_calc.loc[df_calc.index[-1], 'rs_vs_btc_sma7'] = 0.5

captured_rows = []
orig = engine._prepare_single_row
def hook(row, feat, scaler):
    captured_rows.append({f: row.get(f, 0.0) for f in feat})
    return orig(row, feat, scaler)
engine._prepare_single_row = hook

print("Calling predict...")
res = engine.predict('ETHUSDT', df_calc)
print(f'Prediction Output: {res}')

for i, x in enumerate(captured_rows):
    regime = x.get('btc_is_bull_regime')
    trend = x.get('btc_trend_strength')
    rs = x.get('rs_vs_btc')
    sma = x.get('rs_vs_btc_sma7')
    print(f'Model {i}: BTC Regime: {regime}, Trend: {trend}, RS: {rs}, RS_SMA: {sma}')
