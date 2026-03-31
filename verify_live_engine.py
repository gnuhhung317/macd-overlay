import os, sys
import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"d:\Code\Projects\self-projects\macd-overlay - Copy")
sys.path.append(str(BASE_DIR))

from sniper_bot.feature import calculate_features, apply_feature_shift

def main():
    sym = 'ETHUSDT'
    target_time = pd.Timestamp('2026-03-20 12:00:00') # The time the ignition bar fully closed
    
    print(f"Comparing LIVE ENGINE features vs PARQUET backtest features for {sym} at {target_time}...\n")
    
    # 1. Load Parquet Data (Backtest truth)
    pq_path = BASE_DIR / f"bitget-data/symbols_v3/{sym}.parquet"
    if not pq_path.exists():
        print(f"Parquet missing: {pq_path}")
        return
        
    df_pq = pd.read_parquet(pq_path)
    df_pq['timestamp'] = pd.to_datetime(df_pq['timestamp']).dt.tz_localize(None)
    row_pq = df_pq[df_pq['timestamp'] == target_time]
    
    if row_pq.empty:
        print("Timestamp not found in Parquet!")
        return
    row_pq = row_pq.iloc[0]
    
    # 2. Simulate Live Engine
    print("Simulating Live Engine data feed...\n")
    # Load raw data like the API would return it up to target_time + 1 hour (so target is the closed bar)
    raw_path = BASE_DIR / f"bitget-data/ohlcv/{sym}_USDT.parquet"
    df_raw = pd.read_parquet(raw_path)
    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp']).dt.tz_localize(None)
    
    # Filter raw data exactly how it would be at target_time + 30 mins
    # The last closed bar would be target_time. The open bar would be target_time + 1h.
    open_time = target_time + pd.Timedelta(hours=1)
    df_live = df_raw[df_raw['timestamp'] <= open_time].copy()
    
    # btc_df and df_1d need to be built similarly
    btc_raw = pd.read_parquet(BASE_DIR / "bitget-data/ohlcv/BTCUSDT_USDT.parquet")
    btc_raw['timestamp'] = pd.to_datetime(btc_raw['timestamp']).dt.tz_localize(None)
    btc_live = btc_raw[btc_raw['timestamp'] <= target_time].copy()
    
    import numpy as np
    btc_live['log_returns'] = np.log(btc_live['close'] / (btc_live['close'].shift(1) + 1e-9))
    btc_live['sma_200'] = btc_live['close'].rolling(200).mean()
    tr = pd.concat([btc_live['high'] - btc_live['low'], abs(btc_live['high'] - btc_live['close'].shift(1)), abs(btc_live['low'] - btc_live['close'].shift(1))], axis=1).max(axis=1)
    pdm = btc_live['high'].diff(); mdm = -btc_live['low'].diff()
    pdm = pdm.where((pdm > mdm) & (pdm > 0), 0); mdm = mdm.where((mdm > pdm) & (mdm > 0), 0)
    atr_s = tr.rolling(14).mean()
    pdi = 100 * (pdm.rolling(14).mean() / atr_s.replace(0, np.nan))
    mdi = 100 * (mdm.rolling(14).mean() / atr_s.replace(0, np.nan))
    btc_live['adx'] = (100 * abs(pdi - mdi) / (pdi + mdi).replace(0, np.nan)).rolling(14).mean()
    
    # 1D
    df_1d_live = df_live.set_index('timestamp').resample('1D').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    ).dropna().reset_index()
    
    print(f"Feeding {len(df_live)} candles into engine...")
    
    # -------- LIVE ENGINE LOGIC EXACTLY FROM sniper_scanner.py ---------
    # Use only completed candles
    df_calc = df_live.iloc[:-1].copy() # This drops the open_time bar, making target_time the last bar
    
    df_calc = calculate_features(df_calc, df_1d=df_1d_live, btc_df=btc_live)
    df_calc = apply_feature_shift(df_calc)
    
    last_calc_ts = df_calc['timestamp'].iloc[-1]
    print(f"Extracted Live Engine target timestamp: {last_calc_ts}")
    if last_calc_ts != target_time:
        print("Mismatch in target timestamps!")
        return
        
    row_live = df_calc.iloc[-1]
    
    # 3. Comparison
    features = ['rsi_14', 'upper_wick_ratio', 'rs_vs_btc', 'dist_to_ema50_atr', 'close', 'open']
    print(f"\n{'FEATURE':>20} | {'LIVE ENGINE':>15} | {'PARQUET (BT)':>15} | MATCH?")
    print("-" * 65)
    for f in features:
        val_live = row_live.get(f, 'N/A')
        val_pq = row_pq.get(f, 'N/A')
        
        match = "✅"
        if isinstance(val_live, float) and isinstance(val_pq, float):
            if abs(val_live - val_pq) > 1e-4:
                match = "❌"
        elif val_live != val_pq:
            match = "❌"
            
        if isinstance(val_live, float): val_live = f"{val_live:.4f}"
        if isinstance(val_pq, float): val_pq = f"{val_pq:.4f}"
        
        print(f"{f:>20} | {str(val_live):>15} | {str(val_pq):>15} | {match}")

if __name__ == '__main__':
    main()
