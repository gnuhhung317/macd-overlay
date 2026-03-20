
import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# BENCHMARK LOGIC (From backtest_sniper.py)
# ============================================================
def get_backtest_signals(df, features, clf, threshold):
    # Indicator Logic from backtest_sniper.py
    vol_sma = df['volume'].rolling(20).mean().shift(1)
    vol_sma = df['volume'].rolling(20).mean().shift(1)
    c1 = (df['close'] > df['open']) & (df['close'] > df['ema_20'])
    c2 = ((df['close'] - df['open']) / df['open']) > 0.015
    c3 = (df['volume'] > vol_sma * 1.5) & (df['volume'] < vol_sma * 4.0)
    c4 = (df['rsi_14'] >= 55) & (df['rsi_14'] <= 72)
    
    ignition_mask = (c1 & c2 & c3 & c4)
    hits = df[ignition_mask].copy()
    if hits.empty: return pd.DataFrame()
    
    X = hits[features].apply(pd.to_numeric, errors='coerce').fillna(0)
    probas = clf.predict_proba(X)
    hits['prob_long'] = probas[:, 1]
    hits['prob_short'] = probas[:, 2]
    hits['source'] = 'backtest'
    return hits[probas.max(axis=1) >= threshold]

# ============================================================
# SCANNER LOGIC (From scan_sniper.py)
# ============================================================
def get_scan_signals(df, features, clf, threshold):
    df = df.copy()
    # Always exclude incomplete candle in scanner
    df = df.iloc[:-1]
    
    # Indicator logic from scan_sniper.py
    vol_sma = df['volume'].rolling(20).mean().shift(1)
    c1 = (df['close'] > df['open']) & (df['close'] > df['ema_20'])
    c2 = ((df['close'] - df['open']) / df['open']) > 0.015
    c3 = (df['volume'] > vol_sma * 1.5) & (df['volume'] < vol_sma * 4.0)
    c4 = (df['rsi_14'] >= 55) & (df['rsi_14'] <= 72)
    
    hits = df[c1 & c2 & c3 & c4].copy()
    if hits.empty: return pd.DataFrame()
    
    X = hits[features].apply(pd.to_numeric, errors='coerce').fillna(0)
    probas = clf.predict_proba(X)
    hits['prob_long'] = probas[:, 1]
    hits['prob_short'] = probas[:, 2]
    
    # Updated: Uses meta threshold
    hits['final_signal'] = 'WAIT'
    hits.loc[hits['prob_long'] > threshold, 'final_signal'] = '🚀 LONG'
    hits.loc[hits['prob_short'] > threshold, 'final_signal'] = '💀 SHORT'
    hits['source'] = 'scanner'
    return hits[hits['final_signal'] != 'WAIT']

# ============================================================
# SYNC WORKER LOGIC (From sync_worker.py - Simulated)
# ============================================================
def get_worker_signals_simulated(df, features, clf, threshold, extras=None):
    # extras should contain {'df_1d': ..., 'btc_1h': ...} consistent with live logic
    df_completed = df.iloc[:-1].copy()
    
    # 1. Simulate FETCH 500 bars
    # In live loop, it fetches 500 nến, then we calculate features on those 500
    df_window = df_completed.tail(500).copy()
    
    print(f"\n[Worker Sim] Window size: {len(df_window)}")
    print(f"[Worker Sim] Range: {df_window['timestamp'].min()} to {df_window['timestamp'].max()}")
    
    target_ts = pd.to_datetime("2026-03-10 14:00:00")
    if target_ts in df_window['timestamp'].values:
        print(f"[Worker Sim] FOUND target {target_ts} in window.")
    else:
        print(f"[Worker Sim] MISSING target {target_ts} in window!")
        # Let's see what's actually there
        print(f"[Worker Sim] Last 5 timestamps in window:\n{df_window['timestamp'].tail(5)}")

    if len(df_window) < 200: 
        print(f"[Worker Sim] Window too small ({len(df_window)}). Need 200.")
        return pd.DataFrame() 

    # 2. Re-calculate indicators on THIS window only (Crucial for drift check)
    vol_sma = df_window['volume'].rolling(20).mean().shift(1)
    
    c1 = (df_window['close'] > df_window['open']) & (df_window['close'] > df_window['ema_20'])
    c2 = ((df_window['close'] - df_window['open']) / df_window['open']) > 0.015
    c3 = (df_window['volume'] > vol_sma * 1.5) & (df_window['volume'] < vol_sma * 4.0)
    c4 = (df_window['rsi_14'] >= 55) & (df_window['rsi_14'] <= 72)
    
    # Debug T1 filters for the signal timestamp if known
    if target_ts in df_window['timestamp'].values:
        idx = df_window[df_window['timestamp'] == target_ts].index[0]
        row = df_window.loc[idx]
        v_sma_val = vol_sma.loc[idx]
        print(f"\n--- T1 DEBUG (Worker Sim) at {target_ts} ---")
        print(f"Close > Open: {row['close'] > row['open']} ({row['close']:.4f} vs {row['open']:.4f})")
        print(f"Close > EMA20: {row['close'] > row['ema_20']} ({row['close']:.4f} vs {row['ema_20']:.4f})")
        print(f"Body Size > 1.5%: {((row['close'] - row['open']) / row['open']) > 0.015:.4f}")
        print(f"Vol Ignition: {v_sma_val * 1.5:.2f} < {row['volume']:.2f} < {v_sma_val * 4.0:.2f}")
        print(f"RSI Fresh (55-72): {55 <= row['rsi_14'] <= 72} (RSI: {row['rsi_14']:.2f})")
    
    hits = df_window[c1 & c2 & c3 & c4].copy()
    if hits.empty: return pd.DataFrame()
    
    X = hits[features].apply(pd.to_numeric, errors='coerce').fillna(0)
    probas = clf.predict_proba(X)
    hits['prob_long'] = probas[:, 1]
    hits['prob_short'] = probas[:, 2]
    hits['source'] = 'worker'
    
    return hits[probas.max(axis=1) >= threshold]

# ============================================================
# MAIN VERIFICATION
# ============================================================
if __name__ == "__main__":
    BASE_DIR = Path(r"d:\Code\Projects\self-projects\macd-overlay - Copy")
    SYMBOLS_DIR = BASE_DIR / "data" / "processed" / "symbols_v2"
    MODEL_PATH = BASE_DIR / "ml" / "training" / "models" / "1h" / "ensemble_lgbm_tabular.joblib"
    META_PATH = BASE_DIR / "ml" / "training" / "models" / "1h" / "ensemble_meta.joblib"
    
    clf = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH)
    features = meta['features']
    threshold = meta['threshold']
    
    symbol = "BTCUSDT"
    file_path = SYMBOLS_DIR / f"{symbol}.parquet"
    df = pd.read_parquet(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"\nComparing signals for {symbol} (Threshold: {threshold})...")
    
    bt_sigs = get_backtest_signals(df, features, clf, threshold)
    sc_sigs = get_scan_signals(df, features, clf, threshold)
    wk_sigs = get_worker_signals_simulated(df, features, clf, threshold)
    
    print(f"Backtest Signals: {len(bt_sigs)}")
    print(f"Scanner Signals:  {len(sc_sigs)}")
    print(f"Worker Signals:   {len(wk_sigs)}")
    
    if not wk_sigs.empty:
        print("\nSample Worker Logic Features (Last Row):")
        print(wk_sigs.iloc[-1][['timestamp', 'ema_200_1d_dist', 'rsi_14_1d', 'btc_is_bull_regime']])
    else:
        # Check if features were calculated but signal failed threshold
        # We can't easily check without rerunning part of the logic locally
        print("\nWorker detected 0 signals. This usually means T1 filters failed or MTF features are NaN.")
    
    # Merge and compare latest
    combined_list = []
    if not bt_sigs.empty: combined_list.append(bt_sigs[['timestamp', 'prob_long', 'prob_short', 'source']])
    if not sc_sigs.empty: combined_list.append(sc_sigs[['timestamp', 'prob_long', 'prob_short', 'source']])
    if not wk_sigs.empty: combined_list.append(wk_sigs[['timestamp', 'prob_long', 'prob_short', 'source']])
    
    if combined_list:
        all_combined = pd.concat(combined_list)
        print("\nLast 5 unique signal timestamps:")
        latest_ts = sorted(all_combined['timestamp'].unique())[-5:]
        for ts in latest_ts:
            ts_dt = pd.to_datetime(ts)
            print(f"\nAt {ts_dt}:")
            for src in ['backtest', 'scanner', 'worker']:
                row = all_combined[(all_combined['timestamp'] == ts) & (all_combined['source'] == src)]
                if not row.empty:
                    print(f"  [{src:8}] ProbL: {row.iloc[0]['prob_long']:.4f} | ProbS: {row.iloc[0]['prob_short']:.4f} -> PASS")
                else:
                    print(f"  [{src:8}] NO SIGNAL")
    else:
        print("\nNo signals detected by any script in the processing window.")
