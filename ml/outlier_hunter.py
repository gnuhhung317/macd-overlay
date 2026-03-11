import os
import gc
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
from typing import List, Dict, Tuple, Optional

warnings.filterwarnings('ignore')

# ============================================================
# CONFIG & PATHS
# ============================================================
BASE_DIR = Path(r"d:\Code\Projects\self-projects\macd-overlay - Copy")
SYMBOLS_DIR = BASE_DIR / "data" / "processed" / "symbols_v3"
MODEL_PATH = BASE_DIR / "ml" / "training" / "models" / "1h" / "ensemble_lgbm_tabular.joblib"
META_PATH = BASE_DIR / "ml" / "training" / "models" / "1h" / "ensemble_meta.joblib"

SCAN_HORIZON = 120  # 5 days for 1H candles
START_DATE = '2025-01-01'
OUTLIER_MFE_THRESHOLD = 6.0
OUTLIER_MAE_LIMIT = -2.0

def load_ml_assets():
    if not META_PATH.exists() or not MODEL_PATH.exists():
        print("❌ Missing model or meta file!")
        return None, [], 0.6
    meta = joblib.load(META_PATH)
    clf = joblib.load(MODEL_PATH)
    features = meta.get('features', []) if isinstance(meta, dict) else meta
    threshold = meta.get('threshold', 0.6)
    return clf, features, threshold

def calculate_features_extended(df):
    """Calculate features including DNA markers like vol_compression."""
    df = df.copy()
    
    # 0. Basic Price/Time
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 1. Standard Indicators (from backtest_sniper)
    if 'ema_20' not in df.columns:
        df['ema_20'] = df['close'].ewm(span=20).mean()
    if 'ema_50' not in df.columns:
        df['ema_50'] = df['close'].ewm(span=50).mean()
    if 'atr_14' not in df.columns:
        hl = df['high'] - df['low']
        hc = np.abs(df['high'] - df['close'].shift())
        lc = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()

    if 'atr_pct' not in df.columns:
        df['atr_pct'] = (df['atr_14'] / df['close']) * 100
    
    # Sniper-specific features (from backtest_sniper.py)
    if 'upper_wick_ratio' not in df.columns:
        df['upper_wick_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-9)
    if 'dist_to_ema50_atr' not in df.columns:
        df['dist_to_ema50_atr'] = (df['close'] - df['ema_50']) / (df['atr_14'] + 1e-9)
    if 'vol_acceleration' not in df.columns:
        df['vol_acceleration'] = df['volume'] / (df['volume'].shift(1) + 1e-9)

    # 2. DNA Markers
    # Vol Compression: Ratio of current ATR to long-term average ATR (lower = more compressed)
    if 'vol_compression' not in df.columns:
        df['vol_compression'] = df['atr_14'] / (df['atr_14'].rolling(100).mean() + 1e-9)
        
    if 'rsi_14' not in df.columns:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['rsi_14'] = 100 - (100 / (1 + rs))

    if 'vol_ratio' not in df.columns:
        vol_sma = df['volume'].rolling(20).mean().shift(1)
        df['vol_ratio'] = df['volume'] / (vol_sma + 1e-9)

    return df

def identify_signals(df, features, clf, threshold):
    """Filter for Ignition signals and get ML probabilities."""
    # 1. Stage 1 Filter: Ignition Bar
    vol_sma = df['volume'].rolling(20).mean().shift(1)
    c1 = (df['close'] > df['open']) & (df['close'] > df['ema_20'])
    c2 = ((df['close'] - df['open']) / df['open']) > 0.015
    c3 = (df['volume'] > vol_sma * 1.5) & (df['volume'] < vol_sma * 4.0)
    c4 = (df['rsi_14'] >= 55) & (df['rsi_14'] <= 72)
    
    potential_indices = df[c1 & c2 & c3 & c4].index
    if len(potential_indices) == 0:
        return pd.DataFrame()
        
    # 2. Stage 2: ML Scoring
    X_batch = df.loc[potential_indices, features].apply(pd.to_numeric, errors='coerce').fillna(0)
    probas = clf.predict_proba(X_batch)
    
    prob_long = probas[:, 1]
    prob_short = probas[:, 2]
    
    results = df.loc[potential_indices].copy()
    results['prob_long'] = prob_long
    results['prob_short'] = prob_short
    
    # Keep only signals above threshold
    final_signals = results[(results['prob_long'] > threshold) | (results['prob_short'] > threshold)]
    return final_signals

def hunt_outliers_vectorized(df, signals_df, horizon=SCAN_HORIZON):
    """Unbound Forward Scan to find true MFE/MAE."""
    if signals_df.empty:
        return pd.DataFrame()
        
    outliers = []
    symbol = signals_df['symbol'].iloc[0] if 'symbol' in signals_df.columns else "UNKNOWN"
    
    for idx, row in signals_df.iterrows():
        entry_idx = df.index.get_loc(idx)
        if entry_idx + horizon >= len(df):
            continue
            
        entry_price = row['close']
        atr = row['atr_14']
        signal_type = 'LONG' if row['prob_long'] > row['prob_short'] else 'SHORT'
        
        # Slice future window
        future_window = df.iloc[entry_idx + 1 : entry_idx + 1 + horizon].reset_index(drop=True)
        
        if signal_type == 'LONG':
            # MFE is based on Highs
            max_p = future_window['high'].max()
            max_idx = future_window['high'].idxmax()
            # MAE is based on Lows
            min_p = future_window['low'].min()
            min_idx = future_window['low'].idxmin()
            
            mfe_atr = (max_p - entry_price) / (atr + 1e-9)
            mae_atr = (min_p - entry_price) / (atr + 1e-9) # Negative for LONG
        else: # SHORT
            # MFE is based on Lows
            max_p = future_window['low'].min()
            max_idx = future_window['low'].idxmin()
            # MAE is based on Highs
            min_p = future_window['high'].max()
            min_idx = future_window['high'].idxmax()
            
            mfe_atr = (entry_price - max_p) / (atr + 1e-9)
            mae_atr = (entry_price - min_p) / (atr + 1e-9) # Negative for SHORT

        outliers.append({
            'timestamp': row['timestamp'],
            'symbol': symbol,
            'type': signal_type,
            'entry_price': entry_price,
            'atr': atr,
            'mfe_atr': mfe_atr,
            'mae_atr': mae_atr,
            'bars_to_mfe': max_idx + 1,
            'bars_to_mae': min_idx + 1,
            'vol_compression': row.get('vol_compression', 0),
            'rsi_14': row.get('rsi_14', 0),
            'vol_ratio': row.get('vol_ratio', 0),
            'prob_long': row['prob_long'],
            'prob_short': row['prob_short']
        })
        
    return pd.DataFrame(outliers)

def main():
    print(f"\n{'='*60}")
    print(f"🕵️ OUTLIER HUNTER: UNBOUND FORWARD SCAN")
    print(f"Horizon: {SCAN_HORIZON} bars | Threshold: {OUTLIER_MFE_THRESHOLD} ATR")
    print(f"{'='*60}")
    
    clf, features, threshold = load_ml_assets()
    if clf is None: return
    
    all_files = list(SYMBOLS_DIR.glob("*.parquet"))
    print(f"Scanning {len(all_files)} symbols...")
    
    all_outliers = []
    
    for i, file_path in enumerate(all_files):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(all_files)}...")
            
        try:
            df = pd.read_parquet(file_path)
            if df.empty: continue
            
            symbol = Path(file_path).stem.replace('_USDT', '').replace('USDT', '')
            df['symbol'] = symbol
            
            # Pre-calculate features
            df = calculate_features_extended(df)
            
            # Filter for signals
            signals = identify_signals(df, features, clf, threshold)
            if signals.empty: continue
            
            start_ts = pd.to_datetime(START_DATE)
            signals = signals[signals['timestamp'] >= start_ts]
            
            # Hunt outliers
            symbol_outliers = hunt_outliers_vectorized(df, signals)
            if not symbol_outliers.empty:
                all_outliers.append(symbol_outliers)
                
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            
    if not all_outliers:
        print("No outliers found.")
        return
        
    full_df = pd.concat(all_outliers, ignore_index=True)
    
    # DNA EXTRACTION: Filtering for Elite Outliers
    # Case 1: High MFE
    # Case 2: Clean Move (No deep MAE before hitting MFE)
    elite_longs = full_df[
        (full_df['type'] == 'LONG') & 
        (full_df['mfe_atr'] >= OUTLIER_MFE_THRESHOLD) &
        ~((full_df['mae_atr'] < OUTLIER_MAE_LIMIT) & (full_df['bars_to_mae'] < full_df['bars_to_mfe']))
    ]
    
    elite_shorts = full_df[
        (full_df['type'] == 'SHORT') & 
        (full_df['mfe_atr'] >= OUTLIER_MFE_THRESHOLD) &
        ~((full_df['mae_atr'] < OUTLIER_MAE_LIMIT) & (full_df['bars_to_mae'] < full_df['bars_to_mfe']))
    ]
    
    elite_all = pd.concat([elite_longs, elite_shorts])
    
    print(f"\n✅ RESULTS:")
    print(f"Total Signals Scanned: {len(full_df)}")
    print(f"Elite Long Outliers:  {len(elite_longs)}")
    print(f"Elite Short Outliers: {len(elite_shorts)}")
    
    if not elite_all.empty:
        dna_summary = elite_all.describe()
        print("\n--- Outlier DNA (Means) ---")
        print(f"Avg Vol Compression: {elite_all['vol_compression'].mean():.4f}")
        print(f"Avg RSI:             {elite_all['rsi_14'].mean():.2f}")
        print(f"Avg Bars to MFE:     {elite_all['bars_to_mfe'].mean():.1f}")
        
    # Save results
    output_all = BASE_DIR / "ml" / "outlier_results_full.csv"
    output_elite = BASE_DIR / "ml" / "outlier_dna_elite.csv"
    
    full_df.to_csv(output_all, index=False)
    elite_all.to_csv(output_elite, index=False)
    
    print(f"\nFull scan saved:  {output_all}")
    print(f"Elite DNA saved:  {output_elite}")

if __name__ == "__main__":
    main()
