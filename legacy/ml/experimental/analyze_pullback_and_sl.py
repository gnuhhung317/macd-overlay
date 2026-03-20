
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
import random

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.inference import InferenceEngine
from ml.data_pipeline import calculate_features

def load_local_data(symbol, interval='4h', days=120):
    try:
        ohlcv_dir = Path(__file__).parent.parent.parent / 'data' / 'ohlcv'
        file_path = ohlcv_dir / f"{symbol}_USDT.parquet"
        if not file_path.exists(): return None
        df = pd.read_parquet(file_path)
        if 'open_time' in df.columns: df = df.rename(columns={'open_time': 'timestamp'})
        if df['timestamp'].dtype == 'int64': df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        else: df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days+30)
        if df['timestamp'].dt.tz is None: df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        df = df[df['timestamp'] >= cutoff]
        if len(df) == 0: return None
        if interval != '1h':
            df = df.set_index('timestamp')
            df_resampled = df.resample(interval).agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna().reset_index()
            return df_resampled
        return df.reset_index(drop=True)
    except: return None

def prepare_row_features(row, feature_names, scaler):
    # Quick feature prep
    val_map = row.to_dict() # faster lookup?
    # Actually just creating numpy array might be faster if we map indices
    # But sticking to dataframe method for safety
    X = pd.DataFrame(0, index=[0], columns=feature_names)
    present_cols = [c for c in feature_names if c in row.index]
    X[present_cols] = row[present_cols].values
    if scaler: X = pd.DataFrame(scaler.transform(X), columns=feature_names)
    return X

def analyze_symbol(symbol, interval, days, engine):
    df = load_local_data(symbol, interval, days)
    if df is None or len(df) < 250: return []
    try: df = calculate_features(df)
    except: return []
    
    # Get crossovers
    crossovers = []
    # Identify crossover points - check entire provided dataframe range!
    cross_ups = df[df['macd_cross_up'] == 1].index
    cross_downs = df[df['macd_cross_down'] == 1].index
    
    # Filter to ensure we have enough lookback for features (e.g. at least 50 bars)
    min_lookback = 50
    cross_ups = [i for i in cross_ups if i >= min_lookback]
    cross_downs = [i for i in cross_downs if i >= min_lookback]
    
    # DEBUG: Deep inspection
    if len(cross_ups) == 0 and len(cross_downs) == 0 and random.random() < 0.1: 
        print(f"\n{symbol} DEBUG:")
        print(f"  DF Range: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
        print(f"  Rows: {len(df)}")
        print(f"  Total Cross Up: {df['macd_cross_up'].sum()}")
        print(f"  Total Cross Down: {df['macd_cross_down'].sum()}")
    
    for i in cross_ups: crossovers.append({'index': i, 'type': 'BULLISH', 'price': df.loc[i, 'close']})
    for i in cross_downs: crossovers.append({'index': i, 'type': 'BEARISH', 'price': df.loc[i, 'close']})
    crossovers.sort(key=lambda x: x['index'])
    
    if not crossovers: return []
    
    records = []
    
    # Pre-extract model components
    entry_model = engine.entry_model
    sl_model = engine.sl_model
    if not entry_model or not sl_model: return []
    
    for cross in crossovers:
        i = cross['index']
        row = df.iloc[i]
        
        # Check Confidence
        X_entry = prepare_row_features(row, engine.entry_features, engine.entry_scaler)
        conf = float(entry_model.predict_proba(X_entry)[0, 1])
        
        if conf < 0.0: continue # Analyze ALL signals
        
        # Get Predicted SL
        X_sl = prepare_row_features(row, engine.sl_features, engine.sl_scaler)
        pred_sl_pct = max(0.005, min(float(sl_model.predict(X_sl)[0]), 0.15))
        
        # Simulate Future to find MAE (Max Adverse Excursion) and MFE (Max Favorable Excursion)
        entry_price = df.iloc[i+1]['open'] if i+1 < len(df) else row['close']
        
        mae_pct = 0.0
        mfe_pct = 0.0
        
        # Look forward fixed bars (e.g. 20 bars) or until an exit condition
        # To understand pullback, let's look at the "worst case" drawdown before hitting a "decent" profit (e.g. 2%)
        # Or just log max drawdown over next 20 bars.
        
        lookahead = 20
        future_slice = df.iloc[i+1 : min(i+1+lookahead, len(df))]
        
        if len(future_slice) == 0: continue
        
        if cross['type'] == 'BULLISH':
            # Adverse = lowest low vs entry
            min_price = future_slice['low'].min()
            mae_pct = (entry_price - min_price) / entry_price
            
            # Favorable = highest high vs entry
            max_price = future_slice['high'].max()
            mfe_pct = (max_price - entry_price) / entry_price
            
            did_hit_sl = min_price <= (entry_price * (1 - pred_sl_pct))
            
        else: # BEARISH
            # Adverse = highest high vs entry
            max_price = future_slice['high'].max()
            mae_pct = (max_price - entry_price) / entry_price
            
            # Favorable = lowest low vs entry
            min_price = future_slice['low'].min()
            mfe_pct = (entry_price - min_price) / entry_price
            
            did_hit_sl = max_price >= (entry_price * (1 + pred_sl_pct))
            
        records.append({
            'symbol': symbol,
            'type': cross['type'],
            'conf': conf,
            'pred_sl_pct': pred_sl_pct,
            'mae_pct': mae_pct, # Actual Pullback
            'mfe_pct': mfe_pct, # Max Potential Gain
            'sl_hit': did_hit_sl,
            'mae_ratio': mae_pct / pred_sl_pct # Ratio of Pullback to StopLoss ( > 1 means SL hit)
        })
        
    return records

if __name__ == "__main__":
    interval = '4h' # Use 4h as models exist there
    days = 120
    
    print(f"--- Analyzing Pullbacks vs SL Model ({interval}, {days} days) ---")
    
    engine = InferenceEngine(interval)
    
    ohlcv_dir = Path(__file__).parent.parent.parent / 'data' / 'ohlcv'
    files = list(ohlcv_dir.glob("*_USDT.parquet"))
    symbols = [f.stem.replace('_USDT', '') for f in files]
    symbols = [s for s in symbols if 'USDT' in s and 'DOWN' not in s and 'UP' not in s]
    
    # Use a sample of 100 symbols for speed
    random.seed(42)
    random.shuffle(symbols)
    test_symbols = symbols[:100]
    
    all_data = []
    
    for sym in tqdm(test_symbols):
        # Debug first symbol
        if len(all_data) == 0 and sym == test_symbols[0]:
             # Force print inside analyze_symbol by re-running or modifying?
             # Just enable debug prints via flag?
             pass
        stats = analyze_symbol(sym, interval, days, engine)
        if len(stats) == 0 and len(all_data) == 0:
             # Diagnose first failure
             pass # Already added print in function? No, commented out. 
        all_data.extend(stats)
        
    if not all_data:
        print("No signals found.")
        sys.exit()
        
    df_res = pd.DataFrame(all_data)
    
    # Summary Stats
    print("\n" + "="*50)
    print("PULLBACK ANALYSIS RESULTS")
    print("="*50)
    print(f"Total Signals Analyzed: {len(df_res)}")
    print(f"Avg Predicted SL: {df_res['pred_sl_pct'].mean()*100:.2f}%")
    print(f"Avg Actual MAE (Pullback): {df_res['mae_pct'].mean()*100:.2f}%")
    
    sl_hits = df_res[df_res['sl_hit']]
    safe_trades = df_res[~df_res['sl_hit']]
    
    print(f"\nSL Hit Rate: {len(sl_hits)/len(df_res)*100:.1f}%")
    
    # Of the winning trades (MFE > 2%), what was the typical pullback?
    # This answers "How much overlap do I need to survive to win?"
    winners = df_res[df_res['mfe_pct'] >= 0.02] # Potential winners > 2% gain
    print(f"\nPotential Winners (>2% MFE): {len(winners)}")
    if len(winners) > 0:
        winners_sl_hit = winners[winners['sl_hit']]
        print(f"Winners stopped out early: {len(winners_sl_hit)} ({len(winners_sl_hit)/len(winners)*100:.1f}%)")
        print(f"Avg Pullback on Winners: {winners['mae_pct'].mean()*100:.2f}%")
        print(f"Max Pullback on Winners (95th percentile): {winners['mae_pct'].quantile(0.95)*100:.2f}%")
        
        # Suggestion
        suggested_padding = winners['mae_pct'].quantile(0.90) / winners['pred_sl_pct'].median()
        print(f"\n>> Suggestion: To capture 90% of winners, multiply SL by {suggested_padding:.1f}x")
    else:
        print("No significant winners found in sample.")
