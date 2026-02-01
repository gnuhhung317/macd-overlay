
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import random # Import random
warnings.filterwarnings('ignore')

# Add project root to path
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
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days+20) 
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
    df = pd.DataFrame([row], columns=row.index)
    X = pd.DataFrame(0, index=[0], columns=feature_names)
    for col in feature_names:
        if col in df.columns: X[col] = df[col].values
    if scaler: X = pd.DataFrame(scaler.transform(X), columns=feature_names)
    return X

def run_backtest_for_symbol(symbol, interval='4h', days=120, engine=None):
    df = load_local_data(symbol, interval, days)
    if df is None or len(df) < 250: return None
    try: df = calculate_features(df)
    except: return None
    
    start_idx = 200
    if len(df) <= start_idx: return None

    crossovers = []
    # Vectorized detection for speed
    cross_ups = df[df['macd_cross_up'] == 1].index
    cross_downs = df[df['macd_cross_down'] == 1].index
    for i in cross_ups:
        if i >= start_idx: crossovers.append({'index': i, 'type': 'BULLISH', 'price': df.loc[i, 'close'], 'timestamp': df.loc[i, 'timestamp']})
    for i in cross_downs:
        if i >= start_idx: crossovers.append({'index': i, 'type': 'BEARISH', 'price': df.loc[i, 'close'], 'timestamp': df.loc[i, 'timestamp']})
    crossovers.sort(key=lambda x: x['index'])
            
    if not crossovers: return None
        
    results_market = []
    results_limit = []
    limit_fills = 0
    missed_trades = 0
    signals = 0
    
    if engine is None: engine = InferenceEngine(interval)
    entry_model = engine.entry_model
    try:
        entry_features = engine.entry_features
        entry_scaler = engine.entry_scaler
    except: return None

    for cross in crossovers:
        i = cross['index']
        row = df.iloc[i]
        
        if entry_model:
            X = prepare_row_features(row, entry_features, entry_scaler)
            confidence = float(entry_model.predict_proba(X)[0, 1])
        else: confidence = 0.5
            
        if confidence < 0.5: continue
        signals += 1
        
        if engine.sl_model:
            X_sl = prepare_row_features(row, engine.sl_features, engine.sl_scaler)
            sl_pct = max(0.005, min(float(engine.sl_model.predict(X_sl)[0]), 0.15))
        else: sl_pct = 0.02
            
        if engine.tp_model:
            X_tp = prepare_row_features(row, engine.tp_features, engine.tp_scaler)
            tp_pct = float(engine.tp_model.predict(X_tp)[0])
            if getattr(engine, 'tp_predict_rr', False): tp_pct = tp_pct * sl_pct
            tp_pct = max(0.01, min(tp_pct, 0.30))
        else: tp_pct = 0.04
        
        if i+1 >= len(df): continue
        next_bar = df.iloc[i+1]
        entry_price_market = next_bar['open']
        results_market.append(simulate_trade(df, i+1, cross['type'], entry_price_market, sl_pct, tp_pct))
        
        entry_adjust = 0
        if sl_pct > 0.05: entry_adjust = (sl_pct - 0.03) * 0.5
        entry_price_limit = cross['price'] 
        limit_price = entry_price_limit * (1 - entry_adjust) if cross['type'] == 'BULLISH' else entry_price_limit * (1 + entry_adjust)
            
        fill_idx = -1
        wait_bars = 24 
        for k in range(1, wait_bars + 1):
            if i + k >= len(df): break
            bar = df.iloc[i+k]
            if cross['type'] == 'BULLISH':
                if bar['low'] <= limit_price: fill_idx = i + k; break
            else:
                if bar['high'] >= limit_price: fill_idx = i + k; break
        
        if fill_idx != -1:
            limit_fills += 1
            results_limit.append(simulate_trade(df, fill_idx, cross['type'], limit_price, sl_pct, tp_pct))
        else:
            missed_trades += 1
            results_limit.append(0)
            
    return {
        'symbol': symbol, 'signals': signals,
        'market_pnl_R': sum(results_market), 'limit_pnl_R': sum(results_limit),
        'limit_fills': limit_fills, 'missed_trades': missed_trades
    }

def simulate_trade(df, start_idx, type, entry_price, sl_pct, tp_pct):
    if type == 'BULLISH':
        sl_price = entry_price * (1 - sl_pct)
        tp_price = entry_price * (1 + tp_pct)
    else:
        sl_price = entry_price * (1 + sl_pct)
        tp_price = entry_price * (1 - tp_pct)
    for i in range(start_idx, min(start_idx + 100, len(df))):
        row = df.iloc[i]
        if type == 'BULLISH':
            if row['low'] <= sl_price: return -1.0
            if row['high'] >= tp_price: return tp_pct / sl_pct
        else:
            if row['high'] >= sl_price: return -1.0
            if row['low'] <= tp_price: return tp_pct / sl_pct
    return 0 

if __name__ == "__main__":
    interval = '1d'
    days = 120
    
    ohlcv_dir = Path(__file__).parent.parent.parent / 'data' / 'ohlcv'
    files = list(ohlcv_dir.glob("*_USDT.parquet"))
    symbols = [f.stem.replace('_USDT', '') for f in files]
    
    excludes = ['USDC', 'BUSD', 'TUSD', 'UST', 'DAI']
    symbols = [s for s in symbols if not any(x in s for x in excludes)]
    
    # Randomize to avoid specific bad-sector sticking
    random.shuffle(symbols)
    
    print(f"--- Detail Backtest on {len(symbols)} coins ({interval}, {days} days) ---")
    
    engine = InferenceEngine(interval)
    
    output_file = Path("backtest_results.csv")
    if not output_file.exists():
        pd.DataFrame(columns=['symbol', 'signals', 'market_pnl_R', 'limit_pnl_R', 'limit_fills', 'missed_trades']).to_csv(output_file, index=False)
    
    for symbol in symbols:
        print(f"Processing {symbol}...", end='\r')
        try:
            res = run_backtest_for_symbol(symbol, interval, days, engine)
            if res:
                df_res = pd.DataFrame([res])
                df_res.to_csv(output_file, mode='a', header=False, index=False)
        except Exception as e:
            pass
            
    # Final Analysis
    print("\n\nAnalysis:")
    if output_file.exists():
        df = pd.read_csv(output_file)
        if len(df) > 0:
            agg_signals = df['signals'].sum()
            agg_market = df['market_pnl_R'].sum()
            agg_limit = df['limit_pnl_R'].sum()
            agg_fills = df['limit_fills'].sum()
            agg_missed = df['missed_trades'].sum()
            
            print(f"Total Signals: {agg_signals}")
            print(f"Coins Analyzed: {len(df)}")
            print("\n--- Strategy A: Market Entry ---")
            print(f"Total PnL (R): {agg_market:.2f}R")
            print(f"Avg PnL: {agg_market/agg_signals:.2f}R" if agg_signals else "0R")
            
            print("\n--- Strategy B: Limit Entry ---")
            print(f"Filled: {agg_fills} ({agg_fills/agg_signals:.1%})" if agg_signals else "0%")
            print(f"Total PnL (R): {agg_limit:.2f}R")
            print(f"Avg PnL: {agg_limit/agg_signals:.2f}R" if agg_signals else "0R")
            
            diff = agg_limit - agg_market
            print(f"\nDiff (Limit - Market): {diff:+.2f}R")
        else:
            print("No data collected.")
    else:
        print("No output file.")
