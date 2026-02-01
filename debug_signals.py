import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from ml.signal_dashboard import fetch_symbol_signal, get_top_symbols

def debug_signals():
    symbols = get_top_symbols(limit=10)
    timeframe = '4h'
    lookback_days = 7
    
    print(f"Debugging {timeframe} signals for {symbols} with {lookback_days}d lookback...")
    
    for symbol in symbols:
        print(f"\n--- Analyzing {symbol} ---")
        try:
            # Re-implement fetch logic with prints
            from data_processor import BinanceDataProcessor
            from ml.inference import InferenceEngine
            from ml.data_pipeline import calculate_features
            
            processor = BinanceDataProcessor(use_futures=True)
            engine = InferenceEngine(timeframe)
            
            fetch_start = "400 days ago UTC"
            df = processor.get_historical_data(symbol, timeframe, fetch_start, 'now UTC')
            print(f"  Fetched {len(df)} rows.")
            
            # Simulate fix
            if not df.empty:
                df = df.iloc[:-1].copy()
                print("  Dropped last (forming) candle.")
            
            if len(df) < 200:
                print("  Insufficient data ( < 200).")
                continue
                
            df = calculate_features(df)
            
            cutoff_date = pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)
            if df['timestamp'].dt.tz is None:
                 cutoff_date = cutoff_date.tz_localize(None)
            
            recent = df[df['timestamp'] >= cutoff_date]
            print(f"  Recent window: {len(recent)} rows since {cutoff_date}")
            
            cross_up = recent[recent['macd_cross_up'] == 1]
            cross_down = recent[recent['macd_cross_down'] == 1]
            print(f"  Crossovers found: Up={len(cross_up)}, Down={len(cross_down)}")
            
            if not cross_up.empty or not cross_down.empty:
                is_up = not cross_up.empty
                if not cross_up.empty and not cross_down.empty:
                    is_up = cross_up.index[-1] > cross_down.index[-1]
                
                row = cross_up.iloc[-1] if is_up else cross_down.iloc[-1]
                print(f"  Latest crossover: {'BULLISH' if is_up else 'BEARISH'} at {row['timestamp']} (Price: {row['close']})")
                
                # Check prediction
                print(f"  Running prediction for {symbol} at {row['timestamp']}...")
                prediction = engine.predict(symbol, df.loc[:row.name])
                
                if prediction:
                    if prediction.get('error'):
                        print(f"  ❌ Prediction error: {prediction.get('error')}")
                    else:
                        print(f"  ✅ Prediction success! Confidence: {prediction.get('confidence', prediction.get('entry_confidence', 'N/A'))}")
                else:
                    print("  ❌ Prediction returned None")
            else:
                print("  No crossovers in recent window.")
                
        except Exception as e:
            print(f"  💥 Exception: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_signals()
