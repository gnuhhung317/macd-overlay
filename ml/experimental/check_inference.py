
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.inference import InferenceEngine
from data_processor import BinanceDataProcessor

def check_inference():
    symbol = 'BTCUSDT'
    interval = '4h'
    
    processor = BinanceDataProcessor(use_futures=True)
    df = processor.get_historical_data(symbol, interval, '30 days ago UTC', 'now UTC')
    df = processor.calculate_macd(df)
    
    engine = InferenceEngine(interval)
    
    print(f"Checking inference for {symbol} {interval} (last 5 bars)")
    
    for i in range(len(df)-5, len(df)):
        slice_df = df.iloc[:i+1]
        pred = engine.predict(symbol, slice_df)
        
        close = slice_df.iloc[-1]['close']
        print(f"Bar {slice_df.iloc[-1]['timestamp']}: Close={close}")
        print(f"  Conf: {pred['confidence']:.4f}")
        print(f"  SL: {pred['sl_pct']*100:.2f}%")
        print(f"  TP: {pred['tp_pct']*100:.2f}%")
        print("-" * 30)

if __name__ == "__main__":
    check_inference()
