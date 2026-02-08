import pandas as pd
import numpy as np
from ml.realtime_predictor import RealtimePredictor

def verify_features():
    # Create dummy data
    data = {
        'open': np.linspace(100, 110, 100),
        'high': np.linspace(101, 111, 100),
        'low': np.linspace(99, 109, 100),
        'close': np.linspace(100.5, 110.5, 100),
        'volume': np.random.uniform(1000, 2000, 100)
    }
    # Add a spike at the end
    data['volume'][-1] = 5000
    
    df = pd.DataFrame(data)
    
    predictor = RealtimePredictor()
    features = predictor.calculate_features(df, timeframe='4h', funding_rate=0.0001)
    
    if features.empty:
        print("❌ Feature calculation returned empty DataFrame")
        return
        
    last_row = features.iloc[-1]
    
    required_features = ['volume_spike', 'rsi_slope', 'adx']
    for feat in required_features:
        val = last_row.get(feat)
        if val is None:
            print(f"❌ Feature '{feat}' is MISSING")
        elif val == 0 and feat == 'volume_spike':
             print(f"❌ Feature '{feat}' is 0.0 (possibly still broken if volume was spiked)")
        else:
            print(f"✅ Feature '{feat}': {val:.4f}")

if __name__ == "__main__":
    verify_features()
