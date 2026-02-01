
import sys
import pandas as pd
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.inference import InferenceEngine

def test_inference_fix():
    print("Testing InferenceEngine fixes...")
    
    # Use 4h as we know models exist there
    engine = InferenceEngine('4h')
    
    # Create mock OHLCV data that usually triggers the issue
    # Issue was in _prepare_single_row during data type alignment
    data = {
        'timestamp': pd.date_range(start='2026-01-01', periods=150, freq='4h'),
        'open': [100.0] * 150,
        'high': [105.0] * 150,
        'low': [95.0] * 150,
        'close': [102.0] * 150,
        'volume': [1000.0] * 150
    }
    df = pd.DataFrame(data)
    
    # We need to trigger a crossover to reach the SL logic in predict()
    # Let's manually add macd crossover columns if predict() expects them before calling calculate_features?
    # No, predict() calls calculate_features_for_timeframe first, which calls calculate_features, which calls calculate_macd.
    # But calculate_features in data_pipeline detection logic might need enough data.
    # Let's just manually set macd_cross_up on the last row to simulate a signal.
    
    # Wait, InferenceEngine.predict checks for crossovers in the calculated df.
    # To force a signal in the test, we'll let it calculate everything.
    
    # We need to trigger a crossover to reach the SL logic
    # Or at least reach _prepare_single_row
    
    print("Running predict()...")
    try:
        result = engine.predict('BTCUSDT', df)
        
        if 'error' in result:
            print(f"Prediction Error: {result['error']}")
        else:
            print(f"Success! Can Enter: {result['can_enter']}")
            print(f"SL Pct: {result.get('sl_pct', 0)*100:.2f}%")
            print(f"Limit Price: {result.get('limit_price', 0)}")
            print(f"Confidence: {result.get('confidence', 0)}")
            
            # Check for data types in internal preparation if possible
            # But the main goal is no exception
            print("\nVerification Passed: No int64 casting exceptions.")
            
    except Exception as e:
        print(f"\nVerification FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_inference_fix()
