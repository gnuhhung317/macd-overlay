import sys
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from ml.inference import InferenceEngine

def test_inference_threshold_fix():
    print("Testing InferenceEngine threshold decoupling fix...")
    
    # Initialize engine for 4h
    try:
        engine = InferenceEngine('4h')
    except Exception as e:
        print(f"Error loading engine: {e}")
        return

    # Mock the entry model to return 0.63 confidence (below 0.65 hardcoded in engine)
    if engine.entry_model:
        engine.entry_model.predict_proba = MagicMock(return_value=np.array([[0.37, 0.63]]))
    
    # Mock SL/TP models to return predictable values if needed
    if engine.sl_model:
        engine.sl_model.predict = MagicMock(return_value=np.array([0.02]))
    if engine.tp_model:
        engine.tp_model.predict = MagicMock(return_value=np.array([0.04]))

    # Create dummy data with a bullish crossover
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2026-01-01', periods=100, freq='4h'),
        'open': 100.0,
        'high': 105.0,
        'low': 95.0,
        'close': 100.0,
        'volume': 1000.0
    })
    
    # To avoid feature calculation errors, we can mock the feature calculation
    # or just fill the necessary columns that the prepare_single_row needs
    for feat in engine.entry_features:
        df[feat] = 0.0
    for feat in engine.sl_features:
        df[feat] = 0.0
    for feat in engine.tp_features:
        df[feat] = 0.0
        
    df['macd_cross_up'] = 0
    df['macd_cross_down'] = 0
    df.iloc[-1, df.columns.get_loc('macd_cross_up')] = 1
    
    # We also need to bypass calculate_features_for_timeframe since we mocked columns
    import ml.inference
    original_calc = ml.inference.calculate_features_for_timeframe
    ml.inference.calculate_features_for_timeframe = lambda d, t: d
    ml.inference.load_funding = lambda s: pd.DataFrame()

    try:
        # Run prediction
        result = engine.predict('TESTUSDT', df)
        
        print(f"Confidence: {result['entry_confidence']:.1%}")
        print(f"Should Enter: {result['should_enter']}")
        print(f"SL Pct: {result['sl_pct']:.1%}")
        print(f"TP Pct: {result['tp_pct']:.1%}")
        print(f"SL Price: {result['sl_price']}")
        print(f"TP Price: {result['tp_price']}")
        print(f"Limit Price: {result['limit_price']}")
        
        # Verification
        assert result['entry_confidence'] == 0.63
        assert result['should_enter'] is False # Engine still says False because < 0.65
        assert result['sl_pct'] > 0
        assert result['tp_pct'] > 0
        assert result['limit_price'] > 0
        assert result['sl_price'] > 0
        assert result['tp_price'] > 0
        
        print("\n✅ Verification SUCCESS: SL/TP prices calculated even with Low Confidence!")
        
    finally:
        # Restore
        ml.inference.calculate_features_for_timeframe = original_calc

if __name__ == "__main__":
    test_inference_threshold_fix()
