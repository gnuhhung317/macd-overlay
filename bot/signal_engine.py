import pandas as pd
from typing import Dict, Optional, Any
from .config import BotConfig
from pathlib import Path

# Import RealtimePredictor from parent directory
import sys
sys.path.append(str(Path(__file__).parent.parent)) 

try:
    from ml.realtime_predictor import RealtimePredictor
except ImportError:
    # Fallback if specific import fails
    print("Warning: Could not import RealtimePredictor")
    RealtimePredictor = None

class SignalEngine:
    def __init__(self, config: BotConfig):
        self.config = config
        self.predictor = None
        self._init_predictor()

    def _init_predictor(self):
        """Initialize ML predictor"""
        if RealtimePredictor:
            # Threshold from config
            self.predictor = RealtimePredictor(entry_threshold=self.config.strategy.entry_threshold)
            print(f"[SignalEngine] ML Models Loaded: {self.predictor.is_loaded}")
        else:
            print("[SignalEngine] ML Predictor not available!")

    def analyze(self, symbol: str, df: pd.DataFrame, timeframe: str, funding_rate: float = 0.0) -> Dict[str, Any]:
        """
        Analyze dataframe (OHLCV + Indicators) to find signals.
        Returns a dict with signal details or None if no signal.
        """
        result = {
            "signal": "NEUTRAL",
            "confidence": 0.0,
            "sl": 0.0,
            "tp": 0.0,
            "action": "WAIT",
            "metadata": {}
        }
        
        if df.empty or len(df) < 50:
            return result

        # 1. Check Technical Signal (MACD Crossover) on the LAST closed candle
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        is_bullish = (prev_row['macd'] < prev_row['signal']) and (last_row['macd'] > last_row['signal'])
        is_bearish = (prev_row['macd'] > prev_row['signal']) and (last_row['macd'] < last_row['signal'])
        
        if not is_bullish and not is_bearish:
            return result

        # 2. ML Prediction
        ml_prediction = None
        if self.predictor and self.predictor.is_loaded:
            # We need to calculate features first used by ML model
            try:
                features_df = self.predictor.calculate_features(df, timeframe=timeframe, funding_rate=funding_rate)
                
                # Use the predict method which now returns the full dict structure from ThreeStageMLSystem
                ml_prediction = self.predictor.predict(features_df)
                
            except Exception as e:
                print(f"ML Prediction Error for {symbol}: {e}")

        # 3. Decision Logic
        if is_bullish:
            result['signal'] = "BULLISH"
            result['raw_signal'] = "MACD_CROSS_UP"
        elif is_bearish:
            result['signal'] = "BEARISH"
            result['raw_signal'] = "MACD_CROSS_DOWN"

        # Integrate ML
        if ml_prediction:
            result['confidence'] = ml_prediction['entry_confidence']
            result['sl'] = ml_prediction['sl_pct']
            result['tp'] = ml_prediction['tp_pct']
            result['risk_reward'] = ml_prediction.get('risk_reward', 0.0)
            result['metadata']['ml_raw'] = ml_prediction
            
            # Final Action Decision - Use ML's own 'should_enter' which checks threshold AND RR
            if ml_prediction['should_enter']:
                 # Extra check against config threshold just in case models were loaded with different default
                if result['confidence'] >= self.config.strategy.entry_threshold:
                     # Check Min RR from config (redundant if ML system does it, but safe)
                    if result['risk_reward'] >= self.config.strategy.min_rr_ratio:
                        result['action'] = "ENTRY"
                    else:
                        result['action'] = "FILTERED_POOR_RR"
                else:
                    result['action'] = "FILTERED_LOW_CONFIDENCE"
            else:
                 # ML system said NO (could be low confidence or poor RR)
                result['action'] = f"FILTERED_ML_{ml_prediction.get('filter_reason', 'UNKNOWN')}"
        else:
            # Fallback if ML fails
            result['action'] = "FILTERED_NO_ML"
            
        return result
