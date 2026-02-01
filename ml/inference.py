import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Union, Tuple

# Add parent directory to path if trying to run directly
import sys
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

# Define paths locally
DATA_DIR = Path(__file__).parent.parent / 'data'
MODELS_DIR = Path(__file__).parent / 'models'

try:
    from ml.data_pipeline import calculate_features, load_funding
    from ml.multi_timeframe_pipeline import calculate_features_for_timeframe
except ImportError:
    try:
        from data_pipeline import calculate_features, load_funding
        from multi_timeframe_pipeline import calculate_features_for_timeframe
    except ImportError:
        # Fallback if running from within ml directory
        import sys
        sys.path.append(str(Path(__file__).parent))
        from data_pipeline import calculate_features, load_funding
        from multi_timeframe_pipeline import calculate_features_for_timeframe

class InferenceEngine:
    def __init__(self, timeframe: str):
        self.timeframe = timeframe
        self.model_dir = MODELS_DIR / timeframe
        self.entry_model = None
        self.sl_model = None
        self.tp_model = None
        self.encoders = {}
        
        self._load_models()

    def _load_models(self):
        """Load all models for the timeframe."""
        if not self.model_dir.exists():
            raise ValueError(f"No models found for timeframe {self.timeframe}")
            
        # Entry
        entry_path = self.model_dir / 'entry_filter.joblib'
        if entry_path.exists():
            data = joblib.load(entry_path)
            self.entry_model = data['model']
            self.entry_scaler = data.get('scaler')
            self.entry_features = data['feature_names']
        
        # SL
        sl_path = self.model_dir / 'sl_predictor.joblib'
        if sl_path.exists():
            data = joblib.load(sl_path)
            self.sl_model = data['model']
            self.sl_scaler = data.get('scaler')
            self.sl_features = data['feature_names']
            
        # TP
        tp_path = self.model_dir / 'tp_predictor.joblib'
        if tp_path.exists():
            data = joblib.load(tp_path)
            self.tp_model = data['model']
            self.tp_scaler = data.get('scaler')
            self.tp_features = data['feature_names']
            self.tp_predict_rr = data.get('predict_rr', False)

    def _prepare_single_row(self, row: pd.Series, features: list, scaler) -> pd.DataFrame:
        """Prepare a single row as a DataFrame for prediction."""
        # Create a dict with all expected features initialized to 0.0
        row_dict = {feat: 0.0 for feat in features}
        
        # Fill available values from the input row
        for col in features:
            if col in row.index:
                val = row[col]
                # Ensure value is float and finite
                if pd.isna(val) or np.isinf(val):
                    row_dict[col] = 0.0
                else:
                    row_dict[col] = float(val)
        
        # Create single-row DataFrame
        df = pd.DataFrame([row_dict], columns=features)
        
        # Handle scaling - ensure we return a DataFrame with feature names
        if scaler:
            try:
                # transform() returns a numpy array, we MUST wrap it back in a DataFrame
                scaled_values = scaler.transform(df)
                df = pd.DataFrame(scaled_values, columns=features)
            except Exception as e:
                print(f"⚠️  Scaling failed: {e}. Using unscaled features.")
            
        return df

    def predict(self, symbol: str, recent_data: pd.DataFrame) -> Dict:
        """
        Generate prediction for a symbol given its recent OHLCV data.
        
        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT)
            recent_data: DataFrame with OHLCV data. Must have enough rows to calculate features.
            
        Returns:
            Dictionary containing entry decision, confidence, SL, TP
        """
        if len(recent_data) < 100:
             return {"error": f"Insufficient data. Need at least 100 bars, got {len(recent_data)}"}

        # Calculate features (this adds MACD, RSI, etc.)
        # make a copy to avoid warning
        df = recent_data.copy()
        
        # We need to calculate features. 
        # CAUTION: calculate_features expects 'timestamp' column or index? 
        # data_pipeline.calculate_features sets index to timestamp usually.
        # Let's inspect data_pipeline usage. Assuming it takes standard OHLCV.
        
        try:
            # Replicate the exact training feature pipeline
            df = calculate_features_for_timeframe(df, self.timeframe)
            
            # Merge funding rate (critical for some models)
            df_funding = load_funding(symbol)
            if not df_funding.empty:
                df_funding = df_funding[['timestamp', 'funding_rate']].copy()
                df_funding['date'] = df_funding['timestamp'].dt.date
                funding_daily = df_funding.groupby('date')['funding_rate'].mean().reset_index()
                df['date'] = df['timestamp'].dt.date
                df = df.merge(funding_daily, on='date', how='left')
                df['funding_rate'] = df['funding_rate'].fillna(0)
                df = df.drop(columns=['date'])
            else:
                df['funding_rate'] = 0
                
        except Exception as e:
             return {"error": f"Feature calculation failed: {str(e)}"}
             
        # Get the very last row (most recent closed candle)
        # Note: In real-time, we might want the barely closing candle. 
        # Assuming the data passed includes the latest closed candle at the end.
        latest_row = df.iloc[-1]
        
        result = {
            "symbol": symbol,
            "timestamp": latest_row.get('timestamp', datetime.now()),
            "close_price": latest_row['close'],
            "timeframe": self.timeframe,
            "should_enter": False,
            "entry_confidence": 0.0,
            "sl_pct": 0.0,
            "tp_pct": 0.0,
            "sl_price": 0.0,
            "tp_price": 0.0,
            "limit_price": 0.0, # Recommended entry price
            "action": "WAIT"
        }
        
        # Check Signal (MACD Crossover)
        # The training logic relies on macd_cross_up/down
        is_long = latest_row.get('macd_cross_up', 0) == 1
        is_short = latest_row.get('macd_cross_down', 0) == 1
        
        if not is_long and not is_short:
             result["reason"] = "No MACD Crossover Signal"
             return result

        direction = "LONG" if is_long else "SHORT"
        result["direction"] = direction
        
        # Add is_bullish_cross (critical categorical feature for model)
        # We need to make a copy of the row or add it to a dict
        row_with_target = latest_row.to_dict()
        row_with_target['is_bullish_cross'] = 1 if is_long else 0
        latest_row_plus = pd.Series(row_with_target)
        
        # Stage 1: Entry Filter
        if self.entry_model:
            X = self._prepare_single_row(latest_row_plus, self.entry_features, self.entry_scaler)
            # Use .values to bypass XGBoost 2.0+ "feature names" check which fails in Docker
            prob = self.entry_model.predict_proba(X.values)[0, 1]
            result["entry_confidence"] = float(prob)
            
            # Threshold check (could make configurable)
            if prob >= 0.65: # Using default 0.65
                result["should_enter"] = True
                result["action"] = f"ENTER {direction}"
            else:
                 result["should_enter"] = False
                 result["action"] = f"SKIP {direction} (Low Confidence)"
        else:
             # If no entry model, assume valid if signal exists? Or fail?
             # Let's be conservative
             result["should_enter"] = True
             result["entry_confidence"] = 0.5
             result["action"] = f"ENTER {direction} (No Filter)"

        # Stage 2 & 3: SL/TP (Always calculate if we have a signal direction)
        # Previously this was: if result["should_enter"]:
        if direction:
            # SL Model Prediction
            if self.sl_model:
                 X_sl = self._prepare_single_row(latest_row_plus, self.sl_features, self.sl_scaler)
                 sl_pct = float(self.sl_model.predict(X_sl.values)[0])
                 sl_pct = max(0.005, min(sl_pct, 0.15)) # Clip
                 result["sl_pct"] = sl_pct
            else:
                 result["sl_pct"] = 0.02 # Default
            
            # TP Model Prediction
            if self.tp_model:
                 X_tp = self._prepare_single_row(latest_row_plus, self.tp_features, self.tp_scaler)
                 tp_pred = float(self.tp_model.predict(X_tp.values)[0])
                 
                 if self.tp_predict_rr:
                     tp_pct = tp_pred * result["sl_pct"]
                 else:
                     tp_pct = tp_pred
                 
                 tp_pct = max(0.01, min(tp_pct, 0.30)) # Clip
                 result["tp_pct"] = tp_pct
            else:
                 result["tp_pct"] = 0.04

            # --- Optimization: Apply SL Multiplier (Safety Factor) ---
            # Based on pullback analysis, we need more breathing room
            sl_multiplier = 1.5 
            result["sl_pct"] = result["sl_pct"] * sl_multiplier
            
            # --- Limit Offset Logic (Strategy Optimization) ---
            # If SL is wide, we suggest a lower entry (for Long) or higher (for Short)
            # to improve R/R and survival.
            entry_adjust = 0.0
            if result["sl_pct"] > 0.05:
                # If SL is 10%, adjust is (0.1 - 0.03) * 0.5 = 0.035 (3.5%)
                entry_adjust = (result["sl_pct"] - 0.03) * 0.5
                entry_adjust = min(entry_adjust, 0.05) # Cap at 5% adjustment
            
            # Calculate Prices
            price = result["close_price"]
            if direction == "LONG":
                result["limit_price"] = price * (1 - entry_adjust)
                result["sl_price"] = result["limit_price"] * (1 - result["sl_pct"])
                result["tp_price"] = result["limit_price"] * (1 + result["tp_pct"])
            else:
                result["limit_price"] = price * (1 + entry_adjust)
                result["sl_price"] = result["limit_price"] * (1 + result["sl_pct"])
                result["tp_price"] = result["limit_price"] * (1 - result["tp_pct"])
            
            # If no adjustment, limit_price is just current price
            if entry_adjust == 0:
                result["limit_price"] = price
                
        return result

def load_data_for_symbol(symbol: str, timeframe: str) -> pd.DataFrame:
    """
    Load recent data for a symbol. 
    In production, this would fetch from CCXT.
    For now, we load from disk (processed parquet or raw csv).
    """
    # Try loading from processed data first if available (faster)
    # But processed data is usually one big file.
    # Let's check raw directory structure
    csv_path = DATA_DIR / 'raw' / timeframe / f"{symbol.replace('/','')}.csv"
    if csv_path.exists():
        # Load last N rows
        # Pandas read_csv can be slow for huge files, but we mainly need tail.
        # For simplicity, load all and tail.
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
        
    return pd.DataFrame()
