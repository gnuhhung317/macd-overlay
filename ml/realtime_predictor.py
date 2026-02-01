#!/usr/bin/env python3
"""
Realtime ML Predictor - Calculates features from OHLCV data and predicts
Used by api_server.py and telegram_notifier.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional

# Import ML components
try:
    from ml.three_stage_ml import ThreeStageMLSystem
    from ml.data_pipeline import calculate_macd
except ImportError:
    # When running from ml directory
    from three_stage_ml import ThreeStageMLSystem
    from data_pipeline import calculate_macd

MODEL_DIR = Path(__file__).parent / 'models'


class RealtimePredictor:
    """
    Realtime ML Predictor for MACD Crossover signals.
    
    Usage:
        predictor = RealtimePredictor()
        if predictor.is_loaded:
            features = predictor.calculate_features(df)
            prediction = predictor.predict(features)
    """
    
    def __init__(self, entry_threshold: float = 0.5):
        self.ml_system = None
        self.is_loaded = False
        self.entry_threshold = entry_threshold
        
        # Try to load models
        self._load_models()
    
    def _load_models(self):
        """Load ML models if available."""
        entry_path = MODEL_DIR / 'entry_filter.joblib'
        sl_path = MODEL_DIR / 'sl_predictor.joblib'
        tp_path = MODEL_DIR / 'tp_predictor.joblib'
        
        # Check if at least entry model exists
        if not entry_path.exists():
            print("[ML] Entry filter model not found. ML predictions disabled.")
            return
        
        try:
            self.ml_system = ThreeStageMLSystem(
                entry_model_path=str(entry_path) if entry_path.exists() else None,
                sl_model_path=str(sl_path) if sl_path.exists() else None,
                tp_model_path=str(tp_path) if tp_path.exists() else None,
                entry_threshold=self.entry_threshold
            )
            self.is_loaded = True
            print("[ML] Models loaded successfully!")
        except Exception as e:
            print(f"[ML] Failed to load models: {e}")
            self.is_loaded = False
    
    def calculate_features(self, df: pd.DataFrame, timeframe: str = '1h', funding_rate: float = 0.0) -> pd.DataFrame:
        """
        Calculate features from OHLCV DataFrame.
        Must match exactly the logic in ml/multi_timeframe_pipeline.py
        
        Args:
            df: OHLCV DataFrame
            timeframe: Current timeframe string (e.g., '15m', '4h')
            funding_rate: Current funding rate
            
        Returns:
            DataFrame with features for ML prediction.
        """
        # Minimum 50 rows for basic features
        if df.empty or len(df) < 50:
            print(f"[ML] Not enough data: {len(df)} rows (need 50+)")
            return pd.DataFrame()
        
        df = df.copy()
        
        # ===== Price Features =====
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['body_size'] = abs(df['close'] - df['open']) / df['close']
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
        
        # ===== Trend Features =====
        for period in [7, 14, 21, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}']
        
        # Trend strength
        df['trend_7_21'] = df['sma_7'] / df['sma_21']
        df['trend_21_50'] = df['sma_21'] / df['sma_50']
        df['trend_50_200'] = df['sma_50'] / df['sma_200']
        
        # ===== MACD Features =====
        df = calculate_macd(df)
        df['macd_slope'] = df['macd'].diff()
        df['signal_slope'] = df['signal'].diff()
        df['histogram_slope'] = df['histogram'].diff()
        df['macd_acceleration'] = df['macd_slope'].diff()
        
        # ===== Volatility Features =====
        df['atr_14'] = self._calculate_atr(df, 14)
        df['atr_7'] = self._calculate_atr(df, 7)
        df['volatility_7'] = df['returns'].rolling(7).std()
        df['volatility_14'] = df['returns'].rolling(14).std()
        df['volatility_21'] = df['returns'].rolling(21).std()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ===== Momentum Features =====
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        df['rsi_7'] = self._calculate_rsi(df['close'], 7)
        
        # Stochastic
        df['stoch_k'] = self._calculate_stochastic(df, 14)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # Rate of Change
        df['roc_7'] = df['close'].pct_change(7)
        df['roc_14'] = df['close'].pct_change(14)
        df['roc_21'] = df['close'].pct_change(21)
        
        # ===== Volume Features =====
        df['volume_sma_7'] = df['volume'].rolling(7).mean()
        df['volume_sma_14'] = df['volume'].rolling(14).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_14']
        df['volume_trend'] = df['volume_sma_7'] / df['volume_sma_14']
        
        # OBV
        df['obv'] = (np.sign(df['returns']) * df['volume']).cumsum()
        df['obv_sma'] = df['obv'].rolling(14).mean()
        df['obv_trend'] = df['obv'] / df['obv_sma']
        
        # ===== Market Regime =====
        df['is_trending'] = (abs(df['trend_7_21'] - 1) > 0.02).astype(int)
        df['is_volatile'] = (df['volatility_14'] > df['volatility_14'].rolling(50).mean()).astype(int)
        
        # ===== Timeframe & Funding Features (New for Parity) =====
        # Timeframe mapping (hours)
        # Note: We need to handle non-standard timeframes like 15m, 30m by estimating
        tf_map = {
            '1m': 1/60, '3m': 3/60, '5m': 5/60, '15m': 0.25, '30m': 0.5,
            '1h': 1, '2h': 2, '4h': 4, '6h': 6, '8h': 8, '12h': 12, '1d': 24
        }
        df['timeframe_hours'] = tf_map.get(timeframe, 4) # Default to 4h if unknown
        
        # Volatility Scaling (matches multi_timeframe_pipeline.py)
        # 1d is the baseline? Pipeline scales 1h/4h UP to match 1d volatility scale approx?
        # Pipeline logic:
        # if timeframe in ['1h', '4h']:
        #     scale = 24 / tf_map[timeframe]
        #     df['volatility_7_scaled'] = df['volatility_7'] * np.sqrt(scale)
        # else: ...
        
        # We apply the same logic generally
        hours = df['timeframe_hours'].iloc[-1]
        scale = 24 / max(hours, 0.01) # Avoid div by zero
        
        # Only scale if timeframe < 24h? Pipeline only did it for 1h/4h explicitly but we generalize
        if hours < 24:
            df['volatility_7_scaled'] = df['volatility_7'] * np.sqrt(scale)
            df['volatility_14_scaled'] = df['volatility_14'] * np.sqrt(scale)
        else:
            df['volatility_7_scaled'] = df['volatility_7']
            df['volatility_14_scaled'] = df['volatility_14']
            
        # Funding Rate
        df['funding_rate'] = funding_rate
        
        # MACD Crossover Flags (Explicitly needed for is_bullish_cross)
        # calculate_macd gives macd, signal, histogram
        df['macd_cross_up'] = ((df['macd'] > df['signal']) & (df['macd'].shift(1) <= df['signal'].shift(1))).astype(int)
        df['macd_cross_down'] = ((df['macd'] < df['signal']) & (df['macd'].shift(1) >= df['signal'].shift(1))).astype(int)
        
        # Is Bullish Cross (Target Feature)
        # The model likely uses 'is_bullish_cross' to distinguish Long/Short context
        df['is_bullish_cross'] = df['macd_cross_up']
        
        # Bars since cross (Also in training pipeline)
        # Calculating this iteratively is slow, but for realtime (small df) it's fine
        # Or we can just default to 0 for the immediate signal?
        # Pipeline does full calculation. Let's do a simplified version for the latest rows or just fill 0
        # The model uses 'bars_since_cross_up/down'.
        # Vectorized approach:
        cross_up_mask = df['macd_cross_up'] == 1
        cross_down_mask = df['macd_cross_down'] == 1
        
        # We only strictly need it for the last row if that's what we predict on.
        # But let's keep it simple. If we are AT a cross, it is 0.
        # If we are 1 bar after, it is 1.
        # For now, initializing to 0 is "safe-ish" but might affect tree splits.
        # Let's try to do it right for at least the recent window.
        df['bars_since_cross_up'] = 0
        df['bars_since_cross_down'] = 0
        
        # Fast way for last row: look back
        # (Optimization: We assume the caller passes enough history, e.g. 100+ bars)
        
        last_idx = df.index[-1]
        
        # Find last True in mask
        last_up = df.index[df['macd_cross_up'] == 1].max()
        last_down = df.index[df['macd_cross_down'] == 1].max()
        
        if pd.notna(last_up):
             # Assuming purely integer index or we rely on position
             # Since df is likely RangeIndex or checks above, let's use integer position
             # We can't easily do date math without more checks.
             # Reset index to be safe
             df_temp = df.reset_index(drop=True)
             last_up_idx = df_temp.index[df_temp['macd_cross_up'] == 1].max()
             current_idx = df_temp.index[-1]
             if pd.notna(last_up_idx):
                 df.iloc[-1, df.columns.get_loc('bars_since_cross_up')] = current_idx - last_up_idx
        else:
             df.iloc[-1, df.columns.get_loc('bars_since_cross_up')] = 999
             
        if pd.notna(last_down):
             df_temp = df.reset_index(drop=True)
             last_down_idx = df_temp.index[df_temp['macd_cross_down'] == 1].max()
             current_idx = df_temp.index[-1]
             if pd.notna(last_down_idx):
                 df.iloc[-1, df.columns.get_loc('bars_since_cross_down')] = current_idx - last_down_idx
        else:
             df.iloc[-1, df.columns.get_loc('bars_since_cross_down')] = 999
        
        # Clean up
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_stochastic(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Stochastic %K"""
        low_min = df['low'].rolling(period).min()
        high_max = df['high'].rolling(period).max()
        return 100 * (df['close'] - low_min) / (high_max - low_min)
    
    def predict(self, features_df: pd.DataFrame) -> Optional[Dict]:
        """
        Get ML prediction for the last row of features.
        
        Returns:
            {
                'should_enter': bool,
                'entry_confidence': float,
                'sl_pct': float,
                'tp_pct': float,
                'risk_reward': float,
            }
            Or None if ML system not loaded.
        """
        if not self.is_loaded or self.ml_system is None:
            return None
        
        if features_df.empty:
            return None
        
        # Get last row as DataFrame
        last_features = features_df.iloc[[-1]]
        
        return self.ml_system.predict(last_features)
    
    def predict_from_ohlcv(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Convenience method: Calculate features and predict in one call.
        
        Args:
            df: OHLCV DataFrame with at least 200 rows
            
        Returns:
            ML prediction dict or None
        """
        if not self.is_loaded:
            return None
        
        features = self.calculate_features(df)
        if features.empty:
            return None
        
        return self.predict(features)


# Singleton instance for use across modules
_predictor_instance = None

def get_predictor(entry_threshold: float = 0.5) -> RealtimePredictor:
    """Get or create singleton predictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = RealtimePredictor(entry_threshold)
    return _predictor_instance


if __name__ == '__main__':
    # Test
    print("Testing RealtimePredictor...")
    predictor = RealtimePredictor()
    print(f"Models loaded: {predictor.is_loaded}")
