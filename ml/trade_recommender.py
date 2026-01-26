#!/usr/bin/env python3
"""
ML Trade Recommender

Provides intelligent trade recommendations based on MACD crossover signals:
- Entry price adjustment (better entry if SL is far)
- Optimal SL based on volatility
- Optimal TP based on momentum
- Expected bars to peak

Supports multiple timeframes: 1h, 4h, 8h, 12h, 1d
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass
import joblib

MODEL_DIR = Path(__file__).parent / 'models'


@dataclass
class TradeRecommendation:
    """Trade recommendation from ML system."""
    # Signal info
    symbol: str
    timeframe: str
    signal_type: str  # 'LONG' or 'SHORT'
    signal_time: str
    
    # Prices
    current_price: float
    recommended_entry: float  # May be different from current if SL is far
    stop_loss: float
    take_profit: float
    
    # Percentages
    sl_pct: float
    tp_pct: float
    entry_offset_pct: float  # How much entry is adjusted from current price
    
    # Risk/Reward
    risk_reward: float
    
    # ML Confidence
    entry_confidence: float
    should_enter: bool
    
    # Timing
    expected_bars_to_peak: int
    
    # Recommendation text
    recommendation: str
    notes: List[str]


class MLTradeRecommender:
    """
    ML-based trade recommender that combines all 4 stages:
    1. Entry Filter
    2. SL Predictor
    3. TP Predictor
    4. Bars to Peak Predictor
    
    Plus smart entry adjustment based on SL distance.
    """
    
    def __init__(
        self,
        entry_model_path: str = None,
        sl_model_path: str = None,
        tp_model_path: str = None,
        bars_model_path: str = None,
        entry_threshold: float = 0.5,
        min_rr_ratio: float = 1.5,
        sl_adjustment_threshold: float = 0.03  # If SL > 3%, consider entry adjustment
    ):
        # Models
        self.entry_model = None
        self.entry_scaler = None
        self.entry_features = None
        
        self.sl_model = None
        self.sl_scaler = None
        self.sl_features = None
        
        self.tp_model = None
        self.tp_scaler = None
        self.tp_features = None
        
        self.bars_model = None
        self.bars_scaler = None
        self.bars_features = None
        
        # Settings
        self.entry_threshold = entry_threshold
        self.min_rr_ratio = min_rr_ratio
        self.sl_adjustment_threshold = sl_adjustment_threshold
        
        # Default values
        self.default_sl = 0.02
        self.default_tp = 0.04
        self.default_bars = 5
        
        # Load models
        self._load_models(entry_model_path, sl_model_path, tp_model_path, bars_model_path)
    
    def _load_models(self, entry_path, sl_path, tp_path, bars_path):
        """Load all models."""
        # Default paths
        if entry_path is None:
            entry_path = MODEL_DIR / 'entry_filter.joblib'
        if sl_path is None:
            sl_path = MODEL_DIR / 'sl_predictor.joblib'
        if tp_path is None:
            tp_path = MODEL_DIR / 'tp_predictor.joblib'
        if bars_path is None:
            bars_path = MODEL_DIR / 'bars_predictor.joblib'
        
        # Load Entry Filter
        if Path(entry_path).exists():
            data = joblib.load(entry_path)
            self.entry_model = data['model']
            self.entry_scaler = data.get('scaler')
            self.entry_features = data['feature_names']
            print(f"✓ Entry Filter loaded: {len(self.entry_features)} features")
        
        # Load SL Predictor
        if Path(sl_path).exists():
            data = joblib.load(sl_path)
            self.sl_model = data['model']
            self.sl_scaler = data.get('scaler')
            self.sl_features = data['feature_names']
            print(f"✓ SL Predictor loaded: {len(self.sl_features)} features")
        
        # Load TP Predictor
        if Path(tp_path).exists():
            data = joblib.load(tp_path)
            self.tp_model = data['model']
            self.tp_scaler = data.get('scaler')
            self.tp_features = data['feature_names']
            print(f"✓ TP Predictor loaded: {len(self.tp_features)} features")
        
        # Load Bars Predictor
        if Path(bars_path).exists():
            data = joblib.load(bars_path)
            self.bars_model = data['model']
            self.bars_scaler = data.get('scaler')
            self.bars_features = data['feature_names']
            print(f"✓ Bars Predictor loaded: {len(self.bars_features)} features")
    
    def _prepare_features(self, features: pd.DataFrame, feature_names: list, scaler) -> np.ndarray:
        """Prepare features for prediction."""
        X = pd.DataFrame()
        for col in feature_names:
            if col in features.columns:
                val = features[col].values if hasattr(features[col], 'values') else [features[col]]
                X[col] = val
            else:
                X[col] = 0
        
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        if scaler is not None:
            X = scaler.transform(X)
        
        return X
    
    def predict_entry(self, features: pd.DataFrame) -> tuple:
        """Predict entry quality."""
        if self.entry_model is None:
            return True, 0.5
        
        X = self._prepare_features(features, self.entry_features, self.entry_scaler)
        proba = self.entry_model.predict_proba(X)[0, 1]
        should_enter = proba >= self.entry_threshold
        return should_enter, proba
    
    def predict_sl(self, features: pd.DataFrame) -> float:
        """Predict optimal SL percentage."""
        if self.sl_model is None:
            return self.default_sl
        
        X = self._prepare_features(features, self.sl_features, self.sl_scaler)
        sl = self.sl_model.predict(X)[0]
        return np.clip(sl, 0.005, 0.15)  # 0.5% to 15%
    
    def predict_tp(self, features: pd.DataFrame) -> float:
        """Predict optimal TP percentage."""
        if self.tp_model is None:
            return self.default_tp
        
        X = self._prepare_features(features, self.tp_features, self.tp_scaler)
        tp = self.tp_model.predict(X)[0]
        return np.clip(tp, 0.01, 0.50)  # 1% to 50%
    
    def predict_bars(self, features: pd.DataFrame) -> int:
        """Predict bars to peak."""
        if self.bars_model is None:
            return self.default_bars
        
        X = self._prepare_features(features, self.bars_features, self.bars_scaler)
        bars = self.bars_model.predict(X)[0]
        return int(np.round(np.clip(bars, 1, 10)))
    
    def calculate_adjusted_entry(
        self,
        current_price: float,
        sl_pct: float,
        is_long: bool,
        atr_pct: float = None
    ) -> tuple:
        """
        Calculate adjusted entry price if SL is far.
        
        Logic:
        - If SL > threshold (e.g., 3%), suggest limit order at better price
        - Entry offset = (SL - target_SL) * adjustment_factor
        - This gives better RR while keeping same absolute SL level
        
        Returns:
            (adjusted_entry, entry_offset_pct, notes)
        """
        notes = []
        target_sl = 0.02  # Target 2% SL
        
        if sl_pct <= self.sl_adjustment_threshold:
            # SL is reasonable, enter at market
            return current_price, 0.0, notes
        
        # SL is wide - calculate offset
        # Adjustment: offset = (actual_SL - target_SL) * factor
        # Factor depends on how much we want to optimize entry
        excess_sl = sl_pct - target_sl
        adjustment_factor = 0.5  # Take 50% of excess as entry offset
        
        entry_offset_pct = excess_sl * adjustment_factor
        
        # Cap the offset
        max_offset = 0.02  # Max 2% entry adjustment
        entry_offset_pct = min(entry_offset_pct, max_offset)
        
        if is_long:
            # For long: enter lower (limit buy below current)
            adjusted_entry = current_price * (1 - entry_offset_pct)
            notes.append(f"SL rộng ({sl_pct:.1%}) → Đặt limit buy thấp hơn {entry_offset_pct:.1%}")
        else:
            # For short: enter higher (limit sell above current)
            adjusted_entry = current_price * (1 + entry_offset_pct)
            notes.append(f"SL rộng ({sl_pct:.1%}) → Đặt limit sell cao hơn {entry_offset_pct:.1%}")
        
        return adjusted_entry, entry_offset_pct, notes
    
    def get_recommendation(
        self,
        features: pd.DataFrame,
        symbol: str,
        timeframe: str,
        current_price: float,
        is_long: bool,
        signal_time: str = None
    ) -> TradeRecommendation:
        """
        Get full trade recommendation.
        
        Args:
            features: DataFrame with technical indicators
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            current_price: Current close price
            is_long: True for bullish crossover, False for bearish
            signal_time: Timestamp of signal
        
        Returns:
            TradeRecommendation with all details
        """
        notes = []
        
        # Stage 1: Entry Filter
        should_enter, entry_confidence = self.predict_entry(features)
        
        # Stage 2: SL Prediction
        sl_pct = self.predict_sl(features)
        
        # Stage 3: TP Prediction
        tp_pct = self.predict_tp(features)
        
        # Stage 4: Bars to Peak
        expected_bars = self.predict_bars(features)
        
        # Calculate adjusted entry
        adjusted_entry, entry_offset_pct, entry_notes = self.calculate_adjusted_entry(
            current_price, sl_pct, is_long
        )
        notes.extend(entry_notes)
        
        # Calculate SL/TP prices
        if is_long:
            stop_loss = adjusted_entry * (1 - sl_pct)
            take_profit = adjusted_entry * (1 + tp_pct)
        else:
            stop_loss = adjusted_entry * (1 + sl_pct)
            take_profit = adjusted_entry * (1 - tp_pct)
        
        # Risk/Reward
        risk_reward = tp_pct / sl_pct if sl_pct > 0 else 0
        
        # Check RR threshold
        if risk_reward < self.min_rr_ratio:
            should_enter = False
            notes.append(f"RR thấp ({risk_reward:.2f} < {self.min_rr_ratio})")
        
        # Generate recommendation text
        signal_type = 'LONG' if is_long else 'SHORT'
        
        if should_enter:
            if entry_offset_pct > 0:
                rec = f"✅ {signal_type} - Limit Order tại ${adjusted_entry:,.2f}"
            else:
                rec = f"✅ {signal_type} - Market Entry tại ${current_price:,.2f}"
        else:
            rec = f"❌ SKIP - Entry confidence thấp ({entry_confidence:.0%})"
        
        # Add notes about timing
        notes.append(f"Dự kiến đạt đỉnh sau {expected_bars} nến")
        
        # Confidence notes
        if entry_confidence >= 0.7:
            notes.append("🟢 High confidence")
        elif entry_confidence >= 0.5:
            notes.append("🟡 Medium confidence")
        else:
            notes.append("🔴 Low confidence")
        
        return TradeRecommendation(
            symbol=symbol,
            timeframe=timeframe,
            signal_type=signal_type,
            signal_time=signal_time or '',
            current_price=current_price,
            recommended_entry=adjusted_entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            entry_offset_pct=entry_offset_pct,
            risk_reward=risk_reward,
            entry_confidence=entry_confidence,
            should_enter=should_enter,
            expected_bars_to_peak=expected_bars,
            recommendation=rec,
            notes=notes
        )
    
    def format_recommendation(self, rec: TradeRecommendation) -> str:
        """Format recommendation for display/Telegram."""
        lines = [
            f"{'='*40}",
            f"🔔 {rec.symbol} | {rec.timeframe} | {rec.signal_type}",
            f"{'='*40}",
            f"",
            f"📊 {rec.recommendation}",
            f"",
            f"💰 Entry:  ${rec.recommended_entry:,.2f}",
        ]
        
        if rec.entry_offset_pct > 0:
            lines.append(f"   (Limit {rec.entry_offset_pct:.1%} {'dưới' if rec.signal_type == 'LONG' else 'trên'} giá hiện tại)")
        
        lines.extend([
            f"🛑 SL:     ${rec.stop_loss:,.2f} ({rec.sl_pct:.1%})",
            f"🎯 TP:     ${rec.take_profit:,.2f} ({rec.tp_pct:.1%})",
            f"",
            f"📈 Risk/Reward: {rec.risk_reward:.2f}",
            f"🎲 Confidence:  {rec.entry_confidence:.0%}",
            f"⏱️ Peak trong:  ~{rec.expected_bars_to_peak} nến",
            f"",
        ])
        
        if rec.notes:
            lines.append("📝 Notes:")
            for note in rec.notes:
                lines.append(f"   • {note}")
        
        return '\n'.join(lines)


def test_multi_timeframe():
    """Test recommender with multiple timeframes."""
    from data_pipeline import build_dataset, calculate_features
    
    print("="*60)
    print("Multi-Timeframe ML Recommendation Test")
    print("="*60)
    
    # Initialize recommender
    recommender = MLTradeRecommender(
        entry_threshold=0.5,
        min_rr_ratio=1.5,
        sl_adjustment_threshold=0.03
    )
    
    # Test with synthetic data (in real use, this comes from API)
    # For now, load existing processed data
    DATA_DIR = Path(__file__).parent.parent / 'data' / 'processed'
    data_path = DATA_DIR / 'features_1d_full.parquet'
    
    if not data_path.exists():
        data_path = DATA_DIR / 'features_1d_test.parquet'
    
    if not data_path.exists():
        print("No test data found!")
        return
    
    df = pd.read_parquet(data_path)
    
    # Get some sample crossovers
    crossovers = df[(df['macd_cross_up'] == 1) | (df['macd_cross_down'] == 1)].tail(10)
    
    print(f"\nTesting with {len(crossovers)} recent crossovers...")
    
    for _, row in crossovers.iterrows():
        row_df = pd.DataFrame([row])
        
        rec = recommender.get_recommendation(
            features=row_df,
            symbol=row.get('symbol', 'BTCUSDT'),
            timeframe='1d',
            current_price=row['close'],
            is_long=row['macd_cross_up'] == 1,
            signal_time=str(row.get('timestamp', ''))
        )
        
        print(recommender.format_recommendation(rec))
        print()


if __name__ == '__main__':
    test_multi_timeframe()
