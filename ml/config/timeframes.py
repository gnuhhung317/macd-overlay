#!/usr/bin/env python3
"""
Timeframe Configuration for ML Models

Defines supported timeframes and their specific configurations.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

# Supported timeframes
SUPPORTED_TIMEFRAMES = ['1h', '4h', '8h', '12h', '1d', '1w']

@dataclass
class TimeframeConfig:
    """Configuration for a specific timeframe"""
    name: str
    resample_from: str = '1h'  # Base data to resample from
    max_bars: int = 10  # Max bars to hold trade
    
    # Feature engineering
    sma_periods: List[int] = None
    volatility_periods: List[int] = None
    atr_periods: List[int] = None
    
    # Model hyperparameters (can be tuned per timeframe)
    entry_threshold: float = 0.65
    
    # Position management
    default_leverage: float = 3.0
    max_sl_pct: float = 0.15
    max_tp_pct: float = 0.30
    
    def __post_init__(self):
        if self.sma_periods is None:
            self.sma_periods = [7, 14, 21, 50, 100, 200]
        if self.volatility_periods is None:
            self.volatility_periods = [7, 14, 21]
        if self.atr_periods is None:
            self.atr_periods = [7, 14]


# Default configurations per timeframe
TIMEFRAME_CONFIGS: Dict[str, TimeframeConfig] = {
    '1h': TimeframeConfig(
        name='1h',
        resample_from='1h',
        max_bars=24,  # 1 day
        entry_threshold=0.70,  # Higher threshold for noisy 1h
        default_leverage=2.0,
        sma_periods=[7, 14, 21, 50, 100, 200],
    ),
    '4h': TimeframeConfig(
        name='4h',
        resample_from='1h',
        max_bars=12,  # 2 days
        entry_threshold=0.65,
        default_leverage=3.0,
    ),
    '8h': TimeframeConfig(
        name='8h',
        resample_from='1h',
        max_bars=9,  # 3 days
        entry_threshold=0.65,
        default_leverage=3.0,
    ),
    '12h': TimeframeConfig(
        name='12h',
        resample_from='1h',
        max_bars=8,  # 4 days
        entry_threshold=0.60,
        default_leverage=4.0,
    ),
    '1d': TimeframeConfig(
        name='1d',
        resample_from='1h',
        max_bars=10,  # 10 days
        entry_threshold=0.60,
        default_leverage=5.0,
    ),
    '1w': TimeframeConfig(
        name='1w',
        resample_from='1d',  # Resample from 1d or 1h? Let's use 1h to match others, or 1d to be faster
        max_bars=4,   # 4 weeks (about 1 month)
        entry_threshold=0.60,
        default_leverage=2.0,
    ),
}


def get_timeframe_config(timeframe: str) -> TimeframeConfig:
    """Get configuration for a timeframe"""
    if timeframe not in TIMEFRAME_CONFIGS:
        raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: {SUPPORTED_TIMEFRAMES}")
    return TIMEFRAME_CONFIGS[timeframe]


def get_model_path(timeframe: str, model_type: str) -> str:
    """
    Get model path for a specific timeframe and model type.
    
    Args:
        timeframe: '1h', '4h', '8h', '12h', '1d'
        model_type: 'entry_filter', 'sl_predictor', 'tp_predictor'
    
    Returns:
        Path like 'models/1d/entry_filter.joblib'
    """
    return f"models/{timeframe}/{model_type}.joblib"


def get_data_path(timeframe: str) -> str:
    """Get processed data path for a timeframe"""
    return f"data/processed/features_{timeframe}_full.parquet"
