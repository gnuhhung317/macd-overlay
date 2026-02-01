#!/usr/bin/env python3
"""
Shared utilities for ML training modules.
"""
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List
import pandas as pd
import numpy as np

# Add parent to path to allow importing config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Paths
ML_DIR = Path(__file__).parent.parent
DATA_DIR = ML_DIR.parent / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = ML_DIR / 'models'

@dataclass
class TrainingResult:
    """Result of training a single model"""
    timeframe: str
    model_type: str
    best_model_name: str
    cv_score: float
    test_score: float
    test_score_name: str  # 'AUC' or 'MAE'
    feature_count: int
    training_time: float
    model_path: str


def get_feature_columns(df: pd.DataFrame, exclude_atr: bool = False) -> List[str]:
    """
    Get list of feature columns from dataframe
    
    Args:
        df: DataFrame
        exclude_atr: If True, exclude ATR/volatility columns to prevent data leakage
                    when training SL/TP predictors (since targets are ATR-based)
    """
    # CRITICAL: Exclude all future/outcome information
    exclude_cols = {
        # Identifiers and raw OHLCV
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'date',
        
        # Labels and targets (FUTURE INFO - NEVER USE!)
        'label', 'actual_tp', 'actual_sl', 'actual_rr',
        'tp_pct_used', 'sl_pct_used',
        
        # Outcome data (FUTURE INFO - NEVER USE!)
        'bars_to_tp', 'bars_to_sl', 'bars_to_outcome',
        'tp_first', 'sl_first', 'outcome', 'trade_result',
        'max_profit', 'max_drawdown',  # These are trade results!
        
        # Crossover signals (used for filtering, not prediction)
        'macd_cross_up', 'macd_cross_down', 'macd_crossover',
    }
    
    # Exclude ATR and volatility columns for SL/TP prediction
    # Since sl_pct_used = 1.5 * ATR and tp_pct_used = 3.0 * ATR
    # And volatility is highly correlated with ATR
    if exclude_atr:
        exclude_cols.update({
            'atr_7', 'atr_14', 'atr_21', 'atr',
            'volatility_7', 'volatility_14', 'volatility_21',
            'volatility_7_scaled', 'volatility_14_scaled',
            'bb_width',  # BB width is based on volatility
        })
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Filter numeric only
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    return numeric_cols
