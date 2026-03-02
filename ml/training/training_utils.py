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

def evaluate_on_exchanges(model, scaler, feature_cols: List[str], timeframe: str, model_type: str):
    """
    Evaluate trained model on alternative exchanges (Bitget, Bybit) to test robustness.
    """
    from sklearn.metrics import roc_auc_score, mean_absolute_error
    from scipy.stats import spearmanr
    exchanges = ['bitget', 'bybit', 'kraken', 'okx', 'mexc']
    print("\n" + "-"*60)
    print(f"EVALUATING ON OTHER EXCHANGES (Robustness Test)")
    print("-"*60)
    
    for exchange in exchanges:
        exch_data_dir = ML_DIR.parent / f'{exchange}-data' / 'processed'
        data_path = exch_data_dir / f'features_{timeframe}_full.parquet'
        
        if not data_path.exists():
            print(f"  {exchange.upper():<10}: No data found")
            continue
            
        try:
            df = pd.read_parquet(data_path)
            df_cross = df[(df['macd_cross_up'] == 1) | (df['macd_cross_down'] == 1)].copy()
            
            if model_type == 'entry_filter':
                df_cross = df_cross.dropna(subset=['label'])
                y = df_cross['label'].astype(int)
            elif model_type == 'sl_predictor':
                sl_col = 'sl_pct_used' if 'sl_pct_used' in df_cross.columns else 'actual_sl'
                df_cross = df_cross.dropna(subset=[sl_col])
                df_cross = df_cross[(df_cross[sl_col] > 0.005) & (df_cross[sl_col] < 0.15)]
                y = df_cross[sl_col]
            else: # tp_predictor
                tp_col = 'tp_pct_used' if 'tp_pct_used' in df_cross.columns else 'actual_tp'
                df_cross = df_cross.dropna(subset=[tp_col])
                df_cross = df_cross[(df_cross[tp_col] > 0.01) & (df_cross[tp_col] < 1.0)]
                df_cross[tp_col] = df_cross[tp_col].clip(upper=0.30)
                y = df_cross[tp_col]

            raw_cols = [c for c in feature_cols if c != 'is_bullish_cross']
            missing = [c for c in raw_cols if c not in df_cross.columns]
            if missing:
                print(f"  {exchange.upper():<10}: Missing cols: {missing[:3]}...")
                continue
                
            X = df_cross[raw_cols].copy()
            X['is_bullish_cross'] = df_cross['macd_cross_up'].values
            X = X[feature_cols] # Ensure exact order
            X = X.fillna(0).replace([np.inf, -np.inf], 0)
            
            X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
            
            if model_type == 'entry_filter':
                y_proba = model.predict_proba(X_scaled)[:, 1]
                score = roc_auc_score(y, y_proba)
                print(f"  {exchange.upper():<10}: AUC = {score:.4f} ({len(X)} samples)")
            else:
                y_pred = model.predict(X_scaled)
                mae_score = mean_absolute_error(y, y_pred)
                ic_score, _ = spearmanr(y, y_pred)
                print(f"  {exchange.upper():<10}: MAE = {mae_score:.4f}, IC = {ic_score:.4f} ({len(X)} samples)")
                
        except Exception as e:
            print(f"  {exchange.upper():<10}: Error - {e}")
