#!/usr/bin/env python3
"""
SL Predictor Training Module
"""
import sys
import time
from datetime import datetime
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Sklearn imports
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Local imports
# Add parent directory to path to find training_utils
sys.path.insert(0, str(Path(__file__).parent))
from training_utils import (
    TrainingResult, get_feature_columns, 
    PROCESSED_DIR, MODELS_DIR
)


def get_regression_models() -> Dict:
    """Get regression models for SL/TP Predictor"""
    return {
        'XGBoost': xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_split=20,
            random_state=42, n_jobs=-1
        ),
    }


def prepare_sl_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Prepare data for SL Predictor training"""
    df_cross = df[(df['macd_cross_up'] == 1) | (df['macd_cross_down'] == 1)].copy()
    
    # Use sl_pct_used (from pipeline) or actual_sl (legacy)
    sl_col = 'sl_pct_used' if 'sl_pct_used' in df_cross.columns else 'actual_sl'
    df_cross = df_cross.dropna(subset=[sl_col])
    
    # Filter valid SL range
    df_cross = df_cross[(df_cross[sl_col] > 0.005) & (df_cross[sl_col] < 0.15)]
    
    # IMPORTANT: Exclude ATR to prevent data leakage (SL is ATR-based)
    feature_cols = get_feature_columns(df_cross, exclude_atr=True)
    X = df_cross[feature_cols].copy()
    y = df_cross[sl_col]
    
    X['is_bullish_cross'] = df_cross['macd_cross_up'].values
    feature_cols.append('is_bullish_cross')
    
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    
    return X, y, feature_cols


def train_sl_predictor(timeframe: str, tune: bool = False) -> TrainingResult:
    """Train SL Predictor for a specific timeframe"""
    start_time = time.time()
    print(f"\n{'='*60}")
    print(f"Training SL Predictor for {timeframe}")
    print('='*60)
    
    # Load data
    # Priority: specialized train file (small) > full file (huge)
    train_path = PROCESSED_DIR / f'sl_train_{timeframe}.parquet'
    data_path = PROCESSED_DIR / f'features_{timeframe}_full.parquet'
    
    if train_path.exists():
        print(f"  Using optimized training set: {train_path.name}")
        df = pd.read_parquet(train_path)
    elif data_path.exists():
        print(f"  Using full dataset: {data_path.name}")
        df = pd.read_parquet(data_path)
    else:
        raise FileNotFoundError(f"Data not found: {data_path}")

    X, y, feature_cols = prepare_sl_data(df)
    
    print(f"Samples: {len(X)}, Features: {len(feature_cols)}")
    print(f"SL range: {y.min():.2%} - {y.max():.2%}, Mean: {y.mean():.2%}")
    
    # Time-based split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    # Train models
    tscv = TimeSeriesSplit(n_splits=5)
    models = get_regression_models()
    results = {}
    
    for name, model in models.items():
        print(f"  Training {name}...")
        scores = cross_val_score(model, X_train_scaled, y_train, cv=tscv, scoring='neg_mean_absolute_error')
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'cv_score': -scores.mean(),
            'test_mae': test_mae,
            'test_r2': test_r2
        }
        print(f"    CV MAE: {-scores.mean():.4f}, Test MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
    
    # Select best (lowest MAE)
    best_name = min(results.keys(), key=lambda k: results[k]['test_mae'])
    best_model = results[best_name]['model']
    
    # Save
    model_dir = MODELS_DIR / timeframe
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'sl_predictor.joblib'
    
    joblib.dump({
        'model': best_model,
        'scaler': scaler,
        'feature_names': feature_cols,
        'timeframe': timeframe,
        'trained_at': datetime.now().isoformat()
    }, model_path)
    
    training_time = time.time() - start_time
    print(f"\n✓ Best: {best_name}, Test MAE: {results[best_name]['test_mae']:.4f}")
    print(f"✓ Saved to: {model_path}")
    
    return TrainingResult(
        timeframe=timeframe,
        model_type='sl_predictor',
        best_model_name=best_name,
        cv_score=results[best_name]['cv_score'],
        test_score=results[best_name]['test_mae'],
        test_score_name='MAE',
        feature_count=len(feature_cols),
        training_time=training_time,
        model_path=str(model_path)
    )

if __name__ == "__main__":
    if len(sys.argv) > 1:
        train_sl_predictor(sys.argv[1])
    else:
        print("Usage: python train_sl.py [timeframe]")
