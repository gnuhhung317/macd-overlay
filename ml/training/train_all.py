#!/usr/bin/env python3
"""
Multi-Timeframe Model Trainer

Trains Entry Filter, SL Predictor, and TP Predictor for each timeframe.
Supports parallel training and hyperparameter tuning.
"""
import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
import joblib
from dataclasses import dataclass

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SUPPORTED_TIMEFRAMES, get_timeframe_config, get_model_path

# Sklearn imports
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

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


def prepare_entry_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Prepare data for Entry Filter training"""
    # Only crossover rows
    df_cross = df[(df['macd_cross_up'] == 1) | (df['macd_cross_down'] == 1)].copy()
    df_cross = df_cross.dropna(subset=['label'])
    
    feature_cols = get_feature_columns(df_cross)
    X = df_cross[feature_cols].copy()
    y = df_cross['label'].astype(int)
    
    # Add crossover direction
    X['is_bullish_cross'] = df_cross['macd_cross_up'].values
    feature_cols.append('is_bullish_cross')
    
    # Clean
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    
    return X, y, feature_cols


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


def prepare_tp_data(df: pd.DataFrame, max_tp: float = 0.30) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Prepare data for TP Predictor training"""
    df_cross = df[(df['macd_cross_up'] == 1) | (df['macd_cross_down'] == 1)].copy()
    
    # Use tp_pct_used (from pipeline) or actual_tp (legacy)
    tp_col = 'tp_pct_used' if 'tp_pct_used' in df_cross.columns else 'actual_tp'
    df_cross = df_cross.dropna(subset=[tp_col])
    
    # Filter and cap TP
    df_cross = df_cross[(df_cross[tp_col] > 0.01) & (df_cross[tp_col] < 1.0)]
    df_cross[tp_col] = df_cross[tp_col].clip(upper=max_tp)
    
    # IMPORTANT: Exclude ATR to prevent data leakage (TP is ATR-based)
    feature_cols = get_feature_columns(df_cross, exclude_atr=True)
    X = df_cross[feature_cols].copy()
    y = df_cross[tp_col]
    
    X['is_bullish_cross'] = df_cross['macd_cross_up'].values
    feature_cols.append('is_bullish_cross')
    
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    
    return X, y, feature_cols


def get_classification_models() -> Dict:
    """Get classification models for Entry Filter"""
    return {
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric='logloss', verbosity=0
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, class_weight='balanced',
            random_state=42, verbose=-1
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=20,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
    }


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


def train_entry_filter(timeframe: str, tune: bool = False) -> TrainingResult:
    """Train Entry Filter for a specific timeframe"""
    start_time = time.time()
    print(f"\n{'='*60}")
    print(f"Training Entry Filter for {timeframe}")
    print('='*60)
    
    # Load data
    data_path = PROCESSED_DIR / f'features_{timeframe}_full.parquet'
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
    
    df = pd.read_parquet(data_path)
    X, y, feature_cols = prepare_entry_data(df)
    
    print(f"Samples: {len(X)}, Features: {len(feature_cols)}")
    print(f"Win rate: {y.mean():.2%}")
    
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
    models = get_classification_models()
    results = {}
    
    for name, model in models.items():
        print(f"  Training {name}...")
        scores = cross_val_score(model, X_train_scaled, y_train, cv=tscv, scoring='roc_auc')
        model.fit(X_train_scaled, y_train)
        
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        test_auc = roc_auc_score(y_test, y_proba)
        
        results[name] = {
            'model': model,
            'cv_score': scores.mean(),
            'test_auc': test_auc
        }
        print(f"    CV AUC: {scores.mean():.4f}, Test AUC: {test_auc:.4f}")
    
    # Select best
    best_name = max(results.keys(), key=lambda k: results[k]['test_auc'])
    best_model = results[best_name]['model']
    
    # Save
    model_dir = MODELS_DIR / timeframe
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'entry_filter.joblib'
    
    joblib.dump({
        'model': best_model,
        'scaler': scaler,
        'feature_names': feature_cols,
        'timeframe': timeframe,
        'trained_at': datetime.now().isoformat()
    }, model_path)
    
    training_time = time.time() - start_time
    print(f"\n✓ Best: {best_name}, Test AUC: {results[best_name]['test_auc']:.4f}")
    print(f"✓ Saved to: {model_path}")
    
    return TrainingResult(
        timeframe=timeframe,
        model_type='entry_filter',
        best_model_name=best_name,
        cv_score=results[best_name]['cv_score'],
        test_score=results[best_name]['test_auc'],
        test_score_name='AUC',
        feature_count=len(feature_cols),
        training_time=training_time,
        model_path=str(model_path)
    )


def train_sl_predictor(timeframe: str, tune: bool = False) -> TrainingResult:
    """Train SL Predictor for a specific timeframe"""
    start_time = time.time()
    print(f"\n{'='*60}")
    print(f"Training SL Predictor for {timeframe}")
    print('='*60)
    
    # Load data
    data_path = PROCESSED_DIR / f'features_{timeframe}_full.parquet'
    df = pd.read_parquet(data_path)
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


def train_tp_predictor(timeframe: str, tune: bool = False) -> TrainingResult:
    """Train TP Predictor for a specific timeframe"""
    start_time = time.time()
    print(f"\n{'='*60}")
    print(f"Training TP Predictor for {timeframe}")
    print('='*60)
    
    # Load data
    data_path = PROCESSED_DIR / f'features_{timeframe}_full.parquet'
    df = pd.read_parquet(data_path)
    X, y, feature_cols = prepare_tp_data(df)
    
    print(f"Samples: {len(X)}, Features: {len(feature_cols)}")
    print(f"TP range: {y.min():.2%} - {y.max():.2%}, Mean: {y.mean():.2%}")
    
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
    
    # Select best
    best_name = min(results.keys(), key=lambda k: results[k]['test_mae'])
    best_model = results[best_name]['model']
    
    # Save
    model_dir = MODELS_DIR / timeframe
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'tp_predictor.joblib'
    
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
        model_type='tp_predictor',
        best_model_name=best_name,
        cv_score=results[best_name]['cv_score'],
        test_score=results[best_name]['test_mae'],
        test_score_name='MAE',
        feature_count=len(feature_cols),
        training_time=training_time,
        model_path=str(model_path)
    )


def train_all_models_for_timeframe(timeframe: str, tune: bool = False) -> List[TrainingResult]:
    """Train all 3 models for a specific timeframe"""
    results = []
    
    try:
        results.append(train_entry_filter(timeframe, tune))
    except Exception as e:
        print(f"❌ Entry Filter failed for {timeframe}: {e}")
    
    try:
        results.append(train_sl_predictor(timeframe, tune))
    except Exception as e:
        print(f"❌ SL Predictor failed for {timeframe}: {e}")
    
    try:
        results.append(train_tp_predictor(timeframe, tune))
    except Exception as e:
        print(f"❌ TP Predictor failed for {timeframe}: {e}")
    
    return results


def print_summary(all_results: List[TrainingResult]):
    """Print training summary table"""
    print("\n" + "="*100)
    print("TRAINING SUMMARY")
    print("="*100)
    
    # Group by timeframe
    by_timeframe = {}
    for r in all_results:
        if r.timeframe not in by_timeframe:
            by_timeframe[r.timeframe] = {}
        by_timeframe[r.timeframe][r.model_type] = r
    
    print(f"\n{'Timeframe':<10} {'Entry Filter':<25} {'SL Predictor':<25} {'TP Predictor':<25}")
    print("-"*100)
    
    for tf in SUPPORTED_TIMEFRAMES:
        if tf not in by_timeframe:
            print(f"{tf:<10} {'N/A':<25} {'N/A':<25} {'N/A':<25}")
            continue
        
        models = by_timeframe[tf]
        
        entry_str = "N/A"
        if 'entry_filter' in models:
            r = models['entry_filter']
            entry_str = f"{r.best_model_name} AUC={r.test_score:.3f}"
        
        sl_str = "N/A"
        if 'sl_predictor' in models:
            r = models['sl_predictor']
            sl_str = f"{r.best_model_name} MAE={r.test_score:.4f}"
        
        tp_str = "N/A"
        if 'tp_predictor' in models:
            r = models['tp_predictor']
            tp_str = f"{r.best_model_name} MAE={r.test_score:.4f}"
        
        print(f"{tf:<10} {entry_str:<25} {sl_str:<25} {tp_str:<25}")
    
    # Total time
    total_time = sum(r.training_time for r in all_results)
    print(f"\nTotal training time: {total_time/60:.1f} minutes")


def main():
    parser = argparse.ArgumentParser(description='Train ML models for all timeframes')
    parser.add_argument('timeframe', nargs='?', default='all', 
                       help='Timeframe to train (1h, 4h, 8h, 12h, 1d, all)')
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'entry', 'sl', 'tp'],
                       help='Model type to train')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Multi-Timeframe ML Model Trainer")
    print("="*60)
    
    # Determine timeframes
    if args.timeframe == 'all':
        timeframes = SUPPORTED_TIMEFRAMES
    else:
        timeframes = [args.timeframe]
    
    # Check data exists
    for tf in timeframes:
        data_path = PROCESSED_DIR / f'features_{tf}_full.parquet'
        if not data_path.exists():
            print(f"⚠️ Data not found for {tf}: {data_path}")
            print(f"   Run: python multi_timeframe_pipeline.py {tf}")
            timeframes.remove(tf)
    
    if not timeframes:
        print("No valid timeframes to train!")
        return
    
    print(f"\nTimeframes: {timeframes}")
    print(f"Model type: {args.model}")
    print(f"Hyperparameter tuning: {args.tune}")
    
    all_results = []
    
    for tf in timeframes:
        if args.model == 'all':
            results = train_all_models_for_timeframe(tf, args.tune)
            all_results.extend(results)
        elif args.model == 'entry':
            all_results.append(train_entry_filter(tf, args.tune))
        elif args.model == 'sl':
            all_results.append(train_sl_predictor(tf, args.tune))
        elif args.model == 'tp':
            all_results.append(train_tp_predictor(tf, args.tune))
    
    print_summary(all_results)
    
    print("\n✅ Training complete!")


if __name__ == '__main__':
    main()
