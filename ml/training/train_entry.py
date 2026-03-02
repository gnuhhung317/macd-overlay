#!/usr/bin/env python3
"""
Entry Filter Training Module
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb

# Local imports
# Add parent directory to path to find training_utils
sys.path.insert(0, str(Path(__file__).parent))
from training_utils import (
    TrainingResult, get_feature_columns, 
    PROCESSED_DIR, MODELS_DIR
)

import optuna
from optuna.samplers import TPESampler
import warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective_xgboost(trial, X_train, y_train, cv):
    """Hàm mục tiêu cho Optuna để tune XGBoost"""
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'random_state': 42,
        'eval_metric': 'logloss',
        'verbosity': 0
    }
    model = xgb.XGBClassifier(**param)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    return scores.mean()


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
        # 'RandomForest': RandomForestClassifier(
        #     n_estimators=200, max_depth=10, min_samples_split=20,
        #     class_weight='balanced', random_state=42, n_jobs=-1
        # ),
    }


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
    results = {}
    
    if tune:
        print(f"\n🚀 Khởi động Optuna Hyperparameter Tuning cho XGBoost...")
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(lambda trial: objective_xgboost(trial, X_train_scaled, y_train, tscv), n_trials=30)
        
        print("\n✨ Quá trình Tuning hoàn tất!")
        print(f" 🏆 Best CV AUC: {study.best_value:.4f}")
        print(" 🎯 Best Parameters:")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")
            
        best_xgb = xgb.XGBClassifier(**study.best_params, random_state=42, eval_metric='logloss')
        best_xgb.fit(X_train_scaled, y_train)
        
        y_proba = best_xgb.predict_proba(X_test_scaled)[:, 1]
        test_auc = roc_auc_score(y_test, y_proba)
        print(f" 📊 Final Test AUC: {test_auc:.4f}")
        
        results['Tuned_XGBoost'] = {
            'model': best_xgb,
            'cv_score': study.best_value,
            'test_auc': test_auc
        }
    else:
        models = get_classification_models()
        for name, model in models.items():
            print(f"  Training {name}...")
            scores = cross_val_score(model, X_train_scaled, y_train, cv=tscv, scoring='roc_auc', n_jobs=-1)
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
    
    # --- EVALUATE ON OTHER EXCHANGES ---
    from training_utils import evaluate_on_exchanges
    evaluate_on_exchanges(best_model, scaler, feature_cols, timeframe, 'entry_filter')
    
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

if __name__ == "__main__":
    if len(sys.argv) > 1:
        train_entry_filter(sys.argv[1])
    else:
        print("Usage: python train_entry.py [timeframe]")
