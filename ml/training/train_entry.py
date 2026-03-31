#!/usr/bin/env python3
"""
Entry Filter Training Module (Dual Long/Short Architecture)
"""
import sys
import time
from datetime import datetime
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
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
    PROCESSED_DIR, MODELS_DIR, filter_top_symbols_by_volume,
    evaluate_on_exchanges
)

import optuna
from optuna.samplers import TPESampler
import warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)

def plot_and_save_feature_importance(model, feature_names: List[str], timeframe: str, save_dir: Path):
    """Plot and save top-25 feature importance chart."""
    if isinstance(model, dict) and 'long' in model:
        plot_and_save_feature_importance(model['long'], feature_names, f"{timeframe}_long", save_dir)
        plot_and_save_feature_importance(model['short'], feature_names, f"{timeframe}_short", save_dir)
        return
        
    if not hasattr(model, 'feature_importances_'):
        print("Model does not support feature_importances_")
        return
    
    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_})
    total = df_imp['Importance'].sum()
    if total > 0:
        df_imp['Importance'] = df_imp['Importance'] / total
    df_imp = df_imp.sort_values('Importance', ascending=False).head(25)
    
    print(f"\n🧐 Top 10 Most Important Features ({timeframe}):")
    for _, row in df_imp.head(10).iterrows():
        print(f"  {row['Feature']:<35}: {row['Importance']:.4f}")
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=df_imp, palette='viridis')
    plt.title(f'Top 25 Feature Importances ({timeframe})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = save_dir / f'feature_importance_{timeframe}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Feature importance chart saved: {save_path}")

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
        )
    }

def prepare_entry_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Prepare data for Entry Filter training"""
    # Only crossover rows
    df_cross = df[(df['macd_cross_up'] == 1) | (df['macd_cross_down'] == 1)].copy()
    df_cross = df_cross.dropna(subset=['label'])
    
    feature_cols = get_feature_columns(df_cross)
    # Remove is_bullish_cross from feature_cols!
    if 'is_bullish_cross' in feature_cols:
        feature_cols.remove('is_bullish_cross')
        
    X = df_cross[feature_cols].copy()
    y = df_cross['label'].astype(int)
    
    # Add crossover direction temporarily for routing
    X['is_bullish_cross'] = df_cross['macd_cross_up'].values
    
    # Clean
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    
    return X, y, feature_cols

def train_directional_model(direction_name: str, mask_train, mask_test, X_train_full, X_test_full, y_train_full, y_test_full, tscv, tune):
    print(f"\n{'-'*40}")
    print(f"Training {direction_name.upper()} Model")
    print(f"{'-'*40}")
    
    # Extract data for this direction
    X_train = X_train_full[mask_train].drop(columns=['is_bullish_cross'])
    X_test = X_test_full[mask_test].drop(columns=['is_bullish_cross'])
    y_train = y_train_full[mask_train]
    y_test = y_test_full[mask_test]
    
    print(f"{direction_name} Train: {len(X_train)} samples ({y_train.mean():.2%} win rate)")
    print(f"{direction_name} Test:  {len(X_test)} samples ({y_test.mean():.2%} win rate)")
    
    if len(X_train) == 0 or len(X_test) == 0:
        print(f"Warning: Not enough data for {direction_name}")
        return None, None, 0, 0
        
    # Scale
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    if tune:
        print(f"🚀 Khởi động Optuna Hyperparameter Tuning cho {direction_name}...")
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(lambda trial: objective_xgboost(trial, X_train_scaled, y_train, tscv), n_trials=30)
        
        print(f" 🏆 Best CV AUC ({direction_name}): {study.best_value:.4f}")
        best_model = xgb.XGBClassifier(**study.best_params, random_state=42, eval_metric='logloss')
        cv_score = study.best_value
    else:
        best_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric='logloss', verbosity=0
        )
        scores = cross_val_score(best_model, X_train_scaled, y_train, cv=tscv, scoring='roc_auc', n_jobs=-1)
        cv_score = scores.mean()
        
    # Fit final model
    best_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    test_auc = roc_auc_score(y_test, y_proba)
    print(f" 📊 {direction_name} Final Test AUC: {test_auc:.4f}")
    
    # Precision Analysis
    print(f"\n🎯 PRECISION ANALYSIS ({direction_name})")
    print("-" * 65)
    base_win_rate = y_test.mean()
    print(f"Base Win Rate: {base_win_rate:.2%}")
    for thresh in [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.65]:
        mask_prob = y_proba >= thresh
        signals_count = mask_prob.sum()
        if signals_count > 0:
            win_rate = y_test[mask_prob].mean()
            edge = win_rate - base_win_rate
            edge_str = f"+{edge:.2%}" if edge > 0 else f"{edge:.2%}"
            print(f"> {thresh*100:.0f}%          | {signals_count:<16} | {win_rate:.2%}           | {edge_str}")
        else:
            print(f"> {thresh*100:.0f}%          | 0                | N/A              | N/A")
            
    return best_model, scaler, cv_score, test_auc


def train_entry_filter(timeframe: str, tune: bool = False) -> TrainingResult:
    """Train Entry Filter for a specific timeframe (Dual Long/Short)"""
    start_time = time.time()
    print(f"\n{'='*60}")
    print(f"Training Entry Filter (Dual Long/Short) for {timeframe}")
    print('='*60)
    
    data_path = PROCESSED_DIR / f'features_{timeframe}_full.parquet'
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
    
    df = pd.read_parquet(data_path)
    df = filter_top_symbols_by_volume(df, top_n=150, recent_days=180)

    df_cross = df[(df['macd_cross_up'] == 1) | (df['macd_cross_down'] == 1)].copy()
    df_cross = df_cross.dropna(subset=['label'])
    
    # Time-based cutoff
    df_cross_sorted = df_cross.sort_values('timestamp')
    split_idx = int(len(df_cross_sorted) * 0.8)
    cutoff_ts = df_cross_sorted.iloc[split_idx]['timestamp']
    print(f"Time-based cutoff: {cutoff_ts} (80% train / 20% test by crossover time)")
    
    X, y, feature_cols = prepare_entry_data(df)
    
    X['_ts'] = df_cross.loc[X.index, 'timestamp'].values
    X_sorted = X.sort_values('_ts')
    y_sorted = y.loc[X_sorted.index]
    
    train_mask = X_sorted['_ts'] < cutoff_ts
    test_mask = ~train_mask
    
    X_train_full = X_sorted[train_mask].drop(columns=['_ts'])
    X_test_full = X_sorted[test_mask].drop(columns=['_ts'])
    y_train_full = y_sorted[train_mask]
    y_test_full = y_sorted[test_mask]
    
    # record test window for later evaluation
    ts_test = X_sorted.loc[test_mask, '_ts']
    test_start = pd.to_datetime(ts_test.min()) if not ts_test.empty else None
    test_end = pd.to_datetime(ts_test.max()) if not ts_test.empty else None
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Train Long Model
    model_long, scaler_long, cv_long, test_auc_long = train_directional_model(
        'Long', X_train_full['is_bullish_cross'] == 1, X_test_full['is_bullish_cross'] == 1,
        X_train_full, X_test_full, y_train_full, y_test_full, tscv, tune
    )
    
    # Train Short Model
    model_short, scaler_short, cv_short, test_auc_short = train_directional_model(
        'Short', X_train_full['is_bullish_cross'] == 0, X_test_full['is_bullish_cross'] == 0,
        X_train_full, X_test_full, y_train_full, y_test_full, tscv, tune
    )
    
    # Combine results
    avg_cv = (cv_long + cv_short) / 2
    avg_test_auc = (test_auc_long + test_auc_short) / 2
    
    combined_model = {'long': model_long, 'short': model_short}
    combined_scaler = {'long': scaler_long, 'short': scaler_short}
    
    # Save
    model_dir = MODELS_DIR / timeframe
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'entry_filter.joblib'
    
    joblib.dump({
        'model': combined_model,
        'scaler': combined_scaler,
        'feature_names': feature_cols,
        'timeframe': timeframe,
        'is_split_models': True,
        'trained_at': datetime.now().isoformat()
    }, model_path)
    
    training_time = time.time() - start_time
    print(f"\n✓ Saved Dual Models to: {model_path}")
    print(f"  Avg CV: {avg_cv:.4f} | Avg Test AUC: {avg_test_auc:.4f}")
    
    # --- FEATURE IMPORTANCE ---
    plot_and_save_feature_importance(combined_model, feature_cols, timeframe, model_dir)
    
    # --- EVALUATE ON OTHER EXCHANGES ---
    evaluate_on_exchanges(combined_model, combined_scaler, feature_cols, timeframe, 'entry_filter',
                         test_start=test_start, test_end=test_end)
    
    return TrainingResult(
        timeframe=timeframe,
        model_type='entry_filter',
        best_model_name="Dual_XGBoost",
        cv_score=avg_cv,
        test_score=avg_test_auc,
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
