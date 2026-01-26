#!/usr/bin/env python3
"""
ML Model Training for MACD Entry Filter

Trains XGBoost, LightGBM, RandomForest, and GradientBoosting models
to predict good vs bad MACD crossover entries.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import joblib
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, f1_score, precision_score, recall_score
)
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Import feature helper from data pipeline
try:
    from data_pipeline import get_feature_columns
except ImportError:
    from ml.data_pipeline import get_feature_columns

DATA_DIR = Path(__file__).parent.parent / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
MODEL_DIR = Path(__file__).parent / 'models'


# Default features (fallback if get_feature_columns unavailable)
DEFAULT_FEATURE_COLUMNS = [
    # Price
    'returns', 'log_returns', 'high_low_range', 'body_size',
    'upper_shadow', 'lower_shadow',
    
    # Trend
    'price_to_sma_7', 'price_to_sma_14', 'price_to_sma_21',
    'price_to_sma_50', 'price_to_sma_100', 'price_to_sma_200',
    'trend_7_21', 'trend_21_50', 'trend_50_200',
    
    # MACD
    'macd', 'signal', 'histogram',
    'macd_slope', 'signal_slope', 'histogram_slope',
    'macd_acceleration',
    'bars_since_cross_up', 'bars_since_cross_down',
    
    # Volatility
    'atr_7', 'atr_14',
    'volatility_7', 'volatility_14', 'volatility_21',
    'bb_width', 'bb_position',
    
    # Momentum
    'rsi_7', 'rsi_14',
    'stoch_k', 'stoch_d',
    'roc_7', 'roc_14', 'roc_21',
    
    # Volume
    'volume_ratio', 'volume_trend', 'obv_trend',
    
    # Market regime
    'is_trending', 'is_volatile',
    
    # Funding (if available)
    'funding_rate_avg', 'funding_rate_sum'
]


def prepare_training_data(df: pd.DataFrame, feature_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Prepare features and labels for training.
    
    Returns:
        X: Feature matrix
        y: Labels
        df_meta: Metadata (timestamp, symbol) for analysis
    """
    # Only use rows with crossovers
    df_cross = df[(df['macd_cross_up'] == 1) | (df['macd_cross_down'] == 1)].copy()
    
    # Remove rows with NaN labels
    df_cross = df_cross.dropna(subset=['label'])
    
    # Get feature columns (use dynamic function or fallback)
    if feature_cols is None:
        try:
            feature_cols = get_feature_columns(df_cross)
        except:
            feature_cols = DEFAULT_FEATURE_COLUMNS
    
    # Filter to available features
    available_features = [f for f in feature_cols if f in df_cross.columns]
    print(f"Using {len(available_features)} features")
    
    X = df_cross[available_features].copy()
    y = df_cross['label'].astype(int)
    
    # Add crossover direction as feature
    X['is_bullish_cross'] = df_cross['macd_cross_up'].values
    
    # Store metadata for analysis
    meta_cols = ['timestamp', 'symbol'] if 'symbol' in df_cross.columns else ['timestamp']
    df_meta = df_cross[meta_cols].copy()
    
    # Handle missing values
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    print(f"Training samples: {len(X)}")
    print(f"Label distribution: {y.value_counts().to_dict()}")
    print(f"Win rate: {y.mean():.2%}")
    
    return X, y, df_meta


def get_model_configs(tune_hyperparams: bool = False) -> Dict:
    """
    Get model configurations.
    
    Args:
        tune_hyperparams: If True, return param grids for tuning
    """
    base_models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,  # Adjust if imbalanced
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=42,
            verbose=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
    }
    
    if not tune_hyperparams:
        return base_models
    
    # Param grids for tuning
    param_grids = {
        'XGBoost': {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_weight': [1, 3, 5]
        },
        'LightGBM': {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 50, 70],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
    }
    
    return base_models, param_grids


def train_models(X: pd.DataFrame, y: pd.Series, tune_hyperparams: bool = False) -> Dict:
    """
    Train multiple models and compare.
    
    Args:
        X: Feature matrix
        y: Labels
        tune_hyperparams: If True, perform hyperparameter tuning
    """
    # Time series split (no shuffling!)
    tscv = TimeSeriesSplit(n_splits=5)
    
    models = get_model_configs(tune_hyperparams=False)
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Hyperparameter tuning for XGBoost/LightGBM
        if tune_hyperparams and name in ['XGBoost', 'LightGBM']:
            _, param_grids = get_model_configs(tune_hyperparams=True)
            print(f"  Tuning hyperparameters...")
            search = RandomizedSearchCV(
                model, param_grids[name],
                n_iter=20, cv=tscv, scoring='roc_auc',
                random_state=42, n_jobs=-1
            )
            search.fit(X, y)
            model = search.best_estimator_
            print(f"  Best params: {search.best_params_}")
            scores = [search.best_score_]  # Use search score
        else:
            # Standard cross-validation
            scores = cross_val_score(model, X, y, cv=tscv, scoring='roc_auc')
            print(f"  CV ROC-AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
            
            # Train on full data
            model.fit(X, y)
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance = pd.Series(
                model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)
            print(f"  Top 5 features: {importance.head().to_dict()}")
        
        results[name] = {
            'model': model,
            'cv_scores': scores,
            'cv_mean': scores.mean(),
            'cv_std': scores.std()
        }
    
    return results


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, name: str = ''):
    """Detailed model evaluation"""
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\n{'='*50}")
    print(f"Evaluation: {name}")
    print('='*50)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Bad Entry', 'Good Entry']))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\nROC-AUC: {roc_auc:.4f}")
    
    # Precision at different thresholds
    print("\nPrecision at different confidence thresholds:")
    for threshold in [0.5, 0.6, 0.7, 0.8]:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        if y_pred_thresh.sum() > 0:
            precision = (y_test[y_pred_thresh == 1] == 1).mean()
            coverage = y_pred_thresh.mean()
            print(f"  Threshold {threshold}: Precision={precision:.2%}, Coverage={coverage:.2%}")
    
    return {
        'roc_auc': roc_auc,
        'f1': f1_score(y_test, y_pred),
        'predictions': y_pred,
        'probabilities': y_proba
    }


def save_model(model, scaler, feature_names: List[str], filename: str):
    """Save model and metadata"""
    MODEL_DIR.mkdir(exist_ok=True)
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names
    }
    
    joblib.dump(model_data, MODEL_DIR / filename)
    print(f"✓ Model saved to {MODEL_DIR / filename}")


def load_model(filename: str):
    """Load model and metadata"""
    model_data = joblib.load(MODEL_DIR / filename)
    return model_data['model'], model_data['scaler'], model_data['feature_names']


class EntryFilter:
    """ML-based entry filter for MACD crossover signals"""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.threshold = 0.5
        
        if model_path and Path(model_path).exists():
            self.load(model_path)
    
    def load(self, model_path: str):
        """Load trained model"""
        self.model, self.scaler, self.feature_names = load_model(model_path)
    
    def predict(self, features: pd.DataFrame) -> Tuple[bool, float]:
        """
        Predict if entry is good
        
        Returns:
            (should_enter, confidence)
        """
        if self.model is None:
            return True, 0.5  # No model, always enter
        
        # Ensure all features present
        X = features[self.feature_names].copy()
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Scale
        if self.scaler:
            X = self.scaler.transform(X)
        
        # Predict
        proba = self.model.predict_proba(X)[0, 1]
        should_enter = proba >= self.threshold
        
        return should_enter, proba
    
    def set_threshold(self, threshold: float):
        """Set confidence threshold for entry"""
        self.threshold = threshold


def train_and_evaluate(
    data_path: str = None,
    tune_hyperparams: bool = False,
    test_size: float = 0.2,
    save_best: bool = True
) -> Dict:
    """
    Full training and evaluation pipeline.
    
    Args:
        data_path: Path to processed parquet file
        tune_hyperparams: Whether to tune hyperparameters
        test_size: Fraction of data for testing (time-based split)
        save_best: Whether to save the best model
    
    Returns:
        Dictionary with all results
    """
    print("="*60)
    print("MACD Entry Filter - ML Training")
    print("="*60)
    
    # Load data
    if data_path is None:
        data_path = PROCESSED_DIR / 'features_1d_full.parquet'
    else:
        data_path = Path(data_path)
    
    if not data_path.exists():
        print(f"Data not found: {data_path}")
        print("Run data_pipeline.py first!")
        return {}
    
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} rows")
    
    # Prepare data
    X, y, df_meta = prepare_training_data(df)
    
    # Time-based split (no shuffling!)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    meta_test = df_meta.iloc[split_idx:]
    
    print(f"\nTrain: {len(X_train)} samples ({y_train.mean():.2%} win rate)")
    print(f"Test: {len(X_test)} samples ({y_test.mean():.2%} win rate)")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for feature names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    # Train models
    results = train_models(X_train_scaled, y_train, tune_hyperparams=tune_hyperparams)
    
    # Evaluate all models on test set
    print("\n" + "="*60)
    print("Test Set Evaluation")
    print("="*60)
    
    test_results = {}
    for name, res in results.items():
        model = res['model']
        eval_res = evaluate_model(model, X_test_scaled, y_test, name)
        test_results[name] = eval_res
        results[name]['test_roc_auc'] = eval_res['roc_auc']
        results[name]['test_f1'] = eval_res['f1']
    
    # Find best model (by test ROC-AUC)
    best_model_name = max(results.keys(), key=lambda k: results[k].get('test_roc_auc', 0))
    best_model = results[best_model_name]['model']
    
    # Save best model
    if save_best:
        save_model(best_model, scaler, list(X.columns), 'entry_filter.joblib')
    
    # Summary
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Best Model: {best_model_name}")
    print(f"CV ROC-AUC: {results[best_model_name]['cv_mean']:.4f}")
    print(f"Test ROC-AUC: {results[best_model_name]['test_roc_auc']:.4f}")
    print(f"Test F1: {results[best_model_name]['test_f1']:.4f}")
    
    # All models comparison
    print("\nAll Models Comparison (sorted by Test ROC-AUC):")
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('test_roc_auc', 0), reverse=True)
    for name, res in sorted_results:
        cv_score = res.get('cv_mean', 0)
        cv_std = res.get('cv_std', 0)
        test_auc = res.get('test_roc_auc', 0)
        test_f1 = res.get('test_f1', 0)
        print(f"  {name:20s}: CV={cv_score:.4f}(±{cv_std:.4f}), Test AUC={test_auc:.4f}, F1={test_f1:.4f}")
    
    # Trading metrics at different thresholds
    print("\n" + "="*60)
    print("Trading Performance by Confidence Threshold")
    print("="*60)
    
    y_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    for threshold in [0.5, 0.55, 0.6, 0.65, 0.7]:
        mask = y_proba >= threshold
        if mask.sum() > 0:
            win_rate = y_test[mask].mean()
            coverage = mask.mean()
            n_trades = mask.sum()
            print(f"  Threshold {threshold:.2f}: {n_trades:4d} trades, Win={win_rate:.2%}, Coverage={coverage:.2%}")
    
    return {
        'best_model_name': best_model_name,
        'best_model': best_model,
        'scaler': scaler,
        'feature_names': list(X.columns),
        'results': results,
        'test_results': test_results
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MACD Entry Filter')
    parser.add_argument('--data', type=str, default=None, help='Path to processed data')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (default: 0.2)')
    
    args = parser.parse_args()
    
    train_and_evaluate(
        data_path=args.data,
        tune_hyperparams=args.tune,
        test_size=args.test_size
    )
