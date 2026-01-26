#!/usr/bin/env python3
"""
Stage 2 ML: Stop Loss Predictor

Predicts optimal SL level based on market conditions.
Uses regression to predict the optimal SL percentage that maximizes risk-adjusted returns.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import joblib
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent.parent / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
MODEL_DIR = Path(__file__).parent / 'models'


# Features for SL prediction
SL_FEATURES = [
    # Volatility (most important for SL)
    'atr_7', 'atr_14',
    'volatility_7', 'volatility_14', 'volatility_21',
    'high_low_range', 'bb_width',
    
    # Price action
    'body_size', 'upper_shadow', 'lower_shadow',
    
    # Trend strength
    'trend_7_21', 'trend_21_50', 'trend_50_200',
    'price_to_sma_7', 'price_to_sma_21', 'price_to_sma_50',
    
    # Momentum
    'rsi_7', 'rsi_14',
    'stoch_k', 'stoch_d',
    'roc_7', 'roc_14',
    
    # MACD
    'macd', 'signal', 'histogram',
    'macd_slope', 'histogram_slope',
    
    # Volume
    'volume_ratio', 'volume_trend',
    
    # Market regime
    'is_trending', 'is_volatile',
    
    # Funding
    'funding_rate_avg'
]


def calculate_optimal_sl(df: pd.DataFrame, max_bars: int = 10) -> pd.DataFrame:
    """
    Calculate optimal SL for each crossover based on max drawdown analysis.
    
    Optimal SL = the minimum SL that would not have been hit before TP
    while still protecting against excessive loss.
    """
    df = df.copy()
    
    # Process each symbol separately
    if 'symbol' in df.columns:
        return df.groupby('symbol', group_keys=False).apply(
            lambda x: _calculate_optimal_sl_single(x.sort_values('timestamp'), max_bars)
        ).reset_index(drop=True)
    else:
        return _calculate_optimal_sl_single(df, max_bars)


def _calculate_optimal_sl_single(df: pd.DataFrame, max_bars: int) -> pd.DataFrame:
    """Calculate optimal SL for single symbol."""
    df = df.copy().reset_index(drop=True)
    n = len(df)
    
    # Initialize columns
    df['optimal_sl_pct'] = np.nan
    df['max_adverse_excursion'] = np.nan  # MAE - worst drawdown before exit
    df['max_favorable_excursion'] = np.nan  # MFE - best profit before exit
    
    # Get crossover indices
    cross_up_mask = df['macd_cross_up'] == 1
    cross_down_mask = df['macd_cross_down'] == 1
    crossover_mask = cross_up_mask | cross_down_mask
    crossover_indices = np.where(crossover_mask)[0]
    crossover_indices = crossover_indices[crossover_indices < n - max_bars]
    
    if len(crossover_indices) == 0:
        return df
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    is_long_arr = cross_up_mask.values
    
    for idx in crossover_indices:
        entry_price = close[idx]
        is_long = is_long_arr[idx]
        
        future_start = idx + 1
        future_end = min(idx + max_bars + 1, n)
        
        if future_end <= future_start:
            continue
        
        future_high = high[future_start:future_end]
        future_low = low[future_start:future_end]
        
        if is_long:
            # For long: MAE = max drawdown, MFE = max profit
            mae = ((entry_price - future_low.min()) / entry_price) if len(future_low) > 0 else 0
            mfe = ((future_high.max() - entry_price) / entry_price) if len(future_high) > 0 else 0
        else:
            # For short: MAE = max upside move, MFE = max downside move
            mae = ((future_high.max() - entry_price) / entry_price) if len(future_high) > 0 else 0
            mfe = ((entry_price - future_low.min()) / entry_price) if len(future_low) > 0 else 0
        
        # Clip to valid range [0, 0.9999] - can't lose more than ~100%
        # Long: price goes to 0 = -100%
        # Short: price can go to infinity but we cap at 100% for practical purposes
        mae = np.clip(mae, 0, 0.9999)
        mfe = np.clip(mfe, 0, 10)  # MFE can be large for long positions
        
        # Optimal SL: slightly above MAE but reasonable
        # If trade was profitable (MFE > MAE), optimal SL = MAE + small buffer
        # If trade was losing (MAE > MFE), optimal SL = some fraction of MAE
        if mfe > mae:
            # Winning trade: SL could have been tight
            optimal_sl = mae * 1.1 + 0.005  # MAE + 10% buffer + 0.5%
        else:
            # Losing trade: need wider SL or better not to take
            optimal_sl = mae * 0.5  # Use 50% of MAE as optimal (would have cut loss earlier)
        
        # DON'T clip here - let model learn true distribution
        # Clipping only happens at prediction time
        
        df.loc[df.index[idx], 'optimal_sl_pct'] = optimal_sl
        df.loc[df.index[idx], 'max_adverse_excursion'] = mae
        df.loc[df.index[idx], 'max_favorable_excursion'] = mfe
    
    return df


def prepare_sl_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Prepare features and targets for SL prediction."""
    
    # Only use crossover rows with valid optimal SL
    df_cross = df[(df['macd_cross_up'] == 1) | (df['macd_cross_down'] == 1)].copy()
    df_cross = df_cross.dropna(subset=['optimal_sl_pct'])
    
    # Filter outliers: SL should be between 0.5% and 20%
    df_cross = df_cross[(df_cross['optimal_sl_pct'] >= 0.005) & (df_cross['optimal_sl_pct'] <= 0.20)]
    print(f"After filtering outliers: {len(df_cross)} samples")
    
    # Get available features
    available_features = [f for f in SL_FEATURES if f in df_cross.columns]
    print(f"Using {len(available_features)} features for SL prediction")
    
    X = df_cross[available_features].copy()
    y = df_cross['optimal_sl_pct']
    
    # Add crossover direction
    X['is_bullish_cross'] = df_cross['macd_cross_up'].values
    
    # Store metadata
    meta_cols = ['timestamp', 'symbol'] if 'symbol' in df_cross.columns else ['timestamp']
    df_meta = df_cross[meta_cols].copy()
    
    # Handle missing values
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    
    print(f"Training samples: {len(X)}")
    print(f"Optimal SL distribution: mean={y.mean():.2%}, std={y.std():.2%}, min={y.min():.2%}, max={y.max():.2%}")
    
    return X, y, df_meta


def train_sl_models(X: pd.DataFrame, y: pd.Series) -> Dict:
    """Train SL prediction models."""
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name} for SL prediction...")
        
        # Cross-validation with negative MAE (higher is better)
        scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
        mae_scores = -scores
        print(f"  CV MAE: {mae_scores.mean():.4f} (+/- {mae_scores.std():.4f})")
        
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
            'cv_mae': mae_scores.mean(),
            'cv_std': mae_scores.std()
        }
    
    return results


def evaluate_sl_model(model, X_test: pd.DataFrame, y_test: pd.Series, name: str = ''):
    """Evaluate SL prediction model."""
    
    y_pred = model.predict(X_test)
    
    # Clip predictions to valid range
    y_pred = np.clip(y_pred, 0.005, 0.10)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"SL Prediction Evaluation: {name}")
    print('='*50)
    print(f"MAE: {mae:.4f} ({mae*100:.2f}%)")
    print(f"RMSE: {rmse:.4f} ({rmse*100:.2f}%)")
    print(f"R²: {r2:.4f}")
    
    # Prediction distribution
    print(f"\nPredicted SL distribution:")
    print(f"  Mean: {y_pred.mean():.2%}")
    print(f"  Std: {y_pred.std():.2%}")
    print(f"  Min: {y_pred.min():.2%}")
    print(f"  Max: {y_pred.max():.2%}")
    
    # Binned accuracy
    print(f"\nPrediction accuracy by SL bucket:")
    bins = [0, 0.02, 0.03, 0.05, 0.10]
    y_test_binned = pd.cut(y_test, bins)
    y_pred_binned = pd.cut(y_pred, bins)
    
    for bin_label in y_test_binned.unique():
        if pd.notna(bin_label):
            mask = y_test_binned == bin_label
            if mask.sum() > 0:
                bin_mae = mean_absolute_error(y_test[mask], y_pred[mask])
                print(f"  {bin_label}: {mask.sum()} samples, MAE={bin_mae:.4f}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'predictions': y_pred
    }


def save_sl_model(model, scaler, feature_names: List[str], filename: str = 'sl_predictor.joblib'):
    """Save SL predictor model."""
    MODEL_DIR.mkdir(exist_ok=True)
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'model_type': 'sl_predictor'
    }
    
    joblib.dump(model_data, MODEL_DIR / filename)
    print(f"✓ SL Predictor saved to {MODEL_DIR / filename}")


class SLPredictor:
    """ML-based Stop Loss predictor."""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.default_sl = 0.02  # 2% default
        
        if model_path and Path(model_path).exists():
            self.load(model_path)
    
    def load(self, model_path: str):
        """Load trained model."""
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data.get('scaler')
        self.feature_names = model_data['feature_names']
    
    def predict(self, features: pd.DataFrame, min_sl: float = 0.005, max_sl: float = 0.10) -> float:
        """
        Predict optimal SL percentage.
        
        Returns:
            SL as decimal (e.g., 0.02 for 2%)
        """
        if self.model is None:
            return self.default_sl
        
        # Ensure all features present
        X = pd.DataFrame()
        for col in self.feature_names:
            if col in features.columns:
                X[col] = features[col].values
            else:
                X[col] = 0
        
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        if self.scaler:
            X = self.scaler.transform(X)
        
        sl_pred = self.model.predict(X)[0]
        return np.clip(sl_pred, min_sl, max_sl)


def train_and_evaluate_sl(data_path: str = None, test_size: float = 0.2, save_best: bool = True) -> Dict:
    """Full SL predictor training pipeline."""
    
    print("="*60)
    print("Stage 2: Stop Loss Predictor Training")
    print("="*60)
    
    # Load data
    if data_path is None:
        data_path = PROCESSED_DIR / 'features_1d_full.parquet'
        if not data_path.exists():
            data_path = PROCESSED_DIR / 'features_1d_test.parquet'
    else:
        data_path = Path(data_path)
    
    if not data_path.exists():
        print(f"Data not found: {data_path}")
        return {}
    
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} rows")
    
    # Calculate optimal SL for each crossover
    print("\nCalculating optimal SL for each trade...")
    df = calculate_optimal_sl(df)
    
    # Prepare data
    X, y, df_meta = prepare_sl_training_data(df)
    
    if len(X) == 0:
        print("No training data!")
        return {}
    
    # Time-based split
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\nTrain: {len(X_train)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    # Train models
    results = train_sl_models(X_train_scaled, y_train)
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Test Set Evaluation")
    print("="*60)
    
    test_results = {}
    for name, res in results.items():
        eval_res = evaluate_sl_model(res['model'], X_test_scaled, y_test, name)
        test_results[name] = eval_res
        results[name]['test_mae'] = eval_res['mae']
        results[name]['test_r2'] = eval_res['r2']
    
    # Find best model (lowest MAE)
    best_model_name = min(results.keys(), key=lambda k: results[k].get('test_mae', float('inf')))
    best_model = results[best_model_name]['model']
    
    # Save best model
    if save_best:
        save_sl_model(best_model, scaler, list(X.columns), 'sl_predictor.joblib')
    
    # Summary
    print("\n" + "="*60)
    print("SL Predictor Training Summary")
    print("="*60)
    print(f"Best Model: {best_model_name}")
    print(f"CV MAE: {results[best_model_name]['cv_mae']:.4f}")
    print(f"Test MAE: {results[best_model_name]['test_mae']:.4f}")
    print(f"Test R²: {results[best_model_name]['test_r2']:.4f}")
    
    print("\nAll Models (sorted by Test MAE):")
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('test_mae', float('inf')))
    for name, res in sorted_results:
        print(f"  {name:20s}: CV MAE={res['cv_mae']:.4f}, Test MAE={res['test_mae']:.4f}, R²={res['test_r2']:.4f}")
    
    return {
        'best_model_name': best_model_name,
        'best_model': best_model,
        'scaler': scaler,
        'feature_names': list(X.columns),
        'results': results
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SL Predictor (Stage 2)')
    parser.add_argument('--data', type=str, default=None, help='Path to processed data')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    
    args = parser.parse_args()
    
    train_and_evaluate_sl(data_path=args.data, test_size=args.test_size)
