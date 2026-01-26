#!/usr/bin/env python3
"""
Stage 4 ML: Bars to Peak Predictor

Predicts how many bars until the trade reaches its maximum favorable excursion (MFE).
This helps with:
1. Time-based exits - know when to expect the move to complete
2. Trailing stop timing - when to tighten SL
3. Opportunity cost - skip trades that take too long
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
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


# Features for bars-to-peak prediction
# Focus on momentum exhaustion and trend maturity signals
BARS_FEATURES = [
    # Momentum (key for timing)
    'rsi_7', 'rsi_14',
    'stoch_k', 'stoch_d',
    'roc_7', 'roc_14', 'roc_21',
    'macd', 'signal', 'histogram',
    'macd_slope', 'histogram_slope', 'macd_acceleration',
    
    # Trend maturity (how extended is the move?)
    'trend_7_21', 'trend_21_50', 'trend_50_200',
    'price_to_sma_7', 'price_to_sma_21', 'price_to_sma_50', 'price_to_sma_200',
    'bars_since_cross_up', 'bars_since_cross_down',
    
    # Volume (exhaustion signals)
    'volume_ratio', 'volume_trend', 'obv_trend',
    
    # Volatility (affects speed of moves)
    'atr_7', 'atr_14',
    'volatility_7', 'volatility_14', 'volatility_21',
    'bb_width', 'bb_position',
    
    # Price action
    'high_low_range', 'body_size',
    'upper_shadow', 'lower_shadow',
    'returns', 'log_returns',
    
    # Market regime
    'is_trending', 'is_volatile',
    
    # Funding
    'funding_rate_avg'
]


def calculate_bars_to_peak(df: pd.DataFrame, max_bars: int = 10) -> pd.DataFrame:
    """
    Calculate bars to reach MFE (peak for long, trough for short).
    """
    df = df.copy()
    
    if 'symbol' in df.columns:
        return df.groupby('symbol', group_keys=False).apply(
            lambda x: _calculate_bars_to_peak_single(x.sort_values('timestamp'), max_bars)
        ).reset_index(drop=True)
    else:
        return _calculate_bars_to_peak_single(df, max_bars)


def _calculate_bars_to_peak_single(df: pd.DataFrame, max_bars: int) -> pd.DataFrame:
    """Calculate bars to peak for single symbol."""
    df = df.copy().reset_index(drop=True)
    n = len(df)
    
    # Initialize columns
    df['bars_to_peak'] = np.nan
    df['peak_pct'] = np.nan  # The MFE value at peak
    df['early_exit_pct'] = np.nan  # Profit if exited at bar 3
    df['late_penalty_pct'] = np.nan  # How much lost by waiting after peak
    
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
        future_close = close[future_start:future_end]
        
        if is_long:
            # For long: peak is highest high
            mfe_values = (future_high - entry_price) / entry_price
            mfe = mfe_values.max()
            mfe_bar = mfe_values.argmax() + 1  # 1-indexed
            
            # Early exit at bar 3
            if len(future_close) >= 3:
                early_exit = (future_close[2] - entry_price) / entry_price
            else:
                early_exit = 0
            
            # Late penalty: profit lost by waiting after peak
            if mfe_bar < len(future_close):
                final_profit = (future_close[-1] - entry_price) / entry_price
                late_penalty = mfe - final_profit
            else:
                late_penalty = 0
        else:
            # For short: peak is lowest low
            mfe_values = (entry_price - future_low) / entry_price
            mfe = mfe_values.max()
            mfe_bar = mfe_values.argmax() + 1
            
            if len(future_close) >= 3:
                early_exit = (entry_price - future_close[2]) / entry_price
            else:
                early_exit = 0
            
            if mfe_bar < len(future_close):
                final_profit = (entry_price - future_close[-1]) / entry_price
                late_penalty = mfe - final_profit
            else:
                late_penalty = 0
        
        df.loc[df.index[idx], 'bars_to_peak'] = mfe_bar
        df.loc[df.index[idx], 'peak_pct'] = mfe
        df.loc[df.index[idx], 'early_exit_pct'] = early_exit
        df.loc[df.index[idx], 'late_penalty_pct'] = late_penalty
    
    return df


def prepare_bars_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Prepare features and targets for bars-to-peak prediction."""
    
    df_cross = df[(df['macd_cross_up'] == 1) | (df['macd_cross_down'] == 1)].copy()
    df_cross = df_cross.dropna(subset=['bars_to_peak'])
    
    # Filter: bars_to_peak should be valid (1 to max_bars)
    df_cross = df_cross[(df_cross['bars_to_peak'] >= 1) & (df_cross['bars_to_peak'] <= 10)]
    print(f"After filtering: {len(df_cross)} samples")
    
    available_features = [f for f in BARS_FEATURES if f in df_cross.columns]
    print(f"Using {len(available_features)} features for bars-to-peak prediction")
    
    X = df_cross[available_features].copy()
    y = df_cross['bars_to_peak']
    
    # Add crossover direction
    X['is_bullish_cross'] = df_cross['macd_cross_up'].values
    
    # Store metadata
    meta_cols = ['timestamp', 'symbol'] if 'symbol' in df_cross.columns else ['timestamp']
    df_meta = df_cross[meta_cols].copy()
    
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    
    print(f"Training samples: {len(X)}")
    print(f"Bars to peak distribution: mean={y.mean():.1f}, std={y.std():.1f}, min={y.min():.0f}, max={y.max():.0f}")
    
    # Distribution analysis
    print("\nBars distribution:")
    for bar in range(1, 11):
        pct = (y == bar).mean() * 100
        print(f"  Bar {bar}: {pct:.1f}%")
    
    return X, y, df_meta


def train_bars_models(X: pd.DataFrame, y: pd.Series) -> Dict:
    """Train bars-to-peak prediction models."""
    
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
        print(f"\nTraining {name} for bars-to-peak prediction...")
        
        scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
        mae_scores = -scores
        print(f"  CV MAE: {mae_scores.mean():.2f} bars (+/- {mae_scores.std():.2f})")
        
        model.fit(X, y)
        
        if hasattr(model, 'feature_importances_'):
            importance = pd.Series(
                model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)
            print(f"  Top 5 features: {list(importance.head().index)}")
        
        results[name] = {
            'model': model,
            'cv_mae': mae_scores.mean(),
            'cv_std': mae_scores.std()
        }
    
    return results


def evaluate_bars_model(model, X_test: pd.DataFrame, y_test: pd.Series, name: str = ''):
    """Evaluate bars-to-peak prediction model."""
    
    y_pred = model.predict(X_test)
    # Round to nearest bar (discrete)
    y_pred_rounded = np.round(y_pred).clip(1, 10)
    
    mae = mean_absolute_error(y_test, y_pred)
    mae_rounded = mean_absolute_error(y_test, y_pred_rounded)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Accuracy within 1 bar
    within_1 = (np.abs(y_pred_rounded - y_test) <= 1).mean()
    # Accuracy within 2 bars
    within_2 = (np.abs(y_pred_rounded - y_test) <= 2).mean()
    
    print(f"\n{'='*50}")
    print(f"Bars-to-Peak Evaluation: {name}")
    print('='*50)
    print(f"MAE: {mae:.2f} bars")
    print(f"MAE (rounded): {mae_rounded:.2f} bars")
    print(f"RMSE: {rmse:.2f} bars")
    print(f"R²: {r2:.4f}")
    print(f"\nTiming Accuracy:")
    print(f"  Within 1 bar: {within_1:.1%}")
    print(f"  Within 2 bars: {within_2:.1%}")
    
    print(f"\nPredicted distribution:")
    print(f"  Mean: {y_pred.mean():.1f} bars")
    print(f"  Std: {y_pred.std():.1f} bars")
    
    # Confusion by actual bar
    print(f"\nPrediction accuracy by actual bar:")
    for bar in range(1, 11):
        mask = y_test == bar
        if mask.sum() > 10:  # Only show if enough samples
            bar_mae = np.abs(y_pred[mask] - bar).mean()
            print(f"  Bar {bar}: {mask.sum()} samples, MAE={bar_mae:.2f}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'within_1_bar': within_1,
        'within_2_bars': within_2,
        'predictions': y_pred
    }


def save_bars_model(model, scaler, feature_names: List[str], filename: str = 'bars_predictor.joblib'):
    """Save bars-to-peak predictor model."""
    MODEL_DIR.mkdir(exist_ok=True)
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'model_type': 'bars_predictor'
    }
    
    joblib.dump(model_data, MODEL_DIR / filename)
    print(f"✓ Bars Predictor saved to {MODEL_DIR / filename}")


class BarsPredictor:
    """ML-based Bars to Peak predictor."""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.default_bars = 5  # Default: expect peak at bar 5
        
        if model_path and Path(model_path).exists():
            self.load(model_path)
    
    def load(self, model_path: str):
        """Load trained model."""
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data.get('scaler')
        self.feature_names = model_data['feature_names']
    
    def predict(self, features: pd.DataFrame, min_bars: int = 1, max_bars: int = 10) -> int:
        """
        Predict bars to peak.
        
        Returns:
            Expected bar number when MFE will occur (1-10)
        """
        if self.model is None:
            return self.default_bars
        
        X = pd.DataFrame()
        for col in self.feature_names:
            if col in features.columns:
                X[col] = features[col].values
            else:
                X[col] = 0
        
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        if self.scaler:
            X = self.scaler.transform(X)
        
        bars_pred = self.model.predict(X)[0]
        return int(np.round(np.clip(bars_pred, min_bars, max_bars)))


def train_and_evaluate_bars(data_path: str = None, test_size: float = 0.2, save_best: bool = True) -> Dict:
    """Full bars-to-peak predictor training pipeline."""
    
    print("="*60)
    print("Stage 4: Bars to Peak Predictor Training")
    print("="*60)
    
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
    
    print("\nCalculating bars to peak for each trade...")
    df = calculate_bars_to_peak(df)
    
    X, y, df_meta = prepare_bars_training_data(df)
    
    if len(X) == 0:
        print("No training data!")
        return {}
    
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\nTrain: {len(X_train)} samples")
    print(f"Test: {len(X_test)} samples")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    results = train_bars_models(X_train_scaled, y_train)
    
    print("\n" + "="*60)
    print("Test Set Evaluation")
    print("="*60)
    
    test_results = {}
    for name, res in results.items():
        eval_res = evaluate_bars_model(res['model'], X_test_scaled, y_test, name)
        test_results[name] = eval_res
        results[name]['test_mae'] = eval_res['mae']
        results[name]['test_r2'] = eval_res['r2']
        results[name]['within_1_bar'] = eval_res['within_1_bar']
    
    best_model_name = min(results.keys(), key=lambda k: results[k].get('test_mae', float('inf')))
    best_model = results[best_model_name]['model']
    
    if save_best:
        save_bars_model(best_model, scaler, list(X.columns), 'bars_predictor.joblib')
    
    print("\n" + "="*60)
    print("Bars Predictor Training Summary")
    print("="*60)
    print(f"Best Model: {best_model_name}")
    print(f"CV MAE: {results[best_model_name]['cv_mae']:.2f} bars")
    print(f"Test MAE: {results[best_model_name]['test_mae']:.2f} bars")
    print(f"Within 1 bar: {results[best_model_name]['within_1_bar']:.1%}")
    
    print("\nAll Models (sorted by Test MAE):")
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('test_mae', float('inf')))
    for name, res in sorted_results:
        print(f"  {name:20s}: MAE={res['test_mae']:.2f}, Within 1 bar={res['within_1_bar']:.1%}")
    
    return {
        'best_model_name': best_model_name,
        'best_model': best_model,
        'scaler': scaler,
        'feature_names': list(X.columns),
        'results': results
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Bars to Peak Predictor (Stage 4)')
    parser.add_argument('--data', type=str, default=None, help='Path to processed data')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    
    args = parser.parse_args()
    
    train_and_evaluate_bars(data_path=args.data, test_size=args.test_size)
