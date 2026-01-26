#!/usr/bin/env python3
"""
Stage 3 ML: Take Profit Predictor (v2 - Risk/Reward Ratio based)

IMPROVEMENTS over v1:
1. Predict RR Ratio instead of raw TP% (more stable target)
2. Cap extreme outliers (max 30% TP - can't learn black swan events)
3. Add momentum features: ADX, Volume Spike, RSI Slope
4. Use SL from Stage 2 to calculate TP = SL * RR_Ratio

The key insight: TP is tied to risk. Better to predict "how many R can this trade achieve?"
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

# Maximum TP to learn (cap outliers - can't learn 496% pumps)
MAX_TP_PCT = 0.30  # 30% max - reasonable for 10-bar window

# Features for TP/RR prediction (momentum & trend focused)
TP_FEATURES = [
    # Core Momentum (PRIMARY - drives TP potential)
    'rsi_7', 'rsi_14',
    'rsi_slope',  # NEW: Rate of RSI change
    'stoch_k', 'stoch_d',
    'roc_7', 'roc_14', 'roc_21',
    'macd', 'signal', 'histogram',
    'macd_slope', 'histogram_slope', 'macd_acceleration',
    
    # Trend Strength (how far can it go?)
    'adx',  # NEW: Average Directional Index
    'trend_7_21', 'trend_21_50', 'trend_50_200',
    'price_to_sma_7', 'price_to_sma_21', 'price_to_sma_50', 'price_to_sma_200',
    
    # Volume (exhaustion & confirmation)
    'volume_ratio', 'volume_trend', 'obv_trend',
    'volume_spike',  # NEW: Is this a volume breakout?
    
    # Volatility (affects TP distance - via ATR)
    'atr_7', 'atr_14',
    'volatility_7', 'volatility_14', 'volatility_21',
    'bb_width', 'bb_position',
    
    # Price action
    'high_low_range', 'body_size',
    'upper_shadow', 'lower_shadow',
    'returns', 'log_returns',
    
    # Market regime
    'is_trending', 'is_volatile',
    
    # Funding (sentiment)
    'funding_rate'
]


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add advanced momentum features that help predict TP potential.
    
    These features capture "how strong is the push" - critical for TP prediction.
    """
    df = df.copy()
    
    # 1. ADX (Average Directional Index) - Trend strength 0-100
    # Higher ADX = stronger trend = higher TP potential
    if 'atr_14' in df.columns:
        # +DI and -DI
        df['plus_dm'] = df['high'].diff()
        df['minus_dm'] = -df['low'].diff()
        
        # Keep only positive values and handle comparison
        df['plus_dm'] = df['plus_dm'].where(
            (df['plus_dm'] > df['minus_dm']) & (df['plus_dm'] > 0), 0
        )
        df['minus_dm'] = df['minus_dm'].where(
            (df['minus_dm'] > df['plus_dm']) & (df['minus_dm'] > 0), 0
        )
        
        # Smooth with EMA
        df['plus_di'] = 100 * (df['plus_dm'].ewm(span=14, adjust=False).mean() / df['atr_14'])
        df['minus_di'] = 100 * (df['minus_dm'].ewm(span=14, adjust=False).mean() / df['atr_14'])
        
        # DX and ADX
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'] + 1e-10)
        df['adx'] = df['dx'].ewm(span=14, adjust=False).mean()
        
        # Clean up intermediate columns
        df = df.drop(columns=['plus_dm', 'minus_dm', 'plus_di', 'minus_di', 'dx'], errors='ignore')
    else:
        df['adx'] = 25  # Default neutral ADX
    
    # 2. RSI Slope (momentum of momentum)
    # Rising RSI = acceleration = higher TP
    if 'rsi_14' in df.columns:
        df['rsi_slope'] = df['rsi_14'].diff(3) / 3  # 3-bar slope
    else:
        df['rsi_slope'] = 0
    
    # 3. Volume Spike (is this a breakout?)
    # Volume > 2x average = likely breakout = higher TP
    if 'volume' in df.columns:
        vol_sma = df['volume'].rolling(20).mean()
        df['volume_spike'] = (df['volume'] / vol_sma).clip(0, 5)  # Cap at 5x
    else:
        df['volume_spike'] = 1
    
    # Replace NaN/inf
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    return df


def calculate_optimal_tp(df: pd.DataFrame, max_bars: int = 10) -> pd.DataFrame:
    """
    Calculate optimal TP for each crossover based on MFE (Max Favorable Excursion).
    
    Key change: Cap TP at MAX_TP_PCT to remove extreme outliers.
    """
    df = df.copy()
    
    if 'symbol' in df.columns:
        return df.groupby('symbol', group_keys=False).apply(
            lambda x: _calculate_optimal_tp_single(x.sort_values('timestamp'), max_bars)
        ).reset_index(drop=True)
    else:
        return _calculate_optimal_tp_single(df, max_bars)


def _calculate_optimal_tp_single(df: pd.DataFrame, max_bars: int) -> pd.DataFrame:
    """Calculate optimal TP for single symbol with outlier capping."""
    df = df.copy().reset_index(drop=True)
    n = len(df)
    
    # Initialize columns
    df['optimal_tp_pct'] = np.nan
    df['mfe'] = np.nan
    df['mfe_bar'] = np.nan
    df['trend_continuation'] = np.nan
    df['rr_ratio'] = np.nan  # NEW: Risk/Reward ratio
    
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
    
    # Get SL from dynamic labeling (if available)
    sl_used = df['sl_pct_used'].values if 'sl_pct_used' in df.columns else None
    
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
            mfe_values = (future_high - entry_price) / entry_price
            mfe = mfe_values.max()
            mfe_bar = mfe_values.argmax() + 1
            trend_cont = (future_close[-1] - entry_price) / entry_price if len(future_close) > 0 else 0
        else:
            mfe_values = (entry_price - future_low) / entry_price
            mfe = mfe_values.max()
            mfe_bar = mfe_values.argmax() + 1
            trend_cont = (entry_price - future_close[-1]) / entry_price if len(future_close) > 0 else 0
        
        # Optimal TP = 75% of MFE, CAPPED at MAX_TP_PCT
        # This removes extreme outliers that the model can't learn
        optimal_tp = min(mfe * 0.75, MAX_TP_PCT)
        
        # Calculate RR Ratio if SL available
        if sl_used is not None and not np.isnan(sl_used[idx]) and sl_used[idx] > 0:
            rr_ratio = optimal_tp / sl_used[idx]
        else:
            rr_ratio = optimal_tp / 0.02  # Default 2% SL assumption
        
        df.loc[df.index[idx], 'optimal_tp_pct'] = optimal_tp
        df.loc[df.index[idx], 'mfe'] = min(mfe, MAX_TP_PCT * 1.5)  # Also cap MFE for reference
        df.loc[df.index[idx], 'mfe_bar'] = mfe_bar
        df.loc[df.index[idx], 'trend_continuation'] = trend_cont
        df.loc[df.index[idx], 'rr_ratio'] = min(rr_ratio, 10)  # Cap RR at 10:1
    
    return df


def prepare_tp_training_data(df: pd.DataFrame, predict_rr: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Prepare features and targets for TP prediction.
    
    Args:
        df: DataFrame with features and labels
        predict_rr: If True, predict RR Ratio. If False, predict TP% directly.
    """
    # Add momentum features first
    df = add_momentum_features(df)
    
    df_cross = df[(df['macd_cross_up'] == 1) | (df['macd_cross_down'] == 1)].copy()
    df_cross = df_cross.dropna(subset=['optimal_tp_pct'])
    
    # Filter: TP should be at least 0.5% (realistic minimum)
    df_cross = df_cross[df_cross['optimal_tp_pct'] >= 0.005]
    
    # Additional filter: Remove extreme outliers (already capped in calculate_optimal_tp)
    df_cross = df_cross[df_cross['optimal_tp_pct'] <= MAX_TP_PCT]
    
    print(f"After filtering (0.5% <= TP <= {MAX_TP_PCT*100:.0f}%): {len(df_cross)} samples")
    
    available_features = [f for f in TP_FEATURES if f in df_cross.columns]
    print(f"Using {len(available_features)} features for TP prediction")
    
    X = df_cross[available_features].copy()
    
    # Choose target: RR Ratio or TP%
    if predict_rr and 'rr_ratio' in df_cross.columns:
        y = df_cross['rr_ratio'].clip(0.5, 10)  # RR between 0.5:1 and 10:1
        print(f"Target: RR Ratio (mean={y.mean():.2f}, std={y.std():.2f})")
    else:
        y = df_cross['optimal_tp_pct']
        print(f"Target: TP% (mean={y.mean():.2%}, std={y.std():.2%})")
    
    # Add crossover direction
    X['is_bullish_cross'] = df_cross['macd_cross_up'].values
    
    # Store metadata
    meta_cols = ['timestamp', 'symbol'] if 'symbol' in df_cross.columns else ['timestamp']
    df_meta = df_cross[meta_cols].copy()
    
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    
    print(f"Training samples: {len(X)}")
    print(f"Target distribution: mean={y.mean():.4f}, std={y.std():.4f}, min={y.min():.4f}, max={y.max():.4f}")
    
    return X, y, df_meta


def train_tp_models(X: pd.DataFrame, y: pd.Series) -> Dict:
    """Train TP prediction models."""
    
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
        print(f"\nTraining {name} for TP prediction...")
        
        scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
        mae_scores = -scores
        print(f"  CV MAE: {mae_scores.mean():.4f} (+/- {mae_scores.std():.4f})")
        
        model.fit(X, y)
        
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


def evaluate_tp_model(model, X_test: pd.DataFrame, y_test: pd.Series, name: str = '', is_rr_target: bool = False):
    """
    Evaluate TP/RR prediction model.
    
    Args:
        is_rr_target: If True, target is RR Ratio (0.5-10). If False, target is TP% (0.01-0.30)
    """
    y_pred = model.predict(X_test)
    
    # Clip predictions to valid range
    if is_rr_target:
        y_pred = np.clip(y_pred, 0.5, 10)  # RR: 0.5:1 to 10:1
    else:
        y_pred = np.clip(y_pred, 0.01, MAX_TP_PCT)  # TP%: 1% to MAX_TP_PCT
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"TP/RR Prediction Evaluation: {name}")
    print('='*50)
    
    if is_rr_target:
        print(f"MAE: {mae:.3f} RR")
        print(f"RMSE: {rmse:.3f} RR")
    else:
        print(f"MAE: {mae:.4f} ({mae*100:.2f}%)")
        print(f"RMSE: {rmse:.4f} ({rmse*100:.2f}%)")
    print(f"R²: {r2:.4f}")
    
    print(f"\nPredicted distribution:")
    if is_rr_target:
        print(f"  Mean: {y_pred.mean():.2f} RR")
        print(f"  Std: {y_pred.std():.2f}")
    else:
        print(f"  Mean: {y_pred.mean():.2%}")
        print(f"  Std: {y_pred.std():.2%}")
    print(f"  Min: {y_pred.min():.4f}")
    print(f"  Max: {y_pred.max():.4f}")
    
    # Binned accuracy
    print(f"\nPrediction accuracy by bucket:")
    if is_rr_target:
        bins = [0, 1, 2, 3, 5, 10]
        labels_text = ['<1:1', '1-2:1', '2-3:1', '3-5:1', '>5:1']
    else:
        bins = [0, 0.05, 0.10, 0.15, 0.20, 0.30]
        labels_text = ['<5%', '5-10%', '10-15%', '15-20%', '>20%']
    
    y_test_binned = pd.cut(y_test, bins)
    
    for bin_label in sorted(y_test_binned.unique(), key=lambda x: x.left if pd.notna(x) else 0):
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


def save_tp_model(model, scaler, feature_names: List[str], filename: str = 'tp_predictor.joblib'):
    """Save TP predictor model."""
    MODEL_DIR.mkdir(exist_ok=True)
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'model_type': 'tp_predictor'
    }
    
    joblib.dump(model_data, MODEL_DIR / filename)
    print(f"✓ TP Predictor saved to {MODEL_DIR / filename}")


class TPPredictor:
    """ML-based Take Profit predictor."""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.default_tp = 0.03  # 3% default
        
        if model_path and Path(model_path).exists():
            self.load(model_path)
    
    def load(self, model_path: str):
        """Load trained model."""
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data.get('scaler')
        self.feature_names = model_data['feature_names']
    
    def predict(self, features: pd.DataFrame, min_tp: float = 0.01, max_tp: float = 0.15) -> float:
        """Predict optimal TP percentage."""
        if self.model is None:
            return self.default_tp
        
        X = pd.DataFrame()
        for col in self.feature_names:
            if col in features.columns:
                X[col] = features[col].values
            else:
                X[col] = 0
        
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        if self.scaler:
            X = self.scaler.transform(X)
        
        tp_pred = self.model.predict(X)[0]
        return np.clip(tp_pred, min_tp, max_tp)


def train_and_evaluate_tp(
    data_path: str = None, 
    test_size: float = 0.2, 
    save_best: bool = True,
    predict_rr: bool = False  # Set True to predict RR Ratio instead of TP%
) -> Dict:
    """
    Full TP predictor training pipeline.
    
    Args:
        predict_rr: If True, predict Risk/Reward Ratio (more stable).
                   If False, predict TP% directly (capped at 30%).
    """
    
    print("="*60)
    print("Stage 3: Take Profit Predictor Training (v2)")
    print("="*60)
    print(f"Mode: {'RR Ratio' if predict_rr else 'TP%'} prediction")
    print(f"Max TP cap: {MAX_TP_PCT*100:.0f}%")
    
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
    
    print("\nCalculating optimal TP for each trade (capped at {:.0f}%)...".format(MAX_TP_PCT*100))
    df = calculate_optimal_tp(df)
    
    X, y, df_meta = prepare_tp_training_data(df, predict_rr=predict_rr)
    
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
    
    results = train_tp_models(X_train_scaled, y_train)
    
    print("\n" + "="*60)
    print("Test Set Evaluation")
    print("="*60)
    
    test_results = {}
    for name, res in results.items():
        eval_res = evaluate_tp_model(res['model'], X_test_scaled, y_test, name, is_rr_target=predict_rr)
        test_results[name] = eval_res
        results[name]['test_mae'] = eval_res['mae']
        results[name]['test_r2'] = eval_res['r2']
    
    best_model_name = min(results.keys(), key=lambda k: results[k].get('test_mae', float('inf')))
    best_model = results[best_model_name]['model']
    
    if save_best:
        # Save with metadata about prediction mode
        model_data = {
            'model': best_model,
            'scaler': scaler,
            'feature_names': list(X.columns),
            'model_type': 'tp_predictor',
            'predict_rr': predict_rr,  # Important: tells us how to use predictions
            'max_tp_pct': MAX_TP_PCT
        }
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(model_data, MODEL_DIR / 'tp_predictor.joblib')
        print(f"✓ TP Predictor saved to {MODEL_DIR / 'tp_predictor.joblib'}")
    
    print("\n" + "="*60)
    print("TP Predictor Training Summary")
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
        'results': results,
        'predict_rr': predict_rr
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train TP Predictor (Stage 3 v2)')
    parser.add_argument('--data', type=str, default=None, help='Path to processed data')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--rr', action='store_true', help='Predict RR Ratio instead of TP%')
    
    args = parser.parse_args()
    
    train_and_evaluate_tp(
        data_path=args.data, 
        test_size=args.test_size,
        predict_rr=args.rr
    )
