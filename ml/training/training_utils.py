#!/usr/bin/env python3
"""
Shared utilities for ML training modules.
"""
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
import numpy as np

# Add parent to path to allow importing config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Paths
ML_DIR = Path(__file__).parent.parent
DATA_DIR = ML_DIR.parent / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = ML_DIR / 'models'


def filter_top_symbols_by_volume(
    df: pd.DataFrame,
    top_n: int = 150,
    recent_days: int = 180
) -> pd.DataFrame:
    """
    Filter training data to top N most liquid symbols by recent median volume.
    
    Rationale: pump-dump altcoins with random/thin volume create very noisy 
    labels (MACD crossover TP/SL outcomes are essentially random for these coins).
    Keeping only liquid, high-volume coins improves signal-to-noise ratio.
    
    Args:
        df: Full feature DataFrame with 'symbol', 'volume', 'timestamp' columns
        top_n: Number of top symbols to keep (default 150)
        recent_days: Only use recent data for ranking (avoids coins that were 
                     liquid 3 years ago but are now dead)
    
    Returns:
        Filtered DataFrame with only top_n symbols
    """
    if 'symbol' not in df.columns or 'volume' not in df.columns:
        print("  ⚠️  Cannot filter by volume: missing 'symbol' or 'volume' columns")
        return df
    
    # Use recent data to rank (more relevant than full history)
    if 'timestamp' in df.columns:
        cutoff = df['timestamp'].max() - pd.Timedelta(days=recent_days)
        df_recent = df[df['timestamp'] >= cutoff]
    else:
        df_recent = df
    
    if df_recent.empty:
        df_recent = df
    
    # Rank by median volume in recent period (more robust than mean vs outlier spikes)
    symbol_volume = (
        df_recent.groupby('symbol')['volume']
        .median()
        .sort_values(ascending=False)
    )
    
    total_symbols = len(symbol_volume)
    top_symbols = symbol_volume.head(top_n).index.tolist()
    
    df_filtered = df[df['symbol'].isin(top_symbols)].copy()
    
    print(f"\n📊 Volume Filter: kept top {top_n} of {total_symbols} symbols "
          f"(last {recent_days} days median volume)")
    print(f"   Before: {len(df):,} rows | After: {len(df_filtered):,} rows "
          f"({len(df_filtered)/len(df)*100:.0f}%)")
    
    return df_filtered

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
    # CRITICAL: Exclude all future/outcome information AND non-stationary absolute values.
    # Non-stationarity: MACD at BTC $20k is numerically different from MACD at $70k.
    # The model can't generalize across price levels if trained on absolute values.
    # Use the *_pct normalized versions instead.
    exclude_cols = {
        # Identifiers and raw OHLCV (absolute price levels — never use!)
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'date',
        
        # 🚨 NON-STATIONARY ABSOLUTE BTC MACRO PRICES — use btc_returns, btc_adx,
        # btc_is_bull_regime, btc_trend_strength instead (those are relative/stationary)!
        'btc_close', 'btc_sma_200',
        
        # Labels and targets (FUTURE INFO — NEVER USE!)
        'label', 'actual_tp', 'actual_sl', 'actual_rr',
        'tp_pct_used', 'sl_pct_used',
        
        # 🚨 MFE/MAE REGRESSION TARGETS (FUTURE INFO — NEVER USE!)
        'mfe_pct', 'mae_pct',
        
        # Outcome data (FUTURE INFO — NEVER USE!)
        'bars_to_tp', 'bars_to_sl', 'bars_to_outcome',
        'tp_first', 'sl_first', 'outcome', 'trade_result',
        'max_profit', 'max_drawdown',
        
        # Crossover signals (used for filtering rows, not as ML features)
        'macd_cross_up', 'macd_cross_down', 'macd_crossover',
        'wt_cross_up', 'wt_cross_down',  # WaveTrend crossover signals
        'lorentz_buy_signal', 'lorentz_sell_signal',  # Lorentzian entry signals
        'forward_return_pct',  # Label-related (future info)
        
        # 🚨 NON-STATIONARY ABSOLUTE MACD — use macd_pct, signal_pct, histogram_pct instead!
        'macd', 'signal', 'histogram',
        'macd_slope', 'signal_slope', 'histogram_slope', 'macd_acceleration',
        
        # 🚨 NON-STATIONARY ABSOLUTE ATR — use atr_14_pct, atr_7_pct instead!
        'atr_14', 'atr_7', 'atr_21', 'atr',
        
        # 🚨 NON-STATIONARY ABSOLUTE SMA/EMA/KC (price-level dependent)
        # Use price_to_sma_*, trend_*, dist_to_ema_*_pct instead!
        'sma_7', 'sma_14', 'sma_21', 'sma_50', 'sma_100', 'sma_200',
        'ema_7', 'ema_14', 'ema_21', 'ema_50', 'ema_100', 'ema_200',
        'bb_middle', 'bb_upper', 'bb_lower', 'bb_std',
        'kc_middle', 'kc_upper', 'kc_lower',
        
        # 🚨 NON-STATIONARY RAW OBV & VOLUME SMAs (cumulative and absolute)
        # Use obv_trend, volume_ratio, volume_spike, buy_pressure_14 instead!
        'obv', 'obv_sma',
        'volume_sma_7', 'volume_sma_14', 'volume_sma_20',
        'buy_volume', 'sell_volume',
        
        # Intermediate pipeline columns (not features)
        'vol_sma_20', 'recent_high_20', 'recent_low_20',
        'entry_price', 'entry_bar',
        'pullback_long_entry', 'pullback_short_entry',
        'macd_cross_up_filtered', 'macd_cross_down_filtered',
        'confluence_score',
        'volume_surge', 'rsi_sweet_spot', 'trend_aligned',
        'not_at_resistance', 'not_at_support',
        'strong_macd_momentum', 'good_volatility', 'no_recent_whipsaws',
        'is_potential_long_entry', 'is_potential_short_entry',
        'sma_89',
        # WT intermediate columns
        'wt_cross_count_10bars',
    }
    
    # For SL/TP prediction: also exclude ATR% (target is ATR-based, would be circular)
    if exclude_atr:
        exclude_cols.update({
            'atr_14_pct', 'atr_7_pct',
            'volatility_7', 'volatility_14', 'volatility_21',
            'volatility_7_scaled', 'volatility_14_scaled',
            'bb_width',
        })
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Filter numeric only
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    return numeric_cols

def evaluate_on_exchanges(model, scaler, feature_cols: List[str], timeframe: str, model_type: str, 
                             test_start: Optional[pd.Timestamp] = None, 
                             test_end: Optional[pd.Timestamp] = None):
    """
    Evaluate trained model on alternative exchanges (Bitget, Bybit) to test robustness.

    If `test_start`/`test_end` are provided, restrict evaluation to the same
    timestamp window that was used for the holdout test set. This allows a
    fair comparison and prevents the model from seeing future data.
    """
    from sklearn.metrics import roc_auc_score, mean_absolute_error
    from scipy.stats import spearmanr
    exchanges = ['bitget', 'bybit', 'kraken', 'okx', 'mexc']
    print("\n" + "-"*60)
    print(f"EVALUATING ON OTHER EXCHANGES (Robustness Test)")
    if test_start is not None and test_end is not None:
        print(f"   Window: {test_start} -> {test_end}")
    print("-"*60)
    
    for exchange in exchanges:
        exch_data_dir = ML_DIR.parent / f'{exchange}-data' / 'processed'
        data_path = exch_data_dir / f'features_{timeframe}_full.parquet'
        
        if not data_path.exists():
            print(f"  {exchange.upper():<10}: No data found")
            continue
            
        try:
            df = pd.read_parquet(data_path)
            
            # Use Lorentzian signals if available, fallback to MACD
            if 'lorentz_buy_signal' in df.columns:
                df_cross = df[
                    (df['lorentz_buy_signal'] == 1) | (df['lorentz_sell_signal'] == 1)
                ].copy()
                is_bullish_col = 'lorentz_buy_signal'
                is_bearish_col = 'lorentz_sell_signal'
            else:
                df_cross = df[
                    (df['macd_cross_up'] == 1) | (df['macd_cross_down'] == 1)
                ].copy()
                is_bullish_col = 'macd_cross_up'
                is_bearish_col = 'macd_cross_down'
            
            # apply test window filter if requested
            if test_start is not None and test_end is not None:
                df_cross = df_cross[
                    (df_cross['timestamp'] >= test_start) &
                    (df_cross['timestamp'] <= test_end)
                ]
                if df_cross.empty:
                    print(f"  {exchange.upper():<10}: No crossover rows in test window")
                    continue
            
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
                # Try to fill missing columns with zeros (for optional features like btc regime)
                for col in missing:
                    df_cross[col] = 0
                print(f"  {exchange.upper():<10}: Filled {len(missing)} missing cols with 0")
                
            X = df_cross[raw_cols].copy()
            X['is_bullish_cross'] = df_cross[is_bullish_col].values
            X = X[feature_cols] # Ensure exact order
            X = X.fillna(0).replace([np.inf, -np.inf], 0)
            
            if model_type == 'entry_filter':
                if isinstance(model, dict) and ('long_bull' in model or 'long' in model):
                    y_proba = np.zeros(len(X))
                    mask_long = df_cross[is_bullish_col] == 1
                    mask_short = df_cross[is_bearish_col] == 1
                    
                    if 'long_bull' in model:
                        # 4-model MoE: route by direction + regime
                        mask_bull = (df_cross['btc_is_bull_regime'] == 1).values if 'btc_is_bull_regime' in df_cross.columns else np.ones(len(df_cross), dtype=bool)
                        mask_bear = ~mask_bull
                        
                        for mask, key in [
                            (mask_long & mask_bull, 'long_bull'),
                            (mask_long & mask_bear, 'long_bear'),
                            (mask_short & mask_bull, 'short_bull'),
                            (mask_short & mask_bear, 'short_bear'),
                        ]:
                            if mask.sum() > 0 and key in model and model[key] is not None:
                                X_sub = X[mask].copy()
                                X_sub_scaled = pd.DataFrame(scaler[key].transform(X_sub), columns=X_sub.columns, index=X_sub.index)
                                y_proba[mask] = model[key].predict_proba(X_sub_scaled)[:, 1]
                    else:
                        # Legacy 2-model
                        if mask_long.sum() > 0 and 'long' in model:
                            X_long = X[mask_long].copy()
                            X_long_scaled = pd.DataFrame(scaler['long'].transform(X_long), columns=X_long.columns, index=X_long.index)
                            y_proba[mask_long] = model['long'].predict_proba(X_long_scaled)[:, 1]
                            
                        if mask_short.sum() > 0 and 'short' in model:
                            X_short = X[mask_short].copy()
                            X_short_scaled = pd.DataFrame(scaler['short'].transform(X_short), columns=X_short.columns, index=X_short.index)
                            y_proba[mask_short] = model['short'].predict_proba(X_short_scaled)[:, 1]
                else:
                    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
                    y_proba = model.predict_proba(X_scaled)[:, 1]
                
                score = roc_auc_score(y, y_proba)
                print(f"  {exchange.upper():<10}: AUC = {score:.4f} ({len(X)} samples)")
            else:
                X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
                y_pred = model.predict(X_scaled)
                mae_score = mean_absolute_error(y, y_pred)
                ic_score, _ = spearmanr(y, y_pred)
                print(f"  {exchange.upper():<10}: MAE = {mae_score:.4f}, IC = {ic_score:.4f} ({len(X)} samples)")
                
        except Exception as e:
            print(f"  {exchange.upper():<10}: Error - {e}")
