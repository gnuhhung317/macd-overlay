#!/usr/bin/env python3
"""
Multi-Timeframe Data Pipeline

Build features and labels for multiple timeframes: 1h, 4h, 8h, 12h, 1d
This allows testing ML models on different timeframes.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import data_pipeline
from data_pipeline import (
    load_ohlcv_1h,
    load_funding,
    calculate_features,
    calculate_macd,
    calculate_atr,
    calculate_rsi,
    calculate_stochastic,
    calculate_cci,
    calculate_wavetrend,
    generate_labels,
    get_feature_columns
)


def add_entry_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add CONTINUOUS entry quality features for ML model to learn from.
    NO hard-coded thresholds — model decides what matters.
    
    Philosophy: Trader chuyên nghiệp nhìn vào nhiều yếu tố trước khi entry,
    nhưng mỗi yếu tố có trọng số khác nhau tùy market condition.
    Cho model tự học trọng số thay vì hard-code.
    """
    df = df.copy()
    
    # === 1. VOLUME CONTEXT (continuous) ===
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['vol_sma_20'] + 1e-8)
    df['volume_rank_50'] = df['volume'].rolling(50).rank(pct=True)
    df['volume_trend'] = df['vol_sma_20'].pct_change(5)  # Volume expanding/contracting
    
    # === 2. RSI CONTEXT (continuous spectrum) ===
    if 'rsi_14' in df.columns:
        df['rsi_distance_from_50'] = (df['rsi_14'] - 50) / 50  # -1 to +1, 0 = neutral
        df['rsi_overbought_degree'] = np.maximum(df['rsi_14'] - 70, 0) / 30  # 0-1
        df['rsi_oversold_degree'] = np.maximum(30 - df['rsi_14'], 0) / 30    # 0-1
    
    # === 3. PRICE POSITION IN RANGE (continuous) ===
    df['recent_high_20'] = df['high'].rolling(20).max()
    df['recent_low_20'] = df['low'].rolling(20).min()
    price_range = df['recent_high_20'] - df['recent_low_20']
    df['price_position_in_range'] = (df['close'] - df['recent_low_20']) / (price_range + 1e-8)  # 0-1
    df['distance_from_high_pct'] = (df['recent_high_20'] - df['close']) / (df['close'] + 1e-8) * 100
    df['distance_from_low_pct'] = (df['close'] - df['recent_low_20']) / (df['close'] + 1e-8) * 100
    
    # === 4. MACD MOMENTUM QUALITY (continuous) ===
    if 'histogram_pct' in df.columns:
        df['macd_momentum_strength'] = abs(df['histogram_pct'])
        df['macd_momentum_accel'] = df['histogram_pct'].diff()
        df['histogram_rank_50'] = df['histogram_pct'].rolling(50).rank(pct=True)
    
    # === 5. VOLATILITY REGIME (continuous) ===
    if 'atr_14' in df.columns:
        df['atr_pct'] = (df['atr_14'] / df['close']) * 100
        df['volatility_rank_50'] = df['atr_pct'].rolling(50).rank(pct=True)
        vol_fast = df['atr_pct'].rolling(10).mean()
        vol_slow = df['atr_pct'].rolling(50).mean()
        df['volatility_regime_ratio'] = vol_fast / (vol_slow + 1e-8)  # >1 = expanding vol
    
    # === 6. MARKET CHOPPINESS (continuous) ===
    if 'macd_cross_up' in df.columns and 'macd_cross_down' in df.columns:
        df['cross_count_10bars'] = (df['macd_cross_up'].rolling(10).sum() + 
                                    df['macd_cross_down'].rolling(10).sum())
        df['choppiness_score'] = df['cross_count_10bars'] / 10  # 0-1, higher = choppier
    
    # === 6b. WT CHOPPINESS (Lorentzian) ===
    if 'wt_cross_up' in df.columns and 'wt_cross_down' in df.columns:
        df['wt_cross_count_10bars'] = (df['wt_cross_up'].rolling(10).sum() + 
                                        df['wt_cross_down'].rolling(10).sum())
        df['wt_choppiness_score'] = df['wt_cross_count_10bars'] / 10
    
    # === 7. PULLBACK FEATURES (continuous, learnable) ===
    # Instead of binary pullback detection, measure pullback characteristics
    if 'ema_21' in df.columns:
        df['ema21_distance_pct'] = (df['close'] - df['ema_21']) / (df['close'] + 1e-8) * 100
        
    # Price action in last N bars (pullback context)
    for n in [2, 3, 5]:
        df[f'price_change_{n}bar'] = df['close'].pct_change(n) * 100
        df[f'low_change_{n}bar'] = (df['low'] - df['low'].shift(n)) / (df['close'] + 1e-8) * 100
        df[f'high_change_{n}bar'] = (df['high'] - df['high'].shift(n)) / (df['close'] + 1e-8) * 100
        
    # Candle body characteristics (pullback quality)
    body = abs(df['close'] - df['open'])
    total_range = df['high'] - df['low']
    df['candle_body_ratio'] = body / (total_range + 1e-8)  # 0-1, small body = indecision
    df['upper_wick_ratio'] = (df['high'] - df[['close', 'open']].max(axis=1)) / (total_range + 1e-8)
    df['lower_wick_ratio'] = (df[['close', 'open']].min(axis=1) - df['low']) / (total_range + 1e-8)
    
    # === 8. BARS SINCE LAST CROSS (continuous timing) ===
    if 'macd_cross_up' in df.columns and 'macd_cross_down' in df.columns:
        any_cross = (df['macd_cross_up'] == 1) | (df['macd_cross_down'] == 1)
        groups = any_cross.cumsum()
        df['bars_since_any_cross'] = df.groupby(groups).cumcount()
        df['bars_since_any_cross'] = df['bars_since_any_cross'].clip(0, 20)
        
        # Which direction was the last cross? 1=bullish, -1=bearish, 0=unknown
        last_cross_dir = (df['macd_cross_up'] - df['macd_cross_down']).replace(0, np.nan).ffill().fillna(0)
        df['last_cross_direction'] = last_cross_dir
        
        # Price change since last cross
        cross_price = df['close'].where(any_cross).ffill()
        df['price_change_since_cross_pct'] = (df['close'] - cross_price) / (cross_price + 1e-8) * 100
        
        # Volume change since last cross  
        cross_volume = df['volume'].where(any_cross).ffill()
        df['volume_ratio_since_cross'] = df['volume'] / (cross_volume + 1e-8)
    
    # === 9. WT-BASED TIMING FEATURES (Lorentzian) ===
    if 'wt_cross_up' in df.columns and 'wt_cross_down' in df.columns:
        any_wt_cross = (df['wt_cross_up'] == 1) | (df['wt_cross_down'] == 1)
        wt_groups = any_wt_cross.cumsum()
        df['bars_since_any_wt_cross'] = df.groupby(wt_groups).cumcount()
        df['bars_since_any_wt_cross'] = df['bars_since_any_wt_cross'].clip(0, 20)
        
        # Last WT cross direction
        last_wt_dir = (df['wt_cross_up'] - df['wt_cross_down']).replace(0, np.nan).ffill().fillna(0)
        df['last_wt_cross_direction'] = last_wt_dir
        
        # Price change since last WT cross
        wt_cross_price = df['close'].where(any_wt_cross).ffill()
        df['price_change_since_wt_cross_pct'] = (df['close'] - wt_cross_price) / (wt_cross_price + 1e-8) * 100
        
        # Volume change since last WT cross
        wt_cross_volume = df['volume'].where(any_wt_cross).ffill()
        df['volume_ratio_since_wt_cross'] = df['volume'] / (wt_cross_volume + 1e-8)
    
    # === 10. OSCILLATOR CONTEXT (Lorentzian, continuous) ===
    # How extreme are oscillators at the moment? (model learns optimal levels)
    if 'wt1' in df.columns:
        df['wt1_distance_from_zero'] = df['wt1'] / 100  # Normalized
        df['wt1_rank_50'] = df['wt1'].rolling(50).rank(pct=True)
    if 'cci_20' in df.columns:
        df['cci_rank_50'] = df['cci_20'].rolling(50).rank(pct=True)
    if 'rsi_9' in df.columns:
        df['rsi_9_distance_from_50'] = (df['rsi_9'] - 50) / 50
    
    return df


def generate_lorentzian_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate primary entry signals based on WaveTrend crossovers.
    
    Replaces MACD cross as the primary trigger for entry candidates.
    WaveTrend cross provides timing + direction (like MACD cross) but
    operates in the oscillator domain, aligned with Lorentzian Classification.
    
    XGBoost will learn which WT crosses are quality entries from the features
    (RSI, CCI, WT level, ADX, volume, etc.)
    """
    df = df.copy()
    
    # Primary signal: WaveTrend crossover
    # Buy: WT1 crosses above WT2 (bullish momentum shift)
    # Sell: WT1 crosses below WT2 (bearish momentum shift)
    df['lorentz_buy_signal'] = df['wt_cross_up'].astype(int)
    df['lorentz_sell_signal'] = df['wt_cross_down'].astype(int)
    
    return df


def generate_4bar_labels(df: pd.DataFrame, horizon: int = 4) -> pd.DataFrame:
    """
    Generate labels based on N-bar forward return direction.
    
    Inspired by Lorentzian Classification's labeling:
    - For buy signals: label=1 if price goes UP in next N bars (profitable long)
    - For sell signals: label=1 if price goes DOWN in next N bars (profitable short)
    - Non-signal rows: label=NaN (not used for training)
    
    This is simpler and more robust than Triple Barrier:
    - No ATR dependency for target calculation
    - No complex TP/SL barrier logic
    - Clean binary outcome: did price move in the expected direction?
    
    Args:
        df: DataFrame with OHLCV, lorentz_buy_signal, lorentz_sell_signal columns
        horizon: Number of bars to look ahead (default 4, from Pine Script)
    
    Returns:
        DataFrame with 'label' and 'forward_return_pct' columns added
    """
    df = df.copy()
    
    # Calculate future close per symbol (prevent cross-symbol contamination)
    df['_future_close'] = df.groupby('symbol')['close'].shift(-horizon)
    
    # Forward return (% change)
    df['forward_return_pct'] = (df['_future_close'] - df['close']) / (df['close'] + 1e-10)
    
    # Initialize label as NaN (non-signal rows won't have labels)
    df['label'] = np.nan
    
    # Label buy signals: win if price goes UP
    buy_mask = df['lorentz_buy_signal'] == 1
    df.loc[buy_mask, 'label'] = (df.loc[buy_mask, 'forward_return_pct'] > 0).astype(float)
    
    # Label sell signals: win if price goes DOWN  
    sell_mask = df['lorentz_sell_signal'] == 1
    df.loc[sell_mask, 'label'] = (df.loc[sell_mask, 'forward_return_pct'] < 0).astype(float)
    
    # Drop rows where future close is NaN (last N bars per symbol)
    # These can't be labeled since we don't know the future
    no_future = df['_future_close'].isna()
    df.loc[no_future, 'label'] = np.nan
    
    # Clean up intermediate column
    df = df.drop(columns=['_future_close'], errors='ignore')
    
    return df


def resample_to_timeframe(df_1h: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample 1h OHLCV to specified timeframe.
    
    Args:
        df_1h: DataFrame with 1h OHLCV data
        timeframe: Target timeframe ('1h', '4h', '8h', '12h', '1d')
    
    Returns:
        Resampled DataFrame
    """
    if df_1h.empty:
        return pd.DataFrame()
    
    df = df_1h.copy()
    
    # 1h doesn't need resampling
    if timeframe == '1h':
        return df
    
    # Map timeframe to pandas resample rule
    resample_map = {
        '4h': '4h',
        '8h': '8h',
        '12h': '12h',
        '1d': '1D',
        '1w': '1W',
    }
    
    if timeframe not in resample_map:
        raise ValueError(f"Unsupported timeframe: {timeframe}. Use 1h, 4h, 8h, 12h, 1d, or 1w")
    
    df = df.set_index('timestamp')
    
    # Resample OHLCV
    df_resampled = df.resample(resample_map[timeframe]).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    df_resampled = df_resampled.reset_index()
    return df_resampled


def calculate_features_for_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Calculate features with timeframe-aware adjustments.
    
    Different timeframes may need different lookback periods:
    - 1h: More bars available, use longer periods
    - 1d: Fewer bars, use shorter periods
    """
    df = calculate_features(df)
    
    # Add timeframe as feature (useful for multi-timeframe model)
    tf_map = {'1h': 1, '4h': 4, '8h': 8, '12h': 12, '1d': 24, '1w': 168}
    df['timeframe_hours'] = tf_map.get(timeframe, 24)
    
    # Normalize some features by timeframe
    # (e.g., volatility on 1h is naturally lower than 1d)
    if timeframe in ['1h', '4h']:
        # For shorter timeframes, scale volatility
        scale = 24 / tf_map[timeframe]  # 1h: 24x, 4h: 6x
        df['volatility_7_scaled'] = df['volatility_7'] * np.sqrt(scale)
        df['volatility_14_scaled'] = df['volatility_14'] * np.sqrt(scale)
    else:
        df['volatility_7_scaled'] = df['volatility_7']
        df['volatility_14_scaled'] = df['volatility_14']
    
    return df


def build_timeframe_dataset(
    symbols: list = None,
    timeframe: str = '1d',
    limit_symbols: int = None,
    min_rows: int = 200,
    save: bool = True
) -> pd.DataFrame:
    """
    Build complete dataset for a specific timeframe.
    
    Args:
        symbols: List of symbols (without USDT suffix), or None for all
        timeframe: Target timeframe ('1h', '4h', '8h', '12h', '1d')
        limit_symbols: Limit number of symbols (for testing)
        min_rows: Minimum rows required per symbol
        save: Whether to save to parquet
    
    Returns:
        DataFrame with features and labels
    """
    print(f"\n{'='*60}")
    print(f"Building Dataset for {timeframe} Timeframe")
    print(f"{'='*60}")
    
    # Get all symbols
    if symbols is None:
        parquet_files = list(data_pipeline.OHLCV_DIR.glob("*_USDT.parquet"))
        symbols = [f.stem.replace('_USDT', '') for f in parquet_files]
        print(f"Found {len(symbols)} symbols")
    
    if limit_symbols:
        symbols = symbols[:limit_symbols]
        print(f"Limited to {len(symbols)} symbols")
    
    # ---------------------------------------------------------
    # NEW: 1. Process BTC context first for Market Regime
    # ---------------------------------------------------------
    print("=> Extracting Macro Market Regime from BTCUSDT...")
    btc_context = pd.DataFrame()
    btc_symbol = None
    if 'BTCUSDT' in symbols:
        btc_symbol = 'BTCUSDT'
    elif 'BTC' in symbols:
        btc_symbol = 'BTC'
        
    if btc_symbol:
        df_1h_btc = load_ohlcv_1h(btc_symbol)
        if not df_1h_btc.empty:
            df_btc = resample_to_timeframe(df_1h_btc, timeframe)
            df_btc = calculate_features_for_timeframe(df_btc, timeframe)
            if not df_btc.empty:
                btc_context = df_btc[['timestamp', 'close', 'sma_200', 'adx', 'log_returns']].copy()
                btc_context.columns = ['timestamp', 'btc_close', 'btc_sma_200', 'btc_adx', 'btc_returns']
                btc_context['btc_is_bull_regime'] = (btc_context['btc_close'] > btc_context['btc_sma_200']).astype(int)
                btc_context['btc_trend_strength'] = np.where(btc_context['btc_adx'] > 25, 1, 0)
    # ---------------------------------------------------------
    
    all_data = []
    
    for i, symbol in enumerate(symbols, 1):
        # Load 1h data
        df_1h = load_ohlcv_1h(symbol)
        if df_1h.empty:
            continue
        
        # Resample to target timeframe
        df = resample_to_timeframe(df_1h, timeframe)
        if len(df) < min_rows:
            continue
        
        # Add symbol column
        df['symbol'] = symbol
        
        # Calculate features
        df = calculate_features_for_timeframe(df, timeframe)
        
        # Generate Lorentzian entry signals (WaveTrend cross)
        df = generate_lorentzian_signals(df)
        
        # Add learnable entry quality features (continuous, no hard filtering)
        df = add_entry_quality_features(df)
        
        # Merge funding rate (if available)
        df_funding = load_funding(symbol)
        if not df_funding.empty:
            df_funding = df_funding[['timestamp', 'funding_rate']].copy()
            df_funding['date'] = df_funding['timestamp'].dt.date
            funding_daily = df_funding.groupby('date')['funding_rate'].mean().reset_index()
            df['date'] = df['timestamp'].dt.date
            df = df.merge(funding_daily, on='date', how='left')
            df['funding_rate'] = df['funding_rate'].fillna(0)
            df = df.drop(columns=['date'])
        else:
            df['funding_rate'] = 0
            
        # ---------------------------------------------------------
        # NEW: 2. Merge BTC Context & Calculate Relative Strength
        # ---------------------------------------------------------
        if not btc_context.empty:
            df = df.merge(btc_context, on='timestamp', how='left')
            fill_cols = ['btc_is_bull_regime', 'btc_trend_strength', 'btc_returns']
            df[fill_cols] = df[fill_cols].ffill().fillna(0)
            df['rs_vs_btc'] = df['log_returns'] - df['btc_returns']
            df['rs_vs_btc_sma7'] = df['rs_vs_btc'].rolling(7).mean()
            
            # ---------------------------------------------------------
            # BTC REGIME-CONDITIONAL FEATURES
            # These help the model learn different patterns per BTC regime
            # ---------------------------------------------------------
            
            # 1. RSI adjusted by BTC regime
            # In BTC bull: low RSI is "buy the dip"; in BTC bear: low RSI can keep dropping
            if 'rsi_14' in df.columns:
                df['rsi_14_btc_adj'] = df['rsi_14'] * np.where(
                    df['btc_is_bull_regime'] == 1, 1.15, 0.85
                )
            
            # 2. MACD histogram credibility based on BTC alignment
            # Long signals in BTC bear are less reliable
            if 'histogram_pct' in df.columns:
                df['macd_btc_aligned'] = np.where(
                    df['histogram_pct'] > 0,  # Bullish MACD
                    df['histogram_pct'] * (0.5 + 0.5 * df['btc_is_bull_regime']),  # Boost in bull
                    df['histogram_pct'] * (0.5 + 0.5 * (1 - df['btc_is_bull_regime']))  # Boost in bear
                )
            
            # 3. Relative strength outperformance flag
            # Coins outperforming BTC in BTC bull = strong; outperforming in BTC bear = defensive
            df['rs_outperform'] = (df['rs_vs_btc_sma7'] > 0).astype(int)
            df['rs_outperform_regime'] = df['rs_outperform'] * (2 * df['btc_is_bull_regime'] - 1)
            
            # 4. Trend alignment score
            # Measures if coin trend aligns with BTC regime
            if 'trend_state' in df.columns:
                df['trend_btc_align'] = (
                    (df['trend_state'] == 1) & (df['btc_is_bull_regime'] == 1) |
                    (df['trend_state'] == -1) & (df['btc_is_bull_regime'] == 0)
                ).astype(int)
        # ---------------------------------------------------------
        
        all_data.append(df)
        
        if i % 50 == 0:
            print(f"  Processed {i}/{len(symbols)} symbols...")
    
    if not all_data:
        print("No data loaded!")
        return pd.DataFrame()
    
    df_combined = pd.concat(all_data, ignore_index=True)
    print(f"\n✓ Combined: {len(df_combined):,} rows from {len(all_data)} symbols")
    
    # ---------------------------------------------------------
    # Apply T-1 Global Shift to predictive features to prevent Look-ahead bias
    # ---------------------------------------------------------
    from data_pipeline import apply_global_feature_shift
    print("\nApplying T-1 Global Shift to features...")
    df_combined = apply_global_feature_shift(df_combined)
    
    # ---------------------------------------------------------
    # Lorentzian 4-bar Horizon Labeling
    # Replaces ATR-based Triple Barrier with simple directional labels
    # ---------------------------------------------------------
    print("\nGenerating 4-bar horizon labels (Lorentzian Classification)...")
    df_labeled = generate_4bar_labels(df_combined, horizon=4)
    
    # Stats for Lorentzian signals
    signals = df_labeled[
        (df_labeled['lorentz_buy_signal'] == 1) | 
        (df_labeled['lorentz_sell_signal'] == 1)
    ]
    signals_labeled = signals.dropna(subset=['label'])
    
    buy_signals = df_labeled[df_labeled['lorentz_buy_signal'] == 1].dropna(subset=['label'])
    sell_signals = df_labeled[df_labeled['lorentz_sell_signal'] == 1].dropna(subset=['label'])
    
    print(f"\n📊 Lorentzian Entry Dataset:")
    print(f"   Total WT crosses: {len(signals):,}")
    print(f"   With labels: {len(signals_labeled):,}")
    if len(signals_labeled) > 0:
        print(f"   Overall win rate: {signals_labeled['label'].mean():.1%}")
    if len(buy_signals) > 0:
        print(f"   Buy signals: {len(buy_signals):,} (win rate: {buy_signals['label'].mean():.1%})")
    if len(sell_signals) > 0:
        print(f"   Sell signals: {len(sell_signals):,} (win rate: {sell_signals['label'].mean():.1%})")
    print(f"   📖 Model will learn what makes a quality entry from features")
    
    # Forward return stats
    if 'forward_return_pct' in signals_labeled.columns:
        fwd = signals_labeled['forward_return_pct']
        print(f"\n📈 4-bar Forward Return Stats:")
        print(f"   Mean: {fwd.mean():.4f} ({fwd.mean()*100:.2f}%)")
        print(f"   Std: {fwd.std():.4f}")
        print(f"   Median: {fwd.median():.4f}")
    
    # Save
    if save:
        data_pipeline.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        output_path = data_pipeline.PROCESSED_DIR / f'features_{timeframe}_full.parquet'
        df_labeled.to_parquet(output_path)
        print(f"\n✓ Saved to {output_path}")
    
    return df_labeled


def build_all_timeframes(symbols: list = None, limit_symbols: int = None):
    """Build datasets for all timeframes."""
    # timeframes = ['1h', '4h', '8h', '12h', '1d', '1w']
    timeframes = ['8h', '12h', '1d', '1w']
    
    results = {}
    for tf in timeframes:
        try:
            df = build_timeframe_dataset(
                symbols=symbols,
                timeframe=tf,
                limit_symbols=limit_symbols,
                save=True
            )
            results[tf] = {
                'rows': len(df),
                'symbols': df['symbol'].nunique() if 'symbol' in df.columns else 0,
                'signals': len(df[(df['lorentz_buy_signal'] == 1) | (df['lorentz_sell_signal'] == 1)])
            }
        except Exception as e:
            print(f"Error building {tf}: {e}")
            results[tf] = {'error': str(e)}
    
    # Summary
    print("\n" + "="*60)
    print("Summary: All Timeframes")
    print("="*60)
    for tf, stats in results.items():
        if 'error' in stats:
            print(f"  {tf}: ERROR - {stats['error']}")
        else:
            print(f"  {tf}: {stats['rows']:,} rows, {stats['symbols']} symbols, {stats['signals']:,} WT signals")
    
    return results


def compare_timeframe_performance(model_path: str = None):
    """Compare ML model performance across timeframes."""
    import joblib
    from sklearn.metrics import accuracy_score, precision_score
    
    if model_path is None:
        model_path = Path(__file__).parent / 'models' / 'entry_filter.joblib'
    
    if not Path(model_path).exists():
        print("Model not found! Train entry_filter first.")
        return
    
    # Load model
    model_data = joblib.load(model_path)
    model = model_data['model']
    scaler = model_data.get('scaler')
    feature_names = model_data['feature_names']
    
    print(f"\nLoaded model: {model_path}")
    print(f"Features: {len(feature_names)}")
    
    # Test on each timeframe
    results = {}
    timeframes = ['1h', '4h', '8h', '12h', '1d', '1w']
    
    for tf in timeframes:
        data_path = data_pipeline.PROCESSED_DIR / f'features_{tf}_full.parquet'
        if not data_path.exists():
            print(f"  {tf}: No data found")
            continue
        
        df = pd.read_parquet(data_path)
        
        # Filter signals with labels (Lorentzian or MACD fallback)
        if 'lorentz_buy_signal' in df.columns:
            crossovers = df[
                ((df['lorentz_buy_signal'] == 1) | (df['lorentz_sell_signal'] == 1)) &
                (df['label'].notna())
            ].copy()
        else:
            crossovers = df[
                ((df['macd_cross_up'] == 1) | (df['macd_cross_down'] == 1)) &
                (df['label'].notna())
            ].copy()
        
        if len(crossovers) < 100:
            print(f"  {tf}: Not enough data ({len(crossovers)} samples)")
            continue
        
        # Prepare features
        X = crossovers[feature_names].fillna(0).replace([np.inf, -np.inf], 0)
        y = crossovers['label'].astype(int)
        
        if scaler:
            X = scaler.transform(X)
        
        # Predict
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        
        # Metrics
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        
        # At different thresholds
        thresholds = [0.5, 0.6, 0.7]
        prec_at_thresh = {}
        for thresh in thresholds:
            mask = y_proba >= thresh
            if mask.sum() > 0:
                prec_at_thresh[thresh] = y[mask].mean()
            else:
                prec_at_thresh[thresh] = 0
        
        results[tf] = {
            'samples': len(crossovers),
            'accuracy': acc,
            'precision': prec,
            'precision_at_50': prec_at_thresh.get(0.5, 0),
            'precision_at_60': prec_at_thresh.get(0.6, 0),
            'precision_at_70': prec_at_thresh.get(0.7, 0)
        }
        
        print(f"\n📊 {tf}:")
        print(f"   Samples: {len(crossovers):,}")
        print(f"   Accuracy: {acc:.1%}")
        print(f"   Precision: {prec:.1%}")
        print(f"   Precision @60%: {prec_at_thresh.get(0.6, 0):.1%}")
    
    return results


def main():
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'all':
            # Build all timeframes
            build_all_timeframes(limit_symbols=None)
        elif sys.argv[1] == 'compare':
            # Compare performance
            compare_timeframe_performance()
        else:
            # Build specific timeframe
            tf = sys.argv[1]
            limit = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else None
            build_timeframe_dataset(timeframe=tf, limit_symbols=limit)
    else:
        # Default: build 4h for testing
        print("Usage:")
        print("  python multi_timeframe_pipeline.py all     - Build all timeframes")
        print("  python multi_timeframe_pipeline.py 4h      - Build 4h timeframe")
        print("  python multi_timeframe_pipeline.py compare - Compare model on all timeframes")
        print("")
        print("Building 4h as example...")
        build_timeframe_dataset(timeframe='4h', limit_symbols=10)


if __name__ == '__main__':
    main()
