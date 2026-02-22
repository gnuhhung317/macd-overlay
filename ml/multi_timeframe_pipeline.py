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

from data_pipeline import (
    load_ohlcv_1h,
    load_funding,
    calculate_features,
    calculate_macd,
    calculate_atr,
    calculate_rsi,
    calculate_stochastic,
    generate_labels,
    get_feature_columns,
    OHLCV_DIR,
    PROCESSED_DIR
)


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
        parquet_files = list(OHLCV_DIR.glob("*_USDT.parquet"))
        symbols = [f.stem.replace('_USDT', '') for f in parquet_files]
        print(f"Found {len(symbols)} symbols")
    
    if limit_symbols:
        symbols = symbols[:limit_symbols]
        print(f"Limited to {len(symbols)} symbols")
    
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
        
        all_data.append(df)
        
        if i % 50 == 0:
            print(f"  Processed {i}/{len(symbols)} symbols...")
    
    if not all_data:
        print("No data loaded!")
        return pd.DataFrame()
    
    df_combined = pd.concat(all_data, ignore_index=True)
    print(f"\n✓ Combined: {len(df_combined):,} rows from {len(all_data)} symbols")
    
    # Generate labels with ATR-based dynamic targets
    print("\nGenerating labels with ATR-based Triple Barrier Method...")
    
    # ATR multipliers - same across timeframes since ATR adapts naturally
    # TP = 3x ATR, SL = 1.5x ATR
    atr_tp_mult = 3.0
    atr_sl_mult = 1.5
    
    # Fallback fixed targets (used if ATR not available)
    # Shorter timeframes typically have smaller moves
    tf_scale = {
        '1h': 0.5,   # 1.5% TP, 0.75% SL
        '4h': 7,  # 2.25% TP, 1.125% SL
        '8h': 7,   # 2.7% TP, 1.35% SL
        '12h': 7.0,  # 3% TP, 1.5% SL
        '1d': 7.0,   # 3% TP, 1.5% SL
        '1w': 14.0   # 6% TP, 3% SL
    }
    scale = tf_scale.get(timeframe, 1.0)
    tp_pct = 0.03 * scale
    sl_pct = 0.015 * scale
    
    print(f"  ATR multipliers: TP={atr_tp_mult}x, SL={atr_sl_mult}x")
    print(f"  Fallback fixed: TP={tp_pct:.2%}, SL={sl_pct:.2%}")
    
    df_labeled = generate_labels(
        df_combined, 
        tp_pct=tp_pct, 
        sl_pct=sl_pct, 
        max_bars=10,
        use_atr=True,  # Enable ATR-based dynamic targets
        atr_tp_mult=atr_tp_mult,
        atr_sl_mult=atr_sl_mult,
        min_tp_pct=0.20,  # Minimum 20% gain required
        max_tp_pct=1.00   # Allow up to 100% gain (or higher if needed)
    )
    
    # Stats
    crossovers = df_labeled[(df_labeled['macd_cross_up'] == 1) | (df_labeled['macd_cross_down'] == 1)]
    crossovers_labeled = crossovers.dropna(subset=['label'])
    wins = crossovers_labeled['label'] == 1
    
    print(f"\n📊 Crossover Stats:")
    print(f"   Total crossovers: {len(crossovers):,}")
    print(f"   With labels: {len(crossovers_labeled):,}")
    print(f"   Win rate: {wins.mean():.1%}" if len(crossovers_labeled) > 0 else "   Win rate: N/A")
    
    # Dynamic TP/SL statistics
    if 'tp_pct_used' in crossovers_labeled.columns:
        tp_used = crossovers_labeled['tp_pct_used'].dropna()
        sl_used = crossovers_labeled['sl_pct_used'].dropna()
        if len(tp_used) > 0:
            print(f"\n📈 Dynamic TP/SL Stats (ATR-based):")
            print(f"   TP%: min={tp_used.min():.2%}, mean={tp_used.mean():.2%}, max={tp_used.max():.2%}")
            print(f"   SL%: min={sl_used.min():.2%}, mean={sl_used.mean():.2%}, max={sl_used.max():.2%}")
            print(f"   Avg Risk/Reward: {(tp_used / sl_used).mean():.2f}")
    
    # Trade result breakdown
    if 'trade_result' in crossovers_labeled.columns:
        result_counts = crossovers_labeled['trade_result'].value_counts()
        print(f"\n📋 Trade Results:")
        for result, count in result_counts.items():
            pct = count / len(crossovers_labeled) * 100
            print(f"   {result}: {count:,} ({pct:.1f}%)")
    
    # Save
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        output_path = PROCESSED_DIR / f'features_{timeframe}_full.parquet'
        df_labeled.to_parquet(output_path)
        print(f"\n✓ Saved to {output_path}")
    
    return df_labeled


def build_all_timeframes(symbols: list = None, limit_symbols: int = None):
    """Build datasets for all timeframes."""
    # timeframes = ['1h', '4h', '8h', '12h', '1d', '1w']
    timeframes = ['4h', '8h', '12h', '1d', '1w']
    
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
                'crossovers': len(df[(df['macd_cross_up'] == 1) | (df['macd_cross_down'] == 1)])
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
            print(f"  {tf}: {stats['rows']:,} rows, {stats['symbols']} symbols, {stats['crossovers']:,} crossovers")
    
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
        data_path = PROCESSED_DIR / f'features_{tf}_full.parquet'
        if not data_path.exists():
            print(f"  {tf}: No data found")
            continue
        
        df = pd.read_parquet(data_path)
        
        # Filter crossovers with labels
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
