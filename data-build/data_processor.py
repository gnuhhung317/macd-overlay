"""
Data Processing Integration with Feature Building
Handles data loading, merging, and feature engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from feature_v2 import create_market_features


# Directory configurations
DIR_OHLCV = Path("data/ohlcv")
DIR_OI = Path("data/derivatives")
DIR_FUNDING = Path("data/funding")


def load_btc_reference_data() -> pd.DataFrame:
    """Load BTC OHLCV reference data used to build BTC-context features for all coins."""
    path_btc_ohlcv = DIR_OHLCV / "BTC_USDT.parquet"

    if not path_btc_ohlcv.exists():
        print(f"[BTC] BTC reference OHLCV file not found: {path_btc_ohlcv}")
        return pd.DataFrame()

    try:
        btc_df = pd.read_parquet(path_btc_ohlcv)
    except Exception as e:
        print(f"[BTC] Failed to load BTC reference OHLCV: {e}")
        return pd.DataFrame()

    if btc_df.empty:
        print("[BTC] BTC reference OHLCV is empty")
        return pd.DataFrame()

    if 'timestamp' not in btc_df.columns:
        print("[BTC] BTC reference OHLCV missing 'timestamp' column")
        return pd.DataFrame()

    btc_df = btc_df.copy()
    btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'])
    btc_df = btc_df.sort_values('timestamp').reset_index(drop=True)
    return btc_df


def process_symbol_data(symbol: str, btc_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Load and process OHLCV, OI, and Funding data
    Then build all features
    
    Parameters:
    -----------
    symbol : str
        Trading symbol (e.g., 'BTC', '0G')
    btc_df : Optional[pd.DataFrame]
        BTC reference dataframe used for BTC-context features
    
    Returns:
    --------
    pd.DataFrame
        Processed dataframe with all features
    """
    print(f"[{symbol}] Đang xử lý data...")
    
    # Define file paths
    path_ohlcv = DIR_OHLCV / f"{symbol}_USDT.parquet"
    path_oi = DIR_OI / f"{symbol}.parquet"
    path_funding = DIR_FUNDING / f"{symbol}_USDT.parquet"
    
    # Load OHLCV (required)
    if not path_ohlcv.exists():
        print(f"[{symbol}] OHLCV file not found: {path_ohlcv}")
        return pd.DataFrame()
    
    df_ohlcv = pd.read_parquet(path_ohlcv)
    if df_ohlcv.empty:
        print(f"[{symbol}] OHLCV data is empty")
        return pd.DataFrame()
    
    df_ohlcv['timestamp'] = pd.to_datetime(df_ohlcv['timestamp'])
    
    # Load OI (optional)
    df_oi = pd.DataFrame()
    if path_oi.exists():
        try:
            df_oi = pd.read_parquet(path_oi)
            if not df_oi.empty:
                df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp'])
                # Remove symbol column if it exists
                if 'symbol' in df_oi.columns:
                    df_oi = df_oi.drop(columns=['symbol'])
                print(f"[{symbol}] OI columns: {df_oi.columns.tolist()}")
        except Exception as e:
            print(f"[{symbol}] OI loading error: {e}")
    
    # Load Funding Rate (optional)
    # Note: Funding is usually 4h or 8h, will be filled to match 1h OHLCV
    df_funding = pd.DataFrame()
    if path_funding.exists():
        try:
            df_funding = pd.read_parquet(path_funding)
            if not df_funding.empty:
                df_funding['timestamp'] = pd.to_datetime(df_funding['timestamp'])
                print(f"[{symbol}] Funding columns: {df_funding.columns.tolist()}")
        except Exception as e:
            print(f"[{symbol}] Funding loading error: {e}")
    
    # Merge all data
    df_merged = df_ohlcv.copy()
    
    # Merge OI data (left join to keep all OHLCV rows)
    if not df_oi.empty:
        df_merged = pd.merge(df_merged, df_oi, on='timestamp', how='left')
        print(f"[{symbol}] After OI merge: {df_merged.shape}")
    
    # Merge Funding data (left join + forward fill for 4h/8h data)
    if not df_funding.empty:
        df_merged = pd.merge(df_merged, df_funding, on='timestamp', how='left')
        # Forward fill funding rate to propagate 4h/8h data to 1h candles
        df_merged['fundingRate'] = df_merged['fundingRate'].ffill()
        print(f"[{symbol}] After Funding merge: {df_merged.shape}")
    
    # Sort by timestamp
    df_merged = df_merged.sort_values('timestamp').reset_index(drop=True)
    
    # Defensive check: ensure all expected columns exist
    expected_cols = ['fundingRate', 'sum_open_interest', 'top_ls_ratio', 'global_ls_ratio']
    for col in expected_cols:
        if col not in df_merged.columns:
            df_merged[col] = np.nan
        # Forward fill missing values
        df_merged[col] = df_merged[col].ffill().bfill()
    
    print(f"[{symbol}] Before feature building: {df_merged.shape}, columns: {df_merged.columns.tolist()}")
    
    # Build all features
    print(f"[{symbol}] Building features...")
    df_merged = create_market_features(df_merged, btc_df=btc_df)
    
    print(f"[{symbol}] ✓ Complete! Shape: {df_merged.shape}, Features: {df_merged.shape[1] - 6}")
    
    return df_merged


def batch_process_symbols(symbols: list, use_btc_context: bool = True) -> dict:
    """
    Process multiple symbols
    
    Parameters:
    -----------
    symbols : list
        List of symbols to process
    use_btc_context : bool
        If True, add BTC context features to every symbol via BTC reference dataframe
    
    Returns:
    --------
    dict
        Dictionary mapping symbols to processed dataframes
    """
    btc_df = load_btc_reference_data() if use_btc_context else pd.DataFrame()
    if use_btc_context and btc_df.empty:
        print("[BTC] BTC context disabled for this batch (reference data unavailable)")

    results = {}
    for symbol in symbols:
        try:
            symbol_btc_df = btc_df if (use_btc_context and not btc_df.empty) else None
            df = process_symbol_data(symbol, btc_df=symbol_btc_df)
            if not df.empty:
                results[symbol] = df
            else:
                print(f"[{symbol}] Skipped (no data)")
        except Exception as e:
            print(f"[{symbol}] Error: {e}")
    
    return results


def validate_data_quality(df: pd.DataFrame, symbol: str) -> dict:
    """
    Validate data quality and report statistics
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to validate
    symbol : str
        Symbol name for reporting
    
    Returns:
    --------
    dict
        Quality metrics
    """
    metrics = {
        'symbol': symbol,
        'rows': len(df),
        'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
        'missing_values': df.isnull().sum().sum(),
        'nan_percent': (df.isnull().sum().sum() / (len(df) * df.shape[1])) * 100,
        'features': df.shape[1],
    }
    
    # Check for common columns
    common_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'fundingRate']
    missing_cols = [col for col in common_cols if col not in df.columns]
    if missing_cols:
        metrics['missing_columns'] = missing_cols
    
    return metrics


# Example usage
if __name__ == "__main__":
    # Process single symbol
    df = process_symbol_data("BTC")
    if not df.empty:
        print(f"\nData shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(df[['timestamp', 'close', 'volume', 'fundingRate']].head())
        
        # Validate quality
        quality = validate_data_quality(df, "BTC")
        print(f"\nData Quality: {quality}")
    
    # Process multiple symbols
    # symbols = ['BTC', 'ETH', 'SOL', '0G']
    # results = batch_process_symbols(symbols)
    # for symbol, df in results.items():
    #     quality = validate_data_quality(df, symbol)
    #     print(f"{symbol}: {quality['rows']} rows, {quality['nan_percent']:.2f}% NaN")
