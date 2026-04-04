import pandas as pd
import numpy as np

def engineer_liquidity_sweep_features(df: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    """
    Engineer advanced OOS-safe features for Iteration 1: Liquidity Sweep & Leverage Tension.
    All features strictly use rolling windows to prevent look-ahead bias and data leakage.
    Assumes `df` contains: Open, High, Low, Close, Volume, FundingRate.
    """
    df_feat = df.copy()

    # Epsilon to prevent division by zero
    eps = 1e-8

    # --- 1. Smart Money Mechanics: Wick-to-Body Ratios ---
    body = (df_feat['close'] - df_feat['open']).abs()
    lower_wick = df_feat[['open', 'close']].min(axis=1) - df_feat['low']
    upper_wick = df_feat['high'] - df_feat[['open', 'close']].max(axis=1)
    
    df_feat['lower_wick_ratio'] = lower_wick / (body + eps)
    df_feat['upper_wick_ratio'] = upper_wick / (body + eps)

    # --- 2. Smart Money Mechanics: Rolling 24h Extremes (Liquidity Sweeps) ---
    # Detect if the current low sweeps the 24-period low
    df_feat['rolling_24h_low'] = df_feat['low'].shift(1).rolling(window=window).min()
    df_feat['rolling_24h_high'] = df_feat['high'].shift(1).rolling(window=window).max()
    
    # Sweep Booleans (True if swept)
    df_feat['swept_local_low'] = df_feat['low'] < df_feat['rolling_24h_low']
    df_feat['swept_local_high'] = df_feat['high'] > df_feat['rolling_24h_high']

    # --- 3. Robust Z-Scores for Volume ---
    # Using Median and IQR stringently (Robust Z-Score = (X - Median) / IQR)
    vol_median = df_feat['volume'].rolling(window=window).median()
    vol_q75 = df_feat['volume'].rolling(window=window).quantile(0.75)
    vol_q25 = df_feat['volume'].rolling(window=window).quantile(0.25)
    vol_iqr = vol_q75 - vol_q25
    df_feat['volume_robust_z'] = (df_feat['volume'] - vol_median) / (vol_iqr + eps)

    # --- 4. Leverage Tension: Robust Z-Scores for Funding Rate ---
    if 'funding_rate' in df_feat.columns:
        funding_median = df_feat['funding_rate'].rolling(window=window).median()
        funding_q75 = df_feat['funding_rate'].rolling(window=window).quantile(0.75)
        funding_q25 = df_feat['funding_rate'].rolling(window=window).quantile(0.25)
        funding_iqr = funding_q75 - funding_q25
        df_feat['funding_robust_z'] = (df_feat['funding_rate'] - funding_median) / (funding_iqr + eps)
    else:
        # Mocking for testing purposes when Funding Rate isn't present
        df_feat['funding_robust_z'] = 0.0 

    # --- 5. Composite Triggers ---
    # Long Trigger: Sweeps local low, long lower wick (e.g. wick is at least 2x the body), 
    # abnormally high volume (z > 1.5), and funding rate deeply negative (z < -1.5)
    df_feat['long_liquidity_sweep_trigger'] = (
        df_feat['swept_local_low'] & 
        (df_feat['lower_wick_ratio'] > 2.0) & 
        (df_feat['volume_robust_z'] > 1.5) & 
        (df_feat['funding_robust_z'] < -1.5)
    ).astype(int)
    
    # Short Trigger: Sweeps local high, long upper wick, high volume, funding deeply positive
    df_feat['short_liquidity_sweep_trigger'] = (
        df_feat['swept_local_high'] & 
        (df_feat['upper_wick_ratio'] > 2.0) & 
        (df_feat['volume_robust_z'] > 1.5) & 
        (df_feat['funding_robust_z'] > 1.5)
    ).astype(int)

    return df_feat

if __name__ == "__main__":
    # Example Dummy Data run
    print("Generating Mock Data and Testing Features...")
    idx = pd.date_range("2026-01-01", periods=100, freq='1h')
    mock_df = pd.DataFrame({
        'open': np.random.uniform(100, 110, size=100),
        'high': np.random.uniform(110, 115, size=100),
        'low': np.random.uniform(90, 100, size=100),
        'close': np.random.uniform(100, 110, size=100),
        'volume': np.random.uniform(1000, 5000, size=100),
        'funding_rate': np.random.uniform(-0.01, 0.01, size=100)
    }, index=idx)
    
    # Ensure High is highest, Low is lowest
    mock_df['high'] = mock_df[['open', 'close', 'high']].max(axis=1) + 1
    mock_df['low'] = mock_df[['open', 'close', 'low']].min(axis=1) - 1
    
    # Force a liquidity sweep and extreme funding event at index 50
    mock_df.iloc[50, mock_df.columns.get_loc('low')] = 80  # sweep previous lows
    mock_df.iloc[50, mock_df.columns.get_loc('open')] = 90
    mock_df.iloc[50, mock_df.columns.get_loc('close')] = 92 # small body, big lower wick
    mock_df.iloc[50, mock_df.columns.get_loc('volume')] = 20000 # huge volume
    mock_df.iloc[50, mock_df.columns.get_loc('funding_rate')] = -0.1 # trapped shorts
    
    feat_df = engineer_liquidity_sweep_features(mock_df)
    
    # Drop rows with NaN due to rolling windows
    feat_df_clean = feat_df.dropna()
    print("Columns generated:", feat_df.columns.tolist())
    
    triggers = feat_df[feat_df['long_liquidity_sweep_trigger'] == 1]
    print(f"\\nFound {len(triggers)} long setups:")
    print(triggers[['close', 'lower_wick_ratio', 'volume_robust_z', 'funding_robust_z', 'long_liquidity_sweep_trigger']])
