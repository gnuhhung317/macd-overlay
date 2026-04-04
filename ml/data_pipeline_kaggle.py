#!/usr/bin/env python3
"""
Data Pipeline: Load, Merge 1h->1d, Feature Engineering
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Default data directory (Bitget for backward compatibility)
DATA_DIR = Path('/kaggle/input/datasets/hungbui317/macd-coin/macd-overlay - Copy/data')
OHLCV_DIR = DATA_DIR / 'ohlcv'
FUNDING_DIR = DATA_DIR / 'funding'
DERIVATIVES_DIR = DATA_DIR / 'derivatives'

PROCESSED_DIR = 'processed'



def set_data_directory(new_dir: Path):
    """
    Dynamically update the data directories.
    Useful for switching between Bitget and Binance.
    """
    global DATA_DIR, OHLCV_DIR, FUNDING_DIR, DERIVATIVES_DIR, PROCESSED_DIR
    DATA_DIR = new_dir
    OHLCV_DIR = DATA_DIR / 'ohlcv'
    FUNDING_DIR = DATA_DIR / 'funding'
    DERIVATIVES_DIR = Path('/kaggle/input/datasets/hungbui317/macd-coin/data/data')
    PROCESSED_DIR = 'processed'
    print(f"📁 Data directory set to: {DATA_DIR}")


def load_ohlcv_1h(symbol: str) -> pd.DataFrame:
    """Load OHLCV 1h data for a symbol"""
    file_path = OHLCV_DIR / f"{symbol}_USDT.parquet"
    if not file_path.exists():
        print(f"  ⚠️ File not found: {file_path}")
        return pd.DataFrame()
    
    df = pd.read_parquet(file_path)
    
    # Standardize column names
    if 'timestamp' not in df.columns and 'open_time' in df.columns:
        df = df.rename(columns={'open_time': 'timestamp'})
    
    # Ensure timestamp is datetime
    if df['timestamp'].dtype == 'int64':
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def load_funding(symbol: str) -> pd.DataFrame:
    """Load funding rate data for a symbol"""
    file_path = FUNDING_DIR / f"{symbol}_USDT.parquet"
    if not file_path.exists():
        return pd.DataFrame()
    
    df = pd.read_parquet(file_path)
    
    # Standardize column names
    if 'fundingTime' in df.columns:
        df = df.rename(columns={'fundingTime': 'timestamp'})
    if 'fundingRate' in df.columns:
        df = df.rename(columns={'fundingRate': 'funding_rate'})
    
    if 'timestamp' not in df.columns:
        return pd.DataFrame()
    
    if df['timestamp'].dtype == 'int64':
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def load_derivatives(symbol: str) -> pd.DataFrame:
    """Load derivatives data (OI, Long/Short ratio) for a symbol"""
    # Try both naming conventions and fallback to base data dir
    possible_dirs = [DERIVATIVES_DIR]
    file_path = None
    
    for d in possible_dirs:
        if (d / f"{symbol}_USDT.parquet").exists():
            file_path = d / f"{symbol}_USDT.parquet"
            break
        elif (d / f"{symbol}.parquet").exists():
            file_path = d / f"{symbol}.parquet"
            break
            
    if not file_path:
        return pd.DataFrame()
    
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"  ⚠️ Error reading derivatives for {symbol}: {e}")
        return pd.DataFrame()
    
    # Standardize column names
    if 'timestamp' not in df.columns:
        return pd.DataFrame()
        
    if df['timestamp'].dtype == 'int64':
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def resample_to_timeframe(df_1h: pd.DataFrame, timeframe: str = '1d') -> pd.DataFrame:
    """
    Resample 1h OHLCV to target timeframe.
    
    Args:
        df_1h: 1h OHLCV DataFrame
        timeframe: Target timeframe ('1h', '4h', '8h', '12h', '1d')
    
    Returns:
        Resampled DataFrame
    """
    if df_1h.empty:
        return pd.DataFrame()
    
    # Map timeframe to pandas resample rule
    timeframe_map = {
        '1h': '1H',
        '4h': '4H',
        '8h': '8H',
        '12h': '12H',
        '1d': '1D'
    }
    
    if timeframe not in timeframe_map:
        raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {list(timeframe_map.keys())}")
    
    df = df_1h.copy()
    df = df.set_index('timestamp')
    
    # Resample to target timeframe
    df_resampled = df.resample(timeframe_map[timeframe]).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    df_resampled = df_resampled.reset_index()
    return df_resampled


def resample_1h_to_1d(df_1h: pd.DataFrame) -> pd.DataFrame:
    """DEPRECATED: Use resample_to_timeframe instead. Kept for backward compatibility."""
    return resample_to_timeframe(df_1h, '1d')


def calculate_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    """
    Calculate MACD using MACD Overlay formula (Pine Script version):
    MACD = EMA(close, slow-fast) = EMA(close, 14) for default params
    Signal Line = SMA(MACD, signal)
    Histogram = MACD - Signal
    Also calculates SMA(close, 89) as an additional indicator
    """
    df = df.copy()
    # MACD Overlay formula: MACD = EMA(close, slow-fast)
    macd_period = slow - fast  # 26 - 12 = 14
    df['macd'] = df['close'].ewm(span=macd_period, adjust=False).mean()
    
    # Signal = SMA of MACD (not EMA like traditional MACD)
    df['signal'] = df['macd'].rolling(window=signal).mean()
    
    df['histogram'] = df['macd'] - df['signal']
    
    # Additional SMA(close, 89) from Pine Script
    df['sma_89'] = df['close'].rolling(window=89).mean()
    
    return df


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all features for ML.
    Note: This function should be called per-symbol to avoid data bleeding.
    Use calculate_features_grouped() for multi-symbol DataFrames.
    """
    df = df.copy()
    
    # ===== Price Features =====
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    df['body_size'] = abs(df['close'] - df['open']) / df['close']
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
    
    # ===== Trend Features =====
    for period in [7, 14, 21, 50, 100, 200]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}']
    
    # Trend strength
    df['trend_7_21'] = df['sma_7'] / df['sma_21']
    df['trend_21_50'] = df['sma_21'] / df['sma_50']
    df['trend_50_200'] = df['sma_50'] / df['sma_200']
    
    # ===== MACD Features =====
    df = calculate_macd(df)
    
    # 🌟 NORMALIZED MACD (% of close price) — Stationary across price levels!
    # Raw macd/signal/histogram are scale-dependent (different at BTC $20k vs $70k)
    # Normalizing by close makes them comparable across all time periods.
    df['macd_pct'] = (df['macd'] / df['close']) * 100
    df['signal_pct'] = (df['signal'] / df['close']) * 100
    df['histogram_pct'] = (df['histogram'] / df['close']) * 100
    df['macd_slope_pct'] = df['macd_pct'].diff()
    df['signal_slope_pct'] = df['signal_pct'].diff()
    df['histogram_slope_pct'] = df['histogram_pct'].diff()
    df['macd_acceleration_pct'] = df['macd_slope_pct'].diff()
    
    # Keep raw for crossover calculation (used to detect entry signals, not as features)
    df['macd_slope'] = df['macd'].diff()
    df['signal_slope'] = df['signal'].diff()
    df['histogram_slope'] = df['histogram'].diff()
    df['macd_acceleration'] = df['macd_slope'].diff()
    
    # MACD crossover detection (boolean 0/1 — already stationary)
    df['macd_cross_up'] = ((df['macd'] > df['signal']) & (df['macd'].shift(1) <= df['signal'].shift(1))).astype(int)
    df['macd_cross_down'] = ((df['macd'] < df['signal']) & (df['macd'].shift(1) >= df['signal'].shift(1))).astype(int)
    df['macd_crossover'] = df['macd_cross_up'] - df['macd_cross_down']
    
    # 🚀 OPTIMIZED: Vectorized bars since crossover (O(n) instead of O(n²))
    # Use cumsum grouping technique for massive speedup
    def calculate_bars_since_event(event_series):
        """Vectorized calculation of bars since last event using cumsum grouping."""
        event_series = event_series.fillna(0).astype(bool)
        # Create groups that increment at each event
        event_cumsum = event_series.cumsum()
        # Count position within each group (bars since last event)
        bars_since = event_cumsum.groupby(event_cumsum).cumcount()
        # Set to high value where no event has occurred yet
        bars_since = bars_since.where(event_cumsum > 0, 999)
        return bars_since
    
    df['bars_since_cross_up'] = calculate_bars_since_event(df['macd_cross_up'])
    df['bars_since_cross_down'] = calculate_bars_since_event(df['macd_cross_down'])
    
    # ===== Volatility Features =====
    df['atr_14'] = calculate_atr(df, 14)
    df['atr_7'] = calculate_atr(df, 7)
    
    # 🌟 NORMALIZED ATR (% of close) — Stationary across price levels!
    df['atr_14_pct'] = (df['atr_14'] / df['close']) * 100
    df['atr_7_pct'] = (df['atr_7'] / df['close']) * 100
    df['volatility_7'] = df['returns'].rolling(7).std()
    df['volatility_14'] = df['returns'].rolling(14).std()
    df['volatility_21'] = df['returns'].rolling(21).std()
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # ===== Momentum Features =====
    df['rsi_14'] = calculate_rsi(df['close'], 14)
    df['rsi_7'] = calculate_rsi(df['close'], 7)
    df['rsi_9'] = calculate_rsi(df['close'], 9)  # Lorentzian feature
    
    # CCI (Commodity Channel Index) — Lorentzian Classification feature
    df['cci_20'] = calculate_cci(df, 20)
    
    # WaveTrend Oscillator — Lorentzian Classification feature
    df = calculate_wavetrend(df, n1=10, n2=11)
    df['wt_histogram'] = df['wt1'] - df['wt2']  # WT momentum (like MACD histogram)
    
    # RSI Slope (momentum of momentum)
    # BUG FIX: diff(-3) looks into the FUTURE. We must use historical diff: diff(3)
    df['rsi_slope'] = df['rsi_14'].diff(3) / 3
    df['rsi_9_slope'] = df['rsi_9'].diff(3) / 3  # RSI(9) momentum
    df['cci_20_slope'] = df['cci_20'].diff(3) / 3  # CCI momentum
    df['wt1_slope'] = df['wt1'].diff(3) / 3  # WaveTrend momentum
    
    # 🌟 NEW: Volatility Squeeze (Bollinger Bands vs Keltner Channels)
    # Keltner Channels (21-period to match existing EMA, 1.5x ATR)
    df['kc_middle'] = df['ema_21']
    df['kc_upper'] = df['kc_middle'] + 1.5 * df['atr_14']
    df['kc_lower'] = df['kc_middle'] - 1.5 * df['atr_14']
    
    # Squeeze: BB inside KC indicates contracted volatility ready to expand
    # If bb_upper < kc_upper and bb_lower > kc_lower -> Squeeze is ON (value < 0)
    df['bb_width_vs_kc'] = (df['bb_upper'] - df['bb_lower']) / (df['kc_upper'] - df['kc_lower'])
    df['is_squeeze'] = (df['bb_width_vs_kc'] < 1.0).astype(int)
    # Bars in squeeze
    df['squeeze_duration'] = df.groupby((df['is_squeeze'] == 0).cumsum()).cumcount()
    
    # Stochastic
    df['stoch_k'] = calculate_stochastic(df, 14)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    
    # Rate of Change
    df['roc_7'] = df['close'].pct_change(7)
    df['roc_14'] = df['close'].pct_change(14)
    df['roc_21'] = df['close'].pct_change(21)
    
    # ADX (Average Directional Index) - NEW for TP prediction
    df = calculate_adx(df, 14)
    
    # ADX(20) for Lorentzian Classification features
    _adx20_df = calculate_adx(df, 20)
    df['adx_20'] = _adx20_df['adx']
    
    # ===== Volume Features =====
    df['volume_sma_7'] = df['volume'].rolling(7).mean()
    df['volume_sma_14'] = df['volume'].rolling(14).mean()
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    
    # 🌟 OPTIMIZED: Log-transform volume to handle heavy skewness (fat tails)
    # Raw volume_ratio can have extreme spikes (100x+) that bias the model
    df['volume_ratio'] = df['volume'] / (df['volume_sma_14'] + 1e-9)
    df['volume_log_ratio'] = np.log1p(df['volume_ratio'])  # log(1 + x) for stability
    
    df['volume_trend'] = df['volume_sma_7'] / (df['volume_sma_14'] + 1e-9)
    
    # Volume Spike (clipped for stability)
    df['volume_spike'] = (df['volume'] / (df['volume_sma_20'] + 1e-9)).clip(0, 5)
    
    # 🌟 NEW: Advanced Volume Profile (Buy vs Sell Pressure)
    # Estimate buying pressure based on where the close is relative to high/low
    # 1.0 = closed at high (all buy), 0.0 = closed at low (all sell)
    with np.errstate(divide='ignore', invalid='ignore'):
        close_location = (df['close'] - df['low']) / (df['high'] - df['low'])
        close_location = close_location.fillna(0.5)
    
    df['buy_volume'] = df['volume'] * close_location
    df['sell_volume'] = df['volume'] * (1 - close_location)
    
    df['buy_pressure_14'] = df['buy_volume'].rolling(14).sum() / (df['volume'].rolling(14).sum() + 1e-9)
    df['buy_pressure_14'] = df['buy_pressure_14'].fillna(0.5)
    
    # OBV
    df['obv'] = (np.sign(df['returns']) * df['volume']).cumsum()
    df['obv_sma'] = df['obv'].rolling(14).mean()
    df['obv_trend'] = df['obv'] / df['obv_sma']
    
    # ===== Advanced Statistical & Micro-structure Features (For SHAP/RFE) =====
    
    # 1. Candlestick & Micro-structure
    df['wick_to_body_ratio'] = (df['upper_shadow'] + df['lower_shadow']) / (df['body_size'] + 1e-8)
    df['is_doji'] = (df['body_size'] < 0.001).astype(int)
    
    df['rolling_min_24'] = df['low'].rolling(24).min()
    df['rolling_max_24'] = df['high'].rolling(24).max()
    df['sweep_low'] = ((df['low'] <= df['rolling_min_24'].shift(1)) & (df['close'] > df['rolling_min_24'].shift(1))).astype(int)
    df['sweep_high'] = ((df['high'] >= df['rolling_max_24'].shift(1)) & (df['close'] < df['rolling_max_24'].shift(1))).astype(int)
    
    # 2. Skewness / Kurtosis (Fat tails detection for risk)
    df['returns_skew_14'] = df['returns'].rolling(14).skew()
    df['returns_kurt_14'] = df['returns'].rolling(14).kurt()
    
    # 3. Robust Z-Scores (Outlier-resistant)
    median_24 = df['close'].rolling(24).median()
    iqr_24 = df['close'].rolling(24).quantile(0.75) - df['close'].rolling(24).quantile(0.25)
    df['price_robust_z'] = (df['close'] - median_24) / (iqr_24 + 1e-8)
    
    vol_median_24 = df['volume'].rolling(24).median()
    vol_iqr_24 = df['volume'].rolling(24).quantile(0.75) - df['volume'].rolling(24).quantile(0.25)
    df['volume_robust_z'] = (df['volume'] - vol_median_24) / (vol_iqr_24 + 1e-8)
    
    # 4. Advanced Oscillators (Williams %R, TSI, CMF)
    highest_high = df['high'].rolling(14).max()
    lowest_low = df['low'].rolling(14).min()
    df['williams_r'] = (highest_high - df['close']) / (highest_high - lowest_low + 1e-8) * -100
    
    diff = df['close'].diff()
    ema25_diff = diff.ewm(span=25, adjust=False).mean()
    ema13_ema25_diff = ema25_diff.ewm(span=13, adjust=False).mean()
    ema25_abs_diff = abs(diff).ewm(span=25, adjust=False).mean()
    ema13_ema25_abs_diff = ema25_abs_diff.ewm(span=13, adjust=False).mean()
    df['tsi'] = 100 * (ema13_ema25_diff / (ema13_ema25_abs_diff + 1e-8))
    
    mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-8)
    mf_volume = mf_multiplier * df['volume']
    df['cmf_20'] = mf_volume.rolling(20).sum() / (df['volume'].rolling(20).sum() + 1e-8)
    
    # 5. Price-Volume Divergence
    df['price_up_volume_down'] = ((df['returns'] > 0) & (df['volume'] < df['volume'].shift(1))).astype(int)
    df['price_down_volume_down'] = ((df['returns'] < 0) & (df['volume'] < df['volume'].shift(1))).astype(int)
    
    # 6. Vectorized Autocorrelation (Lag-1)
    ret_shift = df['returns'].shift(1)
    roll_cov = df['returns'].rolling(14).cov(ret_shift)
    roll_var = df['returns'].rolling(14).var()
    df['autocorr_1'] = roll_cov / (roll_var + 1e-8)
    
    # 7. Multi-timeframe proxies (Assuming base timeframe is often 1H -> 4H proxy)
    df['rsi_14_4x'] = calculate_rsi(df['close'], 14 * 4)
    df['roc_14_4x'] = df['close'].pct_change(14 * 4)
    
    # 8. Volume Profile & Order Flow Proxies
    # VWAP (Approximation over 24 periods)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['vwap_24'] = (typical_price * df['volume']).rolling(24).sum() / (df['volume'].rolling(24).sum() + 1e-8)
    df['dist_to_vwap_24'] = (df['close'] - df['vwap_24']) / (df['vwap_24'] + 1e-8)
    
    # 9. Efficiency Ratio (Kaufman)
    # ER = Direction / Volatility
    direction = abs(df['close'] - df['close'].shift(14))
    volatility = abs(df['close'].diff()).rolling(14).sum()
    df['kaufman_er'] = direction / (volatility + 1e-8)
    
    # 10. Gap / FVG (Fair Value Gap) Features
    df['fvg_bullish'] = ((df['low'] > df['high'].shift(2)) & (df['close'] > df['open'])).astype(int)
    df['fvg_bearish'] = ((df['high'] < df['low'].shift(2)) & (df['close'] < df['open'])).astype(int)
    
    # ===== Market Regime =====
    df['is_trending'] = (abs(df['trend_7_21'] - 1) > 0.02).astype(int)
    df['is_volatile'] = (df['volatility_14'] > df['volatility_14'].rolling(50).mean()).astype(int)
    
    # 🌟 NEW: Volatility of Volatility (VoV) - Critical for risk prediction
    # High VoV = MAE becomes extremely unpredictable
    df['volatility_of_volatility'] = df['volatility_14'].rolling(14).std()
    
    # ===== Advanced Regime Features =====
    
    # 1. Khoảng cách an toàn tới các đường EMA (Tính bằng %)
    # Giúp mô hình biết giá đã đi quá xa trung bình chưa (rủi ro đảo chiều)
    df['dist_to_ema_21_pct'] = (df['close'] - df['ema_21']) / df['ema_21']
    df['dist_to_ema_200_pct'] = (df['close'] - df['ema_200']) / df['ema_200']
    
    # 2. ADX Trend State (-1, 0, 1)
    # Kết hợp ADX và SMA để xác định rõ: Sideway (0), Uptrend mạnh (1), Downtrend mạnh (-1)
    df['trend_state'] = np.where(
        df['adx'] < 20, 0,  # Không có xu hướng
        np.where(df['close'] > df['sma_50'], 1, -1) # Có xu hướng, check xem trend lên hay xuống
    )
    
    # 3. Phân loại thanh khoản (Liquidity Regime)
    # Lọc các coin rác có volume chồi sụt bất thường
    # Use moving average of volume directly to calculate standard deviation since apply with lambda on rolling is slow, but we stick to user req
    # An optimization: rolling with pandas native methods where possible, but apply is ok for now.
    def calc_cv(x):
        m = np.mean(x)
        return np.std(x)/m if m > 0 else 0
        
    df['liquidity_regime'] = df['volume_sma_14'].rolling(30).apply(calc_cv, raw=True)
    
    # ===== REGIME-CONDITIONED FEATURES =====
    # These help the model learn different patterns per market regime.
    # The idea: RSI=30 in a bull market has different meaning than RSI=30 in a bear market.
    # By creating interaction features, the model can distinguish these scenarios.
    
    # Note: These require BTC macro data to be merged later (in build_dataset).
    # We use placeholders here that will be populated after BTC merge.
    # For now, create features based on local trend_state.
    
    # 1. RSI conditioned on trend state
    # In uptrend, low RSI is "buy the dip"; in downtrend, low RSI can keep dropping
    df['rsi_14_trend_adj'] = df['rsi_14'] * np.where(df['trend_state'] == 1, 1.2, 
                                                     np.where(df['trend_state'] == -1, 0.8, 1.0))
    
    # 2. MACD histogram strength relative to volatility
    # High histogram in low vol = strong signal; high histogram in high vol = noise
    df['macd_hist_vol_adj'] = df['histogram_pct'] / (df['volatility_14'] * 100 + 0.1)
    
    # 3. Volume spike significance in different regimes
    # Volume spike in uptrend often = continuation; in downtrend often = capitulation
    df['volume_spike_trend'] = df['volume_spike'] * df['trend_state']
    
    # 4. Momentum persistence (is the trend accelerating or decelerating?)
    df['momentum_persistence'] = (df['roc_7'] > df['roc_14']).astype(int) * 2 - 1  # 1 = accelerating, -1 = decelerating
    
    # 5. Mean reversion potential (how far stretched from equilibrium?)
    df['mean_reversion_z'] = (df['close'] - df['sma_50']) / (df['bb_std'] + 1e-10)
    df['mean_reversion_z'] = df['mean_reversion_z'].clip(-4, 4)  # Clip extreme values
    
    # ===== LEADING FEATURES (Not Lagging) =====
    
    # 🌟 1. Bollinger Band Squeeze Strength
    # Narrow BB before cross = high volatility expansion potential = higher MFE
    df['bb_squeeze_strength'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-10)
    df['bb_squeeze_pct'] = df['bb_squeeze_strength'] / df['bb_squeeze_strength'].rolling(50).mean()
    
    # 🌟 2. MACD Histogram Acceleration (Rate of change of momentum)
    # Not just histogram value, but how fast it's expanding/contracting
    df['histogram_velocity'] = df['histogram_pct'].diff(3)  # 3-bar momentum change
    df['histogram_acceleration'] = df['histogram_velocity'].diff(3)  # 2nd derivative
    
    # 🌟 3. Distance to Key Support/Resistance (EMA 200)
    # MAE often gets support from EMA 200 in uptrends
    df['dist_to_ema_200'] = (df['close'] - df['ema_200']) / (df['close'] + 1e-10)
    df['dist_to_sma_200'] = (df['close'] - df['sma_200']) / (df['close'] + 1e-10)
    
    # 🌟 4. Cyclical Time Features (Leading indicators for session patterns)
    # Crypto markets have patterns: Asian session vs US session, weekends vs weekdays
    if 'timestamp' in df.columns and hasattr(df['timestamp'].iloc[0], 'hour'):
        df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
    
    # 🌟 5. Relative Performance vs BTC (Alpha)
    # If coin +2% but BTC +5% → coin is actually weak (bearish signal)
    # This will be populated after BTC merge in build_dataset()
    # Placeholder here for consistency
    if 'btc_returns' in df.columns:
        df['relative_performance'] = df['returns'] - df['btc_returns']
        df['alpha_7d'] = df['relative_performance'].rolling(7).mean()
        df['alpha_14d'] = df['relative_performance'].rolling(14).mean()
    
    # 🌟 6. MACD Alignment Indicators
    # When MACD, Signal, and Histogram all pointing same direction = strong conviction
    df['macd_btc_aligned'] = 0  # Placeholder, populated after BTC merge
    df['trend_btc_align'] = 0   # Placeholder
    df['rs_vs_btc'] = 0.0       # Placeholder
    
    # ===== DERIVATIVES & MICRO-STRUCTURE FEATURES (Not Raw) =====
    # These rely on having 'sum_open_interest', 'fundingRate', 'top_ls_ratio', etc.
    
    # 1. Open Interest (OI)
    if 'sum_open_interest' in df.columns:
        oi_median_24 = df['sum_open_interest'].rolling(24).median()
        oi_iqr_24 = df['sum_open_interest'].rolling(24).quantile(0.75) - df['sum_open_interest'].rolling(24).quantile(0.25)
        df['oi_robust_z'] = (df['sum_open_interest'] - oi_median_24) / (oi_iqr_24 + 1e-8)
        
        # OI Rate of Change
        df['oi_roc_1'] = df['sum_open_interest'].pct_change(1)
        df['oi_roc_4'] = df['sum_open_interest'].pct_change(4)
        
        # Price-OI Regime (1: New longs, -1: New shorts, -2: Long liq, 2: Short cover)
        df['oi_price_trend'] = np.where(
            (df['returns'] > 0) & (df['oi_roc_1'] > 0), 1,  
            np.where((df['returns'] < 0) & (df['oi_roc_1'] > 0), -1,  
            np.where((df['returns'] < 0) & (df['oi_roc_1'] < 0), -2,  
            np.where((df['returns'] > 0) & (df['oi_roc_1'] < 0), 2, 0))) 
        )
        
        # CVD Proxy: Volume direction vs OI direction
        df['cvd_proxy_14'] = (df['oi_roc_1'] * df['volume'] * np.sign(df['returns'])).rolling(14).sum()
        
    # 2. Funding Rate
    fund_col = 'fundingRate' if 'fundingRate' in df.columns else ('funding_rate' if 'funding_rate' in df.columns else None)
    if fund_col:
        fr_median_24 = df[fund_col].rolling(24).median()
        fr_iqr_24 = df[fund_col].rolling(24).quantile(0.75) - df[fund_col].rolling(24).quantile(0.25)
        df['funding_robust_z'] = (df[fund_col] - fr_median_24) / (fr_iqr_24 + 1e-8)
        
        # Trạng thái bẫy funding (Funding trap)
        df['funding_trap_long'] = ((df['returns'] < -0.01) & (df['funding_robust_z'] > 1.5)).astype(int)
        df['funding_trap_short'] = ((df['returns'] > 0.01) & (df['funding_robust_z'] < -1.5)).astype(int)

    # 3. Long/Short Ratios
    if 'top_ls_ratio' in df.columns:
        ls_median = df['top_ls_ratio'].rolling(24).median()
        ls_iqr = df['top_ls_ratio'].rolling(24).quantile(0.75) - df['top_ls_ratio'].rolling(24).quantile(0.25)
        df['top_ls_robust_z'] = (df['top_ls_ratio'] - ls_median) / (ls_iqr + 1e-8)
        
        # Retail vs Smart Money Divergence
        if 'global_ls_ratio' in df.columns:
            df['ls_divergence'] = df['top_ls_ratio'] - df['global_ls_ratio']
            
            # Z-Score of divergence to normalize it
            div_median = df['ls_divergence'].rolling(24).median()
            div_iqr = df['ls_divergence'].rolling(24).quantile(0.75) - df['ls_divergence'].rolling(24).quantile(0.25)
            df['ls_divergence_z'] = (df['ls_divergence'] - div_median) / (div_iqr + 1e-8)

    return df


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate ADX (Average Directional Index) - measures trend strength.
    ADX > 25: Strong trend (good for TP targets)
    ADX < 20: Weak/No trend (conservative TP targets)
    """
    df = df.copy()
    
    # +DM and -DM
    df['_plus_dm'] = df['high'].diff()
    df['_minus_dm'] = -df['low'].diff()
    
    # Keep only positive values and handle comparison
    df['_plus_dm'] = df['_plus_dm'].where(
        (df['_plus_dm'] > df['_minus_dm']) & (df['_plus_dm'] > 0), 0
    )
    df['_minus_dm'] = df['_minus_dm'].where(
        (df['_minus_dm'] > df['_plus_dm']) & (df['_minus_dm'] > 0), 0
    )
    
    # ATR for normalization
    atr = df['atr_14'] if 'atr_14' in df.columns else calculate_atr(df, period)
    
    # +DI and -DI
    df['_plus_di'] = 100 * (df['_plus_dm'].ewm(span=period, adjust=False).mean() / (atr + 1e-10))
    df['_minus_di'] = 100 * (df['_minus_dm'].ewm(span=period, adjust=False).mean() / (atr + 1e-10))
    
    # DX and ADX
    df['_dx'] = 100 * abs(df['_plus_di'] - df['_minus_di']) / (df['_plus_di'] + df['_minus_di'] + 1e-10)
    df['adx'] = df['_dx'].ewm(span=period, adjust=False).mean()
    
    # Clean up intermediate columns
    df = df.drop(columns=['_plus_dm', '_minus_dm', '_plus_di', '_minus_di', '_dx'], errors='ignore')
    
    return df


def calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Calculate Average True Range"""
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calculate_rsi(prices: pd.Series, period: int) -> pd.Series:
    """Calculate RSI"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_stochastic(df: pd.DataFrame, period: int) -> pd.Series:
    """Calculate Stochastic %K"""
    low_min = df['low'].rolling(period).min()
    high_max = df['high'].rolling(period).max()
    return 100 * (df['close'] - low_min) / (high_max - low_min)


def calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate Commodity Channel Index (CCI).
    
    CCI measures deviation of price from its statistical mean.
    Used in Lorentzian Classification as oscillator feature.
    
    CCI > +100: Overbought
    CCI < -100: Oversold
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = typical_price.rolling(period).mean()
    mad = typical_price.rolling(period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    cci = (typical_price - sma_tp) / (0.015 * mad + 1e-10)
    return cci


def calculate_wavetrend(df: pd.DataFrame, n1: int = 10, n2: int = 11) -> pd.DataFrame:
    """
    Calculate WaveTrend indicator (LazyBear's version).
    
    WaveTrend is an oscillator combining price action with EMA smoothing.
    Used in Lorentzian Classification for oscillator confluence.
    
    Pine Script equivalent:
        ap = hlc3
        esa = EMA(ap, n1)
        d = EMA(|ap - esa|, n1)
        ci = (ap - esa) / (0.015 * d)
        wt1 = EMA(ci, n2)
        wt2 = SMA(wt1, 4)
    
    Args:
        df: DataFrame with OHLCV data
        n1: Channel length (default 10)
        n2: Average length (default 11)
    
    Returns:
        DataFrame with wt1, wt2, wt_cross_up, wt_cross_down columns added
    """
    df = df.copy()
    
    # HLC3 (typical price)
    ap = (df['high'] + df['low'] + df['close']) / 3
    
    # Exponential smoothing
    esa = ap.ewm(span=n1, adjust=False).mean()
    d = (ap - esa).abs().ewm(span=n1, adjust=False).mean()
    
    # Channel Index
    ci = (ap - esa) / (0.015 * d + 1e-10)
    
    # WaveTrend lines
    df['wt1'] = ci.ewm(span=n2, adjust=False).mean()
    df['wt2'] = df['wt1'].rolling(4).mean()
    
    # WaveTrend crossover detection (analogous to MACD cross)
    df['wt_cross_up'] = (
        (df['wt1'] > df['wt2']) & 
        (df['wt1'].shift(1) <= df['wt2'].shift(1))
    ).astype(int)
    df['wt_cross_down'] = (
        (df['wt1'] < df['wt2']) & 
        (df['wt1'].shift(1) >= df['wt2'].shift(1))
    ).astype(int)
    
    return df


def calculate_features_grouped(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate features with proper groupby to prevent data bleeding between symbols.
    This ensures moving averages and other indicators don't use data from other coins.
    """
    return df.groupby('symbol', group_keys=False).apply(
        lambda x: calculate_features(x.sort_values('timestamp'))
    ).reset_index(drop=True)


def generate_labels(
    df: pd.DataFrame, 
    tp_pct: float = 0.03, 
    sl_pct: float = 0.015, 
    max_bars: int = 10,
    use_atr: bool = True,
    atr_tp_mult: float = 3.0,
    atr_sl_mult: float = 1.5,
    min_tp_pct: float = 0.01,
    max_tp_pct: float = 0.30
) -> pd.DataFrame:
    """
    Generate labels for each MACD crossover using Triple Barrier Method.
    
    Triple Barrier Method:
    - Upper barrier (TP): Dynamic based on ATR or fixed %
    - Lower barrier (SL): Dynamic based on ATR or fixed %
    - Time barrier (Expiration): max_bars - close position if no TP/SL hit
    
    Args:
        df: DataFrame with OHLCV and features (must have 'atr_14' column if use_atr=True)
        tp_pct: Fixed TP % (used when use_atr=False)
        sl_pct: Fixed SL % (used when use_atr=False)
        max_bars: Time barrier - max bars before forced exit
        use_atr: If True, use ATR-based dynamic TP/SL instead of fixed %
        atr_tp_mult: ATR multiplier for TP (default 3.0x ATR)
        atr_sl_mult: ATR multiplier for SL (default 1.5x ATR)
        min_tp_pct: Minimum TP % (prevent TP from being too small)
        max_tp_pct: Maximum TP % (prevent TP from being unrealistic)
    
    Labels:
    - 1: Good entry (hit TP first)
    - 0: Bad entry (hit SL first or timeout loss)
    
    ⚠️ DATA LEAKAGE WARNING:
    The following columns use future data and MUST be dropped before ML training:
    - max_profit, max_drawdown, bars_to_tp, bars_to_sl, trade_result, label
    - tp_pct_used, sl_pct_used (dynamic targets)
    
    Use get_feature_columns() to get safe feature columns for training.
    """
    # Process each symbol separately to avoid cross-symbol lookups
    if 'symbol' in df.columns:
        return df.groupby('symbol', group_keys=False).apply(
            lambda x: _generate_labels_triple_barrier(
                x.sort_values('timestamp'), 
                tp_pct, sl_pct, max_bars,
                use_atr, atr_tp_mult, atr_sl_mult,
                min_tp_pct, max_tp_pct
            )
        ).reset_index(drop=True)
    else:
        return _generate_labels_triple_barrier(
            df, tp_pct, sl_pct, max_bars,
            use_atr, atr_tp_mult, atr_sl_mult,
            min_tp_pct, max_tp_pct
        )


def generate_mfe_mae_labels(df: pd.DataFrame, max_bars: int = 14) -> pd.DataFrame:
    """Generate MFE/MAE regression targets using vectorized operations.

    🚀 OPTIMIZED: ~100x faster than loop-based approach using rolling windows.
    
    Args:
        df: input DataFrame with columns ['close','high','low','macd_cross_up']
        max_bars: number of subsequent bars to inspect

    Returns:
        DataFrame copy with two new columns: ``mfe_pct`` and ``mae_pct``.
        Also adds normalized versions: ``mfe_atr_ratio`` and ``mae_atr_ratio``.
    """
    df = df.copy()

    # 🚀 VECTORIZED: Use rolling windows with shift for future lookback
    # This is 100x faster than looping through crossover indices
    
    # Calculate future max high and min low over next max_bars periods
    # shift(-max_bars) looks ahead, rolling(max_bars).max() gets max in window
    future_high_max = df['high'].shift(-max_bars).rolling(window=max_bars, min_periods=1).max()
    future_low_min = df['low'].shift(-max_bars).rolling(window=max_bars, min_periods=1).min()
    
    # Calculate MFE/MAE as percentage returns
    df['mfe_pct'] = (future_high_max - df['close']) / df['close']
    df['mae_pct'] = (future_low_min - df['close']) / df['close']
    
    # Keep only at crossover points (others set to NaN)
    cross_mask = df['macd_cross_up'] == 1
    df.loc[~cross_mask, 'mfe_pct'] = np.nan
    df.loc[~cross_mask, 'mae_pct'] = np.nan
    

    return df


def _generate_labels_triple_barrier(
    df: pd.DataFrame, 
    tp_pct: float, 
    sl_pct: float, 
    max_bars: int,
    use_atr: bool,
    atr_tp_mult: float,
    atr_sl_mult: float,
    min_tp_pct: float = 0.01,
    max_tp_pct: float = 0.30
) -> pd.DataFrame:
    """
    Triple Barrier Method label generation for a single symbol.
    """
    df = df.copy().reset_index(drop=True)
    n = len(df)
    
    # Initialize output columns
    df['label'] = np.nan
    df['max_profit'] = np.nan
    df['max_drawdown'] = np.nan
    df['bars_to_tp'] = np.nan
    df['bars_to_sl'] = np.nan
    df['trade_result'] = ''
    df['tp_pct_used'] = np.nan  # Track actual TP% used
    df['sl_pct_used'] = np.nan  # Track actual SL% used
    
    # Get crossover indices
    cross_up_mask = df['macd_cross_up'] == 1
    cross_down_mask = df['macd_cross_down'] == 1
    crossover_mask = cross_up_mask | cross_down_mask
    crossover_indices = np.where(crossover_mask)[0]
    
    # Filter out indices too close to the end
    crossover_indices = crossover_indices[crossover_indices < n - max_bars]
    
    if len(crossover_indices) == 0:
        return df
    
    # Pre-extract arrays for speed
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    is_long_arr = cross_up_mask.values
    
    # Get ATR for dynamic TP/SL
    if use_atr and 'atr_14' in df.columns:
        atr = df['atr_14'].values
        # Calculate ATR as % of price for each bar
        atr_pct = atr / close
    else:
        atr_pct = None
    
    # Process each crossover
    labels = []
    max_profits = []
    max_drawdowns = []
    bars_to_tps = []
    bars_to_sls = []
    results = []
    tp_pcts_used = []
    sl_pcts_used = []
    
    for idx in crossover_indices:
        entry_price = close[idx]
        is_long = is_long_arr[idx]
        
        # Calculate dynamic TP/SL based on ATR
        if use_atr and atr_pct is not None and not np.isnan(atr_pct[idx]):
            # ATR-based dynamic targets
            current_atr_pct = atr_pct[idx]
            
            # Clamp ATR to reasonable range (0.5% - 10%)
            current_atr_pct = np.clip(current_atr_pct, 0.005, 0.10)
            
            actual_tp_pct = current_atr_pct * atr_tp_mult
            actual_sl_pct = current_atr_pct * atr_sl_mult
            
            # Clamp final TP/SL to reasonable bounds for swing trading
            # Allowing TP to float between 3% and 20% based on actual coin volatility
            actual_tp_pct = np.clip(actual_tp_pct, 0.03, 0.20)
            actual_sl_pct = np.clip(actual_sl_pct, 0.015, 0.075)
        else:
            # Fixed targets
            actual_tp_pct = tp_pct
            actual_sl_pct = sl_pct
        
        # Calculate TP/SL levels
        if is_long:
            tp_price = entry_price * (1 + actual_tp_pct)
            sl_price = entry_price * (1 - actual_sl_pct)
        else:
            tp_price = entry_price * (1 - actual_tp_pct)
            sl_price = entry_price * (1 + actual_sl_pct)
        
        # Get future window
        future_start = idx + 1
        future_end = min(idx + max_bars + 1, n)
        future_high = high[future_start:future_end]
        future_low = low[future_start:future_end]
        future_close = close[future_end - 1] if future_end > future_start else entry_price
        
        if len(future_high) == 0:
            labels.append(np.nan)
            max_profits.append(np.nan)
            max_drawdowns.append(np.nan)
            bars_to_tps.append(np.nan)
            bars_to_sls.append(np.nan)
            results.append('')
            tp_pcts_used.append(np.nan)
            sl_pcts_used.append(np.nan)
            continue
        
        # Calculate max profit and drawdown
        if is_long:
            profits = (future_high - entry_price) / entry_price
            drawdowns = (entry_price - future_low) / entry_price
            tp_hits = np.where(future_high >= tp_price)[0]
            sl_hits = np.where(future_low <= sl_price)[0]
        else:
            profits = (entry_price - future_low) / entry_price
            drawdowns = (future_high - entry_price) / entry_price
            tp_hits = np.where(future_low <= tp_price)[0]
            sl_hits = np.where(future_high >= sl_price)[0]
        
        max_profit = profits.max() if len(profits) > 0 else 0
        max_drawdown = drawdowns.max() if len(drawdowns) > 0 else 0
        
        # Find first TP/SL hit (bar index, 1-based)
        bars_to_tp = (tp_hits[0] + 1) if len(tp_hits) > 0 else max_bars
        bars_to_sl = (sl_hits[0] + 1) if len(sl_hits) > 0 else max_bars
        hit_tp = len(tp_hits) > 0
        hit_sl = len(sl_hits) > 0
        
        # Determine label using Triple Barrier logic
        if hit_tp and hit_sl:
            # Both barriers hit - which was first?
            if bars_to_tp <= bars_to_sl:
                label = 1
                result = 'TP_HIT'
            else:
                label = 0
                result = 'SL_HIT'
        elif hit_tp:
            label = 1
            result = 'TP_HIT'
        elif hit_sl:
            label = 0
            result = 'SL_HIT'
        else:
            # Time barrier hit - use final PnL
            final_pnl = (future_close - entry_price) / entry_price
            if not is_long:
                final_pnl = -final_pnl
                
            # REQUIRE AT LEAST 1% PROFIT FOR A TIMEOUT_WIN TO BE CONSIDERED A WIN (LABEL 1)
            # Otherwise, a tiny 0.01% profit over 10 days is essentially noise/loss after fees.
            if final_pnl > 0.01:
                label = 1
                result = 'TIMEOUT_WIN'
            elif final_pnl > 0:
                label = 0
                result = 'TIMEOUT_FLAT' 
            else:
                label = 0
                result = 'TIMEOUT_LOSS'
        
        # --- ROBUST SL LABELING (NEW) ---
        # If the trade was a success (hit TP), the "optimal" SL would have been 
        # the max drawdown experienced plus a small safety buffer.
        if label == 1:
            # Winner: Optimal SL = MAE + buffer
            # We want to catch the "survival" SL
            optimal_sl = max_drawdown * 1.5 + 0.005 # 1.5x MAE + 0.5% buffer
        else:
            # Loser: Don't learn to have a massive SL for a trade that fails anyway.
            # Stick to the baseline ATR-based SL.
            optimal_sl = actual_sl_pct
            
        # Final clamp for training stability
        optimal_sl = np.clip(optimal_sl, 0.01, 0.20)
        
        labels.append(label)
        max_profits.append(max_profit)
        max_drawdowns.append(max_drawdown)
        bars_to_tps.append(bars_to_tp)
        bars_to_sls.append(bars_to_sl)
        results.append(result)
        tp_pcts_used.append(actual_tp_pct)
        sl_pcts_used.append(optimal_sl) # USE ROBUST SL FOR TRAINING LABEL
    
    # Assign results back to DataFrame
    df.loc[crossover_indices, 'label'] = labels
    df.loc[crossover_indices, 'max_profit'] = max_profits
    df.loc[crossover_indices, 'max_drawdown'] = max_drawdowns
    df.loc[crossover_indices, 'bars_to_tp'] = bars_to_tps
    df.loc[crossover_indices, 'bars_to_sl'] = bars_to_sls
    df.loc[crossover_indices, 'trade_result'] = results
    df.loc[crossover_indices, 'tp_pct_used'] = tp_pcts_used
    df.loc[crossover_indices, 'sl_pct_used'] = sl_pcts_used
    
    return df


def _generate_labels_vectorized(df: pd.DataFrame, tp_pct: float, sl_pct: float, max_bars: int) -> pd.DataFrame:
    """
    DEPRECATED: Use _generate_labels_triple_barrier instead.
    Kept for backward compatibility.
    """
    return _generate_labels_triple_barrier(
        df, tp_pct, sl_pct, max_bars,
        use_atr=False, atr_tp_mult=3.0, atr_sl_mult=1.5
    )


def _generate_labels_vectorized_old(df: pd.DataFrame, tp_pct: float, sl_pct: float, max_bars: int) -> pd.DataFrame:
    """
    OLD: Vectorized label generation for a single symbol.
    Uses NumPy for ~10x speedup over loop-based approach.
    """
    df = df.copy().reset_index(drop=True)
    n = len(df)
    
    # Initialize output columns
    df['label'] = np.nan
    df['max_profit'] = np.nan
    df['max_drawdown'] = np.nan
    df['bars_to_tp'] = np.nan
    df['bars_to_sl'] = np.nan
    df['trade_result'] = ''
    
    # Get crossover indices
    cross_up_mask = df['macd_cross_up'] == 1
    cross_down_mask = df['macd_cross_down'] == 1
    crossover_mask = cross_up_mask | cross_down_mask
    crossover_indices = np.where(crossover_mask)[0]
    
    # Filter out indices too close to the end
    crossover_indices = crossover_indices[crossover_indices < n - max_bars]
    
    if len(crossover_indices) == 0:
        return df
    
    # Pre-extract arrays for speed
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    is_long_arr = cross_up_mask.values
    
    # Process each crossover
    labels = []
    max_profits = []
    max_drawdowns = []
    bars_to_tps = []
    bars_to_sls = []
    results = []
    
    for idx in crossover_indices:
        entry_price = close[idx]
        is_long = is_long_arr[idx]
        
        # Calculate TP/SL levels
        if is_long:
            tp_price = entry_price * (1 + tp_pct)
            sl_price = entry_price * (1 - sl_pct)
        else:
            tp_price = entry_price * (1 - tp_pct)
            sl_price = entry_price * (1 + sl_pct)
        
        # Get future window
        future_start = idx + 1
        future_end = min(idx + max_bars + 1, n)
        future_high = high[future_start:future_end]
        future_low = low[future_start:future_end]
        future_close = close[future_end - 1] if future_end > future_start else entry_price
        
        if len(future_high) == 0:
            labels.append(np.nan)
            max_profits.append(np.nan)
            max_drawdowns.append(np.nan)
            bars_to_tps.append(np.nan)
            bars_to_sls.append(np.nan)
            results.append('')
            continue
        
        # Calculate max profit and drawdown
        if is_long:
            profits = (future_high - entry_price) / entry_price
            drawdowns = (entry_price - future_low) / entry_price
            tp_hits = np.where(future_high >= tp_price)[0]
            sl_hits = np.where(future_low <= sl_price)[0]
        else:
            profits = (entry_price - future_low) / entry_price
            drawdowns = (future_high - entry_price) / entry_price
            tp_hits = np.where(future_low <= tp_price)[0]
            sl_hits = np.where(future_high >= sl_price)[0]
        
        max_profit = profits.max() if len(profits) > 0 else 0
        max_drawdown = drawdowns.max() if len(drawdowns) > 0 else 0
        
        # Find first TP/SL hit (bar index, 1-based)
        bars_to_tp = (tp_hits[0] + 1) if len(tp_hits) > 0 else max_bars
        bars_to_sl = (sl_hits[0] + 1) if len(sl_hits) > 0 else max_bars
        hit_tp = len(tp_hits) > 0
        hit_sl = len(sl_hits) > 0
        
        # Determine label
        if hit_tp and (not hit_sl or bars_to_tp <= bars_to_sl):
            label = 1
            result = 'TP_HIT'
        elif hit_sl and (not hit_tp or bars_to_sl < bars_to_tp):
            label = 0
            result = 'SL_HIT'
        else:
            # Timeout - use final PnL
            final_pnl = (future_close - entry_price) / entry_price
            if not is_long:
                final_pnl = -final_pnl
            label = 1 if final_pnl > 0 else 0
            result = 'TIMEOUT_WIN' if final_pnl > 0 else 'TIMEOUT_LOSS'
        
        labels.append(label)
        max_profits.append(max_profit)
        max_drawdowns.append(max_drawdown)
        bars_to_tps.append(bars_to_tp)
        bars_to_sls.append(bars_to_sl)
        results.append(result)
    
    # Assign results back to DataFrame
    df.loc[crossover_indices, 'label'] = labels
    df.loc[crossover_indices, 'max_profit'] = max_profits
    df.loc[crossover_indices, 'max_drawdown'] = max_drawdowns
    df.loc[crossover_indices, 'bars_to_tp'] = bars_to_tps
    df.loc[crossover_indices, 'bars_to_sl'] = bars_to_sls
    df.loc[crossover_indices, 'trade_result'] = results
    
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Get safe feature columns for ML training (no data leakage).
    Excludes: label, max_profit, max_drawdown, bars_to_tp, bars_to_sl, trade_result,
              tp_pct_used, sl_pct_used, timestamp, symbol, and other non-feature columns.
    """
    exclude_cols = {
        # Target and leakage columns
        'label', 'max_profit', 'max_drawdown', 'bars_to_tp', 'bars_to_sl', 'trade_result',
        'tp_pct_used', 'sl_pct_used',  # Dynamic targets (leakage!)
        'optimal_tp_pct', 'mfe', 'mfe_bar', 'trend_continuation', 'rr_ratio',  # TP predictor targets
        # Metadata columns
        'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume',
        # Raw indicator values (keep normalized versions)
        'obv', 'bb_middle', 'bb_std', 'bb_upper', 'bb_lower',
        'sma_7', 'sma_14', 'sma_21', 'sma_50', 'sma_100', 'sma_200',
        'ema_7', 'ema_14', 'ema_21', 'ema_50', 'ema_100', 'ema_200',
        'volume_sma_7', 'volume_sma_14', 'volume_sma_20', 'obv_sma'
    }
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def process_symbol(symbol: str, timeframe: str = '1d', include_funding: bool = True) -> pd.DataFrame:
    """
    Process a single symbol: load, resample, merge external data, then add features.
    Features are calculated per-symbol to avoid data bleeding.
    
    Args:
        symbol: Trading pair (e.g., 'BTC', 'ETH')
        timeframe: Target timeframe ('1h', '4h', '8h', '12h', '1d')
        include_funding: Whether to include funding rate data
    """
    print(f"Processing {symbol}...")
    
    # Load 1h data
    df_1h = load_ohlcv_1h(symbol)
    if df_1h.empty:
        return pd.DataFrame()
    
    # Resample to target timeframe
    df = resample_to_timeframe(df_1h, timeframe)
    if len(df) < 100:
        print(f"  ⚠️ Not enough data: {len(df)} bars")
        return pd.DataFrame()
    
    # Add symbol column FIRST (before features, for groupby compatibility)
    df['symbol'] = symbol
    
    timeframe_map = {'1h': '1H', '4h': '4H', '8h': '8H', '12h': '12H', '1d': '1D'}
    resample_rule = timeframe_map.get(timeframe, '1D')
    
    # Load and resample Funding data
    if include_funding:
        df_funding = load_funding(symbol)
        if not df_funding.empty:
            df_funding = df_funding.set_index('timestamp')
            df_funding_resampled = df_funding['funding_rate'].resample(resample_rule).mean().reset_index()
            # Also get funding rate sum (accumulated)
            df_funding_sum = df_funding['funding_rate'].resample(resample_rule).sum().reset_index()
            
            df_funding_resampled.columns = ['timestamp', 'funding_rate_avg']
            df_funding_resampled['funding_rate'] = df_funding_resampled['funding_rate_avg'] # backward compat
            df_funding_resampled['funding_rate_sum'] = df_funding_sum['funding_rate']
            
            df = df.merge(df_funding_resampled, on='timestamp', how='left')
            df['funding_rate'] = df['funding_rate'].fillna(0)
            df['funding_rate_avg'] = df['funding_rate_avg'].fillna(0)
            df['funding_rate_sum'] = df['funding_rate_sum'].fillna(0)
        else:
            df['funding_rate'] = 0.0
    else:
        df['funding_rate'] = 0.0

    # Load and resample Derivatives data
    df_deriv = load_derivatives(symbol)
    if not df_deriv.empty:
        df_deriv = df_deriv.set_index('timestamp')
        
        deriv_cols = ['sum_open_interest', 'top_ls_ratio', 'global_ls_ratio']
        cols_to_resample = [c for c in deriv_cols if c in df_deriv.columns]
        
        if cols_to_resample:
            # We take the last value of the period for snapshot metrics like OI
            df_deriv_resampled = df_deriv[cols_to_resample].resample(resample_rule).last().reset_index()
            df = df.merge(df_deriv_resampled, on='timestamp', how='left')
            
            for col in cols_to_resample:
                df[col] = df[col].ffill().fillna(0)
    else:
        df['sum_open_interest'] = 0.0
        df['top_ls_ratio'] = 1.0

    # Add features (per-symbol, calculate on fully merged df)
    df = calculate_features(df)
    
    print(f"  ✓ {len(df)} bars, {len(df.columns)} features")
    return df


def build_dataset(symbols: List[str] = None, timeframe: str = '1d', min_days: int = 365) -> pd.DataFrame:
    """
    Build full dataset from all symbols with Macro Market Regime.
    
    Args:
        symbols: List of symbols to process (None = auto-detect all)
        timeframe: Target timeframe ('1h', '4h', '8h', '12h', '1d')
        min_days: Minimum bars required (adjusted based on timeframe)
    """
    if symbols is None:
        # Get all symbols from ohlcv directory
        symbols = [f.stem.replace('_USDT', '') for f in OHLCV_DIR.glob('*.parquet')]
        # Filter out quarterly futures
        symbols = [s for s in symbols if not any(x in s for x in ['-26', '-25', '-24'])]
        symbols = symbols[:250]
    
    print(f"Processing {len(symbols)} symbols @ {timeframe} timeframe...")
    
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
        btc_df = process_symbol(btc_symbol, timeframe=timeframe)
        if not btc_df.empty:
            btc_context = btc_df[['timestamp', 'close', 'sma_200', 'adx', 'log_returns']].copy()
            btc_context.columns = ['timestamp', 'btc_close', 'btc_sma_200', 'btc_adx', 'btc_returns']
            
            # Đánh giá Regime của toàn thị trường dựa trên BTC
            btc_context['btc_is_bull_regime'] = (btc_context['btc_close'] > btc_context['btc_sma_200']).astype(int)
            # Đo lường độ mạnh xu hướng của BTC
            btc_context['btc_trend_strength'] = np.where(btc_context['btc_adx'] > 25, 1, 0)
    # ---------------------------------------------------------
    
    all_data = []
    for symbol in symbols:
        try:
            df = process_symbol(symbol, timeframe=timeframe)
            if not df.empty and len(df) >= min_days:
                
                # ---------------------------------------------------------
                # NEW: 2. Merge BTC Context & Calculate Relative Strength
                # ---------------------------------------------------------
                if not btc_context.empty:
                    df = df.merge(btc_context, on='timestamp', how='left')
                    
                    # Trám dữ liệu rỗng (nếu có) bằng ffill
                    fill_cols = ['btc_is_bull_regime', 'btc_trend_strength', 'btc_returns']
                    df[fill_cols] = df[fill_cols].ffill().fillna(0)
                    
                    # 🌟 Relative Strength (RS): Coin này đang mạnh hay yếu hơn BTC?
                    df['rs_vs_btc'] = df['log_returns'] - df['btc_returns']
                    df['rs_vs_btc_sma7'] = df['rs_vs_btc'].rolling(7).mean()
                    
                    # 🌟 Alpha calculation (outperformance vs BTC)
                    df['relative_performance'] = df['returns'] - df['btc_returns']
                    df['alpha_7d'] = df['relative_performance'].rolling(7).mean()
                    df['alpha_14d'] = df['relative_performance'].rolling(14).mean()
                    
                    # 🌟 MACD alignment with BTC trend
                    # When both coin and BTC MACD cross up together = strong conviction
                    if 'macd_cross_up' in df.columns:
                        # This needs BTC MACD data - placeholder for now
                        df['macd_btc_aligned'] = 0  # Will be 1 if both cross up same period
                        # Use merged column df['btc_is_bull_regime'], not original btc_context
                        df['trend_btc_align'] = (df['trend_state'] == 
                                                 np.where(df['btc_is_bull_regime'] == 1, 1, -1)).astype(int)
                # ---------------------------------------------------------
                
                all_data.append(df)
        except Exception as e:
            print(f"  ✗ Error processing {symbol}: {e}")
    
    if not all_data:
        print("No data processed!")
        return pd.DataFrame()
    
    # Combine all
    df_all = pd.concat(all_data, ignore_index=True)
    print(f"\n✓ Total: {len(df_all)} rows from {len(all_data)} symbols")
    
    return df_all


def apply_global_feature_shift(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies a T-1 shift to all predictive features. 
    This guarantees zero look-ahead bias if trading at the crossover confirm (Open of T+1).
    
    ⚠️  CRITICAL: BTC regime features (btc_is_bull_regime, btc_trend_strength) are also shifted
    because we trade at T+1 open, so we only know BTC state from T, not T+1.
    """
    if df.empty:
        return df
        
    df = df.copy()
    
    # Columns that should NOT be shifted (metadata and signal triggers)
    non_shift_cols = {
        'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume',
        'macd_cross_up', 'macd_cross_down', 'macd_crossover',
        'wt_cross_up', 'wt_cross_down',  # WaveTrend signals
        'lorentz_buy_signal', 'lorentz_sell_signal',  # Lorentzian entry signals
        'date', 'fundingTime'
    }
    
    # All feature columns need shift (including BTC features!)
    # We trade at T+1, so we only have info from T
    shift_cols = [c for c in df.columns if c not in non_shift_cols]
    
    # Group by symbol to prevent shifting across different coins
    df[shift_cols] = df.groupby('symbol', group_keys=False)[shift_cols].shift(1)
    
    # Drop rows where shift created NaN (first row per symbol)
    df = df.dropna(subset=shift_cols[:min(3, len(shift_cols))]) 
    
    return df

def save_processed_data(df: pd.DataFrame, filename: str = 'features_1d.parquet'):
    """Save processed data"""
    PROCESSED_DIR.mkdir(exist_ok=True)
    output_path = PROCESSED_DIR / filename
    df.to_parquet(output_path, index=False)
    print(f"✓ Saved to {output_path}")


if __name__ == '__main__':
    
    mfe_mae = True
    horizon=24
    timeframe ='1h'#choices=['1h', '4h', '8h', '12h', '1d']
    
    exchange_data_dir = Path('/kaggle/input/datasets/hungbui317/macd-coin/macd-overlay - Copy/data')
    if exchange_data_dir.exists():
        set_data_directory(exchange_data_dir)
    else:
        print(f"⚠️  -data directory not found, using default bitget-data")
    
    # Build dataset
    print("="*60)
    print("Building ML Dataset")
    print("="*60)
    
    if True:
        # Process ALL symbols
        print("Processing ALL symbols in data/ohlcv folder...")
        symbols = None  # build_dataset will auto-detect
        output_file =  f'features_{timeframe}_full.parquet'
    else:
        # Process top coins for testing
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 
                   'ADAUSDT', 'DOGEUSDT', 'DOTUSDT', 'LINKUSDT', 'AVAXUSDT']
        print(f"Processing {len(symbols)} test symbols...")
        output_file =  f'features_{timeframe}_test.parquet'
    
    df = build_dataset(symbols, timeframe=timeframe)
    
    if not df.empty:
        print(f"\nApplying T-1 Global Shift to features to prevent Look-ahead bias...")
        df = apply_global_feature_shift(df)
        
        if mfe_mae:
            print(f" Generating MFE/MAE regression labels (horizon={horizon} bars @ {timeframe})...")
            df = generate_mfe_mae_labels(df, max_bars=horizon)
        else:
            print(" Generating binary labels...")
            df = generate_labels(df)
            

        # 1. Provide stationary alternatives for features being dropped
        if 'btc_close' in df.columns and 'btc_sma_200' in df.columns:
            df['btc_dist_to_sma_200'] = (df['btc_close'] - df['btc_sma_200']) / (df['btc_sma_200'] + 1e-8)
        
        if 'obv' in df.columns and 'obv_sma' in df.columns and 'volume_sma_14' in df.columns:
            df['obv_oscillator'] = (df['obv'] - df['obv_sma']) / (df['volume_sma_14'] + 1e-8)
            
        if 'rolling_max_24' in df.columns and 'close' in df.columns:
            df['dist_to_max_24'] = (df['rolling_max_24'] - df['close']) / (df['close'] + 1e-8)
            
        if 'rolling_min_24' in df.columns and 'close' in df.columns:
            df['dist_to_min_24'] = (df['close'] - df['rolling_min_24']) / (df['close'] + 1e-8)
            
        if 'sum_open_interest' in df.columns and 'oi_robust_z' not in df.columns:
            df['oi_pct_change'] = df.groupby('symbol')['sum_open_interest'].pct_change().fillna(0)
            
        if 'ema_100' in df.columns and 'close' in df.columns:
            df['dist_to_ema_100'] = (df['close'] - df['ema_100']) / (df['ema_100'] + 1e-8)
            
        # 2. Drop the specified columns
        cols_to_drop = [
            'mae_atr_ratio', 'mae_pct', 'mfe_atr_ratio', 'mfe_pct', 
            'bb_lower', 'bb_middle', 'bb_upper', 'btc_close', 'btc_sma_200', 'dist_to_sma_200', 
            'ema_100', 'ema_14', 'ema_200', 'ema_21', 'ema_50', 'ema_7', 'hour_cos', 'hour_sin', 
            'kc_lower', 'kc_middle', 'kc_upper', 'macd', 'macd_btc_aligned', 'obv', 'obv_sma', 
            'rolling_max_24', 'rolling_min_24', 'signal', 'sma_100', 'sma_14', 'sma_200', 'sma_21', 
            'sma_50', 'sma_7', 'sma_89', 'sum_open_interest', 'vwap_24'
        ]
        
        dropped_cols = [c for c in cols_to_drop if c in df.columns]
        df = df.drop(columns=dropped_cols)
        print(f"Dropped {len(dropped_cols)} non-stationary or high NaN columns.")
        
        # Save
        save_processed_data(df, output_file)
        
        # Stats
        print("\n" + "="*60)
        print("Dataset Statistics")
        print("="*60)
        print(f"Total rows: {len(df)}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Symbols: {df['symbol'].nunique()}")
        print(f"\nCrossover stats:")
        cross_df = df[df['macd_cross_up'] == 1]
        print(f"  Bullish crossovers: {len(cross_df)}")
        if len(cross_df) > 0:
            if 'label' in df.columns:
                print(f"  Win rate: {cross_df['label'].mean():.2%}")
            elif 'mfe_pct' in df.columns and 'mae_pct' in df.columns:
                # For MFE/MAE regression
                valid_mfe = cross_df.dropna(subset=['mfe_pct', 'mae_pct'])
                print(f"  Valid MFE/MAE entries: {len(valid_mfe)}")
                if len(valid_mfe) > 0:
                    print(f"  Avg MFE: {valid_mfe['mfe_pct'].mean():.3f} ({valid_mfe['mfe_pct'].mean()*100:.1f}%)")
                    print(f"  Avg MAE: {valid_mfe['mae_pct'].mean():.3f} ({valid_mfe['mae_pct'].mean()*100:.1f}%)")
        
        cross_df = df[df['macd_cross_down'] == 1]
        print(f"  Bearish crossovers: {len(cross_df)}")
        if len(cross_df) > 0 and 'label' in df.columns:
            print(f"  Win rate: {cross_df['label'].mean():.2%}")


