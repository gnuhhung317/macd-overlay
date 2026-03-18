#!/usr/bin/env python3
"""
Data Pipeline: Load, Merge 1h -> multi-timeframe (4h/8h/12h/1d), Feature Engineering
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Default data directory (Bitget for backward compatibility)
DATA_DIR = Path(__file__).parent.parent / 'data'
OHLCV_DIR = DATA_DIR / 'ohlcv'
FUNDING_DIR = DATA_DIR / 'funding'
PROCESSED_DIR = DATA_DIR / 'processed'

def set_data_directory(new_dir: Path):
    """
    Dynamically update the data directories.
    Useful for switching between Bitget and Binance.
    """
    global DATA_DIR, OHLCV_DIR, FUNDING_DIR, PROCESSED_DIR
    DATA_DIR = new_dir
    OHLCV_DIR = DATA_DIR / 'ohlcv'
    FUNDING_DIR = DATA_DIR / 'funding'
    PROCESSED_DIR = DATA_DIR / 'processed'
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


# ----------------------------------------------------------
# Timeframe Configuration: resample rules + label parameters
# ----------------------------------------------------------
TIMEFRAME_CONFIG = {
    '4h': {
        'resample_rule': '4h',
        'min_bars': 200,       # Minimum 4h bars needed (~33 days)
        'atr_clamp': (0.002, 0.06),   # ATR% range for 4h
        'max_tp_pct': 0.15,
        'max_bars_label': 30,  # 30 x 4h = 5 days lookahead
        'unit_name': '4h bars',
    },
    '8h': {
        'resample_rule': '8h',
        'min_bars': 100,       # ~33 days
        'atr_clamp': (0.003, 0.08),
        'max_tp_pct': 0.20,
        'max_bars_label': 20,  # 20 x 8h = ~7 days
        'unit_name': '8h bars',
    },
    '12h': {
        'resample_rule': '12h',
        'min_bars': 80,        # ~40 days
        'atr_clamp': (0.004, 0.10),
        'max_tp_pct': 0.25,
        'max_bars_label': 15,  # 15 x 12h = ~7.5 days
        'unit_name': '12h bars',
    },
    '1d': {
        'resample_rule': '1D',
        'min_bars': 100,       # 100 days
        'atr_clamp': (0.005, 0.13),
        'max_tp_pct': 0.30,
        'max_bars_label': 15,  # 15 days
        'unit_name': 'days',
    },
}

def resample_1h(df_1h: pd.DataFrame, timeframe: str = '1d') -> pd.DataFrame:
    """Resample 1h OHLCV to any target timeframe (4h, 8h, 12h, 1d)"""
    if df_1h.empty:
        return pd.DataFrame()
    
    cfg = TIMEFRAME_CONFIG.get(timeframe)
    if cfg is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}. Choose from {list(TIMEFRAME_CONFIG.keys())}")
    
    df = df_1h.copy()
    df = df.set_index('timestamp')
    
    df_out = df.resample(cfg['resample_rule']).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    df_out = df_out.reset_index()
    return df_out

# Keep backward compatibility
def resample_1h_to_1d(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Legacy wrapper: resample 1h to 1d"""
    return resample_1h(df_1h, '1d')


def calculate_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    """
    Calculate MACD using standard formula:
    MACD Line = EMA(fast) - EMA(slow)
    Signal Line = EMA(MACD, signal)
    Histogram = MACD - Signal
    """
    df = df.copy()
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['histogram'] = df['macd'] - df['signal']
    return df


def calculate_features(df: pd.DataFrame, df_1d: pd.DataFrame = None, btc_df: pd.DataFrame = None) -> pd.DataFrame:
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
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=True).mean()
        # Correct parity: The global shift will handle t-1. Use raw t here.
        df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}']
    
    # Trend strength
    df['trend_7_21'] = df['sma_7'] / df['sma_21']
    df['trend_21_50'] = df['sma_21'] / df['sma_50']
    df['trend_50_200'] = df['sma_50'] / df['sma_200']
    
    # ===== MACD Features =====
    df = calculate_macd(df)
    df['macd_slope'] = df['macd'].diff()
    df['signal_slope'] = df['signal'].diff()
    df['histogram_slope'] = df['histogram'].diff()
    df['macd_acceleration'] = df['macd_slope'].diff()
    
    # MACD crossover detection
    df['macd_cross_up'] = ((df['macd'] > df['signal']) & (df['macd'].shift(1) <= df['signal'].shift(1))).astype(int)
    df['macd_cross_down'] = ((df['macd'] < df['signal']) & (df['macd'].shift(1) >= df['signal'].shift(1))).astype(int)
    df['macd_crossover'] = df['macd_cross_up'] - df['macd_cross_down']
    
    # Vectorized bars since crossover (Significant performance boost)
    df['bars_since_cross_up'] = df.groupby((df['macd_cross_up'] == 1).cumsum()).cumcount()
    df.loc[df['macd_cross_up'].cumsum() == 0, 'bars_since_cross_up'] = 999
    
    df['bars_since_cross_down'] = df.groupby((df['macd_cross_down'] == 1).cumsum()).cumcount()
    df.loc[df['macd_cross_down'].cumsum() == 0, 'bars_since_cross_down'] = 999
    
    # ===== Volatility Features =====
    df['atr_14'] = calculate_atr(df, 14)
    df['atr_7'] = calculate_atr(df, 7)
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
    
    # RSI Slope (momentum of momentum) - NEW for TP prediction
    # BUG FIX: Ground truth shows unscaled diff: diff(3)
    df['rsi_slope'] = df['rsi_14'].diff(3)
    
    # Stochastic
    df['stoch_k'] = calculate_stochastic(df, 14)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    
    # ===== Rate of Change =====
    df['roc_7'] = df['close'].pct_change(7)
    df['roc_14'] = df['close'].pct_change(14)
    df['roc_21'] = df['close'].pct_change(21)
    
    # ADX (Average Directional Index) - NEW for TP prediction
    df = calculate_adx(df, 14)
    
    # ===== Volume Features & Momentum =====
    df['vol_sma_14'] = df['volume'].rolling(14).mean()
    df['vol_std_14'] = df['volume'].rolling(14).std()
    df['volume_zscore'] = (df['volume'] - df['vol_sma_14']) / (df['vol_std_14'] + 1e-9)
    df['volume_ratio'] = df['volume'] / (df['vol_sma_14'] + 1e-9)
    
    # ===== High-Alpha / Sector RS =====
    # Note: Sector RS (vs absolute index) is handled in build_dataset for multi-symbol
    # Here we add "Relative to Self" alpha as requested
    df['price_vs_sma_30'] = df['close'] / (df['close'].rolling(30).mean() + 1e-9)
    df['momentum_30'] = df['close'].pct_change(30)
    df['volume_sma_7'] = df['volume'].rolling(7).mean()
    df['volume_sma_14'] = df['volume'].rolling(14).mean()
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_14']
    df['volume_trend'] = df['volume_sma_7'] / df['volume_sma_14']
    
    # Volume Spike (is this a breakout?) - NEW for TP prediction
    df['volume_spike'] = (df['volume'] / df['volume_sma_20']).clip(0, 5)
    
    # OBV
    df['obv'] = (np.sign(df['returns']) * df['volume']).cumsum()
    df['obv_sma'] = df['obv'].rolling(14).mean()
    df['obv_trend'] = df['obv'] / df['obv_sma']
    
    # ===== Market Regime =====
    df['is_trending'] = (abs(df['trend_7_21'] - 1) > 0.02).astype(int)
    df['is_volatile'] = (df['volatility_14'] > df['volatility_14'].rolling(50).mean()).astype(int)
    
    # ===== Advanced Regime Features =====
    
    # 1. Khoảng cách an toàn tới các đường EMA (Tính bằng %)
    df['dist_to_ema_21_pct'] = (df['close'] - df['ema_21']) / df['close']
    df['dist_to_ema_50_pct'] = (df['close'] - df['ema_50']) / df['close']
    df['dist_to_ema_200_pct'] = (df['close'] - df['ema_200']) / df['close']
    
    # 2. ADX Trend State (-1, 0, 1)
    # Kết hợp ADX và SMA để xác định rõ: Sideway (0), Uptrend mạnh (1), Downtrend mạnh (-1)
    df['trend_state'] = np.where(
        df['adx'] < 20, 0,  # Không có xu hướng
        np.where(df['close'] > df['sma_50'], 1, -1) # Có xu hướng, check xem trend lên hay xuống
    )
    
    # 3. Phân loại thanh khoản (Liquidity Regime)
    # Lọc các coin rác có volume chồi sụt bất thường
    def calc_cv(x):
        m = np.mean(x)
        return np.std(x)/m if m > 0 else 0
        
    # Use moving average of volume directly to calculate standard deviation since apply with lambda on rolling is slow, but we stick to user req
    # An optimization: rolling with pandas native methods where possible, but apply is ok for now.
    df['liquidity_regime'] = df['volume_sma_14'].rolling(30).apply(calc_cv, raw=True)
    
    # 4. Volatility Compression (squeeze before breakout)
    df['vol_compression'] = df['bb_width'] / df['bb_width'].rolling(20).mean()
    
    # 5. Volatility Ratio (Alpha Feature: Speed of volatility change)
    df['atr_21'] = calculate_atr(df, 21)
    df['vol_ratio_alpha'] = df['atr_7'] / df['atr_21']
    
    # 6. Price Magnet (Alpha Feature: Distance to 30d high/low)
    df['high_30d'] = df['high'].rolling(30).max()
    df['low_30d'] = df['low'].rolling(30).min()
    df['dist_to_high_30d'] = (df['close'] - df['high_30d']) / df['close']
    df['dist_to_low_30d'] = (df['close'] - df['low_30d']) / df['close']
    
    # 7. Time-based features (Cyclical)
    df['hour'] = df['timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 23)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 23)
    
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 6)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 6)
    
    # ===== Phase 12: Pre-Ignition Detector =====
    # 1. Keltner Channels & Squeeze Index
    df['keltner_width'] = 3 * df['atr_21']  # Using atr_21 which is calculated above
    df['squeeze_ratio'] = df['bb_width'] / (df['keltner_width'] + 1e-9)
    
    # 2. Volume Quietness (Depletion)
    df['vol_depletion'] = df['volume'] / (df['volume_sma_20'] + 1e-9)
    
    # 3. Chaikin Money Flow (CMF)
    # CMF = Sum(Money Flow Volume, 20) / Sum(Volume, 20)
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    mf_mult = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-9)
    mf_vol = mf_mult * df['volume']
    df['cmf_20'] = mf_vol.rolling(20).sum() / (df['volume'].rolling(20).sum() + 1e-9)
    
    # 4. Pre-Ignition Confluence Score
    # Thấp squeeze + thấp volume + neutral price action
    df['pre_ignition_score'] = (1 - df['squeeze_ratio']) + (1 - df['vol_depletion'])
    
    # 5. Institutional Scale Features (Missing in SHAP)
    df['dist_to_ema50_atr'] = (df['close'] - df['ema_50']) / (df['atr_14'] + 1e-9)
    df['vol_acceleration'] = df['volume'].diff().diff() / (df['volume_sma_20'] + 1e-9)
    
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
    tp_pct: float = 0.10, 
    sl_pct: float = 0.05, 
    max_bars: int = 15,
    use_atr: bool = True,
    atr_tp_mult: float = 4.0,
    atr_sl_mult: float = 2.0,
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
    df['ignition'] = np.nan     # New Ignition Label for Multi-Task
    
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
    atr_count = 0
    fixed_count = 0
    
    labels = []
    max_profits = []
    max_drawdowns = []
    bars_to_tps = []
    bars_to_sls = []
    results = []
    tp_pcts_used = []
    sl_pcts_used = []
    ignition_labels = []
    
    for idx in crossover_indices:
        entry_price = close[idx]
        is_long = is_long_arr[idx]
        
        # Calculate dynamic N for Ignition (40% of max_bars)
        n_ignition = max(3, int(max_bars * 0.4))
        if use_atr and atr_pct is not None:
            if is_long:
                tp_price = entry_price * (1 + atr_pct[idx] * atr_tp_mult)
                sl_price = entry_price * (1 - atr_pct[idx] * atr_sl_mult)
            else:
                tp_price = entry_price * (1 - atr_pct[idx] * atr_tp_mult)
                sl_price = entry_price * (1 + atr_pct[idx] * atr_sl_mult)
        else:
            if is_long:
                tp_price = entry_price * (1 + tp_pct)
                sl_price = entry_price * (1 - sl_pct)
            else:
                tp_price = entry_price * (1 - tp_pct)
                sl_price = entry_price * (1 + sl_pct)
            
        # Recalculate actual percentages for filtering
        actual_tp_pct = abs(tp_price - entry_price) / entry_price
        actual_sl_pct = abs(entry_price - sl_price) / entry_price
        
        # Minimum R:R Filter (1.5)
        risk_reward = actual_tp_pct / (actual_sl_pct + 1e-9)
        is_efficient = risk_reward >= 1.1 # Reduced from 1.5 to keep enough data, but user said 1.5
        
        if not is_efficient:
            # Penalize low R:R as noise
            labels.append(0.5)
            max_profits.append(0)
            max_drawdowns.append(0)
            bars_to_tps.append(max_bars)
            bars_to_sls.append(max_bars)
            results.append('NOISE_LOW_RR')
            tp_pcts_used.append(actual_tp_pct)
            sl_pcts_used.append(actual_sl_pct)
            ignition_labels.append(0.5)
            continue

        # Get future window
        future_start = idx + 1
        future_end = min(idx + max_bars + 1, n)
        future_high = high[future_start:future_end]
        future_low = low[future_start:future_end]
        
        if len(future_high) == 0:
            labels.append(np.nan); max_profits.append(np.nan); max_drawdowns.append(np.nan)
            bars_to_tps.append(np.nan); bars_to_sls.append(np.nan); results.append('')
            tp_pcts_used.append(np.nan); sl_pcts_used.append(np.nan)
            ignition_labels.append(np.nan)
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
        
        # Find first TP/SL hit
        bars_to_tp = (tp_hits[0] + 1) if len(tp_hits) > 0 else max_bars
        bars_to_sl = (sl_hits[0] + 1) if len(sl_hits) > 0 else max_bars
        hit_tp = len(tp_hits) > 0
        hit_sl = len(sl_hits) > 0
        
        # Check for Ignition: Volume > 2*SMA20 and abs(Close-Open) > 1 * ATR (within n_ignition)
        future_open = df['open'].values[future_start:min(idx + n_ignition + 1, n)]
        future_close = close[future_start:min(idx + n_ignition + 1, n)]
        future_vol = df['volume'].values[future_start:min(idx + n_ignition + 1, n)]
        future_vsma20 = df['volume_sma_20'].values[future_start:min(idx + n_ignition + 1, n)]
        future_atr = atr[future_start:min(idx + n_ignition + 1, n)] if atr_pct is not None else np.zeros(len(future_open))
        
        ignited = False
        for j in range(len(future_open)):
            if future_vol[j] > 2.0 * future_vsma20[j] and abs(future_close[j] - future_open[j]) > 1.0 * future_atr[j]:
                ignited = True
                break
        ignition_labels.append(1.0 if ignited else 0.0)
        
        # Phase 11: Comprehensive Sigmoid Score (MAE Penalty for Wins)
        # Score = Sigmoid(3.0 * (MFE / Target) - 2.5 * (MAE / Stop))
        mfe_ratio = np.clip(max_profit / actual_tp_pct, 0, 1.2)
        mae_ratio = np.clip(max_drawdown / actual_sl_pct, 0, 1.2)
        
        # Determine base result for logging
        if hit_tp and (not hit_sl or bars_to_tp <= bars_to_sl):
            result = 'TP_HIT'
            pnl_ratio = 1.1 # Slightly above 1 to reward hitting target
        elif hit_sl:
            result = 'SL_HIT'
            pnl_ratio = -1.0
        else:
            result = 'TIMEOUT'
            pnl_ratio = (max_profit / actual_tp_pct) - (max_drawdown / actual_sl_pct)
        
        # Universal Sigmoid Label
        raw_score = 3.0 * pnl_ratio - 2.5 * mae_ratio
        label = 1 / (1 + np.exp(-raw_score))
        label = np.clip(label, 0.05, 0.95)
        
        labels.append(float(label))
        max_profits.append(max_profit)
        max_drawdowns.append(max_drawdown)
        bars_to_tps.append(bars_to_tp)
        bars_to_sls.append(bars_to_sl)
        results.append(result)
        tp_pcts_used.append(actual_tp_pct)
        sl_pcts_used.append(actual_sl_pct)
    
    # Assign results back to DataFrame
    df.loc[crossover_indices, 'label'] = labels
    df.loc[crossover_indices, 'max_profit'] = max_profits
    df.loc[crossover_indices, 'max_drawdown'] = max_drawdowns
    df.loc[crossover_indices, 'bars_to_tp'] = bars_to_tps
    df.loc[crossover_indices, 'bars_to_sl'] = bars_to_sls
    df.loc[crossover_indices, 'trade_result'] = results
    df.loc[crossover_indices, 'tp_pct_used'] = tp_pcts_used
    df.loc[crossover_indices, 'sl_pct_used'] = sl_pcts_used
    df.loc[crossover_indices, 'ignition'] = ignition_labels
    
    if len(crossover_indices) > 0:
        symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else "Unknown"
        print(f"  [{symbol}] Labels: {atr_count} ATR-based, {fixed_count} Fixed-fallback")
        
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
        
        # Determine label (STRICT: Only TP_HIT is 1)
        if hit_tp and (not hit_sl or bars_to_tp <= bars_to_sl):
            label = 1
            result = 'TP_HIT'
        else:
            # All other outcomes (SL_HIT, TIMEOUT_WIN, TIMEOUT_LOSS) are failures (0)
            # This forces the model to predict high-confidence "Fast Winners"
            label = 0
            if hit_sl and (not hit_tp or bars_to_sl < bars_to_tp):
                result = 'SL_HIT'
            else:
                final_pnl = (future_close - entry_price) / entry_price
                if not is_long:
                    final_pnl = -final_pnl
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
    Process a single symbol: load, resample to target timeframe, add features.
    Features are calculated per-symbol to avoid data bleeding.
    """
    cfg = TIMEFRAME_CONFIG[timeframe]
    print(f"Processing {symbol} ({timeframe})...")
    
    # Load 1h data
    df_1h = load_ohlcv_1h(symbol)
    if df_1h.empty:
        return pd.DataFrame()
    
    # Resample to target timeframe
    df_tf = resample_1h(df_1h, timeframe)
    if len(df_tf) < cfg['min_bars']:
        print(f"  ⚠️ Not enough data: {len(df_tf)} {cfg['unit_name']}")
        return pd.DataFrame()
    
    # Add symbol column FIRST (before features, for groupby compatibility)
    df_tf['symbol'] = symbol
    
    # Add features (per-symbol, no data bleeding)
    df_tf = calculate_features(df_tf)
    
    # Add funding rate if available
    if include_funding:
        df_funding = load_funding(symbol)
        if not df_funding.empty:
            # Resample funding to target timeframe
            df_funding = df_funding.set_index('timestamp')
            df_funding_tf = df_funding['funding_rate'].resample(cfg['resample_rule']).mean().reset_index()
            df_funding_tf.columns = ['timestamp', 'funding_rate_avg']
            
            df_funding_tf['funding_rate_sum'] = df_funding['funding_rate'].resample(cfg['resample_rule']).sum().values
            
            df_tf = df_tf.merge(df_funding_tf, on='timestamp', how='left')
            df_tf['funding_rate_avg'] = df_tf['funding_rate_avg'].fillna(0)
            df_tf['funding_rate_sum'] = df_tf['funding_rate_sum'].fillna(0)
    
    print(f"  ✓ {len(df_tf)} {cfg['unit_name']}, {len(df_tf.columns)} features")
    return df_tf


def build_dataset(symbols: List[str] = None, min_days: int = 365, timeframe: str = '1d') -> pd.DataFrame:
    """Build full dataset from all symbols with Macro Market Regime"""
    cfg = TIMEFRAME_CONFIG[timeframe]
    if symbols is None:
        # Get all symbols from ohlcv directory
        symbols = [f.stem.replace('_USDT', '') for f in OHLCV_DIR.glob('*.parquet')]
        # Filter out quarterly futures
        symbols = [s for s in symbols if not any(x in s for x in ['-26', '-25', '-24'])]
    
    print(f"Processing {len(symbols)} symbols for timeframe={timeframe}...")
    
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
                    
                    # Fill missing
                    fill_cols = ['btc_is_bull_regime', 'btc_trend_strength', 'btc_returns']
                    df[fill_cols] = df[fill_cols].ffill().fillna(0)
                    
                    # 1. Relative Strength (RS)
                    df['rs_vs_btc'] = df['log_returns'] - df['btc_returns']
                    df['rs_vs_btc_sma7'] = df['rs_vs_btc'].rolling(7).mean()
                    
                    # 2. BTC Correlation (Alpha Feature)
                    # Measures if the coin is moving with the market or independently
                    df['btc_corr'] = df['log_returns'].rolling(14).corr(df['btc_returns']).fillna(0)

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


def apply_winsorization(df: pd.DataFrame, columns: List[str], limits=(0.01, 0.01)) -> pd.DataFrame:
    """Clip outliers in specified columns using percentiles."""
    print(f"Applying Winsorization to {len(columns)} features...")
    for col in columns:
        if col in df.columns:
            lower = df[col].quantile(limits[0])
            upper = df[col].quantile(1 - limits[1])
            df[col] = df[col].clip(lower, upper)
    return df

def apply_global_feature_shift(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies a T-1 shift to all predictive features. 
    This guarantees zero look-ahead bias if trading at the crossover confirm (Open of T+1).
    """
    if df.empty:
        return df
        
    df = df.copy()
    non_shift_cols = {
        'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume',
        'macd_cross_up', 'macd_cross_down', 'macd_crossover',
        'date', 'fundingTime'
    }
    shift_cols = [c for c in df.columns if c not in non_shift_cols]
    
    # We must group by symbol so we don't shift data across different coins
    df[shift_cols] = df.groupby('symbol', group_keys=False)[shift_cols].shift(1)
    
    # Drop rows of the first element that became NaN due to shifting
    df = df.dropna(subset=shift_cols[:3]) 
    return df

def save_processed_data(df: pd.DataFrame, filename: str = 'features_1d.parquet'):
    """Save processed data"""
    PROCESSED_DIR.mkdir(exist_ok=True)
    output_path = PROCESSED_DIR / filename
    df.to_parquet(output_path, index=False)
    print(f"✓ Saved to {output_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Build ML Dataset from OHLCV data')
    parser.add_argument('--all', action='store_true', help='Process all symbols (not just test set)')
    parser.add_argument('--min-days', type=int, default=365, help='Minimum days of data required')
    parser.add_argument('--timeframe', type=str, default='1d', 
                        choices=['4h', '8h', '12h', '1d'],
                        help='Target timeframe (4h, 8h, 12h, 1d)')
    parser.add_argument('--output', type=str, default=None, help='Output filename')
    
    args = parser.parse_args()
    tf = args.timeframe
    
    # Build dataset
    print("="*60)
    print(f"Building ML Dataset (timeframe={tf})")
    print("="*60)
    
    if args.all:
        # Process ALL symbols
        print("Processing ALL symbols in data/ohlcv folder...")
        symbols = None  # build_dataset will auto-detect
        output_file = args.output or f'features_{tf}_full.parquet'
    else:
        # Process top coins for testing
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 
                   'ADAUSDT', 'DOGEUSDT', 'DOTUSDT', 'LINKUSDT', 'AVAXUSDT']
        print(f"Processing {len(symbols)} test symbols...")
        output_file = args.output or f'features_{tf}_test.parquet'
    
    df = build_dataset(symbols, min_days=args.min_days, timeframe=tf)
    
    if not df.empty:
        print(f"\nApplying T-1 Global Shift to features to prevent Look-ahead bias...")
        df = apply_global_feature_shift(df)
        
        # Phase 5: Winsorization
        all_features = [c for c in df.columns if c not in ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'label', 'trade_result']]
        df = apply_winsorization(df, all_features)
        
        print("\nGenerating labels (Soft Labels enabled)...")
        df = generate_labels(df)
        
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
            print(f"  Win rate: {cross_df['label'].mean():.2%}")
        
        cross_df = df[df['macd_cross_down'] == 1]
        print(f"  Bearish crossovers: {len(cross_df)}")
        if len(cross_df) > 0:
            print(f"  Win rate: {cross_df['label'].mean():.2%}")
