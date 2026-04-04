import pandas as pd
import numpy as np
import lightgbm as lgb
import os

# 1. Feature Engineering: FVG + Volatility Squeeze
def engineer_features(df_ohlcv: pd.DataFrame, df_deriv: pd.DataFrame, df_fund: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    print("Engineering FVG & Squeeze features...")
    df = pd.merge(df_ohlcv, df_deriv, left_index=True, right_index=True, how='left')
    df = pd.merge(df, df_fund, left_index=True, right_index=True, how='left').sort_index()

    df = df.ffill().dropna()
    eps = 1e-8
    
    # 1. Fair Value Gap (FVG)
    # Bullish FVG: Low of candle t > High of candle t-2
    # Bearish FVG: High of candle t < Low of candle t-2
    df['bullish_fvg'] = (df['low'] > df['high'].shift(2)).astype(int)
    df['bearish_fvg'] = (df['high'] < df['low'].shift(2)).astype(int)
    
    # Size of the gap
    df['bullish_fvg_size'] = (df['low'] - df['high'].shift(2)).clip(lower=0) / df['close']
    df['bearish_fvg_size'] = (df['low'].shift(2) - df['high']).clip(lower=0) / df['close']
    
    # 2. Volatility Squeeze (Bollinger Bands inside Keltner Channels equivalent)
    # Using ATR vs Std Dev
    df['atr'] = (df['high'] - df['low']).rolling(window).mean()
    df['std_dev'] = df['close'].rolling(window).std()
    
    # Squeeze is on when std_dev is historically low compared to ATR
    df['vol_ratio'] = df['std_dev'] / (df['atr'] + eps)
    vol_ratio_median = df['vol_ratio'].rolling(window*4).median()
    df['is_squeeze'] = (df['vol_ratio'] < vol_ratio_median * 0.8).astype(int)
    
    # 3. Momentum
    df['roc_12'] = df['close'].pct_change(12)
    
    # Funding
    if 'fundingRate' in df.columns:
        fr = df['fundingRate']
    else:
        fr = pd.Series(0, index=df.index)

    fr_median = fr.rolling(window).median()
    fr_iqr = fr.rolling(window).quantile(0.75) - fr.rolling(window).quantile(0.25)
    df['funding_robust_z'] = (fr - fr_median) / (fr_iqr + eps)

    # Trigger: Bullish FVG forms OUT out of a volatility squeeze while momentum is positive
    df['long_fvg_trigger'] = (
        (df['bullish_fvg'] == 1) & 
        (df['is_squeeze'].shift(1) == 1) & 
        (df['roc_12'] > 0.02) & 
        (df['bullish_fvg_size'] > 0.002)
    ).astype(int)

    return df.dropna()

# 2. Targets
def add_targets(df: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    df[f'target_return_{horizon}h'] = np.log(df['close'].shift(-horizon) / df['close'])
    df['target_cls'] = (df[f'target_return_{horizon}h'] > 0).astype(int)
    return df.dropna()

# 3. Pruning
def prune_features(df: pd.DataFrame, features: list):
    split_idx = int(len(df) * 0.5)
    train_df = df.iloc[:split_idx]
    X_train = train_df[features]
    y_train = train_df['target_cls']
    
    model = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({'feature': features, 'importance': importances}).sort_values('importance', ascending=False)
    
    keep_n = max(1, len(features) // 2)
    top_features = feat_imp.head(keep_n)['feature'].tolist()
    print(f"Top features retained: {top_features}")
    return top_features

# 4. Backtest
def walk_forward_backtest(df: pd.DataFrame, trigger_col: str):
    cost = 0.0010  # 10 bps
    df['strategy_returns'] = 0.0
    
    # 12h hold
    pos = df[trigger_col].replace(0, np.nan).ffill(limit=12).fillna(0)
    strat_ret = pos.shift(1) * np.log(df['close'] / df['close'].shift(1))
    
    trades = pos.diff().abs()
    strat_ret -= trades * (cost / 2)
    
    df['strategy_returns'] = strat_ret
    
    annualized_vol = df['strategy_returns'].std() * np.sqrt(365 * 24)
    annualized_ret = df['strategy_returns'].mean() * 365 * 24
    sharpe = annualized_ret / annualized_vol if annualized_vol > 0 else 0
    
    cum_ret = np.exp(df['strategy_returns'].cumsum())
    drawdown = cum_ret / cum_ret.cummax() - 1
    max_dd = drawdown.min()
    
    return sharpe, max_dd

def log_results(metrics: dict, log_path="experiments_log.csv"):
    df_new = pd.DataFrame([metrics])
    if os.path.exists(log_path):
        df_old = pd.read_csv(log_path)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(log_path, index=False)
    print("Metrics logged to", log_path)

if __name__ == "__main__":
    print("=== ITERATION 2: FVG + VOLATILITY SQUEEZE ===")
    try:
        df_ohlcv = pd.read_parquet("data/ohlcv/BTCUSDT_USDT.parquet")
        if 'timestamp' in df_ohlcv.columns:
            df_ohlcv.set_index('timestamp', inplace=True)
            df_ohlcv = df_ohlcv[~df_ohlcv.index.duplicated(keep='first')]
        df_ohlcv = df_ohlcv.rename(columns=str.lower)
    except Exception as e:
        exit()
        
    try:
        df_deriv = pd.read_parquet("data/derivatives/BTCUSDT.parquet")
        df_deriv.set_index('timestamp', inplace=True)
        df_deriv = df_deriv[~df_deriv.index.duplicated(keep='first')]
    except Exception as e:
        df_deriv = pd.DataFrame(index=df_ohlcv.index)
        
    try:
        df_fund = pd.read_parquet("data/funding/BTCUSDT_USDT.parquet")
        df_fund.set_index('timestamp', inplace=True)
        df_fund = df_fund[~df_fund.index.duplicated(keep='first')]
    except Exception as e:
        df_fund = pd.DataFrame(index=df_ohlcv.index)
        
    df_ohlcv = df_ohlcv.iloc[-40000:]
    df_deriv = df_deriv.iloc[-40000:]
    df_fund = df_fund.iloc[-40000:]

    df = engineer_features(df_ohlcv, df_deriv, df_fund)
    df = add_targets(df, horizon=12)
    
    features = ['bullish_fvg_size', 'bearish_fvg_size', 'vol_ratio', 'roc_12', 'funding_robust_z', 'atr']
    top_features = prune_features(df, features)
    
    trigger_count = df['long_fvg_trigger'].sum()
    print(f"Trigger found count: {trigger_count}")
    
    if trigger_count > 0:
        sharpe, max_dd = walk_forward_backtest(df, 'long_fvg_trigger')
    else:
        sharpe, max_dd = 0, 0
        
    print(f"OOS Sharpe Ratio: {sharpe:.4f}")
    print(f"OOS Max Drawdown: {max_dd:.4%}")
    
    metrics = {
        "iteration": "2",
        "name": "FVG + Volatility Squeeze",
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "features": str(top_features)
    }
    log_results(metrics)
    print("=== PIPELINE FINISHED DYNAMICALLY ===")
