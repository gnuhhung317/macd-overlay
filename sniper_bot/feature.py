import pandas as pd
import numpy as np

def load_ohlcv_1h(symbol):
    for name in [f"{symbol}_USDT.parquet", f"{symbol}.parquet"]:
        fp = OHLCV_DIR / name
        if fp.exists():
            df = pd.read_parquet(fp)
            if 'timestamp' not in df.columns and 'open_time' in df.columns: df = df.rename(columns={'open_time':'timestamp'})
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') if df['timestamp'].dtype=='int64' else pd.to_datetime(df['timestamp'])
            return df.sort_values('timestamp').reset_index(drop=True)
    return pd.DataFrame()
def resample_1h(df_1h, tf):
    if df_1h.empty: return pd.DataFrame()
    return df_1h.set_index('timestamp').resample(TF_CONFIG[tf]['rule']).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna().reset_index()
def calculate_rsi(prices, period=14):
    d = prices.diff(); g = d.where(d>0,0).rolling(period).mean(); l = (-d.where(d<0,0)).rolling(period).mean()
    return 100-(100/(1+g/(l.replace(0,np.nan)+1e-9)))
def calculate_macd(df, fast=12, slow=26, signal=9):
    ef=df['close'].ewm(span=fast).mean(); es=df['close'].ewm(span=slow).mean()
    df['macd']=ef-es; df['macd_signal']=df['macd'].ewm(span=signal).mean(); df['macd_histogram']=df['macd']-df['macd_signal']
    df['macd_cross_up']=((df['macd']>df['macd_signal'])&(df['macd'].shift(1)<=df['macd_signal'].shift(1))).astype(int)
    df['macd_cross_down']=((df['macd']<df['macd_signal'])&(df['macd'].shift(1)>=df['macd_signal'].shift(1))).astype(int)
    df['macd_slope']=df['macd'].diff(); df['macd_acceleration']=df['macd_slope'].diff(); return df
def calculate_liquidity_sweep(df, lookback=20):
    df = df.copy()
    df['swing_low'] = df['low'].rolling(window=lookback).min().shift(1)
    df['swing_high'] = df['high'].rolling(window=lookback).max().shift(1)
    df['candle_range'] = df['high'] - df['low'] + 1e-9
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick_ratio'] = df['lower_wick'] / df['candle_range']
    df['upper_wick_ratio'] = df['upper_wick'] / df['candle_range']
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    cond_sweep_bottom = df['low'] < df['swing_low']
    cond_reject_bottom = df['close'] > df['swing_low']
    cond_pinbar_bottom = df['lower_wick_ratio'] > 0.3
    cond_vol_surge = df['volume'] > df['vol_sma_20']
    df['bullish_sweep'] = (cond_sweep_bottom & cond_reject_bottom & cond_pinbar_bottom & cond_vol_surge).astype(int)
    cond_sweep_top = df['high'] > df['swing_high']
    cond_reject_top = df['close'] < df['swing_high']
    cond_pinbar_top = df['upper_wick_ratio'] > 0.3
    df['bearish_sweep'] = (cond_sweep_top & cond_reject_top & cond_pinbar_top & cond_vol_surge).astype(int)
    df['macd_cross_up'] = df['bullish_sweep']
    df['macd_cross_down'] = df['bearish_sweep']
    return df
def calculate_features(df, df_1d=None, btc_df=None):
    df=df.copy(); df['log_returns']=np.log(df['close']/(df['close'].shift(1)+1e-9))
    df['high_low_range']=(df['high']-df['low'])/df['close']; df['body_size']=abs(df['close']-df['open'])/df['close']
    df['candle_range']=df['high']-df['low']+1e-9; df['lower_wick']=df[['open','close']].min(axis=1)-df['low']; df['upper_wick']=df['high']-df[['open','close']].max(axis=1)
    df['lower_wick_ratio_current']=df['lower_wick']/df['candle_range']; df['upper_wick_ratio_current']=df['upper_wick']/df['candle_range']
    for p in [7,14,21,50,100,200]: df[f'ema_{p}']=df['close'].ewm(span=p, adjust=True).mean()
    for p in [10,20,50,200]: df[f'sma_{p}']=df['close'].rolling(p).mean()
    tr=pd.concat([df['high']-df['low'],abs(df['high']-df['close'].shift(1)),abs(df['low']-df['close'].shift(1))],axis=1).max(axis=1)
    df['atr_14']=tr.rolling(14).mean(); df['volatility_14']=df['log_returns'].rolling(14).std()
    df['vol_sma_14']=df['volatility_14'].rolling(14).mean(); df['vol_compression']=df['volatility_14']/(df['vol_sma_14']+1e-9)
    df['volume_sma_20']=df['volume'].rolling(20).mean(); df['volume_std_20']=df['volume'].rolling(20).std()
    df['volume_ratio']=df['volume']/(df['volume_sma_20']+1e-9); df['volume_zscore']=(df['volume']-df['volume_sma_20'])/(df['volume_std_20']+1e-9)
    df['volume_trend']=df['volume'].rolling(7).mean()/(df['volume'].rolling(21).mean()+1e-9); df['volume_spike']=(df['volume_ratio']>2).astype(int)
    df['rsi_14']=calculate_rsi(df['close'], 14); df['rsi_slope']=df['rsi_14'].diff(3) # Match Ground Truth (No /3)
    l14=df['low'].rolling(14).min(); h14=df['high'].rolling(14).max()
    df['stoch_k']=100*(df['close']-l14)/(h14-l14).replace(0,np.nan); df['stoch_d']=df['stoch_k'].rolling(3).mean()
    df['roc_7']=df['close'].pct_change(7); df['roc_14']=df['close'].pct_change(14)
    # Phase 11 Features
    df['sma_30']=df['close'].rolling(30).mean(); df['price_vs_sma_30']=df['close']/(df['sma_30']+1e-9)
    df['momentum_30']=df['close'].pct_change(30)
    pdm=df['high'].diff(); mdm=-df['low'].diff()
    pdm=pdm.where((pdm>mdm)&(pdm>0),0); mdm=mdm.where((mdm>pdm)&(mdm>0),0); atr_s=tr.rolling(14).mean()
    pdi=100*(pdm.rolling(14).mean()/atr_s.replace(0,np.nan)); mdi=100*(mdm.rolling(14).mean()/atr_s.replace(0,np.nan))
    df['adx']=(100*abs(pdi-mdi)/(pdi+mdi).replace(0,np.nan)).rolling(14).mean()
    df['dist_to_high_30d']=(df['close']-df['high'].rolling(30).max())/df['close']
    df['dist_to_low_30d']=(df['close']-df['low'].rolling(30).min())/df['close']
    for e in [21,50,200]: df[f'dist_to_ema_{e}_pct']=(df['close']-df[f'ema_{e}'])/df['close']
    df['trend_state']=np.where(df['close']>df['sma_50'],1,np.where(df['close']<df['sma_50'],-1,0))
    df['is_trending']=(df['adx']>25).astype(int); df['is_volatile']=(df['vol_compression']>1.5).astype(int)
    df['hour_sin']=np.sin(2*np.pi*df['timestamp'].dt.hour/24); df['hour_cos']=np.cos(2*np.pi*df['timestamp'].dt.hour/24)
    df['day_sin']=np.sin(2*np.pi*df['timestamp'].dt.dayofweek/7); df['day_cos']=np.cos(2*np.pi*df['timestamp'].dt.dayofweek/7)
    df['vol_ratio_alpha']=df['volume_ratio']*df['volatility_14']
    df['market_structure_bull']=((df['close']>df['sma_200'])&(df['sma_50']>df['sma_200'])).astype(int)
    bb_mid=df['close'].rolling(20).mean(); bb_std=df['close'].rolling(20).std()
    bb_wd=(bb_mid+2*bb_std - (bb_mid-2*bb_std))/(bb_mid+1e-9)
    df['bb_squeeze']=(bb_wd<bb_wd.rolling(20).quantile(0.2)).astype(int)
    df['vwap_30d']=(df['close']*df['volume']).rolling(30).sum()/(df['volume'].rolling(30).sum()+1e-9)
    df['above_poc']=(df['close']>df['vwap_30d']).astype(int)
    
    # Hit & Run 10% Features
    df['micro_volume']=df['volume']/(df['volume'].rolling(5).mean()+1e-9)
    df['price_accel']=df['close'].pct_change(1)/(df['close'].pct_change(4).replace(0,np.nan)+1e-9)
    df['order_flow_proxy']=(df['close']-df['low'])/(df['high']-df['low']+1e-9)
    
    df=calculate_macd(df); df=df.drop(columns=['macd_cross_up','macd_cross_down'], errors='ignore')
    df=calculate_liquidity_sweep(df)
    df['usd_vol_24h'] = (df['volume'] * df['close']).rolling(24).sum()

    # --- PORTED FROM sync_worker.py (PHASE 11 & BTC CONTEXT) ---
    df['dist_to_ema50_atr'] = (df['close'] - df['ema_50']) / (df['atr_14'] + 1e-9)
    df['vol_acceleration'] = df['volume'] / (df['volume'].shift(1) + 1e-9)
    df['resistance_50'] = df['high'].rolling(50).max().shift(1) # Shift 1 to avoid lookahead
    df['dist_to_res'] = (df['resistance_50'] - df['close']) / (df['close'] + 1e-9)

    # Daily MTF (Anti-Lookahead)
    if df_1d is not None and not df_1d.empty:
        d1d_ema = df_1d['ema_200'].iloc[-1] if 'ema_200' in df_1d.columns else df_1d['close'].ewm(span=200).mean().iloc[-1]
        df['ema_200_1d_dist'] = (df['close'] - d1d_ema) / (d1d_ema + 1e-9)
        df['rsi_14_1d'] = calculate_rsi(df_1d['close'], 14).iloc[-1]
    else:
        df['ema_200_1d_dist'] = np.nan; df['rsi_14_1d'] = np.nan

    # BTC Context
    if btc_df is not None and not btc_df.empty:
        btc_map = btc_df.set_index('timestamp')
        df['btc_close'] = df['timestamp'].map(btc_map['close']).ffill()
        df['btc_ema_200'] = df['timestamp'].map(btc_map['ema_200']).ffill()
        df['btc_adx'] = df['timestamp'].map(btc_map['adx']).ffill()
        df['btc_returns_hist'] = df['timestamp'].map(btc_map['log_returns']).fillna(0)
        
        df['btc_is_bull_regime'] = (df['btc_close'] > df['btc_ema_200']).astype(int)
        df['btc_trend_strength'] = (df['btc_adx'] > 25).astype(int)
        df['rs_vs_btc'] = df['log_returns'] - df['btc_returns_hist']
        df['rs_vs_btc_sma7'] = df['rs_vs_btc'].rolling(7).mean()
        df['btc_corr'] = df['log_returns'].rolling(14).corr(df['btc_returns_hist']).fillna(0)
        
        # Cleanup temp btc columns
        df.drop(columns=['btc_close', 'btc_ema_200', 'btc_adx', 'btc_returns_hist'], inplace=True, errors='ignore')
    else:
        for c in ['btc_is_bull_regime', 'btc_trend_strength', 'btc_corr', 'rs_vs_btc', 'rs_vs_btc_sma7']:
            df[c] = np.nan

    return df.dropna(subset=['macd','swing_low','vol_sma_20','vwap_30d','usd_vol_24h'])
def generate_momentum_labels(df, horizon=12, min_pump=0.10):
    df = df.copy()
    
    # 1. Đảo ngược Data để nhìn về tương lai
    df_rev = df.iloc[::-1].copy()
    
    # 2. Tìm Đỉnh cao nhất (Max High) trong 'horizon' nến tiếp theo
    if 'symbol' in df_rev.columns:
        future_max_high = df_rev.groupby('symbol', group_keys=False)['high'].apply(lambda x: x.rolling(horizon, min_periods=1).max())
    else:
        future_max_high = df_rev['high'].rolling(horizon, min_periods=1).max()
        
    df['future_max_high'] = future_max_high.sort_index()
    df['max_pump_pct'] = (df['future_max_high'] - df['close']) / df['close']
    df['label'] = (df['max_pump_pct'] >= min_pump).astype(int)
    
    if 'usd_vol_24h' in df.columns:
        df.loc[df['usd_vol_24h'] < 1000000, 'label'] = np.nan
        
    df['ignition'] = df['label']
    if 'symbol' in df.columns:
        df['future_return'] = df.groupby('symbol')['close'].shift(-horizon) / df['close'] - 1
    else:
        df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1
    df['trade_result'] = np.where(df['label'] == 1, 'WIN', 'LOSS')
    return df.drop(columns=['future_max_high'])
def apply_winsorization(df,fc,lo=0.01,hi=0.99):
    df=df.copy()
    for c in fc:
        if c in df.columns and df[c].dtype in ['float64','float32','int64']: l,h=df[c].quantile(lo),df[c].quantile(hi); df[c]=df[c].clip(l,h)
    return df
def apply_feature_shift(df):
    ex={'timestamp','symbol','open','high','low','close','volume','label','ignition','trade_result','macd_cross_up','macd_cross_down'}
    sc=[c for c in df.columns if c not in ex]
    
    if 'symbol' in df.columns: df[sc]=df.groupby('symbol')[sc].shift(1)
    else: df[sc]=df[sc].shift(1)
    return df.dropna(subset=sc[:3])
    
def reduce_mem_usage(df):
    """Giảm dung lượng RAM bằng cách ép kiểu dữ liệu."""
    for col in df.columns:
        col_type = df[col].dtype
        
        if not pd.api.types.is_numeric_dtype(col_type):
            continue
            
        try:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.api.types.is_integer_dtype(col_type):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max: df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max: df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max: df[col] = df[col].astype(np.int32)
                else: df[col] = df[col].astype(np.int64)
            else:
                df[col] = df[col].astype(np.float32) # Luôn cast float về float32 để tránh lỗi LGBM
        except Exception as e:
            pass
    return df

print("✓ Pipeline loaded.")

def scan_historical_confluence(df, horizon=12):
    """
    Quét toàn bộ lịch sử để tìm các điểm Hợp Lưu Breakout kinh điển.
    """
    df = df.copy()
    if 'rsi_14' not in df.columns:
        from ml.data_pipeline import calculate_rsi
        df['rsi_14'] = calculate_rsi(df['close'], 14)
        
    df['resistance_50'] = df.groupby('symbol')['high'].transform(lambda x: x.rolling(50).max().shift(1))
    df['ema_20'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=20).mean())
    df['ema_50'] = df.groupby('symbol')['close'].transform(lambda x: x.ewm(span=50).mean())
    
    cond_uptrend = (df['ema_20'] > df['ema_50']) & (df['low'] > df['ema_50'])
    cond_rsi = df['rsi_14'] > 60
    vol_sma_20 = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).mean().shift(1))
    cond_volume = df['volume'] > (vol_sma_20 * 2.5)
    cond_breakout = df['close'] > df['resistance_50']
    
    df['is_golden_setup'] = cond_uptrend & cond_rsi & cond_volume & cond_breakout
    
    # Tính thực tế bay bao nhiêu %
    df['actual_pump_pct'] = df.groupby('symbol')['high'].transform(lambda x: (x.shift(-horizon).rolling(horizon).max() - df['close']) / df['close'])
    
    signals = df[df['is_golden_setup'] == True].copy()
    cols = ['timestamp', 'symbol', 'close', 'resistance_50', 'volume', 'actual_pump_pct']
    return signals[cols]