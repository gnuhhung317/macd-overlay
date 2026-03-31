import os
import sys
import gc
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path

# ============================================================
# CONFIG & FEATURES
# ============================================================
MODEL_FEATURES = [
    'rsi_14','rsi_slope','stoch_k','stoch_d','roc_7','roc_14',
    'volume_ratio','volume_zscore','volume_trend','rs_vs_btc','rs_vs_btc_sma7','vol_compression',
    'dist_to_high_30d','dist_to_low_30d','dist_to_ema_21_pct','dist_to_ema_50_pct','dist_to_ema_200_pct',
    'price_vs_sma_30','momentum_30','macd_slope','macd_acceleration',
    'upper_wick_ratio','dist_to_ema50_atr','vol_acceleration',
    'bb_squeeze','above_poc',
    'micro_volume','price_accel','order_flow_proxy',
    'btc_is_bull_regime','btc_trend_strength','adx','hour_sin','hour_cos','day_sin','day_cos',
    'btc_corr','trend_state','is_trending','is_volatile','ema_200_1d_dist','rsi_14_1d'
]

# Paths - REDIRECTED TO HONEST
BASE_DIR = Path(r"d:\Code\Projects\self-projects\macd-overlay - Copy")
INPUT_FILE = BASE_DIR / "data" / "processed" / "features_1h_honest_dataset.parquet"
OUTPUT_MODEL_DIR = BASE_DIR / "ml" / "training" / "models" / "honest"
OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_MODEL_PATH = OUTPUT_MODEL_DIR / "ensemble_lgbm_tabular.joblib"
OUTPUT_META_PATH = OUTPUT_MODEL_DIR / "ensemble_meta.joblib"

def prepare_cascade_data_optimized(parquet_path):
    print(f"🧹 Loading Honest Dataset: {parquet_path.name}")
    
    # Load data
    cols_to_load = list(set(MODEL_FEATURES + [
        'symbol', 'timestamp', 'open', 'close', 'high', 'low', 'volume', 'usd_vol_24h'
    ]))
    
    df = pd.read_parquet(
        parquet_path, 
        columns=cols_to_load,
        filters=[('timestamp', '<', pd.Timestamp('2025-01-01')), ('usd_vol_24h', '>=', 1000000)]
    )

    if df.empty:
        print("❌ No data satisfied filters!")
        return pd.DataFrame()
        
    df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

    # 3. TÌNH ATR (14) - Standard for Labels
    def calc_atr(df_group):
        high_low = df_group['high'] - df_group['low']
        high_close = np.abs(df_group['high'] - df_group['close'].shift())
        low_close = np.abs(df_group['low'] - df_group['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(14).mean()

    df['atr_14'] = df.groupby('symbol', group_keys=False).apply(calc_atr)
    
    # 4. TRIPLE BARRIER LABELING (Stage 2)
    print("🎯 Labeling trades (Triple Barrier)...")
    def generate_labels(df_group):
        df_group = df_group.copy()
        # Ignition conditions (Stage 1 logic)
        vol_sma = df_group['volume'].rolling(20).mean().shift(1)
        c1 = (df_group['close'] > df_group['open']) & (df_group['close'] > df_group['close'].ewm(span=20).mean())
        c2 = ((df_group['close'] - df_group['open']) / df_group['open']) > 0.015
        c3 = (df_group['volume'] > vol_sma * 1.5) & (df_group['volume'] < vol_sma * 4.0)
        c4 = (df_group['rsi_14'] >= 55) & (df_group['rsi_14'] <= 72)
        
        ignition = c1 & c2 & c3 & c4
        df_group['label'] = 0 # Neutral
        
        # Performance window
        for idx in df_group.index[ignition]:
            if idx + 24 >= len(df_group): continue
            
            entry_price = df_group.loc[idx, 'close']
            atr = df_group.loc[idx, 'atr_14']
            
            # SL/TP Targets
            tp_long = entry_price + (atr * 3.13)
            sl_long = entry_price - (atr * 2.51)
            tp_short = entry_price - (atr * 4.0)
            sl_short = entry_price + (atr * 2.2)
            
            # Check next 24 bars
            window = df_group.loc[idx+1 : idx+24]
            
            # Label Long
            if window['high'].max() >= tp_long:
                df_group.loc[idx, 'label'] = 1
            elif window['low'].min() <= sl_long:
                df_group.loc[idx, 'label'] = 0 # Loss
                
            # Label Short (Override if better)
            if window['low'].min() <= tp_short:
                df_group.loc[idx, 'label'] = 2
            elif window['high'].max() >= sl_short:
                if df_group.loc[idx, 'label'] != 1:
                    df_group.loc[idx, 'label'] = 0
                    
        return df_group

    df = df.groupby('symbol', group_keys=False).apply(generate_labels)
    return df

def train_model():
    df = prepare_cascade_data_optimized(INPUT_FILE)
    if df.empty: return
    
    # Filter for Ignition only
    train_df = df[df['label'] != 0].copy()
    print(f"🚀 Training on {len(train_df)} ignited cases.")
    
    X = train_df[MODEL_FEATURES]
    y = train_df['label']
    
    # Train LGBM
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        objective='multiclass',
        num_class=3,
        random_state=42,
        importance_type='gain'
    )
    
    model.fit(X, y)
    
    # Save
    joblib.dump(model, OUTPUT_MODEL_PATH)
    joblib.dump({'features': MODEL_FEATURES, 'threshold': 0.6}, OUTPUT_META_PATH)
    print(f"✅ HONEST MODEL SAVED TO: {OUTPUT_MODEL_DIR}")

if __name__ == '__main__':
    train_model()
