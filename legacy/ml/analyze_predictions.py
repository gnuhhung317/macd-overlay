import os
import sys
import pandas as pd
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Add current and root directory to sys.path
sys.path.append(os.getcwd())
sys.path.append(str(Path(os.getcwd()).parent))

from ml.training.transformer_model import HybridScorer

# --- CONFIGURATION ---
MODEL_PATH = Path('ml/models/1d/hybrid')
DATA_PATH = Path('bitget-data/ohlcv')
TIMEFRAME = '1d'
TEST_START_DATE = '2024-01-01' 
SEQ_FEATURES = ['log_returns', 'high_low_range', 'body_size', 'volatility_14', 'macd', 'macd_slope', 'volume_ratio', 'volume_zscore']
CONTEXT_FEATURES = ['btc_is_bull_regime', 'btc_trend_strength', 'adx', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'btc_corr', 'trend_state', 'is_trending', 'is_volatile', 'macd_acceleration', 'volume_spike', 'vol_ratio_alpha', 'ema_200_1d_dist', 'rsi_14_1d']
SIGNAL_FEATURES = ['rsi_14', 'rsi_slope', 'stoch_k', 'stoch_d', 'roc_7', 'roc_14', 'volume_ratio', 'volume_zscore', 'volume_trend', 'rs_vs_btc', 'rs_vs_btc_sma7', 'vol_compression', 'dist_to_high_30d', 'dist_to_low_30d', 'dist_to_ema_21_pct', 'dist_to_ema_50_pct', 'dist_to_ema_200_pct', 'price_vs_sma_30', 'momentum_30', 'is_bullish_cross']

def calculate_hybrid_features(df, btc_context=None):
    df = df.copy()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility_14'] = df['log_returns'].rolling(14).std()
    df['body_size'] = abs(df['close'] - df['open']) / df['open']
    df['high_low_range'] = (df['high'] - df['low']) / df['low']
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volume_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
    df['volume_spike'] = (df['volume'] > df['volume'].rolling(20).mean() * 2).astype(int)
    df['volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    df['ema_200'] = df['close'].ewm(span=200).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    tr = pd.concat([df['high'] - df['low'], abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['vol_compression'] = df['atr_14'] / df['atr_14'].rolling(100).mean().replace(0, np.nan)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['rsi_slope'] = df['rsi_14'].diff(3)
    l14 = df['low'].rolling(14).min(); h14 = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * (df['close'] - l14) / (h14 - l14).replace(0, np.nan)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    df['roc_7'] = df['close'].pct_change(7); df['roc_14'] = df['close'].pct_change(14)
    df['price_vs_sma_30'] = df['close'] / (df['sma_30'] + 1e-9); df['momentum_30'] = df['close'].pct_change(30)
    pdm = df['high'].diff(); mdm = -df['low'].diff()
    pdm = pdm.where((pdm > mdm) & (pdm > 0), 0); mdm = mdm.where((mdm > pdm) & (mdm > 0), 0)
    atr_s = tr.rolling(14).mean()
    pdi = 100 * (pdm.rolling(14).mean() / atr_s.replace(0, np.nan))
    mdi = 100 * (mdm.rolling(14).mean() / atr_s.replace(0, np.nan))
    df['adx'] = (100 * abs(pdi - mdi) / (pdi + mdi).replace(0, np.nan)).rolling(14).mean()
    df['dist_to_high_30d'] = (df['close'] - df['high'].rolling(30).max()) / df['close']
    df['dist_to_low_30d'] = (df['close'] - df['low'].rolling(30).min()) / df['close']
    for e in [21, 50, 200]: df[f'dist_to_ema_{e}_pct'] = (df['close'] - df[f'ema_{e}']) / df['close']
    df['trend_state'] = np.where(df['close'] > df['sma_50'], 1, np.where(df['close'] < df['sma_50'], -1, 0))
    df['is_trending'] = (df['adx'] > 25).astype(int); df['is_volatile'] = (df['vol_compression'] > 1.5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24); df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7); df['day_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
    ef = df['close'].ewm(span=12).mean(); es = df['close'].ewm(span=26).mean()
    df['macd'] = ef - es; df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_slope'] = df['macd'].diff(); df['macd_acceleration'] = df['macd_slope'].diff()
    df['macd_cross_up'] = ((df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
    df['macd_cross_down'] = ((df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
    df['is_bullish_cross'] = df['macd_cross_up']; df['vol_ratio_alpha'] = df['volume_ratio'] * df['volatility_14']
    df['ema_200_1d_dist'] = df['dist_to_ema_200_pct']; df['rsi_14_1d'] = df['rsi_14']
    if btc_context is not None and not btc_context.empty:
        df = df.merge(btc_context, on='timestamp', how='left')
        for c in ['btc_is_bull_regime', 'btc_trend_strength', 'btc_returns']: df[c] = df[c].ffill().fillna(0)
        df['rs_vs_btc'] = df['log_returns'] - df['btc_returns']
        df['rs_vs_btc_sma7'] = df['rs_vs_btc'].rolling(7).mean()
        df['btc_corr'] = df['log_returns'].rolling(14).corr(df['btc_returns']).fillna(0)
    return df.dropna(subset=['macd'])

class PredictionAnalyzer:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.lgbm = joblib.load(model_path / 'ensemble_lgbm_tabular.joblib')
        self.meta = joblib.load(model_path / 'ensemble_meta.joblib')
        num_symbols = self.meta.get('num_symbols', 0)
        window_size = self.meta.get('window_size', 50)
        self.nn_model = HybridScorer(
            seq_in_dim=len(SEQ_FEATURES),
            context_in_dim=len(CONTEXT_FEATURES),
            signal_in_dim=len(SIGNAL_FEATURES),
            num_symbols=num_symbols,
            window_size=window_size
        )
        self.nn_model.load_state_dict(torch.load(model_path / 'ensemble_transformer.pth', map_location='cpu'))
        self.nn_model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nn_model.to(self.device)

    def get_predictions(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        window_size = self.meta['window_size']
        results = []
        signals = df[(df['macd_cross_up'] == 1) | (df['macd_cross_down'] == 1)].index
        
        # Symbol Encoding
        try:
            sym_idx = self.meta['sym_encoder'].transform([symbol])[0]
        except:
            sym_idx = 0
            
        for idx in signals:
            if idx < window_size: continue
            row = df.iloc[idx]
            seq = df.iloc[idx - window_size + 1 : idx + 1][SEQ_FEATURES].fillna(0).values.astype(float)
            seq_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device).float()
            ctx = row[CONTEXT_FEATURES].values.astype(float).reshape(1, -1)
            sig = row[SIGNAL_FEATURES].values.astype(float).reshape(1, -1)
            ctx_t = torch.tensor(ctx, dtype=torch.float32).to(self.device)
            sig_t = torch.tensor(sig, dtype=torch.float32).to(self.device)
            sym_t = torch.tensor([sym_idx]).long().to(self.device)
            
            with torch.no_grad():
                nn_logit = self.nn_model(seq_t, ctx_t, sig_t, sym_t)
                nn_prob = torch.sigmoid(nn_logit).item()
                emb = self.nn_model.get_embeddings(seq_t, ctx_t, sig_t, sym_t)
            
            lgbm_input = np.hstack([emb, ctx, sig])
            lgbm_prob = self.lgbm.predict_proba(lgbm_input)[0, 1]
            ensemble_prob = (nn_prob + lgbm_prob) / 2
            
            future_prices = df.iloc[idx+1 : idx+16]['close']
            if future_prices.empty: continue
            if row['macd_cross_up'] == 1:
                target = 1 if (future_prices.max() - row['close']) / row['close'] >= 0.05 else 0
                actual_ret = (future_prices.max() - row['close']) / row['close']
            else:
                target = 1 if (row['close'] - future_prices.min()) / row['close'] >= 0.05 else 0
                actual_ret = (row['close'] - future_prices.min()) / row['close']
            results.append({'timestamp': row['timestamp'], 'nn_prob': nn_prob, 'lgbm_prob': lgbm_prob, 'ensemble_prob': ensemble_prob, 'target': target, 'return': actual_ret, 'disagreement': abs(nn_prob - lgbm_prob)})
        return pd.DataFrame(results)

def run_analysis():
    analyzer = PredictionAnalyzer(MODEL_PATH)
    print("Loading data and building context...")
    btc_df = pd.read_parquet(DATA_PATH / 'BTCUSDT_USDT.parquet')
    btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'], unit='ms')
    btc_df = btc_df.set_index('timestamp').resample('1D').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna().reset_index()
    btc_df['log_returns'] = np.log(btc_df['close'] / btc_df['close'].shift(1))
    btc_df['sma_200'] = btc_df['close'].rolling(200).mean()
    tr = pd.concat([btc_df['high'] - btc_df['low'], abs(btc_df['high'] - btc_df['close'].shift(1)), abs(btc_df['low'] - btc_df['close'].shift(1))], axis=1).max(axis=1)
    atr = tr.rolling(14).mean(); pdm = btc_df['high'].diff(); mdm = -btc_df['low'].diff()
    pdm = pdm.where((pdm > mdm) & (pdm > 0), 0); mdm = mdm.where((mdm > pdm) & (mdm > 0), 0)
    pdi = 100 * (pdm.rolling(14).mean() / atr.replace(0, np.nan)); mdi = 100 * (mdm.rolling(14).mean() / atr.replace(0, np.nan))
    btc_df['btc_adx'] = (100 * abs(pdi - mdi) / (pdi + mdi).replace(0, np.nan)).rolling(14).mean()
    btc_context = btc_df[['timestamp','close','sma_200','btc_adx','log_returns']].copy()
    btc_context.columns = ['timestamp','btc_close','btc_sma_200','btc_adx','btc_returns']
    btc_context['btc_is_bull_regime'] = (btc_context['btc_close'] > btc_context['btc_sma_200']).astype(int)
    btc_context['btc_trend_strength'] = (btc_context['btc_adx'] > 25).astype(int)
    
    symbols = [f.stem.replace('_USDT', '') for f in DATA_PATH.glob('*.parquet')]
    symbols = [s for s in symbols if not any(x in s for x in ['-26','-25','-24'])][:50]
    all_results = []
    for sym in symbols:
        try:
            pt = DATA_PATH / f"{sym}_USDT.parquet"
            if not pt.exists(): continue
            df = pd.read_parquet(pt)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp').resample('1D').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna().reset_index()
            if len(df) < 250: continue
            df = calculate_hybrid_features(df, btc_context)
            df = df[df['timestamp'] >= pd.Timestamp(TEST_START_DATE)]
            if len(df) < 10: continue
            df[CONTEXT_FEATURES] = analyzer.meta['ctx_scaler'].transform(df[CONTEXT_FEATURES].fillna(0))
            df[SIGNAL_FEATURES] = analyzer.meta['sig_scaler'].transform(df[SIGNAL_FEATURES].fillna(0))
            df[SEQ_FEATURES] = analyzer.meta['seq_scaler'].transform(df[SEQ_FEATURES].fillna(0))
            all_results.append(analyzer.get_predictions(sym, df))
            print(f"Processed {sym}")
        except Exception as e: print(f"Error {sym}: {e}")
            
    if not all_results: print("No results."); return
    master_df = pd.concat(all_results)
    print("Generating Reports...")
    master_df['bin'] = pd.qcut(master_df['ensemble_prob'], 10, labels=False, duplicates='drop')
    decile_stats = master_df.groupby('bin').agg({'target': 'mean', 'return': 'mean'}).reset_index()
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); sns.barplot(x='bin', y='target', data=decile_stats, palette='viridis'); plt.title('Win Rate by Decile (Target 1)')
    plt.subplot(1, 2, 2); sns.barplot(x='bin', y='return', data=decile_stats, palette='rocket'); plt.title('Avg Return by Decile')
    plt.savefig('decile_analysis.png')
    plt.figure(figsize=(10, 5)); sns.kdeplot(master_df['nn_prob'], label='Transformer', fill=True, alpha=0.3); sns.kdeplot(master_df['lgbm_prob'], label='LightGBM', fill=True, alpha=0.3); plt.title('Prob Distribution'); plt.legend(); plt.savefig('prob_distribution.png')
    
    # Feature Importance (Simplified column names)
    emb_cols = [f'emb_{i}' for i in range(64)]
    all_feat_names = emb_cols + CONTEXT_FEATURES + SIGNAL_FEATURES
    feat_imp = pd.DataFrame({'feature': all_feat_names, 'importance': analyzer.lgbm.feature_importances_}).sort_values('importance', ascending=False)
    plt.figure(figsize=(10, 8)); sns.barplot(x='importance', y='feature', data=feat_imp.head(20)); plt.title('Top 20 Features (with Embeddings)'); plt.savefig('feature_importance.png')
    
    print(f"Done. Samples: {len(master_df)}, WinRate: {master_df['target'].mean():.2%}")

if __name__ == "__main__":
    run_analysis()
