"""
Hybrid 2.0: Transformer Embeddings + LightGBM (AutoEncoder + Soft Label Regression)

Architecture:
  1. Transformer pre-trained as AutoEncoder (Reconstruction task)
  2. 64-dim embeddings + tabular indicators -> LightGBM Regressor
  3. LightGBM trained on Soft Labels (MFE/MAE probabilities)
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from transformer_model import HybridScorer
from training_utils import PROCESSED_DIR, MODELS_DIR

# ============================================================
# CONFIG
# ============================================================
WINDOW_SIZE = 50
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05

SEQ_FEATURES = ['log_returns', 'high_low_range', 'body_size', 'volatility_14', 'macd', 'macd_slope', 'volume_ratio']
CONTEXT_FEATURES = [
    'btc_is_bull_regime', 'btc_trend_strength', 'adx', 'hour_sin', 'hour_cos', 
    'day_sin', 'day_cos', 'btc_corr', 'trend_state', 'is_trending', 'is_volatile',
    'macd_acceleration', 'volume_spike', 'vol_ratio_alpha'
]
SIGNAL_FEATURES = [
    'rsi_14', 'rsi_slope', 'stoch_k', 'stoch_d', 'roc_7', 'roc_14', 'volume_ratio', 
    'volume_trend', 'rs_vs_btc', 'rs_vs_btc_sma7', 'vol_compression', 
    'dist_to_high_30d', 'dist_to_low_30d', 'dist_to_ema_21_pct', 
    'dist_to_ema_50_pct', 'dist_to_ema_200_pct', 'is_bullish_cross'
]

# ============================================================
# DATASET
# ============================================================
class SimpleDataset(Dataset):
    def __init__(self, seq, ctx, sig, sym, labels):
        self.seq = seq; self.ctx = ctx; self.sig = sig
        self.sym = sym; self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return self.seq[i], self.ctx[i], self.sig[i], self.sym[i], self.labels[i]

def load_data(timeframe: str):
    data_path = PROCESSED_DIR / f'features_{timeframe}_full.parquet'
    if not data_path.exists():
        data_path = Path(f'/kaggle/working/features_{timeframe}_full.parquet')
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
    
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path).sort_values('timestamp').reset_index(drop=True)
    
    mask = ((df['macd_cross_up'] == 1) | (df['macd_cross_down'] == 1)) & df['label'].notnull()
    idx = df[mask].index[df[mask].index >= WINDOW_SIZE - 1]
    
    raw_labels = df.loc[idx, 'label'].astype(float).values
    sym_encoder = LabelEncoder()
    df['symbol_encoded'] = sym_encoder.fit_transform(df['symbol'])
    sym_data = df.loc[idx, 'symbol_encoded'].values
    
    df['is_bullish_cross'] = df['macd_cross_up'].values
    avail_ctx = [f for f in CONTEXT_FEATURES if f in df.columns]
    avail_sig = [f for f in SIGNAL_FEATURES if f in df.columns]
    
    ctx_scaler = RobustScaler().fit(df.loc[idx, avail_ctx].fillna(0).replace([np.inf, -np.inf], 0))
    sig_scaler = RobustScaler().fit(df.loc[idx, avail_sig].fillna(0).replace([np.inf, -np.inf], 0))
    
    ctx_data = ctx_scaler.transform(df.loc[idx, avail_ctx].fillna(0).replace([np.inf, -np.inf], 0)).astype(np.float32)
    sig_data = sig_scaler.transform(df.loc[idx, avail_sig].fillna(0).replace([np.inf, -np.inf], 0)).astype(np.float32)
    
    avail_seq = [f for f in SEQ_FEATURES if f in df.columns]
    seq_scaler = StandardScaler().fit(df[avail_seq].fillna(0).replace([np.inf, -np.inf], 0))
    df_seq_scaled = seq_scaler.transform(df[avail_seq].fillna(0).replace([np.inf, -np.inf], 0)).astype(np.float32)
    
    sequences = np.array([df_seq_scaled[i - WINDOW_SIZE + 1 : i + 1] for i in idx], dtype=np.float32)
    
    print(f"  Rows: {len(idx)} | Seqs: {sequences.shape} | Labels Mean: {raw_labels.mean():.3f}")
    return {
        'sequences': sequences, 'context': ctx_data, 'signal': sig_data,
        'symbols': sym_data, 'raw_labels': raw_labels,
        'avail_seq': avail_seq, 'avail_ctx': avail_ctx, 'avail_sig': avail_sig,
        'ctx_scaler': ctx_scaler, 'sig_scaler': sig_scaler, 'seq_scaler': seq_scaler,
        'sym_encoder': sym_encoder, 'num_symbols': len(sym_encoder.classes_)
    }

# ============================================================
# TRAINING LOGIC
# ============================================================
def run_lgbm(X_tr, X_te, y_tr, y_te, names, title):
    """Common LightGBM Regressor training logic."""
    print(f"\n{'='*50}\n{title}\n{'='*50}")
    m = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.02, max_depth=7, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, objective='cross_entropy',
        random_state=42, n_jobs=-1, verbose=-1
    )
    hm = (y_te >= 0.9) | (y_te <= 0.1); X_h = X_te[hm]; y_h = (y_te[hm] >= 0.9).astype(int)
    m.fit(X_tr, y_tr, eval_set=[(X_h, y_h)], eval_metric='auc',
          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)])
    
    ph = m.predict(X_h); ha = roc_auc_score(y_h, ph)
    yab = (y_te >= 0.5).astype(int)
    pa = m.predict(X_te); aa = roc_auc_score(yab, pa) if len(np.unique(yab)) > 1 else 0.5
    print(f"  🚀 Hard AUC: {ha:.4f} | All AUC: {aa:.4f}")
    imp = m.feature_importances_; top = np.argsort(imp)[-10:][::-1]
    print("  Top features:"); [print(f"    {i+1}. {names[j]:25s}: {imp[j]}") for i, j in enumerate(top)]
    return m, ha, aa

def train_hybrid_lgbm(timeframe: str, pretrain_epochs: int = 30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(timeframe)
    
    t_seq = torch.tensor(data['sequences'], dtype=torch.float32)
    t_ctx = torch.tensor(data['context'], dtype=torch.float32)
    t_sig = torch.tensor(data['signal'], dtype=torch.float32)
    t_sym = torch.tensor(data['symbols'], dtype=torch.long)
    raw_labels = data['raw_labels']
    t_lab = torch.tensor(np.clip(raw_labels, 0.05, 0.95), dtype=torch.float32).unsqueeze(1)
    
    split = int(len(t_lab) * 0.8)
    train_ds = SimpleDataset(t_seq[:split], t_ctx[:split], t_sig[:split], t_sym[:split], t_lab[:split])
    test_ds = SimpleDataset(t_seq[split:], t_ctx[split:], t_sig[split:], t_sym[split:], t_lab[split:])
    
    model = HybridScorer(
        seq_in_dim=len(data['avail_seq']), context_in_dim=t_ctx.shape[1],
        signal_in_dim=t_sig.shape[1], num_symbols=data['num_symbols'], window_size=WINDOW_SIZE
    ).to(device)
    
    # === STEP 1: Pre-train Transformer (AUTOENCODER MODE) ===
    print(f"\n{'='*50}\nStep 1: Pre-training Transformer (AutoEncoder - {pretrain_epochs} epochs)\n{'='*50}")
    
    # Dùng MSE (Mean Squared Error) thay vì BCE vì ta đang so sánh Nến với Nến
    criterion_ae = nn.MSELoss()
    
    # AutoEncoder học rất nhanh, nên ta dùng LR lớn hơn (1e-3)
    optimizer_ae = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_shuffle = DataLoader(train_ds, batch_size=64, shuffle=True)
    
    for epoch in range(pretrain_epochs):
        model.train()
        tl = 0
        for bs, bc, bg, by, bl in train_shuffle:
            bs = bs.to(device)
            optimizer_ae.zero_grad()
            
            # 1. Nén và Giải nén chuỗi nến
            reconstructed_seq = model.forward_ae(bs)
            
            # 2. So sánh bản sao với bản gốc (Không thèm quan tâm đến nhãn Mua/Bán)
            loss = criterion_ae(reconstructed_seq, bs)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_ae.step()
            tl += loss.item()
            
        print(f"  Epoch {epoch+1}/{pretrain_epochs} | Reconstruction MSE: {tl/len(train_shuffle):.4f}")
    
    # === STEP 2: Extract Embeddings ===
    print(f"\nStep 2: Extracting Neural Embeddings")
    model.eval()
    def extract(loader):
        all_x, all_y = [], []
        with torch.no_grad():
            for bs, bc, bg, by, bl in loader:
                emb = model.get_embeddings(bs.to(device), bc.to(device), bg.to(device), by.to(device))
                tab = torch.cat([bc, bg], dim=1).numpy()
                all_x.append(np.hstack([emb, tab])); all_y.append(bl.numpy())
        return np.vstack(all_x), np.concatenate(all_y).flatten()
    
    tr_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    te_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    X_tr, _ = extract(tr_loader); X_te, _ = extract(te_loader)
    y_tr_raw = raw_labels[:split]; y_te_raw = raw_labels[split:]
    
    d_model = 64; tab_names = data['avail_ctx'] + data['avail_sig']
    feat_names = [f'emb_{i}' for i in range(d_model)] + tab_names
    
    # === STEP 3: Train LightGBM ===
    m_full, ha_full, aa_full = run_lgbm(X_tr, X_te, y_tr_raw, y_te_raw, feat_names, "Step 3a: Hybrid (Emb + Tab)")
    m_tab, ha_tab, aa_tab = run_lgbm(X_tr[:, d_model:], X_te[:, d_model:], y_tr_raw, y_te_raw, tab_names, "Step 3b: Baseline (Tab Only)")
    
    print(f"\nFinal Comparison:\n  Tab Only: {ha_tab:.4f}\n  Hybrid:   {ha_full:.4f}\n  Boost:    {ha_full-ha_tab:+.4f}")
    
    # Save artifacts
    out_dir = MODELS_DIR / timeframe; out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(m_full if ha_full >= ha_tab else m_tab, out_dir / 'lgbm_hybrid.joblib')
    torch.save(model.state_dict(), out_dir / 'transformer_embedder.pth')
    joblib.dump({
        'ctx_scaler': data['ctx_scaler'], 'sig_scaler': data['sig_scaler'], 'seq_scaler': data['seq_scaler'],
        'sym_encoder': data['sym_encoder'], 'context_features': data['avail_ctx'],
        'signal_features': data['avail_sig'], 'seq_features': data['avail_seq'],
        'num_symbols': data['num_symbols'], 'window_size': WINDOW_SIZE,
        'hard_auc': max(ha_full, ha_tab), 'trained_at': datetime.now().isoformat()
    }, out_dir / 'hybrid_lgbm_meta.joblib')
    print(f"Saved to {out_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('timeframe')
    parser.add_argument('--pretrain-epochs', type=int, default=30)
    args = parser.parse_args()
    train_hybrid_lgbm(args.timeframe, args.pretrain_epochs)
