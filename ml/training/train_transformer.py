import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import roc_auc_score
# from torchinfo import summary

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from transformer_model import HybridScorer
from training_utils import (
    PROCESSED_DIR, MODELS_DIR, get_feature_columns
)

# Configuration
WINDOW_SIZE = 50
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 5e-5 # Reduced from 1e-4 for finer convergence
WEIGHT_DECAY = 0.05  # Added for AdamW regularization
GAMMA_FOCAL = 5.0    # Increased from 3.0 for Hard Negative Mining
ALPHA_FOCAL = 0.77   # Adjusted for win rate balance

# Features for Branch 1 (Sequence)
SEQ_FEATURES = ['log_returns', 'high_low_range', 'body_size', 'volatility_14', 'macd', 'macd_slope', 'volume_ratio']

# Context Features (Query Bias) - Added "Explosive" signals to drive the Query
CONTEXT_FEATURES = [
    'btc_is_bull_regime', 'btc_trend_strength', 'adx', 'hour_sin', 'hour_cos', 
    'day_sin', 'day_cos', 'btc_corr', 'trend_state', 'is_trending', 'is_volatile',
    'macd_acceleration', 'volume_spike', 'vol_ratio_alpha'  # Explosive signals moved here!
]

# Signal Features (Fusion)
SIGNAL_FEATURES = [
    'rsi_14', 'rsi_slope', 'stoch_k', 'stoch_d', 'roc_7', 'roc_14', 'volume_ratio', 
    'volume_trend', 'rs_vs_btc', 'rs_vs_btc_sma7', 'vol_compression', 
    'dist_to_high_30d', 'dist_to_low_30d', 'dist_to_ema_21_pct', 
    'dist_to_ema_50_pct', 'dist_to_ema_200_pct', 'is_bullish_cross'
]

# SAM class removed — using standard AdamW for noisy financial data
# SAM can trap the model in narrow minima when data is highly noisy

class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss for hard examples
    """
    def __init__(self, alpha=0.25, gamma=3.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # Targets are now continuous probabilities [0, 1]
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        
        # Per-class alpha: target=1 -> alpha, target=0 -> (1-alpha)
        alpha_factor = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        focal_loss = alpha_factor * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class AUCRankingLoss(nn.Module):
    """
    Phase 6: Margin Ranking Loss to directly optimize AUC ordering.
    Forces pred(TP_HIT) > pred(SL_HIT) by at least a margin.
    """
    def __init__(self, margin=0.3, n_pairs=256):
        super().__init__()
        self.margin = margin
        self.n_pairs = n_pairs
    
    def forward(self, logits, targets):
        preds = torch.sigmoid(logits.squeeze(1))
        targets_flat = targets.squeeze(1)
        
        # Find "hard wins" (TP_HIT, label >= 0.9) and "clear losses" (SL_HIT, label <= 0.1)
        pos_idx = (targets_flat >= 0.9).nonzero(as_tuple=True)[0]
        neg_idx = (targets_flat <= 0.1).nonzero(as_tuple=True)[0]
        
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            return torch.tensor(0.0, device=logits.device)
        
        # Sample random pairs (up to n_pairs)
        n = min(self.n_pairs, len(pos_idx) * len(neg_idx))
        pos_sample = pos_idx[torch.randint(len(pos_idx), (n,))]
        neg_sample = neg_idx[torch.randint(len(neg_idx), (n,))]
        
        # Hinge loss: max(0, margin - (pred_pos - pred_neg))
        diff = preds[pos_sample] - preds[neg_sample]
        loss = F.relu(self.margin - diff).mean()
        return loss

class CombinedLoss(nn.Module):
    """
    BCE (perfect for Soft Labels calibration) + RankingLoss (AUC ordering).
    Focal Loss removed: gamma=5 on soft labels crushes gradients → inverse learning.
    """
    def __init__(self, rank_weight=0.5, margin=0.3):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.rank = AUCRankingLoss(margin=margin)
        self.rank_weight = rank_weight

    def forward(self, logits, targets):
        return self.bce(logits, targets) + self.rank_weight * self.rank(logits, targets)

class HybridDataset(Dataset):
    def __init__(self, seq_data: torch.Tensor, context_data: torch.Tensor, signal_data: torch.Tensor, sym_data: torch.Tensor, labels: torch.Tensor):
        self.seq_data = seq_data
        self.context_data = context_data
        self.signal_data = signal_data
        self.sym_data = sym_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seq_data[idx], self.context_data[idx], self.signal_data[idx], self.sym_data[idx], self.labels[idx]

def prepare_hybrid_data(timeframe: str, window_size: int = 50) -> Dict:
    data_path = PROCESSED_DIR / f'features_{timeframe}_full.parquet'
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
    
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 1. Available Sequence Features
    available_seq = [f for f in SEQ_FEATURES if f in df.columns]
    
    # Identify Crossover Points (Points of interest)
    mask = ((df['macd_cross_up'] == 1) | (df['macd_cross_down'] == 1)) & df['label'].notnull()
    cross_indices = df[mask].index.tolist()
    cross_indices = [idx for idx in cross_indices if idx >= window_size - 1]
    
    # 2. Get Labels - PRESERVE soft labels from MAE/MFE scoring
    raw_labels = df.loc[cross_indices, 'label'].astype(float).values
    smoothed_labels = np.clip(raw_labels, 0.05, 0.95)  # Only clamp extremes
    
    # 3. Get Categorical Embeddings Feature
    sym_encoder = LabelEncoder()
    df['symbol_encoded'] = sym_encoder.fit_transform(df['symbol'])
    sym_data = df.loc[cross_indices, 'symbol_encoded'].values
    num_symbols = len(sym_encoder.classes_)
    
    # --- Split Tabular Features into Context and Signal ---
    df['is_bullish_cross'] = df['macd_cross_up'].values
    
    available_context = [f for f in CONTEXT_FEATURES if f in df.columns]
    available_signal = [f for f in SIGNAL_FEATURES if f in df.columns]
    
    context_data = df.loc[cross_indices, available_context].copy().fillna(0).replace([np.inf, -np.inf], 0)
    signal_data = df.loc[cross_indices, available_signal].copy().fillna(0).replace([np.inf, -np.inf], 0)
    
    # Robust Scaling for volatile features, Standard for others
    volatile_ctx = ['btc_returns', 'btc_corr']
    volatile_sig = ['macd_acceleration', 'volume_spike', 'rsi_slope', 'vol_ratio_alpha']
    
    # Simple split scaling approach
    ctx_scaler = RobustScaler()
    sig_scaler = RobustScaler()
    context_scaled = ctx_scaler.fit_transform(context_data)
    signal_scaled = sig_scaler.fit_transform(signal_data)
    
    # --- Sequence Data ---
    sequences = []
    print(f"Generating {len(cross_indices)} sequences...")
    df_seq = df[available_seq].copy().fillna(0).replace([np.inf, -np.inf], 0)
    seq_scaler = StandardScaler()
    df_seq_scaled = seq_scaler.fit_transform(df_seq)
    
    for idx in cross_indices:
        seq = df_seq_scaled[idx - window_size + 1 : idx + 1]
        sequences.append(seq)
    sequences = np.array(sequences, dtype=np.float32)
    
    return {
        'sequences': sequences,
        'context': context_scaled,
        'signal': signal_scaled,
        'symbols': sym_data,
        'labels': smoothed_labels,
        'raw_labels': raw_labels,
        'available_context': available_context,
        'available_signal': available_signal,
        'available_seq': available_seq,
        'ctx_scaler': ctx_scaler,
        'sig_scaler': sig_scaler,
        'seq_scaler': seq_scaler,
        'sym_encoder': sym_encoder,
        'num_symbols': num_symbols
    }

def get_lr_multiplier(epoch, warmup_epochs=5):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 1.0

def train_hybrid(timeframe: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    data = prepare_hybrid_data(timeframe, WINDOW_SIZE)
    sequences = torch.tensor(data['sequences'], dtype=torch.float32)
    context = torch.tensor(data['context'], dtype=torch.float32)
    signal = torch.tensor(data['signal'], dtype=torch.float32)
    symbols = torch.tensor(data['symbols'], dtype=torch.long)
    labels = torch.tensor(data['labels'], dtype=torch.float32).unsqueeze(1)
    raw_labels = data['raw_labels']
    
    split_idx = int(len(labels) * 0.8)
    train_ds = HybridDataset(sequences[:split_idx], context[:split_idx], signal[:split_idx], symbols[:split_idx], labels[:split_idx])
    test_ds = HybridDataset(sequences[split_idx:], context[split_idx:], signal[split_idx:], symbols[split_idx:], labels[split_idx:])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = HybridScorer(
        seq_in_dim=len(data['available_seq']),
        context_in_dim=context.shape[1],
        signal_in_dim=signal.shape[1],
        num_symbols=data['num_symbols'],
        window_size=WINDOW_SIZE
    ).to(device)
    
    # Phase 8: BCE + Ranking Loss (No Focal)
    criterion = CombinedLoss(rank_weight=0.5, margin=0.3)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(EPOCHS, 20))
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Simplified training: AdamW + BCE + Ranking | {n_params:,} params")
    best_auc = 0
    test_raw_labels = raw_labels[split_idx:]
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        # LR Warmup
        lr_scale = get_lr_multiplier(epoch)
        for g in optimizer.param_groups:
            g['lr'] = LEARNING_RATE * lr_scale
            
        for batch_seq, batch_ctx, batch_sig, batch_sym, batch_label in train_loader:
            batch_seq, batch_ctx, batch_sig, batch_sym, batch_label = \
                batch_seq.to(device), batch_ctx.to(device), batch_sig.to(device), batch_sym.to(device), batch_label.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_seq, batch_ctx, batch_sig, batch_sym)
            loss = criterion(outputs, batch_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
            
        # Validation
        model.eval()
        all_preds = []
        with torch.no_grad():
            for batch_seq, batch_ctx, batch_sig, batch_sym, _ in test_loader:
                batch_seq, batch_ctx, batch_sig, batch_sym = \
                    batch_seq.to(device), batch_ctx.to(device), batch_sig.to(device), batch_sym.to(device)
                outputs = model(batch_seq, batch_ctx, batch_sig, batch_sym)
                probs = torch.sigmoid(outputs)
                all_preds.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        
        # 1. HARD AUC: Only clear TP_HIT (>=0.9) vs SL_HIT (<=0.1)
        hard_mask = (test_raw_labels >= 0.9) | (test_raw_labels <= 0.1)
        if hard_mask.sum() > 0:
            hard_targets = (test_raw_labels[hard_mask] >= 0.9).astype(int)
            hard_preds = all_preds[hard_mask]
            if len(np.unique(hard_targets)) > 1:
                val_auc_hard = roc_auc_score(hard_targets, hard_preds)
            else:
                val_auc_hard = 0.5
        else:
            val_auc_hard = 0.5
        
        # 2. ALL AUC: Full dataset with 0.5 threshold
        binary_test_labels = (test_raw_labels >= 0.5).astype(int)
        if len(np.unique(binary_test_labels)) > 1:
            val_auc_all = roc_auc_score(binary_test_labels, all_preds)
        else:
            val_auc_all = 0.5
        
        print(f"Epoch {epoch+1}/{EPOCHS} (LR: {optimizer.param_groups[0]['lr']:.6f}) - Loss: {train_loss/len(train_loader):.4f} | Hard AUC: {val_auc_hard:.4f} | All AUC: {val_auc_all:.4f}")
        
        # Save model based on HARD AUC only
        if val_auc_hard > best_auc:
            best_auc = val_auc_hard
            model_dir = MODELS_DIR / timeframe
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_dir / 'hybrid_transformer.pth')
            
            joblib.dump({
                'ctx_scaler': data['ctx_scaler'],
                'sig_scaler': data['sig_scaler'],
                'seq_scaler': data['seq_scaler'],
                'sym_encoder': data['sym_encoder'],
                'context_features': data['available_context'],
                'signal_features': data['available_signal'],
                'seq_features': data['available_seq'],
                'num_symbols': data['num_symbols'],
                'window_size': WINDOW_SIZE,
                'val_auc_hard': best_auc,
                'val_auc_all': val_auc_all,
                'trained_at': datetime.now().isoformat()
            }, model_dir / 'hybrid_transformer_meta.joblib')
            print(f"  >>> New best model saved! (Hard AUC: {best_auc:.4f})")
    
    print(f"\nFinal Best Hard AUC: {best_auc:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('timeframe', help='Timeframe to train (e.g., 1d)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()
    
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    
    train_hybrid(args.timeframe)
