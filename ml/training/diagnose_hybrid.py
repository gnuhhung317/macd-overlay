import torch
import joblib
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from transformer_model import HybridScorer

def diagnose_hybrid(timeframe: str):
    """
    Diagnose Hybrid model by checking metadata and input dims.
    """
    # Adjust path if running from different dirs
    base_dir = Path(__file__).parent.parent
    model_dir = base_dir / 'models' / timeframe
    model_path = model_dir / 'hybrid_transformer.pth'
    meta_path = model_dir / 'hybrid_transformer_meta.joblib'
    
    if not (model_path.exists() and meta_path.exists()):
        print(f"Model or meta not found in {model_dir}")
        return
        
    meta = joblib.load(meta_path)
    
    # Extract dims from new metadata keys
    seq_in_dim = len(meta['seq_features'])
    ctx_in_dim = len(meta['context_features'])
    sig_in_dim = len(meta['signal_features'])
    num_symbols = meta.get('num_symbols', 0)
    window_size = meta.get('window_size', 50)
    
    print("--- Model Metadata ---")
    print(f"Timeframe: {timeframe}")
    print(f"Sequence Dims: {seq_in_dim}")
    print(f"Context Dims: {ctx_in_dim}")
    print(f"Signal Dims: {sig_in_dim}")
    print(f"Window Size: {window_size}")
    print(f"Val AUC: {meta.get('val_auc', 'N/A'):.4f}")
    
    # Initialize model with new architecture parameters
    model = HybridScorer(
        seq_in_dim=seq_in_dim, 
        context_in_dim=ctx_in_dim, 
        signal_in_dim=sig_in_dim,
        num_symbols=num_symbols,
        window_size=window_size
    )
    
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    print("\n✓ Model loaded successfully with the new architecture.")
    
    # Check feature consistency
    print("\nContext Features (Explosive):")
    for f in meta['context_features']:
        if f in ['macd_acceleration', 'volume_spike', 'vol_ratio_alpha']:
            print(f"  - {f} [ACTIVE]")
        else:
            print(f"  - {f}")

if __name__ == "__main__":
    timeframe = sys.argv[1] if len(sys.argv) > 1 else '1d'
    diagnose_hybrid(timeframe)
