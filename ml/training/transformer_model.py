import torch
import torch.nn as nn
import torch.nn.functional as F

class DropPath(nn.Module):
    def __init__(self, dp=0.1):
        super().__init__(); self.dp=dp
    def forward(self, x):
        if not self.training or self.dp==0: return x
        k=1-self.dp; return x/k*torch.floor(torch.rand((x.shape[0],)+(1,)*(x.ndim-1),device=x.device)+k)

class HybridScorer(nn.Module):
    """
    Simplified Hybrid Scorer ("Back to Basics"):
    1. Sequence Branch: Linear projection + Transformer (no Conv1d multi-scale).
    2. Tabular Branch: ALL tabular features (context + signal + symbol) -> single MLP.
       This unified representation becomes the Cross-Attention Query.
    3. Cross-Attention fuses tabular query with sequence key/value.
    4. Simple classifier head.
    
    ~60% fewer params than previous version. Designed to reduce overfitting.
    """
    def __init__(
        self, 
        seq_in_dim, 
        context_in_dim,
        signal_in_dim,
        num_symbols=0,
        sym_emb_dim=16,
        d_model=64, 
        nhead=4, 
        num_layers=2, 
        window_size=50
    ):
        super().__init__()
        
        # -----------------------------------------
        # Branch 1: Sequence (Simple Linear + Transformer)
        # -----------------------------------------
        self.seq_proj = nn.Linear(seq_in_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, window_size, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 2,  # Reduced from 4x to 2x
            batch_first=True,
            dropout=0.15
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_seq = nn.LayerNorm(d_model)
        self.drop_path = DropPath(0.1)
        
        # -----------------------------------------
        # Branch 2: Unified Tabular (Context + Signal + Symbol -> Query)
        # -----------------------------------------
        self.has_emb = num_symbols > 0
        if self.has_emb:
            self.sym_emb = nn.Embedding(num_symbols, sym_emb_dim)
        
        all_tab_dim = context_in_dim + signal_in_dim + (sym_emb_dim if self.has_emb else 0)
        self.tab_branch = nn.Sequential(
            nn.Linear(all_tab_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        
        # -----------------------------------------
        # Fusion: Cross-Attention + Simple Classifier
        # -----------------------------------------
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=nhead, 
            batch_first=True,
            dropout=0.1
        )
        
        # Gated Multimodal Unit (GMU)
        self.gmu = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # Simplified classifier (no separate signal branch concatenation)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

        # Self-Supervised Reconstruction Head (AutoEncoder)
        self.reconstruct_head = nn.Linear(d_model, seq_in_dim)

    def forward(self, seq_x, context_x=None, signal_x=None, sym_x=None):
        xs = self.seq_proj(seq_x) + self.pos_encoder
        xs_out = self.transformer(xs); xs_enc = self.ln_seq(self.drop_path(xs_out) + xs)
        tab = [context_x, signal_x] + ([self.sym_emb(sym_x)] if self.has_emb and sym_x is not None else [])
        tc = torch.cat(tab, dim=1)
        if self.training: tc = tc + torch.randn_like(tc)*0.01
        te = self.tab_branch(tc)
        sc, _ = self.cross_attn(te.unsqueeze(1), xs_enc, xs_enc); sc = sc.squeeze(1)
        z = self.gmu(torch.cat([sc, te], dim=1)); f = z*sc + (1-z)*te
        return self.classifier(f)

    def forward_ae(self, seq_x):
        """Dùng riêng cho quá trình Pre-train AutoEncoder"""
        xs = self.seq_proj(seq_x) + self.pos_encoder
        xs_out = self.transformer(xs)
        xs_enc = self.ln_seq(self.drop_path(xs_out) + xs)
        return self.reconstruct_head(xs_enc)

    def get_embeddings(self, seq_x, ctx_x, sig_x, sym_x=None):
        self.eval()
        with torch.no_grad():
            xs = self.seq_proj(seq_x) + self.pos_encoder
            return self.ln_seq(self.transformer(xs)).mean(dim=1).cpu().numpy()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridScorer(seq_in_dim=7, context_in_dim=14, signal_in_dim=17, num_symbols=30).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
    
    dummy_seq = torch.randn(8, 50, 7).to(device)
    dummy_ctx = torch.randn(8, 14).to(device)
    dummy_sig = torch.randn(8, 17).to(device)
    dummy_sym = torch.randint(0, 30, (8,)).to(device)
    
    with torch.no_grad():
        output = model(dummy_seq, dummy_ctx, dummy_sig, dummy_sym)
        print(f"Output shape: {output.shape}")
        print(f"Sample logit: {output[0].item():.4f}")
    print("✓ Forward pass OK!")
