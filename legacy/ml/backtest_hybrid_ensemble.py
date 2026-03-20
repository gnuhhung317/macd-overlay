#!/usr/bin/env python3
"""
Hybrid Ensemble Chronological Backtest
- Loads Transformer + LightGBM Ensemble from ml/models/1d/hybrid
- Uses chronological walkthrough with multi-symbol ranking
- Implements Phase 11 feature engineering
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add project root to path (at the end to avoid naming conflicts with 'config')
root_dir = Path(__file__).parent.parent
ml_dir = Path(__file__).parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
if str(ml_dir) not in sys.path:
    sys.path.insert(0, str(ml_dir))

from backtest_3stage import ThreeStageBacktester, BacktestResult, BacktestConfig as BaseConfig, Trade as BaseTrade
from training.transformer_model import HybridScorer

# Feature Lists from Notebook
SEQ_FEATURES = ['log_returns', 'high_low_range', 'body_size', 'volatility_14', 'macd', 'macd_slope', 'volume_ratio', 'volume_zscore']
CONTEXT_FEATURES = ['btc_is_bull_regime', 'btc_trend_strength', 'adx', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'btc_corr', 'trend_state', 'is_trending', 'is_volatile', 'macd_acceleration', 'volume_spike', 'vol_ratio_alpha', 'ema_200_1d_dist', 'rsi_14_1d']
SIGNAL_FEATURES = ['rsi_14', 'rsi_slope', 'stoch_k', 'stoch_d', 'roc_7', 'roc_14', 'volume_ratio', 'volume_zscore', 'volume_trend', 'rs_vs_btc', 'rs_vs_btc_sma7', 'vol_compression', 'dist_to_high_30d', 'dist_to_low_30d', 'dist_to_ema_21_pct', 'dist_to_ema_50_pct', 'dist_to_ema_200_pct', 'price_vs_sma_30', 'momentum_30', 'is_bullish_cross']

@dataclass
class Trade(BaseTrade):
    is_sl_at_be: bool = False # Trailing SL flag

@dataclass
class BacktestConfig(BaseConfig):
    # Phase 6: Advanced Trade Mgmt
    trailing_sl_activation_pct: float = 0.5 # 50% profit distance to move to BE
    time_stop_bars: int = 7 # Check sluggishness after 7 bars

class HybridEnsembleBacktester(ThreeStageBacktester):
    def __init__(self, model_dir: Path, config: BacktestConfig = None):
        super().__init__(config)
        self.model_dir = model_dir
        self.lgbm_model = None
        self.nn_model = None
        self.meta = None
        
        self._load_ensemble_models()

    def _load_ensemble_models(self):
        """Load the 3 components of the Hybrid Ensemble"""
        print(f"Loading Hybrid Ensemble from {self.model_dir}...")
        
        lgbm_path = self.model_dir / 'ensemble_lgbm_tabular.joblib'
        nn_path = self.model_dir / 'ensemble_transformer.pth'
        meta_path = self.model_dir / 'ensemble_meta.joblib'
        
        if not (lgbm_path.exists() and nn_path.exists() and meta_path.exists()):
            raise FileNotFoundError(f"Missing ensemble files in {self.model_dir}")
            
        self.meta = joblib.load(meta_path)
        self.lgbm_model = joblib.load(lgbm_path)
        
        # Initialize NN model
        # Note: HybridScorer signature in transformer_model.py is slightly different from notebook
        # seq_in_dim, context_in_dim, signal_in_dim, num_symbols=0, sym_emb_dim=16, d_model=64, nhead=4, num_layers=2, window_size=50
        num_symbols = self.meta.get('num_symbols', 0)
        window_size = self.meta.get('window_size', 50)
        
        self.nn_model = HybridScorer(
            seq_in_dim=len(SEQ_FEATURES),
            context_in_dim=len(CONTEXT_FEATURES),
            signal_in_dim=len(SIGNAL_FEATURES),
            num_symbols=num_symbols,
            window_size=window_size
        )
        self.nn_model.load_state_dict(torch.load(nn_path, map_location='cpu'))
        self.nn_model.eval()
        print("✓ Ensemble Loaded Successfully")

    def predict_ensemble_confidence(self, symbol: str, timestamp: datetime, df_by_symbol: Dict[str, pd.DataFrame], sig_idx: int = None) -> float:
        """Calculate ensemble confidence for a single signal using pre-scaled features"""
        df_s = df_by_symbol.get(symbol)
        if df_s is None: return 0.5
        
        # Find index for timestamp - optimize by using sig_idx if provided
        if sig_idx is not None:
            idx = sig_idx
        else:
            idx_series = df_s.index[df_s['timestamp'] == timestamp]
            if idx_series.empty: return 0.5
            idx = idx_series[0]
        
        row = df_s.iloc[idx]
        window_size = self.meta['window_size']
        
        # 1. Prepare Tensors (Using pre-scaled columns)
        SEQ_SCALED_COLS = [f'seq_{c}' for c in SEQ_FEATURES]
        if idx < window_size - 1:
            seq_raw = df_s.iloc[0 : idx + 1][SEQ_SCALED_COLS].values.astype(float)
            pad_len = window_size - len(seq_raw)
            seq_scaled = np.zeros((window_size, len(SEQ_FEATURES)))
            seq_scaled[pad_len:] = seq_raw # Already scaled, but fillna handled globally
            seq_scaled = seq_scaled.reshape(1, window_size, -1)
        else:
            seq_scaled = df_s.iloc[idx - window_size + 1 : idx + 1][SEQ_SCALED_COLS].values.astype(float).reshape(1, window_size, -1)
        
        # Tabular data (Already scaled)
        ctx = row[[f'ctx_{c}' for c in CONTEXT_FEATURES]].values.astype(float).reshape(1, -1)
        sig = row[[f'sig_{c}' for c in SIGNAL_FEATURES]].values.astype(float).reshape(1, -1)
        
        # Symbol
        try:
            sym_idx = self.meta['sym_encoder'].transform([symbol])[0]
        except:
            sym_idx = 0 # Fallback
            
        t_seq = torch.from_numpy(seq_scaled).float()
        t_ctx = torch.from_numpy(ctx).float()
        t_sig = torch.from_numpy(sig).float()
        t_sym = torch.tensor([sym_idx]).long()
        
        # 2. NN Prediction
        with torch.no_grad():
            nn_logit = self.nn_model(t_seq, t_ctx, t_sig, t_sym)
            nn_prob = torch.sigmoid(nn_logit).item()
            
            # 3. LightGBM Prediction
            # Re-extract embedding for LGBM
            emb = self.nn_model.get_embeddings(t_seq, t_ctx, t_sig, t_sym)
            X_lgbm = np.hstack([emb, ctx, sig])
            lgbm_prob = self.lgbm_model.predict_proba(X_lgbm)[0, 1]
            
        # 4. Soft Ensemble (Averaging) - Proven superior
        return (nn_prob + lgbm_prob) / 2.0

def calculate_rsi(prices, period=14):
    d = prices.diff(); g = d.where(d>0,0).rolling(period).mean(); l = (-d.where(d<0,0)).rolling(period).mean()
    return 100-(100/(1+g/(l.replace(0,np.nan)+1e-9)))

def calculate_hybrid_features(df, btc_context):
    """Implement the exact feature engineering from the notebook"""
    df = df.copy()
    
    # Basic Features
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    df['body_size'] = abs(df['close'] - df['open']) / df['close']
    
    for p in [7, 14, 21, 50, 100, 200]:
        df[f'ema_{p}'] = df['close'].ewm(span=p).mean()
    for p in [10, 20, 30, 50, 200]:
        df[f'sma_{p}'] = df['close'].rolling(p).mean()
        
    tr = pd.concat([df['high'] - df['low'], abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['volatility_14'] = df['log_returns'].rolling(14).std()
    
    df['vol_sma_14'] = df['volatility_14'].rolling(14).mean()
    df['vol_compression'] = df['volatility_14'] / (df['vol_sma_14'] + 1e-9)
    
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_std_20'] = df['volume'].rolling(20).std()
    df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-9)
    df['volume_zscore'] = (df['volume'] - df['volume_sma_20']) / (df['volume_std_20'] + 1e-9)
    df['volume_trend'] = df['volume'].rolling(7).mean() / (df['volume'].rolling(21).mean() + 1e-9)
    df['volume_spike'] = (df['volume_ratio'] > 2).astype(int)
    
    df['rsi_14'] = calculate_rsi(df['close'], 14)
    df['rsi_slope'] = df['rsi_14'].diff(3)
    
    l14 = df['low'].rolling(14).min()
    h14 = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * (df['close'] - l14) / (h14 - l14).replace(0, np.nan)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    
    df['roc_7'] = df['close'].pct_change(7)
    df['roc_14'] = df['close'].pct_change(14)
    
    df['price_vs_sma_30'] = df['close'] / (df['sma_30'] + 1e-9)
    df['momentum_30'] = df['close'].pct_change(30)
    
    # ADX
    pdm = df['high'].diff(); mdm = -df['low'].diff()
    pdm = pdm.where((pdm > mdm) & (pdm > 0), 0)
    mdm = mdm.where((mdm > pdm) & (mdm > 0), 0)
    atr_s = tr.rolling(14).mean()
    pdi = 100 * (pdm.rolling(14).mean() / atr_s.replace(0, np.nan))
    mdi = 100 * (mdm.rolling(14).mean() / atr_s.replace(0, np.nan))
    df['adx'] = (100 * abs(pdi - mdi) / (pdi + mdi).replace(0, np.nan)).rolling(14).mean()
    
    df['dist_to_high_30d'] = (df['close'] - df['high'].rolling(30).max()) / df['close']
    df['dist_to_low_30d'] = (df['close'] - df['low'].rolling(30).min()) / df['close']
    
    for e in [21, 50, 200]:
        df[f'dist_to_ema_{e}_pct'] = (df['close'] - df[f'ema_{e}']) / df['close']
        
    df['trend_state'] = np.where(df['close'] > df['sma_50'], 1, np.where(df['close'] < df['sma_50'], -1, 0))
    df['is_trending'] = (df['adx'] > 25).astype(int)
    df['is_volatile'] = (df['vol_compression'] > 1.5).astype(int)
    
    df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.dayofweek / 7)
    
    # MACD
    ef = df['close'].ewm(span=12).mean(); es = df['close'].ewm(span=26).mean()
    df['macd'] = ef - es
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_slope'] = df['macd'].diff()
    df['macd_acceleration'] = df['macd_slope'].diff()
    df['macd_cross_up'] = ((df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
    df['macd_cross_down'] = ((df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
    df['is_bullish_cross'] = df['macd_cross_up']
    
    df['vol_ratio_alpha'] = df['volume_ratio'] * df['volatility_14']
    
    # 1D features (Multi-timeframe)
    # Since this is already 1D backtest, we can just use 1D features directly or resample if needed.
    # The notebook resampled 1H to 1D and added dist_to_ema_200_1d_dist, rsi_14_1d.
    # For 1D backtest, these are basically dist_to_ema_200_pct and rsi_14.
    df['ema_200_1d_dist'] = df['dist_to_ema_200_pct']
    df['rsi_14_1d'] = df['rsi_14']
    
    # Merge BTC
    if btc_context is not None and not btc_context.empty:
        df = df.merge(btc_context, on='timestamp', how='left')
        for c in ['btc_is_bull_regime', 'btc_trend_strength', 'btc_returns']:
            df[c] = df[c].ffill().fillna(0)
        df['rs_vs_btc'] = df['log_returns'] - df['btc_returns']
        df['rs_vs_btc_sma7'] = df['rs_vs_btc'].rolling(7).mean()
        df['btc_corr'] = df['log_returns'].rolling(14).corr(df['btc_returns']).fillna(0)
        
    return df.dropna(subset=['macd'])

def run_chron_backtest():
    # 1. SETUP
    MODEL_PATH = Path('ml/models/1d/hybrid')
    DATA_PATH = Path('bitget-data/ohlcv')
    TIMEFRAME = '1d'
    INITIAL_CAPITAL = 10000
    MAX_POSITIONS = 5
    ENTRY_THRESHOLD = 0.70 # Back to 0.70 for ensemble
    LEVERAGE = 2
    
    config = BacktestConfig(
        initial_capital=INITIAL_CAPITAL,
        entry_threshold=ENTRY_THRESHOLD,
        leverage=LEVERAGE,
        max_open_trades=MAX_POSITIONS,
        max_bars=15,
        fee_rate=0.0006,
        slippage=0.0005,
        fixed_position_size=False,
        risk_per_trade=0.01,
        time_stop_bars=7,
        trailing_sl_activation_pct=0.5
    )
    
    backtester = HybridEnsembleBacktester(MODEL_PATH, config)
    
    # 2. LOAD DATA
    print("Loading data...")
    symbols = [f.stem.replace('_USDT','') for f in DATA_PATH.glob('*.parquet')]
    symbols = [s for s in symbols if not any(x in s for x in ['-26','-25','-24'])]
    print(f"Found {len(symbols)} symbols")
    
    # Build BTC Context
    btc_df = pd.read_parquet(DATA_PATH / 'BTCUSDT_USDT.parquet')
    btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'], unit='ms')
    btc_df = btc_df.set_index('timestamp').resample('1D').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna().reset_index()
    
    btc_df['log_returns'] = np.log(btc_df['close'] / btc_df['close'].shift(1))
    btc_df['sma_200'] = btc_df['close'].rolling(200).mean()
    
    # ADX for BTC
    tr = pd.concat([btc_df['high'] - btc_df['low'], abs(btc_df['high'] - btc_df['close'].shift(1)), abs(btc_df['low'] - btc_df['close'].shift(1))], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    pdm = btc_df['high'].diff(); mdm = -btc_df['low'].diff()
    pdm = pdm.where((pdm > mdm) & (pdm > 0), 0); mdm = mdm.where((mdm > pdm) & (mdm > 0), 0)
    pdi = 100 * (pdm.rolling(14).mean() / atr.replace(0, np.nan)); mdi = 100 * (mdm.rolling(14).mean() / atr.replace(0, np.nan))
    btc_df['btc_adx'] = (100 * abs(pdi - mdi) / (pdi + mdi).replace(0, np.nan)).rolling(14).mean()
    
    btc_context = btc_df[['timestamp','close','sma_200','btc_adx','log_returns']].copy()
    btc_context.columns = ['timestamp','btc_close','btc_sma_200','btc_adx','btc_returns']
    btc_context['btc_is_bull_regime'] = (btc_context['btc_close'] > btc_context['btc_sma_200']).astype(int)
    btc_context['btc_trend_strength'] = (btc_context['btc_adx'] > 25).astype(int)
    
    df_by_symbol = {}
    all_signals = []
    
    # 3. APPLY GLOBAL SCALING PRE-LOOP
    print("Applying Global Scalers from Meta...")
    meta = backtester.meta
    
    for sym in symbols:
        try:
            df = pd.read_parquet(DATA_PATH / f"{sym}_USDT.parquet")
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp').resample('1D').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna().reset_index()
            if len(df) < 250: continue
            
            df = calculate_hybrid_features(df, btc_context)
            df['symbol'] = sym
            
            # Scale features once for the whole symbol
            ctx_scaled = meta['ctx_scaler'].transform(df[CONTEXT_FEATURES].fillna(0).values)
            sig_scaled = meta['sig_scaler'].transform(df[SIGNAL_FEATURES].fillna(0).values)
            seq_scaled = meta['seq_scaler'].transform(df[SEQ_FEATURES].fillna(0).values)
            
            for i, col in enumerate(CONTEXT_FEATURES): df[f'ctx_{col}'] = ctx_scaled[:, i]
            for i, col in enumerate(SIGNAL_FEATURES): df[f'sig_{col}'] = sig_scaled[:, i]
            for i, col in enumerate(SEQ_FEATURES): df[f'seq_{col}'] = seq_scaled[:, i]
            
            df_by_symbol[sym] = df
            
            # Find signals
            signals = df[(df['macd_cross_up'] == 1) | (df['macd_cross_down'] == 1)]
            for s_idx, sig_row in signals.iterrows():
                all_signals.append({
                    'timestamp': sig_row['timestamp'],
                    'symbol': sym,
                    'is_long': sig_row['macd_cross_up'] == 1,
                    'idx': s_idx # Store index for faster inference
                })
        except Exception as e:
            print(f"Error processing {sym}: {e}")
            
    # 3. GROUP SIGNALS BY TIME
    signals_by_time = {}
    for sig in all_signals:
        ts = sig['timestamp']
        if ts not in signals_by_time: signals_by_time[ts] = []
        signals_by_time[ts].append(sig)
        
    sorted_times = sorted(signals_by_time.keys())
    print(f"Grouped signals into {len(sorted_times)} timestamps.")
    
    # 4. CHRONOLOGICAL WALK
    capital = INITIAL_CAPITAL
    open_positions = [] # List of Trade objects
    result = BacktestResult()
    result.equity_curve = [INITIAL_CAPITAL]
    result.timestamps = [sorted_times[0]]
    
    print("\nStarting chronological backtest...")
    for current_time in sorted_times:
        # A. Close expired or hit positions
        still_open = []
        for trade in open_positions:
            # === BẢO VỆ BOT: Không check SL/TP trong chính ngày vào lệnh ===
            if current_time == trade.entry_time:
                still_open.append(trade)
                continue
                
            df_s = df_by_symbol.get(trade.symbol)
            if df_s is None: continue
            
            # Find row for current_time
            hist_row = df_s[df_s['timestamp'] == current_time]
            if hist_row.empty: 
                still_open.append(trade)
                continue
            
            row = hist_row.iloc[0]
            trade.bars_held += 1
            
            # Multi-ticker simulation logic (Simplified for daily)
            high, low, close = row['high'], row['low'], row['close']
            
            # A1. Trailing SL (Move to BE)
            if not trade.is_sl_at_be:
                profit_dist = abs(close - trade.entry_price)
                total_dist = abs(trade.tp_price - trade.entry_price)
                # If currently profitable and reached activation threshold
                if (trade.direction == 'LONG' and close > trade.entry_price) or (trade.direction == 'SHORT' and close < trade.entry_price):
                    if profit_dist / (total_dist + 1e-9) >= config.trailing_sl_activation_pct:
                        trade.sl_price = trade.entry_price
                        trade.is_sl_at_be = True
                        # print(f"  [{current_time.date()}] {trade.symbol} SL moved to BE")

            # SL Check (Priority if gap)
            if (trade.direction == 'LONG' and low <= trade.sl_price) or (trade.direction == 'SHORT' and high >= trade.sl_price):
                trade.exit_price = trade.sl_price
                trade.exit_reason = 'SL_HIT'
            # TP Check
            elif (trade.direction == 'LONG' and high >= trade.tp_price) or (trade.direction == 'SHORT' and low <= trade.tp_price):
                trade.exit_price = trade.tp_price
                trade.exit_reason = 'TP_HIT'
            # Time-Stop (Sluggish Exit)
            elif trade.bars_held >= config.time_stop_bars:
                # Calculate current unrealized PnL (rough check)
                is_profitable = (trade.direction == 'LONG' and close > trade.entry_price) or \
                                (trade.direction == 'SHORT' and close < trade.entry_price)
                
                # If still negative or held too long
                if not is_profitable:
                    trade.exit_price = close
                    trade.exit_reason = 'TIME_STOP'
                elif trade.bars_held >= config.max_bars:
                    trade.exit_price = close
                    trade.exit_reason = 'TIMEOUT'
                
            if trade.exit_reason:
                trade.exit_time = current_time
                # Calculate PnL
                if trade.direction == 'LONG':
                    pnl_pct = (trade.exit_price - trade.entry_price) / trade.entry_price
                else:
                    pnl_pct = (trade.entry_price - trade.exit_price) / trade.entry_price
                
                trade.pnl = trade.position_size * pnl_pct - (trade.position_size * config.fee_rate * 2)
                capital += trade.pnl
                result.trades.append(trade)
            else:
                still_open.append(trade)
                
        open_positions = still_open
        
        # B. Open new positions if slots available
        available_slots = MAX_POSITIONS - len(open_positions)
        if available_slots > 0:
            signals = signals_by_time[current_time]
            
            # Calculate confidence for all signals at this time
            for sig in signals:
                sig['confidence'] = backtester.predict_ensemble_confidence(sig['symbol'], current_time, df_by_symbol, sig['idx'])
                
            # Filter and Rank
            valid_signals = [s for s in signals if s['confidence'] >= ENTRY_THRESHOLD]
            valid_signals.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Skip symbols already in position
            open_symbols = {p.symbol for p in open_positions}
            
            for sig in valid_signals:
                if available_slots <= 0: break
                if sig['symbol'] in open_symbols: continue
                
                # Predict SL/TP (Static fallback or using models if available)
                # For this model, TP/SL are structural (Swing High/Low) as per notebook
                df_s = df_by_symbol[sig['symbol']]
                idx = df_s.index[df_s['timestamp'] == current_time][0]
                
                # Extract structural SL/TP (same logic as labeling)
                lookback = 30
                lb_start = max(0, idx-lookback)
                # CHỈ lấy đỉnh đáy quá khứ (TRỪ cây nến hiện tại idx)
                past_data = df_s.iloc[lb_start:idx]
                s_h, s_l = past_data['high'].max(), past_data['low'].min()
                
                entry_price = df_s.loc[idx, 'close']
                atr = df_s.loc[idx, 'atr_14']
                
                # Nới rộng Stop Loss (Breathing Room)
                buf = atr * 1.5 
                
                if sig['is_long']:
                    # SL phải dưới s_l ít nhất 1.5 ATR, nhưng tối đa không quá 10%
                    # TP dùng kháng cự cũ (s_h)
                    sl_price = min(s_l - buf, entry_price * 0.95) 
                    sl_price = max(sl_price, entry_price * 0.90) # Cap risk 10%
                    tp_price = max(s_h, entry_price * 1.05) # Ít nhất phải ăn 5%
                else:
                    sl_price = max(s_h + buf, entry_price * 1.05)
                    sl_price = min(sl_price, entry_price * 1.10)
                    tp_price = min(s_l, entry_price * 0.95)

                # ==========================================
                # RISK MANAGEMENT KỶ LUẬT THÉP
                # ==========================================
                # 1. Tính toán khoảng cách rủi ro
                risk_pct = abs(entry_price - sl_price) / (entry_price + 1e-9)
                
                # 2. R:R Filter (Lọc rác)
                target_pct = abs(tp_price - entry_price) / (entry_price + 1e-9)
                if target_pct / (risk_pct + 1e-9) < 1.1:
                    continue
                
                # 3. Quản trị Vốn (Position Sizing)
                # Giảm rủi ro mỗi lệnh xuống mức an toàn (1%)
                ACTUAL_RISK_PER_TRADE = 0.01 
                
                # Tính Size: Cần mua bao nhiêu $ để nếu chạm SL thì chỉ mất 1% Capital?
                target_position_size = capital * ACTUAL_RISK_PER_TRADE / (risk_pct + 1e-9)
                
                # 4. Giới hạn Đòn Bẩy (Margin Constraints)
                # Một lệnh KHÔNG ĐƯỢC dùng quá 1/MAX_POSITIONS vốn ký quỹ (Tránh all-in)
                max_margin_per_trade = capital / config.max_open_trades
                
                # Chuyển Margin thành Size tối đa cho phép
                max_size_per_trade = max_margin_per_trade * config.leverage
                
                # Chốt Size cuối cùng (Lấy số nhỏ nhất để an toàn)
                position_size = min(target_position_size, max_size_per_trade)
                
                if np.isnan(position_size) or position_size <= 0:
                    continue

                new_trade = Trade(
                    symbol=sig['symbol'],
                    entry_time=current_time,
                    direction='LONG' if sig['is_long'] else 'SHORT',
                    entry_price=entry_price,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    position_size=position_size,
                    confidence=sig['confidence']
                )
                open_positions.append(new_trade)
                open_symbols.add(sig['symbol'])
                available_slots -= 1
                
        result.equity_curve.append(capital)
        result.timestamps.append(current_time)
        
        # Periodic Heartbeat
        if len(result.timestamps) % 200 == 0:
            print(f"  [{current_time.date()}] Equity: ${capital:,.2f} | Open: {len(open_positions)} | Trades: {len(result.trades)}")

    # 5. SUMMARY
    print("\n" + "="*40)
    print("BACKTEST RESULTS (HYBRID ENSEMBLE)")
    print("="*40)
    
    # Remove any nan pnl trades
    result.trades = [t for t in result.trades if not np.isnan(t.pnl)]
    print(f"Total Trades: {len(result.trades)}")
    if len(result.trades) > 0:
        win_rate = len([t for t in result.trades if t.pnl > 0]) / len(result.trades)
        total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
        # Exit reasons breakdown
        reasons = [t.exit_reason for t in result.trades]
        from collections import Counter
        reason_counts = Counter(reasons)
        print("\nExit Reasons:")
        for reason, count in reason_counts.items():
            print(f"  {reason}: {count} ({count/len(reasons):.1%})")
            
        # Plot
        plt.figure(figsize=(12, 10))
        plt.subplot(2, 1, 1)
        plt.plot(result.timestamps, result.equity_curve)
        plt.title(f"Hybrid Ensemble Chron Backtest (1D)\nReturn: {total_return:.1%}, WinRate: {win_rate:.1%}")
        plt.xlabel("Date")
        plt.ylabel("Equity ($)")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        # Plot distribution of exit reasons as bars
        plt.bar(reason_counts.keys(), reason_counts.values())
        plt.title("Exit Reasons Distribution")
        plt.ylabel("Count")
        
        plt.tight_layout()
        plt.savefig("backtest_hybrid_ensemble_1d.png")
        print("\nChart saved to backtest_hybrid_ensemble_1d.png")
    else:
        print("No trades executed.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test-load":
        # Test loading logic
        try:
            MODEL_PATH = Path('ml/models/1d/hybrid')
            backtester = HybridEnsembleBacktester(MODEL_PATH)
            print("Model load test PASSED")
        except Exception as e:
            print(f"Model load test FAILED: {e}")
            sys.exit(1)
    else:
        run_chron_backtest()
