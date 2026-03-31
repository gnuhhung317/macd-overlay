import os
import gc
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
from bisect import bisect_left, bisect_right

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from sniper_bot.feature import calculate_features
warnings.filterwarnings('ignore')

# ============================================================
# DATA STRUCTURES & ENUMS
# ============================================================
class TradeState(Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    CLOSED = "CLOSED"

@dataclass
class BacktestConfig:
    initial_capital: float = 100.0
    risk_per_trade: float = 0.05
    fee_rate: float = 0.001
    slippage: float = 0.01
    max_open_trades: int = 5
    max_bars_hold: int = 48
    start_date: str = '2025-01-01'
    end_date: str = None
    leverage: float = 1.0
    exchange: str = 'binance'
    long_atr_offset: float = 0.0
    short_atr_offset: float = 0.0
    limit_wait_bars: int = 2
    tp_mult_long: float = 3.0
    sl_mult_long: float = 1.5
    tp_mult_short: float = 2.0
    sl_mult_short: float = 1.5
    threshold: Optional[float] = None

@dataclass
class Trade:
    symbol: str
    signal_time: datetime
    type: str  # LONG/SHORT
    limit_price: float
    tp_price: float
    sl_price: float
    atr_val: float
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    state: TradeState = TradeState.PENDING
    result: str = 'PENDING'
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    duration: int = 0
    wait_bars: int = 0
    pos_size_usd: float = 0.0
    mfe_atr: float = 0.0
    mae_atr: float = 0.0

# ============================================================
# ASSETS & FEATURES
# ============================================================
BASE_DIR = Path(r"d:\Code\Projects\self-projects\macd-overlay - Copy")
SYMBOLS_DIR = BASE_DIR / "data" / "processed" / "symbols_v3"
MODEL_PATH = BASE_DIR / "legacy" / "ml" / "models" / "honest" / "model_reversal.joblib"
META_PATH = BASE_DIR / "legacy" / "ml" / "models" / "honest" / "ensemble_meta.joblib"

def load_assets(override_threshold: Optional[float] = None):
    if not META_PATH.exists() or not MODEL_PATH.exists():
        print("❌ Missing model or meta file!")
        return None, [], override_threshold if override_threshold is not None else 0.7
    meta = joblib.load(META_PATH)
    clf = joblib.load(MODEL_PATH)
    features = meta.get('features', []) if isinstance(meta, dict) else meta
    threshold = override_threshold if override_threshold is not None else meta.get('threshold', 0.7)
    return clf, features, threshold


# ============================================================
# STATE MACHINE LOGIC
# ============================================================
def simulate_trade_step(trade: Trade, row: pd.Series, config: BacktestConfig):
    if trade.state == TradeState.CLOSED:
        return False, trade

    # 1. Entry Logic (PENDING -> ACTIVE)
    if trade.state == TradeState.PENDING:
        trade.wait_bars += 1
        is_filled = False
        if trade.type == 'LONG':
            if row['open'] <= trade.limit_price:
                trade.entry_price = row['open']; is_filled = True
            elif row['low'] <= trade.limit_price:
                trade.entry_price = trade.limit_price; is_filled = True
        else: # SHORT
            if row['open'] >= trade.limit_price:
                trade.entry_price = row['open']; is_filled = True
            elif row['high'] >= trade.limit_price:
                trade.entry_price = trade.limit_price; is_filled = True
        
        if is_filled:
            trade.state = TradeState.ACTIVE
            trade.entry_time = row.name
        elif trade.wait_bars >= config.limit_wait_bars:
            trade.state = TradeState.CLOSED; trade.result = 'MISSED'
            return True, trade

    # 2. Active Logic (including the bar it was just filled in)
    if trade.state == TradeState.ACTIVE:
        # A. Update MFE/MAE (Normalize by signal ATR)
        if trade.type == 'LONG':
            trade.mfe_atr = max(trade.mfe_atr, (row['high'] - trade.entry_price) / (trade.atr_val + 1e-9))
            trade.mae_atr = min(trade.mae_atr, (row['low'] - trade.entry_price) / (trade.atr_val + 1e-9))
        else: # SHORT
            trade.mfe_atr = max(trade.mfe_atr, (trade.entry_price - row['low']) / (trade.atr_val + 1e-9))
            trade.mae_atr = min(trade.mae_atr, (trade.entry_price - row['high']) / (trade.atr_val + 1e-9))

        trade.duration += 1
        
        # B. Exit Logic
        closed_this_bar = False
        if trade.type == 'LONG':
            if row['open'] <= trade.sl_price:
                trade.state = TradeState.CLOSED; trade.result = 'LOSS'
                trade.exit_price = row['open'] * (1 - config.slippage)
                closed_this_bar = True
            elif row['low'] <= trade.sl_price:
                trade.state = TradeState.CLOSED; trade.result = 'LOSS'
                trade.exit_price = trade.sl_price * (1 - config.slippage)
                closed_this_bar = True
            elif row['high'] >= trade.tp_price:
                trade.state = TradeState.CLOSED; trade.result = 'WIN'
                trade.exit_price = trade.tp_price * (1 - config.slippage)
                closed_this_bar = True
        else: # SHORT
            if row['open'] >= trade.sl_price:
                trade.state = TradeState.CLOSED; trade.result = 'LOSS'
                trade.exit_price = row['open'] * (1 + config.slippage)
                closed_this_bar = True
            elif row['high'] >= trade.sl_price:
                trade.state = TradeState.CLOSED; trade.result = 'LOSS'
                trade.exit_price = trade.sl_price * (1 + config.slippage)
                closed_this_bar = True
            elif row['low'] <= trade.tp_price:
                trade.state = TradeState.CLOSED; trade.result = 'WIN'
                trade.exit_price = trade.tp_price * (1 + config.slippage)
                closed_this_bar = True

        if closed_this_bar:
            trade.exit_time = row.name
            return True, trade

        # C. Timeout Check
        if trade.duration >= config.max_bars_hold:
            trade.state = TradeState.CLOSED; trade.result = 'TIMEOUT'
            trade.exit_price = row['close']
            trade.exit_time = row.name
            return True, trade

    return trade.state == TradeState.CLOSED, trade
    return False, trade

# ============================================================
# MAIN BACKTEST ENGINE
# ============================================================
def backtest_symbol(file_path, features, clf, threshold, config: BacktestConfig):
    try:
        df = pd.read_parquet(file_path)
        if df.empty: return None, None
        symbol = Path(file_path).stem.replace('_USDT', '').replace('USDT', '')
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        start_ts = pd.to_datetime(config.start_date) if config.start_date else df['timestamp'].min()
        end_ts = pd.to_datetime(config.end_date) if config.end_date else df['timestamp'].max()

        # Optimize: Crop to window + padding for indicators
        if config.start_date:
            padding_bars = 1000
            # Find index of start_date and take 1000 bars before it
            start_idx = df[df['timestamp'] >= start_ts].index
            if len(start_idx) > 0:
                crop_start = max(0, start_idx[0] - padding_bars)
                # Also crop the end to save memory
                end_idx = df[df['timestamp'] <= end_ts].index
                crop_end = end_idx[-1] + padding_bars if len(end_idx) > 0 else len(df)
                df = df.iloc[crop_start:crop_end].reset_index(drop=True)

        # Use centralized feature calculation for perfect parity
        df = calculate_features(df)
        
        scan_indices = df.index
        if config.start_date:
            scan_indices = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)].index
        
        vol_sma = df['volume'].rolling(20).mean().shift(1)
        
        # New Reversal-focused Ignition Mask
        # Matches trainer: green candle, body > 1%, volume > 1.2x SMA, and Reversal Regime
        c1 = df['close'] > df['open']
        c2 = ((df['close'] - df['open']) / df['open']) > 0.010
        c3 = df['volume'] > (vol_sma * 1.2)
        
        # Reversal Regime: Price is significantly below Daily EMA 200
        # If ema_200_1d_dist is missing, we allow it (or could calculate it)
        if 'ema_200_1d_dist' in df.columns:
            c4 = df['ema_200_1d_dist'] < -0.15
            ignition_mask = (c1 & c2 & c3 & c4).reindex(scan_indices, fill_value=False)
        else:
            ignition_mask = (c1 & c2 & c3).reindex(scan_indices, fill_value=False)
            
        final_scan_indices = scan_indices[ignition_mask]
        if len(final_scan_indices) == 0: return None, df
        
        X_batch = df.loc[final_scan_indices, features].apply(pd.to_numeric, errors='coerce').fillna(0)
        probas = clf.predict_proba(X_batch)
        
        potential_signals = []
        for i, idx in enumerate(final_scan_indices):
            # The new binary model has 2 classes [0, 1]. probas[:, 1] is the probability of a LONG win.
            pl = probas[i, 1]
            ps = 0.0 # Reversal model is LONG-only
            if pl > threshold:
                potential_signals.append({
                    'timestamp': df.iloc[idx]['timestamp'], 'symbol': symbol,
                    'type': 'LONG',
                    'prob_long': pl, 'prob_short': ps,
                    'atr_val': df.iloc[idx]['atr_14'], 'close': df.iloc[idx]['close']
                })
        return potential_signals, df
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}"); return None, None

def run_portfolio_simulation(all_signals, full_price_db, config: BacktestConfig):
    if not all_signals: return [], [], 0
    print(f"Processing {len(all_signals)} potential signals against global timeline...")
    
    unique_ts = sorted(set().union(*(df['timestamp'] for df in full_price_db.values())))
    signals_by_time = {}
    for sig in all_signals:
        ts = sig['timestamp']
        if ts not in signals_by_time: signals_by_time[ts] = []
        signals_by_time[ts].append(sig)
    
    price_lookups = {sym: df.set_index('timestamp') for sym, df in full_price_db.items()}
    realized_capital = config.initial_capital; available_capital = config.initial_capital
    active_trades: List[Trade] = []; pending_trades: List[Trade] = []; closed_trades: List[Trade] = []
    equity_curve = []
    
    unique_ts_list = sorted(unique_ts)
    signal_times = sorted(signals_by_time.keys())
    
    ts_idx = 0
    while ts_idx < len(unique_ts_list):
        ts = unique_ts_list[ts_idx]
        
        # A. Process Pending & Active
        still_processing = pending_trades + active_trades
        pending_trades = []; active_trades = []
        
        for t in still_processing:
            if t.symbol in price_lookups and ts in price_lookups[t.symbol].index:
                row = price_lookups[t.symbol].loc[ts]
                _, updated_t = simulate_trade_step(t, row, config)
                
                if updated_t.state == TradeState.CLOSED:
                    if updated_t.result != 'MISSED':
                        fee_entry = updated_t.pos_size_usd * config.fee_rate
                        fee_exit = (updated_t.pos_size_usd * (updated_t.exit_price / updated_t.entry_price)) * config.fee_rate
                        updated_t.fees = fee_entry + fee_exit
                        raw_pnl = (updated_t.exit_price - updated_t.entry_price)/updated_t.entry_price if updated_t.type == 'LONG' else (updated_t.entry_price - updated_t.exit_price)/updated_t.entry_price
                        updated_t.pnl_usd = (updated_t.pos_size_usd * raw_pnl) - updated_t.fees
                        updated_t.pnl_pct = (updated_t.pnl_usd / updated_t.pos_size_usd) * 100
                        realized_capital += updated_t.pnl_usd
                        available_capital += (updated_t.pos_size_usd / config.leverage) + updated_t.pnl_usd
                    else:
                        available_capital += (updated_t.pos_size_usd / config.leverage)
                    closed_trades.append(updated_t)
                elif updated_t.state == TradeState.ACTIVE: active_trades.append(updated_t)
                else: pending_trades.append(updated_t)

        # B. MTM & Equity
        floating_pnl = 0
        for t in active_trades:
            if t.symbol in price_lookups and ts in price_lookups[t.symbol].index:
                curr_p = price_lookups[t.symbol].loc[ts, 'close']
                raw_pnl = (curr_p - t.entry_price)/t.entry_price if t.type == 'LONG' else (t.entry_price - curr_p)/t.entry_price
                floating_pnl += t.pos_size_usd * raw_pnl
        equity_curve.append((ts, realized_capital + floating_pnl))

        # C. New Signals
        if ts in signals_by_time:
            sorted_sigs = sorted(signals_by_time[ts], key=lambda x: max(x['prob_long'], x['prob_short']), reverse=True)
            for sig in sorted_sigs:
                if len(active_trades) + len(pending_trades) < config.max_open_trades:
                    risk_amount = (realized_capital + floating_pnl) * config.risk_per_trade
                    l_p = sig['close'] + (config.long_atr_offset * sig['atr_val']) if sig['type'] == 'LONG' else sig['close'] + (config.short_atr_offset * sig['atr_val'])
                    sl_p = l_p - (config.sl_mult_long * sig['atr_val']) if sig['type'] == 'LONG' else l_p + (config.sl_mult_short * sig['atr_val'])
                    tp_p = l_p + (config.tp_mult_long * sig['atr_val']) if sig['type'] == 'LONG' else l_p - (config.tp_mult_short * sig['atr_val'])
                    
                    sl_dist_pct = abs(sl_p - l_p) / l_p
                    pos_size_usd = min(risk_amount / max(sl_dist_pct, 0.003), available_capital * config.leverage * 0.95)
                    pos_size_usd =  min(pos_size_usd, 10000)
                    if pos_size_usd >= 10:
                        pending_trades.append(Trade(sig['symbol'], ts, sig['type'], l_p, tp_p, sl_p, sig['atr_val'], pos_size_usd=pos_size_usd))
                        available_capital -= (pos_size_usd / config.leverage)

        # D. Optimization: Jump Logic
        if not pending_trades and not active_trades:
            next_sig_idx = bisect_right(signal_times, ts)
            if next_sig_idx < len(signal_times):
                next_ts = signal_times[next_sig_idx]
                ts_idx = bisect_left(unique_ts_list, next_ts)
            else:
                break
        else:
            ts_idx += 1

    return closed_trades, equity_curve, 0

def run_backtest_with_config(config: BacktestConfig):
    clf, features, threshold = load_assets(override_threshold=config.threshold)
    if clf is None: return None, None, None, None
    
    if config.exchange == 'bitget':
        symbols_dir = BASE_DIR / "bitget-data" / "symbols_v3"
    else:
        symbols_dir = BASE_DIR / "data" / "processed" / "symbols_v3"
        
    all_files = list(symbols_dir.glob("*.parquet"))
    print(f"Scanning {len(all_files)} symbols...")
    potential_signals = []; full_price_db = {}
    for i, file_path in enumerate(all_files):
        if i % 100 == 0: print(f"Progress: {i}/{len(all_files)}...")
        res, ohlcv = backtest_symbol(file_path, features, clf, threshold, config)
        if res: potential_signals.extend(res); full_price_db[Path(file_path).stem.replace('_USDT', '').replace('USDT', '')] = ohlcv
    
    trades, equity_curve, _ = run_portfolio_simulation(potential_signals, full_price_db, config)
    if not trades: print("No trades executed."); return potential_signals, full_price_db, trades, equity_curve

    report_df = pd.DataFrame([vars(t) for t in trades])
    report_df = report_df[report_df['result'] != 'MISSED'].sort_values('entry_time')
    
    print(f"\n✅ PORTFOLIO RESULTS (State Machine):")
    final_cap = config.initial_capital + report_df['pnl_usd'].sum()
    print(f"Initial: ${config.initial_capital:.2f} | Final: ${final_cap:.2f} | Return: {((final_cap/config.initial_capital)-1)*100:.2f}%")
    print(f"Trades: {len(report_df)}")
    
    equity_series = pd.DataFrame(equity_curve, columns=['time', 'val']).set_index('time')['val']
    daily_equity = equity_series.resample('D').last().ffill()
    daily_returns = daily_equity.pct_change().dropna()
    sharpe = (daily_returns.mean() / (daily_returns.std() + 1e-9)) * np.sqrt(365) if len(daily_returns) > 1 else 0
    roll_max = equity_series.cummax(); max_dd = ((equity_series - roll_max)/roll_max).min() * 100
    
    min_bal = equity_series.min()
    print(f"MaxDrawdown: {max_dd:.2f}% | Sharpe: {sharpe:.2f} | Min Balance: ${min_bal:.2f}")
    
    output_file = BASE_DIR / "ml" / "backtest_results_quant_sniper.csv"
    report_df.to_csv(output_file, index=False); print(f"Report saved: {output_file}")
    return potential_signals, full_price_db, trades, equity_curve

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=str, default='2025-01-01')
    parser.add_argument('--end', type=str, default='2026-03-01')
    parser.add_argument('--leverage', type=float, default=1.0)
    parser.add_argument('--exchange', type=str, default='binance')
    parser.add_argument('--capital', type=float, default=100.0)
    parser.add_argument('--risk', type=float, default=0.05)
    parser.add_argument('--max-positions', type=int, default=5)
    parser.add_argument('--max-bars-hold', type=int, default=48)
    parser.add_argument('--threshold', type=float, default=None)
    args = parser.parse_args()
    config = BacktestConfig(
        start_date=args.start,
        end_date=args.end,
        leverage=args.leverage,
        exchange=args.exchange,
        initial_capital=args.capital,
        risk_per_trade=args.risk,
        max_open_trades=args.max_positions,
        max_bars_hold=args.max_bars_hold,
        threshold=args.threshold
    )
    run_backtest_with_config(config)
