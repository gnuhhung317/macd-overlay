import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
from typing import List, Dict, Tuple
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# CONFIG & DATA STRUCTURES
# ============================================================
BASE_DIR = Path(r"d:\Code\Projects\self-projects\macd-overlay - Copy")
SYMBOLS_DIR = BASE_DIR / "data" / "processed" / "symbols_v3"
MODEL_PATH = BASE_DIR / "ml" / "training" / "models" / "1h" / "ensemble_lgbm_tabular.joblib"
META_PATH = BASE_DIR / "ml" / "training" / "models" / "1h" / "ensemble_meta.joblib"

@dataclass
class BacktestConfig:
    initial_capital: float = 1000.0
    risk_per_trade: float = 0.02
    fee_rate: float = 0.001
    slippage: float = 0.0005
    max_open_trades: int = 10
    max_bars_hold: int = 48
    start_date: str = '2025-08-01'
    limit_wait_bars: int = 5

@dataclass
class Trade:
    symbol: str
    entry_time: datetime
    exit_time: datetime
    type: str
    entry_price: float
    exit_price: float
    result: str
    pnl_usd: float
    pnl_pct: float
    fees: float
    duration: int
    mfe_atr: float
    mae_atr: float
    pos_size_usd: float

def load_assets():
    meta = joblib.load(META_PATH)
    clf = joblib.load(MODEL_PATH)
    features = meta.get('features', []) if isinstance(meta, dict) else meta
    threshold = meta.get('threshold', 0.6)
    return clf, features, threshold

def calculate_features_gs(df):
    df = df.copy()
    if 'ema_20' not in df.columns:
        df['ema_20'] = df['close'].ewm(span=20).mean()
    if 'ema_50' not in df.columns:
        df['ema_50'] = df['close'].ewm(span=50).mean()
    if 'vol_ratio' not in df.columns:
        vol_sma = pd.Series(df['volume']).rolling(20).mean().shift(1)
        df['vol_ratio'] = df['volume'] / (vol_sma + 1e-9)
    if 'atr_14' not in df.columns:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()
    if 'atr_pct' not in df.columns:
        df['atr_pct'] = (df['atr_14'] / df['close']) * 100
    if 'upper_wick_ratio' not in df.columns:
        df['upper_wick_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-9)
    if 'dist_to_ema50_atr' not in df.columns:
        df['dist_to_ema50_atr'] = (df['close'] - df['ema_50']) / (df['atr_14'] + 1e-9)
    if 'vol_acceleration' not in df.columns:
        df['vol_acceleration'] = df['volume'] / (df['volume'].shift(1) + 1e-9)
    return df

def simulate_trade_gs(df, entry_idx, trade_type, long_offset, short_offset, config: BacktestConfig):
    horizon = config.max_bars_hold
    full_len = len(df)
    
    ident_row = df.iloc[entry_idx]
    close_at_ident = ident_row['close']
    atr_val = ident_row['atr_14']
    
    if trade_type == 'LONG':
        limit_price = close_at_ident + (long_offset * atr_val)
        tp_price = limit_price + (2.0 * atr_val)
        sl_price = limit_price - (1.0 * atr_val)
    else:
        limit_price = close_at_ident + (short_offset * atr_val)
        tp_price = limit_price - (2.5 * atr_val)
        sl_price = limit_price + (1.5 * atr_val)

    is_filled = False
    fill_idx = -1
    for i in range(1, config.limit_wait_bars + 1):
        idx = entry_idx + i
        if idx >= full_len: break
        row = df.iloc[idx]
        if trade_type == 'LONG':
            if row['low'] < limit_price:
                is_filled = True; fill_idx = idx; break
        else:
            if row['high'] > limit_price:
                is_filled = True; fill_idx = idx; break
    
    if not is_filled: return 'MISSED', None, 0, 0, 0, 0, 0
    
    entry_price = limit_price
    max_mfe_atr, min_mae_atr = 0, 0
    result = 'TIMEOUT'
    start_sim_idx = fill_idx
    remaining_horizon = max(1, horizon - (fill_idx - entry_idx))
    
    exit_price = df.iloc[min(start_sim_idx + remaining_horizon, full_len-1)]['close']
    exit_time = df.iloc[min(start_sim_idx + remaining_horizon, full_len-1)]['timestamp']
    duration = remaining_horizon

    for i in range(0, remaining_horizon + 1):
        idx = start_sim_idx + i
        if idx >= full_len: break
        row = df.iloc[idx]
        if trade_type == 'LONG':
            if row['low'] <= sl_price:
                result = 'LOSS'; duration = i + (fill_idx-entry_idx); exit_price = sl_price * (1-config.slippage); exit_time = row['timestamp']; break
            elif row['high'] >= tp_price:
                result = 'WIN'; duration = i + (fill_idx-entry_idx); exit_price = tp_price * (1-config.slippage); exit_time = row['timestamp']; break
            current_mfe_atr = (row['high'] - entry_price) / (atr_val + 1e-9)
            current_mae_atr = (row['low'] - entry_price) / (atr_val + 1e-9)
        else:
            if row['high'] >= sl_price:
                result = 'LOSS'; duration = i + (fill_idx-entry_idx); exit_price = sl_price * (1+config.slippage); exit_time = row['timestamp']; break
            elif row['low'] <= tp_price:
                result = 'WIN'; duration = i + (fill_idx-entry_idx); exit_price = tp_price * (1+config.slippage); exit_time = row['timestamp']; break
            current_mfe_atr = (entry_price - row['low']) / (atr_val + 1e-9)
            current_mae_atr = (entry_price - row['high']) / (atr_val + 1e-9)
        max_mfe_atr = max(max_mfe_atr, current_mfe_atr)
        min_mae_atr = min(min_mae_atr, current_mae_atr)

    return result, exit_time, duration, max_mfe_atr, min_mae_atr, entry_price, exit_price

def run_portfolio_gs(signals, price_db, config: BacktestConfig):
    if not signals: return 0, -100
    signals = sorted(signals, key=lambda x: x['timestamp'])
    realized_capital = config.initial_capital
    available_capital = config.initial_capital
    active_trades: List[Trade] = []
    equity_curve = [realized_capital]
    
    for s in signals:
        curr_time = s['timestamp']
        active_trades.sort(key=lambda x: x.exit_time)
        while active_trades and active_trades[0].exit_time <= curr_time:
            t = active_trades.pop(0)
            realized_capital += t.pnl_usd
            available_capital += t.pos_size_usd + t.pnl_usd
            equity_curve.append(realized_capital)
            
        floating_pnl = 0
        for t in active_trades:
            if t.symbol in price_db and curr_time in price_db[t.symbol].index:
                curr_p = price_db[t.symbol].loc[curr_time]
                pnl_pct = (curr_p - t.entry_price)/t.entry_price if t.type == 'LONG' else (t.entry_price - curr_p)/t.entry_price
                floating_pnl += t.pos_size_usd * pnl_pct
        mtm_equity = realized_capital + floating_pnl

        if len(active_trades) < config.max_open_trades:
            risk_amount = mtm_equity * config.risk_per_trade
            sl_dist = 1.0 * s['atr_val'] if s['type'] == 'LONG' else 1.5 * s['atr_val']
            pos_size_usd = (risk_amount * s['entry_p_raw']) / (sl_dist + 1e-9)
            max_pos = min(available_capital * 0.95, mtm_equity * 0.95)
            pos_size_usd = min(pos_size_usd, max_pos)
            if pos_size_usd < 10: continue

            fee_entry = pos_size_usd * config.fee_rate
            fee_exit = (pos_size_usd * (s['exit_p_raw'] / s['entry_p_raw'])) * config.fee_rate
            raw_pnl_pct = (s['exit_p_raw'] - s['entry_p_raw'])/s['entry_p_raw'] if s['type'] == 'LONG' else (s['entry_p_raw'] - s['exit_p_raw'])/s['entry_p_raw']
            net_pnl_usd = (pos_size_usd * raw_pnl_pct) - (fee_entry + fee_exit)
            
            trade_obj = Trade(s['symbol'], s['timestamp'], s['exit_time'], s['type'], s['entry_p_raw'], s['exit_p_raw'], s['result'], net_pnl_usd, 0, 0, s['duration'], 0, 0, pos_size_usd)
            available_capital -= pos_size_usd
            active_trades.append(trade_obj)
            
    for t in active_trades: realized_capital += t.pnl_usd; equity_curve.append(realized_capital)
    
    total_return = ((realized_capital / config.initial_capital) - 1) * 100
    eq_ser = pd.Series(equity_curve)
    drawdown = (eq_ser - eq_ser.cummax()) / eq_ser.cummax()
    max_dd = drawdown.min() * 100
    return total_return, max_dd

def main():
    config = BacktestConfig()
    clf, features, threshold = load_assets()
    all_files = list(SYMBOLS_DIR.glob("*.parquet"))
    
    print("🔍 Pre-detecting signals...")
    detected_signals = [] # List of tuples (symbol, df, scan_indices, threshold_scores)
    price_db = {}
    
    for file_path in tqdm(all_files):
        try:
            df = pd.read_parquet(file_path)
            if df.empty: continue
            symbol = Path(file_path).stem.replace('_USDT', '').replace('USDT', '')
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
            df = df.sort_values('timestamp').reset_index(drop=True)
            df = calculate_features_gs(df)
            start_ts = pd.to_datetime(config.start_date)
            scan_indices = df[df['timestamp'] >= start_ts].index
            if len(scan_indices) == 0: continue
            
            vol_sma = df['volume'].rolling(20).mean().shift(1)
            c1 = (df['close'] > df['open']) & (df['close'] > df['ema_20'])
            c2 = ((df['close'] - df['open']) / df['open']) > 0.015
            c3 = (df['volume'] > vol_sma * 1.5) & (df['volume'] < vol_sma * 4.0)
            c4 = (df['rsi_14'] >= 55) & (df['rsi_14'] <= 72)
            ignition_mask = (c1 & c2 & c3 & c4).reindex(scan_indices, fill_value=False)
            final_indices = scan_indices[ignition_mask]
            if len(final_indices) == 0: continue
            
            X_batch = df.loc[final_indices, features].apply(pd.to_numeric, errors='coerce').fillna(0)
            probas = clf.predict_proba(X_batch)
            
            detected_signals.append((symbol, df[['timestamp','high','low','close','atr_14']], final_indices, probas))
            price_db[symbol] = df.set_index('timestamp')['close']
        except: continue

    total_signals = sum(len(x[2]) for x in detected_signals)
    print(f"✅ Pre-detection complete. Found {total_signals} base signals across {len(detected_signals)} symbols.")

    long_grid = [-0.3, -0.2, -0.1, 0.0, 0.1]
    short_grid = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = []

    print(f"🚀 Grid Search starting: {len(long_grid) * len(short_grid)} combinations")
    for lo in long_grid:
        for so in short_grid:
            all_sim_signals = []
            for symbol, df, indices, probas in detected_signals:
                prob_long, prob_short = probas[:, 1], probas[:, 2]
                for i, idx in enumerate(indices):
                    pl, ps = prob_long[i], prob_short[i]
                    if pl > threshold or ps > threshold:
                        ttype = 'LONG' if pl > threshold else 'SHORT'
                        res, ext, dur, mfe, mae, en_p, ex_p = simulate_trade_gs(df, idx, ttype, lo, so, config)
                        if res == 'MISSED': continue
                        all_sim_signals.append({
                            'timestamp': df.iloc[idx]['timestamp'], 'symbol': symbol, 'type': ttype,
                            'result': res, 'exit_time': ext, 'duration': dur, 'entry_p_raw': en_p, 'exit_p_raw': ex_p,
                            'atr_val': df.iloc[idx]['atr_14']
                        })
            
            ret, mdd = run_portfolio_gs(all_sim_signals, price_db, config)
            results.append({'long_offset': lo, 'short_offset': so, 'return': ret, 'max_dd': mdd, 'trades': len(all_sim_signals)})
            print(f"LO: {lo:+.1f} | SO: {so:+.1f} | Return: {ret:7.2f}% | DD: {mdd:6.2f}% | Trades: {len(all_sim_signals)}")

    res_df = pd.DataFrame(results).sort_values('return', ascending=False)
    res_df.to_csv(BASE_DIR / "ml" / "grid_search_results.csv", index=False)
    print("\n✅ Grid Search complete. Results saved to ml/grid_search_results.csv")
    print(res_df.head(10))

if __name__ == "__main__":
    main()
