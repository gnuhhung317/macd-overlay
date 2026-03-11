import os
import joblib
import pandas as pd
import numpy as np
import optuna
from pathlib import Path
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
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
    limit_wait_bars: int = 5
    start_date: str = '2025-08-01'

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
    duration: int
    pos_size_usd: float

def load_assets():
    meta = joblib.load(META_PATH)
    clf = joblib.load(MODEL_PATH)
    features = meta.get('features', []) if isinstance(meta, dict) else meta
    threshold = meta.get('threshold', 0.6)
    return clf, features, threshold

def calculate_features_opt(df):
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
    if 'rsi_14' not in df.columns:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['rsi_14'] = 100 - (100 / (1 + rs))
    return df

def simulate_trade_opt(df, entry_idx, trade_type, lo, so, tp_mult, sl_mult, hold_bars, config: BacktestConfig):
    full_len = len(df)
    ident_row = df.iloc[entry_idx]
    close_at_ident = ident_row['close']
    atr_val = ident_row['atr_14']
    
    if trade_type == 'LONG':
        limit_price = close_at_ident + (lo * atr_val)
        tp_price = limit_price + (tp_mult * atr_val)
        sl_price = limit_price - (sl_mult * atr_val)
    else:
        limit_price = close_at_ident + (so * atr_val)
        tp_price = limit_price - (tp_mult * atr_val)
        sl_price = limit_price + (sl_mult * atr_val)

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
    
    if not is_filled: return 'MISSED', None, 0, 0, 0
    
    entry_price = limit_price
    result = 'TIMEOUT'
    start_sim_idx = fill_idx
    remaining_horizon = max(1, hold_bars - (fill_idx - entry_idx))
    
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
        else:
            if row['high'] >= sl_price:
                result = 'LOSS'; duration = i + (fill_idx-entry_idx); exit_price = sl_price * (1+config.slippage); exit_time = row['timestamp']; break
            elif row['low'] <= tp_price:
                result = 'WIN'; duration = i + (fill_idx-entry_idx); exit_price = tp_price * (1+config.slippage); exit_time = row['timestamp']; break

    return result, exit_time, duration, entry_price, exit_price

def run_portfolio_opt(signals, price_db, config: BacktestConfig):
    if not signals: return -100, 0, 0 # Return large negative for Optuna
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
            sl_dist = s['sl_mult'] * s['atr_val']
            pos_size_usd = (risk_amount * s['entry_p_raw']) / (sl_dist + 1e-9)
            max_pos = min(available_capital * 0.95, mtm_equity * 0.95)
            pos_size_usd = min(pos_size_usd, max_pos)
            if pos_size_usd < 10: continue

            fee_entry = pos_size_usd * config.fee_rate
            fee_exit = (pos_size_usd * (s['exit_p_raw'] / s['entry_p_raw'])) * config.fee_rate
            raw_pnl_pct = (s['exit_p_raw'] - s['entry_p_raw'])/s['entry_p_raw'] if s['type'] == 'LONG' else (s['entry_p_raw'] - s['exit_p_raw'])/s['entry_p_raw']
            net_pnl_usd = (pos_size_usd * raw_pnl_pct) - (fee_entry + fee_exit)
            
            trade_obj = Trade(s['symbol'], s['timestamp'], s['exit_time'], s['type'], s['entry_p_raw'], s['exit_p_raw'], s['result'], net_pnl_usd, 0, s['duration'], pos_size_usd)
            available_capital -= pos_size_usd
            active_trades.append(trade_obj)
            
    for t in active_trades: realized_capital += t.pnl_usd; equity_curve.append(realized_capital)
    
    total_return = ((realized_capital / config.initial_capital) - 1) * 100
    eq_ser = pd.Series(equity_curve)
    drawdown = (eq_ser - eq_ser.cummax()) / (eq_ser.cummax() + 1e-9)
    max_dd = abs(drawdown.min() * 100)
    
    calmar = total_return / (max_dd + 1e-9)
    return calmar, total_return, max_dd

def objective(trial, detected_signals, price_db, config, start_date, end_date):
    # Search Space
    lo = trial.suggest_float('long_offset', -0.5, 0.2, step=0.1)
    so = trial.suggest_float('short_offset', 0.0, 1.0, step=0.1)
    tp_long = trial.suggest_float('tp_long', 1.0, 4.0, step=0.5)
    sl_long = trial.suggest_float('sl_long', 0.5, 2.5, step=0.5)
    tp_short = trial.suggest_float('tp_short', 1.0, 4.0, step=0.5)
    sl_short = trial.suggest_float('sl_short', 0.5, 2.5, step=0.5)
    hold_bars = trial.suggest_int('max_hold', 12, 72, step=12)

    all_sim_signals = []
    s_ts = pd.to_datetime(start_date)
    e_ts = pd.to_datetime(end_date)
    total_potential = 0

    for symbol, df, indices, probas in detected_signals:
        mask = (df.loc[indices, 'timestamp'] >= s_ts) & (df.loc[indices, 'timestamp'] < e_ts)
        curr_indices = indices[mask]
        curr_probas = probas[mask.values] # Ensure slice matches
        total_potential += len(curr_indices)
        
        prob_long, prob_short = curr_probas[:, 1], curr_probas[:, 2]
        for i, idx in enumerate(curr_indices):
            pl, ps = prob_long[i], prob_short[i]
            ttype = 'LONG' if pl >= ps else 'SHORT'
            tp = tp_long if ttype == 'LONG' else tp_short
            sl = sl_long if ttype == 'LONG' else sl_short
            
            res, ext, dur, en_p, ex_p = simulate_trade_opt(df, idx, ttype, lo, so, tp, sl, hold_bars, config)
            if res == 'MISSED': continue
            all_sim_signals.append({
                'timestamp': df.iloc[idx]['timestamp'], 'symbol': symbol, 'type': ttype,
                'result': res, 'exit_time': ext, 'duration': dur, 'entry_p_raw': en_p, 'exit_p_raw': ex_p,
                'atr_val': df.iloc[idx]['atr_14'], 'sl_mult': sl
            })
            
    if trial.number == 0:
        print(f"      [Trial 0 Debug] IS Signals: {total_potential} found -> {len(all_sim_signals)} filled")
    
    calmar, ret, mdd = run_portfolio_opt(all_sim_signals, price_db, config)
    return calmar

def run_oos_test(best_params, detected_signals, price_db, config, start_date, end_date):
    lo = best_params['long_offset']
    so = best_params['short_offset']
    tp_long = best_params['tp_long']
    sl_long = best_params['sl_long']
    tp_short = best_params['tp_short']
    sl_short = best_params['sl_short']
    hold_bars = best_params['max_hold']

    all_sim_signals = []
    s_ts = pd.to_datetime(start_date)
    e_ts = pd.to_datetime(end_date)

    for symbol, df, indices, probas in detected_signals:
        mask = (df.loc[indices, 'timestamp'] >= s_ts) & (df.loc[indices, 'timestamp'] < e_ts)
        curr_indices = indices[mask]
        curr_probas = probas[mask.values]
        
        prob_long, prob_short = curr_probas[:, 1], curr_probas[:, 2]
        for i, idx in enumerate(curr_indices):
            pl, ps = prob_long[i], prob_short[i]
            ttype = 'LONG' if pl >= ps else 'SHORT'
            tp = tp_long if ttype == 'LONG' else tp_short
            sl = sl_long if ttype == 'LONG' else sl_short
            
            res, ext, dur, en_p, ex_p = simulate_trade_opt(df, idx, ttype, lo, so, tp, sl, hold_bars, config)
            if res == 'MISSED': continue
            all_sim_signals.append({
                'timestamp': df.iloc[idx]['timestamp'], 'symbol': symbol, 'type': ttype,
                'result': res, 'exit_time': ext, 'duration': dur, 'entry_p_raw': en_p, 'exit_p_raw': ex_p,
                'atr_val': df.iloc[idx]['atr_14'], 'sl_mult': sl
            })
            
    calmar, ret, mdd = run_portfolio_opt(all_sim_signals, price_db, config)
    return ret, mdd, len(all_sim_signals)

def main():
    config = BacktestConfig()
    clf, features, threshold = load_assets()
    all_files = list(SYMBOLS_DIR.glob("*.parquet"))
    
    print("🔍 Pre-detecting signals...")
    detected_signals = [] 
    price_db = {}
    for file_path in tqdm(all_files):
        try:
            df = pd.read_parquet(file_path)
            if df.empty: continue
            symbol = Path(file_path).stem.replace('_USDT', '').replace('USDT', '')
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
            df = calculate_features_opt(df)
            start_ts = pd.to_datetime(config.start_date)
            scan_indices = df[df['timestamp'] >= start_ts].index
            
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
            mask_threshold = (probas[:,1] > threshold) | (probas[:,2] > threshold)
            
            detected_signals.append((symbol, df[['timestamp','high','low','close','atr_14']], final_indices[mask_threshold], probas[mask_threshold]))
            price_db[symbol] = df.set_index('timestamp')['close']
        except: continue

    total_base = sum(len(x[2]) for x in detected_signals)
    print(f"✅ Pre-detection complete. Found {total_base} base signals across {len(detected_signals)} symbols.")

    # Walk-Forward Setting: 4 months IS, 1 month OOS
    folds = [
        ('2025-08-01', '2025-12-01', '2025-12-01', '2026-01-01'),
        ('2025-09-01', '2026-01-01', '2026-01-01', '2026-02-01'),
        ('2025-10-01', '2026-02-01', '2026-02-01', '2026-03-01'),
    ]

    results = []
    print(f"\n🚀 Walk-Forward Optimization starting ({len(folds)} folds)")
    for i, (is_start, is_end, oos_start, oos_end) in enumerate(folds):
        print(f"\n--- Fold {i+1}: IS [{is_start} to {is_end}] | OOS [{oos_start} to {oos_end}] ---")
        
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda t: objective(t, detected_signals, price_db, config, is_start, is_end), n_trials=50)
        
        best_p = study.best_params
        best_val = study.best_value
        print(f"IS Best Calmar: {best_val:.2f} | Params: {best_p}")
        
        oos_ret, oos_mdd, oos_cnt = run_oos_test(best_p, detected_signals, price_db, config, oos_start, oos_end)
        print(f"OOS Result: Return {oos_ret:.2f}% | MaxDD {oos_mdd:.2f}% | Trades {oos_cnt}")
        
        results.append({
            'fold': i+1, 'is_range': f"{is_start}-{is_end}", 'oos_range': f"{oos_start}-{oos_end}",
            'is_calmar': best_val, 'oos_return': oos_ret, 'oos_max_dd': oos_mdd, 'oos_trades': oos_cnt,
            'params': best_p
        })

    report_df = pd.DataFrame(results)
    report_df.to_csv(BASE_DIR / "ml" / "wfo_optuna_results.csv", index=False)
    print("\n✅ WFO Complete. Results saved to ml/wfo_optuna_results.csv")
    print(report_df[['fold', 'oos_return', 'oos_max_dd', 'oos_trades']])

if __name__ == "__main__":
    main()
