import os
import gc
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

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
    initial_capital: float = 100.0
    risk_per_trade: float = 0.05  # 10% of capital per trade
    fee_rate: float = 0.001       # 0.1% per trade
    slippage: float = 0.0005      # 0.05% slippage
    max_open_trades: int = 10
    max_bars_hold: int = 48
    start_date: str = '2025-03-11'
    end_date: str = None
    leverage: float = 1.0
    # Limit Order Params
    long_atr_offset: float = -0.1
    short_atr_offset: float = 0.5
    limit_wait_bars: int = 5

@dataclass
class Trade:
    symbol: str
    entry_time: datetime
    exit_time: datetime
    type: str  # LONG/SHORT
    entry_price: float
    exit_price: float
    result: str  # WIN/LOSS/TIMEOUT/MISSED
    pnl_usd: float
    pnl_pct: float
    fees: float
    duration: int
    mfe_atr: float
    mae_atr: float
    pos_size_usd: float = 0.0

def load_assets():
    if not META_PATH.exists() or not MODEL_PATH.exists():
        print("❌ Missing model or meta file!")
        return None, [], 0.6
    meta = joblib.load(META_PATH)
    clf = joblib.load(MODEL_PATH)
    features = meta.get('features', []) if isinstance(meta, dict) else meta
    threshold = meta.get('threshold', 0.6)
    return clf, features, threshold

def calculate_features_backtest(df):
    """Calculate missing features for backtesting logic if not present."""
    df = df.copy()
    
    # Ensure EMA
    if 'ema_20' not in df.columns:
        df['ema_20'] = df['close'].ewm(span=20).mean()
    if 'ema_50' not in df.columns:
        df['ema_50'] = df['close'].ewm(span=50).mean()
    
    # Volume SMA for ratio
    if 'vol_ratio' not in df.columns:
        vol_sma = df['volume'].rolling(20).mean().shift(1)
        df['vol_ratio'] = df['volume'] / (vol_sma + 1e-9)
    
    # ATR Calculation
    if 'atr_14' not in df.columns:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()
        
    if 'atr_pct' not in df.columns:
        df['atr_pct'] = (df['atr_14'] / df['close']) * 100
    
    # Sniper-specific features
    if 'upper_wick_ratio' not in df.columns:
        df['upper_wick_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-9)
    if 'dist_to_ema50_atr' not in df.columns:
        df['dist_to_ema50_atr'] = (df['close'] - df['ema_50']) / (df['atr_14'] + 1e-9)
    if 'vol_acceleration' not in df.columns:
        df['vol_acceleration'] = df['volume'] / (df['volume'].shift(1) + 1e-9)
    
    return df

def simulate_trade(df, entry_idx, trade_type, config: BacktestConfig):
    """
    Simulate trade with Limit Order validation and Fixed Entry Price.
    """
    horizon = config.max_bars_hold
    full_len = len(df)
    
    ident_row = df.iloc[entry_idx]
    close_at_ident = ident_row['close']
    atr_val = ident_row['atr_14']
    
    # Pre-calculated Entry Level (Fixed, NOT updated in loop)
    if trade_type == 'LONG':
        # return
        limit_price = close_at_ident + (config.long_atr_offset * atr_val)
        tp_price = limit_price + (2.0 * atr_val)
        sl_price = limit_price - (1.0 * atr_val)
    else: # SHORT
        limit_price = close_at_ident + (config.short_atr_offset * atr_val)
        tp_price = limit_price - (2.5 * atr_val)
        sl_price = limit_price + (1.5 * atr_val)

    # 1. LIMIT ORDER VALIDATION: Check if filled within wait bars
    is_filled = False
    fill_idx = -1
    for i in range(1, config.limit_wait_bars + 1):
        idx = entry_idx + i
        if idx >= full_len: break
        
        row = df.iloc[idx]
        if trade_type == 'LONG':
            if row['low'] < limit_price: # Price must PENETRATE limit
                is_filled = True
                fill_idx = idx
                break
        else: # SHORT
            if row['high'] > limit_price: # Price must PENETRATE limit
                is_filled = True
                fill_idx = idx
                break
    
    if not is_filled:
        return 'MISSED', 0, 0, 0, limit_price, limit_price, ident_row['timestamp']

    # 2. TRADE SIMULATION from fill point
    entry_price = limit_price # Assume filled at limit (slightly optimistic, but better than market)
    max_mfe_atr = 0
    min_mae_atr = 0
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
            # INTRA-BAR: Check if both hit in same candle
            # Worst-case: SL hit first
            if row['low'] <= sl_price:
                result = 'LOSS'
                duration = i + (fill_idx - entry_idx)
                exit_price = sl_price * (1 - config.slippage)
                exit_time = row['timestamp']
                break
            elif row['high'] >= tp_price:
                result = 'WIN'
                duration = i + (fill_idx - entry_idx)
                exit_price = tp_price * (1 - config.slippage)
                exit_time = row['timestamp']
                break
            
            # Update MFE/MAE
            current_mfe_atr = (row['high'] - entry_price) / (atr_val + 1e-9)
            current_mae_atr = (row['low'] - entry_price) / (atr_val + 1e-9)
        else: # SHORT
            # Worst-case: SL hit first
            if row['high'] >= sl_price:
                result = 'LOSS'
                duration = i + (fill_idx - entry_idx)
                exit_price = sl_price * (1 + config.slippage)
                exit_time = row['timestamp']
                break
            elif row['low'] <= tp_price:
                result = 'WIN'
                duration = i + (fill_idx - entry_idx)
                exit_price = tp_price * (1 + config.slippage)
                exit_time = row['timestamp']
                break
                
            current_mfe_atr = (entry_price - row['low']) / (atr_val + 1e-9)
            current_mae_atr = (entry_price - row['high']) / (atr_val + 1e-9)

        max_mfe_atr = max(max_mfe_atr, current_mfe_atr)
        min_mae_atr = min(min_mae_atr, current_mae_atr)

    return result, duration, max_mfe_atr, min_mae_atr, entry_price, exit_price, exit_time

def backtest_symbol(file_path, features, clf, threshold, config: BacktestConfig):
    try:
        df = pd.read_parquet(file_path)
        if df.empty: return None, None
        
        symbol = Path(file_path).stem.replace('_USDT', '').replace('USDT', '')
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        df = calculate_features_backtest(df)
        
        scan_indices = df.index
        if config.start_date:
            start_ts = pd.to_datetime(config.start_date)
            scan_indices = df[df['timestamp'] >= start_ts].index
            
        if config.end_date:
            end_ts = pd.to_datetime(config.end_date)
            # Find intersection of previous index filter and end_date filter
            end_indices = df[df['timestamp'] <= end_ts].index
            scan_indices = scan_indices.intersection(end_indices)
            
        if len(scan_indices) == 0: return None, None
        
        # 1. Stage 1 Filter: Ignition Bar
        vol_sma = df['volume'].rolling(20).mean().shift(1)
        resistance_50 = df['high'].rolling(50).max().shift(1)
        dist_to_res = (resistance_50 - df['close']) / (df['close'] + 1e-9)
        
        c1 = (df['close'] > df['open']) & (df['close'] > df['ema_20'])
        c2 = ((df['close'] - df['open']) / df['open']) > 0.015
        c3 = (df['volume'] > vol_sma * 1.5) & (df['volume'] < vol_sma * 4.0)
        c4 = (df['rsi_14'] >= 55) & (df['rsi_14'] <= 72)
        # c5 = dist_to_res > -0.05 (Removed: Counter-productive to profitability)
        c5 = True
        
        ignition_mask = (c1 & c2 & c3 & c4 & c5).reindex(scan_indices, fill_value=False)
        final_scan_indices = scan_indices[ignition_mask]
        
        if len(final_scan_indices) == 0: return None, None
        
        # 2. Stage 2: VECTORIZED Scoring
        X_batch = df.loc[final_scan_indices, features].apply(pd.to_numeric, errors='coerce').fillna(0)
        probas_batch = clf.predict_proba(X_batch)
        
        prob_long = probas_batch[:, 1]
        prob_short = probas_batch[:, 2]
        
        all_potential_signals = []
        for i, idx in enumerate(final_scan_indices):
            pl, ps = prob_long[i], prob_short[i]
            if pl > threshold or ps > threshold:
                trade_type = 'LONG' if pl > threshold else 'SHORT'
                
                result, duration, res_mfe, res_mae, entry_p, exit_p, exit_t = simulate_trade(df, idx, trade_type, config)
                
                if result == 'MISSED': continue # Pro Quant: Ignore signals that never fill
                
                all_potential_signals.append({
                    'timestamp': df.iloc[idx]['timestamp'],
                    'symbol': symbol,
                    'type': trade_type,
                    'entry_price': entry_p,
                    'exit_price': exit_p,
                    'exit_time': exit_t,
                    'prob_long': pl,
                    'prob_short': ps,
                    'result': result,
                    'duration': duration,
                    'mfe_atr': res_mfe,
                    'mae_atr': res_mae,
                    'atr_val': df.iloc[idx]['atr_14']
                })
        
        # Return signals AND the price series for MTM lookups
        price_series = df.set_index('timestamp')['close']
        return all_potential_signals, price_series
        
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        return None, None

def run_portfolio_simulation(all_signals, price_db, config: BacktestConfig):
    if not all_signals: return None, []
    all_signals = sorted(all_signals, key=lambda x: x['timestamp'])
    
    realized_capital = config.initial_capital
    available_capital = config.initial_capital # Free Margin
    active_trades: List[Trade] = []
    closed_trades: List[Trade] = []
    equity_curve = [(all_signals[0]['timestamp'] - timedelta(hours=1), realized_capital)]
    max_open_notional = 0.0
    
    for signal in all_signals:
        curr_time = signal['timestamp']
        
        # 1. Resolve trades that exited BEFORE or AT current signal time
        active_trades.sort(key=lambda x: x.exit_time)
        while active_trades and active_trades[0].exit_time <= curr_time:
            t = active_trades.pop(0)
            realized_capital += t.pnl_usd
            # Return margin + net profit (which could be negative)
            available_capital += (t.pos_size_usd / config.leverage) + t.pnl_usd
            closed_trades.append(t)
            equity_curve.append((t.exit_time, realized_capital))
            
        # 2. Calculate MTM Equity for sizing
        floating_pnl = 0
        used_margin = 0
        for t in active_trades:
            used_margin += t.pos_size_usd / config.leverage
            if t.symbol in price_db and curr_time in price_db[t.symbol].index:
                curr_p = price_db[t.symbol].loc[curr_time]
                if t.type == 'LONG':
                    pnl_pct = (curr_p - t.entry_price) / (t.entry_price + 1e-9)
                else:
                    pnl_pct = (t.entry_price - curr_p) / (t.entry_price + 1e-9)
                floating_pnl += t.pos_size_usd * pnl_pct
                
        mtm_equity = realized_capital + floating_pnl
        current_open_notional = sum(t.pos_size_usd for t in active_trades)
        max_open_notional = max(max_open_notional, current_open_notional)
            
        # 3. Check if we can open new trade
        if len(active_trades) < config.max_open_trades:
            # Risk Management (based on MTM Equity)
            risk_amount = mtm_equity * config.risk_per_trade
            atr = signal['atr_val']
            sl_dist = 1.0 * atr if signal['type'] == 'LONG' else 1.5 * atr
            # Safety: Ensure SL distance is at least 0.3% to prevent extreme volume
            min_sl_dist = signal['entry_price'] * 0.003
            effective_sl_dist = max(sl_dist, min_sl_dist)
            
            # Position sizing based on risk
            pos_size_usd = (risk_amount * signal['entry_price']) / (effective_sl_dist + 1e-9)
            
            # Margin Check: Can't exceed available capital * leverage
            max_allowed_pos = available_capital * config.leverage * 0.95
            pos_size_usd = min(pos_size_usd, max_allowed_pos)
            
            if pos_size_usd < 10: continue

            fee_entry = pos_size_usd * config.fee_rate
            fee_exit = (pos_size_usd * (signal['exit_price'] / (signal['entry_price'] + 1e-9))) * config.fee_rate
            total_fees = fee_entry + fee_exit
            
            if signal['type'] == 'LONG':
                raw_pnl_pct = (signal['exit_price'] - signal['entry_price']) / (signal['entry_price'] + 1e-9)
            else:
                raw_pnl_pct = (signal['entry_price'] - signal['exit_price']) / (signal['entry_price'] + 1e-9)
                
            gross_pnl_usd = pos_size_usd * raw_pnl_pct
            net_pnl_usd = gross_pnl_usd - total_fees
            
            trade_obj = Trade(
                symbol=signal['symbol'],
                entry_time=signal['timestamp'],
                exit_time=signal['exit_time'],
                type=signal['type'],
                entry_price=signal['entry_price'],
                exit_price=signal['exit_price'],
                result=signal['result'],
                pnl_usd=net_pnl_usd,
                pnl_pct=(net_pnl_usd / (pos_size_usd + 1e-9)) * 100,
                fees=total_fees,
                duration=signal['duration'],
                mfe_atr=signal['mfe_atr'],
                mae_atr=signal['mae_atr'],
                pos_size_usd=pos_size_usd
            )
            
            margin_required = pos_size_usd / config.leverage
            available_capital -= margin_required
            active_trades.append(trade_obj)
    
    # 3. Final cleanup
    active_trades.sort(key=lambda x: x.exit_time)
    for t in active_trades:
        realized_capital += t.pnl_usd
        closed_trades.append(t)
        equity_curve.append((t.exit_time, realized_capital))
            
    return closed_trades, equity_curve, max_open_notional

def run_backtest_with_config(config: BacktestConfig):
    print(f"\n{'='*60}")
    print(f"🚀 QUANT-REFINED SNIPER BACKTEST")
    print(f"Fixed Entry Logic | Limit Validation | Vectorized AI")
    print(f"Leverage: {config.leverage}x | Risk: {config.risk_per_trade * 100}% | Range: {config.start_date} to {config.end_date if config.end_date else 'Now'}")
    print(f"{'='*60}")
    
    clf, features, threshold = load_assets()
    if clf is None: return None, None, None, None
    
    all_files = list(SYMBOLS_DIR.glob("*.parquet"))
    print(f"Scanning {len(all_files)} symbols...")
    
    potential_signals = []
    price_db = {}
    for i, file_path in enumerate(all_files):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(all_files)}...")
            
        res, prices = backtest_symbol(file_path, features, clf, threshold, config)
        if res:
            potential_signals.extend(res)
            symbol = Path(file_path).stem.replace('_USDT', '').replace('USDT', '')
            price_db[symbol] = prices
            
    if not potential_signals:
        print("No signals found.")
        return None, None, None, None
        
    print(f"Found {len(potential_signals)} filled signals. Simulating portfolio...")
    trades, equity_curve, max_notional = run_portfolio_simulation(potential_signals, price_db, config)
    
    if not trades:
        print("No trades executed.")
        return potential_signals, price_db, trades, equity_curve

    report_df = pd.DataFrame([vars(t) for t in trades])
    report_df = report_df.sort_values('entry_time')
    
    print(f"\n✅ PORTFOLIO RESULTS:")
    final_cap = config.initial_capital + report_df['pnl_usd'].sum()
    total_return = ((final_cap / config.initial_capital) - 1) * 100
    
    print(f"Initial Capital: ${config.initial_capital:.2f}")
    print(f"Final Capital:   ${final_cap:.2f}")
    print(f"Total Return:    {total_return:.2f}%")
    print(f"Total Trades:    {len(report_df)}")
    
    equity_series = pd.DataFrame(equity_curve, columns=['time', 'val']).set_index('time')['val']
    
    # Advanced Metrics
    # 1. Profit Factor
    total_gains = report_df[report_df['pnl_usd'] > 0]['pnl_usd'].sum()
    total_losses = abs(report_df[report_df['pnl_usd'] < 0]['pnl_usd'].sum())
    profit_factor = total_gains / (total_losses + 1e-9)
    
    # 2. Daily Returns for Sharpe
    daily_equity = equity_series.resample('D').last().ffill()
    daily_returns = daily_equity.pct_change().dropna()
    sharpe = (daily_returns.mean() / (daily_returns.std() + 1e-9)) * np.sqrt(365) if len(daily_returns) > 1 else 0
    
    # 3. Calmar
    roll_max = equity_series.cummax()
    daily_drawdown = (equity_series - roll_max) / (roll_max + 1e-9)
    max_dd = daily_drawdown.min() * 100
    
    ann_return = total_return / (max(1, (equity_series.index[-1] - equity_series.index[0]).days) / 365.25)
    calmar = abs(ann_return / (max_dd + 1e-9)) if max_dd != 0 else 0

    print(f"Peak Open Vol:   ${max_notional:.2f} (Limit: ${config.initial_capital * config.leverage:.2f})")
    print(f"Max Drawdown:    {max_dd:.2f}%")
    print(f"Profit Factor:   {profit_factor:.2f}")
    print(f"Sharpe Ratio:    {sharpe:.2f}")
    print(f"Calmar Ratio:    {calmar:.2f}")

    for t in ['LONG', 'SHORT']:
        subset = report_df[report_df['type'] == t]
        if subset.empty: print(f"\n--- No {t} trades executed ---"); continue
        
        wins = len(subset[subset['result'] == 'WIN'])
        losses = len(subset[subset['result'] == 'LOSS'])
        wr = (wins / len(subset)) * 100
        print(f"\n--- {t} Performance ---")
        print(f"Trades: {len(subset)} | Winrate: {wr:.2f}%")
        print(f"Avg PnL: ${subset['pnl_usd'].mean():.2f}")
        print(f"Median MFE-ATR: {subset['mfe_atr'].median():.2f} | MAE-ATR: {subset['mae_atr'].median():.2f}")

    output_file = BASE_DIR / "ml" / "backtest_results_quant_sniper.csv"
    report_df.to_csv(output_file, index=False)
    print(f"\nQuant report saved: {output_file}")
    
    return potential_signals, price_db, trades, equity_curve

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sniper Model Backtest")
    parser.add_argument('--start', type=str, default='2025-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--leverage', type=float, default=1.0, help='Leverage multiplier')
    parser.add_argument('--capital', type=float, default=100.0, help='Initial capital')
    parser.add_argument('--risk', type=float, default=0.05, help='Risk per trade (0.1 = 10%)')
    
    args = parser.parse_args()
    
    config = BacktestConfig(
        start_date=args.start,
        end_date=args.end,
        leverage=args.leverage,
        initial_capital=args.capital,
        risk_per_trade=args.risk
    )
    
    run_backtest_with_config(config)

if __name__ == "__main__":
    main()
