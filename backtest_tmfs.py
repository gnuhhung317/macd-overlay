
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from enum import Enum
from dataclasses import dataclass, field

# Indicators Implementation
def wilders_smoothing(data, period):
    return data.ewm(alpha=1/period, adjust=False).mean()

def calculate_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return wilders_smoothing(tr, period)

def calculate_supertrend(df, factor, period):
    atr = calculate_atr(df, period)
    hl2 = (df['high'] + df['low']) / 2
    
    upperband = hl2 + (factor * atr)
    lowerband = hl2 - (factor * atr)
    
    final_upperband = np.zeros(len(df))
    final_lowerband = np.zeros(len(df))
    supertrend = np.zeros(len(df))
    direction = np.zeros(len(df))
    
    for i in range(1, len(df)):
        if upperband[i] < final_upperband[i-1] or df['close'].iloc[i-1] > final_upperband[i-1]:
            final_upperband[i] = upperband[i]
        else:
            final_upperband[i] = final_upperband[i-1]
            
        if lowerband[i] > final_lowerband[i-1] or df['close'].iloc[i-1] < final_lowerband[i-1]:
            final_lowerband[i] = lowerband[i]
        else:
            final_lowerband[i] = final_lowerband[i-1]
            
        if direction[i-1] == -1:
            if df['close'].iloc[i] > final_upperband[i]:
                direction[i] = 1
                supertrend[i] = final_lowerband[i]
            else:
                direction[i] = -1
                supertrend[i] = final_upperband[i]
        else:
            if df['close'].iloc[i] < final_lowerband[i]:
                direction[i] = -1
                supertrend[i] = final_upperband[i]
            else:
                direction[i] = 1
                supertrend[i] = final_lowerband[i]
                
    return supertrend, direction

def calculate_dmi(df, period=13):
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    up_move = high - high.shift()
    down_move = low.shift() - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr_s = wilders_smoothing(tr, period)
    plus_dm_s = wilders_smoothing(pd.Series(plus_dm), period)
    minus_dm_s = wilders_smoothing(pd.Series(minus_dm), period)
    
    di_plus = (plus_dm_s / tr_s) * 100
    di_minus = (minus_dm_s / tr_s) * 100
    
    return di_plus, di_minus

class TradeSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

@dataclass
class Trade:
    side: TradeSide
    entry_time: datetime
    entry_price: float
    qty: float
    sl: float
    tp: float
    atr_at_entry: float = 0.0
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_raw: float = 0.0
    pnl_pct: float = 0.0
    tp_hit: bool = False
    be_moved: bool = False
    entry_id: str = ""  # To track multiple entries

def round_to_tick(price, tick):
    return np.round(price / tick) * tick

def run_backtest_v3(df, symbol="BTCUSDT", mintick=0.1):
    """
    Fixed version with Pyramiding support (max 2 positions like Pine)
    """
    # Parameters - Match Pine exactly
    initial_capital = 10000
    equity = initial_capital
    qty_pct = 0.04  # 4% per entry (Pine: default_qty_value = 4)
    commission = 0.0003  # 0.03%
    pyramiding = 2  # Max 2 positions
    
    atr_period = 13
    swing_length = 8
    atr_mult = 1.0
    super_length = 21
    dmi_length = 13
    
    # Calculate Indicators
    st1, _ = calculate_supertrend(df, 0.618, atr_period)
    st2, _ = calculate_supertrend(df, 1.618, atr_period)
    st3, _ = calculate_supertrend(df, 2.618, atr_period)
    
    df['avg_st'] = (st1 + st2 + st3) / 3
    df['smoothed_st'] = df['avg_st'].ewm(span=super_length, adjust=False).mean()
    
    df['di_plus'], df['di_minus'] = calculate_dmi(df, dmi_length)
    df['atr'] = calculate_atr(df, 13)
    df['atr_swing'] = calculate_atr(df, swing_length)
    
    # Pivots - Match Pine ta.pivothigh/low exactly
    # Note: In Pine, pivothigh(8,8) looks back 8 and forward 8
    # But it only confirms after 8 bars (repainting behavior)
    df['pivot_high'] = df['high'].rolling(window=swing_length*2+1, center=True).apply(
        lambda x: x.iloc[swing_length] if x.iloc[swing_length] == x.max() and (x == x.max()).sum() == 1 else np.nan, 
        raw=False
    )
    df['pivot_low'] = df['low'].rolling(window=swing_length*2+1, center=True).apply(
        lambda x: x.iloc[swing_length] if x.iloc[swing_length] == x.min() and (x == x.min()).sum() == 1 else np.nan,
        raw=False
    )
    
    # Forward fill to shift pivot to confirmation bar (like Pine behavior)
    # In Pine, pivot appears at the right-most bar of the lookback
    df['pivot_high'] = df['pivot_high'].shift(-swing_length)
    df['pivot_low'] = df['pivot_low'].shift(-swing_length)
    
    df = df.reset_index(drop=True)
    
    # Initialize arrays
    upper_trendline = np.full(len(df), np.nan)
    lower_trendline = np.full(len(df), np.nan)
    upper_slope = np.zeros(len(df))
    lower_slope = np.zeros(len(df))
    
    # State variables
    upper_breakout_state = 0
    lower_breakout_state = 0
    
    # Active trades list (support pyramiding)
    active_trades: List[Trade] = []
    closed_trades: List[Trade] = []
    
    # Risk settings
    sl_atr_mult_val = 8.0
    tp_perc_val = 0.01  # 1%
    trailing_sl = True
    be_after_tp = True
    trailing_tp_enabled = True
    deviation_perc = 0.0001  # 0.01%
    
    equity_curve = [initial_capital]
    
    print(f"Starting backtest v3 (Pyramiding={pyramiding}) for {len(df)} candles...")
    
    for i in range(swing_length * 2 + 1, len(df) - swing_length):  # Avoid edge effects
        curr_close = df['close'].iloc[i]
        curr_high = df['high'].iloc[i]
        curr_low = df['low'].iloc[i]
        curr_open = df['open'].iloc[i]
        curr_atr = df['atr'].iloc[i]
        curr_atr_swing = df['atr_swing'].iloc[i]
        
        # Update Trendlines
        pivot_h_val = df['pivot_high'].iloc[i]
        pivot_l_val = df['pivot_low'].iloc[i]
        
        ph_hit = not pd.isna(pivot_h_val)
        pl_hit = not pd.isna(pivot_l_val)
        
        prev_upper_state = upper_breakout_state
        prev_lower_state = lower_breakout_state
        
        if ph_hit:
            upper_slope[i] = curr_atr_swing / swing_length * atr_mult
            upper_trendline[i] = pivot_h_val - (upper_slope[i] * swing_length)
            upper_breakout_state = 0
        else:
            upper_slope[i] = upper_slope[i-1]
            if not np.isnan(upper_trendline[i-1]):
                upper_trendline[i] = upper_trendline[i-1] - upper_slope[i]
            if curr_close > upper_trendline[i]:
                upper_breakout_state = 1
        
        if pl_hit:
            lower_slope[i] = curr_atr_swing / swing_length * atr_mult
            lower_trendline[i] = pivot_l_val + (lower_slope[i] * swing_length)
            lower_breakout_state = 0
        else:
            lower_slope[i] = lower_slope[i-1]
            if not np.isnan(lower_trendline[i-1]):
                lower_trendline[i] = lower_trendline[i-1] + lower_slope[i]
            if curr_close < lower_trendline[i]:
                lower_breakout_state = 1
        
        # Count current positions
        long_positions = sum(1 for t in active_trades if t.side == TradeSide.LONG)
        short_positions = sum(1 for t in active_trades if t.side == TradeSide.SHORT)
        
        # Entry Conditions (with pyramiding check)
        can_enter_long = long_positions < pyramiding
        can_enter_short = short_positions < pyramiding
        
        long_trigger = can_enter_long and (upper_breakout_state == 1 and prev_upper_state == 0) and \
                       (df['di_plus'].iloc[i] > df['di_minus'].iloc[i]) and \
                       (curr_close > df['smoothed_st'].iloc[i])
        
        short_trigger = can_enter_short and (lower_breakout_state == 1 and prev_lower_state == 0) and \
                        (df['di_minus'].iloc[i] > df['di_plus'].iloc[i]) and \
                        (curr_close < df['smoothed_st'].iloc[i])
        
        # Manage existing positions first (SL/TP/Exit)
        trades_to_remove = []
        
        for idx, trade in enumerate(active_trades):
            exit_triggered = False
            exit_price = 0
            exit_reason = ""
            
            entry_atr = trade.atr_at_entry
            entry_p = trade.entry_price
            
            if trade.side == TradeSide.LONG:
                # Update Trailing SL
                if trailing_sl:
                    new_sl = curr_high - sl_atr_mult_val * entry_atr
                    new_sl = round_to_tick(new_sl, mintick)
                    if be_after_tp and trade.tp_hit:
                        new_sl = max(new_sl, entry_p)
                    trade.sl = max(trade.sl, new_sl)
                
                # Check SL
                if curr_low <= trade.sl:
                    exit_triggered = True
                    exit_price = trade.sl if curr_open > trade.sl else curr_open
                    exit_reason = "SL Triggered"
                
                # Check TP
                if not trade.tp_hit and curr_high >= trade.tp:
                    trade.tp_hit = True
                    if not trailing_tp_enabled:
                        exit_triggered = True
                        exit_price = trade.tp
                        exit_reason = "TP Target Hit"
                
                # Trailing TP
                if trailing_tp_enabled and trade.tp_hit:
                    trail_offset = round_to_tick(curr_high * deviation_perc, mintick)
                    trail_sl = curr_high - trail_offset
                    trade.sl = max(trade.sl, trail_sl)
                
                # Trend Crossover Exit
                if i >= 2 and not exit_triggered:
                    p1_close = df['close'].iloc[i-1]
                    p1_trend = df['smoothed_st'].iloc[i-1]
                    p2_close = df['close'].iloc[i-2]
                    p2_trend = df['smoothed_st'].iloc[i-2]
                    if p2_close >= p2_trend and p1_close < p1_trend:
                        exit_triggered = True
                        exit_price = curr_open
                        exit_reason = "Trend Crossover"
            
            else:  # SHORT
                if trailing_sl:
                    new_sl = curr_low + sl_atr_mult_val * entry_atr
                    new_sl = round_to_tick(new_sl, mintick)
                    if be_after_tp and trade.tp_hit:
                        new_sl = min(new_sl, entry_p)
                    trade.sl = min(trade.sl, new_sl)
                
                if curr_high >= trade.sl:
                    exit_triggered = True
                    exit_price = trade.sl if curr_open < trade.sl else curr_open
                    exit_reason = "SL Triggered"
                
                if not trade.tp_hit and curr_low <= trade.tp:
                    trade.tp_hit = True
                    if not trailing_tp_enabled:
                        exit_triggered = True
                        exit_price = trade.tp
                        exit_reason = "TP Target Hit"
                
                if trailing_tp_enabled and trade.tp_hit:
                    trail_offset = round_to_tick(curr_low * deviation_perc, mintick)
                    trail_sl = curr_low + trail_offset
                    trade.sl = min(trade.sl, trail_sl)
                
                if i >= 2 and not exit_triggered:
                    p1_close = df['close'].iloc[i-1]
                    p1_trend = df['smoothed_st'].iloc[i-1]
                    p2_close = df['close'].iloc[i-2]
                    p2_trend = df['smoothed_st'].iloc[i-2]
                    if p2_close <= p2_trend and p1_close > p1_trend:
                        exit_triggered = True
                        exit_price = curr_open
                        exit_reason = "Trend Crossover"
            
            if exit_triggered:
                trade.exit_time = df['timestamp'].iloc[i] if 'timestamp' in df.columns else i
                trade.exit_price = exit_price
                trade.exit_reason = exit_reason
                
                raw_diff = (trade.exit_price - trade.entry_price) if trade.side == TradeSide.LONG else (trade.entry_price - trade.exit_price)
                trade.pnl_raw = raw_diff * trade.qty
                trade.pnl_pct = (raw_diff / trade.entry_price) * 100
                
                fees = (trade.entry_price + trade.exit_price) * trade.qty * commission
                equity += trade.pnl_raw - fees
                
                closed_trades.append(trade)
                trades_to_remove.append(idx)
        
        # Remove closed trades
        for idx in sorted(trades_to_remove, reverse=True):
            active_trades.pop(idx)
        
        # Open new positions
        if long_trigger:
            entry_p = curr_close
            sl_p = round_to_tick(entry_p - sl_atr_mult_val * curr_atr, mintick)
            tp_p = round_to_tick(entry_p * (1 + tp_perc_val), mintick)
            pos_qty = (equity * qty_pct) / entry_p
            
            new_trade = Trade(
                TradeSide.LONG, 
                df['timestamp'].iloc[i] if 'timestamp' in df.columns else i, 
                entry_p, pos_qty, sl_p, tp_p,
                entry_id=f"L{len(closed_trades) + len(active_trades)}"
            )
            new_trade.atr_at_entry = curr_atr
            active_trades.append(new_trade)
        
        if short_trigger:
            entry_p = curr_close
            sl_p = round_to_tick(entry_p + sl_atr_mult_val * curr_atr, mintick)
            tp_p = round_to_tick(entry_p * (1 - tp_perc_val), mintick)
            pos_qty = (equity * qty_pct) / entry_p
            
            new_trade = Trade(
                TradeSide.SHORT,
                df['timestamp'].iloc[i] if 'timestamp' in df.columns else i,
                entry_p, pos_qty, sl_p, tp_p,
                entry_id=f"S{len(closed_trades) + len(active_trades)}"
            )
            new_trade.atr_at_entry = curr_atr
            active_trades.append(new_trade)
        
        equity_curve.append(equity)
    
    # Close any remaining trades at the end
    for trade in active_trades:
        last_close = df['close'].iloc[-1]
        trade.exit_price = last_close
        trade.exit_time = df['timestamp'].iloc[-1] if 'timestamp' in df.columns else len(df)-1
        trade.exit_reason = "End of Data"
        
        raw_diff = (trade.exit_price - trade.entry_price) if trade.side == TradeSide.LONG else (trade.entry_price - trade.exit_price)
        trade.pnl_raw = raw_diff * trade.qty
        trade.pnl_pct = (raw_diff / trade.entry_price) * 100
        
        fees = (trade.entry_price + trade.exit_price) * trade.qty * commission
        equity += trade.pnl_raw - fees
        closed_trades.append(trade)
    
    return closed_trades, equity, equity_curve

def calculate_metrics(trades, equity_curve):
    if not trades:
        return {}
    
    returns = pd.Series(equity_curve).pct_change().dropna()
    total_returns = (equity_curve[-1] / equity_curve[0]) - 1
    
    sharpe = (returns.mean() / returns.std() * np.sqrt(252 * 24)) if returns.std() != 0 else 0
    
    gross_profits = sum(t.pnl_raw for t in trades if t.pnl_raw > 0)
    gross_losses = abs(sum(t.pnl_raw for t in trades if t.pnl_raw < 0))
    profit_factor = gross_profits / gross_losses if gross_losses != 0 else np.inf
    
    equity_series = pd.Series(equity_curve)
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    max_dd = drawdown.min()
    
    win_count = len([t for t in trades if t.pnl_raw > 0])
    win_rate = win_count / len(trades) * 100
    
    # Count long vs short
    long_trades = len([t for t in trades if t.side == TradeSide.LONG])
    short_trades = len([t for t in trades if t.side == TradeSide.SHORT])
    
    return {
        "Total Trades": len(trades),
        "Long Trades": long_trades,
        "Short Trades": short_trades,
        "Win Rate": f"{win_rate:.2f}%",
        "Net Profit %": f"{total_returns*100:.2f}%",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Profit Factor": f"{profit_factor:.2f}",
        "Max Drawdown": f"{max_dd*100:.2f}%",
        "Final Equity": f"${equity_curve[-1]:.2f}"
    }

# Example usage with dummy data
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range('2025-01-01', periods=5000, freq='1h')
    np.random.seed(42)
    
    # Generate synthetic OHLCV
    close = 100 + np.cumsum(np.random.randn(5000) * 0.1)
    high = close + np.abs(np.random.randn(5000) * 0.5)
    low = close - np.abs(np.random.randn(5000) * 0.5)
    open_p = close + np.random.randn(5000) * 0.1
    volume = np.random.randint(1000, 10000, 5000)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_p,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    trades, final_equity, equity_curve = run_backtest_v3(df, "TEST", 0.01)
    metrics = calculate_metrics(trades, equity_curve)
    
    print("\n" + "="*50)
    print("BACKTEST V3 (WITH PYRAMIDING)")
    print("="*50)
    for k, v in metrics.items():
        print(f"{k:20}: {v}")
