import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime, timezone
import argparse
from typing import List, Dict, Tuple, Optional
from enum import Enum
from dataclasses import dataclass

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
    # Pine ta.atr(13) uses RMA (Wilders)
    atr = calculate_atr(df, period)
    hl2 = (df['high'] + df['low']) / 2
    
    upperband = hl2 + (factor * atr)
    lowerband = hl2 - (factor * atr)
    
    final_upperband = np.zeros(len(df))
    final_lowerband = np.zeros(len(df))
    supertrend = np.zeros(len(df))
    direction = np.zeros(len(df)) # 1 for up, -1 for down
    
    for i in range(1, len(df)):
        # Upper band
        if upperband[i] < final_upperband[i-1] or df['close'][i-1] > final_upperband[i-1]:
            final_upperband[i] = upperband[i]
        else:
            final_upperband[i] = final_upperband[i-1]
            
        # Lower band
        if lowerband[i] > final_lowerband[i-1] or df['close'][i-1] < final_lowerband[i-1]:
            final_lowerband[i] = lowerband[i]
        else:
            final_lowerband[i] = final_lowerband[i-1]
            
        # Direction
        if i == 1:
            supertrend[0] = final_upperband[0] if df['close'][0] <= final_upperband[0] else final_lowerband[0]
            direction[0] = -1 if df['close'][0] <= final_upperband[0] else 1
            
        if direction[i-1] == -1:
            if df['close'][i] > final_upperband[i]:
                direction[i] = 1
                supertrend[i] = final_lowerband[i]
            else:
                direction[i] = -1
                supertrend[i] = final_upperband[i]
        else:
            if df['close'][i] < final_lowerband[i]:
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

class TradeState(Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    CLOSED = "CLOSED"

@dataclass
class BacktestConfig:
    initial_capital: float = 10000.0
    qty_pct: float = 1
    commission: float = 0.0003
    atr_period: int = 13
    swing_length: int = 8
    atr_mult: float = 1.0
    super_length: int = 21
    dmi_length: int = 13
    sl_atr_mult: float = 8.0
    tp_perc: float = 0.01
    trailing_sl: bool = True
    be_after_tp: bool = True
    trailing_tp_enabled: bool = True
    deviation_perc: float = 0.0001
    min_tick: float = 0.1
    symbol: str = "BTCUSDT"
    fast_len: int = 9
    slow_len: int = 21
    atr_len: int = 14

@dataclass
class Trade:
    symbol: str
    side: TradeSide
    signal_time: datetime
    entry_price: float
    qty: float
    sl: float
    tp: float
    atr_at_entry: float = 0.0
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl_raw: float = 0.0
    pnl_pct: float = 0.0
    tp_hit: bool = False
    state: TradeState = TradeState.PENDING
    fees: float = 0.0

def round_to_tick(price, tick):
    return np.round(price / tick) * tick

def resample_data(df, timeframe):
    if timeframe == "1h":
        return df
    
    print(f"Resampling data to {timeframe}...")
    df = df.set_index('timestamp')
    resampled = df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    resampled = resampled.reset_index()
    return resampled

def calculate_indicators(df, config: BacktestConfig):
    print('Calculating indicators...')
    df['sma_fast'] = df['close'].rolling(window=config.fast_len).mean()
    df['sma_slow'] = df['close'].rolling(window=config.slow_len).mean()
    df['atr'] = calculate_atr(df, config.atr_len)

    df['candle_range'] = df['high'] - df['low']
    df['min_oc'] = df[['open', 'close']].min(axis=1)
    df['max_oc'] = df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df['min_oc'] - df['low']
    df['upper_wick'] = df['high'] - df['max_oc']

    df['reject_low'] = (df['candle_range'] > 0) & (df['lower_wick'] > df['candle_range'] * 0.35)
    df['reject_high'] = (df['candle_range'] > 0) & (df['upper_wick'] > df['candle_range'] * 0.35)
    return df

def calculate_signals(df, config: BacktestConfig):
    print('Calculating signals...')
    df['long_trigger'] = (df['close'] < df['sma_fast']) & (df['close'] > df['sma_slow']) & df['reject_low']
    df['short_trigger'] = (df['close'] > df['sma_fast']) & (df['close'] < df['sma_slow']) & df['reject_high']
    df['exit_signal_long'] = False
    df['exit_signal_short'] = False
    return df

def simulate_trade_step(trade: Trade, row: pd.Series, config: BacktestConfig):
    if trade.state == TradeState.CLOSED:
        return False, trade

    if trade.state == TradeState.ACTIVE:
        curr_high = row['high']
        curr_low = row['low']
        curr_open = row['open']
        curr_close = row['close']

        exit_triggered = False
        exit_price = 0.0
        exit_reason = ""

        # 1. Update Trailing SL (Giữ nguyên logic cực tốt của bạn)
        if config.trailing_sl:
            if trade.side == TradeSide.LONG:
                new_sl = curr_high - config.sl_atr_mult * trade.atr_at_entry
                new_sl = round_to_tick(new_sl, config.min_tick)
                if config.be_after_tp and trade.tp_hit:
                    new_sl = max(new_sl, trade.entry_price)
                trade.sl = max(trade.sl, new_sl)
            else: # SHORT
                new_sl = curr_low + config.sl_atr_mult * trade.atr_at_entry
                new_sl = round_to_tick(new_sl, config.min_tick)
                if config.be_after_tp and trade.tp_hit:
                    new_sl = min(new_sl, trade.entry_price)
                trade.sl = min(trade.sl, new_sl)

        # 2. Trend Crossover Exit (Ưu tiên check trước vì nó trigger từ Open nến mới)
        if trade.side == TradeSide.LONG and row['exit_signal_long']:
            trade.state = TradeState.CLOSED
            trade.exit_price = curr_open
            trade.exit_reason = "Trend Crossover"
            trade.exit_time = row['timestamp']
            return True, trade
        elif trade.side == TradeSide.SHORT and row['exit_signal_short']:
            trade.state = TradeState.CLOSED
            trade.exit_price = curr_open
            trade.exit_reason = "Trend Crossover"
            trade.exit_time = row['timestamp']
            return True, trade

        # 3. Intrabar Routing thông minh (Khắc phục điểm mù bi quan)
        sl_hit = False
        tp_hit = False

        if trade.side == TradeSide.LONG:
            if curr_low <= trade.sl: sl_hit = True
            if curr_high >= trade.tp: tp_hit = True
        else: # SHORT
            if curr_high >= trade.sl: sl_hit = True
            if curr_low <= trade.tp: tp_hit = True

        if sl_hit and tp_hit:
            # Nếu chạm cả hai, dùng heuristic màu nến để đoán đường giá
            is_green = curr_close > curr_open
            
            # Long + Nến xanh: Giá quét râu dưới chạm SL trước khi bật lên High
            # Short + Nến đỏ: Giá quét râu trên chạm SL trước khi sập xuống Low
            if (trade.side == TradeSide.LONG and is_green) or (trade.side == TradeSide.SHORT and not is_green):
                exit_triggered = True
                exit_price = trade.sl
                exit_reason = "SL Triggered (Intrabar Conflict)"
            else:
                if config.trailing_tp_enabled:
                    trade.tp_hit = True
                    # Cập nhật trailing ngay trong nến
                    if trade.side == TradeSide.LONG:
                        trail_sl = curr_high - round_to_tick(curr_high * config.deviation_perc, config.min_tick)
                        trade.sl = max(trade.sl, trail_sl)
                    else:
                        trail_sl = curr_low + round_to_tick(curr_low * config.deviation_perc, config.min_tick)
                        trade.sl = min(trade.sl, trail_sl)
                else:
                    exit_triggered = True
                    exit_price = trade.tp
                    exit_reason = "TP Target Hit (Intrabar Conflict)"

        # Nếu chỉ chạm 1 trong 2
        elif sl_hit:
            exit_triggered = True
            if trade.side == TradeSide.LONG:
                exit_price = trade.sl if curr_open > trade.sl else curr_open
            else:
                exit_price = trade.sl if curr_open < trade.sl else curr_open
            exit_reason = "SL Triggered"

        elif tp_hit and not trade.tp_hit:
            trade.tp_hit = True
            if not config.trailing_tp_enabled:
                exit_triggered = True
                exit_price = trade.tp
                exit_reason = "TP Target Hit"
            else:
                if trade.side == TradeSide.LONG:
                    trail_sl = curr_high - round_to_tick(curr_high * config.deviation_perc, config.min_tick)
                    trade.sl = max(trade.sl, trail_sl)
                else:
                    trail_sl = curr_low + round_to_tick(curr_low * config.deviation_perc, config.min_tick)
                    trade.sl = min(trade.sl, trail_sl)

        if exit_triggered:
            trade.state = TradeState.CLOSED
            trade.exit_price = exit_price
            trade.exit_reason = exit_reason
            trade.exit_time = row['timestamp']
            return True, trade

    return False, trade

def run_backtest_event_driven(df, config: BacktestConfig):
    df = calculate_indicators(df, config)
    df = calculate_signals(df, config)
    
    equity = config.initial_capital
    active_trade: Optional[Trade] = None
    trades: List[Trade] = []
    equity_curve = [equity]
    
    print(f"Starting event-driven backtest for {len(df)} candles...")
    
    # Bắt đầu từ 1 để có thể lookback 1 nến
    for i in range(1, len(df)):
        curr_row = df.iloc[i]
        prev_row = df.iloc[i-1] # Lấy tín hiệu từ nến trước
        
        # 1. Quản lý lệnh đang mở dựa trên giá của nến HIỆN TẠI
        if active_trade:
            closed, updated_trade = simulate_trade_step(active_trade, curr_row, config)
            if closed:
                side_mult = 1 if updated_trade.side == TradeSide.LONG else -1
                raw_diff = (updated_trade.exit_price - updated_trade.entry_price) * side_mult
                updated_trade.pnl_raw = raw_diff * updated_trade.qty
                updated_trade.pnl_pct = (raw_diff / updated_trade.entry_price) * 100
                
                updated_trade.fees = (updated_trade.entry_price + updated_trade.exit_price) * updated_trade.qty * config.commission
                equity += updated_trade.pnl_raw - updated_trade.fees
                trades.append(updated_trade)
                active_trade = None

        # 2. Tìm điểm vào lệnh mới dựa trên tín hiệu của nến TRƯỚC
        if not active_trade:
            if prev_row['long_trigger']:
                entry_p = curr_row['open'] # Khớp ở OPEN của nến hiện tại
                signal_atr = prev_row['atr'] # ATR phải dùng của nến ra signal
                sl_p = round_to_tick(entry_p - config.sl_atr_mult * signal_atr, config.min_tick)
                tp_p = round_to_tick(entry_p * (1 + config.tp_perc), config.min_tick)
                pos_qty = (equity * config.qty_pct) / entry_p
                
                active_trade = Trade(config.symbol, TradeSide.LONG, prev_row['timestamp'], entry_p, pos_qty, sl_p, tp_p)
                active_trade.atr_at_entry = signal_atr
                active_trade.state = TradeState.ACTIVE
                active_trade.entry_time = curr_row['timestamp']
                
            elif prev_row['short_trigger']:
                entry_p = curr_row['open']
                signal_atr = prev_row['atr']
                sl_p = round_to_tick(entry_p + config.sl_atr_mult * signal_atr, config.min_tick)
                tp_p = round_to_tick(entry_p * (1 - config.tp_perc), config.min_tick)
                pos_qty = (equity * config.qty_pct) / entry_p
                
                active_trade = Trade(config.symbol, TradeSide.SHORT, prev_row['timestamp'], entry_p, pos_qty, sl_p, tp_p)
                active_trade.atr_at_entry = signal_atr
                active_trade.state = TradeState.ACTIVE
                active_trade.entry_time = curr_row['timestamp']
        
        equity_curve.append(equity)
        
    return trades, equity, equity_curve

def calculate_metrics(trades, equity_curve):
    if not trades:
        return {}
    
    returns = pd.Series(equity_curve).pct_change().dropna()
    total_returns = (equity_curve[-1] / equity_curve[0]) - 1
    
    # Sharpe Ratio (Hourly to Annualized)
    sharpe = (returns.mean() / returns.std() * np.sqrt(252 * 24)) if returns.std() != 0 else 0
    
    # Profit Factor
    gross_profits = sum(t.pnl_raw for t in trades if t.pnl_raw > 0)
    gross_losses = abs(sum(t.pnl_raw for t in trades if t.pnl_raw < 0))
    profit_factor = gross_profits / gross_losses if gross_losses != 0 else np.inf
    
    # Max Drawdown
    equity_series = pd.Series(equity_curve)
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    max_dd = drawdown.min()
    
    win_count = len([t for t in trades if t.pnl_raw > 0])
    win_rate = win_count / len(trades) * 100
    
    return {
        "Total Trades": len(trades),
        "Win Rate": f"{win_rate:.2f}%",
        "Net Profit %": f"{total_returns*100:.2f}%",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Profit Factor": f"{profit_factor:.2f}",
        "Max Drawdown": f"{max_dd*100:.2f}%",
        "Final Equity": f"${equity_curve[-1]:.2f}"
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="SOLUSDT")
    parser.add_argument("--tick", type=float, default=0.01)
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--timeframe", type=str, default="1h", choices=["1h", "4h", "8h", "12h", "1d"], help="Timeframe to resample to")
    args = parser.parse_args()
    
    data_path = Path("data/ohlcv") / f"{args.symbol}_USDT.parquet"
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return
        
    df = pd.read_parquet(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Date Filtering
    if args.start:
        start_ts = pd.to_datetime(args.start).tz_localize(df['timestamp'].dt.tz)
        df = df[df['timestamp'] >= start_ts]
    if args.end:
        end_ts = pd.to_datetime(args.end).tz_localize(df['timestamp'].dt.tz)
        df = df[df['timestamp'] <= end_ts]
    
    df = df.reset_index(drop=True)
    
    # Resampling
    if args.timeframe != "1h":
        df = resample_data(df, args.timeframe)
        
    if len(df) == 0:
        print("Error: No data in specified range.")
        return
        
    print(f"Loaded {len(df)} candles for {args.symbol} from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    
    config = BacktestConfig(
        symbol=args.symbol,
        min_tick=args.tick
    )
    
    trades, final_equity, equity_curve = run_backtest_event_driven(df, config)
    metrics = calculate_metrics(trades, equity_curve)
    
    print("\n" + "="*40)
    print(f"BACKTEST V2 FINAL RESULTS: {args.symbol}")
    print("="*40)
    for k, v in metrics.items():
        print(f"{k:15}: {v}")
        
    if trades:
        results_df = pd.DataFrame([vars(t) for t in trades])
        # Convert state enum to name for CSV
        results_df['state'] = results_df['state'].apply(lambda x: x.value)
        results_df['side'] = results_df['side'].apply(lambda x: x.value)
        results_df.to_csv(f"backtest_tmfs_v2_final_{args.symbol}.csv", index=False)
        print(f"\nTrades saved to backtest_tmfs_v2_final_{args.symbol}.csv")
    else:
        print("No trades executed.")

if __name__ == "__main__":
    main()

