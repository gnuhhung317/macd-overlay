import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime, timezone
import argparse
from typing import List, Dict, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
import csv

# Shared basic classes
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
    qty_pct: float = 1.0
    commission: float = 0.0003
    min_tick: float = 0.01
    symbol: str = "SOLUSDT"
    sl_atr_mult: float = 3.0
    tp_perc: float = 0.02
    trailing_sl: bool = True
    be_after_tp: bool = True
    trailing_tp_enabled: bool = True
    deviation_perc: float = 0.005  # Nới lỏng thành 0.5% để chặn nhiễu giá (market noise)
    slippage_perc: float = 0.0005 # Cấu hình trượt giá 0.05%
    
    # Strat params
    bb_period: int = 20
    bb_std: float = 2.0
    rsi_period: int = 14
    rsi_lower: float = 30
    rsi_upper: float = 70
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

def round_to_tick(price, tick):
    return np.round(price / tick) * tick

class BaseStrategy:
    def calculate_indicators(self, df, config):
        pass
    def calculate_signals(self, df, config):
        pass

def calculate_dmi(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    tr_s = wilders_smoothing(tr, period)
    plus_dm_s = wilders_smoothing(pd.Series(plus_dm), period)
    minus_dm_s = wilders_smoothing(pd.Series(minus_dm), period)
    
    di_plus = (plus_dm_s / tr_s) * 100
    di_minus = (minus_dm_s / tr_s) * 100
    
    dx = (abs(di_plus - di_minus) / (di_plus + di_minus)) * 100
    adx = wilders_smoothing(dx, period)
    
    return adx

class DonchianBreakoutStrategy(BaseStrategy):
    def calculate_indicators(self, df, config):
        df['atr'] = calculate_atr(df, config.atr_len)
        df['dc_upper'] = df['high'].rolling(config.bb_period).max().shift(1)
        df['dc_lower'] = df['low'].rolling(config.bb_period).min().shift(1)
        exit_period = config.bb_period // 2
        df['dc_exit_long'] = df['low'].rolling(exit_period).min().shift(1)
        df['dc_exit_short'] = df['high'].rolling(exit_period).max().shift(1)
        return df

    def calculate_signals(self, df, config):
        df['long_trigger'] = df['high'] > df['dc_upper']
        df['short_trigger'] = df['low'] < df['dc_lower']
        df['exit_signal_long'] = df['low'] < df['dc_exit_long']
        df['exit_signal_short'] = df['high'] > df['dc_exit_short']
        return df

class DonchianAdxStrategy(BaseStrategy):
    def calculate_indicators(self, df, config):
        df['atr'] = calculate_atr(df, config.atr_len)
        df['adx'] = calculate_dmi(df, 14)
        
        # Donchian Channels
        df['dc_upper'] = df['high'].rolling(config.bb_period).max().shift(1)
        df['dc_lower'] = df['low'].rolling(config.bb_period).min().shift(1)
        
        # Exit channels (shorter period)
        exit_period = config.bb_period // 2
        df['dc_exit_long'] = df['low'].rolling(exit_period).min().shift(1)
        df['dc_exit_short'] = df['high'].rolling(exit_period).max().shift(1)
        
        return df

    def calculate_signals(self, df, config):
        # Entry: Price goes above highest high of last N periods AND ADX > 25
        df['long_trigger'] = (df['high'] > df['dc_upper']) & (df['adx'] > 25)
        df['short_trigger'] = (df['low'] < df['dc_lower']) & (df['adx'] > 25)
        
        # Exit: Price goes below N/2 lowest low
        df['exit_signal_long'] = df['low'] < df['dc_exit_long']
        df['exit_signal_short'] = df['high'] > df['dc_exit_short']
        return df

class VolatilitySqueezeStrategy(BaseStrategy):
    def calculate_indicators(self, df, config):
        length = config.bb_period
        df['atr'] = calculate_atr(df, config.atr_len)
        
        # Bollinger Bands
        df['sma'] = df['close'].rolling(length).mean()
        std = df['close'].rolling(length).std()
        df['bb_upper'] = df['sma'] + (config.bb_std * std)
        df['bb_lower'] = df['sma'] - (config.bb_std * std)
        
        # Keltner Channels (using 1.5 multiplier for standard squeeze)
        df['ema'] = df['close'].ewm(span=length, adjust=False).mean()
        df['kc_upper'] = df['ema'] + (1.5 * df['atr'])
        df['kc_lower'] = df['ema'] - (1.5 * df['atr'])
        
        # Squeeze condition (BB inside KC)
        df['squeeze'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])
        
        # Momentum (Price vs EMA to determine breakout direction)
        df['mom_bull'] = df['close'] > df['ema']
        df['mom_bear'] = df['close'] < df['ema']
        
        # IMPORTANT: shift indicators to avoid lookahead bias
        for col in ['sma', 'ema', 'bb_upper', 'bb_lower', 'kc_upper', 'kc_lower', 'squeeze', 'mom_bull', 'mom_bear', 'atr']:
            df[col] = df[col].shift(1)
            
        return df

    def calculate_signals(self, df, config):
        # Squeeze firing: Squeeze is false NOW, but was true in previous bar
        squeeze_prev = df['squeeze'].shift(1).fillna(0).astype(bool)
        df_squeeze_bool = df['squeeze'].fillna(0).astype(bool)
        df['squeeze_release'] = (~df_squeeze_bool) & squeeze_prev
        
        # Trigger on the release of the squeeze in the direction of momentum
        df['long_trigger'] = df['squeeze_release'] & df['mom_bull'].fillna(False).astype(bool)
        df['short_trigger'] = df['squeeze_release'] & df['mom_bear'].fillna(False).astype(bool)
        
        # Exit: Reversion back to the mean (SMA)
        df['exit_signal_long'] = df['close'] < df['sma']
        df['exit_signal_short'] = df['close'] > df['sma']
        
        return df

def simulate_trade_step(trade: Trade, row: pd.Series, config: BacktestConfig, prev_row: pd.Series):
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

        # LƯU Ý 1: Dùng Mốc Exit từ Strategy (Nới rộng để Strategy linh hoạt)
        val_exit_long = prev_row.get('dc_exit_long', None)
        val_exit_short = prev_row.get('dc_exit_short', None)
        
        # Nếu strategy custom exit thì dùng nó, lấy mặc định là curr_close nếu thiếu để tránh Lookahead Bias
        if val_exit_long is None or pd.isna(val_exit_long): 
            val_exit_long = curr_close
        if val_exit_short is None or pd.isna(val_exit_short):
            val_exit_short = curr_close

        # 1. Đánh giá Signal Exit (Khớp lệnh ngay tại mốc bị phá vỡ)
        if trade.side == TradeSide.LONG and row['exit_signal_long']:
            exit_triggered = True
            exit_price = val_exit_long
            exit_reason = "Signal Exit"
            
        elif trade.side == TradeSide.SHORT and row['exit_signal_short']:
            exit_triggered = True
            exit_price = val_exit_short
            exit_reason = "Signal Exit"

        # 2. Đánh giá SL / TP (Dùng trade.sl và trade.tp CŨ, đã chốt từ nến trước)
        if not exit_triggered:
            sl_hit = False
            tp_hit = False

            if trade.side == TradeSide.LONG:
                if curr_low <= trade.sl: sl_hit = True
                if curr_high >= trade.tp: tp_hit = True
            else:
                if curr_high >= trade.sl: sl_hit = True
                if curr_low <= trade.tp: tp_hit = True

            if sl_hit and tp_hit:
                is_green = curr_close > curr_open
                if (trade.side == TradeSide.LONG and is_green) or (trade.side == TradeSide.SHORT and not is_green):
                    exit_triggered = True
                    exit_price = curr_open if (trade.side == TradeSide.LONG and curr_open < trade.sl) else trade.sl
                    if trade.side == TradeSide.SHORT and curr_open > trade.sl: exit_price = curr_open
                    exit_reason = "SL Triggered (Conflict)"
                else:
                    trade.tp_hit = True # Cập nhật trạng thái chạm TP
                    exit_triggered = True
                    exit_price = curr_open if (trade.side == TradeSide.LONG and curr_open > trade.tp) else trade.tp
                    if trade.side == TradeSide.SHORT and curr_open < trade.tp: exit_price = curr_open
                    exit_reason = "TP Hit (Conflict)"
                    
            elif sl_hit:
                exit_triggered = True
                if trade.side == TradeSide.LONG:
                    exit_price = curr_open if curr_open < trade.sl else trade.sl
                else:
                    exit_price = curr_open if curr_open > trade.sl else trade.sl
                exit_reason = "SL Triggered"
                
            elif tp_hit:
                trade.tp_hit = True
                if not config.trailing_tp_enabled:
                    exit_triggered = True
                    exit_price = curr_open if trade.side == TradeSide.LONG and curr_open > trade.tp else trade.tp
                    if trade.side == TradeSide.SHORT: exit_price = curr_open if curr_open < trade.tp else trade.tp
                    exit_reason = "TP Target Hit"

        # ĐÓNG LỆNH
        if exit_triggered:
            trade.state = TradeState.CLOSED
            trade.exit_price = exit_price
            trade.exit_reason = exit_reason
            trade.exit_time = row['timestamp']
            return True, trade

        # 3. LƯU Ý 2: Lệnh SỐNG SÓT qua cây nến này rồi mới được phép cập nhật Trailing SL cho cây nến TỚI
        if config.trailing_sl:
            if trade.side == TradeSide.LONG:
                new_sl = curr_high - config.sl_atr_mult * trade.atr_at_entry
                new_sl = round_to_tick(new_sl, config.min_tick)
                if config.be_after_tp and trade.tp_hit:
                    new_sl = max(new_sl, trade.entry_price)
                trade.sl = max(trade.sl, new_sl)
            else: 
                new_sl = curr_low + config.sl_atr_mult * trade.atr_at_entry
                new_sl = round_to_tick(new_sl, config.min_tick)
                if config.be_after_tp and trade.tp_hit:
                    new_sl = min(new_sl, trade.entry_price)
                trade.sl = min(trade.sl, new_sl)

        # Cập nhật Trailing TP nếu trade.tp_hit
        if config.trailing_tp_enabled and trade.tp_hit:
            if trade.side == TradeSide.LONG:
                trail_sl = curr_high - round_to_tick(curr_high * config.deviation_perc, config.min_tick)
                trade.sl = max(trade.sl, trail_sl)
            else:
                trail_sl = curr_low + round_to_tick(curr_low * config.deviation_perc, config.min_tick)
                trade.sl = min(trade.sl, trail_sl)

    return False, trade

def run_backtest_event_driven(df, config):
    equity = config.initial_capital
    active_trade: Optional[Trade] = None
    trades: List[Trade] = []
    equity_curve = [equity]
    
    for i in range(1, len(df)):
        curr_row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        current_step_equity = equity
        
        if active_trade:
            closed, updated_trade = simulate_trade_step(active_trade, curr_row, config, prev_row)
            if closed:
                # Áp dụng trượt giá cho đầu thoát lệnh (Exit Slippage)
                if updated_trade.side == TradeSide.LONG:
                    updated_trade.exit_price = updated_trade.exit_price * (1 - config.slippage_perc)
                    side_mult = 1
                else:
                    updated_trade.exit_price = updated_trade.exit_price * (1 + config.slippage_perc)
                    side_mult = -1
                
                raw_diff = (updated_trade.exit_price - updated_trade.entry_price) * side_mult
                updated_trade.pnl_raw = raw_diff * updated_trade.qty
                updated_trade.pnl_pct = (raw_diff / updated_trade.entry_price) * 100
                
                updated_trade.fees = (updated_trade.entry_price + updated_trade.exit_price) * updated_trade.qty * config.commission
                equity += updated_trade.pnl_raw - updated_trade.fees
                trades.append(updated_trade)
                
                current_step_equity = equity
                active_trade = None
            else:
                # TÍNH FLOATING PNL KHI LỆNH ĐANG MỞ (MARK-TO-MARKET)
                side_mult = 1 if active_trade.side == TradeSide.LONG else -1
                floating_pnl = (curr_row['close'] - active_trade.entry_price) * side_mult * active_trade.qty
                current_step_equity = equity + floating_pnl

        if not active_trade:
            if prev_row['long_trigger']:
                # Áp dụng trượt giá cho đầu vào (Entry Slippage)
                entry_p = curr_row['open'] * (1 + config.slippage_perc)
                if pd.isna(prev_row['atr']): continue
                signal_atr = prev_row['atr']
                sl_p = round_to_tick(entry_p - config.sl_atr_mult * signal_atr, config.min_tick)
                tp_p = round_to_tick(entry_p * (1 + config.tp_perc), config.min_tick)
                pos_qty = (equity * config.qty_pct) / entry_p
                
                active_trade = Trade(config.symbol, TradeSide.LONG, prev_row['timestamp'], entry_p, pos_qty, sl_p, tp_p)
                active_trade.atr_at_entry = signal_atr
                active_trade.state = TradeState.ACTIVE
                active_trade.entry_time = curr_row['timestamp']
                
            elif prev_row['short_trigger']:
                # Áp dụng trượt giá cho đầu vào (Entry Slippage)
                entry_p = curr_row['open'] * (1 - config.slippage_perc)
                if pd.isna(prev_row['atr']): continue
                signal_atr = prev_row['atr']
                sl_p = round_to_tick(entry_p + config.sl_atr_mult * signal_atr, config.min_tick)
                tp_p = round_to_tick(entry_p * (1 - config.tp_perc), config.min_tick)
                pos_qty = (equity * config.qty_pct) / entry_p
                
                active_trade = Trade(config.symbol, TradeSide.SHORT, prev_row['timestamp'], entry_p, pos_qty, sl_p, tp_p)
                active_trade.atr_at_entry = signal_atr
                active_trade.state = TradeState.ACTIVE
                active_trade.entry_time = curr_row['timestamp']
        
        equity_curve.append(current_step_equity)
        
    return trades, equity, equity_curve

def calculate_metrics(trades, equity_curve):
    if not trades or len(equity_curve) <= 1:
        return {"Sharpe Ratio": 0, "Max Drawdown": 0, "Net Profit %": 0}
    
    returns = pd.Series(equity_curve).pct_change().dropna()
    total_returns = (equity_curve[-1] / equity_curve[0]) - 1
    
    sharpe = (returns.mean() / returns.std() * np.sqrt(252 * 24)) if returns.std() != 0 else 0
    
    equity_series = pd.Series(equity_curve)
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    max_dd = drawdown.min()
    
    return {
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Net Profit %": total_returns
    }

def log_experiment(strat_name, is_res, oos_res, params):
    file_exists = os.path.isfile("experiments_log.csv")
    with open("experiments_log.csv", mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Strategy", "IS Sharpe", "OOS Sharpe", "OOS Max DD", "Params"])
        writer.writerow([
            strat_name, 
            round(is_res['Sharpe Ratio'], 2), 
            round(oos_res['Sharpe Ratio'], 2), 
            round(oos_res['Max Drawdown']*100, 2), 
            str(params)
        ])

class BBRSIStrategy(BaseStrategy):
    def calculate_indicators(self, df, config):
        length = config.bb_period
        
        df['sma'] = df['close'].rolling(length).mean()
        std = df['close'].rolling(length).std()
        df['bb_upper'] = df['sma'] + (2.5 * std)  
        df['bb_lower'] = df['sma'] - (2.5 * std)  
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        df['atr'] = calculate_atr(df, config.atr_len if hasattr(config, 'atr_len') else 14)
        
        for col in ['sma', 'bb_upper', 'bb_lower', 'rsi', 'atr']:
            df[col] = df[col].shift(1)
            
        return df

    def calculate_signals(self, df, config):
        df['long_trigger'] = (df['close'] < df['bb_lower']) & (df['rsi'] < 30)
        df['short_trigger'] = (df['close'] > df['bb_upper']) & (df['rsi'] > 70)
        
        df['exit_signal_long'] = (df['close'] > df['sma'])
        df['exit_signal_short'] = (df['close'] < df['sma'])
        return df

class MacdEmaStrategy(BaseStrategy):
    def calculate_indicators(self, df, config):
        df['ema_fast'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=200, adjust=False).mean()
        
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        df['atr'] = calculate_atr(df, config.atr_len if hasattr(config, 'atr_len') else 14)
        
        for col in ['ema_fast', 'ema_slow', 'macd', 'macd_signal', 'macd_hist', 'atr']:
            df[col] = df[col].shift(1)
            
        return df

    def calculate_signals(self, df, config):
        macd_cross_up = (df['macd_hist'] > 0) & (df['macd_hist'].shift(1) <= 0)
        macd_cross_down = (df['macd_hist'] < 0) & (df['macd_hist'].shift(1) >= 0)
        
        df['long_trigger'] = (df['close'] > df['ema_slow']) & macd_cross_up & (df['macd'] < 0)
        df['short_trigger'] = (df['close'] < df['ema_slow']) & macd_cross_down & (df['macd'] > 0)
        
        df['exit_signal_long'] = macd_cross_down
        df['exit_signal_short'] = macd_cross_up
        return df

class EMA_ADX_Strategy:
    def calculate_indicators(self, df: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
        df['ema_fast'] = df['close'].ewm(span=9).mean().shift(1)
        df['ema_slow'] = df['close'].ewm(span=21).mean().shift(1)
        df['ema_trend'] = df['close'].ewm(span=200).mean().shift(1)
        df['up_move'] = df['high'] - df['high'].shift(1)
        df['down_move'] = df['low'].shift(1) - df['low']
        df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = (df['high'] - df['close'].shift(1)).abs()
        df['tr3'] = (df['low'] - df['close'].shift(1)).abs()
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean().shift(1)
        atr14 = df['atr'] * 14
        plus_di = 100 * (df['plus_dm'].rolling(14).sum() / (atr14+1e-9))
        minus_di = 100 * (df['minus_dm'].rolling(14).sum() / (atr14+1e-9))
        df['dx'] = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
        df['adx'] = df['dx'].rolling(14).mean().shift(1)
        return df

    def calculate_signals(self, df: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
        cross_up = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
        cross_dn = (df['ema_fast'] < df['ema_slow']) & (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1))
        trend_up = df['close'] > df['ema_trend']
        trend_dn = df['close'] < df['ema_trend']
        trending = df['adx'] > 25
        df['long_trigger'] = cross_up & trend_up & trending
        df['short_trigger'] = cross_dn & trend_dn & trending
        df['exit_signal_long'] = cross_dn
        df['exit_signal_short'] = cross_up
        return df

if __name__ == "__main__":
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    
    print("Testing Volatility Squeeze Strategy (Finding True Edge)...")
    
    portfolio_is_sharpe = []
    portfolio_oos_sharpe = []
    
    for symbol in symbols:
        data_path = Path("data/ohlcv") / f"{symbol}_USDT.parquet"
        if not data_path.exists():
            data_path = Path("data/ohlcv") / f"{symbol.replace('USDT', '')}_USDT.parquet"
        if not data_path.exists():
            data_path = Path("data/ohlcv") / f"{symbol}.parquet"

        if not data_path.exists():
            print(f"[{symbol}] Data not found. Skipping.")
            continue
            
        df = pd.read_parquet(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values("timestamp").reset_index(drop=True)
        
# Strategy logic params
        config = BacktestConfig(symbol=symbol, bb_period=20, sl_atr_mult=2.5, trailing_sl=True, trailing_tp_enabled=True)
        strategy = EMA_ADX_Strategy()

        df_copy = df.copy()
        df_copy = strategy.calculate_indicators(df_copy, config)
        df_copy = strategy.calculate_signals(df_copy, config)

        split_idx = int(len(df_copy) * 0.7)
        df_is = df_copy.iloc[:split_idx].copy().reset_index(drop=True)
        df_oos = df_copy.iloc[split_idx:].copy().reset_index(drop=True)

        print(f"\n--- {symbol} ---")
        t_is, eq_is, curve_is = run_backtest_event_driven(df_is, config)
        metrics_is = calculate_metrics(t_is, curve_is)
        print(f"IS  -> Sharpe: {metrics_is['Sharpe Ratio']:.2f}, DD: {metrics_is['Max Drawdown']*100:.2f}%") 
        portfolio_is_sharpe.append(metrics_is['Sharpe Ratio'])

        t_oos, eq_oos, curve_oos = run_backtest_event_driven(df_oos, config)
        metrics_oos = calculate_metrics(t_oos, curve_oos)
        print(f"OOS -> Sharpe: {metrics_oos['Sharpe Ratio']:.2f}, DD: {metrics_oos['Max Drawdown']*100:.2f}%")
        portfolio_oos_sharpe.append(metrics_oos['Sharpe Ratio'])

        param_str = "BBRSI: bb=20/2.5, rsi=14/30/70, sl=3.0"
        log_experiment(f"EMA_ADX_{symbol}", metrics_is, metrics_oos, param_str)
    print("\n==================================")
    print(f"AVERAGE IS SHARPE : {np.mean(portfolio_is_sharpe):.2f}")
    if portfolio_oos_sharpe:
        print(f"AVERAGE OOS SHARPE: {np.mean(portfolio_oos_sharpe):.2f}")
    print("==================================")
