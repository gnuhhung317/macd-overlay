#!/usr/bin/env python3
"""
Backtester for MACD Crossover Strategy
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent.parent / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
RESULTS_DIR = Path(__file__).parent / 'results'


@dataclass
class Trade:
    """Represents a single trade"""
    symbol: str
    entry_time: datetime
    entry_price: float
    direction: str  # 'LONG' or 'SHORT'
    tp_price: float
    sl_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ''
    pnl: float = 0.0
    pnl_pct: float = 0.0
    bars_held: int = 0


@dataclass
class BacktestResult:
    """Results from backtest"""
    trades: List[Trade] = field(default_factory=list)
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_bars_held: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'total_pnl': self.total_pnl,
            'avg_pnl': self.avg_pnl,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'profit_factor': self.profit_factor,
            'avg_bars_held': self.avg_bars_held
        }


class MACDBacktester:
    """Backtest MACD Crossover strategy"""
    
    def __init__(
        self,
        tp_pct: float = 0.03,      # Take profit %
        sl_pct: float = 0.015,     # Stop loss %
        max_bars: int = 10,         # Max bars to hold
        commission: float = 0.0004, # 0.04% per trade
        trade_long: bool = True,
        trade_short: bool = True
    ):
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.max_bars = max_bars
        self.commission = commission
        self.trade_long = trade_long
        self.trade_short = trade_short
    
    def run(self, df: pd.DataFrame, symbol: str = 'UNKNOWN') -> BacktestResult:
        """Run backtest on a single symbol"""
        trades = []
        
        for i in range(len(df) - self.max_bars - 1):
            row = df.iloc[i]
            
            # Check for entry signal
            is_long_signal = row.get('macd_cross_up', 0) == 1 and self.trade_long
            is_short_signal = row.get('macd_cross_down', 0) == 1 and self.trade_short
            
            if not is_long_signal and not is_short_signal:
                continue
            
            # Entry
            entry_price = row['close']
            entry_time = row['timestamp']
            direction = 'LONG' if is_long_signal else 'SHORT'
            
            # Calculate TP/SL
            if direction == 'LONG':
                tp_price = entry_price * (1 + self.tp_pct)
                sl_price = entry_price * (1 - self.sl_pct)
            else:
                tp_price = entry_price * (1 - self.tp_pct)
                sl_price = entry_price * (1 + self.sl_pct)
            
            trade = Trade(
                symbol=symbol,
                entry_time=entry_time,
                entry_price=entry_price,
                direction=direction,
                tp_price=tp_price,
                sl_price=sl_price
            )
            
            # Simulate trade
            trade = self._simulate_trade(df, i, trade)
            trades.append(trade)
        
        return self._calculate_results(trades)
    
    def _simulate_trade(self, df: pd.DataFrame, entry_idx: int, trade: Trade) -> Trade:
        """Simulate a single trade"""
        for j in range(1, self.max_bars + 1):
            if entry_idx + j >= len(df):
                break
            
            bar = df.iloc[entry_idx + j]
            
            if trade.direction == 'LONG':
                # Check SL first (worst case)
                if bar['low'] <= trade.sl_price:
                    trade.exit_price = trade.sl_price
                    trade.exit_reason = 'SL'
                    trade.exit_time = bar['timestamp']
                    trade.bars_held = j
                    break
                # Check TP
                if bar['high'] >= trade.tp_price:
                    trade.exit_price = trade.tp_price
                    trade.exit_reason = 'TP'
                    trade.exit_time = bar['timestamp']
                    trade.bars_held = j
                    break
            else:  # SHORT
                # Check SL first
                if bar['high'] >= trade.sl_price:
                    trade.exit_price = trade.sl_price
                    trade.exit_reason = 'SL'
                    trade.exit_time = bar['timestamp']
                    trade.bars_held = j
                    break
                # Check TP
                if bar['low'] <= trade.tp_price:
                    trade.exit_price = trade.tp_price
                    trade.exit_reason = 'TP'
                    trade.exit_time = bar['timestamp']
                    trade.bars_held = j
                    break
        
        # Timeout - exit at close
        if trade.exit_price is None:
            last_bar = df.iloc[min(entry_idx + self.max_bars, len(df) - 1)]
            trade.exit_price = last_bar['close']
            trade.exit_reason = 'TIMEOUT'
            trade.exit_time = last_bar['timestamp']
            trade.bars_held = self.max_bars
        
        # Calculate PnL
        if trade.direction == 'LONG':
            trade.pnl_pct = (trade.exit_price - trade.entry_price) / trade.entry_price
        else:
            trade.pnl_pct = (trade.entry_price - trade.exit_price) / trade.entry_price
        
        # Subtract commission (entry + exit)
        trade.pnl_pct -= 2 * self.commission
        trade.pnl = trade.pnl_pct * trade.entry_price
        
        return trade
    
    def _calculate_results(self, trades: List[Trade]) -> BacktestResult:
        """Calculate backtest metrics"""
        result = BacktestResult(trades=trades)
        
        if not trades:
            return result
        
        result.total_trades = len(trades)
        result.winning_trades = sum(1 for t in trades if t.pnl_pct > 0)
        result.losing_trades = sum(1 for t in trades if t.pnl_pct <= 0)
        result.win_rate = result.winning_trades / result.total_trades
        
        pnls = [t.pnl_pct for t in trades]
        result.total_pnl = sum(pnls)
        result.avg_pnl = np.mean(pnls)
        result.avg_bars_held = np.mean([t.bars_held for t in trades])
        
        # Max Drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        result.max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        # Sharpe Ratio (annualized, assuming daily)
        if np.std(pnls) > 0:
            result.sharpe_ratio = np.mean(pnls) / np.std(pnls) * np.sqrt(252)
        
        # Profit Factor
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        if gross_loss > 0:
            result.profit_factor = gross_profit / gross_loss
        
        return result


def run_full_backtest(
    df: pd.DataFrame,
    tp_range: List[float] = None,
    sl_range: List[float] = None
) -> pd.DataFrame:
    """Run backtest with multiple TP/SL combinations"""
    
    if tp_range is None:
        tp_range = [0.01, 0.02, 0.03, 0.04, 0.05]
    if sl_range is None:
        sl_range = [0.005, 0.01, 0.015, 0.02, 0.025]
    
    results = []
    symbols = df['symbol'].unique()
    
    for tp in tp_range:
        for sl in sl_range:
            print(f"Testing TP={tp:.1%}, SL={sl:.1%}...")
            
            backtester = MACDBacktester(tp_pct=tp, sl_pct=sl)
            
            all_trades = []
            for symbol in symbols:
                symbol_df = df[df['symbol'] == symbol].copy()
                if len(symbol_df) < 100:
                    continue
                
                result = backtester.run(symbol_df, symbol)
                all_trades.extend(result.trades)
            
            # Aggregate results
            if all_trades:
                agg_result = backtester._calculate_results(all_trades)
                
                results.append({
                    'tp_pct': tp,
                    'sl_pct': sl,
                    'risk_reward': tp / sl,
                    **agg_result.to_dict()
                })
    
    return pd.DataFrame(results)


def analyze_by_regime(df: pd.DataFrame, backtester: MACDBacktester) -> Dict:
    """Analyze performance by market regime"""
    results = {}
    
    # Trending vs Ranging
    for regime in ['trending', 'ranging']:
        if regime == 'trending':
            regime_df = df[df['is_trending'] == 1]
        else:
            regime_df = df[df['is_trending'] == 0]
        
        if len(regime_df) > 0:
            all_trades = []
            for symbol in regime_df['symbol'].unique():
                symbol_df = regime_df[regime_df['symbol'] == symbol].copy()
                result = backtester.run(symbol_df, symbol)
                all_trades.extend(result.trades)
            
            if all_trades:
                results[regime] = backtester._calculate_results(all_trades).to_dict()
    
    # High vs Low Volatility
    for vol_regime in ['high_vol', 'low_vol']:
        if vol_regime == 'high_vol':
            regime_df = df[df['is_volatile'] == 1]
        else:
            regime_df = df[df['is_volatile'] == 0]
        
        if len(regime_df) > 0:
            all_trades = []
            for symbol in regime_df['symbol'].unique():
                symbol_df = regime_df[regime_df['symbol'] == symbol].copy()
                result = backtester.run(symbol_df, symbol)
                all_trades.extend(result.trades)
            
            if all_trades:
                results[vol_regime] = backtester._calculate_results(all_trades).to_dict()
    
    return results


if __name__ == '__main__':
    print("="*60)
    print("MACD Crossover Backtester")
    print("="*60)
    
    # Load processed data
    data_path = PROCESSED_DIR / 'features_1d_test.parquet'
    if not data_path.exists():
        print(f"Data not found: {data_path}")
        print("Run data_pipeline.py first!")
        exit(1)
    
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} rows, {df['symbol'].nunique()} symbols")
    
    # Quick single test
    print("\n" + "-"*60)
    print("Single Configuration Test")
    print("-"*60)
    
    backtester = MACDBacktester(tp_pct=0.03, sl_pct=0.015, max_bars=10)
    
    all_trades = []
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()
        result = backtester.run(symbol_df, symbol)
        all_trades.extend(result.trades)
    
    final_result = backtester._calculate_results(all_trades)
    
    print(f"Total Trades: {final_result.total_trades}")
    print(f"Win Rate: {final_result.win_rate:.2%}")
    print(f"Total PnL: {final_result.total_pnl:.2%}")
    print(f"Avg PnL: {final_result.avg_pnl:.4%}")
    print(f"Sharpe Ratio: {final_result.sharpe_ratio:.2f}")
    print(f"Profit Factor: {final_result.profit_factor:.2f}")
    print(f"Max Drawdown: {final_result.max_drawdown:.2%}")
    print(f"Avg Bars Held: {final_result.avg_bars_held:.1f}")
    
    # By exit reason
    exit_reasons = pd.Series([t.exit_reason for t in all_trades]).value_counts()
    print(f"\nExit Reasons:")
    for reason, count in exit_reasons.items():
        pct = count / len(all_trades) * 100
        print(f"  {reason}: {count} ({pct:.1f}%)")
    
    # Grid search
    print("\n" + "-"*60)
    print("TP/SL Optimization")
    print("-"*60)
    
    grid_results = run_full_backtest(df)
    
    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    grid_results.to_csv(RESULTS_DIR / 'tp_sl_optimization.csv', index=False)
    
    # Best configuration
    best_idx = grid_results['sharpe_ratio'].idxmax()
    best = grid_results.iloc[best_idx]
    
    print(f"\nBest Configuration (by Sharpe):")
    print(f"  TP: {best['tp_pct']:.1%}")
    print(f"  SL: {best['sl_pct']:.1%}")
    print(f"  Risk/Reward: {best['risk_reward']:.2f}")
    print(f"  Win Rate: {best['win_rate']:.2%}")
    print(f"  Sharpe: {best['sharpe_ratio']:.2f}")
    print(f"  Profit Factor: {best['profit_factor']:.2f}")
    
    # Regime analysis
    print("\n" + "-"*60)
    print("Performance by Market Regime")
    print("-"*60)
    
    regime_results = analyze_by_regime(df, backtester)
    for regime, metrics in regime_results.items():
        print(f"\n{regime.upper()}:")
        print(f"  Trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate']:.2%}")
        print(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
