#!/usr/bin/env python3
"""
Backtest Comparison: ML-Filtered Fixed TP/SL vs Trailing Stop Loss

Compares two exit strategies:
1. ML Filter + Fixed TP/SL: Use ML model to filter entries, fixed TP/SL for exits
2. Trailing Stop Loss: Dynamic stop loss that follows price
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import joblib
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent.parent / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
MODEL_DIR = Path(__file__).parent / 'models'
RESULTS_DIR = Path(__file__).parent / 'results'


@dataclass
class Trade:
    """Represents a single trade"""
    symbol: str
    entry_time: pd.Timestamp
    entry_price: float
    direction: str  # 'LONG' or 'SHORT'
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: str = ''
    pnl_pct: float = 0.0
    bars_held: int = 0
    ml_confidence: float = 0.0
    highest_price: float = 0.0  # For trailing SL tracking
    lowest_price: float = 0.0


@dataclass
class BacktestResult:
    """Aggregated backtest results"""
    method: str = ''
    symbol: str = ''
    total_trades: int = 0
    winning_trades: int = 0
    win_rate: float = 0.0
    total_pnl_pct: float = 0.0
    avg_pnl_pct: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_bars_held: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'method': self.method,
            'symbol': self.symbol,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.win_rate,
            'total_pnl_pct': self.total_pnl_pct,
            'avg_pnl_pct': self.avg_pnl_pct,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'profit_factor': self.profit_factor,
            'avg_bars_held': self.avg_bars_held
        }


class MLEntryFilter:
    """Load and use ML model for entry filtering"""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        if model_path is None:
            model_path = MODEL_DIR / 'entry_filter.joblib'
        
        if Path(model_path).exists():
            self._load_model(model_path)
            print(f"✓ Loaded ML model from {model_path}")
        else:
            print(f"⚠️ ML model not found: {model_path}")
    
    def _load_model(self, path: str):
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data.get('scaler')
        self.feature_names = data['feature_names']
    
    def predict(self, row: pd.Series) -> Tuple[bool, float]:
        """
        Predict if entry should be taken.
        Returns: (should_enter, confidence)
        """
        if self.model is None:
            return True, 0.5
        
        # Extract features
        features = {}
        for fname in self.feature_names:
            if fname in row.index:
                features[fname] = row[fname]
            elif fname == 'is_bullish_cross':
                features[fname] = row.get('macd_cross_up', 0)
            else:
                features[fname] = 0
        
        X = pd.DataFrame([features])
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Scale if scaler exists
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Predict
        proba = self.model.predict_proba(X)[0, 1]
        return proba >= 0.5, proba


class StrategyBacktester:
    """Backtester supporting multiple exit strategies"""
    
    def __init__(
        self,
        # Fixed TP/SL params
        tp_pct: float = 0.05,
        sl_pct: float = 0.025,
        # Trailing SL params
        trailing_activation_pct: float = 0.02,  # Activate trailing after 2% profit
        trailing_distance_pct: float = 0.015,   # Trail 1.5% behind
        # General params
        max_bars: int = 20,
        commission: float = 0.0004,
        ml_threshold: float = 0.5
    ):
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_distance_pct = trailing_distance_pct
        self.max_bars = max_bars
        self.commission = commission
        self.ml_threshold = ml_threshold
        
        # ML filter
        self.ml_filter = None
    
    def load_ml_model(self, model_path: str = None):
        """Load ML model for entry filtering"""
        self.ml_filter = MLEntryFilter(model_path)
    
    def backtest_fixed_tpsl(
        self,
        df: pd.DataFrame,
        symbol: str,
        use_ml_filter: bool = True
    ) -> BacktestResult:
        """
        Method 1: Fixed TP/SL with optional ML filtering
        """
        trades = []
        
        for i in range(len(df) - self.max_bars - 1):
            row = df.iloc[i]
            
            # Check entry signal
            is_long = row.get('macd_cross_up', 0) == 1
            is_short = row.get('macd_cross_down', 0) == 1
            
            if not is_long and not is_short:
                continue
            
            # ML filter
            ml_confidence = 0.5
            if use_ml_filter and self.ml_filter is not None:
                should_enter, ml_confidence = self.ml_filter.predict(row)
                if not should_enter or ml_confidence < self.ml_threshold:
                    continue
            
            # Create trade
            entry_price = row['close']
            direction = 'LONG' if is_long else 'SHORT'
            
            if direction == 'LONG':
                tp_price = entry_price * (1 + self.tp_pct)
                sl_price = entry_price * (1 - self.sl_pct)
            else:
                tp_price = entry_price * (1 - self.tp_pct)
                sl_price = entry_price * (1 + self.sl_pct)
            
            trade = Trade(
                symbol=symbol,
                entry_time=row['timestamp'],
                entry_price=entry_price,
                direction=direction,
                ml_confidence=ml_confidence
            )
            
            # Simulate exit
            for j in range(1, self.max_bars + 1):
                if i + j >= len(df):
                    break
                
                bar = df.iloc[i + j]
                
                if direction == 'LONG':
                    if bar['low'] <= sl_price:
                        trade.exit_price = sl_price
                        trade.exit_reason = 'SL'
                        break
                    if bar['high'] >= tp_price:
                        trade.exit_price = tp_price
                        trade.exit_reason = 'TP'
                        break
                else:
                    if bar['high'] >= sl_price:
                        trade.exit_price = sl_price
                        trade.exit_reason = 'SL'
                        break
                    if bar['low'] <= tp_price:
                        trade.exit_price = tp_price
                        trade.exit_reason = 'TP'
                        break
                
                trade.bars_held = j
            
            # Timeout
            if trade.exit_price is None:
                last_bar = df.iloc[min(i + self.max_bars, len(df) - 1)]
                trade.exit_price = last_bar['close']
                trade.exit_reason = 'TIMEOUT'
                trade.bars_held = self.max_bars
            
            trade.exit_time = df.iloc[i + trade.bars_held]['timestamp']
            
            # Calculate PnL
            if direction == 'LONG':
                trade.pnl_pct = (trade.exit_price - entry_price) / entry_price
            else:
                trade.pnl_pct = (entry_price - trade.exit_price) / entry_price
            
            trade.pnl_pct -= 2 * self.commission
            trades.append(trade)
        
        method_name = 'ML_Fixed_TPSL' if use_ml_filter else 'Fixed_TPSL'
        return self._calculate_results(trades, method_name, symbol)
    
    def backtest_trailing_sl(
        self,
        df: pd.DataFrame,
        symbol: str,
        use_ml_filter: bool = True
    ) -> BacktestResult:
        """
        Method 2: Trailing Stop Loss
        - Initial SL at fixed distance
        - Once in profit by activation_pct, trail SL behind price
        - No fixed TP (let profits run)
        """
        trades = []
        
        for i in range(len(df) - self.max_bars - 1):
            row = df.iloc[i]
            
            # Check entry signal
            is_long = row.get('macd_cross_up', 0) == 1
            is_short = row.get('macd_cross_down', 0) == 1
            
            if not is_long and not is_short:
                continue
            
            # ML filter
            ml_confidence = 0.5
            if use_ml_filter and self.ml_filter is not None:
                should_enter, ml_confidence = self.ml_filter.predict(row)
                if not should_enter or ml_confidence < self.ml_threshold:
                    continue
            
            # Create trade
            entry_price = row['close']
            direction = 'LONG' if is_long else 'SHORT'
            
            trade = Trade(
                symbol=symbol,
                entry_time=row['timestamp'],
                entry_price=entry_price,
                direction=direction,
                ml_confidence=ml_confidence,
                highest_price=entry_price,
                lowest_price=entry_price
            )
            
            # Initial SL
            if direction == 'LONG':
                current_sl = entry_price * (1 - self.sl_pct)
            else:
                current_sl = entry_price * (1 + self.sl_pct)
            
            trailing_activated = False
            
            # Simulate exit with trailing
            for j in range(1, self.max_bars + 1):
                if i + j >= len(df):
                    break
                
                bar = df.iloc[i + j]
                
                if direction == 'LONG':
                    # Update highest price
                    if bar['high'] > trade.highest_price:
                        trade.highest_price = bar['high']
                    
                    # Check if trailing should activate
                    profit_pct = (trade.highest_price - entry_price) / entry_price
                    if profit_pct >= self.trailing_activation_pct:
                        trailing_activated = True
                    
                    # Update trailing SL
                    if trailing_activated:
                        new_sl = trade.highest_price * (1 - self.trailing_distance_pct)
                        current_sl = max(current_sl, new_sl)
                    
                    # Check SL hit
                    if bar['low'] <= current_sl:
                        trade.exit_price = current_sl
                        trade.exit_reason = 'TRAILING_SL' if trailing_activated else 'SL'
                        break
                
                else:  # SHORT
                    # Update lowest price
                    if bar['low'] < trade.lowest_price:
                        trade.lowest_price = bar['low']
                    
                    # Check if trailing should activate
                    profit_pct = (entry_price - trade.lowest_price) / entry_price
                    if profit_pct >= self.trailing_activation_pct:
                        trailing_activated = True
                    
                    # Update trailing SL
                    if trailing_activated:
                        new_sl = trade.lowest_price * (1 + self.trailing_distance_pct)
                        current_sl = min(current_sl, new_sl)
                    
                    # Check SL hit
                    if bar['high'] >= current_sl:
                        trade.exit_price = current_sl
                        trade.exit_reason = 'TRAILING_SL' if trailing_activated else 'SL'
                        break
                
                trade.bars_held = j
            
            # Timeout - exit at close
            if trade.exit_price is None:
                last_bar = df.iloc[min(i + self.max_bars, len(df) - 1)]
                trade.exit_price = last_bar['close']
                trade.exit_reason = 'TIMEOUT'
                trade.bars_held = self.max_bars
            
            trade.exit_time = df.iloc[i + trade.bars_held]['timestamp']
            
            # Calculate PnL
            if direction == 'LONG':
                trade.pnl_pct = (trade.exit_price - entry_price) / entry_price
            else:
                trade.pnl_pct = (entry_price - trade.exit_price) / entry_price
            
            trade.pnl_pct -= 2 * self.commission
            trades.append(trade)
        
        method_name = 'ML_Trailing_SL' if use_ml_filter else 'Trailing_SL'
        return self._calculate_results(trades, method_name, symbol)
    
    def _calculate_results(
        self,
        trades: List[Trade],
        method: str,
        symbol: str
    ) -> BacktestResult:
        """Calculate metrics from trades"""
        result = BacktestResult(method=method, symbol=symbol, trades=trades)
        
        if not trades:
            return result
        
        result.total_trades = len(trades)
        result.winning_trades = sum(1 for t in trades if t.pnl_pct > 0)
        result.win_rate = result.winning_trades / result.total_trades
        
        pnls = [t.pnl_pct for t in trades]
        result.total_pnl_pct = sum(pnls)
        result.avg_pnl_pct = np.mean(pnls)
        result.avg_bars_held = np.mean([t.bars_held for t in trades])
        
        # Max Drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        result.max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        # Sharpe (annualized)
        if np.std(pnls) > 0:
            result.sharpe_ratio = np.mean(pnls) / np.std(pnls) * np.sqrt(252)
        
        # Profit Factor
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        if gross_loss > 0:
            result.profit_factor = gross_profit / gross_loss
        
        return result


def run_comparison(df: pd.DataFrame, ml_threshold: float = 0.5) -> pd.DataFrame:
    """Run full comparison across all symbols"""
    
    backtester = StrategyBacktester(
        tp_pct=0.05,
        sl_pct=0.025,
        trailing_activation_pct=0.02,
        trailing_distance_pct=0.015,
        max_bars=20,
        ml_threshold=ml_threshold
    )
    backtester.load_ml_model()
    
    symbols = df['symbol'].unique()
    all_results = []
    
    print(f"\nBacktesting {len(symbols)} symbols with ML threshold={ml_threshold:.0%}...")
    print("-" * 80)
    
    for symbol in symbols:
        symbol_df = df[df['symbol'] == symbol].copy().reset_index(drop=True)
        
        if len(symbol_df) < 200:
            continue
        
        # Method 1: ML + Fixed TP/SL
        result_ml_fixed = backtester.backtest_fixed_tpsl(symbol_df, symbol, use_ml_filter=True)
        all_results.append(result_ml_fixed.to_dict())
        
        # Method 2: ML + Trailing SL
        result_ml_trailing = backtester.backtest_trailing_sl(symbol_df, symbol, use_ml_filter=True)
        all_results.append(result_ml_trailing.to_dict())
        
        # Method 3: No ML + Fixed TP/SL (baseline)
        result_baseline = backtester.backtest_fixed_tpsl(symbol_df, symbol, use_ml_filter=False)
        all_results.append(result_baseline.to_dict())
        
        # Method 4: No ML + Trailing SL
        result_trailing_only = backtester.backtest_trailing_sl(symbol_df, symbol, use_ml_filter=False)
        all_results.append(result_trailing_only.to_dict())
        
        # Print per-symbol summary
        print(f"{symbol:12s} | "
              f"ML+Fixed: {result_ml_fixed.total_trades:3d} trades, {result_ml_fixed.win_rate:5.1%} win, {result_ml_fixed.total_pnl_pct:+6.1%} | "
              f"ML+Trail: {result_ml_trailing.total_trades:3d} trades, {result_ml_trailing.win_rate:5.1%} win, {result_ml_trailing.total_pnl_pct:+6.1%}")
    
    return pd.DataFrame(all_results)


def aggregate_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results by method"""
    
    agg = results_df.groupby('method').agg({
        'total_trades': 'sum',
        'winning_trades': 'sum',
        'total_pnl_pct': 'sum',
        'avg_pnl_pct': 'mean',
        'max_drawdown': 'max',
        'sharpe_ratio': 'mean',
        'profit_factor': 'mean',
        'avg_bars_held': 'mean'
    }).reset_index()
    
    agg['win_rate'] = agg['winning_trades'] / agg['total_trades']
    
    return agg.sort_values('total_pnl_pct', ascending=False)


if __name__ == '__main__':
    print("=" * 80)
    print("MACD Crossover Strategy Comparison")
    print("Method 1: ML Filter + Fixed TP/SL")
    print("Method 2: ML Filter + Trailing Stop Loss")
    print("=" * 80)
    
    # Load data
    data_path = PROCESSED_DIR / 'features_1d_test.parquet'
    if not data_path.exists():
        print(f"Data not found: {data_path}")
        print("Run data_pipeline.py first!")
        exit(1)
    
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} rows, {df['symbol'].nunique()} symbols")
    
    # Run comparison
    results_df = run_comparison(df, ml_threshold=0.5)
    
    # Aggregate
    print("\n" + "=" * 80)
    print("AGGREGATED RESULTS (All Symbols)")
    print("=" * 80)
    
    agg_results = aggregate_results(results_df)
    
    print(f"\n{'Method':<20} {'Trades':>8} {'Win%':>8} {'Total PnL':>12} {'Sharpe':>8} {'PF':>8}")
    print("-" * 70)
    
    for _, row in agg_results.iterrows():
        print(f"{row['method']:<20} {row['total_trades']:>8.0f} {row['win_rate']:>7.1%} "
              f"{row['total_pnl_pct']:>+11.1%} {row['sharpe_ratio']:>8.2f} {row['profit_factor']:>8.2f}")
    
    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    results_df.to_csv(RESULTS_DIR / 'comparison_by_symbol.csv', index=False)
    agg_results.to_csv(RESULTS_DIR / 'comparison_aggregated.csv', index=False)
    
    print(f"\n✓ Results saved to {RESULTS_DIR}")
    
    # Best per-symbol analysis
    print("\n" + "=" * 80)
    print("BEST METHOD BY SYMBOL")
    print("=" * 80)
    
    best_by_symbol = results_df.loc[results_df.groupby('symbol')['total_pnl_pct'].idxmax()]
    method_counts = best_by_symbol['method'].value_counts()
    
    print("\nMethod with highest PnL per symbol:")
    for method, count in method_counts.items():
        print(f"  {method}: {count} symbols ({count/len(method_counts.index)*100:.1f}%)")
    
    # Compare ML vs No-ML
    print("\n" + "=" * 80)
    print("ML FILTER IMPACT")
    print("=" * 80)
    
    ml_methods = results_df[results_df['method'].str.contains('ML_')]
    no_ml_methods = results_df[~results_df['method'].str.contains('ML_')]
    
    ml_total_pnl = ml_methods.groupby('method')['total_pnl_pct'].sum()
    no_ml_total_pnl = no_ml_methods.groupby('method')['total_pnl_pct'].sum()
    
    print("\nWith ML Filter:")
    for method, pnl in ml_total_pnl.items():
        trades = ml_methods[ml_methods['method'] == method]['total_trades'].sum()
        print(f"  {method}: {trades:.0f} trades, {pnl:+.1%} total PnL")
    
    print("\nWithout ML Filter:")
    for method, pnl in no_ml_total_pnl.items():
        trades = no_ml_methods[no_ml_methods['method'] == method]['total_trades'].sum()
        print(f"  {method}: {trades:.0f} trades, {pnl:+.1%} total PnL")
