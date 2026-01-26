#!/usr/bin/env python3
"""
3-Stage ML Backtest with Realistic Conditions

Features:
1. Entry Filter (Stage 1) with confidence threshold
2. Dynamic SL (Stage 2) and TP (Stage 3) from ML predictions
3. SL-First rule: When both TP/SL hit in same candle, assume SL hit first
4. Position sizing: Fixed risk per trade (1-2% of account)
5. Kelly Criterion optional sizing
6. Max concentration per coin (20%)
7. Slippage and fees (0.05-0.1%)
8. Compare with baseline strategies
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent.parent / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
MODEL_DIR = Path(__file__).parent / 'models'


@dataclass
class Trade:
    """Single trade record"""
    symbol: str
    entry_time: datetime
    exit_time: datetime = None
    direction: str = 'LONG'  # LONG or SHORT
    entry_price: float = 0
    exit_price: float = 0
    sl_price: float = 0
    tp_price: float = 0
    sl_pct: float = 0
    tp_pct: float = 0
    position_size: float = 0  # In USD
    pnl: float = 0
    pnl_pct: float = 0
    exit_reason: str = ''  # TP_HIT, SL_HIT, TIMEOUT
    confidence: float = 0
    fees_paid: float = 0
    bars_held: int = 0


@dataclass
class BacktestConfig:
    """Backtest configuration"""
    initial_capital: float = 10000
    risk_per_trade: float = 0.01  # 1% risk per trade
    max_concentration: float = 0.20  # Max 20% in single coin
    entry_threshold: float = 0.65  # Min confidence to enter
    fee_rate: float = 0.001  # 0.1% per trade (entry + exit = 0.2%)
    slippage: float = 0.0005  # 0.05% slippage
    max_bars: int = 10  # Max bars to hold
    use_kelly: bool = False  # Use Kelly Criterion for sizing
    kelly_fraction: float = 0.5  # Half-Kelly for safety
    allow_shorts: bool = True  # Allow short positions
    max_open_trades: int = 10  # Max concurrent trades
    fixed_position_size: bool = False  # If True, use fixed $ amount instead of % of equity
    position_size_usd: float = 1000  # Fixed position size in USD
    leverage: float = 1.0  # Leverage multiplier (1x, 3x, 5x, 7x, 10x)
    liquidation_threshold: float = 0.80  # Liquidation at 80% margin loss


@dataclass
class BacktestResult:
    """Backtest results"""
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)
    
    # Summary metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0
    total_pnl: float = 0
    total_return: float = 0
    max_drawdown: float = 0
    sharpe_ratio: float = 0
    profit_factor: float = 0
    avg_trade_pnl: float = 0
    avg_winner: float = 0
    avg_loser: float = 0
    best_trade: float = 0
    worst_trade: float = 0
    avg_bars_held: float = 0
    total_fees: float = 0


class ThreeStageBacktester:
    """
    Professional backtester for 3-stage ML system.
    
    Stage 1: Entry Filter (classification) - decides IF we should enter
    Stage 2: SL Predictor (regression) - decides WHERE to place SL
    Stage 3: TP Predictor (regression) - decides WHERE to place TP
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.entry_model = None
        self.sl_model = None
        self.tp_model = None
        self.entry_scaler = None
        self.sl_scaler = None
        self.tp_scaler = None
        self.entry_features = None
        self.sl_features = None
        self.tp_features = None
        
        self._load_models()
    
    def _load_models(self):
        """Load all 3 ML models."""
        # Stage 1: Entry Filter
        entry_path = MODEL_DIR / 'entry_filter.joblib'
        if entry_path.exists():
            data = joblib.load(entry_path)
            self.entry_model = data['model']
            self.entry_scaler = data.get('scaler')
            self.entry_features = data['feature_names']
            print(f"✓ Stage 1 loaded: {len(self.entry_features)} features")
        else:
            print("⚠️ Entry filter not found")
        
        # Stage 2: SL Predictor
        sl_path = MODEL_DIR / 'sl_predictor.joblib'
        if sl_path.exists():
            data = joblib.load(sl_path)
            self.sl_model = data['model']
            self.sl_scaler = data.get('scaler')
            self.sl_features = data['feature_names']
            print(f"✓ Stage 2 loaded: {len(self.sl_features)} features")
        else:
            print("⚠️ SL predictor not found")
        
        # Stage 3: TP Predictor
        tp_path = MODEL_DIR / 'tp_predictor.joblib'
        if tp_path.exists():
            data = joblib.load(tp_path)
            self.tp_model = data['model']
            self.tp_scaler = data.get('scaler')
            self.tp_features = data['feature_names']
            self.tp_predict_rr = data.get('predict_rr', False)
            print(f"✓ Stage 3 loaded: {len(self.tp_features)} features")
        else:
            print("⚠️ TP predictor not found")
    
    def _prepare_features(self, row: pd.Series, feature_names: list, scaler) -> np.ndarray:
        """Prepare features for prediction."""
        X = pd.DataFrame()
        for col in feature_names:
            if col in row.index:
                X[col] = [row[col]]
            else:
                X[col] = [0]
        
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        if scaler is not None:
            X = scaler.transform(X)
        
        return X
    
    def predict_entry(self, row: pd.Series) -> Tuple[bool, float]:
        """Stage 1: Predict if entry is good."""
        if self.entry_model is None:
            return True, 0.5
        
        X = self._prepare_features(row, self.entry_features, self.entry_scaler)
        proba = self.entry_model.predict_proba(X)[0, 1]
        should_enter = proba >= self.config.entry_threshold
        
        return should_enter, proba
    
    def predict_sl(self, row: pd.Series) -> float:
        """Stage 2: Predict optimal SL percentage."""
        if self.sl_model is None:
            return 0.02  # Default 2%
        
        X = self._prepare_features(row, self.sl_features, self.sl_scaler)
        sl_pct = self.sl_model.predict(X)[0]
        return np.clip(sl_pct, 0.005, 0.15)  # 0.5% - 15%
    
    def predict_tp(self, row: pd.Series, sl_pct: float = None) -> float:
        """Stage 3: Predict optimal TP percentage."""
        if self.tp_model is None:
            return 0.04  # Default 4%
        
        X = self._prepare_features(row, self.tp_features, self.tp_scaler)
        pred = self.tp_model.predict(X)[0]
        
        # If model predicts RR ratio, convert to TP%
        if getattr(self, 'tp_predict_rr', False) and sl_pct is not None:
            tp_pct = pred * sl_pct  # RR * SL = TP
        else:
            tp_pct = pred
        
        return np.clip(tp_pct, 0.01, 0.30)  # 1% - 30%
    
    def calculate_position_size(
        self, 
        capital: float, 
        sl_pct: float, 
        confidence: float,
        current_positions: Dict[str, float]
    ) -> float:
        """
        Calculate position size based on risk management rules.
        
        Fixed Risk: risk_amount = capital * risk_per_trade
        Position Size = risk_amount / sl_pct
        
        Kelly Criterion (optional):
        f* = (p * b - q) / b
        where p = win probability, q = 1-p, b = win/loss ratio (RR)
        """
        # Fixed position size mode (with leverage)
        if self.config.fixed_position_size:
            # With leverage, we can control larger position with less margin
            max_position = self.config.position_size_usd * self.config.leverage
            margin_required = self.config.position_size_usd  # Actual capital used
            if margin_required > capital * 0.9:
                margin_required = capital * 0.9
                max_position = margin_required * self.config.leverage
            return max_position
        
        # Calculate total current exposure
        total_exposure = sum(current_positions.values())
        available_capital = capital - total_exposure
        
        if available_capital <= 0:
            return 0
        
        # Fixed risk sizing
        risk_amount = capital * self.config.risk_per_trade
        
        # Kelly criterion (optional)
        if self.config.use_kelly and confidence > 0.5:
            # Estimate RR from model (assume 2:1 as baseline)
            estimated_rr = 2.0
            p = confidence  # Win probability from Stage 1
            q = 1 - p
            b = estimated_rr
            
            # Kelly fraction
            kelly_f = (p * b - q) / b
            kelly_f = max(0, min(kelly_f, 0.25))  # Cap at 25%
            
            # Use half-Kelly for safety
            kelly_f *= self.config.kelly_fraction
            
            # Adjust risk based on Kelly
            risk_amount = capital * kelly_f
        
        # Calculate position size from risk and SL
        if sl_pct > 0:
            position_size = risk_amount / sl_pct
        else:
            position_size = risk_amount / 0.02  # Default 2% SL
        
        # Apply leverage
        position_size *= self.config.leverage
        
        # Apply max concentration limit (on leveraged position)
        max_position = capital * self.config.max_concentration * self.config.leverage
        position_size = min(position_size, max_position)
        
        # Can't exceed available capital * leverage
        position_size = min(position_size, available_capital * self.config.leverage)
        
        return max(0, position_size)
    
    def simulate_trade(
        self,
        entry_row: pd.Series,
        future_data: pd.DataFrame,
        entry_price: float,
        sl_pct: float,
        tp_pct: float,
        direction: str,
        position_size: float
    ) -> Trade:
        """
        Simulate a single trade with SL-First rule.
        
        SL-First Rule: If both SL and TP could be hit in the same candle,
        assume SL was hit first (conservative approach).
        """
        trade = Trade(
            symbol=entry_row.get('symbol', 'UNKNOWN'),
            entry_time=entry_row.get('timestamp', datetime.now()),
            direction=direction,
            entry_price=entry_price,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            position_size=position_size
        )
        
        # Apply slippage to entry
        if direction == 'LONG':
            trade.entry_price *= (1 + self.config.slippage)
            trade.sl_price = trade.entry_price * (1 - sl_pct)
            trade.tp_price = trade.entry_price * (1 + tp_pct)
        else:  # SHORT
            trade.entry_price *= (1 - self.config.slippage)
            trade.sl_price = trade.entry_price * (1 + sl_pct)
            trade.tp_price = trade.entry_price * (1 - tp_pct)
        
        # Entry fee
        entry_fee = position_size * self.config.fee_rate
        trade.fees_paid = entry_fee
        
        # Simulate through future candles
        for i, (idx, row) in enumerate(future_data.iterrows()):
            trade.bars_held = i + 1
            
            high = row['high']
            low = row['low']
            close = row['close']
            
            if direction == 'LONG':
                # Check SL FIRST (conservative)
                sl_hit = low <= trade.sl_price
                tp_hit = high >= trade.tp_price
                
                if sl_hit and tp_hit:
                    # Both hit - SL First rule
                    trade.exit_price = trade.sl_price * (1 - self.config.slippage)
                    trade.exit_reason = 'SL_HIT'
                elif sl_hit:
                    trade.exit_price = trade.sl_price * (1 - self.config.slippage)
                    trade.exit_reason = 'SL_HIT'
                elif tp_hit:
                    trade.exit_price = trade.tp_price * (1 - self.config.slippage)
                    trade.exit_reason = 'TP_HIT'
            else:  # SHORT
                # Check SL FIRST (conservative)
                sl_hit = high >= trade.sl_price
                tp_hit = low <= trade.tp_price
                
                if sl_hit and tp_hit:
                    # Both hit - SL First rule
                    trade.exit_price = trade.sl_price * (1 + self.config.slippage)
                    trade.exit_reason = 'SL_HIT'
                elif sl_hit:
                    trade.exit_price = trade.sl_price * (1 + self.config.slippage)
                    trade.exit_reason = 'SL_HIT'
                elif tp_hit:
                    trade.exit_price = trade.tp_price * (1 + self.config.slippage)
                    trade.exit_reason = 'TP_HIT'
            
            if trade.exit_reason:
                trade.exit_time = row.get('timestamp', datetime.now())
                break
            
            # Max bars reached - exit at close
            if trade.bars_held >= self.config.max_bars:
                trade.exit_price = close * (1 - self.config.slippage if direction == 'LONG' else 1 + self.config.slippage)
                trade.exit_reason = 'TIMEOUT'
                trade.exit_time = row.get('timestamp', datetime.now())
                break
        
        # Calculate PnL with leverage
        if trade.exit_price > 0:
            if direction == 'LONG':
                trade.pnl_pct = (trade.exit_price - trade.entry_price) / trade.entry_price
            else:
                trade.pnl_pct = (trade.entry_price - trade.exit_price) / trade.entry_price
            
            # Exit fee
            exit_fee = position_size * (1 + trade.pnl_pct) * self.config.fee_rate
            trade.fees_paid += exit_fee
            
            # Calculate margin (actual capital used)
            margin = position_size / self.config.leverage
            
            # Net PnL after fees (on leveraged position)
            trade.pnl = position_size * trade.pnl_pct - trade.fees_paid
            
            # Check for liquidation (loss exceeds margin * liquidation_threshold)
            max_loss = -margin * self.config.liquidation_threshold
            if trade.pnl < max_loss:
                trade.pnl = -margin  # Lose entire margin
                trade.exit_reason = 'LIQUIDATED'
        
        return trade
    
    def run_backtest(self, df: pd.DataFrame, verbose: bool = True) -> BacktestResult:
        """
        Run full backtest on historical data with REALISTIC capital constraints.
        
        Key features:
        - Process signals chronologically across ALL symbols
        - Track open positions and their margin usage over time
        - Properly constrain capital (can't open new trades if margin exhausted)
        - Track equity curve by time, not just by trade close
        
        Args:
            df: DataFrame with OHLCV, features, and crossover signals
            verbose: Print progress
        """
        result = BacktestResult()
        capital = self.config.initial_capital
        available_capital = capital  # Capital available for new positions
        
        # Track open positions: {trade_id: {'trade': Trade, 'margin': float, 'exit_time': datetime}}
        open_positions: Dict[int, Dict] = {}
        trade_counter = 0
        
        # ===== STEP 1: Collect all crossover signals across all symbols =====
        all_signals = []
        
        symbols = df['symbol'].unique() if 'symbol' in df.columns else ['UNKNOWN']
        
        for symbol in symbols:
            df_symbol = df[df['symbol'] == symbol].sort_values('timestamp').reset_index(drop=True)
            
            # Find crossover signals
            crossover_mask = (df_symbol['macd_cross_up'] == 1) | (df_symbol['macd_cross_down'] == 1)
            
            for idx in df_symbol[crossover_mask].index:
                # Skip if too close to end
                if idx >= len(df_symbol) - self.config.max_bars:
                    continue
                
                row = df_symbol.iloc[idx]
                future_data = df_symbol.iloc[idx + 1: idx + 1 + self.config.max_bars]
                
                if len(future_data) == 0:
                    continue
                
                all_signals.append({
                    'timestamp': row['timestamp'],
                    'symbol': symbol,
                    'row': row,
                    'future_data': future_data,
                    'is_long': row.get('macd_cross_up', 0) == 1
                })
        
        # Sort signals by timestamp (chronological order)
        all_signals.sort(key=lambda x: x['timestamp'])
        
        if verbose:
            print(f"Found {len(all_signals)} crossover signals across {len(symbols)} symbols")
        
        # ===== STEP 2: Process signals chronologically =====
        equity_timeline = [(df['timestamp'].min(), capital)]  # (timestamp, equity)
        
        for signal in all_signals:
            current_time = signal['timestamp']
            row = signal['row']
            future_data = signal['future_data']
            symbol = signal['symbol']
            is_long = signal['is_long']
            
            # ----- Close expired positions and update capital -----
            closed_trade_ids = []
            for trade_id, pos in open_positions.items():
                if pos['exit_time'] <= current_time:
                    # Position has closed
                    trade = pos['trade']
                    capital += trade.pnl
                    available_capital += pos['margin'] + trade.pnl  # Return margin + PnL
                    result.trades.append(trade)
                    closed_trade_ids.append(trade_id)
            
            # Remove closed positions
            for tid in closed_trade_ids:
                del open_positions[tid]
            
            # Record equity at this timestamp (after closing positions)
            if closed_trade_ids:
                equity_timeline.append((current_time, capital))
            
            # ----- Check if we can open new position -----
            
            # Skip shorts if not allowed
            if not is_long and not self.config.allow_shorts:
                continue
            direction = 'LONG' if is_long else 'SHORT'
            
            # Stage 1: Entry Filter
            should_enter, confidence = self.predict_entry(row)
            if not should_enter:
                continue
            
            # Check max open trades
            if len(open_positions) >= self.config.max_open_trades:
                continue
            
            # Check if we already have position in this symbol
            symbols_in_position = {pos['trade'].symbol for pos in open_positions.values()}
            if symbol in symbols_in_position:
                continue  # Don't open multiple positions in same symbol
            
            # Stage 2: Predict SL
            sl_pct = self.predict_sl(row)
            
            # Stage 3: Predict TP
            tp_pct = self.predict_tp(row, sl_pct)
            
            # Calculate position size based on AVAILABLE capital
            if self.config.fixed_position_size:
                margin_needed = self.config.position_size_usd
                position_size = margin_needed * self.config.leverage
                
                # Check if we have enough available capital
                if margin_needed > available_capital * 0.95:  # Keep 5% buffer
                    if verbose and len(result.trades) < 10:
                        print(f"  ⚠️ Skipping trade: insufficient capital (need ${margin_needed:.0f}, have ${available_capital:.0f})")
                    continue
            else:
                # Risk-based sizing on available capital
                risk_amount = available_capital * self.config.risk_per_trade
                if sl_pct > 0:
                    position_size = risk_amount / sl_pct * self.config.leverage
                else:
                    position_size = risk_amount / 0.02 * self.config.leverage
                
                # Apply max concentration limit
                max_position = available_capital * self.config.max_concentration * self.config.leverage
                position_size = min(position_size, max_position)
                margin_needed = position_size / self.config.leverage
            
            if position_size <= 0 or margin_needed <= 0:
                continue
            
            # ----- Simulate trade -----
            entry_price = row['close']
            trade = self.simulate_trade(
                row, future_data, entry_price,
                sl_pct, tp_pct, direction, position_size
            )
            trade.confidence = confidence
            
            # ----- Track open position -----
            trade_counter += 1
            open_positions[trade_counter] = {
                'trade': trade,
                'margin': margin_needed,
                'exit_time': trade.exit_time or (current_time + pd.Timedelta(days=self.config.max_bars))
            }
            
            # Deduct margin from available capital
            available_capital -= margin_needed
            
            if verbose and len(result.trades) % 100 == 0 and len(result.trades) > 0:
                print(f"  Processed {len(result.trades)} trades... Capital: ${capital:,.2f}, "
                      f"Available: ${available_capital:,.2f}, Open: {len(open_positions)}")
        
        # ===== STEP 3: Close remaining open positions =====
        for trade_id, pos in open_positions.items():
            trade = pos['trade']
            capital += trade.pnl
            result.trades.append(trade)
        
        # ===== STEP 4: Build equity curve =====
        # Sort trades by exit time for proper equity curve
        result.trades.sort(key=lambda t: t.exit_time or t.entry_time)
        
        # Rebuild equity curve from sorted trades
        result.equity_curve = [self.config.initial_capital]
        running_capital = self.config.initial_capital
        for trade in result.trades:
            running_capital += trade.pnl
            result.equity_curve.append(running_capital)
            result.timestamps.append(trade.exit_time or trade.entry_time)
        
        if verbose:
            print(f"  Total trades executed: {len(result.trades)}")
            print(f"  Final capital: ${capital:,.2f}")
        
        # Calculate summary metrics
        self._calculate_metrics(result)
        
        return result
    
    def _calculate_metrics(self, result: BacktestResult):
        """Calculate summary metrics for backtest."""
        if not result.trades:
            return
        
        trades = result.trades
        result.total_trades = len(trades)
        
        # Win/Loss counts
        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]
        result.winning_trades = len(winners)
        result.losing_trades = len(losers)
        result.win_rate = len(winners) / len(trades) if trades else 0
        
        # PnL metrics
        result.total_pnl = sum(t.pnl for t in trades)
        result.total_return = result.total_pnl / self.config.initial_capital
        result.avg_trade_pnl = result.total_pnl / len(trades)
        result.avg_winner = np.mean([t.pnl for t in winners]) if winners else 0
        result.avg_loser = np.mean([t.pnl for t in losers]) if losers else 0
        result.best_trade = max(t.pnl for t in trades)
        result.worst_trade = min(t.pnl for t in trades)
        result.total_fees = sum(t.fees_paid for t in trades)
        
        # Bars held
        result.avg_bars_held = np.mean([t.bars_held for t in trades])
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 1
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Max drawdown
        equity = result.equity_curve
        peak = equity[0]
        max_dd = 0
        for eq in equity:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)
        result.max_drawdown = max_dd
        
        # Sharpe Ratio (simplified - daily returns)
        if len(equity) > 1:
            returns = np.diff(equity) / equity[:-1]
            if np.std(returns) > 0:
                result.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                result.sharpe_ratio = 0
    
    def print_results(self, result: BacktestResult):
        """Print formatted backtest results."""
        print("\n" + "="*70)
        print("3-STAGE ML BACKTEST RESULTS")
        print("="*70)
        
        print(f"\n📊 Configuration:")
        print(f"   Initial Capital: ${self.config.initial_capital:,.2f}")
        print(f"   Leverage: {self.config.leverage:.0f}x")
        print(f"   Risk per Trade: {self.config.risk_per_trade:.1%}")
        print(f"   Entry Threshold: {self.config.entry_threshold:.0%}")
        print(f"   Fee Rate: {self.config.fee_rate:.2%}")
        print(f"   Slippage: {self.config.slippage:.2%}")
        print(f"   Max Bars: {self.config.max_bars}")
        print(f"   Kelly Criterion: {'Yes' if self.config.use_kelly else 'No'}")
        
        print(f"\n📈 Performance Summary:")
        print(f"   Total Trades: {result.total_trades}")
        print(f"   Winning: {result.winning_trades} ({result.win_rate:.1%})")
        print(f"   Losing: {result.losing_trades}")
        
        print(f"\n💰 PnL Metrics:")
        print(f"   Total PnL: ${result.total_pnl:,.2f}")
        print(f"   Total Return: {result.total_return:.1%}")
        print(f"   Avg Trade: ${result.avg_trade_pnl:,.2f}")
        print(f"   Avg Winner: ${result.avg_winner:,.2f}")
        print(f"   Avg Loser: ${result.avg_loser:,.2f}")
        print(f"   Best Trade: ${result.best_trade:,.2f}")
        print(f"   Worst Trade: ${result.worst_trade:,.2f}")
        print(f"   Total Fees Paid: ${result.total_fees:,.2f}")
        
        print(f"\n📉 Risk Metrics:")
        print(f"   Max Drawdown: {result.max_drawdown:.1%}")
        print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"   Profit Factor: {result.profit_factor:.2f}")
        
        print(f"\n⏱️ Trade Duration:")
        print(f"   Avg Bars Held: {result.avg_bars_held:.1f}")
        
        # Exit reason breakdown
        exit_reasons = {}
        for t in result.trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
        
        print(f"\n🚪 Exit Reasons:")
        for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
            pct = count / result.total_trades * 100
            print(f"   {reason}: {count} ({pct:.1f}%)")
        
        # Final equity
        final_equity = result.equity_curve[-1] if result.equity_curve else self.config.initial_capital
        print(f"\n🏆 Final Equity: ${final_equity:,.2f}")
        print("="*70)


def plot_equity_curve(results: Dict[str, BacktestResult], title: str = "Equity Curve", save_path: str = None):
    """
    Plot equity curves for multiple strategies.
    
    Args:
        results: Dict of strategy name -> BacktestResult
        title: Chart title
        save_path: If provided, save chart to this path
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Color palette
    colors = {
        '1x': '#2E86AB',
        '3x': '#A23B72', 
        '5x': '#F18F01',
        '7x': '#C73E1D',
        '10x': '#3B1F2B',
        '3-Stage ML': '#2E86AB',
        'Baseline: All Signals': '#C73E1D',
        'ML Entry Only': '#F18F01'
    }
    
    # 1. Equity Curves (Top Left)
    ax1 = axes[0, 0]
    for name, result in results.items():
        if result.equity_curve:
            color = colors.get(name, '#333333')
            ax1.plot(result.equity_curve, label=name, linewidth=2, color=color)
    
    ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Trade #')
    ax1.set_ylabel('Equity ($)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=list(results.values())[0].equity_curve[0] if results else 10000, 
                color='gray', linestyle='--', alpha=0.5, label='Initial')
    
    # Format y-axis with comma separator
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # 2. Drawdown Chart (Top Right)
    ax2 = axes[0, 1]
    for name, result in results.items():
        if result.equity_curve:
            equity = np.array(result.equity_curve)
            peak = np.maximum.accumulate(equity)
            drawdown = (peak - equity) / peak * 100
            color = colors.get(name, '#333333')
            ax2.fill_between(range(len(drawdown)), drawdown, alpha=0.3, color=color)
            ax2.plot(drawdown, label=name, linewidth=1.5, color=color)
    
    ax2.set_title('Drawdown %', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Trade #')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    
    # 3. Returns Distribution (Bottom Left)
    ax3 = axes[1, 0]
    for name, result in results.items():
        if result.trades:
            returns = [t.pnl_pct * 100 for t in result.trades]
            color = colors.get(name, '#333333')
            ax3.hist(returns, bins=50, alpha=0.5, label=name, color=color, edgecolor='white')
    
    ax3.set_title('Trade Returns Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Return (%)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # 4. Performance Summary Table (Bottom Right)
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary table
    table_data = []
    headers = ['Strategy', 'Trades', 'Win%', 'Return', 'MaxDD', 'Sharpe', 'PF']
    
    for name, res in results.items():
        table_data.append([
            name,
            f"{res.total_trades}",
            f"{res.win_rate:.1%}",
            f"{res.total_return:.1%}",
            f"{res.max_drawdown:.1%}",
            f"{res.sharpe_ratio:.2f}",
            f"{res.profit_factor:.2f}"
        ])
    
    table = ax4.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colColours=['#f0f0f0']*len(headers)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style table
    for i in range(len(headers)):
        table[(0, i)].set_fontsize(12)
        table[(0, i)].set_text_props(weight='bold')
    
    ax4.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\n📊 Chart saved to: {save_path}")
    
    plt.show()
    return fig


def plot_leverage_comparison(results: Dict[str, BacktestResult], initial_capital: float = 10000, save_path: str = None):
    """
    Specialized plot for leverage comparison.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    leverage_colors = {
        '1x': '#2E86AB',
        '3x': '#A23B72', 
        '5x': '#F18F01',
        '7x': '#C73E1D',
        '10x': '#3B1F2B'
    }
    
    # 1. Equity Curves
    ax1 = axes[0, 0]
    for name, result in results.items():
        if result.equity_curve:
            ax1.plot(result.equity_curve, label=name, linewidth=2, color=leverage_colors.get(name, '#333'))
    ax1.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('Equity Curves by Leverage', fontweight='bold')
    ax1.set_xlabel('Trade #')
    ax1.set_ylabel('Equity ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # 2. Log Scale Equity
    ax2 = axes[0, 1]
    for name, result in results.items():
        if result.equity_curve:
            ax2.semilogy(result.equity_curve, label=name, linewidth=2, color=leverage_colors.get(name, '#333'))
    ax2.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title('Equity Curves (Log Scale)', fontweight='bold')
    ax2.set_xlabel('Trade #')
    ax2.set_ylabel('Equity ($) - Log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Max Drawdown Comparison
    ax3 = axes[0, 2]
    names = list(results.keys())
    max_dds = [results[n].max_drawdown * 100 for n in names]
    colors = [leverage_colors.get(n, '#333') for n in names]
    bars = ax3.bar(names, max_dds, color=colors, edgecolor='white', linewidth=2)
    ax3.set_title('Max Drawdown by Leverage', fontweight='bold')
    ax3.set_ylabel('Max Drawdown (%)')
    ax3.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for bar, dd in zip(bars, max_dds):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{dd:.1f}%', ha='center', fontweight='bold')
    
    # 4. Return vs Drawdown Scatter
    ax4 = axes[1, 0]
    for name, result in results.items():
        ax4.scatter(result.max_drawdown * 100, result.total_return * 100, 
                   s=200, label=name, color=leverage_colors.get(name, '#333'), 
                   edgecolor='white', linewidth=2)
        ax4.annotate(name, (result.max_drawdown * 100, result.total_return * 100),
                    textcoords="offset points", xytext=(10, 5), fontweight='bold')
    ax4.set_title('Return vs Max Drawdown', fontweight='bold')
    ax4.set_xlabel('Max Drawdown (%)')
    ax4.set_ylabel('Total Return (%)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Sharpe Ratio
    ax5 = axes[1, 1]
    sharpes = [results[n].sharpe_ratio for n in names]
    bars = ax5.bar(names, sharpes, color=colors, edgecolor='white', linewidth=2)
    ax5.set_title('Sharpe Ratio by Leverage', fontweight='bold')
    ax5.set_ylabel('Sharpe Ratio')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Sharpe=1')
    ax5.axhline(y=2, color='green', linestyle='--', alpha=0.5, label='Sharpe=2')
    for bar, sr in zip(bars, sharpes):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{sr:.2f}', ha='center', fontweight='bold')
    
    # 6. Final Equity Bar Chart
    ax6 = axes[1, 2]
    final_equities = [results[n].equity_curve[-1] if results[n].equity_curve else initial_capital for n in names]
    bars = ax6.bar(names, final_equities, color=colors, edgecolor='white', linewidth=2)
    ax6.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial')
    ax6.set_title('Final Equity by Leverage', fontweight='bold')
    ax6.set_ylabel('Final Equity ($)')
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    for bar, eq in zip(bars, final_equities):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, 
                f'${eq:,.0f}', ha='center', fontweight='bold', fontsize=9)
    
    plt.suptitle('Leverage Comparison Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\n📊 Leverage chart saved to: {save_path}")
    
    plt.show()
    return fig


def run_baseline_comparison(df: pd.DataFrame, config: BacktestConfig) -> Dict[str, BacktestResult]:
    """
    Compare 3-stage ML with baseline strategies.
    
    Baselines:
    1. All Signals: Trade every crossover with fixed 2% SL, 4% TP
    2. Fixed 2:1 RR: Trade every crossover with 2% SL, 4% TP
    3. ML Entry Only: Use ML for entry, but fixed SL/TP
    """
    results = {}
    
    # 1. 3-Stage ML (full system)
    print("\n" + "="*70)
    print("Running 3-Stage ML Backtest...")
    print("="*70)
    
    backtester = ThreeStageBacktester(config)
    results['3-Stage ML'] = backtester.run_backtest(df)
    backtester.print_results(results['3-Stage ML'])
    
    # 2. Baseline: All signals, fixed SL/TP
    print("\n" + "="*70)
    print("Running Baseline: All Signals (Fixed 2%/4%)...")
    print("="*70)
    
    baseline_config = BacktestConfig(
        initial_capital=config.initial_capital,
        risk_per_trade=config.risk_per_trade,
        entry_threshold=0.0,  # Take all signals
        fee_rate=config.fee_rate,
        slippage=config.slippage,
        max_bars=config.max_bars
    )
    
    class FixedSLTPBacktester(ThreeStageBacktester):
        def predict_entry(self, row):
            return True, 0.5  # Always enter
        
        def predict_sl(self, row):
            return 0.02  # Fixed 2%
        
        def predict_tp(self, row, sl_pct=None):
            return 0.04  # Fixed 4%
    
    baseline_bt = FixedSLTPBacktester(baseline_config)
    results['Baseline: All Signals'] = baseline_bt.run_backtest(df)
    baseline_bt.print_results(results['Baseline: All Signals'])
    
    # 3. ML Entry Only (Stage 1 only)
    print("\n" + "="*70)
    print("Running ML Entry Only (Stage 1 + Fixed SL/TP)...")
    print("="*70)
    
    class MLEntryOnlyBacktester(ThreeStageBacktester):
        def predict_sl(self, row):
            return 0.02  # Fixed 2%
        
        def predict_tp(self, row, sl_pct=None):
            return 0.04  # Fixed 4%
    
    ml_entry_bt = MLEntryOnlyBacktester(config)
    results['ML Entry Only'] = ml_entry_bt.run_backtest(df)
    ml_entry_bt.print_results(results['ML Entry Only'])
    
    # Comparison table
    print("\n" + "="*70)
    print("STRATEGY COMPARISON")
    print("="*70)
    print(f"\n{'Strategy':<25} {'Trades':>8} {'Win%':>8} {'Return':>10} {'MaxDD':>8} {'Sharpe':>8} {'PF':>8}")
    print("-"*70)
    
    for name, res in results.items():
        print(f"{name:<25} {res.total_trades:>8} {res.win_rate:>7.1%} "
              f"{res.total_return:>9.1%} {res.max_drawdown:>7.1%} "
              f"{res.sharpe_ratio:>7.2f} {res.profit_factor:>7.2f}")
    
    return results


def run_leverage_comparison(df: pd.DataFrame, base_config: BacktestConfig) -> Dict[str, BacktestResult]:
    """
    Compare different leverage levels (1x, 3x, 5x, 7x, 10x).
    """
    leverage_levels = [1, 3, 5, 7, 10]
    results = {}
    
    print("\n" + "="*70)
    print("LEVERAGE COMPARISON TEST")
    print("="*70)
    
    for lev in leverage_levels:
        print(f"\n{'='*70}")
        print(f"Testing {lev}x Leverage...")
        print("="*70)
        
        config = BacktestConfig(
            initial_capital=base_config.initial_capital,
            risk_per_trade=base_config.risk_per_trade,
            entry_threshold=base_config.entry_threshold,
            fee_rate=base_config.fee_rate,
            slippage=base_config.slippage,
            fixed_position_size=base_config.fixed_position_size,
            position_size_usd=base_config.position_size_usd,
            leverage=lev
        )
        
        backtester = ThreeStageBacktester(config)
        result = backtester.run_backtest(df, verbose=False)
        results[f'{lev}x'] = result
        
        # Count liquidations
        liquidations = sum(1 for t in result.trades if t.exit_reason == 'LIQUIDATED')
        
        print(f"   Trades: {result.total_trades}, Win Rate: {result.win_rate:.1%}")
        print(f"   Return: {result.total_return:.1%}, Max DD: {result.max_drawdown:.1%}")
        print(f"   Liquidations: {liquidations}")
    
    # Summary table
    print("\n" + "="*80)
    print("LEVERAGE COMPARISON SUMMARY")
    print("="*80)
    print(f"\n{'Leverage':<10} {'Trades':>8} {'Win%':>8} {'Return':>12} {'MaxDD':>10} {'Sharpe':>8} {'PF':>8} {'Liq':>6}")
    print("-"*80)
    
    for lev in leverage_levels:
        res = results[f'{lev}x']
        liquidations = sum(1 for t in res.trades if t.exit_reason == 'LIQUIDATED')
        print(f"{lev}x{'':<8} {res.total_trades:>8} {res.win_rate:>7.1%} "
              f"{res.total_return:>11.1%} {res.max_drawdown:>9.1%} "
              f"{res.sharpe_ratio:>7.2f} {res.profit_factor:>7.2f} {liquidations:>6}")
    
    # Risk-adjusted comparison
    print("\n" + "="*80)
    print("RISK-ADJUSTED METRICS")
    print("="*80)
    print(f"\n{'Leverage':<10} {'Return/DD':>12} {'Calmar':>10} {'Final Equity':>15}")
    print("-"*60)
    
    for lev in leverage_levels:
        res = results[f'{lev}x']
        return_dd_ratio = res.total_return / res.max_drawdown if res.max_drawdown > 0 else float('inf')
        # Calmar ratio (annual return / max DD) - simplified
        calmar = (res.total_return / 6) / res.max_drawdown if res.max_drawdown > 0 else float('inf')  # ~6 years
        final_equity = res.equity_curve[-1] if res.equity_curve else base_config.initial_capital
        print(f"{lev}x{'':<8} {return_dd_ratio:>11.2f}x {calmar:>9.2f} ${final_equity:>14,.2f}")
    
    # Plot leverage comparison
    plot_leverage_comparison(results, base_config.initial_capital, 
                            save_path=str(DATA_DIR.parent / 'backtest_leverage_comparison.png'))
    
    return results


def main():
    """Main function to run backtest."""
    import argparse
    
    parser = argparse.ArgumentParser(description='3-Stage ML Backtest')
    parser.add_argument('--data', type=str, default=None, help='Path to data file')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--risk', type=float, default=0.01, help='Risk per trade (0.01 = 1%)')
    parser.add_argument('--threshold', type=float, default=0.65, help='Entry confidence threshold')
    parser.add_argument('--fee', type=float, default=0.001, help='Fee rate (0.001 = 0.1%)')
    parser.add_argument('--slippage', type=float, default=0.0005, help='Slippage (0.0005 = 0.05%)')
    parser.add_argument('--kelly', action='store_true', help='Use Kelly Criterion')
    parser.add_argument('--fixed-size', action='store_true', help='Use fixed position size')
    parser.add_argument('--size-usd', type=float, default=1000, help='Fixed position size in USD')
    parser.add_argument('--leverage', type=float, default=1.0, help='Leverage multiplier (1, 3, 5, 7, 10)')
    parser.add_argument('--compare', action='store_true', help='Run baseline comparison')
    parser.add_argument('--leverage-test', action='store_true', help='Test multiple leverage levels')
    
    args = parser.parse_args()
    
    # Load data
    if args.data:
        data_path = Path(args.data)
    else:
        data_path = PROCESSED_DIR / 'features_1d_full.parquet'
    
    if not data_path.exists():
        print(f"Data not found: {data_path}")
        print("Run multi_timeframe_pipeline.py first!")
        return
    
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} rows from {data_path.name}")
    
    # Filter to test period (last 20% of data)
    # This avoids testing on training data
    test_start_idx = int(len(df) * 0.8)
    df_test = df.iloc[test_start_idx:].copy()
    print(f"Test period: {len(df_test):,} rows ({df_test['timestamp'].min()} to {df_test['timestamp'].max()})")
    
    # Configure backtest
    config = BacktestConfig(
        initial_capital=args.capital,
        risk_per_trade=args.risk,
        entry_threshold=args.threshold,
        fee_rate=args.fee,
        slippage=args.slippage,
        use_kelly=args.kelly,
        fixed_position_size=args.fixed_size,
        position_size_usd=args.size_usd,
        leverage=args.leverage
    )
    
    # Test multiple leverage levels
    if args.leverage_test:
        run_leverage_comparison(df_test, config)
        return
    
    if args.compare:
        # Run comparison with baselines
        results = run_baseline_comparison(df_test, config)
        # Plot comparison
        plot_equity_curve(results, title='Strategy Comparison',
                         save_path=str(DATA_DIR.parent / 'backtest_strategy_comparison.png'))
    else:
        # Run single backtest
        backtester = ThreeStageBacktester(config)
        result = backtester.run_backtest(df_test)
        backtester.print_results(result)
        # Plot single result
        lev_str = f"{config.leverage:.0f}x" if config.leverage > 1 else "1x"
        plot_equity_curve({f'3-Stage ML ({lev_str})': result}, 
                         title=f'3-Stage ML Backtest ({lev_str} Leverage)',
                         save_path=str(DATA_DIR.parent / f'backtest_equity_{lev_str}.png'))


if __name__ == '__main__':
    main()
