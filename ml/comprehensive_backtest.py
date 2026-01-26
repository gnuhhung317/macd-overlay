"""
Comprehensive Backtest Runner with Signal Ranking
- Walk through time chronologically
- At each timestamp: check open positions, find best signals to fill up to max
- Rank signals by ML confidence, pick top N
- Test multiple timeframes × leverages × thresholds
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

from backtest_3stage import ThreeStageBacktester, BacktestConfig, Trade, BacktestResult
import matplotlib.pyplot as plt
import joblib

# ============== CONFIGURATION ==============
TIMEFRAMES = ['1h', '4h', '8h', '12h', '1d']
LEVERAGES = [5, 7]  # Only test 1x, 5x, 7x leverage
THRESHOLDS = [0.65, 0.75]
POSITION_SIZING = ['fixed', 'percent']  # fixed=$1000, percent=5% of equity

FIXED_SIZE_USD = 1000
PERCENT_OF_EQUITY = 0.05
INITIAL_CAPITAL = 10000
MAX_POSITIONS = 10  # Maximum concurrent positions

# Date range filter (set to None to use all data)
# START_DATE = '2025-12-01'  # Format: 'YYYY-MM-DD' or None
# END_DATE = '2026-12-31'    # Format: 'YYYY-MM-DD' or None

START_DATE = '2025-04-01'  # Format: 'YYYY-MM-DD' or None
END_DATE = '2025-06-30'    # Format: 'YYYY-MM-DD' or None

@dataclass
class OpenPosition:
    """Track an open position"""
    trade: Trade
    margin: float
    entry_time: datetime
    expected_exit_time: datetime
    symbol: str


class ImprovedBacktester:
    """
    Improved backtester that walks through time and ranks signals.
    
    At each timestamp:
    1. Close expired/exited positions
    2. Count open positions
    3. If open < max_positions, find ALL signals at this timestamp
    4. Rank signals by ML confidence (descending)
    5. Pick top N signals to fill remaining slots
    """
    
    def __init__(self, config: BacktestConfig, model_dir: str):
        self.config = config
        self.model_dir = model_dir
        
        # Load models
        self.entry_model = None
        self.sl_model = None
        self.tp_model = None
        self.entry_scaler = None
        self.sl_scaler = None
        self.tp_scaler = None
        self.entry_features = None
        self.sl_features = None
        self.tp_features = None
        self.tp_predict_rr = False
        
        self._load_models()
    
    def _load_models(self):
        """Load ML models from directory"""
        try:
            # Entry Filter
            entry_path = os.path.join(self.model_dir, 'entry_filter.joblib')
            if os.path.exists(entry_path):
                data = joblib.load(entry_path)
                self.entry_model = data['model']
                self.entry_scaler = data.get('scaler')
                self.entry_features = data['feature_names']
            
            # SL Predictor
            sl_path = os.path.join(self.model_dir, 'sl_predictor.joblib')
            if os.path.exists(sl_path):
                data = joblib.load(sl_path)
                self.sl_model = data['model']
                self.sl_scaler = data.get('scaler')
                self.sl_features = data['feature_names']
            
            # TP Predictor
            tp_path = os.path.join(self.model_dir, 'tp_predictor.joblib')
            if os.path.exists(tp_path):
                data = joblib.load(tp_path)
                self.tp_model = data['model']
                self.tp_scaler = data.get('scaler')
                self.tp_features = data['feature_names']
                self.tp_predict_rr = data.get('predict_rr', False)
                
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def _prepare_features(self, row: pd.Series, feature_names: list, scaler) -> np.ndarray:
        """Prepare features for prediction"""
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
    
    def predict_entry_confidence(self, row: pd.Series) -> float:
        """Get entry confidence score (0-1)"""
        if self.entry_model is None:
            return 0.5
        
        X = self._prepare_features(row, self.entry_features, self.entry_scaler)
        proba = self.entry_model.predict_proba(X)[0, 1]
        return proba
    
    def predict_sl(self, row: pd.Series) -> float:
        """Predict optimal SL percentage"""
        if self.sl_model is None:
            return 0.02
        
        X = self._prepare_features(row, self.sl_features, self.sl_scaler)
        sl_pct = self.sl_model.predict(X)[0]
        return np.clip(sl_pct, 0.005, 0.15)
    
    def predict_tp(self, row: pd.Series, sl_pct: float = None) -> float:
        """Predict optimal TP percentage"""
        if self.tp_model is None:
            return 0.04
        
        X = self._prepare_features(row, self.tp_features, self.tp_scaler)
        pred = self.tp_model.predict(X)[0]
        
        if self.tp_predict_rr and sl_pct is not None:
            tp_pct = pred * sl_pct
        else:
            tp_pct = pred
        
        return np.clip(tp_pct, 0.01, 0.30)
    
    def simulate_trade(
        self,
        entry_row: pd.Series,
        future_data: pd.DataFrame,
        entry_price: float,
        sl_pct: float,
        tp_pct: float,
        direction: str,
        position_size: float,
        confidence: float
    ) -> Trade:
        """Simulate a single trade"""
        trade = Trade(
            symbol=entry_row.get('symbol', 'UNKNOWN'),
            entry_time=entry_row.get('timestamp', datetime.now()),
            direction=direction,
            entry_price=entry_price,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            position_size=position_size,
            confidence=confidence
        )
        
        # Apply slippage to entry
        if direction == 'LONG':
            trade.entry_price *= (1 + self.config.slippage)
            trade.sl_price = trade.entry_price * (1 - sl_pct)
            trade.tp_price = trade.entry_price * (1 + tp_pct)
        else:
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
                sl_hit = low <= trade.sl_price
                tp_hit = high >= trade.tp_price
                
                if sl_hit and tp_hit:
                    trade.exit_price = trade.sl_price * (1 - self.config.slippage)
                    trade.exit_reason = 'SL_HIT'
                elif sl_hit:
                    trade.exit_price = trade.sl_price * (1 - self.config.slippage)
                    trade.exit_reason = 'SL_HIT'
                elif tp_hit:
                    trade.exit_price = trade.tp_price * (1 - self.config.slippage)
                    trade.exit_reason = 'TP_HIT'
            else:
                sl_hit = high >= trade.sl_price
                tp_hit = low <= trade.tp_price
                
                if sl_hit and tp_hit:
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
            
            # Max bars timeout
            if trade.bars_held >= self.config.max_bars:
                trade.exit_price = close * (1 - self.config.slippage if direction == 'LONG' else 1 + self.config.slippage)
                trade.exit_reason = 'TIMEOUT'
                trade.exit_time = row.get('timestamp', datetime.now())
                break
        
        # Calculate PnL
        if trade.exit_price > 0:
            if direction == 'LONG':
                trade.pnl_pct = (trade.exit_price - trade.entry_price) / trade.entry_price
            else:
                trade.pnl_pct = (trade.entry_price - trade.exit_price) / trade.entry_price
            
            exit_fee = position_size * (1 + trade.pnl_pct) * self.config.fee_rate
            trade.fees_paid += exit_fee
            
            margin = position_size / self.config.leverage
            trade.pnl = position_size * trade.pnl_pct - trade.fees_paid
            
            # Check liquidation
            max_loss = -margin * self.config.liquidation_threshold
            if trade.pnl < max_loss:
                trade.pnl = -margin
                trade.exit_reason = 'LIQUIDATED'
        
        return trade
    
    def run_backtest(self, df: pd.DataFrame, verbose: bool = False) -> BacktestResult:
        """
        Run backtest with signal ranking at each timestamp.
        
        Walk through time:
        1. At each unique timestamp with signals
        2. Close any expired positions first
        3. Count available slots (max_positions - open_count)
        4. Get all signals at this timestamp, compute confidence
        5. Rank by confidence, pick top N to fill slots
        6. Open those positions
        """
        result = BacktestResult()
        capital = self.config.initial_capital
        available_capital = capital
        
        open_positions: Dict[int, OpenPosition] = {}
        trade_counter = 0
        
        # ===== STEP 1: Collect ALL crossover signals =====
        all_signals = []
        symbols = df['symbol'].unique() if 'symbol' in df.columns else ['UNKNOWN']
        
        # Pre-compute data by symbol for quick lookup
        symbol_data = {}
        for symbol in symbols:
            symbol_data[symbol] = df[df['symbol'] == symbol].sort_values('timestamp').reset_index(drop=True)
        
        for symbol in symbols:
            df_symbol = symbol_data[symbol]
            
            # Find crossover signals
            crossover_mask = (df_symbol['macd_cross_up'] == 1) | (df_symbol['macd_cross_down'] == 1)
            
            for idx in df_symbol[crossover_mask].index:
                if idx >= len(df_symbol) - self.config.max_bars:
                    continue
                
                row = df_symbol.iloc[idx]
                future_data = df_symbol.iloc[idx + 1: idx + 1 + self.config.max_bars]
                
                if len(future_data) == 0:
                    continue
                
                # Get entry confidence score
                confidence = self.predict_entry_confidence(row)
                
                all_signals.append({
                    'timestamp': row['timestamp'],
                    'symbol': symbol,
                    'row': row,
                    'future_data': future_data,
                    'is_long': row.get('macd_cross_up', 0) == 1,
                    'confidence': confidence,
                    'idx': idx
                })
        
        if verbose:
            print(f"Found {len(all_signals)} total crossover signals")
        
        # ===== STEP 2: Group signals by timestamp =====
        signals_by_time = {}
        for sig in all_signals:
            ts = sig['timestamp']
            if ts not in signals_by_time:
                signals_by_time[ts] = []
            signals_by_time[ts].append(sig)
        
        # Sort timestamps chronologically
        sorted_timestamps = sorted(signals_by_time.keys())
        
        if verbose:
            print(f"Signals spread across {len(sorted_timestamps)} unique timestamps")
        
        # ===== STEP 3: Walk through time =====
        trades_opened = 0
        trades_skipped_confidence = 0
        trades_skipped_capital = 0
        trades_skipped_max_pos = 0
        
        for current_time in sorted_timestamps:
            # ----- Close expired positions -----
            closed_ids = []
            for pos_id, pos in open_positions.items():
                if pos.expected_exit_time <= current_time:
                    capital += pos.trade.pnl
                    available_capital += pos.margin + pos.trade.pnl
                    result.trades.append(pos.trade)
                    closed_ids.append(pos_id)
            
            for pid in closed_ids:
                del open_positions[pid]
            
            # ----- Get signals at this timestamp -----
            signals_now = signals_by_time[current_time]
            
            # ----- Filter by confidence threshold -----
            valid_signals = [s for s in signals_now if s['confidence'] >= self.config.entry_threshold]
            
            if not valid_signals:
                trades_skipped_confidence += len(signals_now)
                continue
            
            # ----- Rank by confidence (descending) -----
            valid_signals.sort(key=lambda x: x['confidence'], reverse=True)
            
            # ----- Check available slots -----
            current_open = len(open_positions)
            available_slots = self.config.max_open_trades - current_open
            
            if available_slots <= 0:
                trades_skipped_max_pos += len(valid_signals)
                continue
            
            # ----- Get symbols already in position -----
            symbols_in_position = {pos.symbol for pos in open_positions.values()}
            
            # ----- Pick top N signals (that aren't already in position) -----
            signals_to_open = []
            for sig in valid_signals:
                if sig['symbol'] in symbols_in_position:
                    continue  # Skip if already have position in this symbol
                if len(signals_to_open) >= available_slots:
                    break
                signals_to_open.append(sig)
                symbols_in_position.add(sig['symbol'])  # Mark as taken
            
            # ----- Open positions for selected signals -----
            for sig in signals_to_open:
                row = sig['row']
                future_data = sig['future_data']
                symbol = sig['symbol']
                is_long = sig['is_long']
                confidence = sig['confidence']
                
                # Skip shorts if not allowed
                if not is_long and not self.config.allow_shorts:
                    continue
                
                direction = 'LONG' if is_long else 'SHORT'
                
                # Predict SL/TP
                sl_pct = self.predict_sl(row)
                tp_pct = self.predict_tp(row, sl_pct)
                
                # Calculate position size
                if self.config.fixed_position_size:
                    margin_needed = self.config.position_size_usd
                    position_size = margin_needed * self.config.leverage
                    
                    if margin_needed > available_capital * 0.95:
                        trades_skipped_capital += 1
                        continue
                else:
                    risk_amount = available_capital * self.config.risk_per_trade
                    if sl_pct > 0:
                        position_size = risk_amount / sl_pct * self.config.leverage
                    else:
                        position_size = risk_amount / 0.02 * self.config.leverage
                    
                    max_position = available_capital * self.config.max_concentration * self.config.leverage
                    position_size = min(position_size, max_position)
                    margin_needed = position_size / self.config.leverage
                
                if position_size <= 0 or margin_needed <= 0:
                    continue
                
                # Simulate trade
                entry_price = row['close']
                trade = self.simulate_trade(
                    row, future_data, entry_price,
                    sl_pct, tp_pct, direction, position_size, confidence
                )
                
                # Track open position
                trade_counter += 1
                open_positions[trade_counter] = OpenPosition(
                    trade=trade,
                    margin=margin_needed,
                    entry_time=current_time,
                    expected_exit_time=trade.exit_time or (current_time + pd.Timedelta(days=self.config.max_bars)),
                    symbol=symbol
                )
                
                available_capital -= margin_needed
                trades_opened += 1
        
        # ===== STEP 4: Close remaining positions =====
        for pos_id, pos in open_positions.items():
            capital += pos.trade.pnl
            result.trades.append(pos.trade)
        
        # ===== STEP 5: Build equity curve =====
        result.trades.sort(key=lambda t: t.exit_time or t.entry_time)
        
        result.equity_curve = [self.config.initial_capital]
        running_capital = self.config.initial_capital
        for trade in result.trades:
            running_capital += trade.pnl
            result.equity_curve.append(running_capital)
            result.timestamps.append(trade.exit_time or trade.entry_time)
        
        if verbose:
            print(f"  Trades opened: {trades_opened}")
            print(f"  Skipped (low confidence): {trades_skipped_confidence}")
            print(f"  Skipped (max positions): {trades_skipped_max_pos}")
            print(f"  Skipped (no capital): {trades_skipped_capital}")
            print(f"  Final capital: ${capital:,.2f}")
        
        # Calculate metrics
        self._calculate_metrics(result)
        
        return result
    
    def _calculate_metrics(self, result: BacktestResult):
        """Calculate summary metrics"""
        if not result.trades:
            return
        
        trades = result.trades
        result.total_trades = len(trades)
        
        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]
        result.winning_trades = len(winners)
        result.losing_trades = len(losers)
        result.win_rate = len(winners) / len(trades) if trades else 0
        
        result.total_pnl = sum(t.pnl for t in trades)
        result.total_return = result.total_pnl / self.config.initial_capital
        result.avg_trade_pnl = result.total_pnl / len(trades)
        result.avg_winner = np.mean([t.pnl for t in winners]) if winners else 0
        result.avg_loser = np.mean([t.pnl for t in losers]) if losers else 0
        result.best_trade = max(t.pnl for t in trades)
        result.worst_trade = min(t.pnl for t in trades)
        result.total_fees = sum(t.fees_paid for t in trades)
        result.avg_bars_held = np.mean([t.bars_held for t in trades])
        
        gross_profit = sum(t.pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 1
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        equity = result.equity_curve
        peak = equity[0]
        max_dd = 0
        for eq in equity:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)
        result.max_drawdown = max_dd
        
        if len(equity) > 1:
            returns = np.diff(equity) / np.array(equity[:-1])
            if np.std(returns) > 0:
                result.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                result.sharpe_ratio = 0


def check_models_exist(timeframe: str) -> bool:
    """Check if models exist for a timeframe"""
    # Check timeframe-specific folder first
    model_dir = os.path.join(os.path.dirname(__file__), 'models', timeframe)
    if os.path.exists(model_dir):
        required_files = ['entry_filter.joblib', 'sl_predictor.joblib', 'tp_predictor.joblib']
        for f in required_files:
            if not os.path.exists(os.path.join(model_dir, f)):
                return False
        return True
    
    # For 1d, also check root models folder (legacy)
    if timeframe == '1d':
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        required_files = ['entry_filter.joblib', 'sl_predictor.joblib', 'tp_predictor.joblib']
        for f in required_files:
            if not os.path.exists(os.path.join(model_dir, f)):
                return False
        return True
    
    return False


def get_model_dir(timeframe: str) -> str:
    """Get correct model directory for timeframe"""
    # Check timeframe-specific folder first
    model_dir = os.path.join(os.path.dirname(__file__), 'models', timeframe)
    if os.path.exists(model_dir):
        return model_dir
    
    # For 1d, also check root models folder (legacy)
    if timeframe == '1d':
        return os.path.join(os.path.dirname(__file__), 'models')
    
    return model_dir


def run_single_backtest(
    timeframe: str,
    leverage: int,
    threshold: float,
    sizing_type: str,
    verbose: bool = False
) -> dict:
    """Run a single backtest configuration with signal ranking"""
    
    # Setup model paths
    model_dir = get_model_dir(timeframe)
    
    # Config
    config = BacktestConfig(
        initial_capital=INITIAL_CAPITAL,
        leverage=leverage,
        fee_rate=0.0004,
        slippage=0.0002,
        max_bars=50,
        use_kelly=False,
        risk_per_trade=0.02,
        max_concentration=0.25,
        entry_threshold=threshold,
        max_open_trades=MAX_POSITIONS,
        fixed_position_size=(sizing_type == 'fixed'),
        position_size_usd=FIXED_SIZE_USD if sizing_type == 'fixed' else INITIAL_CAPITAL * PERCENT_OF_EQUITY
    )
    
    # Create improved backtester with signal ranking
    try:
        backtester = ImprovedBacktester(config, model_dir)
    except Exception as e:
        return {
            'timeframe': timeframe,
            'leverage': leverage,
            'threshold': threshold,
            'sizing': sizing_type,
            'error': f'Error initializing backtester: {e}',
            'total_trades': 0
        }
    
    # Load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')
    data_file = os.path.join(data_dir, f'features_{timeframe}_full.parquet')
    
    if not os.path.exists(data_file):
        data_file = os.path.join(data_dir, f'crossover_data_{timeframe}.parquet')
    
    if not os.path.exists(data_file):
        return {
            'timeframe': timeframe,
            'leverage': leverage,
            'threshold': threshold,
            'sizing': sizing_type,
            'error': f'Data file not found',
            'total_trades': 0
        }
    
    try:
        df = pd.read_parquet(data_file)
    except Exception as e:
        return {
            'timeframe': timeframe,
            'leverage': leverage,
            'threshold': threshold,
            'sizing': sizing_type,
            'error': f'Error loading data: {e}',
            'total_trades': 0
        }
    
    # Use all symbols in dataset
    symbols = df['symbol'].unique() if 'symbol' in df.columns else ['UNKNOWN']
    
    # Filter by date range if specified
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if START_DATE is not None:
            start_dt = pd.to_datetime(START_DATE)
            df = df[df['timestamp'] >= start_dt]
        
        if END_DATE is not None:
            end_dt = pd.to_datetime(END_DATE)
            df = df[df['timestamp'] <= end_dt]
    
    if len(df) == 0:
        return {
            'timeframe': timeframe,
            'leverage': leverage,
            'threshold': threshold,
            'sizing': sizing_type,
            'error': f'No data in date range {START_DATE} to {END_DATE}',
            'total_trades': 0
        }
    
    # Get actual date range from filtered data
    start_date = df['timestamp'].min() if 'timestamp' in df.columns else None
    end_date = df['timestamp'].max() if 'timestamp' in df.columns else None
    
    # Run improved backtest with signal ranking
    try:
        results = backtester.run_backtest(df, verbose=verbose)
        
        # Get actual trade date range
        if results.trades:
            trade_start = min(t.entry_time for t in results.trades)
            trade_end = max(t.exit_time or t.entry_time for t in results.trades)
        else:
            trade_start = start_date
            trade_end = end_date
        
        # Extract metrics
        return {
            'timeframe': timeframe,
            'leverage': leverage,
            'threshold': threshold,
            'sizing': sizing_type,
            'start_date': trade_start.strftime('%Y-%m-%d') if trade_start else 'N/A',
            'end_date': trade_end.strftime('%Y-%m-%d') if trade_end else 'N/A',
            'total_trades': results.total_trades,
            'win_rate': results.win_rate * 100 if results.win_rate < 1 else results.win_rate,
            'total_return_pct': results.total_return * 100 if results.total_return < 10 else results.total_return,
            'max_drawdown_pct': results.max_drawdown * 100 if results.max_drawdown < 1 else results.max_drawdown,
            'sharpe_ratio': results.sharpe_ratio,
            'profit_factor': results.profit_factor,
            'avg_trade_pct': results.avg_trade_pnl / INITIAL_CAPITAL * 100 if results.avg_trade_pnl else 0,
            'final_equity': INITIAL_CAPITAL + results.total_pnl,
            'avg_confidence': np.mean([t.confidence for t in results.trades]) if results.trades else 0,
            'liquidations': len([t for t in results.trades if t.exit_reason == 'LIQUIDATED']),
            'num_symbols': len(symbols),
            'error': None
        }
        
    except Exception as e:
        import traceback
        return {
            'timeframe': timeframe,
            'leverage': leverage,
            'threshold': threshold,
            'sizing': sizing_type,
            'error': f'{str(e)}: {traceback.format_exc()[:200]}',
            'total_trades': 0
        }


def run_comprehensive_backtest():
    """Run all backtest combinations with signal ranking"""
    
    print("=" * 80)
    print("COMPREHENSIVE BACKTEST RUNNER (with Signal Ranking)")
    print("=" * 80)
    print(f"\nTimeframes: {TIMEFRAMES}")
    print(f"Leverages: {LEVERAGES}")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Position Sizing: {POSITION_SIZING}")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,}")
    print(f"Fixed Size: ${FIXED_SIZE_USD:,}")
    print(f"Percent of Equity: {PERCENT_OF_EQUITY*100}%")
    print(f"Max Positions: {MAX_POSITIONS}")
    print(f"Symbols: All symbols in dataset")
    print(f"Date Range: {START_DATE or 'Beginning'} to {END_DATE or 'End'}")
    print("\n📌 Signal Ranking: At each timestamp, signals are ranked by ML confidence")
    print("   and top signals are selected to fill available position slots.")
    
    # Check available timeframes
    available_timeframes = []
    for tf in TIMEFRAMES:
        if check_models_exist(tf):
            available_timeframes.append(tf)
            print(f"  ✓ {tf}: Models found")
        else:
            print(f"  ✗ {tf}: Models NOT found - skipping")
    
    if not available_timeframes:
        print("\n❌ No models available! Please train models first.")
        return
    
    # Calculate total combinations
    total_combinations = len(available_timeframes) * len(LEVERAGES) * len(THRESHOLDS) * len(POSITION_SIZING)
    print(f"\nTotal combinations to test: {total_combinations}")
    print("=" * 80)
    
    results = []
    completed = 0
    
    for tf, lev, thresh, sizing in product(available_timeframes, LEVERAGES, THRESHOLDS, POSITION_SIZING):
        completed += 1
        print(f"\n[{completed}/{total_combinations}] Testing: {tf} | {lev}x | thresh={thresh} | {sizing}")
        
        result = run_single_backtest(tf, lev, thresh, sizing)
        results.append(result)
        
        if result.get('error'):
            print(f"  ❌ Error: {result['error']}")
        else:
            print(f"  ✓ Trades: {result['total_trades']}, Return: {result['total_return_pct']:.1f}%, "
                  f"MaxDD: {result['max_drawdown_pct']:.1f}%, Sharpe: {result['sharpe_ratio']:.2f}")
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Filter successful results
    df_success = df_results[df_results['error'].isna() & (df_results['total_trades'] > 0)]
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    if len(df_success) == 0:
        print("No successful backtests!")
        return df_results
    
    # Sort by Sharpe ratio
    df_sorted = df_success.sort_values('sharpe_ratio', ascending=False)
    
    # Print date range
    if 'start_date' in df_success.columns:
        sample_row = df_success.iloc[0]
        print(f"\n📅 Data Period: {sample_row.get('start_date', 'N/A')} to {sample_row.get('end_date', 'N/A')}")
        print(f"📊 Symbols tested: {sample_row.get('num_symbols', 'N/A')}")
    
    # Top 10 configurations
    print("\n📊 TOP 10 CONFIGURATIONS (by Sharpe Ratio):")
    print("-" * 130)
    print(f"{'Rank':<5} {'TF':<5} {'Lev':<5} {'Thresh':<7} {'Size':<8} {'Trades':<8} {'Return%':<10} {'MaxDD%':<8} {'Sharpe':<8} {'PF':<8} {'AvgConf':<8} {'Period':<20}")
    print("-" * 130)
    
    for i, (_, row) in enumerate(df_sorted.head(10).iterrows(), 1):
        avg_conf = row.get('avg_confidence', 0) * 100
        period = f"{row.get('start_date', '')[:10]}~{row.get('end_date', '')[:10]}"
        print(f"{i:<5} {row['timeframe']:<5} {row['leverage']:<5} {row['threshold']:<7.2f} "
              f"{row['sizing']:<8} {row['total_trades']:<8} {row['total_return_pct']:<10.1f} "
              f"{row['max_drawdown_pct']:<8.1f} {row['sharpe_ratio']:<8.2f} {row['profit_factor']:<8.2f} {avg_conf:<7.1f}% {period:<20}")
    
    # Best by each metric
    print("\n🏆 BEST BY METRIC:")
    print("-" * 60)
    
    metrics_to_check = [
        ('total_return_pct', 'Highest Return', True),
        ('sharpe_ratio', 'Best Sharpe Ratio', True),
        ('profit_factor', 'Best Profit Factor', True),
        ('max_drawdown_pct', 'Lowest Max DD', False),
        ('win_rate', 'Highest Win Rate', True)
    ]
    
    for metric, label, ascending in metrics_to_check:
        if ascending:
            best = df_success.loc[df_success[metric].idxmax()]
        else:
            best = df_success.loc[df_success[metric].idxmin()]
        
        print(f"\n{label}:")
        print(f"  {best['timeframe']} | {best['leverage']}x | thresh={best['threshold']} | {best['sizing']}")
        print(f"  Return: {best['total_return_pct']:.1f}%, MaxDD: {best['max_drawdown_pct']:.1f}%, "
              f"Sharpe: {best['sharpe_ratio']:.2f}, PF: {best['profit_factor']:.2f}")
    
    # Aggregate analysis
    print("\n\n📈 AGGREGATE ANALYSIS:")
    print("=" * 80)
    
    # By Timeframe
    print("\n📅 By Timeframe:")
    tf_agg = df_success.groupby('timeframe').agg({
        'total_trades': 'mean',
        'total_return_pct': 'mean',
        'max_drawdown_pct': 'mean',
        'sharpe_ratio': 'mean',
        'profit_factor': 'mean'
    }).round(2)
    print(tf_agg.to_string())
    
    # By Leverage
    print("\n💰 By Leverage:")
    lev_agg = df_success.groupby('leverage').agg({
        'total_trades': 'mean',
        'total_return_pct': 'mean',
        'max_drawdown_pct': 'mean',
        'sharpe_ratio': 'mean',
        'profit_factor': 'mean'
    }).round(2)
    print(lev_agg.to_string())
    
    # By Threshold
    print("\n🎯 By Threshold:")
    thresh_agg = df_success.groupby('threshold').agg({
        'total_trades': 'mean',
        'total_return_pct': 'mean',
        'max_drawdown_pct': 'mean',
        'sharpe_ratio': 'mean',
        'profit_factor': 'mean'
    }).round(2)
    print(thresh_agg.to_string())
    
    # By Position Sizing
    print("\n📐 By Position Sizing:")
    sizing_agg = df_success.groupby('sizing').agg({
        'total_trades': 'mean',
        'total_return_pct': 'mean',
        'max_drawdown_pct': 'mean',
        'sharpe_ratio': 'mean',
        'profit_factor': 'mean'
    }).round(2)
    print(sizing_agg.to_string())
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'comprehensive_backtest_results_{timestamp}.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    # Create visualization
    create_visualizations(df_success, timestamp)
    
    return df_results


def create_visualizations(df: pd.DataFrame, timestamp: str):
    """Create visualization charts"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Backtest Results', fontsize=14, fontweight='bold')
    
    # 1. Sharpe by Timeframe and Leverage
    ax1 = axes[0, 0]
    pivot1 = df.pivot_table(values='sharpe_ratio', index='timeframe', columns='leverage', aggfunc='mean')
    pivot1.plot(kind='bar', ax=ax1, colormap='viridis')
    ax1.set_title('Sharpe Ratio by Timeframe & Leverage')
    ax1.set_xlabel('Timeframe')
    ax1.set_ylabel('Sharpe Ratio')
    ax1.legend(title='Leverage')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Return by Threshold
    ax2 = axes[0, 1]
    pivot2 = df.pivot_table(values='total_return_pct', index='threshold', columns='sizing', aggfunc='mean')
    pivot2.plot(kind='bar', ax=ax2, colormap='coolwarm')
    ax2.set_title('Return % by Threshold & Sizing')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Return %')
    ax2.legend(title='Sizing')
    ax2.tick_params(axis='x', rotation=0)
    
    # 3. Risk-Reward scatter
    ax3 = axes[0, 2]
    for lev in df['leverage'].unique():
        subset = df[df['leverage'] == lev]
        ax3.scatter(subset['max_drawdown_pct'], subset['total_return_pct'], 
                   label=f'{lev}x', alpha=0.7, s=50)
    ax3.set_xlabel('Max Drawdown %')
    ax3.set_ylabel('Total Return %')
    ax3.set_title('Risk-Reward by Leverage')
    ax3.legend()
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 4. Profit Factor by Timeframe
    ax4 = axes[1, 0]
    pivot4 = df.pivot_table(values='profit_factor', index='timeframe', columns='threshold', aggfunc='mean')
    pivot4.plot(kind='bar', ax=ax4, colormap='plasma')
    ax4.set_title('Profit Factor by Timeframe & Threshold')
    ax4.set_xlabel('Timeframe')
    ax4.set_ylabel('Profit Factor')
    ax4.legend(title='Threshold')
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. Win Rate distribution
    ax5 = axes[1, 1]
    df.boxplot(column='win_rate', by='leverage', ax=ax5)
    ax5.set_title('Win Rate Distribution by Leverage')
    ax5.set_xlabel('Leverage')
    ax5.set_ylabel('Win Rate %')
    plt.suptitle('')
    
    # 6. Heatmap: Timeframe vs Leverage (Sharpe)
    ax6 = axes[1, 2]
    pivot6 = df.pivot_table(values='sharpe_ratio', index='timeframe', columns='leverage', aggfunc='mean')
    im = ax6.imshow(pivot6.values, cmap='RdYlGn', aspect='auto')
    ax6.set_xticks(range(len(pivot6.columns)))
    ax6.set_yticks(range(len(pivot6.index)))
    ax6.set_xticklabels(pivot6.columns)
    ax6.set_yticklabels(pivot6.index)
    ax6.set_xlabel('Leverage')
    ax6.set_ylabel('Timeframe')
    ax6.set_title('Sharpe Ratio Heatmap')
    plt.colorbar(im, ax=ax6)
    
    # Add values to heatmap
    for i in range(len(pivot6.index)):
        for j in range(len(pivot6.columns)):
            val = pivot6.values[i, j]
            if not np.isnan(val):
                ax6.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9)
    
    plt.tight_layout()
    chart_file = f'comprehensive_backtest_charts_{timestamp}.png'
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Charts saved to: {chart_file}")


if __name__ == '__main__':
    run_comprehensive_backtest()
