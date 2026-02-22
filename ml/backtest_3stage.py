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
from copy import deepcopy
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
try:
    import mplfinance as mpf
except ImportError:
    mpf = None
    print("⚠️ mplfinance not installed. Install with: pip install mplfinance")
import warnings
warnings.filterwarnings('ignore')
from config import SUPPORTED_TIMEFRAMES, get_timeframe_config

DATA_DIR = Path(__file__).parent.parent / 'bitget-data'
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
    initial_capital: float = 100
    risk_per_trade: float = 0.01  # 1% risk per trade
    max_concentration: float = 0.20  # Max 20% in single coin
    entry_threshold: float = 0.65  # Min confidence to enter
    fee_rate: float = 0.001  # 0.1% per trade (entry + exit = 0.2%)
    slippage: float = 0.0005  # 0.05% slippage
    max_bars: int = 10  # Max bars to hold
    use_kelly: bool = True  # Use Kelly Criterion for sizing
    kelly_fraction: float = 0.5  # Half-Kelly for safety
    allow_shorts: bool = True  # Allow short positions
    max_open_trades: int = 10  # Max concurrent trades
    fixed_position_size: bool = False  # If True, use fixed $ amount instead of % of equity
    position_size_usd: float = 1000  # Fixed position size in USD
    leverage: float = 1.0  # Leverage multiplier (1x, 3x, 5x, 7x, 10x)
    liquidation_threshold: float = 0.80  # Liquidation at 80% margin loss
    timeframe: Optional[str] = None  # e.g., '1d', '4h', '12h' - if set, filter df by 'timeframe' column
    require_fresh_crossover_after_exit: bool = True  # Require a fresh MACD crossover after a trade exit before allowing a new entry for the same symbol
    margin_mode: str = 'ISOLATED'  # 'ISOLATED' or 'CROSS'
    use_trailing_stop: bool = False
    trailing_start_pct: float = 0.02  # Start trailing after 2% profit
    trailing_step_pct: float = 0.01   # Keep SL 1% away from peak
    max_position_size_usd: float = 10000.0  # Max size per trade in USD (hard cap)
    
    # limit entry options
    entry_pullback_pct: float = 0.0  # If > 0, place LIMIT order at (Price * (1 - pct)) instead of Market
    entry_pullback_timeout: int = 3  # Cancel limit order if not filled after N bars
    
    # scanner entry zone options
    use_scanner_filter: bool = False  # If True, filter trades using SmartScanner Entry Zone logic
    scanner_mae: float = 0.00  # Max Adverse Excursion for zone calculation
    scanner_mfe: float = 0.12  # Max Favorable Excursion for zone calculation
    scanner_lookback_days: int = 6  # Wait up to N days for a good entry zone
    allowed_zones: List[str] = field(default_factory=lambda: [ "DEEP MERGE"])

    # Date filtering
    start_date: Optional[str] = None
    end_date: Optional[str] = None


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
        """Load all 3 ML models based on timeframe configuration."""
        # Determine model directory based on timeframe
        if self.config.timeframe and self.config.timeframe in ['1d', '4h', '8h', '12h', '1h', '1w']:
            model_dir = MODEL_DIR / self.config.timeframe
            if not model_dir.exists():
                print(f"⚠️ Models for {self.config.timeframe} not found, falling back to default models")
                model_dir = MODEL_DIR
            else:
                print(f"📂 Using models for timeframe: {self.config.timeframe}")
        else:
            model_dir = MODEL_DIR
            if self.config.timeframe:
                print(f"⚠️ Unsupported timeframe {self.config.timeframe}, using default models")
        
        # Stage 1: Entry Filter
        entry_path = model_dir / 'entry_filter.joblib'
        if entry_path.exists():
            data = joblib.load(entry_path)
            self.entry_model = data['model']
            self.entry_scaler = data.get('scaler')
            self.entry_features = data['feature_names']
            print(f"✓ Stage 1 loaded: {len(self.entry_features)} features from {entry_path.parent.name}")
        else:
            print(f"⚠️ Entry filter not found at {entry_path}")
        
        # Stage 2: SL Predictor
        sl_path = model_dir / 'sl_predictor.joblib'
        if sl_path.exists():
            data = joblib.load(sl_path)
            self.sl_model = data['model']
            self.sl_scaler = data.get('scaler')
            self.sl_features = data['feature_names']
            print(f"✓ Stage 2 loaded: {len(self.sl_features)} features from {sl_path.parent.name}")
        else:
            print(f"⚠️ SL predictor not found at {sl_path}")
        
        # Stage 3: TP Predictor
        tp_path = model_dir / 'tp_predictor.joblib'
        if tp_path.exists():
            data = joblib.load(tp_path)
            self.tp_model = data['model']
            self.tp_scaler = data.get('scaler')
            self.tp_features = data['feature_names']
            self.tp_predict_rr = data.get('predict_rr', False)
            print(f"✓ Stage 3 loaded: {len(self.tp_features)} features from {tp_path.parent.name}")
        else:
            print(f"⚠️ TP predictor not found at {tp_path}")
    
    def _prepare_features(self, row: pd.Series, feature_names: list, scaler) -> np.ndarray:
        """Prepare features for prediction (Optimized)."""
        # Fast extraction using reindex for Series
        # This ensures all features are present in correct order
        # Reshape to (1, n_features) for scikit-learn
        vals = row.reindex(feature_names, fill_value=0).values.reshape(1, -1)
        
        # Handle Inf/NaN efficiently with numpy (faster than dataframe replace)
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Use scaler if available
        if scaler is not None:
             # Most scalers work fine with numpy arrays
             # To ensure column name compatibility if scaler was strict, we might need a DF,
             # but usually numpy is fine and much faster.
             return scaler.transform(vals)
        
        return vals
    
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
    
    def analyze_entry_zone(self, is_long: bool, signal_price: float, current_price: float) -> str:
        """
        Determine if the current price is in a good entry zone relative to the signal price.
        Logic ported from ml/scanner.py.
        """
        MAE = self.config.scanner_mae
        MFE = self.config.scanner_mfe
        
        status = "UNKNOWN"
        
        if is_long:
            limit_price = signal_price * (1 - MAE)
            profit_limit = signal_price * (1 + MFE * 0.5)
            
            if current_price < limit_price: status = "DEEP MERGE" 
            elif limit_price <= current_price <= signal_price: status = "DISCOUNT" 
            elif signal_price < current_price <= signal_price * 1.01: status = "GOOD ENTRY" 
            elif current_price > profit_limit: status = "TOO LATE" 
            else: status = "CHASING" 
        else:
            limit_price = signal_price * (1 + MAE)
            profit_limit = signal_price * (1 - MFE * 0.5)
            
            if current_price > limit_price: status = "DEEP MERGE"
            elif signal_price <= current_price <= limit_price: status = "DISCOUNT"
            elif signal_price * 0.99 <= current_price < signal_price: status = "GOOD ENTRY"
            elif current_price < profit_limit: status = "TOO LATE"
            else: status = "CHASING"
            
        return status

    def calculate_position_size(
        self, 
        capital: float, 
        sl_pct: float, 
        tp_pct: float,
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
            # Estimate RR from model
            if sl_pct > 0 and tp_pct > 0:
                estimated_rr = tp_pct / sl_pct
            else:
                estimated_rr = 2.0  # Fallback
                
            p = confidence  # Win probability from Stage 1
            q = 1 - p
            b = estimated_rr
            
            # Kelly fraction
            kelly_f = (p * b - q) / b
            kelly_f = max(0, min(kelly_f, 0.25))  # Cap at 25% (safety)
            
            # Use fractional Kelly
            kelly_f *= self.config.kelly_fraction
            
            # Adjust risk based on Kelly
            risk_amount = capital * kelly_f
        
        # Calculate position size from risk and SL
        if sl_pct > 0:
            position_size = risk_amount / sl_pct
        else:
            position_size = risk_amount / 0.02  # Default 2% SL
        
        # ⚠️ FIX: Leverage should NOT be a multiplier for risk-based sizing.
        # Risk-based sizing ALREADY calculates the necessary position size to lose 'risk_amount'
        # if 'sl_pct' is hit. Leverage is simply the mechanism that allows this size.
        # The true constraint is: position_size <= capital * leverage
        position_size *= self.config.leverage
        # Apply max concentration limit
        max_position = capital * self.config.max_concentration * self.config.leverage
        position_size = min(position_size, max_position)
        
        # Hard cap on position size by leverage
        position_size = min(position_size, available_capital * self.config.leverage)
        
        # Hard cap on position size in USD
        position_size = min(position_size, self.config.max_position_size_usd)
        
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
        
        # ⚠️ CRITICAL: Validate entry price before any calculations
        if entry_price <= 0 or entry_price > 1_000_000:
            # Return invalid trade that won't be processed
            trade.exit_reason = 'INVALID_PRICE'
            trade.pnl = 0
            return trade
        
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
            current_time = row.get('timestamp', datetime.now())
            
            high = row['high']
            low = row['low']
            close = row['close']
            open_price = row.get('open', close)  # fallback to close if open not available
            
            # --- TRAILING STOP LOGIC ---
            if self.config.use_trailing_stop:
                if direction == 'LONG':
                    # Check if high price triggered trailing update
                    current_high = high
                    price_change_pct = (current_high - trade.entry_price) / trade.entry_price
                    
                    if price_change_pct >= self.config.trailing_start_pct:
                        # Calculate new SL based on High
                        new_sl = current_high * (1 - self.config.trailing_step_pct)
                        # Only move SL up
                        if new_sl > trade.sl_price:
                            trade.sl_price = new_sl
                            
                    # Check if Low hit the new SL
                    if low <= trade.sl_price:
                        # Slippage applies to exit
                        trade.exit_price = trade.sl_price * (1 - self.config.slippage)
                        trade.exit_reason = 'TRAILING_STOP'
                        trade.exit_time = current_time
                        break
                        
                else: # SHORT
                    # Check if low price triggered trailing update
                    current_low = low
                    price_change_pct = (trade.entry_price - current_low) / trade.entry_price
                    
                    if price_change_pct >= self.config.trailing_start_pct:
                        # Calculate new SL based on Low
                        new_sl = current_low * (1 + self.config.trailing_step_pct)
                        # Only move SL down
                        if new_sl < trade.sl_price:
                            trade.sl_price = new_sl
                            
                    # Check if High hit the new SL
                    if high >= trade.sl_price:
                        # Slippage applies to exit
                        trade.exit_price = trade.sl_price * (1 + self.config.slippage)
                        trade.exit_reason = 'TRAILING_STOP'
                        trade.exit_time = current_time
                        break
            
            # --- STANDARD SL/TP LOGIC ---
            if direction == 'LONG':
                # Check SL FIRST (conservative)
                sl_hit = low <= trade.sl_price
                tp_hit = high >= trade.tp_price
                
                if sl_hit and tp_hit:
                    # Both hit - SL First rule with enhanced slippage due to high volatility
                    raw_exit = min(trade.sl_price, open_price, low)
                    trade.exit_price = raw_exit * (1 - self.config.slippage * 2)  # Double slippage
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
                    # Both hit - SL First rule with enhanced slippage due to high volatility
                    raw_exit = max(trade.sl_price, open_price, high)
                    trade.exit_price = raw_exit * (1 + self.config.slippage * 2)  # Double slippage
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
        
        # If the loop finished without hitting SL/TP/TIMEOUT (i.e. ran out of historical data at the end of the dataset)
        if not trade.exit_reason and trade.bars_held > 0:
            # Force exit at the last seen close price
            trade.exit_price = close * (1 - self.config.slippage if direction == 'LONG' else 1 + self.config.slippage)
            trade.exit_reason = 'END_OF_DATA'
            trade.exit_time = row.get('timestamp', datetime.now())
        
        # Calculate PnL with leverage
        if trade.exit_price > 0 and trade.entry_price > 0:
            if direction == 'LONG':
                trade.pnl_pct = (trade.exit_price - trade.entry_price) / trade.entry_price
            else:
                trade.pnl_pct = (trade.entry_price - trade.exit_price) / trade.entry_price
            
            # ⚠️ Sanity check for PnL percentage
            if abs(trade.pnl_pct) > 10:  # More than 1000% gain/loss - likely data error
                trade.pnl_pct = np.clip(trade.pnl_pct, -0.95, 10)  # Cap at -95% to +1000%
            
            # Exit fee
            exit_fee = position_size * (1 + abs(trade.pnl_pct)) * self.config.fee_rate
            trade.fees_paid += exit_fee
            
            # Calculate margin (actual capital used)
            margin = position_size / self.config.leverage
            
            # Net PnL after fees (on leveraged position)
            trade.pnl = position_size * trade.pnl_pct - trade.fees_paid
            
            # ⚠️ Additional safety check for extreme PnL
            if abs(trade.pnl) > margin * 50:  # More than 50x margin - likely error
                trade.pnl = np.clip(trade.pnl, -margin * 10, margin * 20)
            
            # Check for liquidation (loss exceeds margin * liquidation_threshold)
            # Only in ISOLATED mode. In CROSS mode, we allow PnL to exceed margin (draw from balance)
            if self.config.margin_mode == 'ISOLATED':
                max_loss = -margin * self.config.liquidation_threshold
                if trade.pnl < max_loss:
                    trade.pnl = -margin  # Lose entire margin
                    trade.exit_reason = 'LIQUIDATED'
        
        return trade
    
    def run_backtest(self, df: pd.DataFrame, verbose: bool = True) -> BacktestResult:
        """
        Run full backtest using TIME-STEPPED simulation with a PENDING POOL.
        This closely matches live bot behavior.
        """
        result = BacktestResult()
        capital = self.config.initial_capital
        available_capital = capital
        
        # Track positions and signals
        open_positions: Dict[int, Dict] = {}  # tid -> {trade, margin, exit_time}
        pending_pool: List[Dict] = []         # List of signals waiting for entry
        trade_counter = 0
        last_exit: Dict[str, Tuple[datetime, bool]] = {} # symbol -> (exit_time, is_long)

        # 1. Pre-process: Group data and signals by symbol and time
        symbols = df['symbol'].unique() if 'symbol' in df.columns else ['UNKNOWN']
        df_by_symbol = {s: df[df['symbol'] == s].sort_values('timestamp').reset_index(drop=True) for s in symbols}
        
        # Pre-process signals
        signals_by_time = {}
        for symbol, df_s in df_by_symbol.items():
            mask = (df_s['macd_cross_up'] == 1) | (df_s['macd_cross_down'] == 1)
            for _, row in df_s[mask].iterrows():
                ts = row['timestamp']
                if ts not in signals_by_time: signals_by_time[ts] = []
                signals_by_time[ts].append({
                    'symbol': symbol,
                    'row': row,
                    'is_long': row['macd_cross_up'] == 1,
                    'signal_price': row['close'],
                    'timestamp': ts
                })

        # Pre-process price maps with ffill for robust real-time check
        if verbose: print("  Preparing synchronized price maps (Open, Low, High, Close)...")
        # Use pivot_table with ffill to handle gaps
        open_map = df.pivot_table(index='timestamp', columns='symbol', values='open').fillna(method='ffill')
        low_map = df.pivot_table(index='timestamp', columns='symbol', values='low').fillna(method='ffill')
        high_map = df.pivot_table(index='timestamp', columns='symbol', values='high').fillna(method='ffill')
        close_map = df.pivot_table(index='timestamp', columns='symbol', values='close').fillna(method='ffill')
        
        idx_map = {}        # {(symbol, timestamp): row_index} mapping for slicing
        for symbol, df_s in df_by_symbol.items():
            for idx, row in enumerate(df_s.itertuples()):
                idx_map[(symbol, row.timestamp)] = idx
        
        if verbose:
            print(f"  Pre-indexed {len(df_by_symbol)} symbols and {len(signals_by_time)} signal timestamps.")

        # 2. Get all unique timestamps for simulation
        all_timestamps = sorted(df['timestamp'].unique())
        equity_timeline = [(all_timestamps[0], capital)]
        
        # 3. Main Time-Stepped Loop
        if verbose:
            print(f"🚀 Starting Optimized Time-Stepped Backtest: {len(all_timestamps)} steps...")

        for current_time in all_timestamps:
            # --- REAL-TIME MTM EQUITY & LIQUIDATION CHECK ---
            if open_positions:
                floating_pnl_total = 0.0
                liquidated_ids = []
                
                for tid, pos in open_positions.items():
                    trade = pos['trade']
                    symbol = trade.symbol
                    
                    # Robust Price Lookup using Ffilled Maps
                    if symbol in low_map.columns:
                        worst_price = low_map.at[current_time, symbol] if trade.direction == 'LONG' else high_map.at[current_time, symbol]
                    else:
                        worst_price = trade.entry_price
                    
                    if trade.direction == 'LONG':
                        pnl_pct = (worst_price - trade.entry_price) / trade.entry_price
                    else:
                        pnl_pct = (trade.entry_price - worst_price) / trade.entry_price
                    
                    floating_pnl = trade.position_size * pnl_pct
                    margin = trade.position_size / self.config.leverage
                    
                    # Local Liquidation
                    if floating_pnl < -margin * self.config.liquidation_threshold:
                        if self.config.margin_mode == 'ISOLATED':
                            trade.exit_price = worst_price
                            trade.exit_time = current_time
                            trade.pnl = -margin
                            trade.exit_reason = 'LIQUIDATED'
                            liquidated_ids.append(tid)
                            continue 
                        else:
                            floating_pnl = -margin
                    
                    floating_pnl_total += floating_pnl
                
                for tid in liquidated_ids:
                    pos = open_positions[tid]
                    capital += pos['trade'].pnl
                    available_capital += pos['margin'] + pos['trade'].pnl
                    result.trades.append(pos['trade'])
                    del open_positions[tid]
                    if verbose: print(f"  💀 Local Liquidation: {pos['trade'].symbol} at {current_time}")

                # Global Liquidation Check
                mtm_equity = capital + (floating_pnl_total if self.config.margin_mode == 'CROSS' else 0)
                if mtm_equity <= 0:
                    if verbose:
                        print(f"💀 GLOBAL LIQUIDATION at {current_time} (Mode: {self.config.margin_mode}, MTM Equity: ${mtm_equity:.2f})")
                    # Break simulation and cleanup result.trades
                    self._handle_global_liquidation(result, current_time, [], [])
                    break

            # A. Close expired positions
            closed_any = False
            closed_ids = []
            for tid, pos in open_positions.items():
                if pos['exit_time'] <= current_time:
                    trade = pos['trade']
                    capital += trade.pnl
                    available_capital += pos['margin'] + trade.pnl
                    result.trades.append(trade)
                    closed_ids.append(tid)
                    last_exit[trade.symbol] = (trade.exit_time or current_time, trade.direction == 'LONG')
                    closed_any = True
            
            for tid in closed_ids: del open_positions[tid]
            if closed_any: equity_timeline.append((current_time, capital))

            # B. Add new signals to Pending Pool
            if current_time in signals_by_time:
                for sig in signals_by_time[current_time]:
                    should_enter, confidence = self.predict_entry(sig['row'])
                    if should_enter:
                        sig['confidence'] = confidence
                        sig['expiry'] = current_time + pd.Timedelta(days=self.config.scanner_lookback_days)
                        pending_pool.append(sig)

            # C. Clean Up Pending Pool (Expired signals)
            pending_pool = [s for s in pending_pool if s['expiry'] > current_time]
            
            # D. Identify Candidates from Pending Pool (Good Entry Zones)
            candidates = []
            if current_time not in low_map.index: continue

            for sig in pending_pool:
                symbol = sig['symbol']
                if symbol not in low_map.columns: continue
                
                # Check for existing position
                if any(p['trade'].symbol == symbol for p in open_positions.values()):
                    continue
                
                # Check for fresh crossover requirement
                if self.config.require_fresh_crossover_after_exit and symbol in last_exit:
                    last_exit_time, _ = last_exit[symbol]
                    if current_time <= last_exit_time: continue
                    else: del last_exit[symbol]

                price_now = open_map.at[current_time, symbol]
                
                if self.config.use_scanner_filter:
                    zone = self.analyze_entry_zone(sig['is_long'], sig['signal_price'], price_now)
                    if zone not in self.config.allowed_zones: continue
                
                # Valid candidate
                candidates.append({
                    'signal': sig,
                    'price': price_now
                })

            # E. Prioritize Candidates by Confidence
            candidates.sort(key=lambda x: x['signal']['confidence'], reverse=True)

            # F. Open Positions
            for cand in candidates:
                if len(open_positions) >= self.config.max_open_trades: break
                
                sig = cand['signal']
                row_sig = sig['row']
                entry_price = cand['price']
                
                sl_pct = self.predict_sl(row_sig)
                tp_pct = self.predict_tp(row_sig, sl_pct)
                direction = 'LONG' if sig['is_long'] else 'SHORT'
                
                # Future data starting from current_time to end of max_bars
                df_s = df_by_symbol[sig['symbol']]
                current_idx = idx_map[(sig['symbol'], current_time)]
                future_data = df_s.iloc[current_idx : current_idx + self.config.max_bars + 1]
                
                # Calculate size
                open_margins = {tid: open_positions[tid]['margin'] for tid in open_positions}
                size = self.calculate_position_size(capital, sl_pct, tp_pct, sig['confidence'], open_margins)
                margin = size / self.config.leverage
                
                if size <= 0 or margin > available_capital * 0.98: continue

                trade = self.simulate_trade(row_sig, future_data, entry_price, sl_pct, tp_pct, direction, size)
                trade.confidence = sig['confidence']
                
                trade_counter += 1
                open_positions[trade_counter] = {
                    'trade': trade,
                    'margin': margin,
                    'exit_time': trade.exit_time or (current_time + pd.Timedelta(days=self.config.max_bars))
                }
                available_capital -= margin
                
                # Remove from pool once entered
                pending_pool = [s for s in pending_pool if s['symbol'] != sig['symbol']]
                
                if verbose and len(result.trades) < 5:
                    print(f"  ✅ Entry: {sig['symbol']} at {entry_price:.4f} (Conf: {sig['confidence']:.2%})")

            # Progress log
            if verbose and all_timestamps.index(current_time) % 50 == 0 and len(result.trades) > 0:
                 float_pnl = 0.0
                 if self.config.margin_mode == 'CROSS':
                     for tid, pos in open_positions.items():
                         t = pos['trade']
                         if t.symbol in low_map.columns:
                             wp = low_map.at[current_time, t.symbol] if t.direction == 'LONG' else high_map.at[current_time, t.symbol]
                             pp = (wp - t.entry_price) / t.entry_price if t.direction == 'LONG' else (t.entry_price - wp) / t.entry_price
                             float_pnl += t.position_size * pp
                 
                 display_equity = capital + float_pnl
                 print(f"  [{current_time}] Equity(MTM): ${display_equity:,.2f}, Pool: {len(pending_pool)}, Open: {len(open_positions)}")

        # 4. Final Cleanup
        for _, pos in open_positions.items():
            result.trades.append(pos['trade'])
        
        self._calculate_mtm_equity(df, result)
        self._calculate_metrics(result)
        return result

    def _calculate_mtm_equity(self, df: pd.DataFrame, result: BacktestResult):
        """
        Calculate Mark-to-Market (MtM) equity curve.
        Captures floating PnL of open positions at every timestamp.
        
        UPDATED: Calculates 'Conservative Equity' using Low/High prices (wicks)
        to capture true intra-candle drawdown.
        """
        if not result.trades:
            result.equity_curve = [self.config.initial_capital] * len(df['timestamp'].unique())
            result.timestamps = df['timestamp'].unique()
            return

        # 1. Create price maps (pivot tables) for High, Low, Close
        # We need these to calculate worst-case floating PnL
        close_map = df.pivot_table(index='timestamp', columns='symbol', values='close').fillna(method='ffill')
        low_map = df.pivot_table(index='timestamp', columns='symbol', values='low').fillna(method='ffill')
        high_map = df.pivot_table(index='timestamp', columns='symbol', values='high').fillna(method='ffill')
        
        market_timestamps = close_map.index.sort_values()
        
        # 2. Track equity at each timestamp
        equity_curve = []
        timestamps = []
        
        # Sort trades by entry time
        trades = sorted(result.trades, key=lambda t: t.entry_time)
        active_trades = []
        trade_idx = 0
        
        current_capital = self.config.initial_capital # Cash
        
        # Iterate through time
        for ts in market_timestamps:
            # A. Update realized PnL from closed trades (at this timestamp)
            # Remove trades that closed BEFORE or AT this timestamp from active list
            # and add their PnL to capital
            
            # First, check new entries
            while trade_idx < len(trades) and trades[trade_idx].entry_time <= ts:
                active_trades.append(trades[trade_idx])
                trade_idx += 1
            
            # Now update active trades status and calculate Floating PnL
            remaining_active = []
            floating_pnl_total = 0.0
            
            for trade in active_trades:
                # If trade closed before or at this timestamp
                if trade.exit_time and trade.exit_time <= ts:
                    current_capital += trade.pnl
                    # Trade is closed, don't add to remaining
                    continue
                
                # --- MTM PnL CALCULATION ---
                symbol = trade.symbol
                if symbol in low_map.columns:
                    worst_price = low_map.at[ts, symbol] if trade.direction == 'LONG' else high_map.at[ts, symbol]
                    
                    if pd.isna(worst_price):
                        worst_price = trade.entry_price
                        
                    pnl_pct = (worst_price - trade.entry_price) / trade.entry_price if trade.direction == 'LONG' else (trade.entry_price - worst_price) / trade.entry_price
                    floating_pnl = trade.position_size * pnl_pct
                    floating_pnl_total += floating_pnl
                
                remaining_active.append(trade)
            
            active_trades = remaining_active
            
            # --- ACCOUNT VALUE CALCULATION ---
            # Total Equity
            # CROSS: Cash + Floating PnL
            # ISOLATED: Cash (Floating loss doesn't affect other trades)
            total_equity = current_capital + (floating_pnl_total if self.config.margin_mode == 'CROSS' else 0)
            
            # The simulation loop should have already caught GLOBAL LIQUIDATION.
            # Here we just record the values for the result object.
            if total_equity <= 0:
                total_equity = 0
            
            equity_curve.append(total_equity)
            timestamps.append(ts)
            
        result.equity_curve = equity_curve
        result.timestamps = timestamps
    
    def _handle_global_liquidation(self, result: BacktestResult, death_time: datetime, equity_curve: list, timestamps: list):
        """
        Handle the event where the entire account is blown (Cross Margin).
        1. Truncate equity curve.
        2. Mark all open trades as LIQUIDATED at death_time.
        3. Remove future trades.
        """
        result.equity_curve = equity_curve
        result.timestamps = timestamps
        
        # Valid trades are those that entered BEFORE death time
        valid_trades = []
        for trade in result.trades:
            if trade.entry_time > death_time:
                continue
            
            # If trade was open at death_time, force close it
            if trade.exit_time is None or trade.exit_time > death_time:
                trade.exit_time = death_time
                trade.exit_reason = 'GLOBAL_LIQUIDATION'
                # PnL is already captured in the equity curve crashing to 0
                # But for trade stats, we can set it to the value at death?
                # Actually, the account is 0. 
                pass 
                
            valid_trades.append(trade)
            
        result.trades = valid_trades
    
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
        print(f"   Timeframe: {self.config.timeframe if self.config.timeframe else 'ALL'}")
        print(f"   Require Fresh Crossover After Exit: {'Yes' if self.config.require_fresh_crossover_after_exit else 'No'}")
        
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


def plot_max_positions_comparison(results: Dict[str, BacktestResult], initial_capital: float = 100, save_path: str = None):
    """
    Specialized plot for max positions comparison.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    position_colors = {
        '7 Positions': '#2E86AB',
        '10 Positions': '#A23B72', 
        '15 Positions': '#F18F01',
        '20 Positions': '#C73E1D'
    }
    
    # 1. Equity Curves
    ax1 = axes[0, 0]
    for name, result in results.items():
        if result.equity_curve:
            ax1.plot(result.equity_curve, label=name, linewidth=2, color=position_colors.get(name, '#333'))
    ax1.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('Equity Curves by Max Positions', fontweight='bold')
    ax1.set_xlabel('Trade #')
    ax1.set_ylabel('Equity ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # 2. Log Scale Equity
    ax2 = axes[0, 1]
    for name, result in results.items():
        if result.equity_curve:
            ax2.semilogy(result.equity_curve, label=name, linewidth=2, color=position_colors.get(name, '#333'))
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
    colors = [position_colors.get(n, '#333') for n in names]
    bars = ax3.bar(names, max_dds, color=colors, edgecolor='white', linewidth=2)
    ax3.set_title('Max Drawdown by Max Positions', fontweight='bold')
    ax3.set_ylabel('Max Drawdown (%)')
    ax3.grid(True, alpha=0.3, axis='y')
    # Rotate x labels
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    # Add value labels
    for bar, dd in zip(bars, max_dds):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{dd:.1f}%', ha='center', fontweight='bold')
    
    # 4. Return vs Drawdown Scatter
    ax4 = axes[1, 0]
    for name, result in results.items():
        ax4.scatter(result.max_drawdown * 100, result.total_return * 100, 
                   s=200, label=name, color=position_colors.get(name, '#333'), 
                   edgecolor='white', linewidth=2)
        ax4.annotate(name.replace(' Positions', ''), (result.max_drawdown * 100, result.total_return * 100),
                    textcoords="offset points", xytext=(10, 5), fontweight='bold')
    ax4.set_title('Return vs Max Drawdown', fontweight='bold')
    ax4.set_xlabel('Max Drawdown (%)')
    ax4.set_ylabel('Total Return (%)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Total Trades Count
    ax5 = axes[1, 1]
    trade_counts = [results[n].total_trades for n in names]
    bars = ax5.bar(names, trade_counts, color=colors, edgecolor='white', linewidth=2)
    ax5.set_title('Total Trades by Max Positions', fontweight='bold')
    ax5.set_ylabel('Total Trades')
    ax5.grid(True, alpha=0.3, axis='y')
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
    for bar, count in zip(bars, trade_counts):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{count}', ha='center', fontweight='bold')
    
    # 6. Final Equity Bar Chart
    ax6 = axes[1, 2]
    final_equities = [results[n].equity_curve[-1] if results[n].equity_curve else initial_capital for n in names]
    bars = ax6.bar(names, final_equities, color=colors, edgecolor='white', linewidth=2)
    ax6.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial')
    ax6.set_title('Final Equity by Max Positions', fontweight='bold')
    ax6.set_ylabel('Final Equity ($)')
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)
    for bar, eq in zip(bars, final_equities):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
                f'${eq:,.0f}', ha='center', fontweight='bold', fontsize=9, rotation=45)
    
    plt.suptitle('Max Open Positions Comparison Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path is None:
        save_path = str(Path(__file__).parent / 'results' / 'max_positions_comparison.png')
        # Create results directory if it doesn't exist
        Path(save_path).parent.mkdir(exist_ok=True)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n📊 Plot saved: {save_path}")
    
    plt.show()
    return fig


def plot_leverage_comparison(results: Dict[str, BacktestResult], initial_capital: float = 100, save_path: str = None):
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
        max_bars=config.max_bars,
        timebase=config.timebase,
        require_fresh_crossover_after_exit=config.require_fresh_crossover_after_exit
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


def run_max_positions_comparison(df: pd.DataFrame, base_config: BacktestConfig) -> Dict[str, BacktestResult]:
    """
    Compare different max open positions (7, 10, 15, 20).
    """
    max_positions = [7,8,9,10]
    results = {}
    
    print("\n" + "="*80)
    print("MAX OPEN POSITIONS COMPARISON TEST")
    print("="*80)
    
    for max_pos in max_positions:
        print(f"\n🔄 Testing Max Positions = {max_pos}...")
        
        config = deepcopy(base_config)
        config.max_open_trades = max_pos
        
        backtester = ThreeStageBacktester(config)
        result = backtester.run_backtest(df, verbose=False)
        results[f'{max_pos} Positions'] = result
        
        # Count liquidations and extreme PnL trades
        liquidations = sum(1 for t in result.trades if t.exit_reason == 'LIQUIDATED')
        extreme_pnl_trades = sum(1 for t in result.trades if abs(t.pnl_pct) > 5)  # >500%
        
        print(f"   ✅ Completed:")
        print(f"      Total Trades: {result.total_trades}")
        print(f"      Final Equity: ${result.equity_curve[-1]:,.2f}" if result.equity_curve else "      Final Equity: N/A")
        print(f"      Total Return: {result.total_return:.1%}")
        print(f"      Max Drawdown: {result.max_drawdown:.1%}")
        print(f"      Liquidations: {liquidations}")
        print(f"      Extreme PnL Trades: {extreme_pnl_trades}")
    
    # Summary table
    print("\n" + "="*90)
    print("MAX POSITIONS COMPARISON SUMMARY")
    print("="*90)
    print(f"\n{'Max Pos':<8} {'Trades':>8} {'Win%':>8} {'Return':>12} {'MaxDD':>10} {'Sharpe':>8} {'PF':>8} {'Liq':>6}")
    print("-"*90)
    
    for max_pos in max_positions:
        res = results[f'{max_pos} Positions']
        liquidations = sum(1 for t in res.trades if t.exit_reason == 'LIQUIDATED')
        print(f"{max_pos:<8} {res.total_trades:>8} {res.win_rate:>7.1%} "
              f"{res.total_return:>11.1%} {res.max_drawdown:>9.1%} "
              f"{res.sharpe_ratio:>7.2f} {res.profit_factor:>7.2f} {liquidations:>6}")
    
    # Risk-adjusted comparison
    print("\n" + "="*80)
    print("RISK-ADJUSTED METRICS COMPARISON")
    print("="*80)
    print(f"\n{'Max Pos':<8} {'Return/DD':>12} {'Calmar':>10} {'Final Equity':>15}")
    print("-"*60)
    
    for max_pos in max_positions:
        res = results[f'{max_pos} Positions']
        return_dd_ratio = res.total_return / res.max_drawdown if res.max_drawdown > 0 else float('inf')
        # Calmar ratio (annual return / max DD) - simplified
        calmar = (res.total_return / 6) / res.max_drawdown if res.max_drawdown > 0 else float('inf')  # ~6 years
        final_equity = res.equity_curve[-1] if res.equity_curve else base_config.initial_capital
        print(f"{max_pos:<8} {return_dd_ratio:>11.2f}x {calmar:>9.2f} ${final_equity:>14,.2f}")
    
    # Plot comparison
    plot_max_positions_comparison(results, base_config.initial_capital)
    
    return results


def run_leverage_comparison(df: pd.DataFrame, base_config: BacktestConfig) -> Dict[str, BacktestResult]:
    """
    Compare different leverage levels (1x, 3x, 5x, 7x, 10x).
    """
    leverage_levels = [5, 10, 15, 20, 25]
    results = {}
    
    print("\n" + "="*70)
    print("LEVERAGE COMPARISON TEST")
    print("="*70)
    
    for lev in leverage_levels:
        print(f"\n{'='*70}")
        print(f"Testing {lev}x Leverage...")
        print("="*70)
        
        config = deepcopy(base_config)
        config.leverage = lev
        
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


def run_pullback_comparison(df: pd.DataFrame, base_config: BacktestConfig):
    """
    Compare strategy performance: Market Entry vs Pullback Limit Entry.
    """
    print("\n" + "="*60)
    print("🔄 RUNNING PULLBACK ENTRY COMPARISON")
    print("="*60)
    
    results = {}
    
    # 1. Market Entry (Baseline)
    print("\n🔹 Running Market Entry (Baseline)...")
    base_config.entry_pullback_pct = 0.0
    backtester_market = ThreeStageBacktester(base_config)
    results['Market Entry'] = backtester_market.run_backtest(df, verbose=False)
    
    # 2. Pullback Limit Entry
    # Test a few pullback levels
    pullback_levels = [0.005, 0.01, 0.02] # 0.5%, 1.0%, 2.0%
    timeout = 10 # Increase timeout
    
    for pct in pullback_levels:
        print(f"\n🔹 Running Pullback Entry (Limit: -{pct:.1%}, Timeout: {timeout} bars)...")
        pullback_config = deepcopy(base_config)
        pullback_config.entry_pullback_pct = pct
        pullback_config.entry_pullback_timeout = timeout
        
        backtester_pullback = ThreeStageBacktester(pullback_config)
        results[f'Limit -{pct:.1%}'] = backtester_pullback.run_backtest(df, verbose=False)
    
    # Print Comparison Table
    print("\n" + "="*100)
    print(f"{'METRIC':<25} | {'MARKET':<15} | {'LIMIT -0.5%':<15} | {'LIMIT -1.0%':<15} | {'DIFF (0.5%)':<10}")
    print("-" * 100)
    
    metrics = [
        ('Final Capital ($)', 'equity_curve', lambda x: x[-1] if x else 0),
        ('Total Return (%)', 'total_return', lambda x: x * 100),
        ('Max Drawdown (%)', 'max_drawdown', lambda x: x * 100),
        ('Win Rate (%)', 'win_rate', lambda x: x * 100),
        ('Total Trades', 'total_trades', lambda x: x),
        ('Profit Factor', 'profit_factor', lambda x: x),
        ('Avg Trade ($)', 'avg_trade_pnl', lambda x: x),
    ]
    
    res_market = results['Market Entry']
    res_limit_05 = results.get('Limit -0.5%', res_market) # Fallback
    res_limit_10 = results.get('Limit -1.0%', res_market)
    
    for label, attr, fmt_func in metrics:
        if attr == 'equity_curve':
            val_m = fmt_func(res_market.equity_curve)
            val_05 = fmt_func(res_limit_05.equity_curve)
            val_10 = fmt_func(res_limit_10.equity_curve)
        else:
            val_m = fmt_func(getattr(res_market, attr))
            val_05 = fmt_func(getattr(res_limit_05, attr))
            val_10 = fmt_func(getattr(res_limit_10, attr))
            
        diff = val_05 - val_m
        print(f"{label:<25} | {val_m:>15.2f} | {val_05:>15.2f} | {val_10:>15.2f} | {diff:>+10.2f}")
        
    print("="*100)
    
    # Plot Comparison
    lev_str = f"{base_config.leverage:.0f}x" if base_config.leverage > 1 else "1x"
    plot_equity_curve(results, title=f"Market vs Pullback Entry (Lev {lev_str})", 
                     save_path=str(DATA_DIR.parent / 'backtest_pullback_comparison.png'))


def run_timeout_comparison(df, base_config):
    """Test standard vs different timeout periods (max_bars)."""
    print("\n" + "="*70)
    print(f"🔍 TIMEOUT OPTIMIZATION (max_bars) - Timeframe: {base_config.timeframe}")
    print("="*70)
    
    # Test different max_bars settings
    timeouts = [5, 8, 10, 12, 15, 20]
    results = []
    
    # Run individual backtests
    for bars in timeouts:
        print(f"Testing max_bars = {bars}...", end=" ", flush=True)
        config = deepcopy(base_config)
        config.max_bars = bars
        
        tester = ThreeStageBacktester(config)
        res = tester.run_backtest(df, verbose=False)
        results.append(res)
        print(f"Done: {res.total_trades} trades, {res.total_return*100:.1f}% return")
    
    # Print summary table
    print("\n" + "="*100)
    print(f"{'MAX BARS':<10} | {'TRADES':>8} | {'WIN%':>8} | {'RETURN':>10} | {'MAX DD':>10} | {'SHARPE':>10} | {'PF':>8}")
    print("-" * 100)
    
    for i, bars in enumerate(timeouts):
        res = results[i]
        print(f"{bars:<10} | {res.total_trades:>8} | {res.win_rate*100:>7.1f}% | {res.total_return*100:>9.1f}% | {res.max_drawdown*100:>9.1f}% | {res.sharpe_ratio:>10.2f} | {res.profit_factor:>8.2f}")
    print("="*100 + "\n")

    # Generate plot for timeouts
    plt.figure(figsize=(12, 6))
    for i, bars in enumerate(timeouts):
        res = results[i]
        plt.plot(res.timestamps, res.equity_curve, label=f'Bars={bars}')
    
    plt.title(f"Timeout (max_bars) Optimization - {base_config.timeframe} (Lev {base_config.leverage}x)")
    plt.xlabel("Date")
    plt.ylabel("Capital (USD)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = DATA_DIR.parent / f'backtest_timeout_comparison_{base_config.timeframe}.png'
    plt.savefig(plot_path)
    print(f"📊 Comparison chart saved to: {plot_path}")


def main():
    """Main function to run backtest."""
    import argparse
    
    parser = argparse.ArgumentParser(description='3-Stage ML Backtest')
    parser.add_argument('--data', type=str, default=None, help='Path to data file')
    parser.add_argument('--capital', type=float, default=100, help='Initial capital')
    parser.add_argument('--risk', type=float, default=0.01, help='Risk per trade (0.01 = 1%%)')
    parser.add_argument('--threshold', type=float, default=0.65, help='Entry confidence threshold')
    parser.add_argument('--fee', type=float, default=0.001, help='Fee rate (0.001 = 0.1%%)')
    parser.add_argument('--slippage', type=float, default=0.0005, help='Slippage (0.0005 = 0.05%%)')
    parser.add_argument('--kelly', action='store_true', help='Use Kelly Criterion')
    parser.add_argument('--fixed-size', action='store_true', help='Use fixed position size')
    parser.add_argument('--size-usd', type=float, default=1000, help='Fixed position size in USD')
    parser.add_argument('--leverage', type=float, default=1.0, help='Leverage multiplier (1, 3, 5, 7, 10)')
    parser.add_argument('--max-positions', type=int, default=10, help='Max open positions (default: 10)')
    parser.add_argument('--compare', action='store_true', help='Run baseline comparison')
    parser.add_argument('--leverage-test', action='store_true', help='Test multiple leverage levels')
    parser.add_argument('--max-positions-test', action='store_true', help='Test multiple max positions (7, 10, 15, 20)')
    parser.add_argument('--start', type=str, default=None, help='Start date (YYYY-MM-DD or ISO) to filter test period')
    parser.add_argument('--end', type=str, default=None, help='End date (YYYY-MM-DD or ISO) to filter test period')
    parser.add_argument('--timeframe', type=str, default=None, help="Specify timeframe (1d, 4h, 8h, 12h) - load corresponding data file")
    parser.add_argument('--plot-trades', type=str, default='0', help='Number of individual trade charts to create (e.g. 20, 50) or "all"')
    parser.add_argument('--plot-individual', action='store_true', help='Plot sample individual trades with candlesticks')
    parser.add_argument('--margin-mode', type=str, default='ISOLATED', choices=['ISOLATED', 'CROSS'], help='Margin mode: ISOLATED (default) or CROSS')
    
    # Trailing Stop arguments
    parser.add_argument('--trailing', action='store_true', help='Enable Trailing Stop')
    parser.add_argument('--trailing-start', type=float, default=0.1, help='Trailing start pct (e.g. 0.02 for 2%%)')
    parser.add_argument('--trailing-step', type=float, default=0.05, help='Trailing step pct (e.g. 0.01 for 1%%)')
    parser.add_argument('--compare-trailing', action='store_true', help='Run Trailing Stop Comparison')
    parser.add_argument('--compare-kelly', action='store_true', help='Run Kelly vs Even Sizing Comparison')
    parser.add_argument('--compare-pullback', action='store_true', help='Run Market vs Pullback Entry Comparison')
    parser.add_argument('--compare-timeframes', action='store_true', help='Run Multi-Timeframe Comparison')
    
    # Pullback options
    parser.add_argument('--entry-pullback', type=float, default=0.0, help='Pullback pct for limit entry (e.g. 0.005 for 0.5%%)')
    parser.add_argument('--entry-timeout', type=int, default=3, help='Timeout bars for limit entry')
    parser.add_argument('--compare-timeout', action='store_true', help='Run Timeout (max_bars) Optimization')
    parser.add_argument('--max-bars', type=int, default=10, help='Max bars to hold trade (timeout)')
    
    # Scanner Filter arguments
    parser.add_argument('--use-scanner', action='store_true', help='Enable SmartScanner Entry Zone filtering')
    parser.add_argument('--scanner-mae', type=float, default=0.04, help='Max Adverse Excursion for zone (default: 0.04)')
    parser.add_argument('--scanner-mfe', type=float, default=0.12, help='Max Favorable Excursion for zone (default: 0.12)')
    parser.add_argument('--scanner-lookback', type=int, default=6, help='Lookback days for scanner entry (default: 6)')
    
    args = parser.parse_args()
    
    # Load data based on timeframe
    if args.compare_timeframes:
        # Comparison mode handles its own data loading
        df = None
        data_path = None
    elif args.data:
        data_path = Path(args.data)
    else:
        # Map timeframe to file
        timeframe_files = {
            '1d': 'features_1d_full.parquet',
            '4h': 'features_4h_full.parquet', 
            '8h': 'features_8h_full.parquet',
            '12h': 'features_12h_full.parquet',
            '1h': 'features_1h_full.parquet',
            '1w': 'features_1w_full.parquet'
        }
        
        if args.timeframe and args.timeframe in timeframe_files:
            data_path = PROCESSED_DIR / timeframe_files[args.timeframe]
        else:
            data_path = PROCESSED_DIR / 'features_1d_full.parquet'  # Default
            if args.timeframe:
                print(f"⚠️ Timeframe {args.timeframe} not supported, using 1d data")
    
    if args.compare_timeframes:
        df = pd.DataFrame() # Dummy for initialization
    else:
        if not data_path.exists():
            print(f"Data not found: {data_path}")
            print("Available files:")
            for tf, fname in timeframe_files.items():
                fpath = PROCESSED_DIR / fname
                status = "✓" if fpath.exists() else "✗"
                print(f"  {status} {tf}: {fname}")
            print("Run multi_timeframe_pipeline.py first!")
            return
        
        df = pd.read_parquet(data_path)
        print(f"Loaded {len(df):,} rows from {data_path.name}")
    
    # Filter data based on date range
    if not args.compare_timeframes:
        if args.start or args.end:
            try:
                start = pd.to_datetime(args.start) if args.start else None
                end = pd.to_datetime(args.end) if args.end else None
            except Exception as e:
                print(f"Invalid start/end date: {e}")
                return

            # Use entire dataset filtered by date range (don't split train/test)
            df_test = df.copy()
            if start is not None:
                df_test = df_test[df_test['timestamp'] >= start]
            if end is not None:
                df_test = df_test[df_test['timestamp'] <= end]

            if df_test.empty:
                print(f"No data in range {start} to {end}. Aborting.")
                return

            unique_symbols = df_test['symbol'].nunique() if 'symbol' in df_test.columns else 1
            print(f"Test period: {len(df_test):,} rows across {unique_symbols} symbols")
            print(f"Date range: {df_test['timestamp'].min()} to {df_test['timestamp'].max()}")
        else:
            # 🔧 FIX: Use chronological split for testing
            print("📅 Using default test period (last 6 months chronologically)...")
            df = df.sort_values('timestamp')  # Ensure chronological order
            
            # Use last 6 months for testing (more realistic than random 20%)
            latest_date = df['timestamp'].max()
            test_start_date = latest_date - pd.DateOffset(months=6)
            
            df_test = df[df['timestamp'] >= test_start_date].copy()
            
            # Fallback to last 20% if not enough recent data
            if len(df_test) < 1000:  # Minimum threshold
                print("⚠️ Not enough data in last 6 months, using last 20% of chronological data")
                test_start_idx = int(len(df) * 0.8)
                df_test = df.iloc[test_start_idx:].copy()
            
            unique_symbols = df_test['symbol'].nunique() if 'symbol' in df_test.columns else 1
            print(f"✅ Test period: {len(df_test):,} rows across {unique_symbols} symbols")
            print(f"Date range: {df_test['timestamp'].min()} to {df_test['timestamp'].max()}")
    else:
        df_test = None # Dummy
    
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
        leverage=args.leverage,
        max_open_trades=args.max_positions,
        timeframe=args.timeframe,
        margin_mode=args.margin_mode,
        use_trailing_stop=args.trailing,
        trailing_start_pct=args.trailing_start,
        trailing_step_pct=args.trailing_step,
        entry_pullback_pct=args.entry_pullback,
        entry_pullback_timeout=args.entry_timeout,
        start_date=args.start,
        end_date=args.end,
        max_bars=args.max_bars,
        use_scanner_filter=args.use_scanner,
        scanner_mae=args.scanner_mae,
        scanner_mfe=args.scanner_mfe,
        scanner_lookback_days=args.scanner_lookback
    )
    
    # Run timeout comparison if requested
    if args.compare_timeout:
        run_timeout_comparison(df_test, config)
        return
    
    # Run multi-timeframe comparison if requested
    if args.compare_timeframes:
        run_timeframe_comparison(config)
        return
    
    # Run Comparison: Kelly vs Even
    if args.compare_kelly:
        run_kelly_comparison(df_test, config)
        return

    # Run Comparison: Pullback
    if args.compare_pullback:
        run_pullback_comparison(df_test, config)
        return

    # Run Trailing Comparison
    if args.compare_trailing:
        run_trailing_comparison(df_test, config)
        return
    
    # Test multiple max positions
    if args.max_positions_test:
        run_max_positions_comparison(df_test, config)
        return
    
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
        
        # Plot trades (limited by --plot-trades parameter) - for single symbol view
        # plot_backtest_trades(df_test, result.trades, 
        #                    title=f'Backtest Trades ({lev_str})',
        #                    save_path=str(DATA_DIR.parent / 'output' / f'backtest_trades_{lev_str}.png'),
        #                    trade_limit='50')  # Fixed limit for single symbol view
        
        # Plot ALL trades timeline (always show all trades timeline)
        # plot_all_trades_timeline(result.trades,
        #                        title=f'All Trades Timeline ({lev_str} Leverage)',
        #                        save_path=str(DATA_DIR.parent / f'output/all_trades_timeline_{lev_str}.png'))
        
        # Plot ALL trades summary across all symbols
        #plot_all_trades_summary(result.trades,
        #                       title=f'All Trades Summary ({lev_str} Leverage)', 
        #                       save_path=str(DATA_DIR.parent / f'output/backtest_all_trades_{lev_str}.png'))
        
        # Determine number of individual trade charts to create
        if args.plot_trades == 'all':
            num_individual_charts = min(len(result.trades), 50)  # Cap at 50 to avoid too many files
        else:
            try:
                num_individual_charts = int(args.plot_trades)
                num_individual_charts = min(num_individual_charts, len(result.trades))  # Can't exceed available trades
            except ValueError:
                print(f"⚠️ Invalid plot-trades value '{args.plot_trades}', defaulting to 8")
                num_individual_charts = 8
        
        # Plot individual trades
        if num_individual_charts > 0:
            plot_sample_individual_trades(df_test, result.trades, lev_str, num_samples=num_individual_charts)



def plot_sample_individual_trades(df: pd.DataFrame, trades: List[Trade], lev_str: str, num_samples: int = 8):
    """
    Plot individual trade charts. Selection strategy varies based on num_samples.
    
    Args:
        df: Full DataFrame with OHLCV data
        trades: List of all Trade objects
        lev_str: Leverage string for file naming
        num_samples: Number of individual trade charts to create
    """
    if len(trades) < 1:
        print("⚠️ No trades to plot")
        return
    
    # Cap at available trades
    num_samples = min(num_samples, len(trades))
    
    print(f"📈 Creating {num_samples} individual trade charts...")
    
    # Selection strategy based on requested number
    samples = []
    
    if num_samples <= 10:
        # Small number: pick curated samples (best, worst, etc.)
        samples = _select_curated_trades(trades, num_samples)
    else:
        # Large number: pick diverse sampling across all trades
        samples = _select_diverse_trades(trades, num_samples)
    
    # Plot each sample
    output_dir = DATA_DIR.parent / 'output' / 'individual_trades'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, (label, trade) in enumerate(samples):
        # Create descriptive filename with index
        pnl_str = f"{'+' if trade.pnl >= 0 else ''}{trade.pnl:.0f}USD"
        filename = f'{i+1:03d}_{label}_{trade.symbol}_{pnl_str}_{lev_str}.png'
        save_path = output_dir / filename
        
        print(f"  📊 {i+1:2d}/{num_samples}: {label} - {trade.symbol} {trade.direction} ${trade.pnl:+.0f}")
        
        # Create detailed title
        duration_str = f"{trade.bars_held}bars" if trade.bars_held > 0 else "N/A"
        chart_title = f'{label}: {trade.symbol} {trade.direction}\nPnL: ${trade.pnl:+.0f} | Duration: {duration_str} | Exit: {trade.exit_reason}'
        
        try:
            plot_individual_trade(
                df=df,
                trade=trade, 
                title=chart_title,
                save_path=str(save_path),
                buffer_bars=15  # Smaller buffer for cleaner view
            )
        except Exception as e:
            print(f"  ⚠️ Failed to plot {label}: {e}")
    
    print(f"💾 Created {len(samples)} individual trade charts in: {output_dir}")


def _select_curated_trades(trades: List[Trade], num_samples: int) -> List[Tuple[str, Trade]]:
    """Select curated high-quality trade samples for small numbers (≤10)."""
    samples = []
    
    # 1. Best winner
    if trades:
        best_winner = max(trades, key=lambda t: t.pnl)
        samples.append(('Best_Winner', best_winner))
    
    # 2. Worst loser
    if len(samples) < num_samples and trades:
        worst_loser = min(trades, key=lambda t: t.pnl)
        if worst_loser != samples[-1][1]:  # Avoid duplicates
            samples.append(('Worst_Loser', worst_loser))
    
    # 3. Best TP hit
    if len(samples) < num_samples:
        tp_hits = [t for t in trades if 'TP' in str(t.exit_reason) and t.pnl > 0]
        if tp_hits:
            best_tp = max(tp_hits, key=lambda t: t.pnl)
            if best_tp not in [s[1] for s in samples]:
                samples.append(('Best_TP_Hit', best_tp))
    
    # 4. Worst SL hit
    if len(samples) < num_samples:
        sl_hits = [t for t in trades if 'SL' in str(t.exit_reason) and t.pnl < 0]
        if sl_hits:
            worst_sl = min(sl_hits, key=lambda t: t.pnl)
            if worst_sl not in [s[1] for s in samples]:
                samples.append(('Worst_SL_Hit', worst_sl))
    
    # 5-6. Top symbols
    if len(samples) < num_samples:
        symbol_counts = {}
        for t in trades:
            symbol_counts[t.symbol] = symbol_counts.get(t.symbol, 0) + 1
        
        top_symbols = sorted(symbol_counts.items(), key=lambda x: -x[1])[:2]
        for symbol, count in top_symbols:
            if len(samples) >= num_samples:
                break
            symbol_trades = [t for t in trades if t.symbol == symbol]
            if symbol_trades:
                # Pick median trade for this symbol
                median_trade = sorted(symbol_trades, key=lambda t: t.pnl)[len(symbol_trades)//2]
                if median_trade not in [s[1] for s in samples]:
                    samples.append((f'Symbol_{symbol}', median_trade))
    
    # Fill remaining with diverse picks
    if len(samples) < num_samples:
        remaining_trades = [t for t in trades if t not in [s[1] for s in samples]]
        remaining_trades = sorted(remaining_trades, key=lambda t: t.pnl)
        
        # Pick from different percentiles
        for i in range(num_samples - len(samples)):
            if i < len(remaining_trades):
                percentile = (i + 1) * 100 // (num_samples - len(samples) + 1)
                idx = int(len(remaining_trades) * percentile / 100)
                idx = min(idx, len(remaining_trades) - 1)
                samples.append((f'P{percentile}_Trade', remaining_trades[idx]))
    
    return samples[:num_samples]


def _select_diverse_trades(trades: List[Trade], num_samples: int) -> List[Tuple[str, Trade]]:
    """Select diverse trade sampling for large numbers (>10)."""
    samples = []
    
    # Always include a few key trades
    if trades:
        best_winner = max(trades, key=lambda t: t.pnl)
        samples.append(('Best_Winner', best_winner))
        
        worst_loser = min(trades, key=lambda t: t.pnl)
        if worst_loser != best_winner:
            samples.append(('Worst_Loser', worst_loser))
    
    # Fill rest with systematic sampling
    remaining_trades = [t for t in trades if t not in [s[1] for s in samples]]
    
    # Sort by different criteria and sample
    criteria = [
        ('PnL', lambda t: t.pnl),
        ('Duration', lambda t: t.bars_held),
        ('Time', lambda t: t.entry_time),
        ('Symbol', lambda t: t.symbol)
    ]
    
    samples_per_criteria = (num_samples - len(samples)) // len(criteria)
    
    for criterion_name, sort_key in criteria:
        if len(samples) >= num_samples:
            break
            
        sorted_trades = sorted(remaining_trades, key=sort_key)
        n_to_pick = min(samples_per_criteria, num_samples - len(samples), len(sorted_trades))
        
        # Pick evenly distributed samples
        for i in range(n_to_pick):
            idx = int(i * len(sorted_trades) / n_to_pick)
            idx = min(idx, len(sorted_trades) - 1)
            trade = sorted_trades[idx]
            
            if trade not in [s[1] for s in samples]:
                samples.append((f'{criterion_name}_{i+1}', trade))
                remaining_trades.remove(trade)
    
    # Fill any remaining slots randomly
    while len(samples) < num_samples and remaining_trades:
        trade = remaining_trades.pop(0)
        samples.append((f'Random_{len(samples)+1}', trade))
    
    return samples[:num_samples]


def plot_individual_trade(df: pd.DataFrame, trade: Trade, title: str = None, save_path: str = None, buffer_bars: int = 50):
    """
    Plot diverse sample individual trades: best winners, worst losers, different symbols, exit reasons etc.
    
    Args:
        df: Full DataFrame with OHLCV data
        trades: List of all Trade objects
        lev_str: Leverage string for file naming
        num_samples: Number of sample trades to plot
    """
    if len(trades) < 4:
        print("⚠️ Not enough trades to plot samples")
        return
    
    print(f"📈 Creating {num_samples} individual trade charts...")
    
    # Sort trades by PnL
    sorted_trades = sorted(trades, key=lambda t: t.pnl)
    
    # Select diverse sample trades
    samples = []
    
    # 1. Best winner
    best_winner = max(trades, key=lambda t: t.pnl)
    samples.append(('01_Best_Winner', best_winner))
    
    # 2. Worst loser
    worst_loser = min(trades, key=lambda t: t.pnl)
    samples.append(('02_Worst_Loser', worst_loser))
    
    # 3. Best TP hit
    tp_hits = [t for t in trades if 'TP' in str(t.exit_reason) and t.pnl > 0]
    if tp_hits:
        best_tp = max(tp_hits, key=lambda t: t.pnl)
        samples.append(('03_Best_TP_Hit', best_tp))
    
    # 4. Worst SL hit
    sl_hits = [t for t in trades if 'SL' in str(t.exit_reason) and t.pnl < 0]
    if sl_hits:
        worst_sl = min(sl_hits, key=lambda t: t.pnl)
        samples.append(('04_Worst_SL_Hit', worst_sl))
    
    # 5. Longest duration winner
    long_winners = [t for t in trades if t.pnl > 0 and t.bars_held > 0]
    if long_winners:
        longest_winner = max(long_winners, key=lambda t: t.bars_held)
        samples.append(('05_Longest_Winner', longest_winner))
    
    # 6. Quick winner (shortest duration)
    quick_winners = [t for t in long_winners if t.bars_held < np.percentile([t.bars_held for t in long_winners], 25)]
    if quick_winners:
        quickest_winner = min(quick_winners, key=lambda t: t.bars_held)
        samples.append(('06_Quickest_Winner', quickest_winner))
    
    # 7. Different symbols - pick top symbols
    symbol_counts = {}
    for t in trades:
        symbol_counts[t.symbol] = symbol_counts.get(t.symbol, 0) + 1
    
    top_symbols = sorted(symbol_counts.items(), key=lambda x: -x[1])[:3]
    for i, (symbol, count) in enumerate(top_symbols):
        symbol_trades = [t for t in trades if t.symbol == symbol and t.pnl != 0]
        if symbol_trades:
            # Pick the median trade for this symbol
            symbol_trades_sorted = sorted(symbol_trades, key=lambda t: t.pnl)
            median_trade = symbol_trades_sorted[len(symbol_trades_sorted)//2]
            samples.append((f'0{7+i}_Symbol_{symbol}', median_trade))
    
    # 8. Random samples from different time periods
    if len(samples) < num_samples:
        remaining_trades = [t for t in trades if t not in [s[1] for s in samples]]
        if remaining_trades:
            # Sort by time and pick from different periods
            time_sorted = sorted(remaining_trades, key=lambda t: t.entry_time)
            for i in range(min(2, num_samples - len(samples))):
                idx = int(len(time_sorted) * (i + 1) / 3)  # Pick from different thirds
                samples.append((f'1{i}_Random_Period_{i+1}', time_sorted[idx]))
    
    # Plot each sample
    output_dir = DATA_DIR.parent / 'output' / 'individual_trades'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for label, trade in samples[:num_samples]:
        # Create descriptive filename
        pnl_str = f"{'+' if trade.pnl >= 0 else ''}{trade.pnl:.0f}USD"
        filename = f'{label}_{trade.symbol}_{pnl_str}_{lev_str}.png'
        save_path = output_dir / filename
        
        print(f"  📊 {label}: {trade.symbol} {trade.direction} - ${trade.pnl:+.0f} ({trade.exit_reason})")
        
        # Create detailed title
        duration_str = f"{trade.bars_held}bars" if trade.bars_held > 0 else "N/A"
        chart_title = f'{label.replace("_", " ")}: {trade.symbol} {trade.direction}\nPnL: ${trade.pnl:+.0f} | Duration: {duration_str} | Exit: {trade.exit_reason}'
        
        try:
            plot_individual_trade(
                df=df,
                trade=trade, 
                title=chart_title,
                save_path=str(save_path),
                buffer_bars=20  # Smaller buffer for cleaner view
            )
        except Exception as e:
            print(f"  ⚠️ Failed to plot {label}: {e}")
    
    print(f"💾 Created {len(samples)} individual trade charts in: {output_dir}")


def plot_individual_trade(df: pd.DataFrame, trade: Trade, title: str = None, save_path: str = None, buffer_bars: int = 50):
    """
    Plot detailed individual trade with candlesticks, entry, TP, SL levels.
    
    Args:
        df: DataFrame with OHLCV data for the specific symbol
        trade: Single Trade object to visualize
        title: Chart title
        save_path: Path to save the plot
        buffer_bars: Number of bars to show before/after trade for context
    """
    if df.empty or not trade:
        print("⚠️ No data or trade to plot")
        return
        
    if mpf is None:
        print("⚠️ mplfinance not available, using basic line plot")
        _plot_individual_trade_basic(df, trade, title, save_path, buffer_bars)
        return
    
    print(f"📊 Plotting individual trade: {trade.symbol} {trade.direction} @ {trade.entry_time}")
    
    # Filter to specific symbol if needed
    if 'symbol' in df.columns:
        df = df[df['symbol'] == trade.symbol].copy()
    
    # Ensure timestamp is index 
    df_plot = df.copy()
    if 'timestamp' in df_plot.columns:
        df_plot.set_index('timestamp', inplace=True)
    
    # Find trade period with buffer
    entry_idx = df_plot.index.get_indexer([trade.entry_time], method='nearest')[0]
    if trade.exit_time:
        exit_idx = df_plot.index.get_indexer([trade.exit_time], method='nearest')[0]
    else:
        exit_idx = min(entry_idx + trade.bars_held, len(df_plot) - 1)
    
    # Add buffer
    start_idx = max(0, entry_idx - buffer_bars)
    end_idx = min(len(df_plot) - 1, exit_idx + buffer_bars)
    
    # Extract OHLCV data for the period
    df_period = df_plot.iloc[start_idx:end_idx + 1].copy()
    
    if len(df_period) < 5:
        print("⚠️ Not enough data around trade period")
        return
    
    # Prepare data for mplfinance (needs specific column names)
    ohlc_data = pd.DataFrame({
        'Open': df_period['open'],
        'High': df_period['high'], 
        'Low': df_period['low'],
        'Close': df_period['close'],
        'Volume': df_period.get('volume', 0)
    }, index=df_period.index)
    
    # Remove any NaN values
    ohlc_data = ohlc_data.dropna()
    
    if ohlc_data.empty:
        print("⚠️ No valid OHLC data for trade period")
        return
    
    # Prepare additional plots (lines for TP, SL)
    additional_plots = []
    
    # TP line (horizontal)
    if trade.tp_price and trade.tp_price > 0:
        tp_line = [trade.tp_price] * len(ohlc_data)
        additional_plots.append(
            mpf.make_addplot(tp_line, color='green', linestyle='--', width=2, alpha=0.8)
        )
    
    # SL line (horizontal)  
    if trade.sl_price and trade.sl_price > 0:
        sl_line = [trade.sl_price] * len(ohlc_data)
        additional_plots.append(
            mpf.make_addplot(sl_line, color='red', linestyle='--', width=2, alpha=0.8)
        )
    
    # Entry line (vertical - we'll use scatter for points instead)
    entry_prices = [np.nan] * len(ohlc_data)
    exit_prices = [np.nan] * len(ohlc_data)
    
    # Find closest index for entry
    entry_time_idx = ohlc_data.index.get_indexer([trade.entry_time], method='nearest')[0]
    if 0 <= entry_time_idx < len(entry_prices):
        entry_prices[entry_time_idx] = trade.entry_price
    
    # Find closest index for exit
    if trade.exit_time and trade.exit_price:
        exit_time_idx = ohlc_data.index.get_indexer([trade.exit_time], method='nearest')[0]
        if 0 <= exit_time_idx < len(exit_prices):
            exit_prices[exit_time_idx] = trade.exit_price
    
    # Add entry/exit markers
    additional_plots.append(
        mpf.make_addplot(entry_prices, type='scatter', markersize=200, 
                        marker='^' if trade.direction == 'LONG' else 'v',
                        color='blue', alpha=0.8)
    )
    
    if any(p for p in exit_prices if not np.isnan(p)):
        additional_plots.append(
            mpf.make_addplot(exit_prices, type='scatter', markersize=150,
                           marker='o', color='green' if trade.pnl > 0 else 'red', alpha=0.8)
        )
    
    # Create title with trade details
    if not title:
        pnl_str = f"${trade.pnl:+.0f}" if abs(trade.pnl) >= 1 else f"${trade.pnl:+.2f}"
        title = f"{trade.symbol} {trade.direction} Trade - {pnl_str} ({trade.exit_reason})"
    
    # Plot configuration
    style = mpf.make_marketcolors(up='green', down='red', edge='inherit', wick={'up':'green', 'down':'red'})
    plot_style = mpf.make_mpf_style(marketcolors=style, gridstyle=':', y_on_right=False)
    
    # Create the plot
    fig, axes = mpf.plot(
        ohlc_data,
        type='candle',
        addplot=additional_plots,
        style=plot_style,
        title=title,
        ylabel='Price ($)',
        volume=False,
        figsize=(16, 10),
        returnfig=True
    )
    
    # Add annotations to the main price axis (axes[0])
    ax = axes[0]
    
    # Entry annotation
    if 0 <= entry_time_idx < len(ohlc_data):
        entry_time = ohlc_data.index[entry_time_idx]
        ax.annotate(
            f'ENTRY\n${trade.entry_price:.2f}\nConf: {trade.confidence:.2f}',
            xy=(entry_time_idx, trade.entry_price), 
            xytext=(10, 20), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.7, edgecolor='white'),
            color='white', fontweight='bold', fontsize=10, ha='center'
        )
    
    # TP annotation
    if trade.tp_price and trade.tp_price > 0:
        ax.annotate(
            f'TP: ${trade.tp_price:.2f}',
            xy=(len(ohlc_data) * 0.8, trade.tp_price),
            xytext=(0, 10), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7, edgecolor='white'),
            color='white', fontweight='bold', fontsize=9
        )
    
    # SL annotation 
    if trade.sl_price and trade.sl_price > 0:
        ax.annotate(
            f'SL: ${trade.sl_price:.2f}',
            xy=(len(ohlc_data) * 0.8, trade.sl_price),
            xytext=(0, -15), textcoords='offset points', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7, edgecolor='white'),
            color='white', fontweight='bold', fontsize=9
        )
    
    # Exit annotation
    if trade.exit_time and trade.exit_price and 0 <= exit_time_idx < len(ohlc_data):
        pnl_str = f"${trade.pnl:+.0f}" if abs(trade.pnl) >= 1 else f"${trade.pnl:+.2f}"
        ax.annotate(
            f'EXIT\n${trade.exit_price:.2f}\n{pnl_str}',
            xy=(exit_time_idx, trade.exit_price),
            xytext=(-10, -30), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', 
                     facecolor='green' if trade.pnl > 0 else 'red', alpha=0.7, edgecolor='white'),
            color='white', fontweight='bold', fontsize=10, ha='center'
        )
    
    # Add trade statistics text box
    stats_text = f"""Trade Stats:
Duration: {trade.bars_held} bars
Entry: ${trade.entry_price:.2f}
Exit: ${trade.exit_price:.2f} ({trade.exit_reason})
PnL: ${trade.pnl:+.2f} ({trade.pnl_pct:+.2%})
TP%: {trade.tp_pct:.2%} | SL%: {trade.sl_pct:.2%}
Fees: ${trade.fees_paid:.2f}"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', fontsize=10, 
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Highlight trade period with shaded area
    if trade.exit_time:
        ax.axvspan(entry_time_idx, exit_time_idx, alpha=0.1, 
                  color='green' if trade.pnl > 0 else 'red')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Individual trade plot saved to: {save_path}")
    
    # plt.show()
    plt.close()


def _plot_individual_trade_basic(df: pd.DataFrame, trade: Trade, title: str = None, save_path: str = None, buffer_bars: int = 50):
    """
    Fallback basic line plot for individual trade when mplfinance is not available.
    """
    # Filter to specific symbol if needed
    if 'symbol' in df.columns:
        df = df[df['symbol'] == trade.symbol].copy()
    
    # Similar logic but with basic matplotlib plotting
    df_plot = df.copy()
    if 'timestamp' in df_plot.columns:
        df_plot.set_index('timestamp', inplace=True)
    
    # Find trade period
    entry_idx = df_plot.index.get_indexer([trade.entry_time], method='nearest')[0]
    if trade.exit_time:
        exit_idx = df_plot.index.get_indexer([trade.exit_time], method='nearest')[0]
    else:
        exit_idx = min(entry_idx + trade.bars_held, len(df_plot) - 1)
    
    start_idx = max(0, entry_idx - buffer_bars)
    end_idx = min(len(df_plot) - 1, exit_idx + buffer_bars)
    df_period = df_plot.iloc[start_idx:end_idx + 1].copy()
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot price line
    ax.plot(df_period.index, df_period['close'], label='Close Price', color='black', linewidth=1)
    
    # Entry marker
    ax.scatter(trade.entry_time, trade.entry_price, 
              marker='^' if trade.direction == 'LONG' else 'v',
              color='blue', s=200, label=f'Entry ${trade.entry_price:.2f}', zorder=5)
    
    # Exit marker
    if trade.exit_time and trade.exit_price:
        ax.scatter(trade.exit_time, trade.exit_price, 
                  marker='o', color='green' if trade.pnl > 0 else 'red', 
                  s=150, label=f'Exit ${trade.exit_price:.2f}', zorder=5)
    
    # TP/SL lines
    if trade.tp_price and trade.tp_price > 0:
        ax.axhline(y=trade.tp_price, color='green', linestyle='--', 
                  alpha=0.8, label=f'TP ${trade.tp_price:.2f}')
    
    if trade.sl_price and trade.sl_price > 0:
        ax.axhline(y=trade.sl_price, color='red', linestyle='--',
                  alpha=0.8, label=f'SL ${trade.sl_price:.2f}')
    
    # Trade period highlight
    if trade.exit_time:
        ax.axvspan(trade.entry_time, trade.exit_time, alpha=0.1,
                  color='green' if trade.pnl > 0 else 'red')
    
    if not title:
        pnl_str = f"${trade.pnl:+.0f}" if abs(trade.pnl) >= 1 else f"${trade.pnl:+.2f}"
        title = f"{trade.symbol} {trade.direction} Trade - {pnl_str} ({trade.exit_reason})"
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Price ($)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Individual trade plot saved to: {save_path}")
    
    # plt.show()
    plt.close()


def plot_all_trades_summary(trades: List[Trade], title: str = "All Trades Summary", save_path: str = None):
    """
    Plot summary of ALL trades across all symbols (no price chart needed).
    
    Args:
        trades: List of ALL Trade objects
        title: Chart title  
        save_path: Path to save the plot
    """
    if not trades:
        print("⚠️ No trades to plot")
        return
        
    print(f"📊 Plotting summary for {len(trades)} trades across all symbols")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Cumulative PnL over time (Top Left)
    ax1 = axes[0, 0]
    sorted_trades = sorted(trades, key=lambda t: t.exit_time if t.exit_time else t.entry_time)
    
    trade_dates = [t.exit_time for t in sorted_trades]
    trade_pnl_abs = [t.pnl for t in sorted_trades]
    
    if trade_pnl_abs:
        cum_pnl_abs = np.cumsum(trade_pnl_abs)
        total_pnl = cum_pnl_abs[-1]
        
        ax1.plot(trade_dates, cum_pnl_abs, label=f'Total PnL: ${total_pnl:,.0f}', 
                color='purple', linewidth=2)
        ax1.fill_between(trade_dates, cum_pnl_abs, alpha=0.1, color='purple')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Add markers for significant trades
        big_wins = [t for t in sorted_trades if t.pnl > np.percentile(trade_pnl_abs, 90)]
        big_losses = [t for t in sorted_trades if t.pnl < np.percentile(trade_pnl_abs, 10)]
        
        for t in big_wins:
            ax1.scatter(t.exit_time, cum_pnl_abs[sorted_trades.index(t)], 
                       color='green', s=100, marker='^', zorder=5)
        for t in big_losses:
            ax1.scatter(t.exit_time, cum_pnl_abs[sorted_trades.index(t)], 
                       color='red', s=100, marker='v', zorder=5)
    
    ax1.set_title('Cumulative PnL - All Symbols', fontweight='bold')  
    ax1.set_ylabel('PnL ($)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. PnL Distribution (Top Right)
    ax2 = axes[0, 1]
    pnl_values = [t.pnl for t in trades]
    
    ax2.hist(pnl_values, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Break Even')
    ax2.axvline(x=np.median(pnl_values), color='orange', linestyle='-', label=f'Median: ${np.median(pnl_values):.0f}')
    
    ax2.set_title('PnL Distribution', fontweight='bold')
    ax2.set_xlabel('PnL ($)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Trades by Symbol (Bottom Left) 
    ax3 = axes[1, 0]
    symbol_counts = {}
    symbol_pnl = {}
    for t in trades:
        symbol_counts[t.symbol] = symbol_counts.get(t.symbol, 0) + 1
        symbol_pnl[t.symbol] = symbol_pnl.get(t.symbol, 0) + t.pnl
    
    # Show top 10 symbols by trade count
    top_symbols = sorted(symbol_counts.items(), key=lambda x: -x[1])[:10]
    symbols, counts = zip(*top_symbols) if top_symbols else ([], [])
    
    bars = ax3.bar(range(len(symbols)), counts, color='lightgreen', edgecolor='black')
    ax3.set_title('Trades by Symbol (Top 10)', fontweight='bold')
    ax3.set_xlabel('Symbol')
    ax3.set_ylabel('Number of Trades')
    ax3.set_xticks(range(len(symbols)))
    ax3.set_xticklabels(symbols, rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add trade counts on bars
    for bar, count in zip(bars, counts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{count}', ha='center', fontweight='bold')
    
    # 4. Win/Loss by Direction (Bottom Right)
    ax4 = axes[1, 1]
    long_trades = [t for t in trades if t.direction == 'LONG']
    short_trades = [t for t in trades if t.direction == 'SHORT']
    
    long_wins = len([t for t in long_trades if t.pnl > 0])
    long_losses = len([t for t in long_trades if t.pnl <= 0])
    short_wins = len([t for t in short_trades if t.pnl > 0])
    short_losses = len([t for t in short_trades if t.pnl <= 0])
    
    categories = ['Long Wins', 'Long Losses', 'Short Wins', 'Short Losses']
    values = [long_wins, long_losses, short_wins, short_losses]
    colors = ['green', 'red', 'lightgreen', 'lightcoral']
    
    bars = ax4.bar(categories, values, color=colors, edgecolor='black')
    ax4.set_title('Win/Loss by Direction', fontweight='bold')
    ax4.set_ylabel('Number of Trades')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add counts on bars
    for bar, val in zip(bars, values):
        if val > 0:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{val}', ha='center', fontweight='bold')
    
    # Add summary text
    total_trades = len(trades)
    win_rate = len([t for t in trades if t.pnl > 0]) / total_trades * 100 if total_trades > 0 else 0
    avg_pnl = np.mean(pnl_values) if pnl_values else 0
    
    plt.figtext(0.02, 0.02, 
                f'Total Trades: {total_trades} | Win Rate: {win_rate:.1f}% | Avg PnL: ${avg_pnl:.0f} | Symbols: {len(symbol_counts)}',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 All trades summary saved to: {save_path}")
    
    # plt.show()
    plt.close()


def plot_all_trades_timeline(trades: List[Trade], title: str = "All Trades Timeline", save_path: str = None):
    """
    Plot ALL trades from ALL symbols on a timeline (no price chart needed).
    
    Args:
        trades: List of ALL Trade objects from all symbols
        title: Chart title
        save_path: Path to save the plot
    """
    if not trades:
        print("⚠️ No trades to plot")
        return
        
    print(f"📊 Plotting timeline for ALL {len(trades)} trades across all symbols")
    
    # Sort trades by entry time
    sorted_trades = sorted(trades, key=lambda t: t.entry_time)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # 1. Trades timeline scattered by symbol (Top plot)
    symbols = list(set(t.symbol for t in trades))
    symbol_colors = plt.cm.tab20(np.linspace(0, 1, len(symbols)))
    symbol_color_map = {sym: color for sym, color in zip(symbols, symbol_colors)}
    
    # Plot all trade entries/exits
    for i, trade in enumerate(sorted_trades):
        color = symbol_color_map[trade.symbol]
        
        # Entry marker
        marker = '^' if trade.direction == 'LONG' else 'v'
        entry_color = 'green' if trade.pnl > 0 else 'red'
        
        # Plot entry
        ax1.scatter(trade.entry_time, i, marker=marker, color=entry_color, 
                   s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Plot exit (if available)
        if trade.exit_time:
            ax1.scatter(trade.exit_time, i, marker='o', color=entry_color,
                       s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
            
            # Draw line connecting entry to exit
            ax1.plot([trade.entry_time, trade.exit_time], [i, i], 
                    color=entry_color, alpha=0.3, linewidth=2)
        
        # Add symbol labels for significant trades
        if abs(trade.pnl) > np.percentile([abs(t.pnl) for t in trades], 90):
            ax1.annotate(f'{trade.symbol}\n${trade.pnl:+.0f}', 
                        (trade.entry_time, i), xytext=(5, 0), 
                        textcoords='offset points', fontsize=8, alpha=0.8)
    
    ax1.set_title(f'{title} - All {len(trades)} Trades', fontweight='bold')
    ax1.set_ylabel('Trade Index')
    ax1.grid(True, alpha=0.3)
    ax1.legend(['Long Win', 'Long Loss', 'Short Win', 'Short Loss'], loc='upper right')
    
    # Add stats text
    win_count = len([t for t in trades if t.pnl > 0])
    total_pnl = sum(t.pnl for t in trades)
    win_rate = win_count / len(trades) * 100
    
    ax1.text(0.02, 0.98, f'Total: {len(trades)} trades\nWin Rate: {win_rate:.1f}%\nTotal PnL: ${total_pnl:,.0f}', 
            transform=ax1.transAxes, verticalalignment='top', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 2. Cumulative PnL over time (Bottom plot)
    trade_dates = [t.exit_time if t.exit_time else t.entry_time for t in sorted_trades]
    trade_pnl_abs = [t.pnl for t in sorted_trades]
    
    if trade_pnl_abs:
        cum_pnl_abs = np.cumsum(trade_pnl_abs)
        
        ax2.plot(trade_dates, cum_pnl_abs, label=f'Cumulative PnL (${cum_pnl_abs[-1]:,.0f})', 
                color='purple', linewidth=2)
        ax2.fill_between(trade_dates, cum_pnl_abs, alpha=0.1, color='purple')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Break Even')
        
        # Mark significant drawdowns/peaks
        peaks_idx = []
        for i in range(1, len(cum_pnl_abs)-1):
            if cum_pnl_abs[i] > cum_pnl_abs[i-1] and cum_pnl_abs[i] > cum_pnl_abs[i+1]:
                peaks_idx.append(i)
        
        for idx in peaks_idx[-5:]:  # Show last 5 peaks
            ax2.scatter(trade_dates[idx], cum_pnl_abs[idx], 
                       color='green', s=50, zorder=5)
    
    ax2.set_title('Cumulative PnL - All Symbols', fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('PnL ($)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 All trades timeline saved to: {save_path}")
    
    # plt.show()
    plt.close()


def plot_backtest_trades(df: pd.DataFrame, trades: List[Trade], title: str = "Backtest Trades", save_path: str = None, trade_limit: Optional[str] = None):
    """
    Plot equity curves and trade entries/exits on price chart.
    
    Args:
        df: DataFrame with OHLCV data (must have datetime index or timestamp column)
        trades: List of Trade objects
        title: Chart title
        save_path: Path to save the plot
        trade_limit: If set to a number string, plot only the last N trades. If "all" or None, plot all.
    """
    if df.empty or not trades:
        print("⚠️ No data or trades to plot")
        return
    
    # Limit trades if requested (use last N trades by exit_time)
    if trade_limit is not None and trade_limit != 'all':
        try:
            n = int(trade_limit)
            if n > 0:
                trades = sorted(trades, key=lambda t: (t.exit_time or t.entry_time))
                trades = trades[-n:]
                print(f"📊 Plotting last {len(trades)} trades (limited from {trade_limit})")
        except (ValueError, TypeError):
            # ignore invalid trade_limit values and fall back to all trades
            print(f"⚠️ Invalid trade limit '{trade_limit}', plotting all trades")
    else:
        print(f"📊 Plotting all {len(trades)} trades")

    # Ensure timestamp is index
    df_plot = df.copy()
    if 'timestamp' in df_plot.columns:
        df_plot.set_index('timestamp', inplace=True)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Check for multiple symbols - provide options for handling
    if 'symbol' in df_plot.columns:
        symbols = df_plot['symbol'].unique()
        if len(symbols) > 1:
            
            # Count trades per symbol 
            trade_counts = {}
            for t in trades:
                trade_counts[t.symbol] = trade_counts.get(t.symbol, 0) + 1
            
            # Sort symbols by trade count
            sorted_symbols = sorted(trade_counts.items(), key=lambda x: -x[1])
            
            print(f"📊 Found {len(symbols)} symbols with trades:")
            for symbol, count in sorted_symbols[:10]:  # Show top 10
                print(f"   {symbol}: {count} trades")
            
            # Option 1: Use symbol with most trades (current behavior)
            if trade_counts:
                best_symbol = sorted_symbols[0][0]
                print(f"📉 Plotting for single symbol: {best_symbol} ({trade_counts[best_symbol]} trades)")
                df_plot = df_plot[df_plot['symbol'] == best_symbol] 
                trades = [t for t in trades if t.symbol == best_symbol]
                title += f" ({best_symbol})"
            else:
                print("⚠️ No trades found for any symbol")
                return

    # 1. Price Chart with Trades
    ax1.plot(df_plot.index, df_plot['close'], label='Close Price', color='gray', alpha=0.5, linewidth=1)
    
    # Plot trades
    long_entries = []
    short_entries = []
    win_exits = []
    loss_exits = []
    
    # For annotations
    annotations = []

    for t in trades:
        # Entry Marker
        if t.direction == 'LONG':
            long_entries.append((t.entry_time, t.entry_price, t.confidence))
            color = 'green' if t.pnl > 0 else 'red'
        else:
            short_entries.append((t.entry_time, t.entry_price, t.confidence))
            color = 'green' if t.pnl > 0 else 'red'
            
        # Exit Marker & Line
        if t.pnl > 0:
            win_exits.append((t.exit_time, t.exit_price))
            # Draw line
            ax1.plot([t.entry_time, t.exit_time], [t.entry_price, t.exit_price], 
                    color='green', alpha=0.3, linewidth=1, linestyle='--')
        else:
            loss_exits.append((t.exit_time, t.exit_price))
            ax1.plot([t.entry_time, t.exit_time], [t.entry_price, t.exit_price], 
                    color='red', alpha=0.3, linewidth=1, linestyle='--')
        
        # Add annotation for high confidence or outliers
        # Only annotate if confidence is available and > 0
        if t.confidence > 0:
            annotations.append({
                'time': t.entry_time,
                'price': t.entry_price,
                'text': f"{t.confidence:.2f}",
                'color': 'blue' if t.direction == 'LONG' else 'orange'
            })
        
        # Add PnL annotation at exit point
        if t.exit_time and t.exit_price:
            pnl_text = f"${t.pnl:+.0f}" if abs(t.pnl) >= 1 else f"${t.pnl:+.2f}"
            annotations.append({
                'time': t.exit_time,
                'price': t.exit_price,
                'text': pnl_text,
                'color': 'green' if t.pnl > 0 else 'red'
            })

    # Plot Entry Makers (size/alpha by confidence if possible, or just standard)
    if long_entries:
        times, prices, confs = zip(*long_entries)
        # Scale size by confidence (e.g., 50 to 150)
        sizes = [c * 150 for c in confs] if confs[0] > 0 else 50
        sc = ax1.scatter(times, prices, marker='^', c=confs, cmap='Blues', vmin=0.5, vmax=1.0, s=sizes, label='Long Entry', zorder=5, edgecolors='black')
        plt.colorbar(sc, ax=ax1, label='Long Confidence')
        
    if short_entries:
        times, prices, confs = zip(*short_entries)
        sizes = [c * 150 for c in confs] if confs[0] > 0 else 50
        sc2 = ax1.scatter(times, prices, marker='v', c=confs, cmap='Oranges', vmin=0.5, vmax=1.0, s=sizes, label='Short Entry', zorder=5, edgecolors='black')
        plt.colorbar(sc2, ax=ax1, label='Short Confidence')
        
    if win_exits:
        times, prices = zip(*win_exits)
        ax1.scatter(times, prices, marker='o', color='green', s=30, label='Take Profit', zorder=5)
        
    if loss_exits:
        times, prices = zip(*loss_exits)
        ax1.scatter(times, prices, marker='x', color='red', s=30, label='Stop Loss', zorder=5)
    
    # Add text annotations with offset to avoid overlap
    for i, ann in enumerate(annotations):
        # Use different offsets for entry vs exit annotations
        if 'confidence' in str(ann['text']) or ann['color'] in ['blue', 'orange']:
            xytext = (0, 15)  # Above for entry confidence
        else:
            xytext = (0, -15)  # Below for PnL 
            
        ax1.annotate(ann['text'], (ann['time'], ann['price']), 
                     xytext=xytext, textcoords='offset points', 
                     fontsize=8, color=ann['color'], ha='center', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor=ann['color']))
    
    # 🔵 FOCUS ON TRADE AREA: Set x-axis limits to focus around trades only
    if trades:
        # Find time range of trades
        all_times = []
        for t in trades:
            all_times.append(t.entry_time)
            if t.exit_time:
                all_times.append(t.exit_time)
        
        if all_times:
            min_time = min(all_times)
            max_time = max(all_times)
            
            # Add 10% buffer on each side
            time_delta = max_time - min_time
            buffer = time_delta * 0.1
            
            # Set focus area
            focus_start = min_time - buffer
            focus_end = max_time + buffer
            
            ax1.set_xlim(focus_start, focus_end)
            
            print(f"📍 Focusing chart on trade period: {focus_start.strftime('%Y-%m-%d')} to {focus_end.strftime('%Y-%m-%d')}")
    
    ax1.set_title(f'{title} - Price & Trades (Color/Size = Confidence)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative PnL
    # Sort trades by exit time
    sorted_trades = sorted(trades, key=lambda t: t.exit_time if t.exit_time else t.entry_time)
    
    trade_dates = [t.exit_time for t in sorted_trades]
    trade_pnl_abs = [t.pnl for t in sorted_trades]  # Absolute PnL in dollars
    trade_pnl_pct = [t.pnl_pct * 100 for t in sorted_trades]  # Percentage return
    
    # Calculate summary stats
    total_pnl = sum(trade_pnl_abs)
    total_pct = sum(trade_pnl_pct) 
    avg_pnl = total_pnl / len(trades) if trades else 0
    
    # Reconstruct equity curves aligned with time
    if trade_pnl_abs:
        cum_pnl_abs = np.cumsum(trade_pnl_abs)
        cum_pnl_pct = np.cumsum(trade_pnl_pct)
        
        # Plot absolute PnL (more meaningful for understanding actual gains)
        ax2.plot(trade_dates, cum_pnl_abs, label=f'Cumulative PnL (${total_pnl:,.0f})', 
                color='purple', linewidth=2)
        ax2.fill_between(trade_dates, cum_pnl_abs, alpha=0.1, color='purple')
        
        # Add horizontal line at zero
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Break Even')
        
        # Add summary text
        ax2.text(0.02, 0.98, f'Total: ${total_pnl:,.0f}\nAvg: ${avg_pnl:,.0f}/trade\nTrades: {len(trades)}', 
                transform=ax2.transAxes, verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax2.set_title('Cumulative PnL ($)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('PnL ($)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plt.show()  # Show the chart
    plt.close()


def run_trailing_comparison(df: pd.DataFrame, base_config: BacktestConfig):
    """
    Compare strategy performance WITH and WITHOUT Trailing Stop.
    """
    print("\n" + "="*60)
    print("🔄 RUNNING TRAILING STOP COMPARISON")
    print("="*60)
    
    results = {}
    
    # 1. Baseline (No Trailing)
    print("\n🔹 Running Baseline (Fixed SL/TP)...")
    base_config.use_trailing_stop = False
    backtester_base = ThreeStageBacktester(base_config)
    results['Baseline (Fixed SL/TP)'] = backtester_base.run_backtest(df, verbose=False)
    
    # 2. Trailing Stop
    print(f"\n🔹 Running Trailing Stop (Start: {base_config.trailing_start_pct:.1%}, Step: {base_config.trailing_step_pct:.1%})...")
    trailing_config = deepcopy(base_config)
    trailing_config.use_trailing_stop = True
    backtester_trail = ThreeStageBacktester(trailing_config)
    results[f'Trailing ({trailing_config.trailing_start_pct:.0%}/{trailing_config.trailing_step_pct:.0%})'] = backtester_trail.run_backtest(df, verbose=False)
    
    # Print Comparison Table
    print("\n" + "="*80)
    print(f"{'METRIC':<25} | {'BASELINE':<20} | {'TRAILING STOP':<20} | {'DIFF':<10}")
    print("-" * 80)
    
    # Helper to get value
    def get_val(r, attr):
        return getattr(r, attr)
        
    metrics = [
        ('Final Capital ($)', 'equity_curve', lambda x: x[-1] if x else 0),
        ('Total Return (%)', 'total_return', lambda x: x * 100),
        ('Max Drawdown (%)', 'max_drawdown', lambda x: x * 100),
        ('Win Rate (%)', 'win_rate', lambda x: x * 100),
        ('Total Trades', 'total_trades', lambda x: x),
        ('Profit Factor', 'profit_factor', lambda x: x),
        ('Avg Trade ($)', 'avg_trade_pnl', lambda x: x),
    ]
    
    base_res = results['Baseline (Fixed SL/TP)']
    trail_res = list(results.values())[1]
    
    for label, attr, fmt_func in metrics:
        if attr == 'equity_curve':
            val_base = fmt_func(base_res.equity_curve)
            val_trail = fmt_func(trail_res.equity_curve)
        else:
            val_base = fmt_func(getattr(base_res, attr))
            val_trail = fmt_func(getattr(trail_res, attr))
            
        diff = val_trail - val_base
        print(f"{label:<25} | {val_base:>18.2f} | {val_trail:>18.2f} | {diff:>+9.2f}")
        
    print("="*80)
    
    # Plot Comparison
    lev_str = f"{base_config.leverage:.0f}x" if base_config.leverage > 1 else "1x"
    plot_equity_curve(results, title=f"Trailing Stop Comparison (Lev {lev_str})", 
                     save_path=str(DATA_DIR.parent / 'backtest_trailing_comparison.png'))
    
    plt.show()  # Show the chart
    plt.close()


def run_kelly_comparison(df: pd.DataFrame, base_config: BacktestConfig):
    """
    Compare strategy performance with EVEN Sizing vs KELLY Sizing.
    """
    print("\n" + "="*60)
    print("🔄 RUNNING KELLY SIZING COMPARISON")
    print("="*60)
    
    results = {}
    
    # 1. Even Sizing (Fixed Risk per Trade)
    print("\n🔹 Running Even Sizing (Fixed Risk)...")
    base_config.use_kelly = False
    # Relax constraints for comparison to be visible
    base_config.max_concentration = 1.0  # Allow up to 100% per trade (if risk allows)
    # base_config.initial_capital = 10000 # Use larger capital to avoid small balance issues
    
    backtester_even = ThreeStageBacktester(base_config)
    results['Even Sizing'] = backtester_even.run_backtest(df, verbose=False)
    
    # 2. Kelly Criterion
    print(f"\n🔹 Running Kelly Criterion (Fraction: {base_config.kelly_fraction:.2f})...")
    kelly_config = deepcopy(base_config)
    kelly_config.use_kelly = True
    kelly_config.max_concentration = 1.0  # Allow up to 100% per trade
    
    backtester_kelly = ThreeStageBacktester(kelly_config)
    results['Kelly Sizing'] = backtester_kelly.run_backtest(df, verbose=False)
    
    # Print Comparison Table
    print("\n" + "="*80)
    print(f"{'METRIC':<25} | {'EVEN SIZING':<20} | {'KELLY SIZING':<20} | {'DIFF':<10}")
    print("-" * 80)
    
    # Metrics to compare
    metrics = [
        ('Final Capital ($)', 'equity_curve', lambda x: x[-1] if x else 0),
        ('Total Return (%)', 'total_return', lambda x: x * 100),
        ('Max Drawdown (%)', 'max_drawdown', lambda x: x * 100),
        ('Win Rate (%)', 'win_rate', lambda x: x * 100),
        ('Total Trades', 'total_trades', lambda x: x),
        ('Profit Factor', 'profit_factor', lambda x: x),
        ('Avg Trade ($)', 'avg_trade_pnl', lambda x: x),
    ]
    
    res_even = results['Even Sizing']
    res_kelly = results['Kelly Sizing']
    
    for label, attr, fmt_func in metrics:
        if attr == 'equity_curve':
            val_even = fmt_func(res_even.equity_curve)
            val_kelly = fmt_func(res_kelly.equity_curve)
        else:
            val_even = fmt_func(getattr(res_even, attr))
            val_kelly = fmt_func(getattr(res_kelly, attr))
            
        diff = val_kelly - val_even
        print(f"{label:<25} | {val_even:>18.2f} | {val_kelly:>18.2f} | {diff:>+9.2f}")
        
    print("="*80)
    
    # Plot Comparison
    lev_str = f"{base_config.leverage:.0f}x" if base_config.leverage > 1 else "1x"
    plot_equity_curve(results, title=f"Kelly vs Even Sizing Comparison (Lev {lev_str})", 
                     save_path=str(DATA_DIR.parent / 'backtest_kelly_comparison.png'))
    


def run_timeframe_comparison(base_config: BacktestConfig):
    """
    Compare strategy performance across all supported timeframes.
    """
    print("\n" + "="*70)
    print("🚀 RUNNING MULTI-TIMEFRAME COMPARISON")
    print("="*70)
    
    results: Dict[str, BacktestResult] = {}
    
    # We use a copy of the config but override the timeframe
    comparison_timeframes = [tf for tf in SUPPORTED_TIMEFRAMES if tf != '1h']
    
    for tf in comparison_timeframes:
        print(f"\n🔹 Testing Timeframe: {tf.upper()}...")
        
        data_path = PROCESSED_DIR / f'features_{tf}_full.parquet'
        if not data_path.exists():
            print(f"   ⚠️ Data not found for {tf}, skipping.")
            continue
            
        try:
            df = pd.read_parquet(data_path)
            
            # Filter for test period
            df = df.sort_values('timestamp')
            
            if base_config.start_date or base_config.end_date:
                df_test = df.copy()
                if base_config.start_date:
                    start_dt = pd.to_datetime(base_config.start_date)
                    df_test = df_test[df_test['timestamp'] >= start_dt]
                if base_config.end_date:
                    end_dt = pd.to_datetime(base_config.end_date)
                    df_test = df_test[df_test['timestamp'] <= end_dt]
                
                if df_test.empty:
                    print(f"   ⚠️ No data in range {base_config.start_date} to {base_config.end_date}, skipping.")
                    continue
            else:
                # Default: last 6 months or 20%
                latest_date = df['timestamp'].max()
                test_start_date = latest_date - pd.DateOffset(months=6)
                df_test = df[df['timestamp'] >= test_start_date].copy()
                
                if len(df_test) < 500:
                    test_start_idx = int(len(df) * 0.8)
                    df_test = df.iloc[test_start_idx:].copy()
            
            # Setup config for this timeframe
            tf_config = get_timeframe_config(tf)
            config = deepcopy(base_config)
            config.timeframe = tf
            config.max_bars = tf_config.max_bars
            
            # Create backtester and run
            backtester = ThreeStageBacktester(config)
            result = backtester.run_backtest(df_test, verbose=False)
            results[tf] = result
            
            print(f"   ✅ Done: {result.total_trades} trades, {result.total_return:.1%} return")
            
        except Exception as e:
            print(f"   ❌ Error testing {tf}: {e}")
            
    if not results:
        print("No results to compare!")
        return
        
    # Print Comparison Table
    print("\n" + "="*100)
    print(f"{'TIMEFRAME':<12} | {'TRADES':>8} | {'WIN%':>8} | {'RETURN':>12} | {'MAX DD':>10} | {'SHARPE':>8} | {'PF':>8}")
    print("-" * 100)
    
    for tf in comparison_timeframes:
        if tf not in results: continue
        res = results[tf]
        print(f"{tf.upper():<12} | {res.total_trades:>8} | {res.win_rate:>7.1%} | {res.total_return:>11.1%} | {res.max_drawdown:>9.1%} | {res.sharpe_ratio:>8.2f} | {res.profit_factor:>8.2f}")
        
    print("="*100)
    
    # Plot Comparison
    lev_str = f"{base_config.leverage:.0f}x" if base_config.leverage > 1 else "1x"
    plot_equity_curve(results, title=f"Timeframe Comparison (Lev {lev_str})", 
                     save_path=str(DATA_DIR.parent / 'backtest_timeframe_comparison_advanced.png'))
    
    print(f"\n📊 Comparison chart saved to: {DATA_DIR.parent / 'backtest_timeframe_comparison_advanced.png'}")
    plt.show()


if __name__ == '__main__':
    main()