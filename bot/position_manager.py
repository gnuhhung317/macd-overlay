import pandas as pd
from typing import Dict, List, Any
import time
from datetime import datetime
from .config import BotConfig
from .db import DatabaseManager
from .executor import ExchangeExecutor
from .signal_engine import SignalEngine
from .data_provider import DataProvider

class PositionManager:
    def __init__(
        self, 
        config: BotConfig, 
        db: DatabaseManager, 
        executor: ExchangeExecutor,
        signal_engine: SignalEngine,
        data_provider: DataProvider,
        notifier: Any = None
    ):
        self.config = config
        self.db = db
        self.executor = executor
        self.signal_engine = signal_engine
        self.data_provider = data_provider
        self.notifier = notifier
        
        # In-memory state tracking to reduce DB reads
        self.active_positions = {}
        self._load_state()

    def _load_state(self):
        """Restore active positions from DB on startup"""
        trades = self.db.get_active_trades()
        for t in trades:
            self.active_positions[t['symbol']] = t
        print(f"[PositionManager] Restored {len(self.active_positions)} active positions")

    def process_symbol(self, symbol: str, timeframe: str):
        """Main logic loop for a single symbol"""
        
        # 1. Check if we already have a position
        current_trade = self.active_positions.get(symbol)
        
        # 2. Fetch Data
        df = self.data_provider.fetch_closed_candles(symbol, timeframe)
        if df.empty:
            return

        # Augment with indicators
        df = self.data_provider.calculate_indicators(df)
        
        # 3. Handle Active Position (Management)
        if current_trade:
            self._manage_active_position(current_trade, symbol, df)
            return

        # 4. Handle No Position
        # LEGACY: self._scan_for_entry(symbol, df, timeframe)
        # We now use SmartScanner in main.py for new entries. 
        # So we do nothing here to avoid double scanning / WAIT logs.
        return

    def _manage_active_position(self, trade: Dict, symbol: str, df: pd.DataFrame):
        """Logic to manage an open trade (check SL/TP)"""
        current_price = df.iloc[-1]['close'] # Use close of last candle for now, or get_current_price
        # Better: use get_current_price for realtime accuracy
        live_price = self.data_provider.get_current_price(symbol)
        if live_price > 0:
            current_price = live_price
            
        direction = trade['direction']
        sl_price = trade['sl_price']
        tp_price = trade['tp_price']
        
        exit_reason = None
        exit_price = 0.0
        pnl = 0.0
        
        # Check SL/TP
        if direction == 'LONG':
            if current_price <= sl_price:
                exit_reason = 'SL_HIT'
                exit_price = sl_price
            elif current_price >= tp_price:
                exit_reason = 'TP_HIT'
                exit_price = tp_price
        else: # SHORT
            if current_price >= sl_price:
                exit_reason = 'SL_HIT'
                exit_price = sl_price
            elif current_price <= tp_price:
                exit_reason = 'TP_HIT'
                exit_price = tp_price
                
        # Check Timeout (Bars Held)
        entry_time_str = trade.get('entry_time')
        if entry_time_str and not exit_reason:
            try:
                # Handle potential string or datetime object
                if isinstance(entry_time_str, str):
                    entry_time = pd.Timestamp(entry_time_str)
                else:
                    entry_time = entry_time_str
                    
                # Calculate time elapsed
                time_elapsed = datetime.now() - entry_time
                
                # Parse timeframe to minutes
                tf = self.config.strategy.timeframes[0] if self.config.strategy.timeframes else '1d'
                msg_tf = tf
                
                minutes_per_candle = 1440 # Default 1d
                if tf.endswith('m'):
                    minutes_per_candle = int(tf[:-1])
                elif tf.endswith('h'):
                    minutes_per_candle = int(tf[:-1]) * 60
                elif tf.endswith('d'):
                    minutes_per_candle = int(tf[:-1]) * 1440
                elif tf.endswith('w'):
                    minutes_per_candle = int(tf[:-1]) * 10080
                    
                bars_held = time_elapsed.total_seconds() / 60 / minutes_per_candle
                
                timeout = getattr(self.config.strategy, 'timeout_candles', 10)
                
                if bars_held > timeout:
                    exit_reason = 'TIMEOUT'
                    exit_price = current_price
                    print(f"⌛ {symbol} TIMEOUT ({bars_held:.1f} bars > {timeout})")
                    
            except Exception as e:
                print(f"⚠️ Error checking timeout for {symbol}: {e}")
                
        if exit_reason:
            print(f"🛑 {symbol} {exit_reason}! Price: {current_price} (SL: {sl_price}, TP: {tp_price})")
            
            # Close trade
            if self.config.exchange.dry_run:
                # Calculate PnL locally for simulation
                size = trade['size']
                entry_price = trade['entry_price']
                leverage = trade.get('leverage', 1)
                
                # Apply slippage
                slippage = self.config.exchange.slippage
                if direction == 'LONG':
                    if exit_reason == 'SL_HIT':
                        exit_price = exit_price * (1 - slippage)
                    else:
                        exit_price = exit_price * (1 - slippage)
                    pnl_pct = (exit_price - entry_price) / entry_price
                else:
                    if exit_reason == 'SL_HIT':
                        exit_price = exit_price * (1 + slippage)
                    else:
                        exit_price = exit_price * (1 + slippage)
                    pnl_pct = (entry_price - exit_price) / entry_price
                
                # PnL = Position Value * % Change
                # Position Value = Margin * Leverage
                # Or simply size * pnl_pct for linear contracts (assuming size in USDT)
                pnl = size * pnl_pct
                
                # Update executor balance
                if hasattr(self.executor, 'update_balance'):
                    self.executor.update_balance(pnl)
                
                # Update DB
                self.db.update_trade(trade['id'], {
                    'status': 'CLOSED',
                    'exit_price': exit_price,
                    'exit_time': datetime.now(),
                    'exit_reason': exit_reason,
                    'pnl': pnl
                })
                
                # Clean up local state
                del self.active_positions[symbol]
                print(f"✅ Trade Closed: {symbol} | PnL: ${pnl:.2f}")

                # Send Telegram Alert
                if self.notifier:
                    emoji = "🤑" if pnl > 0 else "😭"
                    msg = (
                        f"{emoji} <b>Trade Closed: {symbol}</b>\n"
                        f"Type: {exit_reason}\n"
                        f"PnL: <b>${pnl:.2f} ({pnl_pct*100:.2f}%)</b>\n"
                        f"Exit Price: {exit_price}\n"
                        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    self.notifier.send_message(msg)

            else:
                # Real execution - allow exchange to handle SL/TP or market close here
                # For now we assume if SL/TP orders were placed, we just track.
                # But if we need to force close:
                self.executor.close_position(symbol)
                # We would wait for WS update to partial/full fill to update DB
                pass

    def _calculate_position_size(self, capital: float, sl_pct: float, confidence: float) -> float:
        """Calculate position size using Kelly or Fixed Risk"""
        # 1. Fixed Risk Sizing
        risk_amount = capital * self.config.risk.max_risk_per_trade
        
        # 2. Kelly Sizing (Optional)
        if self.config.risk.use_kelly and confidence > 0.5:
            # Estimate RR = 2.0 (conservative baseline)
            estimated_rr = 2.0
            p = confidence
            q = 1 - p
            b = estimated_rr
            
            kelly_f = (p * b - q) / b
            kelly_f = max(0, min(kelly_f, 0.25)) # Cap at 25%
            kelly_f *= self.config.risk.kelly_fraction # Safety fraction
            
            # Use Kelly risk if it suggests less risk than fixed (or more? usually we replace fixed)
            # Strategy: Use Kelly to determine risk_amount
            risk_amount = capital * kelly_f
            
        # 3. Calculate Size from Risk & SL
        if sl_pct <= 0: sl_pct = 0.02
        position_size = risk_amount / sl_pct
        
        # 4. Apply Limits
        # ⚠️ High Risk: Multiply size by leverage to match backtest behavior
        position_size = position_size * self.config.exchange.leverage
        
        # Max Concentration limit (applied to the leveraged position size)
        max_position = capital * self.config.risk.max_concentration * self.config.exchange.leverage
        position_size = min(position_size, max_position)
        
        # Available Capital check
        # For Dry Run, we use virtual balance. For Real, we use get_balance
        # Note: 'capital' passed in is already available balance
        if position_size > capital * self.config.exchange.leverage:
            position_size = capital * self.config.exchange.leverage
            
        return max(0, position_size)

    def _scan_for_entry(self, symbol: str, df: pd.DataFrame, timeframe: str):
        """Logic to find and execute new entries"""
        
        # Risk Check: Max Slots
        if len(self.active_positions) >= self.config.risk.max_open_positions:
            return

        # NEW: Fresh Crossover Check
        if self.config.strategy.require_fresh_crossover:
            last_exit = self.db.get_last_trade_exit(symbol)
            if last_exit:
                # Crossover time is the timestamp of the LAST CLOSED candle (since we look at closed candles)
                crossover_time = df.iloc[-1]['timestamp']
                
                # Ensure crossover is AFTER last exit
                # If crossover_time <= last_exit, it means this signal belongs to a past event we already traded
                if crossover_time <= last_exit:
                   # print(f"⏳ Skipping {symbol}: Waiting for fresh crossover (Exit: {last_exit}, Cross: {crossover_time})")
                    return

        # Fetch Funding Rate
        funding_rate = self.data_provider.get_funding_rate(symbol)

        # Analyze Signal
        analysis = self.signal_engine.analyze(symbol, df, timeframe, funding_rate=funding_rate)
        
        # Log signal for audit
        self.db.log_signal({
            'symbol': symbol,
            'timeframe': timeframe,
            'confidence': analysis['confidence'],
            'sl_pct': analysis['sl'],
            'tp_pct': analysis['tp'],
            'action': analysis['action'],
            'raw_data': analysis['metadata']
        })

        if analysis['action'] == "ENTRY":
            self._execute_entry(symbol, analysis)

    def execute_calculated_signal(self, signal_data: Dict[str, Any], timeframe: str):
        """
        Execute a signal that was already calculated by the SmartScanner.
        Logs to DB first, then checks if actionable.
        """
        print(f"DEBUG: execute_calculated_signal for {signal_data['symbol']} at {signal_data['timestamp']}")
        symbol = signal_data['symbol']
        timestamp = signal_data['timestamp']
        
        # 0. Check Fresh Crossover to prevent re-entering a past signal
        if self.config.strategy.require_fresh_crossover:
            last_exit = self.db.get_last_trade_exit(symbol)
            if last_exit:
                if isinstance(timestamp, str):
                    crossover_time = pd.Timestamp(timestamp).to_pydatetime()
                else:
                    crossover_time = timestamp
                
                # Make naive for comparison
                if getattr(crossover_time, 'tzinfo', None) is not None:
                     crossover_time = crossover_time.replace(tzinfo=None)
                safe_last_exit = last_exit.replace(tzinfo=None) if getattr(last_exit, 'tzinfo', None) is not None else last_exit
                     
                if crossover_time <= safe_last_exit:
                    print(f"⏳ Skipping {symbol}: Signal timestamp {crossover_time} is older than last exit {safe_last_exit}")
                    return

        # 1. Log to DB (if new)
        # Check deduplication
        if not self.db.check_signal_exists(symbol, timeframe, timestamp):
             # Format for DB
            log_entry = {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S") if hasattr(timestamp, 'strftime') else str(timestamp),
                'confidence': signal_data['confidence'],
                'sl_pct': signal_data['sl_pct'],
                'tp_pct': signal_data['tp_pct'],
                'action': signal_data['type'], # Store LONG/SHORT in action for now, or "SIGNAL_LONG"
                'raw_data': {
                    'status': signal_data['status'],
                    'signal_price': signal_data['signal_price'],
                    'current_price': signal_data['current_price'],
                    'risk_reward': signal_data.get('risk_reward', 0),
                    'meta': signal_data.get('meta', {})
                }
            }
            # Special logic: If it's a "TOO LATE" signal, we might still want to log it for history, 
            # but maybe mark it. For now, we log everything the scanner finds as a "Signal".
            self.db.log_signal(log_entry)
            # print(f"📝 Logged Signal: {symbol} {signal_data['type']} ({signal_data['status']})")
        
        # 2. Execution Logic
        if symbol in self.active_positions:
            return

        # 3. Filter Entry Zone: Only enter if price is in a favorable area
        allowed_zones = self.config.strategy.allowed_zones
        # print(f"DEBUG: Checking {symbol} zone '{signal_data['status']}' against {allowed_zones}")
        if not any(zone in signal_data['status'] for zone in allowed_zones):
            # Block CHASING and TOO LATE
            print(f"🚫 {symbol} filtered by Entry Zone: {signal_data['status']}")
            return
        
        # 4. Refined Score Filter
        min_score = self.config.strategy.min_refined_score
        refined_score = signal_data.get('refined_score', 0.0)
        if refined_score < min_score:
            print(f"🚫 {symbol} filtered by Refined Score: {refined_score:.2f} < {min_score:.2f}")
            return
        # print(f"DEBUG: {symbol} passed zone and score filters")

        # Convert Scanner format to internal Analysis format
        analysis = {
            "action": "ENTRY",
            "signal": "BULLISH" if signal_data['type'] == 'LONG' else "BEARISH",
            "confidence": signal_data['confidence'],
            "sl": signal_data['sl_pct'],
            "tp": signal_data['tp_pct'],
            "risk_reward": signal_data.get('risk_reward', 0),
            "metadata": signal_data.get('meta', {})
        }
        
        print(f"🤖 SmartScanner Entry for {symbol} ({signal_data['status']}) | Conf: {analysis['confidence']:.2%}")
        self._execute_entry(symbol, analysis)

    def _execute_entry(self, symbol: str, analysis: Dict):
        """Execute the entry order"""
        print(f"🚀 ENTRY SIGNAL FOUND: {symbol} | Conf: {analysis['confidence']:.2%} | RR: {analysis.get('risk_reward', 0):.2f}")
        
        # 1. Calc Sizing
        balance = self.executor.get_balance()
        if balance <= 0:
            print("❌ Insufficient Balance")
            return

        # Pseudo-ISOLATED: Cap SL at 0.99/leverage so we simulate isolated liquidation
        if self.config.exchange.margin_mode.upper() == "ISOLATED":
            pseudo_iso_sl = 0.99 / self.config.exchange.leverage
            ml_sl = analysis.get('sl', pseudo_iso_sl)
            
            if ml_sl <= 0:
                ml_sl = pseudo_iso_sl # Fallback if ML didn't provide one
                
            analysis['sl'] = min(ml_sl, pseudo_iso_sl)
            
            if analysis['sl'] == pseudo_iso_sl:
                print(f"🛡️ Pseudo-ISOLATED Mode: Capping ML SL at liquidation risk: {pseudo_iso_sl:.2%} (1/leverage)")
            else:
                print(f"🛡️ Pseudo-ISOLATED Mode: Using ML SL {ml_sl:.2%} (Safer than {pseudo_iso_sl:.2%})")

        sl_pct = analysis['sl'] if analysis['sl'] > 0 else 0.01
        
        final_size = self._calculate_position_size(balance, sl_pct, analysis['confidence'])
        
        if final_size <= 0:
            print("❌ Calculated size is 0")
            return
        
        # Get Current Price for SL/TP absolute values
        current_price = self.data_provider.get_current_price(symbol)
        if current_price <= 0:
            print("❌ Could not get current price")
            return

        direction = "LONG" if analysis['signal'] == "BULLISH" else "SHORT"
        
        if direction == "LONG":
            sl_price = current_price * (1 - analysis['sl'])
            tp_price = current_price * (1 + analysis['tp'])
        else:
            sl_price = current_price * (1 + analysis['sl'])
            tp_price = current_price * (1 - analysis['tp'])

        # Calculate Trailing Stop Parameters
        trailing_callback = self.config.strategy.trailing_stop_callback
        activation_price = 0.0
        
        if trailing_callback > 0 and self.config.strategy.trailing_stop_activation_pct > 0:
            act_pct = self.config.strategy.trailing_stop_activation_pct
            if direction == "LONG":
                activation_price = current_price * (1 + act_pct)
            else:
                activation_price = current_price * (1 - act_pct)

        # 2. Execute Order
        try:
            order_result = self.executor.place_order(
                symbol=symbol,
                side=direction.lower(),
                size=final_size,
                leverage=self.config.exchange.leverage,
                sl_price=sl_price,
                tp_price=tp_price,
                trailing_callback=trailing_callback,
                activation_price=activation_price
            )
            
            if not order_result or 'order_id' not in order_result:
                print(f"❌ Order placement failed for {symbol}")
                return

            # 3. Save to DB
            trade_record = {
                "symbol": symbol,
                "direction": direction,
                "status": "OPEN",
                "entry_price": current_price,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "size": final_size,
                "leverage": self.config.exchange.leverage,
                "raw_data": order_result,
                "entry_time": datetime.now()
            }
            
            trade_id = self.db.add_trade(trade_record)
            
            # Update local state
            trade_record['id'] = trade_id
            self.active_positions[symbol] = trade_record
            print(f"✅ Trade Executed & Saved: ID {trade_id} | Size: ${final_size:.0f}")

            # Send Telegram Alert
            if self.notifier:
                arrow = "🟢" if direction == "LONG" else "🔴"
                msg = (
                    f"{arrow} <b>New Trade Executed: {symbol}</b>\n"
                    f"Side: <b>{direction}</b>\n"
                    f"Entry: {current_price}\n"
                    f"Size: ${final_size:.0f} (Lev x{self.config.exchange.leverage})\n"
                    f"SL: {sl_price:.4f}\n"
                    f"TP: {tp_price:.4f}\n"
                    f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                self.notifier.send_message(msg)
            
        except Exception as e:
            print(f"❌ Execution Failed: {e}")

    def sync_positions(self):
        """
        Reconcile Bot's internal state with Exchange's real positions.
        If a position is missing on Exchange but Open in Bot -> Close in Bot.
        """
        try:
            # 1. Get Real Positions
            real_positions = self.executor.get_open_positions()
            real_symbols = {p['symbol']: p for p in real_positions}
            
            # 2. Check Bot's Active Positions
            # Create a copy of keys to avoid runtime error during deletion
            bot_symbols = list(self.active_positions.keys())
            
            for symbol in bot_symbols:
                trade = self.active_positions[symbol]
                
                # If Bot thinks it's open, but Exchange doesn't have it
                if symbol not in real_symbols:
                    print(f"⚠️ Mismatch: {symbol} found in Bot but NOT on Exchange. Closing in DB...")
                    
                    # Mark as Closed in DB
                    self.db.update_trade(trade['id'], {
                        'status': 'CLOSED',
                        'exit_time': datetime.now(),
                        'exit_reason': 'EXTERNAL_CLOSE', # Manually closed or Liquidated
                        'pnl': 0.0, # Unknown PnL if closed externally, or fetch from history if possible
                        'raw_data': {'note': 'Closed externally detected by Sync'}
                    })
                    
                    # Remove from Memory
                    del self.active_positions[symbol]
                    
                    # Notify
                    if self.notifier:
                        self.notifier.send_message(f"⚠️ <b>Position Sync:</b> {symbol} was closed externally.")
                        
                else:
                    # Optional: Update PnL or Size if changed partial close?
                    # For now just existence check is enough for safety.
                    pass
                    
            # 3. (Optional) Reverse Sync: If Exchange has position but Bot doesn't?
            # We skip this for now to avoid importing random trades.
            
        except Exception as e:
            print(f"❌ Sync Error: {e}")
