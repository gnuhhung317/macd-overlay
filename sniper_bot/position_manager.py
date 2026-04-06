import pandas as pd
from typing import Dict, List, Any
import time
import json
import logging
from datetime import datetime
from .config import SniperBotConfig
from bot.db import DatabaseManager
from bot.executor import ExchangeExecutor
from bot.data_provider import DataProvider

logger = logging.getLogger("PositionManager")

class PositionManager:
    def __init__(
        self, 
        config: SniperBotConfig, 
        db: DatabaseManager, 
        executor: ExchangeExecutor,
        data_provider: DataProvider,
        signal_engine: Any = None, # Unused in SniperBot
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
        """Main logic loop for a single symbol (Management only for SniperBot)"""
        
        # 1. Check if we already have a position
        current_trade = self.active_positions.get(symbol)
        
        # 2. Handle Active Position (Management)
        if current_trade:
            # We don't need to fetch candles here for SL/TP if we use get_current_price 
            # or if we only need the latest close.
            # However, _manage_active_position handles it internally.
            self._manage_active_position(current_trade, symbol, None)
            
        return

    def _manage_active_position(self, trade: Dict, symbol: str, df: pd.DataFrame = None):
        """Logic to manage an open trade (check SL/TP)"""
        current_price = 0.0
        
        # 1. Try to get Live Price first
        live_price = self.data_provider.get_current_price(symbol)
        if live_price > 0:
            current_price = live_price
        elif df is not None and not df.empty:
            current_price = df.iloc[-1]['close']
        else:
            return # Cannot manage without price
            
        direction = trade['direction']
        sl_price = trade['sl_price']
        tp_price = trade['tp_price']
        
        exit_reason = None
        exit_price = 0.0
        pnl = 0.0
        
        # Check SL/TP (Only if they are set > 0)
        if direction == 'LONG':
            if sl_price > 0 and current_price <= sl_price:
                exit_reason = 'SL_HIT'
                exit_price = sl_price
            elif tp_price > 0 and current_price >= tp_price:
                exit_reason = 'TP_HIT'
                exit_price = tp_price
        else: # SHORT
            if sl_price > 0 and current_price >= sl_price:
                exit_reason = 'SL_HIT'
                exit_price = sl_price
            elif tp_price > 0 and current_price <= tp_price:
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
            exit_time = datetime.now()
            pnl = 0.0
            pnl_pct = 0.0
            
            if self.config.exchange.dry_run:
                # Calculate PnL locally for simulation
                size = trade['size']
                entry_price = trade['entry_price']
                
                # Apply slippage
                slippage = self.config.exchange.slippage if hasattr(self.config.exchange, 'slippage') else 0.001
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
                
                pnl = size * pnl_pct
                
                # Update executor balance
                if hasattr(self.executor, 'update_balance'):
                    self.executor.update_balance(pnl)
                
                print(f"✅ [DRY RUN] Trade Closed: {symbol} | PnL: ${pnl:.2f}")
            else:
                # LIVE / TESTNET CLOSE
                print(f"📡 Sending Close Order for {symbol} ({exit_reason})...")
                success = self.executor.close_position(symbol)
                if not success:
                    print(f"❌ Failed to close position for {symbol} on exchange!")
                    return # Don't update DB yet if it failed
                
                # For Live Trades, PnL is often updated via WS/Sync later, 
                # but we can try to estimate or just log the close.
                # Real PnL will be reconciled by Sync.
                exit_price = current_price
                print(f"✅ [LIVE] Position Close Order Sent for {symbol}")

            # Update DB
            self.db.update_trade(trade['id'], {
                'status': 'CLOSED',
                'exit_price': exit_price,
                'exit_time': exit_time,
                'exit_reason': exit_reason,
                'pnl': pnl
            })
            
            # Clean up local state
            if symbol in self.active_positions:
                del self.active_positions[symbol]

            # Send Telegram Alert
            if self.notifier:
                emoji = "🤑" if pnl > 0 or exit_reason == 'TP_HIT' else "🛑"
                pnl_str = f"PnL: <b>${pnl:.2f} ({pnl_pct*100:.2f}%)</b>\n" if self.config.exchange.dry_run else ""
                msg = (
                    f"{emoji} <b>Trade Closed: {symbol}</b>\n"
                    f"Type: {exit_reason}\n"
                    f"{pnl_str}"
                    f"Exit Price: {exit_price}\n"
                    f"Time: {exit_time.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                self.notifier.send_message(msg)

    def _check_limit_order_timeouts(self):
        """Cancel limit orders that haven't filled after limit_wait_bars"""
        limit_wait_bars = getattr(self.config.strategy, 'limit_wait_bars', 5)
        # Assuming 1 bar = timeframe duration
        tf = self.config.strategy.timeframes[0] if self.config.strategy.timeframes else '1h'
        minutes_per_bar = 60 # Default 1h
        if tf.endswith('m'): minutes_per_bar = int(tf[:-1])
        elif tf.endswith('h'): minutes_per_bar = int(tf[:-1]) * 60
        elif tf.endswith('d'): minutes_per_bar = int(tf[:-1]) * 1440
        
        timeout_seconds = limit_wait_bars * minutes_per_bar * 60
        
        now = datetime.now()
        bot_symbols = list(self.active_positions.keys())
        
        for symbol in bot_symbols:
            trade = self.active_positions[symbol]
            # Check if it's a pending LIMIT order (size might be set but not yet 'filled' if we tracked it better)
            # In current logic, we just check entry_time + status
            if trade['status'] == 'OPEN':
                # Check exchange to see if it's actually a position or still an order
                pass # Sync handle this mostly, but we can proactively cancel here

            else:
                # Real execution - allow exchange to handle SL/TP or market close here
                # For now we assume if SL/TP orders were placed, we just track.
                # But if we need to force close:
                self.executor.close_position(symbol)
                # We would wait for WS update to partial/full fill to update DB
                pass

    def _calculate_position_size(self, total_capital: float, available_capital: float, sl_pct: float, confidence: float) -> float:
        """
        Calculate position size using Fixed Risk (Total Equity) 
        and cap it by Available Buying Power (Available Margin).
        """
        # 1. Fixed Risk Sizing based on Total Equity (Consistensy)
        risk_amount = total_capital * self.config.risk.max_risk_per_trade
        
        # 2. Calculate Size from Risk & SL (Total Nominal Value)
        # Position Size = (Capital * Risk%) / SL%
        if sl_pct <= 0: sl_pct = 0.02
        position_size = risk_amount / sl_pct
        
        # 3. Cap by Leveraged Buying Power (Available Margin * Leverage)
        # We use available_capital (Free Balance) here to prevent "exceeds balance" errors.
        # 0.9 safety buffer for fees/slippage.
        max_buying_power = available_capital * self.config.exchange.leverage * 0.90
        
        if position_size > max_buying_power:
            print(f"⚠️ Sizing Cap: Reducing {position_size:.0f} to available buying power {max_buying_power:.0f}")
            position_size = max_buying_power
            
        # Hard Cap on Position Size (USD) from Config
        if hasattr(self.config.risk, 'max_position_size_usd'):
            position_size = min(position_size, self.config.risk.max_position_size_usd)
            
        return max(0, position_size)


    def execute_calculated_signal(self, signal_data: Dict[str, Any], timeframe: str):
        """
        Execute a signal that was already calculated by the SmartScanner.
        Logs to DB first, then checks if actionable.
        """
        print(f"DEBUG: execute_calculated_signal for {signal_data['symbol']} at {signal_data['timestamp']}")
        symbol = signal_data['symbol']
        timestamp = signal_data['timestamp']
        logger.info(
            "Signal received: symbol=%s tf=%s ts=%s side=%s conf=%.4f",
            symbol,
            timeframe,
            timestamp,
            signal_data.get('type'),
            float(signal_data.get('confidence', 0.0)),
        )
        
        # 0. Fresh Crossover check bypassed for localized SniperBot (handled by scanner logic)

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
                    'limit_price': signal_data.get('limit_price', 0.0),
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
            logger.info("Skip entry: already active position for %s", symbol)
            return

        # 3. Filter Entry Zone is bypassed for SniperBot (handled in scanner)
        # 4. Refined Score Filter is bypassed for SniperBot (handled in scanner)
        
        # Convert Scanner format to internal Analysis format
        analysis = {
            "action": "ENTRY",
            "signal": "BULLISH" if signal_data['type'] == 'LONG' else "BEARISH",
            "confidence": signal_data['confidence'],
            "sl": signal_data['sl_pct'],
            "tp": signal_data['tp_pct'],
            "limit_price": signal_data.get('limit_price', 0.0),
            "risk_reward": signal_data.get('risk_reward', 0),
            "metadata": signal_data.get('meta', {})
        }
        
        print(f"🤖 SmartScanner Entry for {symbol} ({signal_data['status']}) | Conf: {analysis['confidence']:.2%}")
        logger.info(
            "Entry candidate accepted: symbol=%s status=%s rr=%.3f",
            symbol,
            signal_data.get('status', 'NA'),
            float(analysis.get('risk_reward', 0.0)),
        )
        self._execute_entry(symbol, analysis)

    def _execute_entry(self, symbol: str, analysis: Dict):
        """Execute the entry order"""
        print(f"🚀 ENTRY SIGNAL FOUND: {symbol} | Conf: {analysis['confidence']:.2%} | RR: {analysis.get('risk_reward', 0):.2f}")
        
        # 1. Calc Sizing
        total_balance = self.executor.get_balance()
        available_balance = self.executor.get_available_balance()
        
        if available_balance <= 0:
            print(f"❌ Insufficient Available Balance: {available_balance}")
            logger.warning("Entry rejected %s: insufficient available balance %.4f", symbol, float(available_balance))
            return

        # Pseudo-ISOLATED: Cap SL at 0.99/leverage so we simulate isolated liquidation
        if self.config.exchange.margin_mode.upper() == "ISOLATED":
            pseudo_iso_sl = 0.99 / self.config.exchange.leverage
            atr_sl_pct = analysis.get('sl', pseudo_iso_sl)
            
            if atr_sl_pct <= 0:
                atr_sl_pct = pseudo_iso_sl # Fallback
                
            analysis['sl'] = min(atr_sl_pct, pseudo_iso_sl)
            
            if analysis['sl'] == pseudo_iso_sl:
                print(f"🛡️ Pseudo-ISOLATED Mode: Capping ATR SL at liquidation risk: {pseudo_iso_sl:.2%} (1/leverage)")
            else:
                print(f"🛡️ Pseudo-ISOLATED Mode: Using ATR SL {atr_sl_pct:.2%} (Safer than {pseudo_iso_sl:.2%})")

        sl_pct = analysis['sl'] if analysis['sl'] > 0 else 0.01
        
        final_size = self._calculate_position_size(total_balance, available_balance, sl_pct, analysis['confidence'])
        
        if final_size <= 0:
            print("❌ Calculated size is 0")
            logger.warning("Entry rejected %s: calculated position size is 0", symbol)
            return
        
        # Get Current Price for SL/TP absolute values
        current_price = self.data_provider.get_current_price(symbol)
        if current_price <= 0:
            print("❌ Could not get current price")
            logger.warning("Entry rejected %s: could not fetch current price", symbol)
            return

        direction = "LONG" if analysis['signal'] == "BULLISH" else "SHORT"
        
        # Use SL/TP calculated from ATR in scanner
        limit_price = analysis.get('limit_price', 0)
        base_price = limit_price if limit_price > 0 else current_price
        
        if direction == "LONG":
            sl_price = base_price * (1 - analysis['sl'])
            tp_price = base_price * (1 + analysis['tp'])
        else:
            sl_price = base_price * (1 + analysis['sl'])
            tp_price = base_price * (1 - analysis['tp'])

        # --- Pre-entry price checks ---
        # If TP already reached -> skip
        # If SL already reached -> allow optional rebase when within threshold
        rebase_threshold = getattr(self.config.strategy, 'rebase_threshold', 0.005)  # default 0.5%
        latest_price = self.data_provider.get_current_price(symbol)
        if latest_price <= 0:
            print("❌ Could not get latest price for pre-entry check")
            logger.warning("Entry rejected %s: could not fetch latest price for pre-entry check", symbol)
            return

        mark_price = latest_price

        if direction == 'LONG':
            # TP hit -> skip
            if mark_price >= tp_price:
                print(f"⚠️ {symbol} mark {mark_price} >= TP {tp_price}. Skipping entry.")
                logger.warning("Entry skipped %s: TP already reached (mark>=tp)", symbol)
                return

            # SL hit -> consider rebase
            if mark_price <= sl_price:
                deviation = abs(mark_price - base_price) / base_price if base_price > 0 else 1.0
                if deviation <= rebase_threshold:
                    print(f"🔁 {symbol} hit SL but within rebase {deviation:.2%}. Rebasing entry to mark price.")
                    base_price = mark_price
                    sl_price = base_price * (1 - analysis['sl'])
                    tp_price = base_price * (1 + analysis['tp'])
                    # convert limit entry to market to get immediate fill
                    if limit_price > 0:
                        print("🔁 Converting LIMIT entry to MARKET due to rebase.")
                        limit_price = 0
                else:
                    print(f"⚠️ {symbol} hit SL and deviation {deviation:.2%} > rebase threshold {rebase_threshold:.2%}. Skipping entry.")
                    return
        else:  # SHORT
            if mark_price <= tp_price:
                print(f"⚠️ {symbol} mark {mark_price} <= TP {tp_price}. Skipping entry.")
                logger.warning("Entry skipped %s: TP already reached (mark<=tp)", symbol)
                return

            if mark_price >= sl_price:
                deviation = abs(mark_price - base_price) / base_price if base_price > 0 else 1.0
                if deviation <= rebase_threshold:
                    print(f"🔁 {symbol} hit SL but within rebase {deviation:.2%}. Rebasing entry to mark price.")
                    base_price = mark_price
                    sl_price = base_price * (1 + analysis['sl'])
                    tp_price = base_price * (1 - analysis['tp'])
                    if limit_price > 0:
                        print("🔁 Converting LIMIT entry to MARKET due to rebase.")
                        limit_price = 0
                else:
                    print(f"⚠️ {symbol} hit SL and deviation {deviation:.2%} > rebase threshold {rebase_threshold:.2%}. Skipping entry.")
                    return

        # 2. Execute Order
        try:
            # Trailing stop removed to fix StrategyConfig attribute error
            order_result = self.executor.place_order(
                symbol=symbol,
                side=direction.lower(),
                size=final_size,
                leverage=self.config.exchange.leverage,
                sl_price=sl_price,
                tp_price=tp_price,
                order_type='LIMIT' if limit_price > 0 else 'MARKET',
                price=limit_price
            )
            
            if not order_result or 'order_id' not in order_result:
                print(f"❌ Order placement failed for {symbol}")
                logger.warning("Entry rejected %s: executor returned no order_id", symbol)
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
            logger.info(
                "Trade opened: id=%s symbol=%s side=%s size=%.2f entry=%.6f sl=%.6f tp=%.6f",
                trade_id,
                symbol,
                direction,
                float(final_size),
                float(current_price),
                float(sl_price),
                float(tp_price),
            )

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
            logger.exception("Execution failed for %s", symbol)

    def sync_positions(self):
        """
        Reconcile Bot's internal state with Exchange's real positions and orders.
        Prevents purging pending LIMIT orders.
        """
        if self.config.exchange.dry_run:
            # Local paper mode has no exchange-side state. Keep positions managed locally.
            return

        try:
            # 1. Get Real State
            real_positions = self.executor.get_open_positions()
            real_orders = self.executor.get_open_orders()
            
            real_symbols = {p['symbol']: p for p in real_positions}
            order_symbols = {o['symbol'] for o in real_orders}
            
            # Get timeout settings
            limit_wait_bars = getattr(self.config.strategy, 'limit_wait_bars', 5)
            tf = self.config.strategy.timeframes[0] if self.config.strategy.timeframes else '1h'
            
            def get_minutes(tf_str):
                if tf_str.endswith('m'): return int(tf_str[:-1])
                if tf_str.endswith('h'): return int(tf_str[:-1]) * 60
                if tf_str.endswith('d'): return int(tf_str[:-1]) * 1440
                if tf_str.endswith('w'): return int(tf_str[:-1]) * 10080
                return 60
            
            minutes_per_bar = get_minutes(tf)
            limit_timeout_seconds = limit_wait_bars * minutes_per_bar * 60

            # 2. Check Bot's Active Positions
            bot_symbols = list(self.active_positions.keys())
            
            for symbol in bot_symbols:
                trade = self.active_positions[symbol]
                
                # Case A: Found in Positions -> CHECK TIMEOUT
                if symbol in real_symbols:
                    real_p = real_symbols[symbol]
                    # Check timeout for existing position
                    entry_time = trade.get('entry_time')
                    if not entry_time or not isinstance(entry_time, (datetime, pd.Timestamp)):
                        # Default to exchange timestamp if internal missing
                        entry_time = real_p.get('entry_time', datetime.now())
                        if isinstance(entry_time, str): entry_time = pd.Timestamp(entry_time)
                    
                    waited_seconds = (datetime.now() - entry_time).total_seconds()
                    
                    # Calculate timeout in seconds
                    timeout_candles = getattr(self.config.strategy, 'timeout_candles', 48)
                    timeout_seconds = timeout_candles * minutes_per_bar * 60
                    
                    if waited_seconds > timeout_seconds:
                        print(f"⌛ Position Timeout for {symbol} ({waited_seconds/3600:.1f}h > {timeout_seconds/3600:.1f}h). Closing...")
                        if not self.config.exchange.dry_run:
                            self.executor.close_position(symbol)
                        
                        self.db.update_trade(trade['id'], {
                            'status': 'CLOSED',
                            'exit_time': datetime.now(),
                            'exit_reason': 'TIMEOUT',
                            'pnl': real_p.get('pnl', 0.0)
                        })
                        del self.active_positions[symbol]
                        if self.notifier:
                            self.notifier.send_message(f"⌛ <b>Position Timeout:</b> {symbol} closed after {waited_seconds/3600:.1f} hours.")
                    continue
                
                # Case B: Found in Orders -> Check Timeout
                matches = [o for o in real_orders if o['symbol'] == symbol]
                if matches:
                    # Use the creation timestamp provided by the exchange
                    order_timestamp = matches[0]['timestamp']
                    if isinstance(order_timestamp, str):
                        order_timestamp = pd.Timestamp(order_timestamp)
                    
                    waited_seconds = (datetime.now() - order_timestamp).total_seconds()
                    if waited_seconds > limit_timeout_seconds:
                        print(f"⌛ Limit Timeout for {symbol} ({waited_seconds/60:.1f}m > {limit_timeout_seconds/60:.1f}m). Cancelling...")
                        # 1. Cancel on exchange
                        if not self.config.exchange.dry_run:
                            raw = trade.get('raw_data', {})
                            if isinstance(raw, str):
                                try: raw = json.loads(raw)
                                except: raw = {}
                                
                            order_id = raw.get('order_id') or raw.get('id')
                            if order_id:
                                self.executor.cancel_order(symbol, order_id)
                            else:
                                # Fallback: cancel all for symbol
                                self.executor.cancel_order(symbol, "all")
                        
                        # 2. Close in DB
                        self.db.update_trade(trade['id'], {
                            'status': 'CLOSED',
                            'exit_time': datetime.now(),
                            'exit_reason': 'LIMIT_TIMEOUT',
                            'pnl': 0.0
                        })
                        del self.active_positions[symbol]
                    continue

                # Case C: Not in Positions and Not in Orders -> Closed externally
                print(f"⚠️ Mismatch: {symbol} found in Bot but NOT on Exchange (Pos/Order). Closing in DB...")
                self.db.update_trade(trade['id'], {
                    'status': 'CLOSED',
                    'exit_time': datetime.now(),
                    'exit_reason': 'EXTERNAL_CLOSE',
                    'pnl': 0.0
                })
                del self.active_positions[symbol]
                if self.notifier:
                    self.notifier.send_message(f"⚠️ <b>Position Sync:</b> {symbol} was closed externally or filled/cancelled.")
                    
            # 3. Reverse Sync (Import missing positions)
            for symbol, real_p in real_symbols.items():
                if symbol not in bot_symbols:
                    print(f"🔄 Reverse Sync: Found {symbol} on exchange but not in local DB. Importing...")
                    trade_record = {
                        "symbol": symbol,
                        "direction": real_p['side'],
                        "status": "OPEN",
                        "entry_price": real_p['entry_price'],
                        "sl_price": real_p.get('sl_price', 0.0),
                        "tp_price": real_p.get('tp_price', 0.0),
                        "size": real_p['size'],
                        "leverage": real_p.get('leverage', self.config.exchange.leverage),
                        "raw_data": {"note": "Imported via Sync"},
                        "entry_time": real_p.get('entry_time', datetime.now())
                    }
                    trade_id = self.db.add_trade(trade_record)
                    trade_record['id'] = trade_id
                    self.active_positions[symbol] = trade_record
        except Exception as e:
            print(f"❌ Sync Error: {e}")
