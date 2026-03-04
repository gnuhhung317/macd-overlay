from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
from .config import BotConfig

class ExchangeExecutor(ABC):
    @abstractmethod
    def get_balance(self) -> float:
        """Get available USDT balance"""
        pass

    @abstractmethod
    def place_order(self, symbol: str, side: str, size: float, leverage: int, sl_price: float, tp_price: float, trailing_callback: float = 0.0, activation_price: float = 0.0) -> Dict[str, Any]:
        """Place market order with TP/SL and optional Trailing Stop"""
        pass

    @abstractmethod
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order"""
        pass
        
    @abstractmethod
    def close_position(self, symbol: str) -> bool:
        """Close position immediately"""
        pass

    @abstractmethod
    def get_open_positions(self) -> list:
        """Get all open positions from exchange"""
        pass

class DryRunExecutor(ExchangeExecutor):
    def __init__(self, config: BotConfig):
        self.config = config
        self.balance = 10000.0 # Virtual $10k
        print("[Executor] Initialized in DRY RUN mode")

    def get_balance(self) -> float:
        return self.balance

    def place_order(self, symbol: str, side: str, size: float, leverage: int, sl_price: float, tp_price: float, trailing_callback: float = 0.0, activation_price: float = 0.0) -> Dict[str, Any]:
        display_tp = "None (Trailing Stop)" if trailing_callback > 0 else tp_price
        print(f"⚠️ [DRY RUN] Placing {side} {symbol} | Size: {size} | Lev: {leverage}x | SL: {sl_price} | TP: {display_tp}")
        if trailing_callback > 0:
            act_str = f" | Act: {activation_price}" if activation_price > 0 else ""
            print(f"⚠️ [DRY RUN] Placing TRAILING STOP for {symbol} | Callback: {trailing_callback}%{act_str}")
        return {
            "order_id": f"dry_run_{symbol}_{side}",
            "status": "filled",
            "filled_price": 0.0, # Would be filled by caller with real price
            "timestamp": "now"
        }

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        print(f"⚠️ [DRY RUN] Cancelling order {order_id} for {symbol}")
        return True

    def close_position(self, symbol: str) -> bool:
        print(f"⚠️ [DRY RUN] Closing position for {symbol}")
        return True

    def update_balance(self, amount: float):
        """Update virtual balance (e.g. adding PnL)"""
        self.balance += amount
        print(f"💰 [DRY RUN] Balance updated: ${self.balance:,.2f} ({'+' if amount >= 0 else ''}{amount:,.2f})")

    def get_open_positions(self) -> list:
        return []

import ccxt

class CCXTExecutor(ExchangeExecutor):
    def __init__(self, config: BotConfig):
        self.config = config
        exchange_id = config.exchange.name.lower()
        exchange_class = getattr(ccxt, exchange_id)
        
        exchange_args = {
            'apiKey': config.exchange.api_key,
            'secret': config.exchange.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap', # assume futures
                'positionMode': False  # Force One-Way / Unilateral mode
            }
        }
        if config.exchange.password:
            exchange_args['password'] = config.exchange.password
        elif hasattr(config.exchange, 'passphrase') and config.exchange.passphrase:
            exchange_args['password'] = config.exchange.passphrase
            
        self.client = exchange_class(exchange_args)
        
        # Load markets for precision and symbol details
        try:
            self.client.load_markets()
            print(f"[Executor] Initialized CCXT Executor for {exchange_id.upper()} (Real Trading)")
        except Exception as e:
            print(f"⚠️ Error loading markets for {exchange_id}: {e}")

    def _get_ccxt_symbol(self, symbol: str) -> str:
        # e.g. BTCUSDT -> BTC/USDT:USDT (futures) or BTC/USDT
        # Many exchanges require base/quote. Let's find it from loaded markets
        for s in self.client.symbols:
            if s.replace('/', '').replace(':', '') == symbol or s.replace('/', '').split(':')[0] == symbol:
                if self.client.markets[s]['swap']:
                    return s
        # fallback
        return symbol.replace("USDT", "/USDT:USDT")

    def get_balance(self) -> float:
        try:
            balance = self.client.fetch_balance()
            if 'USDT' in balance:
                return float(balance['USDT']['total']) # Use Total (Equity/Margin Balance) instead of Free
            return 0.0
        except Exception as e:
            print(f"❌ Error getting balance via CCXT: {e}")
            return 0.0

    def place_order(self, symbol: str, side: str, size: float, leverage: int, sl_price: float, tp_price: float, trailing_callback: float = 0.0, activation_price: float = 0.0) -> Dict[str, Any]:
        ccxt_symbol = self._get_ccxt_symbol(symbol)
        try:
            # 1. Set Leverage
            try:
                self.client.set_leverage(leverage, ccxt_symbol)
            except Exception as e:
                pass # might not be supported or already set
                
            # 2. Set Margin Mode
            try:
                margin_mode = self.config.exchange.margin_mode.lower() # isolated or cross
                self.client.set_margin_mode(margin_mode, ccxt_symbol)
            except Exception as e:
                pass

            # 3. Calculate Quantity
            ticker = self.client.fetch_ticker(ccxt_symbol)
            current_price = float(ticker['last'])
            
            quantity = size / current_price
            
            # Use CCXT precision formatting
            quantity = float(self.client.amount_to_precision(ccxt_symbol, quantity))
            if quantity <= 0:
                print(f"❌ Calculated quantity is 0 for {ccxt_symbol}")
                return {}

            ccxt_side = 'buy' if side.upper() == 'LONG' else 'sell'
            
            print(f"🚀 Placing {side} {ccxt_symbol}: Qty {quantity} @ Market via CCXT")
            
            # Set SL/TP params using unified CCXT structure for better exchange compatibility
            params = {
                'stopLoss': {
                    'triggerPrice': self.client.price_to_precision(ccxt_symbol, sl_price)
                },
                'takeProfit': {
                    'triggerPrice': self.client.price_to_precision(ccxt_symbol, tp_price)
                }
            }
            
            # Bitget Adaptive Logic: Handle both Unilateral and Hedge modes
            if self.config.exchange.name.lower() == 'bitget':
                try:
                    # Attempt 1: Assume Unilateral mode (omit tradeSide)
                    order = self.client.create_order(
                        symbol=ccxt_symbol,
                        type='market',
                        side=ccxt_side,
                        amount=quantity,
                        params=params
                    )
                except Exception as e:
                    # Error 40774 indicates a mode mismatch
                    if '40774' in str(e):
                        print(f"🔄 Bitget 40774 detected. Retrying with tradeSide='open' (Hedge Mode)...")
                        params['tradeSide'] = 'open'
                        # In Hedge mode, we also specify the position side
                        params['posSide'] = 'long' if ccxt_side == 'buy' else 'short'
                        
                        order = self.client.create_order(
                            symbol=ccxt_symbol,
                            type='market',
                            side=ccxt_side,
                            amount=quantity,
                            params=params
                        )
                    else:
                        raise e
            else:
                # Standard order placement for other exchanges
                order = self.client.create_order(
                    symbol=ccxt_symbol,
                    type='market',
                    side=ccxt_side,
                    amount=quantity,
                    params=params
                )
            
            print(f"✅ Order & Standard SL/TP Placed for {ccxt_symbol}")

            # Ensure we return a valid dict even if some fields are None
            avg_price = order.get('average')
            if avg_price is None:
                avg_price = ticker.get('last', 0.0)

            return {
                "order_id": order.get('id', 'unknown'),
                "status": order.get('status', 'open'),
                "filled_price": float(avg_price),
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            print(f"❌ CCXT Execution Error: {e}")
            return {}

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        ccxt_symbol = self._get_ccxt_symbol(symbol)
        try:
            self.client.cancel_order(order_id, ccxt_symbol)
            return True
        except:
            return False
            
    def close_position(self, symbol: str) -> bool:
        ccxt_symbol = self._get_ccxt_symbol(symbol)
        try:
            positions = self.client.fetch_positions([ccxt_symbol])
            for p in positions:
                if p['symbol'] == ccxt_symbol and float(p['contracts']) > 0:
                    side = 'sell' if p['side'] == 'long' else 'buy'
                    params = {'reduceOnly': True}
                    
                    if self.config.exchange.name.lower() == 'bitget':
                        try:
                            # Attempt to use ccxt's native close_position which handles Hedge / One-Way automatically
                            self.client.close_position(
                                symbol=ccxt_symbol,
                                side=p['side'] # 'long' or 'short'
                            )
                        except Exception as e:
                            print(f"❌ Fallback: Native close_position failed for {ccxt_symbol}: {e}. Trying market order...")
                            # Fallback to standard order placement if close_position raises an error
                            self.client.create_order(
                                symbol=ccxt_symbol,
                                type='market',
                                side=side,
                                amount=float(p['contracts']),
                                params=params
                            )
                    else:
                        # Standard order placement for other exchanges
                        self.client.create_order(
                            symbol=ccxt_symbol,
                            type='market',
                            side=side,
                            amount=float(p['contracts']),
                            params=params
                        )
                    print(f"⚠️ Closed position for {ccxt_symbol}")
                    return True
            return True
        except Exception as e:
            print(f"❌ Error closing position via CCXT: {e}")
            return False

    def get_open_positions(self) -> list:
        try:
            all_positions = self.client.fetch_positions()
            active_positions = []
            
            for p in all_positions:
                if float(p.get('contracts', 0)) > 0:
                    raw_symbol = p['symbol'].split(':')[0].replace('/', '')
                    
                    entry_time_ms = p.get('timestamp') or p.get('lastUpdateTimestamp')
                    entry_time = datetime.fromtimestamp(entry_time_ms / 1000.0) if entry_time_ms else datetime.now()
                    
                    active_positions.append({
                        "symbol": raw_symbol,
                        "size": float(p['contracts']),
                        "entry_price": float(p['entryPrice']),
                        "mark_price": float(p.get('markPrice', p['entryPrice'])),
                        "pnl": float(p.get('unrealizedPnl', 0)),
                        "leverage": int(p.get('leverage', self.config.exchange.leverage)),
                        "side": p['side'].upper(),
                        "entry_time": entry_time
                    })
            return active_positions
        except Exception as e:
            print(f"❌ Error fetching positions via CCXT: {e}")
            return []

try:
    from binance.client import Client
    from binance.enums import *
    from binance.exceptions import BinanceAPIException
except ImportError:
    print("⚠️ python-binance not installed. Real trading will fail.")
    Client = None

class BinanceExecutor(ExchangeExecutor):
    def __init__(self, config: BotConfig):
        self.config = config
        if not self.config.exchange.api_key or not self.config.exchange.api_secret:
             print("⚠️ API Key/Secret missing for Binance Executor!")
        
        self.client = Client(self.config.exchange.api_key, self.config.exchange.api_secret)
        print("[Executor] Initialized Binance Executor (Real Trading)")
        
        # Cache symbol info for precision
        self.symbol_info = {}
        try:
            info = self.client.futures_exchange_info()
            for s in info['symbols']:
                self.symbol_info[s['symbol']] = s
        except Exception as e:
            print(f"⚠️ Error fetching exchange info: {e}")

    def _get_precision(self, symbol: str) -> int:
        """Get quantity precision for symbol"""
        if symbol in self.symbol_info:
            return self.symbol_info[symbol]['quantityPrecision']
        return 3 # Default fallback

    def _get_tick_size(self, symbol: str) -> float:
        """Get minimum price movement (tickSize) for symbol"""
        if symbol in self.symbol_info:
            for f in self.symbol_info[symbol].get('filters', []):
                if f.get('filterType') == 'PRICE_FILTER':
                    return float(f.get('tickSize', 0.0))
        return 0.0

    def _get_price_precision(self, symbol: str) -> int:
        """Get price precision for symbol"""
        if symbol in self.symbol_info:
            return self.symbol_info[symbol]['pricePrecision']
        return 2 # Default fallback

    def format_price(self, symbol: str, price: float) -> float:
        """Format price to perfectly align with exchange tickSize"""
        tick_size = self._get_tick_size(symbol)
        precision = self._get_price_precision(symbol)
        
        if tick_size <= 0:
            return round(price, precision)
            
        ticks = round(price / tick_size)
        return round(ticks * tick_size, precision)

    def get_balance(self) -> float:
        """Get USDT Balance (Total Equity/Margin Balance)"""
        try:
            balances = self.client.futures_account_balance()
            for b in balances:
                if b['asset'] == 'USDT':
                    # Support multiple potential keys for better compatibility (Standard vs Portfolio)
                    # Priority: marginBalance (Equity) -> balance (Wallet) -> crossMarginBalance
                    val = b.get('marginBalance') or b.get('balance') or b.get('walletBalance') or b.get('totalMarginBalance')
                    if val is not None:
                        return float(val)
            return 0.0
        except Exception as e:
            print(f"❌ Error getting balance: {e}")
            return 0.0

    def place_order(self, symbol: str, side: str, size: float, leverage: int, sl_price: float, tp_price: float, trailing_callback: float = 0.0, activation_price: float = 0.0) -> Dict[str, Any]:
        """
        Place Market Order + SL/TP + Optional Trailing Stop
        size is in USDT
        """
        try:
            # 1. Set Leverage & Margin Type
            try:
                self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
            except: pass # Might already be set

            try:
                margin_type = self.config.exchange.margin_mode.upper() # ISOLATED or CROSS
                
                # Pseudo-ISOLATED: Override ISOLATED to CROSS on the exchange
                if margin_type == "ISOLATED" or margin_type == "CROSS":
                    margin_type = "CROSSED" # Binance API expects "CROSSED"
                    
                self.client.futures_change_margin_type(symbol=symbol, marginType=margin_type)
            except Exception as e:
                # Ignore "No need to change margin type" (err code -4046)
                if "-4046" not in str(e):
                    print(f"⚠️ Could not change margin type to {margin_type} for {symbol}: {e}")
            
            # 2. Convert USDT Size to Quantity
            price_tick = self.client.futures_symbol_ticker(symbol=symbol)
            current_price = float(price_tick['price'])
            
            quantity = size / current_price
            prec = self._get_precision(symbol)
            quantity = round(quantity, prec)
            
            if quantity <= 0:
                print(f"❌ Calculated quantity is 0 for {symbol} (Size: {size})")
                return {}

            binance_side = SIDE_BUY if side.upper() == 'LONG' else SIDE_SELL
            
            # 3. Market Open
            print(f"🚀 Placing {side} {symbol}: Qty {quantity} @ Market")
            order = self.client.futures_create_order(
                symbol=symbol,
                side=binance_side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            
            # 4. Place SL & TP (Reduce Only)
            
            # SL
            sl_side = SIDE_SELL if side.upper() == 'LONG' else SIDE_BUY
            self.client.futures_create_order(
                symbol=symbol,
                side=sl_side,
                type='STOP_MARKET',
                stopPrice=self.format_price(symbol, sl_price),
                closePosition=True
            )
            
            # TP
            if trailing_callback <= 0:
                self.client.futures_create_order(
                    symbol=symbol,
                    side=sl_side,
                    type='TAKE_PROFIT_MARKET',
                    stopPrice=self.format_price(symbol, tp_price),
                    closePosition=True
                )
                print(f"✅ Order & Standard SL/TP Placed for {symbol}")
            else:
                print(f"✅ Order & Standard SL Placed for {symbol} (TP skipped due to Trailing Stop)")
            
            # 5. Place Native Trailing Stop (Reduce Only) if configured
            if trailing_callback > 0:
                print(f"🚀 Placing TRAILING_STOP_MARKET for {symbol} (Callback {trailing_callback}%)")
                ts_kwargs = {
                    'symbol': symbol,
                    'side': sl_side,
                    'type': 'TRAILING_STOP_MARKET',
                    'quantity': quantity,
                    'callbackRate': trailing_callback,
                    'reduceOnly': True
                }
                if activation_price > 0:
                    ts_kwargs['activatePrice'] = self.format_price(symbol, activation_price)
                    
                ts_order = self.client.futures_create_order(**ts_kwargs)
                print(f"✅ Trailing Stop Placed (AlgoID: {ts_order.get('algoId', 'Unknown')})")

            return {
                "order_id": order['orderId'],
                "status": order['status'],
                "filled_price": float(order.get('avgPrice', current_price)),
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            print(f"❌ Execution Error: {e}")
            return {}

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        try:
            self.client.futures_cancel_order(symbol=symbol, orderId=order_id)
            return True
        except:
            return False
        
    def close_position(self, symbol: str) -> bool:
        """Close entire position"""
        try:
            # Check current position amount
            positions = self.client.futures_position_information(symbol=symbol)
            amt = 0.0
            for p in positions: # usually returns list [0] if specific symbol queried
                if p['symbol'] == symbol:
                    amt = float(p['positionAmt'])
            
            if amt == 0:
                return True
                
            side = SIDE_SELL if amt > 0 else SIDE_BUY
            
            # Market Close
            self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=abs(amt),
                reduceOnly=True
            )
            print(f"⚠️ Closed position for {symbol}")
            return True
            
        except Exception as e:
            print(f"❌ Error closing position: {e}")
            return False

    def get_open_positions(self) -> list:
        """Fetch all open positions from Binance Futures"""
        try:
            # Using futures_position_information is better than account which returns all symbols
            positions = self.client.futures_position_information(recvWindow=20000)
            active_positions = []
            
            for p in positions:
                amt = float(p['positionAmt'])
                if amt != 0:
                    # found an open position
                    entry_price = float(p['entryPrice'])
                    mark_price = float(p.get('markPrice', 0))
                    unrealized_pnl = float(p['unRealizedProfit'])
                    leverage = int(p.get('leverage', self.config.exchange.leverage))
                    
                    entry_time_ms = p.get('updateTime', 0)
                    entry_time = datetime.fromtimestamp(entry_time_ms / 1000.0) if entry_time_ms else datetime.now()
                    
                    active_positions.append({
                        "symbol": p['symbol'],
                        "size": amt,
                        "entry_price": entry_price,
                        "mark_price": mark_price,
                        "pnl": unrealized_pnl,
                        "leverage": leverage,
                        "side": "LONG" if amt > 0 else "SHORT",
                        "entry_time": entry_time
                    })
            return active_positions
        except Exception as e:
            print(f"❌ Error fetching positions: {e}")
            return []

def get_executor(config: BotConfig) -> ExchangeExecutor:
    if config.exchange.dry_run:
        return DryRunExecutor(config)
    elif config.exchange.name.lower() == 'binance':
        return BinanceExecutor(config)
    else:
        return CCXTExecutor(config)
