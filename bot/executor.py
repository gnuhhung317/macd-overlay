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
        print(f"⚠️ [DRY RUN] Placing {side} {symbol} | Size: {size} | Lev: {leverage}x | SL: {sl_price} | TP: {tp_price}")
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

class BitgetExecutor(ExchangeExecutor):
    def __init__(self, config: BotConfig):
        self.config = config
        # Here we would initialize the real Bitget/CCXT client using api_key/secret
        print("[Executor] Initialized Bitget Executor (Real Trading)")

    def get_balance(self) -> float:
        # Placeholder for real API call
        return 0.0

    def place_order(self, symbol: str, side: str, size: float, leverage: int, sl_price: float, tp_price: float, trailing_callback: float = 0.0, activation_price: float = 0.0) -> Dict[str, Any]:
        # Placeholder
        raise NotImplementedError("Real Bitget execution not yet fully implemented")

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        return False
        
    def close_position(self, symbol: str) -> bool:
        return False

    def get_open_positions(self) -> list:
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

    def _get_price_precision(self, symbol: str) -> int:
        """Get price precision for symbol"""
        if symbol in self.symbol_info:
            return self.symbol_info[symbol]['pricePrecision']
        return 2 # Default fallback

    def get_balance(self) -> float:
        """Get USDT Balance"""
        try:
            balances = self.client.futures_account_balance()
            for b in balances:
                if b['asset'] == 'USDT':
                    return float(b['availableBalance'])
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
            price_prec = self._get_price_precision(symbol)
            
            # SL
            sl_side = SIDE_SELL if side.upper() == 'LONG' else SIDE_BUY
            self.client.futures_create_order(
                symbol=symbol,
                side=sl_side,
                type='STOP_MARKET',
                stopPrice=round(sl_price, price_prec),
                closePosition=True
            )
            
            # TP
            self.client.futures_create_order(
                symbol=symbol,
                side=sl_side,
                type='TAKE_PROFIT_MARKET',
                stopPrice=round(tp_price, price_prec),
                closePosition=True
            )
            
            print(f"✅ Order & Standard SL/TP Placed for {symbol}")
            
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
                    ts_kwargs['activationPrice'] = round(activation_price, price_prec)
                    
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
                    
                    active_positions.append({
                        "symbol": p['symbol'],
                        "size": amt,
                        "entry_price": entry_price,
                        "mark_price": mark_price,
                        "pnl": unrealized_pnl,
                        "leverage": leverage,
                        "side": "LONG" if amt > 0 else "SHORT"
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
        return BitgetExecutor(config)
