
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from sniper_bot.position_manager import PositionManager
from sniper_bot.config import SniperBotConfig

class TestPositionTimeout(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock(spec=SniperBotConfig)
        self.config.exchange = MagicMock()
        self.config.exchange.dry_run = False
        self.config.exchange.leverage = 20
        self.config.strategy = MagicMock()
        self.config.strategy.timeframes = ['1h']
        self.config.strategy.timeout_candles = 2 # 2 hours timeout for testing
        
        self.db = MagicMock()
        self.executor = MagicMock()
        self.data_provider = MagicMock()
        self.notifier = MagicMock()
        
        # Mock DB get_active_trades to return nothing initially
        self.db.get_active_trades.return_value = []
        
        self.pm = PositionManager(
            self.config, self.db, self.executor, self.data_provider, notifier=self.notifier
        )

    def test_sync_positions_timeout(self):
        # 1. Setup an active position in PM that is 3 hours old
        symbol = "BTCUSDT"
        entry_time = datetime.now() - timedelta(hours=3)
        trade = {
            'id': 1,
            'symbol': symbol,
            'direction': 'LONG',
            'status': 'OPEN',
            'entry_time': entry_time,
            'size': 1000
        }
        self.pm.active_positions = {symbol: trade}
        
        # 2. Mock executor.get_open_positions to return this position
        self.executor.get_open_positions.return_value = [{
            'symbol': symbol,
            'side': 'LONG',
            'size': 1000,
            'entry_time': entry_time,
            'pnl': 10.0
        }]
        self.executor.get_open_orders.return_value = []
        
        # 3. Call sync_positions
        self.pm.sync_positions()
        
        # 4. Verify close_position was called
        self.executor.close_position.assert_called_with(symbol)
        
        # 5. Verify DB was updated
        self.db.update_trade.assert_called()
        args, kwargs = self.db.update_trade.call_args
        self.assertEqual(args[0], 1)
        self.assertEqual(args[1]['status'], 'CLOSED')
        self.assertEqual(args[1]['exit_reason'], 'TIMEOUT')
        
        # 6. Verify local state was cleaned
        self.assertNotIn(symbol, self.pm.active_positions)

    def test_manage_active_position_live_close(self):
        # 1. Setup an active position
        symbol = "ETHUSDT"
        entry_time = datetime.now() - timedelta(hours=3) # Older than 2h timeout
        trade = {
            'id': 2,
            'symbol': symbol,
            'direction': 'LONG',
            'status': 'OPEN',
            'entry_time': entry_time,
            'size': 500,
            'sl_price': 1000,
            'tp_price': 3000,
            'entry_price': 2000
        }
        self.pm.active_positions = {symbol: trade}
        
        # 2. Mock data_provider to return a price
        self.data_provider.get_current_price.return_value = 2100
        
        # 3. Call _manage_active_position
        self.pm._manage_active_position(trade, symbol)
        
        # 4. Verify close_position was called due to timeout
        self.executor.close_position.assert_called_with(symbol)
        
        # 5. Verify DB was updated
        self.db.update_trade.assert_called()
        self.assertNotIn(symbol, self.pm.active_positions)

if __name__ == '__main__':
    unittest.main()
