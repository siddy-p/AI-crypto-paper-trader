import unittest
import os
import shutil
from unittest.mock import MagicMock, patch
from src import config
from src import database
from src.trader import Trader

class TestTrader(unittest.TestCase):
    def setUp(self):
        # Set up a temporary database path for tests
        self.temp_db_path = "test_trading_system.db"
        config.DB_PATH = self.temp_db_path
        database.init_db()
        
        # Mock dependencies
        self.mock_client = MagicMock()
        self.mock_model_manager = MagicMock()
        
        # Configure Trader with order execution disabled (default safe simulation)
        config.DISABLE_ORDER_EXECUTION = True
        config.INITIAL_USDT_BALANCE = 10000.0
        config.POSITION_SIZE_USDT   = 100.0
        config.STOP_LOSS_PCT        = 0.015
        config.TAKE_PROFIT_PCT      = 0.030
        config.CONFIDENCE_THRESHOLD = 60.0
        config.MAX_OPEN_POSITIONS   = 5
        
        self.trader = Trader(self.mock_client, self.mock_model_manager)
        self.trader.symbols = ["BTCUSDT", "ETHUSDT"]
        self.trader.latest_prices = {"BTCUSDT": 60000.0, "ETHUSDT": 30000.0}
        
        # Clear database records
        conn = database.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM trades")
        cursor.execute("DELETE FROM portfolio_history")
        conn.commit()
        conn.close()
        
        # Write initial portfolio history entry
        database.log_portfolio(10000.0, 10000.0)
        self.trader.sync_portfolio_balance()

    def tearDown(self):
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)

    def test_position_sizing_risk(self):
        # Starting portfolio value: $10,000
        # Position size: Fixed $100
        
        portfolio_val = self.trader.get_portfolio_value()
        available_usdt = self.trader.get_available_usdt()
        
        self.assertEqual(portfolio_val, 10000.0)
        self.assertEqual(available_usdt, 10000.0)
        
        # Simulate open position execution
        self.trader.open_long_position("BTCUSDT", 60000.0, 85.0)
        
        open_trades = database.get_open_trades()
        self.assertEqual(len(open_trades), 1)
        
        # Available USDT should be: 10000 - 100 = 9900
        self.assertAlmostEqual(self.trader.simulated_usdt, 9900.0)
        
        # Quantity bought: 100 / 60000
        self.assertAlmostEqual(open_trades[0]["quantity"], 100.0 / 60000.0)

    def test_stop_loss_hit(self):
        # Open simulated position at 100.0. SL is 1% down = 99.0
        database.add_trade("ETHUSDT", "BUY", 100.0, 10.0, 80.0, 99.0, 102.0)
        
        # Check exits with price 99.5 (SL not hit)
        self.trader.check_positions_exits("ETHUSDT", 99.5)
        open_trades = database.get_open_trades()
        self.assertEqual(len(open_trades), 1)
        
        # Check exits with price 98.9 (SL hit!)
        self.trader.check_positions_exits("ETHUSDT", 98.9)
        open_trades = database.get_open_trades()
        self.assertEqual(len(open_trades), 0)
        
        closed_trades = database.get_trade_history()
        self.assertEqual(len(closed_trades), 1)
        self.assertEqual(closed_trades[0]["exit_reason"], "STOP_LOSS")
        self.assertEqual(closed_trades[0]["exit_price"], 98.9)
        # P&L: (98.9 - 100.0) * 10 = -11.0
        self.assertAlmostEqual(closed_trades[0]["pnl"], -11.0)

    def test_take_profit_hit(self):
        # Open simulated position at 100.0. TP is 2% up = 102.0
        database.add_trade("ETHUSDT", "BUY", 100.0, 10.0, 80.0, 99.0, 102.0)
        
        # Check exits with price 102.1 (TP hit!)
        self.trader.check_positions_exits("ETHUSDT", 102.1)
        open_trades = database.get_open_trades()
        self.assertEqual(len(open_trades), 0)
        
        closed_trades = database.get_trade_history()
        self.assertEqual(len(closed_trades), 1)
        self.assertEqual(closed_trades[0]["exit_reason"], "TAKE_PROFIT")
        self.assertEqual(closed_trades[0]["exit_price"], 102.1)
        # P&L: (102.1 - 100.0) * 10 = +21.0
        self.assertAlmostEqual(closed_trades[0]["pnl"], 21.0)

    @patch('src.database.get_open_trades')
    def test_safety_execution_flag(self, mock_get_open):
        mock_get_open.return_value = []
        config.DISABLE_ORDER_EXECUTION = False
        config.BINANCE_API_KEY = "test_key"
        config.BINANCE_API_SECRET = "test_secret"
        self.mock_client.get_account_balance.return_value = {"USDT": 10000.0}
        self.mock_client.place_market_buy.return_value = {"fills": [{"price": "60000.0", "qty": "0.16"}]}
        
        # Run live buy path
        self.trader.open_long_position("BTCUSDT", 60000.0, 90.0)
        
        # Verify that client API method was called
        self.mock_client.place_market_buy.assert_called_once()

if __name__ == '__main__':
    unittest.main()
