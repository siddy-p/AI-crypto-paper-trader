import unittest
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src import config
from src.model import CryptoModelManager

class TestModel(unittest.TestCase):
    def setUp(self):
        self.temp_model_path = "test_xgboost_model.json"
        config.MODEL_PATH = self.temp_model_path
        self.manager = CryptoModelManager()
        
        # Create a mock database of candles
        self.mock_candles = []
        np.random.seed(42)
        base_price = 100.0
        for i in range(200):
            base_price += np.random.randn()
            self.mock_candles.append({
                "open_time": 1718000000000 + (i * 60000),
                "open": base_price + np.random.randn() * 0.1,
                "high": base_price + 0.5,
                "low": base_price - 0.5,
                "close": base_price,
                "volume": 500.0 + np.random.uniform(-50, 50)
            })

    def tearDown(self):
        if os.path.exists(self.temp_model_path):
            os.remove(self.temp_model_path)

    @patch('src.database.get_cached_candles')
    def test_prepare_training_data(self, mock_get_candles):
        mock_get_candles.return_value = self.mock_candles
        
        X, y = self.manager.prepare_training_data(["BTCUSDT"])
        
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertEqual(len(X), len(y))
        
        # Verify columns exist
        for col in self.manager.feature_cols:
            self.assertIn(col, X.columns)
            
        # Target classes must be binary: 0 or 1
        self.assertTrue(set(y.unique()).issubset({0, 1}))

    @patch('src.database.get_cached_candles')
    def test_model_training_and_saving(self, mock_get_candles):
        mock_get_candles.return_value = self.mock_candles
        
        # Run training
        success = self.manager.train(["BTCUSDT"])
        
        self.assertTrue(success)
        self.assertIsNotNone(self.manager.model)
        self.assertTrue(os.path.exists(self.temp_model_path))

    def test_predict_probability(self):
        # Setup mock indicators dataframe
        size = 65
        df = pd.DataFrame({
            'open':  np.random.uniform(90, 110, size),
            'high':  np.random.uniform(90, 110, size),
            'low':   np.random.uniform(90, 110, size),
            'close': np.random.uniform(90, 110, size),
            'volume': np.random.uniform(100, 1000, size),
            'ema_20': np.random.uniform(90, 110, size),
            'ema_50': np.random.uniform(90, 110, size),
            'rsi_14': np.random.uniform(20, 80, size),
            'macd_line':   np.random.uniform(-1, 1, size),
            'macd_signal': np.random.uniform(-1, 1, size),
            'macd_hist':   np.random.uniform(-1, 1, size),
            'bb_middle': np.random.uniform(90, 110, size),
            'bb_upper':  np.random.uniform(100, 120, size),
            'bb_lower':  np.random.uniform(80, 100, size),
            'volume_change_pct': np.random.uniform(-50, 200, size),
            'atr_14':      np.random.uniform(0.1, 2.0, size),
            'momentum_5':  np.random.uniform(-2, 2, size),
            'momentum_15': np.random.uniform(-4, 4, size),
            'momentum_30': np.random.uniform(-6, 6, size),
            # New indicators added in overhaul
            'stoch_rsi_k': np.random.uniform(0, 100, size),
            'stoch_rsi_d': np.random.uniform(0, 100, size),
            'obv':         np.cumsum(np.random.uniform(-1000, 1000, size)),
        })
        
        # Predict should return a default 0.5 if no model is loaded
        self.manager.model = None
        prob_default = self.manager.predict_probability(df)
        self.assertEqual(prob_default, 0.5)
        
        # Mock actual XGBoost model prediction
        self.manager.model = MagicMock()
        self.manager.model.predict_proba.return_value = np.array([[0.2, 0.8]])
        
        prob = self.manager.predict_probability(df)
        self.assertEqual(prob, 0.8)

if __name__ == '__main__':
    unittest.main()
