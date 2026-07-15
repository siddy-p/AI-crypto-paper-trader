import unittest
import pandas as pd
import numpy as np
from src.indicators import calculate_all_indicators, generate_normalized_features

class TestIndicators(unittest.TestCase):
    def setUp(self):
        # Generate 100 periods of mock candle data
        self.size = 100
        np.random.seed(42)
        
        # Simulate a random walk price action
        close_prices = 100.0 + np.cumsum(np.random.randn(self.size))
        high_prices = close_prices + np.random.uniform(0.1, 2.0, self.size)
        low_prices = close_prices - np.random.uniform(0.1, 2.0, self.size)
        open_prices = close_prices + np.random.randn(self.size)
        volumes = np.random.uniform(100, 1000, self.size)
        
        self.df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        })

    def test_calculate_all_indicators_columns(self):
        # Ensure indicators execute and append all required outputs
        df_indicators = calculate_all_indicators(self.df.copy())
        
        expected_cols = [
            'ema_20', 'ema_50', 'rsi_14', 'macd_line', 'macd_signal', 
            'macd_hist', 'bb_middle', 'bb_upper', 'bb_lower', 
            'volume_change_pct', 'atr_14', 'momentum_5', 'momentum_15', 'momentum_30'
        ]
        
        for col in expected_cols:
            self.assertIn(col, df_indicators.columns)
            self.assertFalse(df_indicators[col].isnull().all(), f"Column {col} is all NaNs")

    def test_normalize_features(self):
        df_indicators = calculate_all_indicators(self.df.copy())
        features = generate_normalized_features(df_indicators)
        
        expected_feature_cols = [
            'feat_close_to_ema20', 'feat_close_to_ema50', 'feat_ema20_to_ema50',
            'feat_rsi_scaled', 'feat_macd_line', 'feat_macd_signal', 'feat_macd_hist',
            'feat_bb_pct_b', 'feat_bb_width', 'feat_atr_scaled', 'feat_volume_change',
            'feat_momentum_5', 'feat_momentum_15', 'feat_momentum_30'
        ]
        
        for col in expected_feature_cols:
            self.assertIn(col, features.columns)
            # Ensure price scale normalization maps close to ratio limits
            # e.g., scaled RSI should be in range [0, 1]
            if col == 'feat_rsi_scaled':
                self.assertTrue((features[col].dropna() <= 1.0).all())
                self.assertTrue((features[col].dropna() >= 0.0).all())

    def test_short_dataframe_handling(self):
        # Dataframe under 50 rows should return without indicators to prevent crashes
        short_df = self.df.iloc[:20].copy()
        result = calculate_all_indicators(short_df)
        self.assertEqual(len(result.columns), 5) # Should only contain original cols
        
if __name__ == '__main__':
    unittest.main()
