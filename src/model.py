import os
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from src import config
from src import database
from src.indicators import calculate_all_indicators, generate_normalized_features

logger = logging.getLogger(__name__)

class CryptoModelManager:
    def __init__(self):
        self.model_path = config.MODEL_PATH
        self.model = None
        
        # Define the feature list expected by the model
        self.feature_cols = [
            'feat_close_to_ema20',
            'feat_close_to_ema50',
            'feat_ema20_to_ema50',
            'feat_rsi_scaled',
            'feat_macd_line',
            'feat_macd_signal',
            'feat_macd_hist',
            'feat_bb_pct_b',
            'feat_bb_width',
            'feat_atr_scaled',
            'feat_volume_change',
            'feat_momentum_5',
            'feat_momentum_15',
            'feat_momentum_30'
        ]
        
        # Load model if it exists
        self.load_model()

    def load_model(self):
        """Loads the saved XGBoost model weights if they exist."""
        if os.path.exists(self.model_path):
            try:
                self.model = xgb.XGBClassifier()
                self.model.load_model(self.model_path)
                logger.info(f"Loaded existing model from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load model from {self.model_path}: {e}")
                self.model = None
        else:
            logger.info("No existing model found. Will need to train on startup.")

    def save_model(self):
        """Saves the current model weights to disk."""
        if self.model:
            try:
                self.model.save_model(self.model_path)
                logger.info(f"Saved model to {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to save model to {self.model_path}: {e}")

    def prepare_training_data(self, symbols):
        """
        Loads all cached candles, computes indicators, generates features,
        labels targets, and aggregates them into a single training dataframe.
        """
        all_features = []
        all_targets = []
        
        logger.info(f"Preparing training data for {len(symbols)} symbols...")
        
        for symbol in symbols:
            candles = database.get_cached_candles(symbol)
            if len(candles) < 100:
                logger.warning(f"Insufficient candles for {symbol} ({len(candles)} found). Skipping from training.")
                continue
                
            # Convert to DataFrame
            df = pd.DataFrame(candles)
            df = calculate_all_indicators(df)
            
            # 1. Generate labels: 1 if high price increases by >= 0.5% in the next 15 minutes, else 0
            # Next 15 minutes = index t+1 to t+15
            high_shifted = [df['high'].shift(-h) for h in range(1, 16)]
            future_max_high = pd.concat(high_shifted, axis=1).max(axis=1)
            
            # Target is 1 if future max high >= close * 1.005 (0.5% gain)
            df['target'] = (future_max_high >= (df['close'] * 1.005)).astype(int)
            
            # Generate normalized features
            feats = generate_normalized_features(df)
            
            # Drop rows with NaNs (first 50 due to indicators, last 15 due to lookahead target)
            feats['target'] = df['target']
            clean_df = feats.dropna()
            
            if len(clean_df) > 0:
                all_features.append(clean_df[self.feature_cols])
                all_targets.append(clean_df['target'])
                
        if not all_features:
            logger.error("No training data could be prepared.")
            return None, None
            
        X = pd.concat(all_features, axis=0)
        y = pd.concat(all_targets, axis=0)
        
        logger.info(f"Training data ready. Total samples: {len(X)} (Positive class: {sum(y)} - {round(sum(y)/len(y)*100, 2)}%)")
        return X, y

    def train(self, symbols):
        """
        Trains the XGBoost model on all available historical data
        and saves it.
        """
        X, y = self.prepare_training_data(symbols)
        if X is None or len(X) < 200:
            logger.error("Not enough historical data to train the model. Minimum 200 samples required.")
            return False
            
        try:
            logger.info("Training XGBoost Classifier...")
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Setup XGBoost classifier
            self.model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss",
                early_stopping_rounds=30
            )
            
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Evaluate performance on validation set
            val_preds = self.model.predict(X_val)
            val_probs = self.model.predict_proba(X_val)[:, 1]
            accuracy = (val_preds == y_val).mean()
            logger.info(f"Model training complete. Validation Accuracy: {round(accuracy * 100, 2)}%")
            
            self.save_model()
            return True
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            return False

    def predict_probability(self, df):
        """
        Generates the prediction probability for a single symbol state.
        Expects a DataFrame with the latest closed candle and its indicators calculated.
        """
        if self.model is None:
            logger.warning("No model loaded. Predicting default neutral probability (0.5).")
            return 0.5
            
        try:
            # Generate the feature vector for the last row (latest kline)
            feats = generate_normalized_features(df)
            latest_feat = feats.iloc[[-1]]
            
            # Verify no NaNs are present in features
            if latest_feat[self.feature_cols].isnull().values.any():
                logger.debug("Latest feature vector contains NaNs due to bootstrapping. Returning 0.5.")
                return 0.5
                
            # Run prediction probability for class 1
            prob = self.model.predict_proba(latest_feat[self.feature_cols])[0, 1]
            return float(prob)
        except Exception as e:
            logger.error(f"Error running model prediction: {e}")
            return 0.5
