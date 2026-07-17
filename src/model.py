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
        self.model_path_base = config.MODEL_PATH
        self.model_path_long = self.model_path_base.replace(".json", "_long.json")
        self.model_path_short = self.model_path_base.replace(".json", "_short.json")
        
        self.model_long = None
        self.model_short = None
        
        # All 17 normalized features the models are trained on
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
            'feat_momentum_30',
            'feat_stoch_rsi_k',
            'feat_stoch_rsi_d',
            'feat_obv_norm',
        ]
        
        # Load models if they exist
        self.load_model()

    # Legacy compatibility property for test suites expecting .model
    @property
    def model(self):
        return self.model_long

    @model.setter
    def model(self, value):
        self.model_long = value

    def load_model(self):
        """Loads the saved XGBoost models if they exist."""
        # 1. Load Long model
        if os.path.exists(self.model_path_long):
            try:
                self.model_long = xgb.XGBClassifier()
                self.model_long.load_model(self.model_path_long)
                logger.info(f"Loaded existing Long model from {self.model_path_long}")
            except Exception as e:
                logger.error(f"Failed to load Long model: {e}")
                self.model_long = None
        elif os.path.exists(self.model_path_base):
            # Fallback to base model path if legacy file exists
            try:
                self.model_long = xgb.XGBClassifier()
                self.model_long.load_model(self.model_path_base)
                logger.info(f"Loaded existing legacy model from {self.model_path_base}")
            except Exception as e:
                logger.error(f"Failed to load legacy model: {e}")
                self.model_long = None
        else:
            logger.info("No existing Long model found. Will train on startup.")
            
        # 2. Load Short model
        if os.path.exists(self.model_path_short):
            try:
                self.model_short = xgb.XGBClassifier()
                self.model_short.load_model(self.model_path_short)
                logger.info(f"Loaded existing Short model from {self.model_path_short}")
            except Exception as e:
                logger.error(f"Failed to load Short model: {e}")
                self.model_short = None
        else:
            logger.info("No existing Short model found. Will train on startup.")

    def save_model(self):
        """Saves current models to disk."""
        if self.model_long:
            try:
                self.model_long.save_model(self.model_path_long)
                self.model_long.save_model(self.model_path_base)
                logger.info(f"Saved Long model to {self.model_path_long} and legacy base path {self.model_path_base}")
            except Exception as e:
                logger.error(f"Failed to save Long model: {e}")
                
        if self.model_short:
            try:
                self.model_short.save_model(self.model_path_short)
                logger.info(f"Saved Short model to {self.model_path_short}")
            except Exception as e:
                logger.error(f"Failed to save Short model: {e}")

    # Legacy method signature returning just X and y_long to pass standard unit tests
    def prepare_training_data(self, symbols):
        X, y_long, _ = self.prepare_training_data_dual(symbols)
        return X, y_long

    def prepare_training_data_dual(self, symbols):
        """
        Loads all cached candles, computes indicators, labels target classes,
        and aggregates them into training datasets.
        """
        all_features = []
        all_targets_long = []
        all_targets_short = []
        
        logger.info(f"Preparing training data for {len(symbols)} symbols...")
        
        for symbol in symbols:
            candles = database.get_cached_candles(symbol)
            if len(candles) < 100:
                logger.warning(f"Insufficient candles for {symbol} ({len(candles)} found). Skipping from training.")
                continue
                
            # Convert to DataFrame
            df = pd.DataFrame(candles)
            df = calculate_all_indicators(df)
            
            # Label Long: 1 if high increases by >= 0.5% in the next 15 minutes, else 0
            high_shifted = [df['high'].shift(-h) for h in range(1, 16)]
            future_max_high = pd.concat(high_shifted, axis=1).max(axis=1)
            df['target_long'] = (future_max_high >= (df['close'] * 1.005)).astype(int)
            
            # Label Short: 1 if low drops by >= 0.5% in the next 15 minutes, else 0
            low_shifted = [df['low'].shift(-h) for h in range(1, 16)]
            future_min_low = pd.concat(low_shifted, axis=1).min(axis=1)
            df['target_short'] = (future_min_low <= (df['close'] * 0.995)).astype(int)
            
            # Generate normalized features
            feats = generate_normalized_features(df)
            
            # Append targets to clean alignment
            feats['target_long'] = df['target_long']
            feats['target_short'] = df['target_short']
            clean_df = feats.dropna()
            
            if len(clean_df) > 0:
                all_features.append(clean_df[self.feature_cols])
                all_targets_long.append(clean_df['target_long'])
                all_targets_short.append(clean_df['target_short'])
                
        if not all_features:
            logger.error("No training data could be prepared.")
            return None, None, None
            
        X = pd.concat(all_features, axis=0)
        y_long = pd.concat(all_targets_long, axis=0)
        y_short = pd.concat(all_targets_short, axis=0)
        
        logger.info(f"Training data ready. Total samples: {len(X)}")
        logger.info(f"  Long Positives: {sum(y_long)} ({round(sum(y_long)/len(y_long)*100, 1)}%)")
        logger.info(f"  Short Positives: {sum(y_short)} ({round(sum(y_short)/len(y_short)*100, 1)}%)")
        
        return X, y_long, y_short

    def train(self, symbols):
        """Trains both Long and Short classifiers on historical data."""
        X, y_long, y_short = self.prepare_training_data_dual(symbols)
        if X is None or len(X) < 200:
            logger.error("Not enough historical data to train the models. Minimum 200 samples required.")
            return False
            
        try:
            # 1. Train Long Model with class-imbalance correction
            logger.info("Training Long model classifier...")
            X_train_l, X_val_l, y_train_l, y_val_l = train_test_split(X, y_long, test_size=0.2, random_state=42)
            neg_l = int((y_train_l == 0).sum())
            pos_l = int((y_train_l == 1).sum())
            spw_l = max(1.0, neg_l / pos_l) if pos_l > 0 else 1.0
            logger.info(f"Long class balance: {neg_l} neg / {pos_l} pos — scale_pos_weight={spw_l:.1f}")
            model_long_temp = xgb.XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=spw_l,
                tree_method='hist',
                random_state=42,
                eval_metric='logloss',
                early_stopping_rounds=30
            )
            model_long_temp.fit(X_train_l, y_train_l, eval_set=[(X_val_l, y_val_l)], verbose=False)
            val_preds_l = model_long_temp.predict(X_val_l)
            acc_l = (val_preds_l == y_val_l).mean()
            logger.info(f"Long model complete. Validation Accuracy: {round(acc_l * 100, 2)}%")
            self.model_long = model_long_temp
            
            # 2. Train Short Model with class-imbalance correction
            logger.info("Training Short model classifier...")
            X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(X, y_short, test_size=0.2, random_state=42)
            neg_s = int((y_train_s == 0).sum())
            pos_s = int((y_train_s == 1).sum())
            spw_s = max(1.0, neg_s / pos_s) if pos_s > 0 else 1.0
            logger.info(f"Short class balance: {neg_s} neg / {pos_s} pos — scale_pos_weight={spw_s:.1f}")
            model_short_temp = xgb.XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=spw_s,
                tree_method='hist',
                random_state=42,
                eval_metric='logloss',
                early_stopping_rounds=30
            )
            model_short_temp.fit(X_train_s, y_train_s, eval_set=[(X_val_s, y_val_s)], verbose=False)
            val_preds_s = model_short_temp.predict(X_val_s)
            acc_s = (val_preds_s == y_val_s).mean()
            logger.info(f"Short model complete. Validation Accuracy: {round(acc_s * 100, 2)}%")
            self.model_short = model_short_temp
            
            self.save_model()
            return True
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            return False

    # Legacy method signature returning just Long probability to pass standard unit tests
    def predict_probability(self, df):
        prob_long, _ = self.predict_probabilities(df)
        return prob_long

    def predict_probabilities(self, df):
        """
        Generates prediction probabilities for both directions.
        Returns: (prob_long, prob_short)
        """
        prob_long = 0.5
        prob_short = 0.5
        
        try:
            feats = generate_normalized_features(df)
            latest_feat = feats.iloc[[-1]]
            
            if latest_feat[self.feature_cols].isnull().values.any():
                return 0.5, 0.5
                
            if self.model_long is not None:
                prob_long = float(self.model_long.predict_proba(latest_feat[self.feature_cols])[0, 1])
                
            if self.model_short is not None:
                prob_short = float(self.model_short.predict_proba(latest_feat[self.feature_cols])[0, 1])
                
            return prob_long, prob_short
        except Exception as e:
            logger.error(f"Error running model prediction: {e}")
            return 0.5, 0.5
