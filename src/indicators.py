import pandas as pd
import numpy as np

def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates technical indicators on a DataFrame containing at least
    ['open', 'high', 'low', 'close', 'volume'] columns.
    Returns the DataFrame with additional indicator columns.
    """
    if len(df) < 50:
        # Return unmodified or with empty columns to prevent crashing on short history
        return df

    # Make sure columns are float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    # 1. EMA 20 and EMA 50
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

    # 2. RSI (14) using Wilder's smoothing
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    
    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['rsi_14'] = df['rsi_14'].fillna(50)  # Default neutral

    # 3. MACD (12, 26, 9)
    df['macd_ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['macd_ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_line'] = df['macd_ema_12'] - df['macd_ema_26']
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal']

    # 4. Bollinger Bands (20, 2)
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    # Fill standard deviation/bounds on first candles
    df['bb_middle'] = df['bb_middle'].fillna(df['close'])
    df['bb_upper'] = df['bb_upper'].fillna(df['close'])
    df['bb_lower'] = df['bb_lower'].fillna(df['close'])

    # 5. Volume change percentage (current volume vs average of last 10 periods)
    avg_volume_10 = df['volume'].rolling(window=10).mean()
    df['volume_change_pct'] = ((df['volume'] - avg_volume_10) / avg_volume_10.replace(0, np.nan)) * 100.0
    df['volume_change_pct'] = df['volume_change_pct'].fillna(0.0)

    # 6. ATR (14) for volatility
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.ewm(alpha=1/14, adjust=False).mean()
    df['atr_14'] = df['atr_14'].fillna(0.0)

    # 7. Price momentum over the last 5, 15, and 30 minutes (percentage price changes)
    df['momentum_5'] = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5).replace(0, np.nan)) * 100.0
    df['momentum_15'] = ((df['close'] - df['close'].shift(15)) / df['close'].shift(15).replace(0, np.nan)) * 100.0
    df['momentum_30'] = ((df['close'] - df['close'].shift(30)) / df['close'].shift(30).replace(0, np.nan)) * 100.0
    
    # Fill initial NaN momentum entries with 0.0
    df['momentum_5'] = df['momentum_5'].fillna(0.0)
    df['momentum_15'] = df['momentum_15'].fillna(0.0)
    df['momentum_30'] = df['momentum_30'].fillna(0.0)

    return df

def generate_normalized_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates normalized features that are price-scale-independent.
    This allows a single machine learning model to generalize across assets 
    with different price denominations (e.g. BTC/USDT at 60k vs DOGE/USDT at 0.1).
    """
    features = pd.DataFrame(index=df.index)
    
    close = df['close']
    
    # Relative Moving Averages
    features['feat_close_to_ema20'] = (close - df['ema_20']) / df['ema_20']
    features['feat_close_to_ema50'] = (close - df['ema_50']) / df['ema_50']
    features['feat_ema20_to_ema50'] = (df['ema_20'] - df['ema_50']) / df['ema_50']
    
    # Scaled RSI (from [0, 100] to [0, 1])
    features['feat_rsi_scaled'] = df['rsi_14'] / 100.0
    
    # Relative MACD
    features['feat_macd_line'] = df['macd_line'] / close
    features['feat_macd_signal'] = df['macd_signal'] / close
    features['feat_macd_hist'] = df['macd_hist'] / close
    
    # Bollinger %B (where current price lies between upper and lower band)
    bb_width = df['bb_upper'] - df['bb_lower']
    features['feat_bb_pct_b'] = (close - df['bb_lower']) / bb_width.replace(0, np.nan)
    features['feat_bb_pct_b'] = features['feat_bb_pct_b'].fillna(0.5)  # Default to middle
    
    # Normalized Bollinger Band Width
    features['feat_bb_width'] = bb_width / df['bb_middle']
    
    # Normalized Volatility (ATR / Close)
    features['feat_atr_scaled'] = df['atr_14'] / close
    
    # Volume Change (scaled from percent to decimal)
    features['feat_volume_change'] = df['volume_change_pct'] / 100.0
    
    # Momentum (scaled from percent to decimal)
    features['feat_momentum_5'] = df['momentum_5'] / 100.0
    features['feat_momentum_15'] = df['momentum_15'] / 100.0
    features['feat_momentum_30'] = df['momentum_30'] / 100.0
    
    return features
