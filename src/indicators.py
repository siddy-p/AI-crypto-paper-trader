import pandas as pd
import numpy as np

def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates technical indicators on a DataFrame containing at least
    ['open', 'high', 'low', 'close', 'volume'] columns.
    Returns the DataFrame with additional indicator columns.
    """
    if len(df) < 50:
        return df

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    # 1. EMA 20 and EMA 50
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

    # 2. RSI (14) — Wilder's RMA: alpha=1/period (FIXED from incorrect com=13)
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['rsi_14'] = df['rsi_14'].fillna(50)

    # 3. MACD (12, 26, 9)
    df['macd_ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['macd_ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_line']   = df['macd_ema_12'] - df['macd_ema_26']
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist']   = df['macd_line'] - df['macd_signal']

    # 4. Bollinger Bands (20, 2)
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std']    = df['close'].rolling(window=20).std()
    df['bb_upper']  = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower']  = df['bb_middle'] - (df['bb_std'] * 2)
    df['bb_middle'] = df['bb_middle'].fillna(df['close'])
    df['bb_upper']  = df['bb_upper'].fillna(df['close'])
    df['bb_lower']  = df['bb_lower'].fillna(df['close'])

    # 5. Volume change % vs 10-period average
    avg_volume_10 = df['volume'].rolling(window=10).mean()
    df['volume_change_pct'] = ((df['volume'] - avg_volume_10) / avg_volume_10.replace(0, np.nan)) * 100.0
    df['volume_change_pct'] = df['volume_change_pct'].fillna(0.0)

    # 6. ATR (14) — Wilder's RMA
    high_low    = df['high'] - df['low']
    high_close  = (df['high'] - df['close'].shift()).abs()
    low_close   = (df['low']  - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.ewm(alpha=1/14, adjust=False).mean()
    df['atr_14'] = df['atr_14'].fillna(0.0)

    # 7. Price momentum over 5, 15, 30 minutes
    df['momentum_5']  = ((df['close'] - df['close'].shift(5))  / df['close'].shift(5).replace(0, np.nan)) * 100.0
    df['momentum_15'] = ((df['close'] - df['close'].shift(15)) / df['close'].shift(15).replace(0, np.nan)) * 100.0
    df['momentum_30'] = ((df['close'] - df['close'].shift(30)) / df['close'].shift(30).replace(0, np.nan)) * 100.0
    df['momentum_5']  = df['momentum_5'].fillna(0.0)
    df['momentum_15'] = df['momentum_15'].fillna(0.0)
    df['momentum_30'] = df['momentum_30'].fillna(0.0)

    # 8. Stochastic RSI (14, 14, 3, 3) — leading reversal indicator
    rsi       = df['rsi_14']
    rsi_min   = rsi.rolling(window=14).min()
    rsi_max   = rsi.rolling(window=14).max()
    rsi_range = (rsi_max - rsi_min).replace(0, np.nan)
    stoch_k_raw      = ((rsi - rsi_min) / rsi_range) * 100.0
    df['stoch_rsi_k'] = stoch_k_raw.rolling(window=3).mean().fillna(50.0)
    df['stoch_rsi_d'] = df['stoch_rsi_k'].rolling(window=3).mean().fillna(50.0)

    # 9. On-Balance Volume (OBV) — volume confirms price direction
    obv     = [0]
    closes  = df['close'].values
    volumes = df['volume'].values
    for i in range(1, len(df)):
        if closes[i] > closes[i - 1]:
            obv.append(obv[-1] + volumes[i])
        elif closes[i] < closes[i - 1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv

    return df


def generate_normalized_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates normalized, price-scale-independent features for the ML model.
    Includes original 14 features + Stochastic RSI + OBV = 17 total features.
    """
    features = pd.DataFrame(index=df.index)
    close = df['close']

    # Relative Moving Averages
    features['feat_close_to_ema20'] = (close - df['ema_20']) / df['ema_20']
    features['feat_close_to_ema50'] = (close - df['ema_50']) / df['ema_50']
    features['feat_ema20_to_ema50'] = (df['ema_20'] - df['ema_50']) / df['ema_50']

    # RSI (scaled to [0, 1])
    features['feat_rsi_scaled'] = df['rsi_14'] / 100.0

    # MACD normalized by price
    features['feat_macd_line']   = df['macd_line']   / close
    features['feat_macd_signal'] = df['macd_signal'] / close
    features['feat_macd_hist']   = df['macd_hist']   / close

    # Bollinger %B and bandwidth
    bb_width = df['bb_upper'] - df['bb_lower']
    features['feat_bb_pct_b'] = (close - df['bb_lower']) / bb_width.replace(0, np.nan)
    features['feat_bb_pct_b'] = features['feat_bb_pct_b'].fillna(0.5)
    features['feat_bb_width'] = bb_width / df['bb_middle']

    # ATR / Close
    features['feat_atr_scaled'] = df['atr_14'] / close

    # Volume change
    features['feat_volume_change'] = df['volume_change_pct'] / 100.0

    # Momentum
    features['feat_momentum_5']  = df['momentum_5']  / 100.0
    features['feat_momentum_15'] = df['momentum_15'] / 100.0
    features['feat_momentum_30'] = df['momentum_30'] / 100.0

    # Stochastic RSI (scaled to [0, 1])
    features['feat_stoch_rsi_k'] = df['stoch_rsi_k'] / 100.0
    features['feat_stoch_rsi_d'] = df['stoch_rsi_d'] / 100.0

    # OBV z-score (20-period rolling, clipped to [-5, 5])
    obv_mean = df['obv'].rolling(window=20).mean()
    obv_std  = df['obv'].rolling(window=20).std().replace(0, np.nan)
    features['feat_obv_norm'] = ((df['obv'] - obv_mean) / obv_std).fillna(0.0).clip(-5, 5)

    return features
