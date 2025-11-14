import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from pykalman import KalmanFilter
import os

# === CONFIG ===
INPUT_FILE = "data/mnq_5m.csv"
OUT_FEATURES = "data/dataset_features.parquet"
OUT_LABELS = "data/dataset_labels.parquet"

def apply_kalman(series):
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=series.iloc[0],
        initial_state_covariance=1,
        observation_covariance=1,
        transition_covariance=0.01
    )
    state_means, _ = kf.smooth(series.values)
    return pd.Series(state_means.flatten(), index=series.index)

def build_dataset():
    df = pd.read_csv(INPUT_FILE)

    # timestamp → datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # === Basic features ===
    df['returns'] = df['close'].pct_change()
    df['delta'] = df['close'] - df['open']
    df['range'] = df['high'] - df['low']

    # === Technical indicators ===
    df['ema_fast'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=21, adjust=False).mean()
    df['sma50'] = df['close'].rolling(50).mean()
    df['sma200'] = df['close'].rolling(200).mean()

    # RSI
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    rs = up.rolling(14).mean() / down.rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['signal'] = df['macd'].ewm(span=9).mean()

    # === Kalman filter smoothing ===
    df['close_kalman'] = apply_kalman(df['close'])

    # === Savgol smoothing ===
    df['close_savgol'] = savgol_filter(df['close'], window_length=31, polyorder=3, mode='nearest')

    # === Lag features ===
    df['lag1'] = df['close'].shift(1)
    df['lag2'] = df['close'].shift(2)
    df['lag3'] = df['close'].shift(3)

    # === Target label: future direction ===
    df['future'] = df['close'].shift(-5)
    df['target'] = (df['future'] > df['close']).astype(int)

    # remove NaN
    df = df.dropna()

    # Save
    features = df[['close','close_kalman','close_savgol','returns','delta','range',
                   'ema_fast','ema_slow','sma50','sma200','rsi','macd','signal',
                   'lag1','lag2','lag3']]

    labels = df[['target']]

    features.to_parquet(OUT_FEATURES, index=False)
    labels.to_parquet(OUT_LABELS, index=False)

    print("Dataset built successfully!")
    print("Features:", features.shape, "Labels:", labels.shape)

if __name__ == "__main__":
    build_dataset()
