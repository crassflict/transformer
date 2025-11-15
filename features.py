# features.py

import numpy as np
import pandas as pd

from pykalman import KalmanFilter
from scipy.signal import savgol_filter

FEATURE_COLUMNS = [
    "close",
    "volume",
    "ema_fast",
    "ema_slow",
    "macd",
    "macd_signal",
    "lag_1",
    "lag_2",
    "lag_3",
    "ret_1",
    "ret_5",
    "delta",
    "rsi",
    "kalman_close",
    "savgol_close",
]


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    gain_ema = pd.Series(gain, index=series.index).ewm(span=window, adjust=False).mean()
    loss_ema = pd.Series(loss, index=series.index).ewm(span=window, adjust=False).mean()

    rs = gain_ema / (loss_ema + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series,
                 fast_span: int = 12,
                 slow_span: int = 26,
                 signal_span: int = 9) -> tuple[pd.Series, pd.Series]:
    ema_fast = compute_ema(series, fast_span)
    ema_slow = compute_ema(series, slow_span)
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal_span, adjust=False).mean()
    return macd, macd_signal


def apply_kalman(series: pd.Series) -> pd.Series:
    # Kalman très simple 1D
    values = series.values.astype(float)
    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=values[0],
                      initial_state_covariance=1,
                      observation_covariance=1,
                      transition_covariance=0.01)
    state_means, _ = kf.filter(values)
    return pd.Series(state_means.flatten(), index=series.index)


def apply_savgol(series: pd.Series,
                 window_length: int = 11,
                 polyorder: int = 3) -> pd.Series:
    # window_length doit être impair et <= len(series)
    if len(series) < window_length:
        return series.copy()
    if window_length % 2 == 0:
        window_length += 1
    smoothed = savgol_filter(series.values.astype(float),
                             window_length=window_length,
                             polyorder=polyorder)
    return pd.Series(smoothed, index=series.index)


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df doit contenir au minimum: ['close', 'volume'].
    Ajoute toutes les colonnes de FEATURE_COLUMNS (sauf label).
    """
    df = df.copy()

    close = df["close"]
    volume = df["volume"]

    # EMAs
    df["ema_fast"] = compute_ema(close, span=9)
    df["ema_slow"] = compute_ema(close, span=21)

    # MACD
    macd, macd_signal = compute_macd(close)
    df["macd"] = macd
    df["macd_signal"] = macd_signal

    # Lags & returns
    df["lag_1"] = close.shift(1)
    df["lag_2"] = close.shift(2)
    df["lag_3"] = close.shift(3)

    df["ret_1"] = close.pct_change(1)
    df["ret_5"] = close.pct_change(5)

    # Delta
    df["delta"] = close.diff()

    # RSI
    df["rsi"] = compute_rsi(close, window=14)

    # Kalman & Savgol
    df["kalman_close"] = apply_kalman(close)
    df["savgol_close"] = apply_savgol(close, window_length=11, polyorder=3)

    return df
