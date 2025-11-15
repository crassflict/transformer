from __future__ import annotations
"""
Feature engineering partagé entre :
- ml_build_dataset.py (dataset historique)
- serveur FastAPI (temps réel, Option B)

On part sur :
- close, volume
- EMA rapide & lente
- RSI
- lags de prix
- returns courts
- Kalman filter sur le prix
- Savitzky–Golay smoothing sur le prix
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from pykalman import KalmanFilter
from scipy.signal import savgol_filter


@dataclass
class FeatureConfig:
    ema_fast: int = 9
    ema_slow: int = 21
    rsi_period: int = 14
    lag_periods: Tuple[int, ...] = (1, 2, 3)
    return_periods: Tuple[int, ...] = (1, 5)
    kalman_dim_state: int = 1
    savgol_window: int = 11  # doit être impair
    savgol_poly: int = 3


FEATURE_COLUMNS: List[str] = [
    "close",
    "volume",
    "ema_fast",
    "ema_slow",
    "rsi",
    "close_kalman",
    "close_savgol",
    "lag_1",
    "lag_2",
    "lag_3",
    "ret_1",
    "ret_5",
]

LABEL_COLUMN: str = "label"


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _apply_kalman(close: pd.Series, dim_state: int = 1) -> pd.Series:
    if close.empty:
        return close.copy()

    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=close.iloc[0],
    )
    state_means, _ = kf.smooth(close.values.astype(float))
    smoothed = pd.Series(state_means.flatten(), index=close.index, name="close_kalman")
    return smoothed


def _apply_savgol(close: pd.Series, window: int, poly: int) -> pd.Series:
    if len(close) < window:
        return pd.Series(np.nan, index=close.index, name="close_savgol")

    filt = savgol_filter(
        close.values.astype(float),
        window_length=window,
        polyorder=poly,
    )
    return pd.Series(filt, index=close.index, name="close_savgol")


def add_features(df: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """
    Prend un DataFrame avec au minimum :
        - 'timestamp' (optionnel mais recommandé)
        - 'close'
        - 'volume' (peut être rempli à 0 si absent)
    Retourne un DataFrame avec toutes les colonnes de FEATURE_COLUMNS
    + la colonne LABEL_COLUMN ('label' -> 0 ou 1).
    """
    if cfg is None:
        cfg = FeatureConfig()

    df = df.copy()

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    if "volume" not in df.columns:
        df["volume"] = 0.0

    close = df["close"]

    df["ema_fast"] = close.ewm(span=cfg.ema_fast, adjust=False).mean()
    df["ema_slow"] = close.ewm(span=cfg.ema_slow, adjust=False).mean()

    df["rsi"] = _compute_rsi(close, period=cfg.rsi_period)

    df["close_kalman"] = _apply_kalman(close, dim_state=cfg.kalman_dim_state)
    df["close_savgol"] = _apply_savgol(
        close, window=cfg.savgol_window, poly=cfg.savgol_poly
    )

    for p in cfg.lag_periods:
        df[f"lag_{p}"] = close.shift(p)

    for p in cfg.return_periods:
        df[f"ret_{p}"] = close.pct_change(p)

    future_close = close.shift(-1)
    future_ret = (future_close / close) - 1.0
    df[LABEL_COLUMN] = (future_ret > 0).astype(int)

    df = df.dropna().reset_index(drop=True)

    return df
