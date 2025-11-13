"""
ml_build_dataset.py

Construit un dataset de features (avec Kalman + Savgol + Delta) pour XGBoost
à partir d'un historique MNQ/NQ.

Entrée:
    - CSV avec colonnes au minimum:
        datetime, open, high, low, close, volume
    - Optionnel:
        delta        = delta par chandelle (buy_vol - sell_vol)
        cum_delta    = cumulative delta (sera converti en delta par chandelle)

Sortie:
    - data/dataset_features.parquet
    - data/dataset_features.csv
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# ==========================
#  FILTRES KALMAN & SAVGOL
# ==========================

def kalman_filter(prices: pd.Series, R: float = 0.01, Q: float = 1e-5) -> pd.Series:
    """
    Kalman filter 1D simple.
    prices : pd.Series de prix (close)
    R : variance du bruit d'observation (plus grand = lissage plus fort)
    Q : variance du bruit du modèle (plus grand = suit plus vite le marché)
    """
    prices = prices.astype(float)
    n = len(prices)

    if n == 0:
        return prices.copy()

    xhat = np.zeros(n)  # estimate
    P = np.zeros(n)     # error covariance

    xhat[0] = prices.iloc[0]
    P[0] = 1.0

    for k in range(1, n):
        # Prediction
        xhat_minus = xhat[k - 1]
        P_minus = P[k - 1] + Q

        # Mise à jour
        K = P_minus / (P_minus + R)
        xhat[k] = xhat_minus + K * (prices.iloc[k] - xhat_minus)
        P[k] = (1 - K) * P_minus

    return pd.Series(xhat, index=prices.index, name="close_kalman")


def safe_savgol(series: pd.Series, window: int = 11, polyorder: int = 2) -> pd.Series:
    """
    Applique Savitzky-Golay de façon safe:
    - si la série est trop courte, renvoie la série d'origine
    """
    series = series.astype(float)
    n = len(series)
    if n < window:
        return series.copy()
    filt = savgol_filter(series.values, window_length=window, polyorder=polyorder)
    return pd.Series(filt, index=series.index, name=(series.name or "") + "_savgol")


# ==========================
#  INDICATEURS TECHNIQUES
# ==========================

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)

    gain = pd.Series(up, index=series.index).rolling(window=period).mean()
    loss = pd.Series(down, index=series.index).rolling(window=period).mean()

    rs = gain / (loss + 1e-9)
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    return rsi_val


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def macd(series: pd.Series,
         fast: int = 12,
         slow: int = 26,
         signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# ==========================
#  FEATURES
# ==========================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les colonnes de features au DataFrame df.
    df doit contenir: open, high, low, close, volume.
    Optionnel: delta ou cum_delta
    """

    df = df.sort_values("datetime").reset_index(drop=True)

    # --- Delta : cum_delta -> delta chandelle si besoin ---
    if "delta" not in df.columns:
        # On essaie de trouver une colonne de cumulative delta
        for name in ["cum_delta", "cumulative_delta", "cumdelta", "cumulativedelta"]:
            if name in df.columns:
                df["delta"] = df[name].diff().fillna(0.0)
                break

    # S'il n'y a ni delta ni cum_delta, on met delta à 0 (le script reste utilisable)
    if "delta" not in df.columns:
        df["delta"] = 0.0

    # --- Lissage Kalman sur le close ---
    df["close_kalman"] = kalman_filter(df["close"], R=0.01, Q=1e-5)
    df["close_kalman_diff"] = df["close"] - df["close_kalman"]

    # --- Lissage Savgol sur le volume ---
    df["volume_savgol"] = safe_savgol(df["volume"], window=11, polyorder=2)
    df["volume_savgol_diff"] = df["volume"] - df["volume_savgol"]

    # --- Lissage Savgol sur le delta chandelle ---
    df["delta_raw"] = df["delta"].astype(float)
    df["delta_savgol"] = safe_savgol(df["delta_raw"], window=11, polyorder=2)
    df["delta_ma20"] = df["delta_savgol"].rolling(window=20).mean()
    df["delta_slope"] = df["delta_savgol"].diff()

    # --- Bougies (sur prix brut) ---
    df["body"] = (df["close"] - df["open"]).abs()
    df["range"] = df["high"] - df["low"]
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["body_to_range"] = df["body"] / (df["range"] + 1e-9)

    # --- EMA / SMA sur prix lissé Kalman ---
    price_ref = df["close_kalman"]

    df["ema_9"] = ema(price_ref, 9)
    df["ema_21"] = ema(price_ref, 21)
    df["ema_50"] = ema(price_ref, 50)
    df["sma_200"] = price_ref.rolling(window=200).mean()

    # Distances
    df["close_minus_ema9"] = price_ref - df["ema_9"]
    df["close_minus_ema21"] = price_ref - df["ema_21"]
    df["ema9_minus_ema21"] = df["ema_9"] - df["ema_21"]
    df["close_minus_sma200"] = price_ref - df["sma_200"]

    # --- RSI sur prix lissé ---
    df["rsi_14"] = rsi(price_ref, 14)
    df["rsi_14_slope"] = df["rsi_14"].diff()

    # --- MACD sur prix lissé ---
    macd_line, signal_line, hist = macd(price_ref, 12, 26, 9)
    df["macd_line"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist

    # --- Volume & volatilité ---
    df["volume_ma20"] = df["volume_savgol"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume_savgol"] / (df["volume_ma20"] + 1e-9)

    df["atr_14"] = atr(df["high"], df["low"], df["close"], 14)
    df["atr_slope"] = df["atr_14"].diff()
    df["volatility_ratio"] = df["atr_14"] / (df["close"] + 1e-9)

    return df


# ==========================
#  LABELS (ce que XGBoost doit prédire)
# ==========================

def build_labels(df: pd.DataFrame,
                 horizon: int = 10,
                 min_move_atr: float = 0.5) -> pd.DataFrame:
    """
    Label binaire:
        1  = bon setup long  (prix monte >= min_move_atr * ATR)
        -1 = bon setup short (prix baisse <= -min_move_atr * ATR)
        0  = neutre (pas utilisé pour l'entraînement)

    horizon: nombre de bougies futures examinées
    """

    df["close_future"] = df["close"].shift(-horizon)
    df["future_return"] = df["close_future"] - df["close"]
    df["future_return_atr"] = df["future_return"] / (df["atr_14"] + 1e-9)

    cond_long = df["future_return_atr"] >= min_move_atr
    cond_short = df["future_return_atr"] <= -min_move_atr

    df["label"] = 0
    df.loc[cond_long, "label"] = 1
    df.loc[cond_short, "label"] = -1

    return df


# ==========================
#  MAIN
# ==========================

def main():
    # Fichier d'entrée - adapte ce chemin selon ton repo
    input_path = os.path.join("data", "mnq_1m.csv")

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Fichier introuvable: {input_path}. "
            f"Assure-toi d'avoir un CSV historique dans data/mnq_1m.csv"
        )

    print(f"Lecture du fichier: {input_path}")
    df = pd.read_csv(input_path)

    # Normalisation noms de colonnes
    df.columns = [c.lower() for c in df.columns]

    # Date/heure
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    elif "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"])
    else:
        raise ValueError("La colonne 'datetime' ou 'date' est requise dans le CSV.")

    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Colonne requise manquante dans le CSV: {col}")

    print("Construction des features (Kalman + Savgol + Delta)...")
    df = build_features(df)

    print("Construction des labels...")
    df = build_labels(df, horizon=10, min_move_atr=0.5)

    feature_cols = [
        # Bougies
        "body", "range", "upper_wick", "lower_wick", "body_to_range",
        # Prix lissé
        "close_kalman", "close_kalman_diff",
        # Volume lissé
        "volume_savgol", "volume_savgol_diff",
        # Delta lissé
        "delta_raw", "delta_savgol", "delta_ma20", "delta_slope",
        # MAs
        "ema_9", "ema_21", "ema_50", "sma_200",
        "close_minus_ema9", "close_minus_ema21",
        "ema9_minus_ema21", "close_minus_sma200",
        # RSI / MACD
        "rsi_14", "rsi_14_slope",
        "macd_line", "macd_signal", "macd_hist",
        # Volume & vol
        "volume", "volume_ma20", "volume_ratio",
        "atr_14", "atr_slope", "volatility_ratio",
    ]

    all_cols = ["datetime"] + feature_cols + ["label", "future_return_atr"]

    df_clean = df[all_cols].dropna().reset_index(drop=True)

    # On enlève les exemples neutres (label = 0)
    df_clean = df_clean[df_clean["label"] != 0].reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    output_path_parquet = os.path.join("data", "dataset_features.parquet")
    output_path_csv = os.path.join("data", "dataset_features.csv")

    print(f"Sauvegarde du dataset dans:\n  {output_path_parquet}\n  {output_path_csv}")
    df_clean.to_parquet(output_path_parquet, index=False)
    df_clean.to_csv(output_path_csv, index=False)

    print("Terminé. Aperçu des 5 premières lignes:")
    print(df_clean.head())


if __name__ == "__main__":
    main()
