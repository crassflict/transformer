# ml_build_dataset.py

import pandas as pd
from pathlib import Path
from features import FEATURE_COLUMNS, LABEL_COLUMN

RAW_FILE = Path("data/mnq_5m.csv")          # ton historique 5 min
FEATURES_FILE = Path("data/dataset_features.parquet")
LABELS_FILE = Path("data/dataset_labels.parquet")


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def main():
    print(f"Loading raw data from {RAW_FILE} ...")
    df = pd.read_csv(RAW_FILE)

    # Assure-toi que les colonnes existent dans ton CSV
    # (ajuste les noms si besoin)
    # On suppose: time, open, high, low, close, volume
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)

    # ====== BASE ======
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    # EMA rapides / lentes
    df["ema_fast"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=21, adjust=False).mean()

    # Lags de prix
    df["lag_1"] = df["close"].shift(1)
    df["lag_2"] = df["close"].shift(2)
    df["lag_3"] = df["close"].shift(3)

    # Returns
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_5"] = df["close"].pct_change(5)

    # RSI
    df["rsi"] = compute_rsi(df["close"], period=14)

    # ====== LABEL ======
    # 1 si la prochaine bougie monte, sinon 0
    df[LABEL_COLUMN] = (df["close"].shift(-1) > df["close"]).astype(int)

    # On enlève les lignes qui ont des NaN (lags, RSI, returns)
    df = df.dropna().reset_index(drop=True)

    # Vérification que toutes les colonnes existent bien
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le dataframe final: {missing}")

    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]

    FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving features to {FEATURES_FILE} ...")
    X.to_parquet(FEATURES_FILE, index=False)

    print(f"Saving labels to {LABELS_FILE} ...")
    y.to_frame(name=LABEL_COLUMN).to_parquet(LABELS_FILE, index=False)

    print("Dataset build completed.")
    print(f"Features shape: {X.shape}, labels shape: {y.shape}")


if __name__ == "__main__":
    main()
