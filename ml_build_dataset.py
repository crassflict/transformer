import os
import pandas as pd
import numpy as np

from features import FEATURE_COLUMNS, LABEL_COLUMN


def find_source_csv() -> str:
    """
    Cherche un fichier de prix à utiliser pour construire le dataset.
    On essaye d'abord mnq_5m.csv, puis nq.csv.
    """
    candidates = [
        "data/mnq_5m.csv",
        "data/nq.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"Using source data: {path}")
            return path

    raise FileNotFoundError(
        f"Aucun fichier de prix trouvé. Cherché: {', '.join(candidates)}"
    )


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    gain_ema = pd.Series(gain, index=series.index).ewm(
        span=period, adjust=False
    ).mean()
    loss_ema = pd.Series(loss, index=series.index).ewm(
        span=period, adjust=False
    ).mean()

    rs = gain_ema / loss_ema
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def main():
    # 1. Charger les données brutes
    csv_path = find_source_csv()
    df = pd.read_csv(csv_path)

    # On suppose au minimum ces colonnes :
    # time, open, high, low, close, volume
    required = ["close", "volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Colonne requise manquante dans le CSV source: {col}")

    # 2. Calcul des indicateurs de base
    df["ema_fast"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=21, adjust=False).mean()

    # Lags de prix
    df["lag_1"] = df["close"].shift(1)
    df["lag_2"] = df["close"].shift(2)
    df["lag_3"] = df["close"].shift(3)

    # Retours (returns)
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_5"] = df["close"].pct_change(5)

    # RSI
    df["rsi"] = compute_rsi(df["close"], period=14)

    # 3. Label: 1 si prochaine bougie monte, 0 sinon
    df[LABEL_COLUMN] = (df["close"].shift(-1) > df["close"]).astype(int)

    # 4. Nettoyage des NaN (échauffement des indicateurs)
    df = df.dropna().reset_index(drop=True)

    # 5. Vérifier qu'on a bien toutes les colonnes demandées
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans le dataframe final: {missing}\n"
            f"Colonnes présentes: {list(df.columns)}"
        )

    print("Colonnes finales dans le dataset :")
    print(df.columns.tolist())

    # 6. Séparer features et labels
    features = df[FEATURE_COLUMNS]
    labels = df[[LABEL_COLUMN]]

    os.makedirs("data", exist_ok=True)

    features_path = "data/dataset_features.parquet"
    labels_path = "data/dataset_labels.parquet"

    print(f"Saving features to {features_path}")
    features.to_parquet(features_path, index=False)

    print(f"Saving labels to {labels_path}")
    labels.to_parquet(labels_path, index=False)

    print("Dataset construit avec succès.")


if __name__ == "__main__":
    main()
