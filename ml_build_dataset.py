# ml_build_dataset.py

import os
import pandas as pd

from features import FEATURE_COLUMNS, add_all_features

DATA_DIR = "data"
SOURCE_CSV = os.path.join(DATA_DIR, "mnq_5m.csv")

FEATURES_FILE = os.path.join(DATA_DIR, "dataset_features.parquet")
LABELS_FILE = os.path.join(DATA_DIR, "dataset_labels.parquet")


def main():
    print(f"Using source data: {SOURCE_CSV}")

    df = pd.read_csv(SOURCE_CSV)

    # On s'assure des noms de colonnes
    # Attendu: timestamp, open, high, low, close, volume
    df.columns = [c.lower() for c in df.columns]

    # Tri par timestamp si nécessaire
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    # Ajout de toutes les features
    df = add_all_features(df)

    # Création du label (direction du prochain bar)
    future_ret = df["close"].shift(-1).pct_change(0)  # close(t+1)/close(t)-1
    # plus simple et clair:
    future_ret = df["close"].shift(-1) / df["close"] - 1.0

    # label binaire: 1 si prochaine bougie monte, 0 sinon
    df["label"] = (future_ret > 0).astype(int)

    # On enlève les lignes avec NaN (du aux lags/RSI etc. et au label shift)
    df = df.dropna().reset_index(drop=True)

    print("Colonnes finales dans le dataset :")
    print(df.columns.tolist())

    # Séparation features / labels
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    print(f"Saving features to {FEATURES_FILE}")
    X.to_parquet(FEATURES_FILE, index=False)

    print(f"Saving labels to {LABELS_FILE}")
    y.to_parquet(LABELS_FILE, index=False)

    print("Dataset construit avec succès.")


if __name__ == "__main__":
    main()
