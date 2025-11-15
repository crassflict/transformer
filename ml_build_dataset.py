import numpy as np
import pandas as pd
from features import FEATURE_COLUMNS, add_all_features

DATA_FILE = "data/mnq_5m.csv"
FEATURES_FILE = "data/dataset_features.parquet"
LABELS_FILE = "data/dataset_labels.parquet"


def build_labels(df: pd.DataFrame) -> pd.Series:
    """
    Crée la colonne 'label' à partir du close(t+1) / close(t) - 1.
    Label = 1  si futur rendement > quantile 66%
            -1 si futur rendement < quantile 33%
             0 sinon
    """

    # Rendement futur simple, sans pct_change (pas de warning)
    future_ret = df["close"].shift(-1) / df["close"] - 1

    # Seuils
    upper = future_ret.quantile(0.66)
    lower = future_ret.quantile(0.33)

    labels = np.zeros(len(df), dtype=int)
    labels[future_ret > upper] = 1
    labels[future_ret < lower] = -1

    # Dernière ligne n'a pas de futur (NaN), on met label 0 par sécurité
    labels[np.isnan(future_ret.values)] = 0

    return pd.Series(labels, index=df.index, name="label")


def main():
    print(f"Using source data: {DATA_FILE}")

    # 1) Charger le CSV 5 min
    df = pd.read_csv(DATA_FILE)

    # 2) Ajouter toutes les features (EMA, MACD, RSI, lags, ret, delta, Kalman, Savgol, etc.)
    df = add_all_features(df)

    # 3) Créer la colonne 'label'
    df["label"] = build_labels(df)

    # 4) Afficher les colonnes finales pour contrôle
    print("Colonnes finales dans le dataset :")
    print(list(df.columns))

    # 5) Séparer features et label
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    # 6) Sauvegarder les features
    print(f"Saving features to {FEATURES_FILE}")
    X.to_parquet(FEATURES_FILE, index=False)

    # 7) Sauvegarder les labels (en DataFrame, pas Series)
    print(f"Saving labels to {LABELS_FILE}")
    labels_df = pd.DataFrame({"label": y})
    labels_df.to_parquet(LABELS_FILE, index=False)

    print("Dataset construit avec succès.")


if __name__ == "__main__":
    main()
