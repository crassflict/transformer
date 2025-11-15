import pandas as pd
from features import FEATURE_COLUMNS, add_all_features

DATA_FILE = "data/mnq_5m.csv"
FEATURES_FILE = "data/dataset_features.parquet"
LABELS_FILE = "data/dataset_labels.parquet"


def main():
    print(f"Using source data: {DATA_FILE}")

    # 1) Charger le CSV 5 min
    df = pd.read_csv(DATA_FILE)

    # 2) Ajouter toutes les features (EMA, MACD, RSI, lags, ret, delta, Kalman, Savgol, etc.)
    df = add_all_features(df)

    # 3) Afficher les colonnes finales pour contrôle
    print("Colonnes finales dans le dataset :")
    print(list(df.columns))

    # 4) Séparer features et label
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    # 5) Sauvegarder les features
    print(f"Saving features to {FEATURES_FILE}")
    X.to_parquet(FEATURES_FILE, index=False)

    # 6) Sauvegarder les labels (en DataFrame, pas Series)
    print(f"Saving labels to {LABELS_FILE}")
    labels_df = pd.DataFrame({"label": y})
    labels_df.to_parquet(LABELS_FILE, index=False)

    print("Dataset construit avec succès.")


if __name__ == "__main__":
    main()
