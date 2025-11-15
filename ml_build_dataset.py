import json
from pathlib import Path

import pandas as pd

from features import add_features, FEATURE_COLUMNS, LABEL_COLUMN

DATA_DIR = Path("data")
RAW_CSV = DATA_DIR / "mnq_5m.csv"
OUT_FEATURES = DATA_DIR / "dataset_features.parquet"
OUT_LABELS = DATA_DIR / "dataset_labels.parquet"
OUT_FEATURE_LIST = DATA_DIR / "feature_list.json"


def load_raw_data(path: Path) -> pd.DataFrame:
    print(f"Loading raw data from {path}...")
    df = pd.read_csv(path)

    for col in ["timestamp", "time", "datetime", "DateTime", "date"]:
        if col in df.columns:
            df["timestamp"] = pd.to_datetime(df[col])
            break

    if "close" not in df.columns:
        for c in ["Close", "CLOSE", "ClosePrice", "last"]:
            if c in df.columns:
                df["close"] = df[c]
                break

    if "close" not in df.columns:
        raise ValueError("Impossible de trouver une colonne 'close' dans le CSV.")

    if "volume" not in df.columns:
        for c in ["Volume", "VOL", "vol"]:
            if c in df.columns:
                df["volume"] = df[c]
                break

    if "volume" not in df.columns:
        print("Avertissement : pas de colonne volume trouvée, remplie à 0.")
        df["volume"] = 0.0

    return df


def main() -> None:
    df_raw = load_raw_data(RAW_CSV)

    print("Building features with Option B (Kalman, Savgol, etc.)...")
    df_feat = add_features(df_raw)

    print(f"Resulting dataset shape: {df_feat.shape}")
    print(df_feat.head())

    features = df_feat[FEATURE_COLUMNS].copy()
    labels = df_feat[[LABEL_COLUMN]].copy()

    print(f"Saving features to {OUT_FEATURES} ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUT_FEATURES, index=False)

    print(f"Saving labels to {OUT_LABELS} ...")
    labels.to_parquet(OUT_LABELS, index=False)

    print(f"Saving feature list to {OUT_FEATURE_LIST} ...")
    feature_info = {
        "features": FEATURE_COLUMNS,
        "label": LABEL_COLUMN,
    }
    with OUT_FEATURE_LIST.open("w", encoding="utf-8") as f:
        json.dump(feature_info, f, indent=2)

    print("Dataset build DONE.")


if __name__ == "__main__":
    main()
