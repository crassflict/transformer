# out/ml_build_dataset.py
"""
Construit le dataset d'entraînement à partir des données brutes
(Option B : toutes les features sont calculées côté Python).

- lit data/mnq_5m.csv
- applique features.add_features
- sauvegarde data/dataset_features.parquet
- sauvegarde data/feature_list.json (pour info)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from features import add_features, FEATURE_COLUMNS, LABEL_COLUMN

# chemins
DATA_DIR = Path("data")
RAW_CSV = DATA_DIR / "mnq_5m.csv"         # adapte le nom si besoin
OUT_PARQUET = DATA_DIR / "dataset_features.parquet"
OUT_FEATURE_LIST = DATA_DIR / "feature_list.json"


def load_raw_data(path: Path) -> pd.DataFrame:
    print(f"Loading raw data from {path}...")
    df = pd.read_csv(path)

    # on essaye de détecter une colonne de temps
    for col in ["timestamp", "time", "datetime", "DateTime", "date"]:
        if col in df.columns:
            df["timestamp"] = pd.to_datetime(df[col])
            break

    # on renomme la colonne de close si besoin
    if "close" not in df.columns:
        for c in ["Close", "CLOSE", "ClosePrice", "last"]:
            if c in df.columns:
                df["close"] = df[c]
                break

    if "close" not in df.columns:
        raise ValueError("Impossible de trouver une colonne 'close' dans le CSV.")

    # idem pour volume
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

    print("Building features...")
    df_feat = add_features(df_raw)

    print(f"Resulting dataset shape: {df_feat.shape}")
    print(df_feat.head())

    # sauvegarde parquet (features + label)
    print(f"Saving features parquet to {OUT_PARQUET} ...")
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_parquet(OUT_PARQUET, index=False)

    # sauvegarde liste de features pour info / logs
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
