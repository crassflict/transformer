# features.py

# Colonnes utilisées comme entrée pour XGBoost
FEATURE_COLUMNS = [
    "close",
    "volume",
    "ema_fast",
    "ema_slow",
    "lag_1",
    "lag_2",
    "lag_3",
    "ret_1",
    "ret_5",
    "rsi",
]

# Nom de la colonne cible (0 = short/flat, 1 = long)
LABEL_COLUMN = "label"
