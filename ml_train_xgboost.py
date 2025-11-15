# ml_train_xgboost.py

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from features import FEATURE_COLUMNS, LABEL_COLUMN

FEATURES_FILE = Path("data/dataset_features.parquet")
LABELS_FILE = Path("data/dataset_labels.parquet")
MODEL_FILE = Path("data/model_xgb.json")
REPORT_FILE = Path("data/xgb_report.txt")


def main():
    print(f"Loading features from {FEATURES_FILE} ...")
    X_df = pd.read_parquet(FEATURES_FILE)

    print(f"Loading labels from {LABELS_FILE} ...")
    y_df = pd.read_parquet(LABELS_FILE)

    # Vérifie que toutes les colonnes sont là
    missing = [c for c in FEATURE_COLUMNS if c not in X_df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans X_df: {missing}")

    X = X_df[FEATURE_COLUMNS].values.astype(np.float32)
    y = y_df[LABEL_COLUMN].values.astype(int)

    print(f"Dataset shapes: X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
    )

    print("Training XGBoost model ...")
    model.fit(X_train, y_train)

    print("Evaluating model ...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("==== XGBoost Evaluation ====")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"test_accuracy: {acc:.4f}")
    print(f"test_precision: {prec:.4f}")
    print(f"test_recall: {rec:.4f}")
    print(f"test_f1: {f1:.4f}")

    # Importances
    importances = model.feature_importances_
    feat_imp = sorted(
        zip(FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True
    )

    # Sauvegarde du modèle
    FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving model to {MODEL_FILE} ...")
    model.save_model(MODEL_FILE)

    # Sauvegarde d’un petit rapport texte
    print(f"Saving report to {REPORT_FILE} ...")
    with open(REPORT_FILE, "w") as f:
        f.write("==== XGBoost Evaluation ====\n")
        f.write(f"Features: {FEATURE_COLUMNS}\n")
        f.write(f"test_accuracy: {acc:.4f}\n")
        f.write(f"test_precision: {prec:.4f}\n")
        f.write(f"test_recall: {rec:.4f}\n")
        f.write(f"test_f1: {f1:.4f}\n\n")

        f.write("Feature importance:\n")
        for name, val in feat_imp:
            f.write(f"{name}: {val:.5f}\n")

    print("Training script completed.")


if __name__ == "__main__":
    main()
