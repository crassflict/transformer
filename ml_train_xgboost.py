# ml_train_xgboost.py

import os
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from xgboost import XGBClassifier

from features import FEATURE_COLUMNS

DATA_DIR = "data"
FEATURES_FILE = os.path.join(DATA_DIR, "dataset_features.parquet")
LABELS_FILE = os.path.join(DATA_DIR, "dataset_labels.parquet")

MODEL_FILE = os.path.join(DATA_DIR, "model_xgb.json")
REPORT_FILE = os.path.join(DATA_DIR, "xgb_report.txt")


def main():
    print(f"Loading features from {FEATURES_FILE} ...")
    X_df = pd.read_parquet(FEATURES_FILE)

    print(f"Loading labels from {LABELS_FILE} ...")
    y_df = pd.read_parquet(LABELS_FILE)

    # Sanity check
    missing = [c for c in FEATURE_COLUMNS if c not in X_df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans X_df: {missing}")

    X = X_df[FEATURE_COLUMNS].values
    y = y_df.values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=4,
    )

    print("Training XGBoost...")
    model.fit(X_train, y_train)

    # Évaluation
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("==== XGBoost Evaluation (features Option B) ====")
    print(f"test_accuracy: {acc:.4f}")
    print(f"test_precision: {prec:.4f}")
    print(f"test_recall: {rec:.4f}")
    print(f"test_f1: {f1:.4f}")

    # Sauvegarde du modèle
    print(f"Saving model to {MODEL_FILE}")
    model.save_model(MODEL_FILE)

    # Rapport texte
    report = {
        "test_accuracy": acc,
        "test_precision": prec,
        "test_recall": rec,
        "test_f1": f1,
        "feature_columns": FEATURE_COLUMNS,
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("==== XGBoost Evaluation (Option B features) ====\n")
        for k, v in report.items():
            f.write(f"{k}: {v}\n")

    print(f"Report written to {REPORT_FILE}")


if __name__ == "__main__":
    main()
