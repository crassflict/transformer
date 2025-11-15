import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from features import FEATURE_COLUMNS

FEATURES_FILE = "data/dataset_features.parquet"
LABELS_FILE = "data/dataset_labels.parquet"
MODEL_FILE = "data/model_xgb.json"
REPORT_FILE = "data/xgb_report.txt"


def main():
    print(f"Loading features from {FEATURES_FILE} ...")
    X_df = pd.read_parquet(FEATURES_FILE)

    print(f"Loading labels from {LABELS_FILE} ...")
    y_df = pd.read_parquet(LABELS_FILE)

    # --- Vérifier les colonnes ---
    missing_features = [c for c in FEATURE_COLUMNS if c not in X_df.columns]
    if missing_features:
        raise ValueError(f"Colonnes manquantes dans X_df: {missing_features}")

    if "label" not in y_df.columns:
        raise ValueError("La colonne 'label' est manquante dans dataset_labels.parquet")

    # X : features, y_raw : labels -1 / 0 / 1
    X = X_df[FEATURE_COLUMNS].values
    y_raw = y_df["label"].astype(int).values

    # Mapping -1,0,1  -->  0,1,2 pour XGBoost multi-class
    # -1 (short) -> 0
    #  0 (flat)  -> 1
    #  1 (long)  -> 2
    y = (y_raw + 1).astype(int)

    print("Répartition des labels bruts (avant mapping) :")
    unique_raw, counts_raw = np.unique(y_raw, return_counts=True)
    for cls, cnt in zip(unique_raw, counts_raw):
        print(f"  label {cls}: {cnt}")

    print("Répartition des labels après mapping (0=short, 1=flat, 2=long) :")
    unique_m, counts_m = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique_m, counts_m):
        print(f"  classe {cls}: {cnt}")

    # --- Split train / test ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Modèle XGBoost multi-class ---
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
    )

    print("Training XGBoost...")
    model.fit(X_train, y_train)

    # --- Evaluation ---
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred, digits=4)

    print("\n==== XGBoost Evaluation (multi-class 0=short,1=flat,2=long) ====")
    print(f"test_accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(cls_report)

    # --- Sauvegarde du modèle ---
    print(f"Saving model to {MODEL_FILE}")
    Path(MODEL_FILE).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_FILE)

    # --- Sauvegarde du rapport texte ---
    print(f"Saving report to {REPORT_FILE}")
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("==== XGBoost Evaluation (multi-class 0=short,1=flat,2=long) ====\n")
        f.write(f"test_accuracy: {acc:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(cls_report)

        f.write("\n\nFeature importance:\n")
        importances = model.feature_importances_
        for name, imp in sorted(
            zip(FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True
        ):
            f.write(f"{name}: {imp:.5f}\n")

    print("Training finished successfully.")


if __name__ == "__main__":
    main()
