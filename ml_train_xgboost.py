import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# chemins d'entrée / sortie
FEATURES_FILE = "data/dataset_features.parquet"
LABELS_FILE = "data/dataset_labels.parquet"

MODEL_FILE = "data/model_xgb.json"
FEATURES_META_FILE = "data/feature_list.json"
REPORT_FILE = "data/xgb_report.txt"


def load_data():
    print(f"Loading features from: {FEATURES_FILE}")
    X = pd.read_parquet(FEATURES_FILE)

    print(f"Loading labels from: {LABELS_FILE}")
    y = pd.read_parquet(LABELS_FILE)["target"]

    print(f"Shapes -> X: {X.shape}, y: {y.shape}")
    return X, y


def train_test_split_time_series(X, y, test_ratio=0.2):
    n = len(X)
    split_idx = int(n * (1 - test_ratio))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test


def build_model():
    # hyperparamètres raisonnables pour commencer
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=3,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=4,
        tree_method="hist",  # plus rapide sur CPU
    )
    return model


def evaluate_and_log(model, X_train, y_train, X_test, y_test, feature_names):
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)

    y_train_pred = (model.predict_proba(X_train)[:, 1] > 0.5).astype(int)
    y_test_pred = (model.predict_proba(X_test)[:, 1] > 0.5).astype(int)

    metrics = {}

    metrics["train_accuracy"] = float(accuracy_score(y_train, y_train_pred))
    metrics["test_accuracy"] = float(accuracy_score(y_test, y_test_pred))
    metrics["test_precision"] = float(precision_score(y_test, y_test_pred, zero_division=0))
    metrics["test_recall"] = float(recall_score(y_test, y_test_pred, zero_division=0))
    metrics["test_f1"] = float(f1_score(y_test, y_test_pred, zero_division=0))

    report = classification_report(y_test, y_test_pred, zero_division=0)

    print("==== XGBoost Evaluation ====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("\nClassification report:\n", report)

    # importance des features
    importance = model.feature_importances_
    feat_importance = sorted(
        zip(feature_names, importance), key=lambda t: t[1], reverse=True
    )

    # écriture dans un rapport texte
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("==== XGBoost Evaluation ====\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\nClassification report:\n")
        f.write(report)
        f.write("\n\nFeature importance:\n")
        for name, imp in feat_importance:
            f.write(f"{name}: {imp:.5f}\n")

    print(f"\nReport written to: {REPORT_FILE}")


def save_model_and_metadata(model, feature_names):
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)

    # modèle XGBoost en JSON (format natif XGBoost)
    model.save_model(MODEL_FILE)
    print(f"Model saved to: {MODEL_FILE}")

    # liste des features pour Quantower / future debug
    with open(FEATURES_META_FILE, "w", encoding="utf-8") as f:
        json.dump({"features": feature_names}, f, indent=2)

    print(f"Feature list saved to: {FEATURES_META_FILE}")


def main():
    X, y = load_data()
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split_time_series(X, y, test_ratio=0.2)

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    model = build_model()

    print("Training XGBoost model...")
    model.fit(X_train, y_train)

    print("Training done. Evaluating...")
    evaluate_and_log(model, X_train, y_train, X_test, y_test, feature_names)

    save_model_and_metadata(model, feature_names)


if __name__ == "__main__":
    main()
