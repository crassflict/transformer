from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from features import FEATURE_COLUMNS, LABEL_COLUMN

DATA_DIR = Path("data")
FEATURES_FILE = DATA_DIR / "dataset_features.parquet"
LABELS_FILE = DATA_DIR / "dataset_labels.parquet"
MODEL_FILE = Path("model_xgb.json")
REPORT_FILE = Path("xgb_report.txt")


def main() -> None:
    print(f"Loading features from {FEATURES_FILE} ...")
    X_df = pd.read_parquet(FEATURES_FILE)
    print(f"Loading labels from {LABELS_FILE} ...")
    y_df = pd.read_parquet(LABELS_FILE)

    X = X_df[FEATURE_COLUMNS].values
    y = y_df[LABEL_COLUMN].values.astype(int)

    print(f"Dataset shape: X={X.shape}, y={y.shape}, positive ratio={y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        shuffle=False,
    )

    model = XGBClassifier(
        max_depth=4,
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
    )

    print("Training XGBoost...")
    model.fit(X_train, y_train)

    def eval_split(Xp, yp, name: str) -> str:
        y_pred_proba = model.predict_proba(Xp)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        acc = accuracy_score(yp, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            yp, y_pred, average="binary", zero_division=0
        )

        rep = classification_report(yp, y_pred, digits=4)
        txt = []
        txt.append(f"=== {name} ===")
        txt.append(f"accuracy: {acc:.4f}")
        txt.append(f"precision: {prec:.4f}")
        txt.append(f"recall: {rec:.4f}")
        txt.append(f"f1: {f1:.4f}")
        txt.append("")
        txt.append("Classification report:")
        txt.append(rep)
        txt.append("")
        return "\n".join(txt)

    report_train = eval_split(X_train, y_train, "TRAIN")
    report_test = eval_split(X_test, y_test, "TEST")

    full_report_lines = []
    full_report_lines.append("==== XGBoost Evaluation (Option B full features) ====")
    full_report_lines.append("")
    full_report_lines.append(report_train)
    full_report_lines.append(report_test)
    full_report_lines.append("Feature order:")
    for i, name in enumerate(FEATURE_COLUMNS):
        full_report_lines.append(f"{i:2d}: {name}")

    full_report = "\n".join(full_report_lines)
    print(full_report)

    importance = model.feature_importances_
    imp_lines = ["\nFeature importance:"]
    for name, val in sorted(
        zip(FEATURE_COLUMNS, importance), key=lambda t: t[1], reverse=True
    ):
        imp_lines.append(f"{name}: {val:.5f}")
    imp_txt = "\n".join(imp_lines)
    print(imp_txt)

    full_report += imp_txt + "\n"

    print(f"Saving model to {MODEL_FILE} ...")
    model.save_model(MODEL_FILE)

    print(f"Saving report to {REPORT_FILE} ...")
    with REPORT_FILE.open("w", encoding="utf-8") as f:
        f.write(full_report)

    print("Training DONE.")


if __name__ == "__main__":
    main()
