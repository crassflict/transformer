import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path

DATA_DIR = Path("data")
FEATURES_FILE = DATA_DIR / "dataset_features.parquet"
LABELS_FILE = DATA_DIR / "dataset_labels.parquet"
MODEL_FILE = Path("model_xgb.json")
REPORT_FILE = Path("xgb_report.txt")

# ======================================================================
# 1. Charger le dataset
# ======================================================================

print("Loading dataset...")
X = pd.read_parquet(FEATURES_FILE)
y = pd.read_parquet(LABELS_FILE)["label"]

# On garde seulement les 4 features utilisées par l'API / C#
FEATURE_COLUMNS = ["close", "ema_fast", "ema_slow", "rsi"]
X = X[FEATURE_COLUMNS].copy()

print(f"Shape X: {X.shape}, y: {y.shape}")
print("Using features:", FEATURE_COLUMNS)

# ======================================================================
# 2. Split train / test
# ======================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False,  # série temporelle
)

# ======================================================================
# 3. Entraîner XGBoost
# ======================================================================

print("Training XGBoost classifier...")

model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    eval_metric="logloss",
    n_jobs=4,
)

model.fit(X_train, y_train)

# ======================================================================
# 4. Évaluation
# ======================================================================

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1       :", f1)

# ======================================================================
# 5. Sauver le modèle + rapport
# ======================================================================

print(f"Saving model to {MODEL_FILE} ...")
model.save_model(MODEL_FILE)

report_lines = [
    "==== XGBoost Evaluation (4 features: close, ema_fast, ema_slow, rsi) ====\n",
    f"test_accuracy: {acc:.4f}\n",
    f"test_precision: {prec:.4f}\n",
    f"test_recall: {rec:.4f}\n",
    f"test_f1: {f1:.4f}\n",
    "\n",
    "Feature importance:\n",
]

importances = model.feature_importances_
for name, imp in zip(FEATURE_COLUMNS, importances):
    report_lines.append(f"{name}: {imp:.5f}\n")

REPORT_FILE.write_text("".join(report_lines), encoding="utf-8")

print("Done. Report written to", REPORT_FILE)
