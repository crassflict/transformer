import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib
import os

FEATURES_FILE = "data/dataset_features.parquet"
LABELS_FILE = "data/dataset_labels.parquet"
MODEL_FILE = "data/model_xgb.json"

def train():
    X = pd.read_parquet(FEATURES_FILE)
    y = pd.read_parquet(LABELS_FILE)['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    # Save model
    model.save_model(MODEL_FILE)

    print("Training complete. Model saved →", MODEL_FILE)

if __name__ == "__main__":
    train()
