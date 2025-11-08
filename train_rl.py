import os, time, json
import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_env import FuturesEnv

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "out")
MODEL_DIR  = os.environ.get("MODEL_DIR",  "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "ppo_mnq.zip")
JSON_PATH  = os.path.join(OUTPUT_DIR, "latest.json")
REPORT_CSV = os.path.join(OUTPUT_DIR, "report.csv")

def load_data(ticker="NQ=F", period="6mo", interval="1h") -> pd.DataFrame:
    # NQ=F = E-mini Nasdaq continuous; MNQ suit la même direction
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    df = df[["Open","High","Low","Close"]].dropna()
    df.columns = ["open","high","low","close"]
    df = df.reset_index().rename(columns={"Datetime":"time"})
    return df

def split_train_test(df, test_ratio=0.2):
    n = len(df)
    k = max(int(n * (1 - test_ratio)), 100)
    return df.iloc[:k].reset_index(drop=True), df.iloc[k:].reset_index(drop=True)

def train_and_signal():
    df = load_data()
    train, test = split_train_test(df, 0.2)

    # Entraînement rapide pour CI (augmenter plus tard)
    env = DummyVecEnv([lambda: FuturesEnv(train, fee=0.00005, window=32)])
    model = PPO("MlpPolicy", env, verbose=0, n_steps=256, batch_size=256, learning_rate=3e-4)
    model.learn(total_timesteps=20_000)
    model.save(MODEL_PATH)

    # Évalue sur 'test' et produit un signal sur la dernière bougie
    env_test = DummyVecEnv([lambda: FuturesEnv(test, fee=0.00005, window=32)])
    obs = env_test.reset()
    # déroule vite fait pour placer la politique dans le contexte de test
    for _ in range(len(test)-40):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, = env_test.step(action)
        if done: break

    # Dernier état -> action actuelle
    action, _ = model.predict(obs, deterministic=True)
    action_map = {0:"sell", 1:"flat", 2:"buy"}
    act = action_map[int(action)]

    last_price = float(test["close"].iloc[-1])
    payload = {
        "symbol": "MNQ",           # adapte au symbole de ton broker dans Quantower
        "action": act,
        "qty": 1,                  # micro-contrat: 1 = 1x MNQ
        "ts": int(time.time()),
        "price": last_price
    }
    with open(JSON_PATH, "w") as f:
        json.dump(payload, f)

    # Mini rapport
    pd.DataFrame([{"last_price": last_price, "action": act}]).to_csv(REPORT_CSV, index=False)
    return payload

if __name__ == "__main__":
    sig = train_and_signal()
    print("Signal:", sig)
