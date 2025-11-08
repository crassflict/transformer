import os, json, time
import pandas as pd
from datafeed import get_kraken_ohlc
from strategy import add_indicators, rule_signal, backtest_long_only

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "out")
os.makedirs(OUTPUT_DIR, exist_ok=True)
JSON_PATH = os.path.join(OUTPUT_DIR, "latest.json")

def run_once(pair="XBTUSD", interval=5):
    df = get_kraken_ohlc(pair=pair, interval=interval)
    df = add_indicators(df).dropna().reset_index(drop=True)
    df["signal"] = (df.apply(lambda r: 1 if rule_signal(r)=="buy" else 0, axis=1)).astype(int)
    bt = backtest_long_only(df)
    last = df.iloc[-1]
    action = "buy" if last["signal"] == 1 else "flat"
    payload = {
        "symbol": "BTCUSDT",
        "action": action,
        "qty": 0.001,
        "ts": int(time.time()),
        "price": float(last["close"])
    }
    with open(JSON_PATH, "w") as f:
        json.dump(payload, f)
    # Sauvegarde un mini rapport lisible dans les artefacts CI
    rep = bt.tail(1)[["equity"]].copy()
    rep["price"] = last["close"]
    rep["action"] = action
    rep.to_csv(os.path.join(OUTPUT_DIR, "report.csv"), index=False)
    return payload

if __name__ == "__main__":
    sig = run_once()
    print("Signal:", sig, "-> écrit dans", JSON_PATH)
