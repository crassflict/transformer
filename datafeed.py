import requests
import pandas as pd

def get_kraken_ohlc(pair="XBTUSD", interval=5, limit=1500) -> pd.DataFrame:
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}"
    r = requests.get(url, timeout=20)
    j = r.json()
    if j.get("error"):
        raise RuntimeError(j["error"])
    key = [k for k in j["result"].keys() if k != "last"][0]
    cols = ["time","open","high","low","close","vwap","volume","count"]
    df = pd.DataFrame(j["result"][key], columns=cols)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.astype({"open":float,"high":float,"low":float,"close":float,"volume":float})
    df = df[["time","open","high","low","close","volume"]].drop_duplicates("time").reset_index(drop=True)
    if len(df) > limit:
        df = df.tail(limit).reset_index(drop=True)
    return df
