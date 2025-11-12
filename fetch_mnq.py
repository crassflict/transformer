# fetch_mnq.py — télécharge MNQ=F en 5 minutes et écrit data/mnq_5m.csv
import pandas as pd, os, sys
try:
    import yfinance as yf
except Exception as e:
    print("Missing yfinance. Install in workflow step.")
    sys.exit(1)

os.makedirs("data", exist_ok=True)
ticker = yf.Ticker("MNQ=F")  # Micro E-mini Nasdaq-100
# 60j de 5 minutes (limite gratuite Yahoo)
df = ticker.history(period="60d", interval="5m", auto_adjust=False)
df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
df = df.dropna(subset=["open","high","low","close","volume"]).copy()
df.reset_index(names="timestamp", inplace=True)
# Harmonise les colonnes pour bot.py
# (timestamp, open, high, low, close, volume)
out_cols = ["timestamp","open","high","low","close","volume"]
df = df[out_cols]
df.to_csv("data/mnq_5m.csv", index=False)
print(f"Wrote data/mnq_5m.csv with {len(df)} rows")
