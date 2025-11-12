# fetch_mnq.py — Télécharge MNQ=F (5m) via Yahoo Finance -> data/mnq_5m.csv
import os, sys
import pandas as pd

try:
    import yfinance as yf
except Exception:
    print("Missing yfinance. It will be installed by the workflow.", file=sys.stderr)
    sys.exit(1)

os.makedirs("data", exist_ok=True)

# 60 jours en 5 minutes (limite gratuite)
ticker = yf.Ticker("MNQ=F")
df = ticker.history(period="60d", interval="5m", auto_adjust=False)

if df is None or df.empty:
    print("ERROR: Empty dataframe from Yahoo (rate limit or symbol).", file=sys.stderr)
    sys.exit(2)

# Nettoyage + normalisation
df = df.dropna(subset=["Open","High","Low","Close","Volume"]).copy()
# Réinitialise l'index (compat pandas toutes versions)
df.reset_index(inplace=True)

# La colonne temps peut s'appeler 'Datetime' ou 'Date' selon la version
time_col = None
for c in ("Datetime", "Date", "index"):
    if c in df.columns:
        time_col = c
        break
if time_col is None:
    # fallback: premier champ non numérique
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            time_col = c
            break
if time_col is None:
    print("ERROR: Could not detect timestamp column.", file=sys.stderr)
    sys.exit(3)

df.rename(columns={
    time_col: "timestamp",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}, inplace=True)

out_cols = ["timestamp","open","high","low","close","volume"]
df = df[out_cols]

# Sauvegarde
out_path = "data/mnq_5m.csv"
df.to_csv(out_path, index=False)
print(f"Wrote {out_path} rows={len(df)}")
