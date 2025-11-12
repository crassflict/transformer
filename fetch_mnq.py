# fetch_mnq.py — Télécharge MNQ=F (Micro E-mini Nasdaq 100) sur Yahoo Finance
# Crée data/mnq_5m.csv pour le bot.

import os, sys, time, json
import pandas as pd

# Sécurise l'import de yfinance
try:
    import yfinance as yf
except Exception as e:
    print("❌ yfinance manquant, installer avec pip install yfinance", file=sys.stderr)
    sys.exit(1)

# Contourne restrictions et caches
from urllib import request
request.install_opener(request.build_opener(request.ProxyHandler({})))
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

os.makedirs("data", exist_ok=True)

symbol = "MNQ=F"
interval = "5m"
period = "60d"
out_path = "data/mnq_5m.csv"

print(f"📡 Téléchargement Yahoo Finance: {symbol} ({interval}, {period})")

success = False
for attempt in range(1, 4):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=False, prepost=False)
        if df is not None and not df.empty:
            success = True
            break
        print(f"⚠️ Tentative {attempt}: aucune donnée reçue, retry dans 5s…")
        time.sleep(5)
    except Exception as e:
        print(f"⚠️ Erreur tentative {attempt}: {e}")
        time.sleep(5)

if not success:
    print("🚫 Yahoo Finance a refusé ou renvoyé un dataset vide.")
    sys.exit(2)

# Nettoyage
df = df.dropna(subset=["Open","High","Low","Close","Volume"]).copy()
df.reset_index(inplace=True)

time_col = None
for c in ("Datetime","Date","index"):
    if c in df.columns:
        time_col = c
        break
if time_col is None:
    print("❌ Impossible de détecter la colonne de temps", file=sys.stderr)
    sys.exit(3)

df.rename(columns={
    time_col:"timestamp",
    "Open":"open",
    "High":"high",
    "Low":"low",
    "Close":"close",
    "Volume":"volume"
}, inplace=True)

df = df[["timestamp","open","high","low","close","volume"]]

df.to_csv(out_path, index=False)
print(f"✅ Fichier écrit: {out_path} ({len(df)} lignes)")
print(df.head(3).to_string(index=False))

# Petit JSON résumé (utile dans les logs ou le README)
meta = {
    "symbol": symbol,
    "rows": len(df),
    "first": str(df['timestamp'].iloc[0]),
    "last": str(df['timestamp'].iloc[-1])
}
with open("data/last_fetch.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"📘 Résumé sauvegardé -> data/last_fetch.json")
