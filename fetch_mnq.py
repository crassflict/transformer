# fetch_mnq.py — Télécharge MNQ=F (Micro E-mini Nasdaq 100) sur Yahoo Finance
# Sauvegarde les données en 5 minutes dans data/mnq_5m.csv

import os, sys
import pandas as pd

# Essaye d'importer yfinance, sinon échoue proprement
try:
    import yfinance as yf
except Exception:
    print("Missing yfinance. Install it before running this script.", file=sys.stderr)
    sys.exit(1)

# --- Contourne les restrictions de Yahoo (serveurs GitHub bloqués parfois)
from urllib import request
request.install_opener(request.build_opener(request.ProxyHandler({})))
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

# --- Préparation du dossier data
os.makedirs("data", exist_ok=True)

print("📡 Téléchargement des vraies données MNQ=F (5 minutes / 60 jours)...")

# Télécharge les 60 derniers jours de données MNQ=F en intervalle 5m
ticker = yf.Ticker("MNQ=F")
df = ticker.history(period="60d", interval="5m", auto_adjust=False)

# Vérification de la validité
if df is None or df.empty:
    print("⚠️ ERREUR: Yahoo n'a retourné aucune donnée. (Blocage temporaire ou rate limit)")
    sys.exit(2)

# --- Nettoyage et normalisation
df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).copy()
df.reset_index(inplace=True)

# Cherche la colonne temps automatiquement
time_col = None
for c in ("Datetime", "Date", "index"):
    if c in df.columns:
        time_col = c
        break
if time_col is None:
    # fallback: prend la première non numérique
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            time_col = c
            break
if time_col is None:
    print("❌ ERREUR: Impossible de détecter la colonne timestamp.", file=sys.stderr)
    sys.exit(3)

# Renomme pour correspondre au format du bot
df.rename(columns={
    time_col: "timestamp",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}, inplace=True)

# Réorganise les colonnes
out_cols = ["timestamp", "open", "high", "low", "close", "volume"]
df = df[out_cols]

# --- Sauvegarde CSV
out_path = "data/mnq_5m.csv"
df.to_csv(out_path, index=False)
print(f"✅ Fichier écrit: {out_path} ({len(df)} lignes)")

# --- Aperçu
print(df.head().to_string(index=False))
