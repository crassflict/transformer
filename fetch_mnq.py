# fetch_mnq.py — Télécharge MNQ=F (Micro E-mini Nasdaq-100) en 5m via l'API JSON Yahoo directe
# Écrit data/mnq_5m.csv sans dépendances externes (stdlib only).

import os, sys, time, json, math
from urllib import request, error
from datetime import datetime, timezone

OUT_DIR = "data"
OUT_CSV = os.path.join(OUT_DIR, "mnq_5m.csv")

# Yahoo chart API (même source que yfinance, mais sans la lib)
URL = "https://query1.finance.yahoo.com/v8/finance/chart/MNQ=F?range=60d&interval=5m&includePrePost=false"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/plain, */*",
    "Connection": "keep-alive",
}

def fetch_json(url: str, retries: int = 3, sleep_s: int = 5):
    req = request.Request(url, headers=HEADERS)
    last_err = None
    for i in range(1, retries + 1):
        try:
            with request.urlopen(req, timeout=20) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")
                data = resp.read()
                return json.loads(data.decode("utf-8"))
        except Exception as e:
            last_err = e
            print(f"⚠️ tentative {i}/{retries} échouée: {e}", file=sys.stderr)
            time.sleep(sleep_s)
    raise RuntimeError(f"Échec de téléchargement après {retries} tentatives: {last_err}")

def to_csv(records):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write("timestamp,open,high,low,close,volume\n")
        for r in records:
            f.write("{},{:.2f},{:.2f},{:.2f},{:.2f},{}\n".format(
                r["ts"], r["open"], r["high"], r["low"], r["close"], r["volume"]
            ))

def main():
    print("📡 Yahoo JSON direct (MNQ=F, 5m, 60d)")
    j = fetch_json(URL)

    # structure: chart -> result[0] -> timestamp[], indicators -> quote[0] -> open/high/low/close/volume arrays
    try:
        res = j["chart"]["result"][0]
        ts_list = res["timestamp"]
        quote = res["indicators"]["quote"][0]
        opens  = quote["open"]
        highs  = quote["high"]
        lows   = quote["low"]
        closes = quote["close"]
        vols   = quote["volume"]
    except Exception as e:
        print("❌ Format de réponse inattendu:", e, file=sys.stderr)
        sys.exit(2)

    if not ts_list or not opens:
        print("❌ Données vides dans la réponse Yahoo", file=sys.stderr)
        sys.exit(3)

    # Assemble records, filtre NaN/None
    def is_num(x): 
        return x is not None and not (isinstance(x, float) and math.isnan(x))
    records = []
    for i in range(min(len(ts_list), len(opens), len(highs), len(lows), len(closes), len(vols))):
        o, h, l, c, v = opens[i], highs[i], lows[i], closes[i], vols[i]
        if not (is_num(o) and is_num(h) and is_num(l) and is_num(c) and v is not None):
            continue
        # timestamps Yahoo en epoch seconds UTC
        ts = datetime.fromtimestamp(ts_list[i], tz=timezone.utc).isoformat()
        records.append({
            "ts": ts,
            "open": float(o),
            "high": float(h),
            "low": float(l),
            "close": float(c),
            "volume": int(v),
        })

    if not records:
        print("❌ Aucun enregistrement utilisable après nettoyage", file=sys.stderr)
        sys.exit(4)

    to_csv(records)
    print(f"✅ Écrit {OUT_CSV} ({len(records)} lignes)")
    # petit aperçu
    for r in records[:3]:
        print(r)

if __name__ == "__main__":
    main()
