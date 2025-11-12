# bot.py
# Nasdaq futures adaptive bot — apprend entre les runs et produit un rapport complet

import csv, os, random, math, json
from datetime import datetime, timedelta, timezone

STATE_FILE = "state.json"

# --- Mémoire persistante entre les runs ---
def load_state(bot):
    if os.path.isfile(STATE_FILE):
        try:
            s = json.load(open(STATE_FILE))
            bot.ema_fast_p = int(s.get("ema_fast_p", bot.ema_fast_p))
            bot.ema_slow_p = int(s.get("ema_slow_p", bot.ema_slow_p))
            bot.rsi_buy   = float(s.get("rsi_buy", bot.rsi_buy))
            bot.rsi_sell  = float(s.get("rsi_sell", bot.rsi_sell))
            print(f"✅ Loaded previous state: {s}")
        except Exception as e:
            print(f"⚠️ Could not load state: {e}")

def save_state(bot):
    s = {
        "ema_fast_p": bot.ema_fast_p,
        "ema_slow_p": bot.ema_slow_p,
        "rsi_buy": bot.rsi_buy,
        "rsi_sell": bot.rsi_sell,
    }
    json.dump(s, open(STATE_FILE, "w"))
    print(f"💾 Saved state: {s}")

# --- Paramètres du contrat (MNQ) ---
TICK_SIZE = 0.25
TICK_VALUE = 5.0
COMM_PER_SIDE = 2.5
SLIPPAGE_TICKS = 0.5

# --- Utilitaires mathématiques ---
def points_to_usd(points):
    ticks = points / TICK_SIZE
    return ticks * TICK_VALUE

def ema(prev, price, period):
    if period <= 1: return price
    alpha = 2.0/(period+1.0)
    return alpha*price + (1-alpha)*(prev if prev is not None else price)

def rsi_wilder(gains_prev, losses_prev, change, period):
    gain = max(change, 0.0)
    loss = max(-change, 0.0)
    if gains_prev is None or losses_prev is None:
        return gain, loss, 50.0
    avg_gain = (gains_prev*(period-1) + gain)/period
    avg_loss = (losses_prev*(period-1) + loss)/period
    rs = (avg_gain / avg_loss) if avg_loss > 0 else 999.0
    rsi = 100.0 - (100.0/(1.0+rs))
    return avg_gain, avg_loss, rsi

def load_csv(path):
    out = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ts = row["timestamp"]
            if ts.isdigit():
                ts = datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
            o = float(row["open"]); h = float(row["high"]); l = float(row["low"]); c = float(row["close"]); v = float(row["volume"])
            out.append((ts,o,h,l,c,v))
    return out

def ensure_data(path="data/nq.csv", minutes=60*24*30):
    """Charge MNQ=F si dispo, sinon génère des données synthétiques."""
    os.makedirs("data", exist_ok=True)
    if os.path.isfile("data/mnq_5m.csv"):
        print("✅ Found real MNQ=F data (data/mnq_5m.csv)")
        return load_csv("data/mnq_5m.csv")

    print("⚠️ No real data found — generating synthetic dataset.")
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    price = 16000.0
    rows = []
    for i in range(minutes):
        ts = start + timedelta(minutes=i)
        drift = 0.05 * math.sin(i/800.0)
        shock = random.gauss(0, 2.0)
        close = max(1000.0, price + drift + shock)
        high = max(price, close) + random.random()*0.8
        low  = min(price, close) - random.random()*0.8
        open_ = price
        vol = random.randint(10, 1000)
        rows.append((ts.isoformat(), open_, high, low, close, vol))
        price = close
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp","open","high","low","close","volume"])
        w.writerows(rows)
    return rows

# --- Classe principale du bot ---
class Bot:
    def __init__(self, ema_fast=21, ema_slow=55, rsi_len=14, rsi_buy=52.0, rsi_sell=48.0):
        self.ema_fast_p = ema_fast
        self.ema_slow_p = ema_slow
        self.rsi_len = rsi_len
        self.rsi_buy = rsi_buy
        self.rsi_sell = rsi_sell
        self.reset_state()

    def reset_state(self):
        self.e_fast = None
        self.e_slow = None
        self.prev_close = None
        self.rsi_gain = None
        self.rsi_loss = None
        self.rsi_val = 50.0
        self.position = 0
        self.entry = None
        self.cash = 0.0
        self.trades = 0
        self.wins = 0

    def decide(self, close):
        go_long  = (self.e_fast and self.e_slow and self.e_fast > self.e_slow and self.rsi_val >= self.rsi_buy)
        go_short = (self.e_fast and self.e_slow and self.e_fast < self.e_slow and self.rsi_val <= self.rsi_sell)
        if go_long and self.position <= 0: return 1
        if go_short and self.position >= 0: return -1
        return 0

    def fill_price(self, px, side):
        slip = SLIPPAGE_TICKS * TICK_SIZE
        if side == 1:   return px + slip
        if side == -1:  return px - slip
        if self.position == 1: return px - slip
        if self.position == -1: return px + slip
        return px

    def step(self, ts, close):
        self.e_fast = ema(self.e_fast, close, self.ema_fast_p)
        self.e_slow = ema(self.e_slow, close, self.ema_slow_p)
        if self.prev_close is not None:
            ch = close - self.prev_close
            self.rsi_gain, self.rsi_loss, self.rsi_val = rsi_wilder(self.rsi_gain, self.rsi_loss, ch, self.rsi_len)
        self.prev_close = close
        action = self.decide(close)
        if action == 0: return

        if self.position != 0 and action != self.position:
            exit_px = self.fill_price(close, 0)
            signed = 1 if self.position == 1 else -1
            pnl_points = (exit_px - self.entry) * signed
            pnl_usd = points_to_usd(pnl_points) - COMM_PER_SIDE
            self.cash += pnl_usd
            self.trades += 1
            if pnl_usd > 0: self.wins += 1
            else: self.tweak_params(loss=True)
            self.position = 0; self.entry = None

        if action != 0:
            fill = self.fill_price(close, action)
            self.cash -= COMM_PER_SIDE
            self.position = action
            self.entry = fill

    def close_all(self, close):
        if self.position == 0: return
        exit_px = self.fill_price(close, 0)
        signed = 1 if self.position == 1 else -1
        pnl_points = (exit_px - self.entry) * signed
        pnl_usd = points_to_usd(pnl_points) - COMM_PER_SIDE
        self.cash += pnl_usd
        self.trades += 1
        if pnl_usd > 0: self.wins += 1
        else: self.tweak_params(loss=True)
        self.position = 0; self.entry = None

    def tweak_params(self, loss: bool):
        if not loss: return
        self.ema_fast_p = max(2, self.ema_fast_p + random.choice([-2,-1,1,2]))
        self.ema_slow_p = max(self.ema_fast_p+1, self.ema_slow_p + random.choice([-4,-2,2,4]))
        self.rsi_buy = min(70.0, max(50.0, self.rsi_buy + random.choice([-2,-1,1,2])))
        self.rsi_sell = max(30.0, min(50.0, self.rsi_sell + random.choice([-2,-1,1,2])))

# --- Exécution principale ---
def main():
    rows = ensure_data()
    bot = Bot()
    load_state(bot)

    for ts,o,h,l,c,v in rows:
        bot.step(ts, float(c))

    if rows:
        bot.close_all(float(rows[-1][4]))

    winrate = (bot.wins / bot.trades * 100.0) if bot.trades else 0.0
    summary_lines = [
        "==== Simple Nasdaq Bot Report ====",
        f"PnL net (USD): {bot.cash:.2f}",
        f"Trades: {bot.trades} | Win rate: {winrate:.2f}%",
        f"Params finaux -> EMA_fast:{bot.ema_fast_p} EMA_slow:{bot.ema_slow_p} "
        f"RSI_buy:{bot.rsi_buy:.1f} RSI_sell:{bot.rsi_sell:.1f}",
        "Données: data/mnq_5m.csv (Yahoo) ou data/nq.csv (synthétique)",
    ]
    text = "\n".join(summary_lines)
    print(text)

    # Écriture rapport complet
    os.makedirs("out", exist_ok=True)
    ts_now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    with open(f"out/run_{ts_now}.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")

    with open("out/summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "ts": ts_now,
            "pnl_usd": round(bot.cash, 2),
            "trades": bot.trades,
            "winrate_pct": round(winrate, 2),
            "ema_fast": bot.ema_fast_p,
            "ema_slow": bot.ema_slow_p,
            "rsi_buy": bot.rsi_buy,
            "rsi_sell": bot.rsi_sell,
            "data_source": "data/mnq_5m.csv" if os.path.isfile("data/mnq_5m.csv") else "data/nq.csv",
        }, f, indent=2)

    save_state(bot)

if __name__ == "__main__":
    main()
