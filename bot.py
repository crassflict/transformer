# bot.py
# Nasdaq futures simple learning bot — zero dependency
# - Lit un CSV OHLCV si présent (data/nq.csv), sinon génère des données synthétiques
# - Stratégie EMA/RSI simple : long, short ou flat
# - "Apprend de ses erreurs" : à chaque trade perdant, il ajuste légèrement ses paramètres
# - Affiche un petit report final (PnL, nb trades, winrate)

import csv, os, random, math
from datetime import datetime, timedelta, timezone

# -----------------------
# Config "contrat" (NQ)
# -----------------------
TICK_SIZE = 0.25       # points
TICK_VALUE = 5.0       # USD / tick
COMM_PER_SIDE = 2.5    # USD par entrée/sortie
SLIPPAGE_TICKS = 0.5

# -----------------------
# Utilitaires
# -----------------------
def ensure_data(path="data/nq.csv", minutes=60*24*30):
    """Charge un CSV si présent, sinon crée des données synthétiques 1-min."""
    if os.path.isfile(path):
        return load_csv(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    price = 16000.0
    rows = []
    for i in range(minutes):
        ts = start + timedelta(minutes=i)
        # petit drift sinus + bruit gaussien
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

def ema(prev, price, period):
    if period <= 1: return price
    alpha = 2.0/(period+1.0)
    return alpha*price + (1-alpha)*(prev if prev is not None else price)

def rsi_wilder(gains_prev, losses_prev, change, period):
    gain = max(change, 0.0)
    loss = max(-change, 0.0)
    if gains_prev is None or losses_prev is None:
        # premier seed
        return gain, loss, 50.0
    avg_gain = (gains_prev*(period-1) + gain)/period
    avg_loss = (losses_prev*(period-1) + loss)/period
    rs = (avg_gain / avg_loss) if avg_loss > 0 else 999.0
    rsi = 100.0 - (100.0/(1.0+rs))
    return avg_gain, avg_loss, rsi

def points_to_usd(points):
    ticks = points / TICK_SIZE
    return ticks * TICK_VALUE

# -----------------------
# Bot
# -----------------------
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
        self.position = 0   # -1 short, 0 flat, 1 long
        self.entry = None
        self.cash = 0.0
        self.trades = 0
        self.wins = 0

    def decide(self, close):
        # Signal simple
        go_long  = (self.e_fast is not None and self.e_slow is not None and self.e_fast > self.e_slow and self.rsi_val >= self.rsi_buy)
        go_short = (self.e_fast is not None and self.e_slow is not None and self.e_fast < self.e_slow and self.rsi_val <= self.rsi_sell)

        if go_long and self.position <= 0:
            return 1
        if go_short and self.position >= 0:
            return -1
        return 0  # keep

    def fill_price(self, px, side):
        slip = SLIPPAGE_TICKS * TICK_SIZE
        if side == 1:   # buy worse (higher)
            return px + slip
        if side == -1:  # sell worse (lower)
            return px - slip
        # closing: penalize too
        if self.position == 1:
            return px - slip
        if self.position == -1:
            return px + slip
        return px

    def step(self, ts, close):
        # indicators
        self.e_fast = ema(self.e_fast, close, self.ema_fast_p)
        self.e_slow = ema(self.e_slow, close, self.ema_slow_p)
        if self.prev_close is not None:
            ch = close - self.prev_close
            self.rsi_gain, self.rsi_loss, self.rsi_val = rsi_wilder(self.rsi_gain, self.rsi_loss, ch, self.rsi_len)
        self.prev_close = close

        action = self.decide(close)

        # execute
        if action == 0:
            return

        # flip/close if needed
        if self.position != 0 and action != self.position:
            exit_px = self.fill_price(close, 0)
            signed = 1 if self.position == 1 else -1
            pnl_points = (exit_px - self.entry) * signed
            pnl_usd = points_to_usd(pnl_points) - COMM_PER_SIDE
            self.cash += pnl_usd
            self.trades += 1
            if pnl_usd > 0: self.wins += 1
            # learning on loss
            if pnl_usd <= 0:
                self.tweak_params(loss=True)

            self.position = 0
            self.entry = None

        # open if needed
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
        if pnl_usd <= 0:
            self.tweak_params(loss=True)
        self.position = 0
        self.entry = None

    def tweak_params(self, loss: bool):
        """Petite adaptation stochastique après un trade perdant."""
        if not loss:
            return
        # bouger légèrement les paramètres vers quelque chose de différent
        self.ema_fast_p = max(2, self.ema_fast_p + random.choice([-2,-1,1,2]))
        self.ema_slow_p = max(self.ema_fast_p+1, self.ema_slow_p + random.choice([-4,-2,2,4]))
        self.rsi_buy = min(70.0, max(50.0, self.rsi_buy + random.choice([-2,-1,1,2])))
        self.rsi_sell = max(30.0, min(50.0, self.rsi_sell + random.choice([-2,-1,1,2])))

# -----------------------
# Run
# -----------------------
def main():
    rows = ensure_data()  # [(ts, o,h,l,c,v), ...]
    bot = Bot()

    # Warmup 100 bar pour stabiliser EMA/RSI
    for i, row in enumerate(rows):
        ts, o,h,l,c,v = row
        bot.step(ts, float(c))

    # Close tout à la dernière barre
    if rows:
        bot.close_all(float(rows[-1][4]))

    # Report
    winrate = (bot.wins / bot.trades * 100.0) if bot.trades else 0.0
    print("==== Simple Nasdaq Bot Report ====")
    print(f"PnL net (USD): {bot.cash:.2f}")
    print(f"Trades: {bot.trades} | Win rate: {winrate:.2f}%")
    print(f"Params finaux -> EMA_fast:{bot.ema_fast_p} EMA_slow:{bot.ema_slow_p} RSI_buy:{bot.rsi_buy:.1f} RSI_sell:{bot.rsi_sell:.1f}")
    print("Données: data/nq.csv (générées si absent)")

if __name__ == "__main__":
    main()

