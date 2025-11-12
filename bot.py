# bot.py
# Nasdaq futures adaptive bot — avec apprentissage et vraies données MNQ (Yahoo Finance)

import csv, os, random, math, json
from datetime import datetime, timedelta, timezone

STATE_FILE = "state.json"

# --- Fonction de persistance d'état (apprentissage entre runs) ---
def load_state(bot):
    if os.path.isfile(STATE_FILE):
        try:
            s = json.load(open(STATE_FILE))
            bot.ema_fast_p = int(s.get("ema_fast_p", bot.ema_fast_p))
            bot.ema_slow_p = int(s.get("ema_slow_p", bot.ema_slow_p))
            bot.rsi_buy   = float(s.get("rsi_buy", bot.rsi_buy))
            bot.rsi_sell  = float(s.get("rsi_sell", bot.rsi_sell))
            print(f"Loaded previous state: {s}")
        except Exception:
            pass

def save_state(bot):
    s = {
        "ema_fast_p": bot.ema_fast_p,
        "ema_slow_p": bot.ema_slow_p,
        "rsi_buy": bot.rsi_buy,
        "rsi_sell": bot.rsi_sell,
    }
    json.dump(s, open(STATE_FILE, "w"))
    print(f"Saved state: {s}")

# --- Paramètres marché ---
TICK_SIZE = 0.25
TICK_VALUE = 5.0
COMM_PER_SIDE = 2.5
SLIPPAGE_TICKS = 0.5

# --- Utilitaires ---
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
