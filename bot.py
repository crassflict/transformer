# bot.py
# Nasdaq futures adaptive bot — multi-indicateurs, multi-timeframes, stdlib only
# - Lit data/mnq_5m.csv (créé par fetch_mnq.py) sinon génère data/nq.csv synthétique
# - Indicateurs: EMA9, EMA21, SMA50, SMA200, MACD(12,26,9), RSI(14), Volume & VMA(20),
#   VWAP (rolling), Volume Profile POC (fenêtre), Cumulative Delta (approx)
# - Multi-timeframes: 5m (base) + 15m (agrégation 3x5m) + 60m (12x5m) pour tendance
# - Logique: score bull/bear pondéré par des poids adaptatifs (qui s’ajustent après pertes)
# - Persistance des hyperparamètres dans state.json, rapport dans out/

import csv, os, random, math, json
from datetime import datetime, timedelta, timezone

DATA_REAL = "data/mnq_5m.csv"
DATA_SYN  = "data/nq.csv"
STATE_FILE = "state.json"

# ====== Utils numériques (stdlib) ======
def ema(prev, x, period):
    if period <= 1: return x
    a = 2.0/(period+1.0)
    return (x if prev is None else (a*x + (1-a)*prev))

def sma(window):
    s, q = 0.0, []
    def push(x):
        nonlocal s, q
        q.append(x); s += x
        return s/len(q)
    def push_n(x, maxlen):
        nonlocal s, q
        q.append(x); s += x
        if len(q) > maxlen:
            s -= q.pop(0)
        return s/len(q)
    return push, push_n

def clamp(x, lo, hi): return lo if x<lo else (hi if x>hi else x)

def pct_change(prev, x):
    if prev is None or prev == 0: return 0.0
    return (x-prev)/prev

# ====== Lecture CSV ou génération synthétique ======
def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ts = row["timestamp"]
            if ts.isdigit():
                ts = datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
            o = float(row["open"]); h = float(row["high"]); l = float(row["low"]); c = float(row["close"]); v = float(row["volume"])
            rows.append((ts,o,h,l,c,v))
    return rows

def ensure_data(minutes=60*24*30):
    os.makedirs("data", exist_ok=True)
    if os.path.isfile(DATA_REAL):
        print("✅ Found real MNQ=F data (data/mnq_5m.csv)")
        return load_csv(DATA_REAL)
    print("⚠️ No real data found — generating synthetic dataset.")
    start = datetime(2024,1,1,tzinfo=timezone.utc)
    price = 16000.0
    out=[]
    for i in range(minutes):
        ts = start + timedelta(minutes=i)
        drift = 0.05*math.sin(i/800)
        shock = random.gauss(0, 2.0)
        close = max(1000.0, price + drift + shock)
        high  = max(price, close) + random.random()*0.8
        low   = min(price, close) - random.random()*0.8
        open_ = price
        vol   = random.randint(50, 1500)
        out.append((ts.isoformat(), open_, high, low, close, vol))
        price = close
    with open(DATA_SYN,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["timestamp","open","high","low","close","volume"]); w.writerows(out)
    return out

# ====== Agrégation OHLCV pour multi-timeframe ======
def aggregate_ohlcv(bars, group_size):
    # bars: list of (ts,o,h,l,c,v) in chronological order (5m)
    out=[]; buf=[]
    for b in bars:
        buf.append(b)
        if len(buf)==group_size:
            ts0,o0,_,_,_,_ = buf[0]
            h = max(x[2] for x in buf)
            l = min(x[3] for x in buf)
            c = buf[-1][4]
            o = o0
            v = sum(x[5] for x in buf)
            out.append((ts0,o,h,l,c,v))
            buf=[]
    return out

# ====== Bot ======
class Bot:
    def __init__(self):
        # hyperparams (adaptatifs)
        self.ema_fast = 9
        self.ema_mid  = 21
        self.sma50_win = 50
        self.sma200_win = 200
        self.rsi_len = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.vma_win = 20
        self.vwap_win = 50
        self.vp_win = 200   # volume profile window (nb barres 5m)
        self.delta_win = 300 # fenêtre pour cum delta (affichage/normalisation)

        # poids (scores de décision)
        self.w_macd = 1.0
        self.w_trend = 1.5
        self.w_rsi = 0.8
        self.w_vol = 0.6
        self.w_vwap = 0.7
        self.w_poc = 0.7
        self.w_mtf = 1.2

        # seuils décision
        self.score_enter = 1.2
        self.score_exit  = 0.3

        # état runtime
        self.reset_state()

    def reset_state(self):
        self.pos = 0         # -1 short, 0 flat, 1 long
        self.entry = None
        self.cash = 0.0
        self.trades = 0
        self.wins = 0

        # 5m indicators
        self.e9 = None; self.e21=None
        self.sma50_s, self.sma50 = sma(None)[0], None  # not used
        # rolling SMA implementations:
        self._sma50_sum=0.0; self._sma50_q=[]
        self._sma200_sum=0.0; self._sma200_q=[]
        self._vma_sum=0.0; self._vma_q=[]
        self._vwap_num=0.0; self._vwap_den=0.0; self._vwap_q=[]  # keep tuples (typical*vol, vol) to roll

        # MACD
        self._ema_macd_fast=None; self._ema_macd_slow=None; self._ema_signal=None

        # RSI
        self._rsi_gain=None; self._rsi_loss=None; self.rsi=50.0

        # Volume profile buffer: list of (price,vol)
        self._vp_buf=[]  # store tuples (close, volume)

        # Cumulative delta (approx)
        self.cum_delta=0.0

        # multi-timeframe trend flags
        self.mtf_bull = 0.0  # -1..+1
        self.mtf_bear = 0.0

    # ---------- helpers ----------
    def _push_sma(self, q, s, x, maxlen):
        q.append(x); s += x
        if len(q)>maxlen: s -= q.pop(0)
        return s, (s/len(q)), q

    def _sma_val(self, q, s): return (s/len(q)) if q else None

    def _update_rsi(self, close, prev_close):
        if prev_close is None: return
        change = close - prev_close
        gain = max(change, 0.0); loss = max(-change, 0.0)
        if self._rsi_gain is None:  # seed
            self._rsi_gain, self._rsi_loss = gain, loss
            self.rsi = 50.0
            return
        n=self.rsi_len
        self._rsi_gain = (self._rsi_gain*(n-1)+gain)/n
        self._rsi_loss = (self._rsi_loss*(n-1)+loss)/n
        rs = self._rsi_gain / (self._rsi_loss + 1e-12)
        self.rsi = 100.0 - (100.0/(1.0+rs))

    def _update_macd(self, close):
        self._ema_macd_fast = ema(self._ema_macd_fast, close, self.macd_fast)
        self._ema_macd_slow = ema(self._ema_macd_slow, close, self.macd_slow)
        macd = (self._ema_macd_fast - self._ema_macd_slow) if (self._ema_macd_fast is not None and self._ema_macd_slow is not None) else 0.0
        self._ema_signal = ema(self._ema_signal, macd, self.macd_signal)
        signal = self._ema_signal if self._ema_signal is not None else 0.0
        hist = macd - signal
        return macd, signal, hist

    def _update_vwap(self, typical, vol):
        # rolling window VWAP over last vwap_win bars
        self._vwap_q.append((typical*vol, vol))
        self._vwap_num += typical*vol; self._vwap_den += vol
        if len(self._vwap_q)>self.vwap_win:
            num,den = self._vwap_q.pop(0)
            self._vwap_num -= num; self._vwap_den -= den
        return self._vwap_num/max(1.0,self._vwap_den)

    def _update_volume_profile(self, close, vol):
        self._vp_buf.append((close, vol))
        if len(self._vp_buf) > self.vp_win:
            self._vp_buf.pop(0)
        # compute simple POC = price level (rounded to 1.0) with max volume
        buckets={}
        for p,v in self._vp_buf:
            key = round(p)  # 1 point bucket (peut ajuster)
            buckets[key] = buckets.get(key,0)+v
        if not buckets: return None
        poc = max(buckets.items(), key=lambda kv: kv[1])[0]
        return float(poc)

    def _update_cum_delta(self, close, prev_close, vol):
        if prev_close is None: return self.cum_delta
        direction = 1 if close>prev_close else (-1 if close<prev_close else 0)
        self.cum_delta += direction * vol
        return self.cum_delta

    # ---------- décision ----------
    def decide(self, feats):
        # feats dict keys: close, prev_close, e9,e21,sma50,sma200, rsi, macd, signal, hist,
        # vma, vol, vwap, poc, cum_delta, mtf_trend (-1..+1)
        score = 0.0

        # Trend alignment EMA9>EMA21>price above SMA50/SMA200 (bull) or l’inverse
        trend_bull = 0.0
        if feats["e9"] and feats["e21"] and feats["e9"]>feats["e21"]: trend_bull += 0.5
        if feats["sma50"] and feats["close"]>feats["sma50"]: trend_bull += 0.3
        if feats["sma200"] and feats["close"]>feats["sma200"]: trend_bull += 0.2
        trend_bear = (1.0 - trend_bull)

        # MACD contribution
        macd_bull = 1.0 if feats["macd"]>feats["signal"] and feats["hist"]>0 else 0.0
        macd_bear = 1.0 if feats["macd"]<feats["signal"] and feats["hist"]<0 else 0.0

        # RSI gating
        rsi_bull = 1.0 if feats["rsi"]>=55 else 0.0
        rsi_bear = 1.0 if feats["rsi"]<=45 else 0.0

        # Volume confirmation
        vol_conf = 1.0 if feats["vol"]>1.05*feats["vma"] else 0.0

        # VWAP: prix au-dessus = bull, en dessous = bear
        vwap_bull = 1.0 if feats["close"]>feats["vwap"] else 0.0
        vwap_bear = 1.0 - vwap_bull

        # Position relative au POC (volume profile)
        poc = feats["poc"]
        poc_bull = 1.0 if poc is not None and feats["close"]>poc else 0.0
        poc_bear = 1.0 if poc is not None and feats["close"]<poc else 0.0

        # Cumulative delta (approx)
        cd_bull = 1.0 if feats["cum_delta"]>0 else 0.0
        cd_bear = 1.0 if feats["cum_delta"]<0 else 0.0

        # Multi-timeframe trend
        mtf = feats["mtf_trend"]  # -1..+1
        mtf_bull = max(0.0, mtf)
        mtf_bear = max(0.0, -mtf)

        # Score
        score += self.w_trend*(trend_bull - trend_bear)
        score += self.w_macd*(macd_bull - macd_bear)
        score += self.w_rsi*(rsi_bull - rsi_bear)
        score += self.w_vol*(vol_conf)  # pas d'effet négatif si faible
        score += self.w_vwap*(vwap_bull - vwap_bear)
        score += self.w_poc*(poc_bull - poc_bear)
        score += self.w_mtf*(mtf_bull - mtf_bear)
        score += 0.4*(cd_bull - cd_bear)

        # Décision
        if score >= self.score_enter and self.pos <= 0: return 1, score
        if score <= -self.score_enter and self.pos >= 0: return -1, score
        if abs(score) < self.score_exit and self.pos != 0: return 0, score  # sortir si plus de momentum
        return 2, score  # hold

    # ---------- step ----------
    def step(self, bar, prev_bar, tf15_trend, tf60_trend):
        ts,o,h,l,c,v = bar
        prev_close = prev_bar[4] if prev_bar else None

        # EMAs 9/21
        self.e9  = ema(self.e9,  c, self.ema_fast)
        self.e21 = ema(self.e21, c, self.ema_mid)

        # SMA50/SMA200 rolling
        self._sma50_sum, sma50, self._sma50_q = self._push_sma(self._sma50_q, self._sma50_sum, c, self.sma50_win)
        self._sma200_sum, sma200, self._sma200_q = self._push_sma(self._sma200_q, self._sma200_sum, c, self.sma200_win)
        sma50_val  = self._sma50_sum/len(self._sma50_q) if self._sma50_q else None
        sma200_val = self._sma200_sum/len(self._sma200_q) if self._sma200_q else None

        # RSI
        self._update_rsi(c, prev_close)

        # MACD
        macd, signal, hist = self._update_macd(c)

        # Volume MA
        self._vma_sum, vma_val, self._vma_q = self._push_sma(self._vma_q, self._vma_sum, v, self.vma_win)

        # VWAP (rolling)
        typical = (h+l+c)/3.0
        vwap_val = self._update_vwap(typical, v)

        # Volume Profile POC
        poc = self._update_volume_profile(c, v)

        # Cumulative Delta approx
        cum_delta = self._update_cum_delta(c, prev_close, v)

        # Multi-timeframe trend (-1..+1) simple: signe(EMA9-EMA21) 15m/60m moyenné
        mtf_trend = 0.5*tf15_trend + 0.5*tf60_trend

        feats = {
            "close": c, "prev_close": prev_close,
            "e9": self.e9, "e21": self.e21,
            "sma50": sma50_val, "sma200": sma200_val,
            "rsi": self.rsi,
            "macd": macd, "signal": signal, "hist": hist,
            "vma": vma_val if vma_val else v, "vol": v,
            "vwap": vwap_val if vwap_val else c,
            "poc": poc,
            "cum_delta": cum_delta,
            "mtf_trend": mtf_trend,
        }

        action, score = self.decide(feats)
        realized = 0.0

        def fill(px, side):
            slip = 0.5*0.25
            if side == 1:   return px + slip
            if side == -1:  return px - slip
            if self.pos == 1:  return px - slip
            if self.pos == -1: return px + slip
            return px

        if action in (1,-1):  # open/flip
            if self.pos != 0 and action != self.pos:
                exit_px = fill(c, 0)
                signed = 1 if self.pos==1 else -1
                pnl_pts = (exit_px - self.entry)*signed
                pnl = pnl_pts / 0.25 * 5.0 - 2.5
                self.cash += pnl; self.trades += 1
                if pnl > 0: self.wins += 1
                else: self.adapt_on_loss(feats)  # learning

                self.pos = 0; self.entry=None
            # open
            self.entry = fill(c, action)
            self.cash -= 2.5
            self.pos = action

        elif action == 0 and self.pos != 0:  # exit
            exit_px = fill(c, 0)
            signed = 1 if self.pos==1 else -1
            pnl_pts = (exit_px - self.entry)*signed
            pnl = pnl_pts / 0.25 * 5.0 - 2.5
            self.cash += pnl; self.trades += 1
            if pnl > 0: self.wins += 1
            else: self.adapt_on_loss(feats)
            self.pos=0; self.entry=None

        # else hold

    def adapt_on_loss(self, feats):
        # Ajuste légèrement les poids/seuils vers ce qui aurait mieux "filtré"
        # Si trade perdant en long et momentum faible: rendre l’entrée plus sélective
        # simple heuristique stochastique
        for wname in ["w_trend","w_macd","w_rsi","w_vol","w_vwap","w_poc","w_mtf"]:
            delta = random.choice([-0.1,-0.05,0.05,0.1])
            setattr(self, wname, clamp(getattr(self,wname)+delta, 0.2, 2.5))
        self.score_enter = clamp(self.score_enter + random.choice([-0.1,0.1]), 0.8, 2.0)
        self.score_exit  = clamp(self.score_exit  + random.choice([-0.05,0.05]), 0.1, 0.8)
        # fenêtres lissages (petits pas)
        self.vp_win = int(clamp(self.vp_win + random.choice([-10,10]), 50, 600))
        self.vwap_win = int(clamp(self.vwap_win + random.choice([-5,5]), 20, 200))

# ====== Persistance ======
def load_state(bot: Bot):
    if os.path.isfile(STATE_FILE):
        try:
            s = json.load(open(STATE_FILE,"r"))
            for k,v in s.items():
                if hasattr(bot,k): setattr(bot,k,v)
            print(f"✅ Loaded state: {s}")
        except Exception as e:
            print(f"⚠️ Could not load state: {e}")

def save_state(bot: Bot):
    keys = ["ema_fast","ema_mid","sma50_win","sma200_win","rsi_len","macd_fast","macd_slow","macd_signal",
            "vma_win","vwap_win","vp_win","delta_win",
            "w_macd","w_trend","w_rsi","w_vol","w_vwap","w_poc","w_mtf",
            "score_enter","score_exit"]
    s = {k:getattr(bot,k) for k in keys}
    json.dump(s, open(STATE_FILE,"w"))
    print(f"💾 Saved state: {s}")

# ====== Multi-timeframe tendance simple ======
def tf_trend_from_ohlcv(bars, ema_fast=9, ema_slow=21):
    e9=e21=None
    last=None
    for _,_,_,_,c,_ in bars:
        e9 = ema(e9,c,ema_fast)
        e21= ema(e21,c,ema_slow)
        last = (e9,e21)
    if last is None: return 0.0
    e9,e21 = last
    if e9 is None or e21 is None: return 0.0
    return 1.0 if e9>e21 else -1.0

# ====== Main ======
def main():
    rows = ensure_data()  # 5m
    # Construire 15m et 60m
    tf15 = aggregate_ohlcv(rows, 3)
    tf60 = aggregate_ohlcv(rows, 12)

    bot = Bot()
    load_state(bot)

    prev=None
    idx15=0; idx60=0
    for i,bar in enumerate(rows):
        # synchroniser tendance TF sup (on prend la dernière close connue)
        if i%3==0 and idx15 < len(tf15): idx15 = min(idx15+1, len(tf15))
        if i%12==0 and idx60 < len(tf60): idx60 = min(idx60+1, len(tf60))
        t15_trend = tf_trend_from_ohlcv(tf15[:idx15]) if idx15>0 else 0.0
        t60_trend = tf_trend_from_ohlcv(tf60[:idx60]) if idx60>0 else 0.0

        bot.step(bar, prev, t15_trend, t60_trend)
        prev = bar

    # close fin
    if rows and bot.pos!=0:
        _,_,_,_,c,_ = rows[-1]
        # simulate exit cost
        slip = 0.5*0.25
        if bot.pos==1: exit_px = c - slip
        else: exit_px = c + slip
        signed = 1 if bot.pos==1 else -1
        pnl_pts = (exit_px - bot.entry)*signed
        pnl = pnl_pts/0.25*5.0 - 2.5
        bot.cash += pnl; bot.trades += 1
        if pnl>0: bot.wins+=1
        else: bot.adapt_on_loss({})
        bot.pos=0; bot.entry=None

    # rapport
    winrate = (bot.wins/bot.trades*100.0) if bot.trades else 0.0
    lines = [
        "==== Simple Nasdaq Bot Report ====",
        f"PnL net (USD): {bot.cash:.2f}",
        f"Trades: {bot.trades} | Win rate: {winrate:.2f}%",
        f"Params -> EMA9:{bot.ema_fast} EMA21:{bot.ema_mid} SMA50:{bot.sma50_win} SMA200:{bot.sma200_win} RSI:{bot.rsi_len}",
        f"MACD({bot.macd_fast},{bot.macd_slow},{bot.macd_signal}) VMA:{bot.vma_win} VWAP_win:{bot.vwap_win} VP_win:{bot.vp_win}",
        f"Weights -> trend:{bot.w_trend:.2f} macd:{bot.w_macd:.2f} rsi:{bot.w_rsi:.2f} vol:{bot.w_vol:.2f} vwap:{bot.w_vwap:.2f} poc:{bot.w_poc:.2f} mtf:{bot.w_mtf:.2f}",
        f"Thresholds -> enter:{bot.score_enter:.2f} exit:{bot.score_exit:.2f}",
        f"Données: {DATA_REAL if os.path.isfile(DATA_REAL) else DATA_SYN}",
    ]
    text = "\n".join(lines)
    print(text)

    os.makedirs("out", exist_ok=True)
    ts_now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    with open(f"out/run_{ts_now}.txt","w",encoding="utf-8") as f:
        f.write(text+"\n")

    with open("out/summary.json","w",encoding="utf-8") as f:
        json.dump({
            "ts": ts_now,
            "pnl_usd": round(bot.cash,2),
            "trades": bot.trades,
            "winrate_pct": round(winrate,2),
            "ema9": bot.ema_fast, "ema21": bot.ema_mid,
            "sma50": bot.sma50_win, "sma200": bot.sma200_win,
            "rsi_len": bot.rsi_len,
            "macd": [bot.macd_fast, bot.macd_slow, bot.macd_signal],
            "vma_win": bot.vma_win, "vwap_win": bot.vwap_win, "vp_win": bot.vp_win,
            "weights": {"trend":bot.w_trend,"macd":bot.w_macd,"rsi":bot.w_rsi,"vol":bot.w_vol,"vwap":bot.w_vwap,"poc":bot.w_poc,"mtf":bot.w_mtf},
            "thresholds": {"enter":bot.score_enter,"exit":bot.score_exit},
            "data_source": DATA_REAL if os.path.isfile(DATA_REAL) else DATA_SYN,
        }, f, indent=2)

    save_state(bot)

if __name__ == "__main__":
    main()
