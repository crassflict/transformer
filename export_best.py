# export_best.py — extrait les params utiles de state.json vers config.json (pour Quantower)

import json, os

STATE = "state.json"
CONFIG = "config.json"

DEFAULTS = {
    "ema9": 9, "ema21": 21, "sma50": 50, "sma200": 200,
    "rsi_len": 14, "macd_fast":12, "macd_slow":26, "macd_signal":9,
    "vma_win":20, "vwap_win":145, "vp_win":70,
    "w_trend":1.5, "w_macd":1.0, "w_rsi":0.8, "w_vol":0.6, "w_vwap":0.7, "w_poc":0.7, "w_mtf":1.2,
    "score_enter":1.2, "score_exit":0.3
}

def main():
    if not os.path.isfile(STATE):
        print("No state.json yet; writing defaults config.json")
        cfg = DEFAULTS
    else:
        s = json.load(open(STATE, "r", encoding="utf-8"))
        cfg = {
            "ema9": int(s.get("ema_fast", DEFAULTS["ema9"])),
            "ema21": int(s.get("ema_mid", DEFAULTS["ema21"])),
            "sma50": int(s.get("sma50_win", DEFAULTS["sma50"])),
            "sma200": int(s.get("sma200_win", DEFAULTS["sma200"])),
            "rsi_len": int(s.get("rsi_len", DEFAULTS["rsi_len"])),
            "macd_fast": int(s.get("macd_fast", DEFAULTS["macd_fast"])),
            "macd_slow": int(s.get("macd_slow", DEFAULTS["macd_slow"])),
            "macd_signal": int(s.get("macd_signal", DEFAULTS["macd_signal"])),
            "vma_win": int(s.get("vma_win", DEFAULTS["vma_win"])),
            "vwap_win": int(s.get("vwap_win", DEFAULTS["vwap_win"])),
            "vp_win": int(s.get("vp_win", DEFAULTS["vp_win"])),
            "w_trend": float(s.get("w_trend", DEFAULTS["w_trend"])),
            "w_macd": float(s.get("w_macd", DEFAULTS["w_macd"])),
            "w_rsi": float(s.get("w_rsi", DEFAULTS["w_rsi"])),
            "w_vol": float(s.get("w_vol", DEFAULTS["w_vol"])),
            "w_vwap": float(s.get("w_vwap", DEFAULTS["w_vwap"])),
            "w_poc": float(s.get("w_poc", DEFAULTS["w_poc"])),
            "w_mtf": float(s.get("w_mtf", DEFAULTS["w_mtf"])),
            "score_enter": float(s.get("score_enter", DEFAULTS["score_enter"])),
            "score_exit": float(s.get("score_exit", DEFAULTS["score_exit"])),
        }
    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print("Wrote", CONFIG)

if __name__ == "__main__":
    main()
