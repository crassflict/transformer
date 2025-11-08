import pandas as pd
import numpy as np

def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    out = 100 - (100 / (1 + rs))
    return out.fillna(method="bfill").fillna(50.0)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema21"] = ema(df["close"], 21)
    df["ema55"] = ema(df["close"], 55)
    df["rsi14"] = rsi(df["close"], 14)
    return df

def rule_signal(row) -> str:
    return "buy" if (row["ema21"] > row["ema55"] and row["rsi14"] > 50) else "flat"

def backtest_long_only(df: pd.DataFrame, fee=0.0006, slip=0.0002) -> pd.DataFrame:
    d = df.copy()
    d["ret"] = d["close"].pct_change().fillna(0)
    d["pos"] = d["signal"].shift(1).fillna(0)
    trade_change = d["pos"].diff().abs().fillna(0)
    d["strategy_ret"] = d["pos"] * d["ret"] - trade_change * (fee + slip)
    d["equity"] = (1 + d["strategy_ret"]).cumprod()
    return d
