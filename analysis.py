"""Indicators, scoring, trade plans and back-test for the PSX morning scanner.

Everything here works on a plain OHLCV DataFrame indexed by date with columns
Open, High, Low, Close, Volume (oldest first). No network access.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- indicators

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(100.0).where(dn.notna())


def macd(close: pd.Series):
    line = ema(close, 12) - ema(close, 26)
    signal = ema(line, 9)
    return line, signal, line - signal


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df.Close.shift()
    tr = pd.concat([df.High - df.Low, (df.High - pc).abs(), (df.Low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df.Close.diff()).fillna(0)
    return (direction * df.Volume).cumsum()


def mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tp = (df.High + df.Low + df.Close) / 3
    flow = tp * df.Volume
    pos = flow.where(tp > tp.shift(), 0.0).rolling(n).sum()
    neg = flow.where(tp < tp.shift(), 0.0).rolling(n).sum()
    ratio = pos / neg.replace(0, np.nan)
    return (100 - 100 / (1 + ratio)).fillna(100.0).where(pos.notna())


def cmf(df: pd.DataFrame, n: int = 20) -> pd.Series:
    rng = (df.High - df.Low).replace(0, np.nan)
    mfm = ((df.Close - df.Low) - (df.High - df.Close)) / rng
    return (mfm.fillna(0) * df.Volume).rolling(n).sum() / df.Volume.rolling(n).sum()


def pivots(high: float, low: float, close: float) -> dict:
    p = (high + low + close) / 3
    return {
        "pivot": p,
        "r1": 2 * p - low,
        "r2": p + (high - low),
        "s1": 2 * p - high,
        "s2": p - (high - low),
    }


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA20"] = sma(df.Close, 20)
    df["SMA50"] = sma(df.Close, 50)
    df["RSI"] = rsi(df.Close)
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(df.Close)
    df["ATR"] = atr(df)
    df["OBV"] = obv(df)
    df["MFI"] = mfi(df)
    df["CMF"] = cmf(df)
    df["VolAvg20"] = df.Volume.rolling(20).mean().shift(1)  # average of the 20 days before
    df["High20"] = df.High.rolling(20).max().shift(1)       # prior 20-day high
    return df


# ------------------------------------------------------------------- scoring

# Weights add up to 100 for the technical + flow part. News adds up to +/-10.
WEIGHTS = {
    "trend": 20,       # close > SMA20 > SMA50
    "breakout": 15,    # close above prior 20-day high
    "close_strength": 10,  # closed near the day's high
    "volume": 20,      # volume vs 20-day average
    "money_flow": 15,  # OBV rising, CMF > 0, MFI healthy
    "momentum": 20,    # RSI in the 50-70 sweet spot, MACD histogram rising
}


def score_row(df: pd.DataFrame, i: int = -1) -> tuple[float, list[str], dict]:
    """Score the setup as of row i (default: last session). Returns score 0-100,
    a list of plain-English reasons and the component scores."""
    r = df.iloc[i]
    prev = df.iloc[i - 1]
    reasons: list[str] = []
    comp = {}

    # trend
    t = 0.0
    if pd.notna(r.SMA20) and r.Close > r.SMA20:
        t += 0.5
    if pd.notna(r.SMA50) and pd.notna(r.SMA20) and r.SMA20 > r.SMA50:
        t += 0.5
    comp["trend"] = t
    if t == 1:
        reasons.append("Uptrend (above 20- and 50-day averages)")
    elif t == 0:
        reasons.append("Below its 20-day average")

    # breakout
    b = 1.0 if pd.notna(r.High20) and r.Close > r.High20 else 0.0
    comp["breakout"] = b
    if b:
        reasons.append("Closed above its 20-day high (breakout)")

    # close strength (close location in the day's range)
    rng = r.High - r.Low
    cl = (r.Close - r.Low) / rng if rng > 0 else 0.5
    comp["close_strength"] = cl
    if cl >= 0.8:
        reasons.append("Closed near the day's high")
    elif cl <= 0.2:
        reasons.append("Closed near the day's low")

    # volume
    vr = r.Volume / r.VolAvg20 if pd.notna(r.VolAvg20) and r.VolAvg20 > 0 else 1.0
    v = float(np.clip((vr - 1) / 2, 0, 1))  # 1x -> 0, 3x+ -> 1
    if r.Close < prev.Close:
        v = 0.0  # heavy volume on a down day is selling, not buying
    comp["volume"] = v
    if vr >= 2 and r.Close >= prev.Close:
        reasons.append(f"Volume {vr:.1f}x its 20-day average on an up day")
    elif vr >= 2:
        reasons.append(f"Heavy selling: volume {vr:.1f}x average on a down day")

    # money flow
    obv_up = df.OBV.iloc[i] > df.OBV.iloc[i - 10] if len(df) > 10 else False
    m = (0.4 if obv_up else 0) + (0.3 if pd.notna(r.CMF) and r.CMF > 0 else 0)
    if pd.notna(r.MFI) and 50 <= r.MFI <= 80:
        m += 0.3
    comp["money_flow"] = m
    cmf_pos = pd.notna(r.CMF) and r.CMF > 0
    if obv_up and cmf_pos:
        reasons.append("Money flowing in (OBV rising, CMF positive)")
    elif not obv_up and pd.notna(r.CMF) and r.CMF < 0:
        reasons.append("Money flowing out (OBV falling, CMF negative)")

    # momentum
    mo = 0.0
    if pd.notna(r.RSI):
        if 50 <= r.RSI <= 70:
            mo += 0.6
        elif 70 < r.RSI <= 78:
            mo += 0.3
    if pd.notna(r.MACD_hist) and pd.notna(prev.MACD_hist) and r.MACD_hist > prev.MACD_hist:
        mo += 0.4
    comp["momentum"] = mo
    if pd.notna(r.RSI) and r.RSI > 75:
        reasons.append(f"RSI {r.RSI:.0f}: overbought, chase risk")

    score = sum(WEIGHTS[k] * comp[k] for k in WEIGHTS)
    return float(score), reasons, comp


def trade_plan(df: pd.DataFrame, i: int = -1) -> dict:
    """Next-session levels from session i: entry zone, targets, stop."""
    r = df.iloc[i]
    pv = pivots(r.High, r.Low, r.Close)
    entry_lo = min(pv["pivot"], r.Close)
    entry_hi = max(pv["pivot"], r.Close)
    stop = pv["s1"]
    risk = entry_hi - stop
    reward = pv["r1"] - entry_hi
    return {
        **pv,
        "entry_lo": entry_lo,
        "entry_hi": entry_hi,
        "stop": stop,
        "target1": pv["r1"],
        "target2": pv["r2"],
        "reward_risk": reward / risk if risk > 0 else np.nan,
        "upside_r1_pct": (pv["r1"] / r.Close - 1) * 100,
        "upside_r2_pct": (pv["r2"] / r.Close - 1) * 100,
    }


# ---------------------------------------------------------------- news tone

POS = ["profit up", "profit rises", "profit jumps", "record profit", "surge", "soar", "jump",
       "dividend", "bonus", "growth", "expansion", "upgrade", "buy rating", "beats", "rally",
       "gains", "approval", "approved", "contract", "order", "discovery", "higher", "strong"]
NEG = ["loss", "decline", "falls", "fall", "drop", "plunge", "slump", "downgrade", "default",
       "penalty", "fine", "probe", "lower", "weak", "shutdown", "halt", "suspend", "cut",
       "tumble", "sell-off", "selloff"]


def headline_tone(title: str) -> int:
    t = title.lower()
    p = sum(w in t for w in POS)
    n = sum(w in t for w in NEG)
    return int(np.sign(p - n))


def news_points(tones: list[int]) -> float:
    """+/-10 points from recent headline tones (capped)."""
    if not tones:
        return 0.0
    return float(np.clip(sum(tones) * 4, -10, 10))


# ---------------------------------------------------------------- back-test

def backtest(histories: dict[str, pd.DataFrame], days: int = 30, top_n: int = 10,
             min_avg_vol: float = 1_000_000) -> pd.DataFrame:
    """Re-run the technical ranking for each of the last `days` sessions and check
    what the top picks did the NEXT session using its high/low:
      target  = next high reached R1 (and low did not hit S1 first, unknowable -> counted 'both')
      stopped = next low fell to S1 without reaching R1
      open    = neither: closed in between, result = next close vs entry
    Uses technical scores only (old news is not stored)."""
    prepared = {s: add_indicators(d) for s, d in histories.items() if len(d) >= 60}
    if not prepared:
        return pd.DataFrame()
    all_dates = sorted(set().union(*[set(d.index) for d in prepared.values()]))
    test_dates = all_dates[-(days + 1):-1]
    rows = []
    for dt in test_dates:
        cands = []
        for s, d in prepared.items():
            if dt not in d.index:
                continue
            i = d.index.get_loc(dt)
            if i < 55 or i + 1 >= len(d):
                continue
            if not (d.VolAvg20.iloc[i] >= min_avg_vol):
                continue
            sc, _, _ = score_row(d, i)
            cands.append((sc, s, i))
        cands.sort(reverse=True)
        for rank, (sc, s, i) in enumerate(cands[:top_n], 1):
            d = prepared[s]
            plan = trade_plan(d, i)
            nxt = d.iloc[i + 1]
            entry = d.Close.iloc[i]
            hit_t = nxt.High >= plan["target1"]
            hit_s = nxt.Low <= plan["stop"]
            if hit_t and not hit_s:
                outcome, ret = "Target hit", (plan["target1"] / entry - 1) * 100
            elif hit_s and not hit_t:
                outcome, ret = "Stopped", (plan["stop"] / entry - 1) * 100
            elif hit_t and hit_s:
                outcome, ret = "Both touched", (nxt.Close / entry - 1) * 100
            else:
                outcome, ret = "Neither", (nxt.Close / entry - 1) * 100
            rows.append({"date": dt, "rank": rank, "symbol": s, "score": round(sc, 1),
                         "outcome": outcome, "return_pct": round(ret, 2)})
    return pd.DataFrame(rows)


def backtest_summary(bt: pd.DataFrame) -> dict:
    if bt.empty:
        return {}
    return {
        "trades": len(bt),
        "days": bt.date.nunique(),
        "target_rate": (bt.outcome == "Target hit").mean() * 100,
        "stop_rate": (bt.outcome == "Stopped").mean() * 100,
        "avg_return": bt.return_pct.mean(),
        "win_rate": (bt.return_pct > 0).mean() * 100,
    }
