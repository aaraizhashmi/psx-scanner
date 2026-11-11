"""Free data fetchers for the PSX scanner.

Sources (all free, public):
  - PSX Data Portal  https://dps.psx.com.pk  (symbols, market watch, daily history,
    company announcements, index history)
  - Google News RSS  (headlines per company)

Set the environment variable PSX_MOCK=1 to use generated sample data instead
(for testing without internet). Every fetcher fails softly: on an error it
returns an empty result and the app says what is missing.
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://dps.psx.com.pk"
PKT = timezone(timedelta(hours=5))
HEADERS = {"User-Agent": "Mozilla/5.0 (PSX morning scanner; personal use)"}
MOCK = os.environ.get("PSX_MOCK") == "1"

_session = requests.Session()
_session.headers.update(HEADERS)


def now_pkt() -> datetime:
    return datetime.now(PKT)


def _num(x) -> float:
    try:
        return float(str(x).replace(",", "").replace("%", "").strip())
    except ValueError:
        return np.nan


# ------------------------------------------------------------------ symbols

def get_symbols() -> pd.DataFrame:
    """symbol, name, sector for every listed security."""
    if MOCK:
        return _mock_symbols()
    r = _session.get(f"{BASE}/symbols", timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df.columns = [c.lower() for c in df.columns]
    out = pd.DataFrame({
        "symbol": df["symbol"].astype(str).str.upper(),
        "name": df.get("name", df["symbol"]),
        "sector": df.get("sectorname", pd.Series([""] * len(df))),
    })
    for flag in ("isetf", "isdebt"):
        if flag in df:
            out = out[~df[flag].fillna(False).astype(bool).values]
    return out.reset_index(drop=True)


# ------------------------------------------------------------- market watch

def get_market_watch() -> pd.DataFrame:
    """Latest session for every stock: symbol, listed_in, ldcp, open, high, low,
    current, change, change_pct, volume, shariah (in KMI All Share)."""
    if MOCK:
        return _mock_market_watch()
    r = _session.get(f"{BASE}/market-watch", timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table")
    if table is None:
        return pd.DataFrame()
    headers = [th.get_text(" ", strip=True).lower() for th in table.select("thead th")]
    rows = []
    for tr in table.select("tbody tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        cells = [td.get_text(" ", strip=True) for td in tds]
        sym = tds[0].get("data-search") or (tds[0].find("strong") or tds[0]).get_text(" ", strip=True)
        cells[0] = re.split(r"\s+", sym.strip())[0].upper()
        rows.append(dict(zip(headers, cells)))
    df = pd.DataFrame(rows)

    def col(*names):
        for n in names:
            for c in df.columns:
                if c.startswith(n):
                    return df[c]
        return pd.Series([np.nan] * len(df))

    out = pd.DataFrame({
        "symbol": col("symbol"),
        "listed_in": col("listed").fillna("").astype(str),
        "ldcp": col("ldcp").map(_num),
        "open": col("open").map(_num),
        "high": col("high").map(_num),
        "low": col("low").map(_num),
        "current": col("current").map(_num),
        "change_pct": col("change (%)", "change%").map(_num),
        "volume": col("volume").map(_num),
    })
    out["shariah"] = out.listed_in.str.upper().str.contains("KMIALLSHR|KMI30|KMI ALL", regex=True)
    return out.dropna(subset=["current"]).reset_index(drop=True)


# ------------------------------------------------------------ daily history

def _month_history(symbol: str, year: int, month: int) -> pd.DataFrame:
    r = _session.post(f"{BASE}/historical", data={"month": month, "year": year, "symbol": symbol},
                      timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    headers = [th.get_text(strip=True).upper() for th in soup.select("th")]
    rows = []
    for tr in soup.select("tr"):
        cells = [td.get_text(strip=True) for td in tr.select("td")]
        if len(cells) == len(headers) and cells:
            rows.append(dict(zip(headers, cells)))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["TIME"], format="%b %d, %Y", errors="coerce")
    out = pd.DataFrame({
        "Open": df["OPEN"].map(_num), "High": df["HIGH"].map(_num),
        "Low": df["LOW"].map(_num), "Close": df["CLOSE"].map(_num),
        "Volume": df["VOLUME"].map(_num),
    })
    out.index = df["Date"]
    return out.dropna()


def get_history(symbol: str, months: int = 6) -> pd.DataFrame:
    """Daily OHLCV for the last `months` months, completed sessions only."""
    if MOCK:
        return _mock_history(symbol)
    today = now_pkt()
    frames = []
    y, m = today.year, today.month
    for _ in range(months):
        try:
            frames.append(_month_history(symbol, y, m))
        except Exception:
            pass
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    # During market hours today's row (if any) is incomplete: drop it.
    if len(df) and df.index[-1].date() == today.date() and today.hour < 16:
        df = df.iloc[:-1]
    return df


def get_histories(symbols: list[str], months: int = 6, workers: int = 8, progress=None) -> dict:
    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(get_history, s, months): s for s in symbols}
        for k, f in enumerate(as_completed(futs), 1):
            s = futs[f]
            try:
                df = f.result()
                if len(df):
                    out[s] = df
            except Exception:
                pass
            if progress:
                progress(k / len(futs), s)
    return out


# --------------------------------------------------------------- KSE-100

def get_index_history(index: str = "KSE100") -> pd.DataFrame:
    """Daily closes for an index (Close, Volume)."""
    if MOCK:
        d = _mock_history(index, base=170000, vol=3e8)
        return d[["Close", "Volume"]]
    r = _session.get(f"{BASE}/timeseries/eod/{index}", timeout=30)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data).iloc[:, :3]
    df.columns = ["ts", "Close", "Volume"]
    df.index = pd.to_datetime(df.ts, unit="s")
    return df[["Close", "Volume"]].sort_index()


# --------------------------------------------------------- announcements

def get_announcements() -> pd.DataFrame:
    """Recent company announcements: date, symbol, title."""
    if MOCK:
        return _mock_announcements()
    r = _session.get(f"{BASE}/announcements/companies", timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table")
    if table is None:
        return pd.DataFrame()
    headers = [th.get_text(" ", strip=True).lower() for th in table.select("thead th")]
    rows = [dict(zip(headers, [td.get_text(" ", strip=True) for td in tr.find_all("td")]))
            for tr in table.select("tbody tr")]
    df = pd.DataFrame(rows)
    pick = lambda key: next((c for c in df.columns if key in c), None)
    c_date, c_sym, c_title = pick("date"), pick("symbol"), pick("title")
    if not (c_sym and c_title):
        return pd.DataFrame()
    return pd.DataFrame({
        "date": df[c_date] if c_date else "",
        "symbol": df[c_sym].str.upper().str.strip(),
        "title": df[c_title],
    })


# ------------------------------------------------------------------ news

def get_news(query: str, days: int = 2, limit: int = 6) -> list[dict]:
    """Recent headlines from Google News RSS for a query."""
    if MOCK:
        return _mock_news(query)
    url = (f"https://news.google.com/rss/search?q={quote_plus(query)}+when:{days}d"
           f"&hl=en-PK&gl=PK&ceid=PK:en")
    r = _session.get(url, timeout=20)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    items = []
    for it in root.iter("item"):
        items.append({
            "title": (it.findtext("title") or "").strip(),
            "link": it.findtext("link") or "",
            "source": (it.find("source").text if it.find("source") is not None else ""),
            "published": it.findtext("pubDate") or "",
        })
        if len(items) >= limit:
            break
    return items


# ============================================================ mock data

_rng = np.random.default_rng(7)
_MOCK_SYMS = [
    ("FCCL", "Fauji Cement Company", "CEMENT", True), ("DGKC", "D.G. Khan Cement", "CEMENT", True),
    ("LUCK", "Lucky Cement", "CEMENT", True), ("MLCF", "Maple Leaf Cement", "CEMENT", True),
    ("PPL", "Pakistan Petroleum", "OIL & GAS EXPLORATION", True),
    ("OGDC", "Oil & Gas Development Co", "OIL & GAS EXPLORATION", True),
    ("MARI", "Mari Energies", "OIL & GAS EXPLORATION", True), ("PSO", "Pakistan State Oil", "OIL & GAS MARKETING", True),
    ("HASCOL", "Hascol Petroleum", "OIL & GAS MARKETING", False), ("EFERT", "Engro Fertilizers", "FERTILIZER", True),
    ("FFC", "Fauji Fertilizer", "FERTILIZER", True), ("MEBL", "Meezan Bank", "COMMERCIAL BANKS", True),
    ("BOP", "Bank of Punjab", "COMMERCIAL BANKS", False), ("UBL", "United Bank", "COMMERCIAL BANKS", False),
    ("SEARL", "The Searle Company", "PHARMACEUTICALS", True), ("PAEL", "Pak Elektron", "CABLE & ELECTRICAL", True),
    ("PIBTL", "Pakistan Intl Bulk Terminal", "TRANSPORT", True), ("FFL", "Fauji Foods", "FOOD", True),
    ("NML", "Nishat Mills", "TEXTILE", True), ("SYS", "Systems Ltd", "TECHNOLOGY", True),
    ("TRG", "TRG Pakistan", "TECHNOLOGY", True), ("CNERGY", "Cnergyico PK", "REFINERY", True),
    ("KEL", "K-Electric", "POWER", True), ("HUBC", "Hub Power", "POWER", True),
    ("WTL", "WorldCall Telecom", "TECHNOLOGY", False), ("ATRL", "Attock Refinery", "REFINERY", True),
    ("POL", "Pakistan Oilfields", "OIL & GAS EXPLORATION", True), ("ENGROH", "Engro Holdings", "INV", True),
]
_cache: dict = {}


def _mock_symbols():
    return pd.DataFrame([{"symbol": s, "name": n, "sector": sec} for s, n, sec, _ in _MOCK_SYMS])


def _mock_history(symbol, base=None, vol=None):
    key = (symbol, base)
    if key in _cache:
        return _cache[key]
    rng = np.random.default_rng(abs(hash(symbol)) % 2**32)
    n = 130
    dates = pd.bdate_range(end=pd.Timestamp(now_pkt().date()) - pd.Timedelta(days=1), periods=n)
    base = base or float(rng.uniform(10, 400))
    drift = rng.normal(0.0008, 0.001)
    ret = rng.normal(drift, 0.022, n)
    close = base * np.exp(np.cumsum(ret))
    op = close * (1 + rng.normal(0, 0.006, n))
    hi = np.maximum(op, close) * (1 + np.abs(rng.normal(0, 0.01, n)))
    lo = np.minimum(op, close) * (1 - np.abs(rng.normal(0, 0.01, n)))
    v0 = vol or float(rng.uniform(4e5, 1.5e7))
    v = v0 * np.exp(rng.normal(0, 0.45, n))
    v[-1] *= rng.choice([1, 1, 2.5, 3.5])
    df = pd.DataFrame({"Open": op, "High": hi, "Low": lo, "Close": close, "Volume": v.round()}, index=dates)
    _cache[key] = df
    return df


def _mock_market_watch():
    rows = []
    for s, _, _, sh in _MOCK_SYMS:
        h = _mock_history(s)
        r, p = h.iloc[-1], h.iloc[-2]
        rows.append({"symbol": s, "listed_in": "KMIALLSHR,ALLSHR" if sh else "ALLSHR",
                     "ldcp": p.Close, "open": r.Open, "high": r.High, "low": r.Low,
                     "current": r.Close, "change_pct": (r.Close / p.Close - 1) * 100,
                     "volume": r.Volume, "shariah": sh})
    return pd.DataFrame(rows)


def _mock_announcements():
    return pd.DataFrame([
        {"date": now_pkt().strftime("%b %d, %Y"), "symbol": "FCCL", "title": "Financial Results for the Quarter Ended September 30, 2026"},
        {"date": now_pkt().strftime("%b %d, %Y"), "symbol": "PPL", "title": "Board Meeting to consider interim dividend"},
        {"date": now_pkt().strftime("%b %d, %Y"), "symbol": "SEARL", "title": "Material Information"},
    ])


def _mock_news(query):
    sample = [
        ("profit jumps 51% on strong local demand", 1), ("shares surge as volumes rise", 1),
        ("faces lower margins amid cost pressure", -1), ("board approves expansion plan", 1),
        ("stock falls after weak quarter", -1), ("market update: KSE-100 gains", 1),
    ]
    rng = np.random.default_rng(abs(hash(query)) % 2**32)
    k = int(rng.integers(0, 3))
    return [{"title": f"{query.split()[0]} {sample[j][0]}", "link": "https://example.com",
             "source": "Sample", "published": ""} for j in rng.choice(len(sample), k, replace=False)]
