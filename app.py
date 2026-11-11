"""PSX Morning Scanner: day-trading watchlist for all liquid PSX stocks.

Run locally:   streamlit run app.py
Test offline:  PSX_MOCK=1 streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import analysis as A
import psx_data as D

st.set_page_config(page_title="PSX Morning Scanner", page_icon="📈", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-left: 0.8rem; padding-right: 0.8rem; max-width: 1100px;}
.card {border: 1px solid rgba(128,128,128,.35); border-radius: 12px; padding: 12px 14px; margin-bottom: 10px;}
.card h4 {margin: 0 0 2px 0;}
.muted {opacity: .7; font-size: .85rem;}
.lv {display:grid; grid-template-columns: repeat(auto-fit, minmax(92px, 1fr)); gap:6px; margin:8px 0;}
.lv div {border-radius:8px; padding:6px 8px; background: rgba(128,128,128,.10); font-size:.82rem;}
.lv b {display:block; font-size:.98rem;}
.pos {color:#1a9a5a;} .neg {color:#d0453f;}
.tag {display:inline-block; font-size:.75rem; padding:1px 8px; border-radius:999px; border:1px solid rgba(128,128,128,.4); margin:2px 4px 2px 0;}
</style>
""", unsafe_allow_html=True)

TTL = 3 * 60 * 60  # cache data for 3 hours


# ------------------------------------------------------------- cached data

@st.cache_data(ttl=TTL, show_spinner=False)
def symbols():
    try:
        return D.get_symbols()
    except Exception:
        return pd.DataFrame(columns=["symbol", "name", "sector"])


@st.cache_data(ttl=TTL, show_spinner=False)
def market_watch():
    return D.get_market_watch()


@st.cache_data(ttl=TTL, show_spinner=False)
def histories(syms: tuple):
    bar = st.progress(0.0, text="Downloading price history…")
    out = D.get_histories(list(syms), months=6,
                          progress=lambda f, s: bar.progress(f, text=f"Downloading price history… {s}"))
    bar.empty()
    return out


@st.cache_data(ttl=TTL, show_spinner=False)
def history_one(sym: str):
    return D.get_history(sym, months=8)


@st.cache_data(ttl=TTL, show_spinner=False)
def index_hist():
    try:
        return D.get_index_history("KSE100")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=TTL, show_spinner=False)
def announcements():
    try:
        return D.get_announcements()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=TTL, show_spinner=False)
def news(query: str):
    try:
        return D.get_news(query)
    except Exception:
        return []


# ---------------------------------------------------------------- settings

with st.sidebar:
    st.header("Settings")
    shariah_only = st.toggle("Shariah-compliant only (KMI All Share)", value=False)
    min_avg_vol = st.number_input("Min. average daily volume (shares)", 100_000, 50_000_000,
                                  1_000_000, step=250_000)
    min_price = st.number_input("Min. share price (PKR)", 1.0, 1000.0, 5.0, step=1.0)
    top_n = st.slider("Picks to show", 5, 20, 10)
    cost_pct = st.number_input("Round-trip cost estimate (%)", 0.0, 3.0, 0.5, step=0.1,
                               help="Brokerage, taxes and fees for buying and selling. Check with your broker.")
    use_news = st.toggle("Score news headlines", value=True)
    if st.button("🔄 Refresh all data"):
        st.cache_data.clear()
        st.rerun()
    if D.MOCK:
        st.warning("SAMPLE DATA MODE (PSX_MOCK=1). Numbers are made up.")

# --------------------------------------------------------------- pipeline

st.title("📈 PSX Morning Scanner")
if D.MOCK:
    st.warning("SAMPLE DATA MODE: these numbers are made up, for testing only.")
now = D.now_pkt()
st.caption(f"Updated {now:%a %d %b %Y, %H:%M} PKT · End-of-day data from the PSX Data Portal · "
           "Ranks setups by rules. It does not predict prices.")

try:
    mw = market_watch()
except Exception as e:
    st.error(f"Couldn't load the PSX market watch ({e}). The PSX site may be down or changed. Try Refresh later.")
    st.stop()

syms = symbols()
names = dict(zip(syms.symbol, syms.name)) if len(syms) else {}
sectors = dict(zip(syms.symbol, syms.sector)) if len(syms) else {}

universe = mw[(mw.current >= min_price) & (mw.volume >= min_avg_vol * 0.3)]
if shariah_only:
    universe = universe[universe.shariah]
if universe.empty:
    st.warning("No stocks pass the filters. Lower the minimum volume in Settings.")
    st.stop()

hist = histories(tuple(sorted(universe.symbol)))

rows = []
for s, h in hist.items():
    if len(h) < 60:
        continue
    d = A.add_indicators(h)
    last = d.iloc[-1]
    if not (last.VolAvg20 >= min_avg_vol):
        continue
    sc, reasons, comp = A.score_row(d)
    plan = A.trade_plan(d)
    rows.append({
        "symbol": s, "name": names.get(s, s), "sector": sectors.get(s, ""),
        "score": sc, "tech_score": sc, "reasons": reasons, "plan": plan,
        "close": last.Close, "chg_pct": (last.Close / d.Close.iloc[-2] - 1) * 100,
        "vol": last.Volume, "vol_ratio": last.Volume / last.VolAvg20 if last.VolAvg20 else np.nan,
        "rsi": last.RSI, "mfi": last.MFI, "cmf": last.CMF, "date": d.index[-1],
    })
ranked = pd.DataFrame(rows)
if ranked.empty:
    st.warning("Not enough history for the filtered stocks.")
    st.stop()
ranked = ranked.sort_values("tech_score", ascending=False).reset_index(drop=True)

ann = announcements()
ann_by_sym = ann.groupby("symbol")["title"].apply(list).to_dict() if len(ann) else {}

# News for the top candidates only (keeps it fast)
ranked["headlines"] = [[] for _ in range(len(ranked))]
ranked["news_pts"] = 0.0
if use_news:
    k = min(len(ranked), top_n * 2 + 5)
    for i in range(k):
        n = news(f'"{ranked.at[i, "name"]}"')
        tones = [A.headline_tone(x["title"]) for x in n]
        for x, t in zip(n, tones):
            x["tone"] = t
        ranked.at[i, "headlines"] = n
        ranked.at[i, "news_pts"] = A.news_points(tones)
# Room to move: a setup whose R1 gain doesn't beat costs is pushed down.
ranked["room_pen"] = [(-15.0 if p["upside_r1_pct"] - cost_pct <= 0 else
                       -7.0 if (not np.isnan(p["reward_risk"]) and p["reward_risk"] < 1) else 0.0)
                      for p in ranked.plan]
ranked["score"] = ranked.tech_score + ranked.news_pts + ranked.room_pen
ranked = ranked.sort_values("score", ascending=False).reset_index(drop=True)
last_session = ranked.date.max()

# Market mood
ix = index_hist()
mood_txt, mood_ok = "KSE-100 trend unavailable", None
if len(ix) > 25:
    c = ix.Close
    s20 = c.rolling(20).mean().iloc[-1]
    chg = (c.iloc[-1] / c.iloc[-2] - 1) * 100
    mood_ok = bool(c.iloc[-1] > s20)
    mood_txt = (f"KSE-100 {c.iloc[-1]:,.0f} ({chg:+.2f}% last session), "
                f"{'above' if mood_ok else 'below'} its 20-day average")

with st.sidebar.expander("Data status"):
    st.write(f"Market watch: {len(mw)} stocks ({int(mw.shariah.sum())} Shariah)")
    st.write(f"Price histories loaded: {len(hist)} of {len(universe)}")
    st.write(f"Announcements: {len(ann)}")
    st.write(f"KSE-100 history: {len(ix)} days")
    st.write(f"PSX server: {D.server_info()}")
    st.caption("If a number here is 0, that PSX page changed or is down.")

tabs = st.tabs(["🏆 Top picks", "🔎 Stock check", "🌐 Market & flows", "📊 Track record", "ℹ️ How it works"])

# ================================================================ TOP PICKS
with tabs[0]:
    if mood_ok is False:
        st.warning(f"⚠️ {mood_txt}. Weak market: be choosier and size smaller.")
    elif mood_ok:
        st.success(f"✅ {mood_txt}.")
    else:
        st.info(mood_txt)
    st.caption(f"Based on the session of {last_session:%a %d %b}. Plan for the next session. "
               f"{len(ranked)} stocks scanned{' (Shariah only)' if shariah_only else ''}.")

    for i, r in ranked.head(top_n).iterrows():
        p = r.plan
        net_up = p["upside_r1_pct"] - cost_pct
        strength = "High" if r.score >= 70 else "Medium" if r.score >= 50 else "Low"
        cls = "pos" if r.chg_pct >= 0 else "neg"
        tags = "".join(f'<span class="tag">{t}</span>' for t in r.reasons)
        ann_list = ann_by_sym.get(r.symbol, [])
        ann_html = "".join(f'<div class="muted">📢 {a}</div>' for a in ann_list[:2])
        warn = ""
        if net_up <= 0:
            warn = '<div class="muted neg">Move to R1 is smaller than your trading costs.</div>'
        elif not np.isnan(p["reward_risk"]) and p["reward_risk"] < 1:
            warn = '<div class="muted neg">Reward is smaller than risk at R1.</div>'
        st.markdown(f"""
<div class="card">
  <h4>#{i+1} {r.symbol} <span class="muted">· {r['name']} · {r.sector}</span></h4>
  <div>Score <b>{r.score:.0f}</b> ({strength}) · Close <b>{r.close:,.2f}</b>
   <span class="{cls}">{r.chg_pct:+.2f}%</span> · Vol {r.vol:,.0f} ({r.vol_ratio:.1f}× avg) · RSI {r.rsi:.0f}</div>
  <div class="lv">
    <div>Buy zone<b>{p['entry_lo']:,.2f}–{p['entry_hi']:,.2f}</b></div>
    <div>Target 1 (R1)<b>{p['target1']:,.2f}</b><span class="pos">+{p['upside_r1_pct']:.1f}%</span></div>
    <div>Target 2 (R2)<b>{p['target2']:,.2f}</b><span class="pos">+{p['upside_r2_pct']:.1f}%</span></div>
    <div>Stop-loss (S1)<b>{p['stop']:,.2f}</b></div>
    <div>Reward : risk<b>{p['reward_risk']:.1f}</b></div>
  </div>
  {tags}{ann_html}{warn}
</div>""", unsafe_allow_html=True)
        if r.headlines:
            with st.expander(f"News for {r.symbol} ({len(r.headlines)})"):
                for h in r.headlines:
                    icon = {1: "🟢", -1: "🔴"}.get(h.get("tone", 0), "⚪")
                    st.markdown(f"{icon} [{h['title']}]({h['link']}) <span class='muted'>{h['source']}</span>",
                                unsafe_allow_html=True)

    with st.expander("Full ranking table"):
        t = ranked[["symbol", "sector", "score", "tech_score", "news_pts", "room_pen", "close", "chg_pct", "vol_ratio", "rsi"]].copy()
        t.columns = ["Symbol", "Sector", "Score", "Technical", "News", "Room penalty", "Close", "Change %", "Vol × avg", "RSI"]
        st.dataframe(t.round(2), use_container_width=True, hide_index=True)

# ============================================================== STOCK CHECK
with tabs[1]:
    all_syms = sorted(mw.symbol.unique())
    default = ranked.symbol.iloc[0] if len(ranked) else all_syms[0]
    sym = st.selectbox("Any PSX stock", all_syms, index=all_syms.index(default) if default in all_syms else 0)
    h = hist.get(sym)
    if h is None:
        with st.spinner("Loading…"):
            h = history_one(sym)
    if h is None or len(h) < 30:
        st.warning("Not enough price history for this stock.")
    else:
        d = A.add_indicators(h)
        sc, reasons, _ = A.score_row(d)
        p = A.trade_plan(d)
        last = d.iloc[-1]
        sh = bool(mw.loc[mw.symbol == sym, "shariah"].any())
        st.subheader(f"{sym} · {names.get(sym, '')}")
        st.caption(f"{sectors.get(sym, '')} · {'Shariah-compliant' if sh else 'Not in KMI All Share'} · "
                   f"session {d.index[-1]:%d %b}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Close", f"{last.Close:,.2f}", f"{(last.Close / d.Close.iloc[-2] - 1) * 100:+.2f}%")
        c2.metric("Setup score", f"{sc:.0f}/100")
        c3.metric("RSI (14)", f"{last.RSI:.0f}")
        c4.metric("Volume vs avg", f"{last.Volume / last.VolAvg20:.1f}×" if last.VolAvg20 else "–")
        st.markdown(f"""<div class="lv">
<div>S2<b>{p['s2']:,.2f}</b></div><div>S1 (stop)<b>{p['s1']:,.2f}</b></div>
<div>Pivot<b>{p['pivot']:,.2f}</b></div><div>R1<b>{p['r1']:,.2f}</b></div><div>R2<b>{p['r2']:,.2f}</b></div>
<div>20-day avg<b>{last.SMA20:,.2f}</b></div><div>50-day avg<b>{last.SMA50:,.2f}</b></div>
<div>MFI<b>{last.MFI:.0f}</b></div><div>CMF<b>{last.CMF:+.2f}</b></div><div>ATR<b>{last.ATR:,.2f}</b></div>
</div>""", unsafe_allow_html=True)
        st.markdown(" ".join(f'<span class="tag">{t}</span>' for t in reasons), unsafe_allow_html=True)

        dd = d.tail(90)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2],
                            vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=dd.index, open=dd.Open, high=dd.High, low=dd.Low, close=dd.Close,
                                     name="Price", showlegend=False), 1, 1)
        fig.add_trace(go.Scatter(x=dd.index, y=dd.SMA20, name="20-day", line=dict(width=1.2)), 1, 1)
        fig.add_trace(go.Scatter(x=dd.index, y=dd.SMA50, name="50-day", line=dict(width=1.2)), 1, 1)
        for lvl, nm in [(p["r1"], "R1"), (p["r2"], "R2"), (p["s1"], "S1")]:
            fig.add_hline(y=lvl, line_dash="dot", line_width=1, annotation_text=nm, row=1, col=1)
        colors = np.where(dd.Close >= dd.Open, "#1a9a5a", "#d0453f")
        fig.add_trace(go.Bar(x=dd.index, y=dd.Volume, marker_color=colors, name="Volume", showlegend=False), 2, 1)
        fig.add_trace(go.Scatter(x=dd.index, y=dd.VolAvg20, name="Avg vol", line=dict(width=1)), 2, 1)
        fig.add_trace(go.Scatter(x=dd.index, y=dd.RSI, name="RSI", line=dict(width=1.2)), 3, 1)
        fig.add_hline(y=70, line_dash="dot", line_width=1, row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_width=1, row=3, col=1)
        fig.update_layout(height=560, margin=dict(l=8, r=8, t=10, b=8), xaxis_rangeslider_visible=False,
                          legend=dict(orientation="h", y=1.02, x=0))
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        for a in ann_by_sym.get(sym, []):
            st.markdown(f"📢 {a}")
        for x in news(f'"{names.get(sym, sym)}"'):
            icon = {1: "🟢", -1: "🔴"}.get(A.headline_tone(x["title"]), "⚪")
            st.markdown(f"{icon} [{x['title']}]({x['link']}) <span class='muted'>{x['source']}</span>",
                        unsafe_allow_html=True)

# ========================================================== MARKET & FLOWS
with tabs[2]:
    st.subheader("Market mood")
    st.write(mood_txt)
    if len(ix) > 25:
        f2 = go.Figure(go.Scatter(x=ix.index[-120:], y=ix.Close.iloc[-120:], name="KSE-100"))
        f2.add_trace(go.Scatter(x=ix.index[-120:], y=ix.Close.rolling(20).mean().iloc[-120:], name="20-day"))
        f2.update_layout(height=280, margin=dict(l=8, r=8, t=10, b=8), legend=dict(orientation="h"))
        st.plotly_chart(f2, use_container_width=True, config={"displayModeBar": False})

    st.subheader("Sectors last session")
    m2 = mw.copy()
    m2["sector"] = m2.symbol.map(sectors).fillna("Other")
    m2["value"] = m2.current * m2.volume
    sec = (m2.groupby("sector")
             .agg(stocks=("symbol", "count"), avg_change=("change_pct", "mean"), traded_value=("value", "sum"))
             .sort_values("traded_value", ascending=False).head(15))
    sec["avg_change"] = sec.avg_change.round(2)
    sec["traded_value"] = (sec.traded_value / 1e6).round(0)
    sec.columns = ["Stocks", "Avg change %", "Traded value (PKR m)"]
    st.dataframe(sec, use_container_width=True)

    st.subheader("Volume leaders")
    vl = mw.sort_values("volume", ascending=False).head(15)[["symbol", "current", "change_pct", "volume", "shariah"]]
    vl.columns = ["Symbol", "Price", "Change %", "Volume", "Shariah"]
    st.dataframe(vl, use_container_width=True, hide_index=True)

    st.subheader("Foreign & local investor flows (FIPI / LIPI)")
    st.markdown(
        "NCCPL publishes daily net buying and selling by foreigners, mutual funds, banks, insurance "
        "and individuals, by sector, after the close. Its site isn't machine-readable for free, so "
        "check it directly:\n\n"
        "- [NCCPL – FIPI / LIPI reports](https://www.nccpl.com.pk/)\n"
        "- [Mettis Global – market news & flows](https://mettisglobal.news/)\n"
        "- [Business Recorder – Markets](https://www.brecorder.com/markets)")

    st.subheader("Market news")
    for x in news("Pakistan Stock Exchange KSE-100"):
        st.markdown(f"- [{x['title']}]({x['link']}) <span class='muted'>{x['source']}</span>",
                    unsafe_allow_html=True)

# ============================================================ TRACK RECORD
with tabs[3]:
    st.subheader("Would these picks have worked?")
    st.caption("Re-runs the technical ranking on each of the last sessions and checks what the top picks "
               "did the next day. News isn't included (old headlines aren't stored).")
    days = st.select_slider("Sessions to test", [10, 20, 30, 40], value=30)
    with st.spinner("Back-testing…"):
        bt = A.backtest({s: h for s, h in hist.items()}, days=days, top_n=top_n, min_avg_vol=min_avg_vol)
    if bt.empty:
        st.info("Not enough history to test yet.")
    else:
        bt["net_pct"] = bt.return_pct - cost_pct
        sm = A.backtest_summary(bt)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Picks tested", sm["trades"])
        c2.metric("Hit target (R1)", f"{sm['target_rate']:.0f}%")
        c3.metric("Hit stop (S1)", f"{sm['stop_rate']:.0f}%")
        c4.metric("Avg result after costs", f"{bt.net_pct.mean():+.2f}%")
        st.caption("'Both touched' means the next day reached both levels; the order isn't known from "
                   "daily data, so its result uses the close. A positive average after costs over many "
                   "sessions is the minimum before trusting the list with real money.")
        daily = bt.groupby("date").net_pct.mean().cumsum()
        f3 = go.Figure(go.Scatter(x=daily.index, y=daily.values, mode="lines+markers"))
        f3.update_layout(height=260, margin=dict(l=8, r=8, t=10, b=8),
                         yaxis_title="Cumulative avg % per pick")
        st.plotly_chart(f3, use_container_width=True, config={"displayModeBar": False})
        st.dataframe(bt.sort_values(["date", "rank"], ascending=[False, True]),
                     use_container_width=True, hide_index=True)

# ============================================================ HOW IT WORKS
with tabs[4]:
    st.markdown(f"""
**What the score means (0–100, plus up to ±10 for news)**

| Part | Points | Looks for |
|---|---|---|
| Trend | {A.WEIGHTS['trend']} | Close above 20-day average, 20-day above 50-day |
| Breakout | {A.WEIGHTS['breakout']} | Close above the previous 20-day high |
| Close strength | {A.WEIGHTS['close_strength']} | Closed near the day's high |
| Volume | {A.WEIGHTS['volume']} | Volume well above its 20-day average, on an up day |
| Money flow | {A.WEIGHTS['money_flow']} | OBV rising, Chaikin Money Flow > 0, MFI 50–80 |
| Momentum | {A.WEIGHTS['momentum']} | RSI 50–70 (not overbought), MACD histogram rising |
| News | ±10 | Tone of the last 2 days' headlines (simple keyword check) |
| Room penalty | −7 / −15 | Reward smaller than risk at R1 (−7), or R1 gain smaller than your costs (−15) |

**Levels** come from classic pivots on the last session: buy zone between pivot and close, targets R1/R2,
stop at S1.

**How to use it**
1. Check the list before 9:30. Skip the day if the market mood is weak.
2. Only buy inside the buy zone; don't chase a stock that opens far above it.
3. Place the sell order at R1 (or R2) and the stop at S1 straight away.
4. Watch the Track record tab. If picks don't beat costs over a few weeks, don't trust them.

**Limits**
- End-of-day data: use your broker app for live prices and the order book.
- News tone is a keyword check, not real understanding; read the headline.
- Financial results aren't scored; announcements are only flagged.
- Shariah status uses PSX's KMI All Share index membership.
- This is a rules-based screen, not investment advice.
""")
