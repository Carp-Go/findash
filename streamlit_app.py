"""FinDash — single-file cloud edition (Streamlit Community Cloud).

Self-contained: market data (Stooq), macro (FRED), geo risk (GDELT),
cross-sectional factors. Free/public sources, polite caching, attributed.
"""
from __future__ import annotations

import io

import httpx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

UA = {"User-Agent": "FinDash prototype (research dashboard; contact via GitHub repo)"}

st.set_page_config(page_title="FinDash", layout="wide")
st.title("FinDash — Global Financial Analytics (prototype)")
st.caption("Sources: Stooq, FRED (St. Louis Fed), GDELT — free/public tiers, attributed. "
           "Cache TTL 5–15 min.")

INDICES = {  # name: (stooq symbol, FRED fallback series or None)
    "S&P 500": ("^spx", "SP500"), "Nasdaq 100": ("^ndx", "NASDAQ100"),
    "Dow Jones": ("^dji", "DJIA"), "Nikkei 225": ("^nkx", "NIKKEI225"),
    "DAX": ("^dax", None), "FTSE 100": ("^ukx", None),
    "EUR/USD": ("eurusd", "DEXUSEU"), "WTI Crude": ("cl.f", "DCOILWTICO"),
    "Brent": ("cb.f", "DCOILBRENTEU"), "Bitcoin/USD": ("btcusd", "CBBTCUSD"),
}
MACRO = {
    "US 10Y yield": "DGS10", "10Y-2Y curve": "T10Y2Y",
    "CPI (index)": "CPIAUCSL", "Unemployment %": "UNRATE", "VIX": "VIXCLS",
}


import time

FRED_MARKET_IDS = ("SP500", "NASDAQ100", "DJIA", "NIKKEI225", "DEXUSEU",
                   "DCOILWTICO", "DCOILBRENTEU", "CBBTCUSD")
FRED_MACRO_IDS = ("DGS10", "T10Y2Y", "CPIAUCSL", "UNRATE", "VIXCLS")


def _fred_key() -> str:
    try:
        return st.secrets.get("FRED_API_KEY", "")
    except Exception:
        return ""


@st.cache_data(ttl=900, show_spinner=False)
def _fred_api_series(series: str) -> pd.DataFrame:
    """Official FRED API (free key, 120 req/min) — the reliable path."""
    try:
        r = httpx.get("https://api.stlouisfed.org/fred/series/observations",
                      params={"series_id": series, "api_key": _fred_key(),
                              "file_type": "json", "observation_start": "2015-01-01"},
                      headers=UA, timeout=25)
        obs = r.json().get("observations", [])
        df = pd.DataFrame(obs)
        if df.empty:
            return df
        df = df[["date", "value"]]
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.dropna(subset=["value"])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=120, show_spinner=False)
def _fred_table(ids: tuple) -> pd.DataFrame:
    """Keyless fallback: ONE request for many series."""
    try:
        r = httpx.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                      params={"id": ",".join(ids)}, headers=UA, timeout=20,
                      follow_redirects=True)
        if r.status_code == 200 and r.text.lower().startswith("observation_date"):
            df = pd.read_csv(io.StringIO(r.text), na_values=".")
            df = df.rename(columns={df.columns[0]: "date"})
            df["date"] = pd.to_datetime(df["date"])
            return df
    except Exception:
        pass
    return pd.DataFrame()


def _fred_csv(series: str) -> pd.DataFrame:
    if _fred_key():
        df = _fred_api_series(series)
        if not df.empty:
            return df
    table = _fred_table(FRED_MARKET_IDS if series in FRED_MARKET_IDS else FRED_MACRO_IDS)
    if table.empty or series not in table.columns:
        return pd.DataFrame()
    out = table[["date", series]].rename(columns={series: "value"})
    return out.dropna(subset=["value"])


@st.cache_data(ttl=300, show_spinner=False)
def load_prices(key: str) -> pd.DataFrame:
    stooq_sym, fred_id = INDICES[key]
    try:
        r = httpx.get("https://stooq.com/q/d/l/", params={"s": stooq_sym, "i": "d"},
                      headers=UA, timeout=6, follow_redirects=True)
        df = pd.read_csv(io.StringIO(r.text))
        if "Close" in df.columns:
            df.columns = [c.lower() for c in df.columns]
            df["date"] = pd.to_datetime(df["date"])
            return df
    except Exception:
        pass
    if fred_id:  # fallback: FRED close-only series
        f = _fred_csv(fred_id)
        if not f.empty:
            return f.rename(columns={"value": "close"})
    return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def load_macro(series: str) -> pd.DataFrame:
    return _fred_csv(series)


@st.cache_data(ttl=900, show_spinner=False)
def load_geo(query: str, mode: str = "ArtList") -> pd.DataFrame:
    try:
        r = httpx.get("https://api.gdeltproject.org/api/v2/doc/doc",
                      params={"query": query, "mode": mode, "timespan": "24h",
                              "maxrecords": 75, "format": "json"},
                      headers=UA, timeout=25, follow_redirects=True)
        data = r.json()
        if mode == "ArtList":
            return pd.DataFrame(data.get("articles", []))
        tl = data.get("timeline", [{}])
        df = pd.DataFrame(tl[0].get("data", [])) if tl else pd.DataFrame()
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%SZ")
        return df
    except Exception:
        return pd.DataFrame()


def zscore(s: pd.Series) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    mu, sd = s.mean(), s.std()
    s = s.clip(mu - 3 * sd, mu + 3 * sd)
    return (s - s.mean()) / s.std(ddof=0)


tab_mkt, tab_cmp, tab_macro, tab_geo, tab_factor = st.tabs(
    ["Markets", "Comparative", "Macro", "Geo Risk", "Factors"])

with tab_mkt:
    cols = st.columns(5)
    for i, name in enumerate(INDICES):
        df = load_prices(name)
        if df.empty or len(df) < 2:
            cols[i % 5].metric(name, "n/a")
            continue
        last, prev = df["close"].iloc[-1], df["close"].iloc[-2]
        cols[i % 5].metric(name, f"{last:,.2f}", f"{(last/prev-1)*100:+.2f}%")
    sel = st.selectbox("Chart", list(INDICES))
    df = load_prices(sel)
    if not df.empty:
        d = df.tail(250)
        if "open" in d.columns:
            fig = go.Figure(go.Candlestick(x=d["date"], open=d["open"], high=d["high"],
                                           low=d["low"], close=d["close"]))
        else:
            fig = px.line(d, x="date", y="close")
        fig.update_layout(height=420, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Market data: Stooq (EOD) with FRED fallback.")

with tab_cmp:
    picks = st.multiselect("Compare (normalized to 100)",
                           list(INDICES), default=["S&P 500", "DAX", "Gold"])
    lookback = st.slider("Days", 30, 1000, 250)
    series = {}
    for p in picks:
        d = load_prices(p)
        if not d.empty:
            s = d.set_index("date")["close"].tail(lookback)
            series[p] = s / s.iloc[0] * 100
    if series:
        cmp_df = pd.DataFrame(series).dropna()
        st.plotly_chart(px.line(cmp_df, height=420), use_container_width=True)
        st.subheader("Rolling 60d correlation matrix")
        st.dataframe(cmp_df.pct_change().tail(60).corr().round(2),
                     use_container_width=True)

with tab_macro:
    any_macro = False
    cols = st.columns(len(MACRO))
    for i, (name, sid) in enumerate(MACRO.items()):
        d = load_macro(sid)
        if not d.empty:
            any_macro = True
            cols[i].metric(name, f"{d['value'].iloc[-1]:,.2f}")
        else:
            cols[i].metric(name, "n/a")
    if not any_macro:
        st.warning("FRED is unreachable from this server right now.")
    sel_m = st.selectbox("Series", list(MACRO))
    d = load_macro(MACRO[sel_m])
    if not d.empty:
        st.plotly_chart(px.line(d.tail(2000), x="date", y="value", height=380),
                        use_container_width=True)
    st.caption("Source: FRED, Federal Reserve Bank of St. Louis (and original sources).")

with tab_geo:
    q = st.text_input("GDELT query", '(conflict OR sanctions OR "military escalation")')
    tone = load_geo(q, mode="TimelineTone")
    if not tone.empty and "value" in tone.columns:
        st.subheader("Average news tone (lower = more negative)")
        st.plotly_chart(px.line(tone, x="date", y="value", height=280),
                        use_container_width=True)
    arts = load_geo(q)
    if not arts.empty:
        show = [c for c in ("title", "domain", "seendate", "sourcecountry")
                if c in arts.columns]
        st.subheader("Latest coverage")
        st.dataframe(arts[show].head(25), use_container_width=True, hide_index=True)
    st.caption("Source: GDELT Project (free & open).")

with tab_factor:
    st.write("Cross-sectional factors across the index universe (proxy demo on "
             "index level; production runs on constituents).")
    panel = {}
    for name in INDICES:
        d = load_prices(name)
        if not d.empty:
            panel[name] = d.set_index("date")["close"]
    pdf = pd.DataFrame(panel).dropna(how="all").ffill().dropna()
    if len(pdf) > 300:
        rows = {}
        rows["mom_12_1"] = pdf.shift(21).iloc[-1] / pdf.shift(252).iloc[-1] - 1
        rows["reversal_1m"] = -(pdf.iloc[-1] / pdf.shift(21).iloc[-1] - 1)
        rows["low_vol"] = -pdf.pct_change().tail(252).std() * np.sqrt(252)
        out = pd.DataFrame({k: zscore(v) for k, v in rows.items()}).round(2)
        st.dataframe(out, use_container_width=True)
        st.caption("z-scores, winsorized ±3σ, point-in-time safe by construction.")
    else:
        st.info("Not enough overlapping history loaded yet.")
