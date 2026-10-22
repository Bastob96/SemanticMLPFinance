"""Live prediction page.

This is the one part of the dashboard that is not read only. It downloads
current prices, refreshes on a timer, and applies the Stage 1 recipe (2025 Q1 to
Q2 training rows, the committed feature code and model settings) to the newest
finished US session. It never reads Q4 results and never selects anything using
them. It is a demonstration, not a validated forecast or a trading signal.
"""

from __future__ import annotations

from datetime import datetime, time as dtime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .charts import BLUE, INK, MUTED, RED, TEAL, _finish
from .components import callout, card, hero
from .data import ROOT, DashboardData

TICKERS = ["AAPL", "QQQ", "MSFT", "NVDA"]
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
NY = ZoneInfo("America/New_York")
ADELAIDE = ZoneInfo("Australia/Adelaide")
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}
SET_LABELS = {"A": "A · AAPL", "B": "B · + QQQ", "C": "C · + peers"}
MODEL_LABELS = {"logistic_regression": "LR", "random_forest": "RF"}
REFRESH_OPTIONS = {"Off": None, "Every 30 seconds": 30, "Every minute": 60, "Every 5 minutes": 300}


# ------------------------------------------------------------------ data

def _to_long(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn a yfinance multi ticker download into Ticker, Date, OHLCV rows."""
    if raw is None or raw.empty or not isinstance(raw.columns, pd.MultiIndex):
        raise ValueError("Yahoo Finance returned no usable data")
    if set(raw.columns.get_level_values(0)) & set(TICKERS):
        raw = raw.swaplevel(axis=1)
    frames = []
    for ticker in TICKERS:
        sub = raw.xs(ticker, axis=1, level=1)[OHLCV].copy()
        sub.index.name = "Date"
        sub.insert(0, "Ticker", ticker)
        frames.append(sub.reset_index())
    return pd.concat(frames, ignore_index=True)


def clean_prices(long: pd.DataFrame) -> pd.DataFrame:
    """Drop incomplete rows and any date that is missing one of the four tickers."""
    data = long.copy()
    data["Date"] = pd.to_datetime(data["Date"]).dt.tz_localize(None).dt.normalize()
    data = data.dropna(subset=OHLCV).drop_duplicates(["Ticker", "Date"])
    counts = data.groupby("Date")["Ticker"].nunique()
    data = data[data["Date"].isin(counts[counts == len(TICKERS)].index)]
    return data.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def drop_unfinished_session(data: pd.DataFrame, now_utc: datetime) -> tuple[pd.DataFrame, bool]:
    """Remove today's US session until the market has closed, so features use finished days."""
    now_ny = now_utc.astimezone(NY)
    latest = data["Date"].max()
    unfinished = latest.date() >= now_ny.date() and now_ny.time() < dtime(16, 30)
    if unfinished:
        return data[data["Date"] < latest].reset_index(drop=True), True
    return data, False


@st.cache_data(ttl=60, show_spinner=False)
def fetch_daily() -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(TICKERS, period="6mo", interval="1d", auto_adjust=True,
                      progress=False, threads=False)
    return clean_prices(_to_long(raw))


@st.cache_data(ttl=30, show_spinner=False)
def fetch_intraday() -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(TICKERS, period="5d", interval="5m", auto_adjust=True,
                      progress=False, threads=False)
    long = _to_long(raw)
    long["Date"] = pd.to_datetime(long["Date"])
    if long["Date"].dt.tz is not None:
        long["Date"] = long["Date"].dt.tz_convert(NY)
    else:
        long["Date"] = long["Date"].dt.tz_localize("UTC").dt.tz_convert(NY)
    return long.dropna(subset=["Close"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)


@st.cache_resource(show_spinner="Fitting the Stage 1 models on 2025 Q1 to Q2 ...")
def fitted_models(path: str, mtime: float):
    """Committed recipe: training rows only, Logistic Regression seed 42, Random Forest seeds 0 to 2."""
    from src.data import quarterly_split
    from src.features import FEATURE_SETS, build_stage1_features
    from src.models import fit_logistic_regression, fit_random_forest

    features = build_stage1_features(pd.read_csv(path))
    train = quarterly_split(features, 2025)["train"]
    y = train["target_next_day_up"]
    models = {}
    for exp, cols in FEATURE_SETS.items():
        x = train[cols]
        models[(exp, "logistic_regression")] = [fit_logistic_regression(x, y, 42)]
        models[(exp, "random_forest")] = [fit_random_forest(x, y, s) for s in (0, 1, 2)]
    return models, dict(FEATURE_SETS), len(train)


def latest_feature_row(prices: pd.DataFrame) -> pd.Series:
    """Features for the newest finished day, using the repository's own feature code.

    The repository builder drops the last day because its label is unknown. A
    placeholder next day is added only so that the real last day survives. Every
    feature at day t uses data up to t, so the placeholder never influences them.
    """
    from src.features import build_stage1_features

    last = prices["Date"].max()
    placeholder = prices[prices["Date"] == last].copy()
    placeholder["Date"] = last + pd.Timedelta(days=1)
    features = build_stage1_features(pd.concat([prices, placeholder], ignore_index=True))
    row = features[features["Date"] == last]
    if row.empty:
        raise ValueError("Not enough history to build features for the latest day")
    return row.iloc[0]


def live_calls(row: pd.Series, path: str) -> tuple[pd.DataFrame, int]:
    models, feature_sets, n_train = fitted_models(path, Path(path).stat().st_mtime)
    out = []
    for (exp, model), fitted in models.items():
        x = pd.DataFrame([row[feature_sets[exp]].astype(float)])
        prob = float(np.mean([m.predict_proba(x)[0, 1] for m in fitted]))
        out.append({
            "Feature set": SET_LABELS[exp],
            "Model": MODEL_LABELS[model],
            "Probability up": prob,
            "Call": "Up" if prob >= 0.5 else "Down",
        })
    return pd.DataFrame(out), n_train


def market_status(now_utc: datetime) -> str:
    ny = now_utc.astimezone(NY)
    open_now = ny.weekday() < 5 and dtime(9, 30) <= ny.time() < dtime(16, 0)
    stamp = ny.strftime("%a %H:%M New York time")
    return (f"US market appears open ({stamp})" if open_now
            else f"US market appears closed ({stamp})") + ". Public holidays are not checked."


# ----------------------------------------------------------------- charts

def _intraday_figure(today: pd.DataFrame) -> go.Figure:
    series = []
    for t in TICKERS:
        one = today[today["Ticker"] == t].sort_values("Date")
        base = float(one["Close"].iloc[0])
        series.append(pd.DataFrame({
            "Time (New York)": one["Date"].dt.tz_localize(None),
            "Change since session open (%)": (one["Close"] / base - 1) * 100,
            "Ticker": t,
        }))
    frame = pd.concat(series)
    fig = px.line(frame, x="Time (New York)", y="Change since session open (%)", color="Ticker",
                  color_discrete_map={"AAPL": BLUE, "QQQ": TEAL, "MSFT": MUTED, "NVDA": "#f59e0b"})
    fig.update_traces(line_width=1.6)
    fig.update_traces(line_width=3.2, selector=dict(name="AAPL"))
    return _finish(fig, height=380, y_title="Change since session open (%)")


def _probability_figure(calls: pd.DataFrame) -> go.Figure:
    labels = calls["Feature set"] + " · " + calls["Model"]
    fig = go.Figure(go.Bar(
        x=calls["Probability up"], y=labels, orientation="h",
        marker_color=[TEAL if c == "Up" else RED for c in calls["Call"]],
        text=[f"{p:.3f} · {c}" for p, c in zip(calls["Probability up"], calls["Call"])],
        textposition="outside",
        hovertemplate="%{y}<br>Probability up %{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=0.5, line_dash="dash", line_color=INK)
    fig.update_xaxes(range=[0, 1], title="Predicted probability that AAPL closes up")
    fig.update_yaxes(autorange="reversed")
    return _finish(fig, height=330)


# ------------------------------------------------------------------- page

def live(data: DashboardData) -> None:
    hero(
        "Live prediction · Research demonstration",
        "Stage 1 models on current prices",
        "Prices refresh on a timer and the Stage 1 recipe is applied to the newest finished US session. "
        "This page is separate from the frozen results elsewhere in the dashboard.",
    )
    callout(
        "<strong>Not a trading signal.</strong> On the untouched Q4 test period these models performed at about "
        "coin toss level, so a call here is a model output to examine, not a forecast to act on. "
        "This page never reads or uses Q4 results.",
        "warning",
    )

    csv_path = str(ROOT / data.manifest["canonical_data"])
    c1, c2 = st.columns([2, 3])
    choice = c1.selectbox("Auto refresh", list(REFRESH_OPTIONS), index=1)
    c2.caption(
        "The price panel and the model calls update by themselves on this timer, with no page reload. "
        "Prices come from Yahoo Finance and may be delayed. The model call uses the last finished US session, "
        "so it changes once per trading day, while the intraday chart moves during the session."
    )

    @st.fragment(run_every=REFRESH_OPTIONS[choice])
    def panel() -> None:
        now_utc = datetime.now(timezone.utc)
        top, button = st.columns([5, 1])
        top.caption(market_status(now_utc))
        if button.button("Refresh now"):
            fetch_daily.clear()
            fetch_intraday.clear()

        offline_reason = None
        try:
            prices, partial = drop_unfinished_session(fetch_daily(), now_utc)
            source = "Yahoo Finance (live)"
        except Exception as exc:  # network, package or format problem
            offline_reason = f"{type(exc).__name__}: {exc}"
            prices, partial = clean_prices(pd.read_csv(csv_path)), False
            source = "Saved file (offline)"
        if offline_reason:
            st.warning(
                "Live download failed, so the saved repository data is shown instead. "
                f"Details: {offline_reason}"
            )

        if offline_reason is None:
            try:
                intra = fetch_intraday()
                session_date = intra["Date"].dt.date.max()
                today = intra[intra["Date"].dt.date == session_date]
                prev = intra[intra["Date"].dt.date < session_date]
                cols = st.columns(len(TICKERS))
                for col, t in zip(cols, TICKERS):
                    one = today[today["Ticker"] == t].sort_values("Date")
                    last = float(one["Close"].iloc[-1])
                    ref_rows = prev[prev["Ticker"] == t].sort_values("Date")
                    ref = float(ref_rows["Close"].iloc[-1]) if not ref_rows.empty else float(one["Close"].iloc[0])
                    col.metric(t, f"${last:,.2f}", f"{(last / ref - 1) * 100:+.2f}% vs previous close")
                st.plotly_chart(_intraday_figure(today), width="stretch", config=PLOT_CONFIG)
                st.caption(f"Session shown: {session_date:%a %d %b %Y}, 5 minute bars. "
                           f"Latest bar: {today['Date'].max():%H:%M} New York time.")
            except Exception as exc:
                st.info(f"Intraday chart unavailable right now ({type(exc).__name__}: {exc}). "
                        "The model call below still works.")

        try:
            row = latest_feature_row(prices)
            calls, n_train = live_calls(row, csv_path)
        except Exception as exc:
            st.error(f"Could not compute the model calls: {type(exc).__name__}: {exc}")
            return

        asof = pd.Timestamp(row["Date"]).date()
        st.subheader(f"Model calls for the session after {asof:%a %d %b %Y}")
        if partial:
            st.caption("Today's US session is still in progress and is excluded from the features.")
        k1, k2, k3 = st.columns(3)
        with k1:
            card("Models calling Up", f"{int((calls['Call'] == 'Up').sum())} of {len(calls)}",
                 "Three feature sets, two models each")
        with k2:
            card("Mean probability up", f"{calls['Probability up'].mean():.1%}",
                 "Average over the six calls")
        with k3:
            card("Data source", source, "Falls back to saved data if the download fails")
        st.plotly_chart(_probability_figure(calls), width="stretch", config=PLOT_CONFIG)
        st.dataframe(calls.assign(**{"Probability up": calls["Probability up"].round(3)}),
                     hide_index=True, width="stretch")
        st.caption(
            f"Models fitted once on the {n_train} training rows of 2025 Q1 to Q2, as in the committed pipeline. "
            f"Last updated {datetime.now(ADELAIDE):%H:%M:%S} Adelaide time."
        )

    panel()
