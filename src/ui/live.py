"""Live prediction page.

This is the one part of the dashboard that is not read only. In live stream mode
it follows Yahoo Finance prices as trades print, and it can also refresh on a
timer (30 seconds, 1, 2 or 5 minutes). It applies the Stage 1 recipe (2025 Q1 to
Q2 training rows, the committed feature code and model settings) either to the
newest finished US session or, while the market is open, to today's session
with the live price as a provisional close. It also adds the one factor that
Claude proposed and the Stage 2 rules kept (NVDA intraday range relative to
AAPL), so each model is called with and without the discovered factor. It never
reads Q4 results and never selects anything using them. It is a demonstration,
not a validated forecast or a trading signal.
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
from .live_feed import LiveFeed, append_live_points, provisional_today
from src.stage2.formula import evaluate, wide_prices

TICKERS = ["AAPL", "QQQ", "MSFT", "NVDA"]
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
NY = ZoneInfo("America/New_York")
ADELAIDE = ZoneInfo("Australia/Adelaide")
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}
CLAUDE_SET = "C+Claude"
SET_LABELS = {"A": "A · AAPL", "B": "B · + QQQ", "C": "C · + peers", CLAUDE_SET: "C · + Claude factor"}
FACTOR_IN_WORDS = {
    "nvda_range_vs_aapl_range": "NVDA's high minus low as a share of its close, minus the same measure for AAPL.",
}
MODEL_LABELS = {"logistic_regression": "LR", "random_forest": "RF"}
LIVE_STREAM = "Live stream"
REFRESH_OPTIONS = {
    LIVE_STREAM: 1,
    "Every 30 seconds": 30,
    "Every minute": 60,
    "Every 2 minutes": 120,
    "Every 5 minutes": 300,
    "Off": None,
}
BASIS_PROVISIONAL = "Live price (provisional)"
BASIS_FINISHED = "Last finished session"


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


@st.cache_data(ttl=5, show_spinner=False)
def fetch_latest() -> dict:
    """Newest one minute close per ticker. Used when the stream is not delivering ticks."""
    import yfinance as yf

    raw = yf.download(TICKERS, period="1d", interval="1m", auto_adjust=True,
                      progress=False, threads=False)
    long = _to_long(raw).dropna(subset=["Close"])
    long["Date"] = pd.to_datetime(long["Date"])
    last = long.sort_values("Date").groupby("Ticker").tail(1)
    if set(last["Ticker"]) != set(TICKERS):
        raise ValueError("Yahoo Finance returned no recent prices for every ticker")
    return {r.Ticker: float(r.Close) for r in last.itertuples()}


@st.cache_resource(show_spinner=False)
def get_feed() -> LiveFeed:
    feed = LiveFeed(TICKERS)
    feed.start()
    return feed


def claude_factor(data: DashboardData) -> dict | None:
    """The factor Claude proposed that the Stage 2 rules kept, read from the committed scoreboard."""
    board = data.stage2_scoreboard
    kept = board[(board["proposer"] == "claude") & (board["status"] == "kept")]
    if kept.empty:
        return None
    row = kept.sort_values("auc_gain", ascending=False).iloc[0]
    return {"name": str(row["name"]), "formula": str(row["formula"]), "hypothesis": str(row["hypothesis"]),
            "auc_gain": float(row["auc_gain"]), "years_helped": int(row["years_helped"]),
            "max_corr": float(row["max_corr"])}


def noise_summary(data: DashboardData, threshold: float = 0.01) -> tuple[int, int]:
    """How many of the pure noise factors reached the gain bar, and how many were tried."""
    gains = data.noise_null["auc_gain"].dropna()
    return int((gains >= threshold).sum()), int(len(gains))


def q4_factor_gain(data: DashboardData) -> dict[str, float]:
    """Mean change in Q4 balanced accuracy from adding one factor, averaged over years and both models."""
    metrics = data.stage2_final_metrics
    out = {}
    for proposer in ("claude", "random"):
        sub = metrics[metrics["run_id"].str.contains(proposer)]
        key = ["year", "model"]
        base = sub[sub["features"] == "stage1_22"].set_index(key)["balanced_accuracy"]
        plus = sub[sub["features"] == "stage1_plus_llm"].set_index(key)["balanced_accuracy"]
        out[proposer] = float((plus - base).mean())
    return out


def factor_value(formula: str, prices: pd.DataFrame) -> pd.Series:
    """Evaluate a Stage 2 factor on long Ticker, Date, OHLCV prices, indexed by date."""
    return evaluate(formula, wide_prices(prices))


@st.cache_resource(show_spinner="Fitting the Stage 1 models on 2025 Q1 to Q2 ...")
def fitted_models(path: str, mtime: float, factor_name: str = "", factor_formula: str = ""):
    """Committed recipe: training rows only, Logistic Regression seed 42, Random Forest seeds 0 to 2.

    When a factor is given, a fourth feature set (the 22 Stage 1 factors plus that factor) is fitted the
    same way as the Stage 2 final test: the factor is clipped to the training 1st and 99th percentiles.
    """
    from src.data import quarterly_split
    from src.features import FEATURE_SETS, build_stage1_features
    from src.models import fit_logistic_regression, fit_random_forest

    raw = pd.read_csv(path)
    features = build_stage1_features(raw)
    sets = dict(FEATURE_SETS)
    clip = {}
    if factor_formula:
        features["Date"] = pd.to_datetime(features["Date"])
        features = features.merge(factor_value(factor_formula, raw).rename(factor_name),
                                  left_on="Date", right_index=True, how="left")
        sets[CLAUDE_SET] = list(FEATURE_SETS["C"]) + [factor_name]
    train = quarterly_split(features, 2025)["train"]
    if factor_formula:
        train = train.dropna(subset=[factor_name]).copy()
        lo, hi = train[factor_name].quantile([0.01, 0.99])
        train[factor_name] = train[factor_name].clip(lo, hi)
        clip[factor_name] = (float(lo), float(hi))
    y = train["target_next_day_up"]
    models = {}
    for exp, cols in sets.items():
        x = train[cols]
        models[(exp, "logistic_regression")] = [fit_logistic_regression(x, y, 42)]
        models[(exp, "random_forest")] = [fit_random_forest(x, y, s) for s in (0, 1, 2)]
    return models, sets, len(train), clip


def latest_feature_row(prices: pd.DataFrame, factor: dict | None = None) -> pd.Series:
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
    row = row.iloc[0].copy()
    if factor:
        # The factor only looks backwards, so its value on the last day uses data up to that day.
        row[factor["name"]] = float(factor_value(factor["formula"], prices).get(last, np.nan))
    return row


def live_calls(row: pd.Series, path: str, factor: dict | None = None) -> tuple[pd.DataFrame, int, dict | None]:
    """Model calls for one feature row. The third value describes the Claude factor, or is None without it."""
    models, feature_sets, n_train, clip = fitted_models(
        path, Path(path).stat().st_mtime,
        factor["name"] if factor else "", factor["formula"] if factor else "")
    info = None
    if factor:
        value = float(row.get(factor["name"], np.nan))
        lo, hi = clip[factor["name"]]
        info = {"value": value, "low": lo, "high": hi}
    out = []
    for (exp, model), fitted in models.items():
        if exp == CLAUDE_SET and (info is None or not np.isfinite(info["value"])):
            continue  # the factor could not be computed today, so only the Stage 1 calls are shown
        values = row[feature_sets[exp]].astype(float)
        if exp == CLAUDE_SET:
            values[factor["name"]] = float(np.clip(info["value"], lo, hi))
        x = pd.DataFrame([values])
        prob = float(np.mean([m.predict_proba(x)[0, 1] for m in fitted]))
        out.append({
            "Feature set": SET_LABELS[exp],
            "Model": MODEL_LABELS[model],
            "Probability up": prob,
            "Call": "Up" if prob >= 0.5 else "Down",
        })
    return pd.DataFrame(out), n_train, info


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
    return _finish(fig, height=max(330, 52 * len(calls)))




# ------------------------------------------------------------------- page

def _factor_section(calls: pd.DataFrame, factor: dict, info: dict, data: DashboardData) -> None:
    """Show the discovered factor, today's value, the evidence for it, and its effect on each call."""
    st.divider()
    st.subheader("The discovered factor inside the prediction")
    plain = FACTOR_IN_WORDS.get(factor["name"], factor["hypothesis"])
    noise_hits, noise_total = noise_summary(data)
    gains = q4_factor_gain(data)
    left, right = st.columns([1.4, 1])
    with left:
        st.markdown(f"**{factor['name']}** · proposed by Claude and kept by the Stage 2 rules")
        st.markdown(plain)
        st.markdown(f"*Hypothesis:* {factor['hypothesis']}")
        st.code(factor["formula"], language="text", wrap_lines=True)
    with right:
        if np.isfinite(info["value"]):
            card("Factor value now", f"{info['value']:.4f}",
                 f"Training range, 1st to 99th percentile: {info['low']:.4f} to {info['high']:.4f}")
        else:
            card("Factor value now", "Not available", "A price needed for the formula is missing today")
    e1, e2, e3 = st.columns(3)
    with e1:
        card("Stage 2 evidence", f"+{factor['auc_gain']:.4f} AUC",
             f"Mean gain on Q3, helped in {factor['years_helped']} of 4 years, "
             f"highest correlation with an existing factor {factor['max_corr']:.2f}")
    with e2:
        card("Noise check", f"{noise_hits} of {noise_total}",
             "Pure noise factors that also reached a gain of 0.01 or more")
    with e3:
        card("Q4 test, mean gain", f"{gains['claude']:+.3f} vs {gains['random']:+.3f}",
             "Balanced accuracy change from adding the Claude factor versus a random factor")
    pivot = calls.pivot(index="Model", columns="Feature set", values="Probability up")
    base, plus = SET_LABELS["C"], SET_LABELS[CLAUDE_SET]
    if base in pivot.columns and plus in pivot.columns:
        compare = pd.DataFrame({
            "Model": pivot.index,
            "22 factors, probability up": pivot[base].round(3).values,
            "Plus Claude factor, probability up": pivot[plus].round(3).values,
            "Change": (pivot[plus] - pivot[base]).round(3).values,
            "22 factors call": ["Up" if v >= 0.5 else "Down" for v in pivot[base]],
            "Plus Claude factor call": ["Up" if v >= 0.5 else "Down" for v in pivot[plus]],
        })
        st.markdown("**Does the factor change the call?**")
        st.dataframe(compare, hide_index=True, width="stretch")
    callout(
        "<strong>Read this as a hypothesis, not a result.</strong> The factor passed the Stage 2 keep rule, "
        "but pure noise factors reach the same gain bar about as often, and on the untouched Q4 test its mean "
        "gain was close to that of a random factor. A changed call here does not show that the factor works.",
        "info",
    )



def market_is_open(now_utc: datetime) -> bool:
    ny = now_utc.astimezone(NY)
    return ny.weekday() < 5 and dtime(9, 30) <= ny.time() < dtime(16, 0)


def _live_prices(live_mode: bool, today: pd.DataFrame):
    """Best available prices: stream ticks, then 5 second polling, then the last 5 minute bar."""
    if live_mode:
        feed = get_feed()
        ticks = feed.fresh_prices()
        if ticks:
            when = pd.Timestamp(datetime.fromtimestamp(max(t.trade_time for t in ticks.values()), tz=NY))
            return {s: t.price for s, t in ticks.items()}, "stream", when
        try:
            return fetch_latest(), "polling", pd.Timestamp(datetime.now(NY))
        except Exception:
            pass
    last_bar = {t: float(today[today["Ticker"] == t].sort_values("Date")["Close"].iloc[-1]) for t in TICKERS}
    return last_bar, "bars", today["Date"].max()


def _intraday_state():
    """Today's 5 minute bars and earlier ones, from the cached download."""
    intra = fetch_intraday()
    session_date = intra["Date"].dt.date.max()
    today = intra[intra["Date"].dt.date == session_date]
    prev = intra[intra["Date"].dt.date < session_date]
    return session_date, today, prev


def _status_line(live_mode: bool, mode: str, open_now: bool) -> str:
    if not live_mode:
        return ("Refreshing on a timer. Choose Live stream to follow prices as they change.")
    if not open_now:
        return ":gray[●] US market closed, so no new prices are expected until it opens. Showing the latest prices."
    feed = get_feed()
    if mode == "stream":
        return f":green[●] Live stream connected, {feed.describe()}."
    if mode == "polling":
        return (f":orange[●] The stream is not delivering ticks ({feed.describe()}), "
                "so prices are polled from Yahoo Finance every 5 seconds instead.")
    return ":red[●] Live prices are unavailable right now, so the latest 5 minute bar is shown."


def _calls_for(prices: pd.DataFrame, csv_path: str, key: tuple, factor: dict | None = None):
    """Recompute the calls only when the inputs changed, so a one second refresh stays cheap."""
    cache = st.session_state.setdefault("_live_calls", {})
    if key not in cache:
        row = latest_feature_row(prices, factor)
        calls, n_train, info = live_calls(row, csv_path, factor)
        cache.clear()
        cache[key] = (pd.Timestamp(row["Date"]).date(), calls, n_train, info)
    return cache[key]


def live(data: DashboardData) -> None:
    hero(
        "Live prediction · Research demonstration",
        "Stage 1 models and the discovered factor on current prices",
        "In live stream mode prices and the model calls update as trades print. Timer modes refresh every "
        "30 seconds, 1, 2 or 5 minutes. This page is separate from the frozen results elsewhere in the dashboard.",
    )
    callout(
        "<strong>Not a trading signal.</strong> On the untouched Q4 test period these models performed at about "
        "coin toss level, so a call here is a model output to examine, not a forecast to act on. "
        "This page never reads or uses Q4 results.",
        "warning",
    )

    csv_path = str(ROOT / data.manifest["canonical_data"])
    factor = claude_factor(data)
    c1, c2, c3 = st.columns([2, 2, 3])
    choice = c1.selectbox("Refresh mode", list(REFRESH_OPTIONS), index=0)
    basis = c2.radio("Model call basis", [BASIS_PROVISIONAL, BASIS_FINISHED])
    c3.caption(
        "Live stream follows Yahoo Finance as trades print (price boxes every second, chart and calls every 5 seconds), and falls back to "
        "polling every 5 seconds if the stream is blocked. Timer modes re-download on that interval. "
        "Prices come from Yahoo Finance and may differ slightly from an exchange feed. "
        "Provisional calls use today's session so far with the live price as the close. "
        "Last finished session uses only completed days and changes once per trading day."
    )
    live_mode = choice == LIVE_STREAM
    interval = REFRESH_OPTIONS[choice]
    # In live stream mode the cheap price boxes refresh every second and the heavier chart and model
    # calls every 5 seconds, so the page stays responsive. Timer modes refresh everything together.
    heavy_interval = 5 if live_mode else interval

    @st.fragment(run_every=interval)
    def price_boxes() -> None:
        now_utc = datetime.now(timezone.utc)
        top, button = st.columns([5, 1])
        top.caption(market_status(now_utc))
        if button.button("Refresh now"):
            fetch_daily.clear()
            fetch_intraday.clear()
            fetch_latest.clear()
        try:
            session_date, today, prev = _intraday_state()
            live_prices, mode, _when = _live_prices(live_mode, today)
        except Exception:
            return  # the panel below explains what went wrong
        st.markdown(_status_line(live_mode, mode, market_is_open(now_utc)))
        cols = st.columns(len(TICKERS))
        for col, t in zip(cols, TICKERS):
            one = today[today["Ticker"] == t].sort_values("Date")
            last = float(live_prices[t])
            ref_rows = prev[prev["Ticker"] == t].sort_values("Date")
            ref = float(ref_rows["Close"].iloc[-1]) if not ref_rows.empty else float(one["Close"].iloc[0])
            col.metric(t, f"${last:,.2f}", f"{(last / ref - 1) * 100:+.2f}% vs previous close")

    @st.fragment(run_every=heavy_interval)
    def panel() -> None:
        now_utc = datetime.now(timezone.utc)
        now_ny = pd.Timestamp(now_utc.astimezone(NY))

        offline_reason = None
        try:
            prices, partial = drop_unfinished_session(fetch_daily(), now_utc)
            source, source_note = "Yahoo Finance (live)", "Falls back to saved data if the download fails"
        except Exception as exc:  # network, package or format problem
            offline_reason = f"{type(exc).__name__}: {exc}"
            prices, partial = clean_prices(pd.read_csv(csv_path)), False
            source, source_note = "Saved file (offline)", "The live download failed"
        if offline_reason:
            st.warning(
                "Live download failed, so the saved repository data is shown instead. "
                f"Details: {offline_reason}"
            )

        live_info = None
        if offline_reason is None:
            try:
                session_date, today, _prev = _intraday_state()
                live_prices, mode, when = _live_prices(live_mode, today)
                # Only extend the chart while the market is open. After the close the last bar is final,
                # and adding a point at the current time would draw a long flat line across the night.
                chart_today = (append_live_points(today, live_prices, when)
                               if market_is_open(now_utc) else today)
                live_info = (live_prices, today, session_date)
                if mode == "stream":
                    source, source_note = "Yahoo Finance (live stream)", get_feed().describe().capitalize()
                st.plotly_chart(_intraday_figure(chart_today), width="stretch", config=PLOT_CONFIG)
                st.caption(f"Session shown: {session_date:%a %d %b %Y}, 5 minute bars plus the live price. "
                           f"Latest point: {chart_today['Date'].max():%H:%M:%S} New York time.")
            except Exception as exc:
                st.info(f"Intraday chart unavailable right now ({type(exc).__name__}: {exc}). "
                        "The model call below still works.")

        provisional = None
        if live_info is not None and basis == BASIS_PROVISIONAL:
            live_prices, today, session_date = live_info
            provisional = provisional_today(prices, today, live_prices, session_date, now_ny)
        try:
            if provisional is not None:
                key = ("provisional", tuple(round(live_prices[t], 2) for t in TICKERS), provisional["Date"].max())
                asof, calls, n_train, info = _calls_for(provisional, csv_path, key, factor)
            else:
                asof, calls, n_train, info = _calls_for(prices, csv_path, ("finished", prices["Date"].max()), factor)
        except Exception as exc:
            st.error(f"Could not compute the model calls: {type(exc).__name__}: {exc}")
            return

        if provisional is not None:
            st.subheader(f"Provisional model calls for today's session, {asof:%a %d %b %Y}")
            st.caption(
                "Built from today's open, high and low so far, with the live price as the close and volume "
                "projected to a full session. These calls move as prices move. The final calls for today are "
                "made after the close."
            )
            basis_note = "Live price, provisional"
        else:
            st.subheader(f"Model calls for the session after {asof:%a %d %b %Y}")
            if partial:
                st.caption("Today's US session is still in progress and is excluded from the features.")
            elif basis == BASIS_PROVISIONAL and live_info is not None:
                st.caption("The market is closed or today's session is complete, so the last finished session is used.")
            basis_note = "Last finished session"
        k1, k2, k3 = st.columns(3)
        n_sets = calls["Feature set"].nunique()
        with k1:
            card("Models calling Up", f"{int((calls['Call'] == 'Up').sum())} of {len(calls)}",
                 f"{n_sets} feature sets, two models each. Basis: {basis_note}")
        with k2:
            card("Mean probability up", f"{calls['Probability up'].mean():.1%}",
                 f"Average over the {len(calls)} calls")
        with k3:
            card("Data source", source, source_note)
        st.plotly_chart(_probability_figure(calls), width="stretch", config=PLOT_CONFIG)
        st.dataframe(calls.assign(**{"Probability up": calls["Probability up"].round(3)}),
                     hide_index=True, width="stretch")
        if factor is not None and info is not None:
            _factor_section(calls, factor, info, data)
        st.caption(
            f"Models fitted once on the {n_train} training rows of 2025 Q1 to Q2, as in the committed pipeline. "
            f"Last updated {datetime.now(ADELAIDE):%H:%M:%S} Adelaide time."
        )

    price_boxes()
    panel()
