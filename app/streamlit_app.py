"""PG-S2-36 results and live prediction app.

Run from the repository root:

    streamlit run app/streamlit_app.py --server.runOnSave true

Two kinds of "live":
  1. Live prediction page: downloads current prices, refreshes on a timer, and
     recomputes the Stage 1 model calls without a page reload.
  2. Development: with --server.runOnSave true the page reloads by itself when
     this file is saved, and any edited results file is picked up on the next
     interaction (files are cached by modification time, not forever).

Research demonstration only. Not financial advice.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
STAGE2 = ROOT / "Stage2_Factor_Files"
DATA = ROOT / "data" / "processed" / "aligned_daily_ohlcv_5y.csv"

SOURCES = {
    "Original committed run": RESULTS / "stage1_results_summary.csv",
    "Re-verified run (Abu, 16 Sep 2026)": RESULTS / "abu_stage1_2025" / "evaluation_comparison.csv",
}
PREDICTIONS = {
    "Original committed run": RESULTS / "stage1_predictions.csv",
    "Re-verified run (Abu, 16 Sep 2026)": RESULTS / "abu_stage1_2025" / "stage1_predictions.csv",
}
BACKTEST_DAILY = RESULTS / "abu_stage1_2025" / "backtest_daily.csv"
BACKTEST_SUMMARY = RESULTS / "abu_stage1_2025" / "backtest_summary.csv"
CAND_DIR = STAGE2 / "data" / "generated" / "run_20260913T011756Z"
CANDIDATES = CAND_DIR / "validated_candidates.json"
CANDIDATE_META = CAND_DIR / "generation_metadata.json"
STAGE2_VALIDATION = RESULTS / "stage2_local_validation" / "stage2_validation_comparison.csv"
PHASE2_RESULTS = RESULTS / "phase2_abc_validation_test_results(taiyu).csv"
PHASE2_IMPORTANCE = RESULTS / "phase2_experiment_c_factor_importance(Taiyu).csv"

EXPERIMENT_LABELS = {
    "baseline": "Baseline",
    "A": "A: AAPL only",
    "B": "B: AAPL + QQQ",
    "C": "C: AAPL + QQQ + MSFT + NVDA",
}
MODEL_LABELS = {
    "majority_class": "Majority class",
    "previous_direction": "Previous direction",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}
METRICS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "auc": "AUC",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
}

DISCLAIMER = (
    "Academic research only. Nothing here is financial advice, a trading signal or a "
    "forecast of future prices. Results come from one 63 day test period (Q4 2025) and "
    "are close to chance."
)
LIVE_DISCLAIMER = (
    "Research demonstration only. This is not financial advice and not a trading signal. "
    "On the untouched test period these models performed at about coin toss level, so a call "
    "here is a model output to examine, not a forecast to act on."
)

TICKERS = ["AAPL", "QQQ", "MSFT", "NVDA"]
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
NY = ZoneInfo("America/New_York")
ADELAIDE = ZoneInfo("Australia/Adelaide")


# --------------------------------------------------------------- file loading
# Files are cached by modification time, so editing a results file changes the
# page on the next interaction instead of staying stale for the whole session.

@st.cache_data(show_spinner=False)
def _read_csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _read_json(path: str, mtime: float):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_csv(path: Path) -> pd.DataFrame | None:
    return _read_csv(str(path), path.stat().st_mtime) if path.is_file() else None


def load_json(path: Path):
    return _read_json(str(path), path.stat().st_mtime) if path.is_file() else None


def missing(path: Path) -> None:
    st.warning(f"File not found: `{path.relative_to(ROOT)}`. Pull the latest repository files.")


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def side_note() -> None:
    st.sidebar.divider()
    st.sidebar.caption(DISCLAIMER)


# ----------------------------------------------------------------- live data

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
def fetch_daily() -> tuple[pd.DataFrame, str]:
    import yfinance as yf

    raw = yf.download(
        TICKERS, period="6mo", interval="1d", auto_adjust=True,
        progress=False, threads=False,
    )
    return clean_prices(_to_long(raw)), datetime.now(timezone.utc).isoformat()


@st.cache_data(ttl=30, show_spinner=False)
def fetch_intraday() -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        TICKERS, period="5d", interval="5m", auto_adjust=True,
        progress=False, threads=False,
    )
    long = _to_long(raw)
    long["Date"] = pd.to_datetime(long["Date"])
    if long["Date"].dt.tz is not None:
        long["Date"] = long["Date"].dt.tz_convert(NY)
    else:
        long["Date"] = long["Date"].dt.tz_localize("UTC").dt.tz_convert(NY)
    return long.dropna(subset=["Close"]).sort_values(["Date", "Ticker"]).reset_index(drop=True)


@st.cache_resource(show_spinner="Fitting the Stage 1 models on 2025 Q1 to Q2 ...")
def fitted_models(mtime: float):
    """Same recipe as the committed pipeline: training rows only, seeds 42 and 0,1,2."""
    from src.data.stage1_split import quarterly_split
    from src.features.stage1_features import FEATURE_SETS, build_stage1_features
    from src.models.stage1_models import fit_logistic_regression, fit_random_forest

    features = build_stage1_features(pd.read_csv(DATA))
    train = quarterly_split(features, 2025)["train"]
    y = train["target_next_day_up"]
    models = {}
    for exp, cols in FEATURE_SETS.items():
        x = train[cols]
        models[(exp, "logistic_regression")] = [fit_logistic_regression(x, y, 42)]
        models[(exp, "random_forest")] = [fit_random_forest(x, y, s) for s in (0, 1, 2)]
    return models, {e: c for e, c in FEATURE_SETS.items()}, len(train)


def latest_feature_row(prices: pd.DataFrame) -> pd.Series:
    """Features for the newest finished day, using the repository's own feature code.

    The repository builder drops the last day because its label is unknown. A
    placeholder next day is added only so that the real last day survives. Every
    feature at day t uses data up to t, so the placeholder never influences them.
    """
    from src.features.stage1_features import build_stage1_features

    last = prices["Date"].max()
    last_rows = prices[prices["Date"] == last].copy()
    placeholder = last_rows.copy()
    placeholder["Date"] = last + pd.Timedelta(days=1)
    features = build_stage1_features(pd.concat([prices, placeholder], ignore_index=True))
    row = features[features["Date"] == last]
    if row.empty:
        raise ValueError("Not enough history to build features for the latest day")
    return row.iloc[0]


def live_calls(row: pd.Series) -> pd.DataFrame:
    models, feature_sets, _ = fitted_models(DATA.stat().st_mtime)
    out = []
    for (exp, model), fitted in models.items():
        x = pd.DataFrame([row[feature_sets[exp]].astype(float)])
        prob = float(np.mean([m.predict_proba(x)[0, 1] for m in fitted]))
        out.append({
            "Feature set": EXPERIMENT_LABELS[exp],
            "Model": MODEL_LABELS[model],
            "Probability up": prob,
            "Call": "Up" if prob >= 0.5 else "Down",
        })
    return pd.DataFrame(out)


def market_status(now_utc: datetime) -> str:
    ny = now_utc.astimezone(NY)
    open_now = ny.weekday() < 5 and dtime(9, 30) <= ny.time() < dtime(16, 0)
    stamp = ny.strftime("%a %H:%M New York time")
    return (f"US market appears open ({stamp})" if open_now
            else f"US market appears closed ({stamp})") + ". Public holidays are not checked."


# --------------------------------------------------------------------- pages

def page_overview() -> None:
    st.title("Interpretable financial factors for AAPL direction")
    st.caption("PG-S2-36 capstone, University of Adelaide. Supervisor: Associate Professor Lingqiao Liu (AIML).")
    st.info(DISCLAIMER)

    st.subheader("Research question")
    st.markdown(
        "Can a language model propose financial factors that are readable by people and that improve "
        "next day prediction of whether Apple (AAPL) closes higher, compared with hand built features?"
    )

    st.subheader("Setup")
    left, right = st.columns(2)
    with left:
        st.markdown(
            "**Target:** AAPL next trading day direction (up or down).\n\n"
            "**Inputs:** AAPL, plus QQQ, MSFT and NVDA as related signals.\n\n"
            "**Split (2025):** Q1 to Q2 training, Q3 validation, Q4 untouched test."
        )
    with right:
        st.markdown(
            "**Stage 1:** feature pipeline, two simple baselines, Logistic Regression and Random Forest, "
            "with ablations A, B and C.\n\n"
            "**Stage 2:** a language model proposes eight readable factors, checked for format and leakage."
        )

    summary = load_csv(SOURCES["Original committed run"])
    if summary is None:
        missing(SOURCES["Original committed run"])
        return
    test = summary[summary["split"] == "test"]
    best = test[test["model"].isin(["logistic_regression", "random_forest"])].sort_values(
        "balanced_accuracy_mean", ascending=False).iloc[0]
    prev = test[test["model"] == "previous_direction"].iloc[0]
    st.subheader("Headline result (Q4 2025 test period, 63 trading days)")
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Best ML balanced accuracy",
        f"{best['balanced_accuracy_mean']:.3f}",
        help=f"{EXPERIMENT_LABELS[best['experiment']]}, {MODEL_LABELS[best['model']]}",
    )
    c2.metric("Previous direction baseline", f"{prev['balanced_accuracy_mean']:.3f}")
    c3.metric("Coin toss", "0.500")
    st.markdown(
        "The best machine learning model does not beat the simple previous direction baseline on the "
        "test period. A negative or inconclusive result is a valid research outcome for this project."
    )


def page_factors() -> None:
    st.title("Language model factors")
    data = load_json(CANDIDATES)
    if data is None:
        missing(CANDIDATES)
        return
    meta = load_json(CANDIDATE_META) or {}
    items = data["candidates"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Accepted candidates", len(items))
    c2.metric("Status", str(meta.get("status", "n/a")).capitalize())
    c3.metric("Prompt version", str(data.get("prompt_version", "n/a")))
    st.caption(
        "Each factor is a readable formula over price and volume columns. Each was checked locally "
        "for format and leakage. None was kept or dropped using test period results."
    )
    families = sorted({i["family"] for i in items})
    chosen = st.multiselect("Filter by family", families, default=families)
    for item in items:
        if item["family"] not in chosen:
            continue
        with st.expander(f"{item['factor_id']}  {item['name']}  ({item['family']})"):
            st.markdown(f"**Hypothesis.** {item['hypothesis']}")
            st.code(item["formula"], language="text")
            st.markdown(
                f"**Inputs:** {', '.join(item['inputs'])}  \n"
                f"**Lookback:** {item['lookback_days']} days  \n"
                f"**Expected direction:** {item['expected_direction']}"
            )
            st.markdown(f"**Leakage self check.** {item['leakage_self_check']}")
            st.markdown(f"**Difference from baseline features.** {item['difference_from_baseline']}")


def page_stage1() -> None:
    st.title("Stage 1 results")
    source = st.radio("Result source", list(SOURCES), horizontal=True)
    split = st.radio("Period", ["test", "validation"], horizontal=True,
                     format_func=lambda s: "Test (Q4 2025)" if s == "test" else "Validation (Q3 2025)")
    frame = load_csv(SOURCES[source])
    if frame is None:
        missing(SOURCES[source])
        return
    frame = frame[frame["split"] == split].copy()
    frame["Setup"] = frame["experiment"].map(EXPERIMENT_LABELS)
    frame["Model name"] = frame["model"].map(MODEL_LABELS)
    frame["Label"] = frame["Setup"] + " | " + frame["Model name"]

    metric = st.selectbox("Metric", list(METRICS), index=1, format_func=METRICS.get)
    col = f"{metric}_mean"
    chart = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X(f"{col}:Q", title=METRICS[metric], scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Label:N", sort=None, title=None),
            color=alt.Color("Model name:N", legend=alt.Legend(title="Model")),
            tooltip=["Label", alt.Tooltip(f"{col}:Q", format=".3f")],
        )
    )
    rule = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
    st.altair_chart(chart + rule, width="stretch")
    st.caption("Dashed line: 0.5, the level of a coin toss for balanced accuracy and AUC.")

    table = frame[["Setup", "Model name"]].copy()
    for key, label in METRICS.items():
        mean = frame[f"{key}_mean"]
        std = frame.get(f"{key}_std")
        table[label] = [
            f"{m:.3f}" if pd.isna(s) or s == 0 else f"{m:.3f} ± {s:.3f}"
            for m, s in zip(mean, std if std is not None else [np.nan] * len(mean))
        ]
    st.dataframe(table.rename(columns={"Model name": "Model"}), hide_index=True, width="stretch")
    st.caption("Random Forest rows are means over seeds 0, 1 and 2, shown as mean ± standard deviation.")


def page_predictions() -> None:
    st.title("Test period predictions")
    source = st.radio("Result source", list(PREDICTIONS), horizontal=True, key="pred_source")
    frame = load_csv(PREDICTIONS[source])
    if frame is None:
        missing(PREDICTIONS[source])
        return
    frame = frame[frame["split"] == "test"]
    c1, c2 = st.columns(2)
    exp = c1.selectbox("Feature set", ["A", "B", "C"], format_func=EXPERIMENT_LABELS.get)
    model = c2.selectbox("Model", ["logistic_regression", "random_forest"], format_func=MODEL_LABELS.get)
    sel = frame[(frame["experiment"] == exp) & (frame["model"] == model)]
    if sel.empty:
        st.warning("No rows for this choice.")
        return
    sel = (sel.groupby(["Date", "target_date", "actual"], as_index=False)["probability_up"].mean()
           .sort_values("target_date"))
    sel["Predicted"] = np.where(sel["probability_up"] >= 0.5, "Up", "Down")
    sel["Actual"] = np.where(sel["actual"] == 1, "Up", "Down")
    sel["Correct"] = np.where(sel["Predicted"] == sel["Actual"], "Correct", "Wrong")
    sel["target_date"] = pd.to_datetime(sel["target_date"])

    m1, m2, m3 = st.columns(3)
    m1.metric("Days predicted", len(sel))
    m2.metric("Correct calls", f"{(sel['Correct'] == 'Correct').sum()} ({pct((sel['Correct'] == 'Correct').mean())})")
    m3.metric("Actual up days", pct((sel["Actual"] == "Up").mean()))

    line = alt.Chart(sel).mark_line(color="#9aa0a6").encode(
        x=alt.X("target_date:T", title="Day predicted"),
        y=alt.Y("probability_up:Q", title="Predicted probability of up", scale=alt.Scale(domain=[0, 1])),
    )
    points = alt.Chart(sel).mark_circle(size=70).encode(
        x="target_date:T", y="probability_up:Q",
        color=alt.Color("Correct:N", scale=alt.Scale(domain=["Correct", "Wrong"], range=["#2e7d32", "#c62828"])),
        tooltip=[alt.Tooltip("target_date:T", title="Day"), "Predicted", "Actual",
                 alt.Tooltip("probability_up:Q", format=".3f")],
    )
    rule = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
    st.altair_chart(line + points + rule, width="stretch")
    st.caption("Random Forest probabilities are averaged over seeds 0, 1 and 2. A call is Up at 0.5 or above.")

    with st.expander("Day by day table"):
        shown = sel[["target_date", "probability_up", "Predicted", "Actual", "Correct"]].copy()
        shown["target_date"] = shown["target_date"].dt.date
        shown["probability_up"] = shown["probability_up"].round(3)
        st.dataframe(shown.rename(columns={"target_date": "Day", "probability_up": "Probability up"}),
                     hide_index=True, width="stretch")

    daily = load_csv(BACKTEST_DAILY)
    summary = load_csv(BACKTEST_SUMMARY)
    if daily is not None and summary is not None:
        st.subheader("Simple backtest diagnostic (Abu)")
        st.caption(
            "Long or flat on the configuration chosen by validation balanced accuracy. Same close execution, "
            "no costs. This is a diagnostic, not evidence of an executable strategy."
        )
        daily = daily.assign(target_date=pd.to_datetime(daily["target_date"]))
        long = daily.melt("target_date", ["strategy_equity", "buy_hold_equity"], "Series", "Equity")
        long["Series"] = long["Series"].map({"strategy_equity": "Model long or flat",
                                              "buy_hold_equity": "Buy and hold AAPL"})
        st.altair_chart(
            alt.Chart(long).mark_line().encode(
                x=alt.X("target_date:T", title="Day"), y=alt.Y("Equity:Q", scale=alt.Scale(zero=False)),
                color="Series:N"),
            width="stretch",
        )
        s = summary.rename(columns={"strategy": "Strategy", "cumulative_return": "Return",
                                    "max_drawdown": "Max drawdown", "sharpe": "Sharpe"})
        s["Strategy"] = s["Strategy"].map({"long_flat": "Model long or flat",
                                           "buy_and_hold_aapl": "Buy and hold AAPL"}).fillna(s["Strategy"])
        st.dataframe(s.assign(**{
            "Return": (s["Return"] * 100).round(2).astype(str) + "%",
            "Max drawdown": (s["Max drawdown"] * 100).round(2).astype(str) + "%",
            "Sharpe": s["Sharpe"].round(2)}), hide_index=True, width="stretch")


def page_live() -> None:
    st.title("Live prediction")
    st.warning(LIVE_DISCLAIMER)

    options = {"Off": None, "Every 30 seconds": 30, "Every minute": 60, "Every 5 minutes": 300}
    c1, c2 = st.columns([2, 3])
    choice = c1.selectbox("Auto refresh", list(options), index=1)
    c2.caption(
        "The price panel and the model calls update by themselves on this timer, with no page reload. "
        "Prices come from Yahoo Finance and may be delayed. The model call uses the last finished US "
        "session, so it changes once per trading day, while the intraday chart moves all session."
    )

    @st.fragment(run_every=options[choice])
    def panel() -> None:
        now_utc = datetime.now(timezone.utc)
        top, button = st.columns([5, 1])
        top.caption(market_status(now_utc))
        if button.button("Refresh now"):
            fetch_daily.clear()
            fetch_intraday.clear()

        offline_reason = None
        try:
            daily, _ = fetch_daily()
            prices, partial = drop_unfinished_session(daily, now_utc)
            source_label = "Yahoo Finance (live download)"
        except Exception as exc:  # network, package or format problem
            offline_reason = f"{type(exc).__name__}: {exc}"
            prices = clean_prices(pd.read_csv(DATA))
            partial = False
            source_label = "Saved repository file (offline)"

        if offline_reason:
            st.warning(
                "Live download failed, so the saved repository data is shown instead. "
                f"Details: {offline_reason}"
            )

        # ---- live price panel
        if offline_reason is None:
            try:
                intra = fetch_intraday()
                session_date = intra["Date"].dt.date.max()
                today = intra[intra["Date"].dt.date == session_date]
                prev = intra[intra["Date"].dt.date < session_date]
                cols = st.columns(len(TICKERS))
                series = []
                for col, t in zip(cols, TICKERS):
                    one = today[today["Ticker"] == t].sort_values("Date")
                    last = float(one["Close"].iloc[-1])
                    ref_rows = prev[prev["Ticker"] == t].sort_values("Date")
                    ref = float(ref_rows["Close"].iloc[-1]) if not ref_rows.empty else float(one["Close"].iloc[0])
                    col.metric(t, f"${last:,.2f}", f"{(last / ref - 1) * 100:+.2f}% vs previous close")
                    base = float(one["Close"].iloc[0])
                    series.append(pd.DataFrame({
                        "Time (New York)": one["Date"].dt.tz_localize(None),
                        "Change since session open (%)": (one["Close"] / base - 1) * 100,
                        "Ticker": t,
                    }))
                chart_data = pd.concat(series)
                chart = alt.Chart(chart_data).mark_line().encode(
                    x=alt.X("Time (New York):T"),
                    y=alt.Y("Change since session open (%):Q", scale=alt.Scale(zero=False)),
                    color="Ticker:N",
                    size=alt.condition(alt.datum.Ticker == "AAPL", alt.value(3), alt.value(1.3)),
                    tooltip=["Ticker", "Time (New York):T", alt.Tooltip("Change since session open (%):Q", format=".2f")],
                )
                st.altair_chart(chart, width="stretch")
                st.caption(f"Session shown: {session_date:%a %d %b %Y}, 5 minute bars. Latest bar: "
                           f"{today['Date'].max():%H:%M} New York time.")
            except Exception as exc:
                st.info(f"Intraday chart unavailable right now ({type(exc).__name__}: {exc}). "
                        "The model call below still works.")

        # ---- model calls
        try:
            row = latest_feature_row(prices)
            calls = live_calls(row)
        except Exception as exc:
            st.error(f"Could not compute the model calls: {type(exc).__name__}: {exc}")
            return

        asof = pd.Timestamp(row["Date"]).date()
        st.subheader(f"Model calls for the session after {asof:%a %d %b %Y}")
        if partial:
            st.caption("Today's US session is still in progress and is excluded from the features.")
        ups = int((calls["Call"] == "Up").sum())
        k1, k2, k3 = st.columns(3)
        k1.metric("Models calling Up", f"{ups} of {len(calls)}")
        k2.metric("Mean probability up", pct(float(calls["Probability up"].mean())))
        k3.metric("Data source", source_label)

        bars = alt.Chart(calls.assign(Label=calls["Feature set"].str.split(":").str[0] + " | " + calls["Model"])).mark_bar().encode(
            x=alt.X("Probability up:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Label:N", sort=None, title=None),
            color=alt.Color("Call:N", scale=alt.Scale(domain=["Up", "Down"], range=["#2e7d32", "#c62828"])),
            tooltip=["Feature set", "Model", alt.Tooltip("Probability up:Q", format=".3f"), "Call"],
        )
        rule = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
        st.altair_chart(bars + rule, width="stretch")
        st.dataframe(calls.assign(**{"Probability up": calls["Probability up"].round(3)}),
                     hide_index=True, width="stretch")

        _, _, ntrain = fitted_models(DATA.stat().st_mtime)
        now_local = datetime.now(ADELAIDE)
        st.caption(
            f"Models fitted once on the {ntrain} training rows of 2025 Q1 to Q2, as in the committed pipeline. "
            f"Last updated {now_local:%H:%M:%S} Adelaide time."
        )

    panel()


def page_stage2() -> None:
    st.title("Stage 2 comparison")
    st.caption("Does adding the language model factors improve on the Stage 1 features?")

    val = load_csv(STAGE2_VALIDATION)
    if val is None:
        missing(STAGE2_VALIDATION)
    else:
        st.subheader("Validation period (Q3 2025), classification")
        st.caption("Q4 was not loaded in this phase. Stage 1 sets are compared with the same sets plus the factors.")
        val = val.copy()
        val["Setup"] = val["experiment"].str.replace("_", " ")
        val["Model"] = val["model"].map(MODEL_LABELS)
        chart = alt.Chart(val).mark_bar().encode(
            x=alt.X("balanced_accuracy_mean:Q", title="Balanced accuracy", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Setup:N", sort=None, title=None),
            color=alt.Color("Model:N", legend=alt.Legend(title="Model")),
            tooltip=["Setup", "Model", alt.Tooltip("balanced_accuracy_mean:Q", format=".3f"),
                     alt.Tooltip("auc_mean:Q", format=".3f")],
        )
        rule = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
        st.altair_chart(chart + rule, width="stretch")
        st.dataframe(
            val[["Setup", "Model", "accuracy_mean", "balanced_accuracy_mean", "auc_mean", "f1_mean"]]
            .rename(columns={"accuracy_mean": "Accuracy", "balanced_accuracy_mean": "Balanced accuracy",
                             "auc_mean": "AUC", "f1_mean": "F1"})
            .round(3), hide_index=True, width="stretch")

    res = load_csv(PHASE2_RESULTS)
    if res is not None:
        st.subheader("Regression comparison (Taiyu)")
        st.caption("A separate comparison that predicts the size of the next day return, then reads the direction.")
        st.dataframe(res.round(4), hide_index=True, width="stretch")
    imp = load_csv(PHASE2_IMPORTANCE)
    if imp is not None:
        st.subheader("Factor importance in the largest set (Experiment C)")
        top = imp.sort_values("Importance", ascending=False)
        st.altair_chart(
            alt.Chart(top).mark_bar().encode(
                x=alt.X("Importance:Q"), y=alt.Y("Factor:N", sort=None, title=None),
                color=alt.Color("Type:N", legend=alt.Legend(title="Factor type")),
                tooltip=["Factor", "Type", alt.Tooltip("Importance:Q", format=".5f")]),
            width="stretch")
        st.caption("Importance values are small and several are negative, which suggests little reliable signal.")


def page_limits() -> None:
    st.title("Limitations")
    st.markdown(
        "- **Weak evidence.** The best machine learning model does not beat the previous direction baseline on the "
        "63 day test period.\n"
        "- **One short test window.** Results from one quarter can change a lot with a different period.\n"
        "- **Markets change.** Relationships between factors and returns shift over time.\n"
        "- **Language model factors are hypotheses.** Each was checked for format and leakage, and none was kept "
        "or dropped using test period results.\n"
        "- **Negative results count.** The project aims to find out whether the approach works, not to show that it does.\n"
        "- **Live page.** It reuses the Stage 1 recipe on current prices. Yahoo data can be delayed or revised, and "
        "its calls are not validated forecasts.\n"
        "- **No investment use.** This is not financial advice and not a system for trading."
    )
    st.subheader("Reproducibility")
    st.markdown(
        "Saved result pages read files in the repository `results` and `Stage2_Factor_Files` folders and recompute "
        "nothing. Only the Live prediction page downloads prices and fits models."
    )


PAGES = {
    "Overview": page_overview,
    "Language model factors": page_factors,
    "Stage 1 results": page_stage1,
    "Test period predictions": page_predictions,
    "Live prediction": page_live,
    "Stage 2 comparison": page_stage2,
    "Limitations": page_limits,
}


def main() -> None:
    st.set_page_config(page_title="PG-S2-36 Results", page_icon="📈", layout="wide")
    st.sidebar.title("PG-S2-36")
    page = st.sidebar.radio("Page", list(PAGES), label_visibility="collapsed")
    PAGES[page]()
    side_note()


if __name__ == "__main__":
    main()