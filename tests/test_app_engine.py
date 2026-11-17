"""The Factor Lab app's engine: no look-ahead, live-data handling, and that the pieces add up."""

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.app import engine
from src.app.engine import Lab, auc_of, check_prices, fetch_live, last_complete_day, load_snapshot, next_trading_day, pool_members
from src.stage2 import direction
from src.stage2.formula import FormulaError

NY = ZoneInfo("America/New_York")


@pytest.fixture(scope="module")
def raw():
    return load_snapshot()


@pytest.fixture(scope="module")
def lab(raw):
    return Lab(raw)


@pytest.fixture(scope="module")
def calls(lab):
    return lab.calls(lab.signal(pool_members(lab.cat, "llm_blind")["key"]))


def test_nothing_changes_when_the_future_is_removed(raw, lab, calls):
    """The call shown for a past date is exactly what the app would have shown on that date."""
    cut = pd.Timestamp("2025-03-14")
    early = Lab(raw[raw["Date"] <= cut])
    keys = pool_members(lab.cat, "llm_blind")["key"]
    early_calls = early.calls(early.signal(keys))
    days = early.index
    pd.testing.assert_frame_equal(early.values, lab.values.loc[days])
    pd.testing.assert_frame_equal(early.z, lab.z.loc[days])
    pd.testing.assert_frame_equal(early.edge_history, lab.edge_history.loc[early.edge_history.index])
    for column in ("signal", "p_up", "base_rate"):
        np.testing.assert_allclose(early_calls[column], calls.loc[days, column], rtol=1e-9, equal_nan=True)
    assert (early_calls["lean"].fillna("") == calls.loc[days, "lean"].fillna("")).all()


def test_directions_use_only_days_whose_outcome_is_known(lab):
    position = lab.refits[3]
    known = np.flatnonzero(lab._known_before(position))
    assert known.max() == position - 2          # the outcome of day p-2 is the close of day p-1, the last one before p
    assert not np.isnan(lab.up.iloc[known]).any()


def test_edge_matches_the_research_code(lab):
    """Same number as src.stage2.direction.edges, which the research results were computed with."""
    keys = list(lab.values.columns[:40])
    days = lab.index[(lab.index >= "2023-01-01") & (lab.index <= "2024-12-31")]
    blocks = [(days[days.year == y], lab.up.loc[days[days.year == y]].to_numpy()) for y in (2023, 2024)]
    table = lab.values[keys]
    table.index = pd.to_datetime(table.index)
    expected = direction.edges(table, blocks).to_numpy()
    got = engine.edges(lab.values.loc[days, keys].to_numpy(), lab.up.loc[days].to_numpy(), days.year.to_numpy())
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_contributions_add_up_to_the_signal(lab, calls):
    members = pool_members(lab.cat, "llm_blind")
    for day in (lab.index[-1], pd.Timestamp("2024-11-05")):
        parts = lab.contributions(day, members)
        assert parts["contribution"].sum() == pytest.approx(calls.loc[day, "signal"], abs=1e-9)
        assert parts["percentile"].between(0, 1).all()


def test_a_factor_that_knows_the_answer_is_found_and_noise_is_not(lab):
    rng = np.random.default_rng(1)
    informed = pd.DataFrame({i: lab.up.fillna(0.5) + rng.standard_normal(len(lab.index)) * 2 for i in range(30)}, index=lab.index)
    signal = lab.signal(range(30), z=lab._standardise(informed), edge_history=lab._measure(informed))
    assert auc_of(signal, lab.up) > 0.75
    noise = lab.random_reference(pools=6, size=50)
    scores = [auc_of(noise[c], lab.up) for c in noise]
    assert 0.44 < np.mean(scores) < 0.56


def test_probability_waits_for_enough_history(lab, calls):
    first = lab.index[lab.refits[0]]
    assert calls.loc[:first].iloc[:-1]["signal"].isna().all()               # no call before the first quarter
    early = calls.loc[first:].iloc[:60]
    assert early["signal"].notna().all() and early["p_up"].isna().all()    # a signal, but no probability yet
    late = calls.loc["2024-01-01":]
    assert late["p_up"].between(0.2, 0.8).all()
    assert set(late["lean"]) <= {"UP", "DOWN"} and set(late["stance"]) <= {"above", "usual", "below"}


def test_stated_directions_give_a_different_signal(lab, calls):
    members = pool_members(lab.cat, "llm_blind")
    stated = dict(zip(members["key"], members["stated"].map({"up": 1.0, "down": -1.0})))
    other = lab.signal(members["key"], stated=stated)
    assert not np.allclose(other.dropna(), calls["signal"].dropna())


def test_pools(lab):
    blind = pool_members(lab.cat, "llm_blind")
    assert blind["key"].is_unique and (blind["round"] == 1).all() and len(blind) > 150
    assert len(pool_members(lab.cat, "base")) == 22
    assert (pool_members(lab.cat, "templates")["source"] == "Templates (no LLM)").all()


def test_score_and_periods(lab, calls):
    periods = lab.periods()
    sealed = periods["Sealed quarters (Oct-Dec 2022-2025)"]
    assert set(lab.index[sealed].quarter) == {4} and lab.index[sealed].max().year == 2025
    fresh = lab.index[periods["Fresh 2026 (never used in the research)"]]
    assert fresh.min().year == 2026
    score = lab.score(calls, sealed, p_value=False)
    assert score["days_with_chance"] == score["days_above"] + score["days_usual"] + score["days_below"] <= score["days"]
    assert 0.4 < score["auc"] < 0.7
    equity = lab.equity(calls, periods["All days since July 2022"])
    assert equity.iloc[-1]["Always hold AAPL"] == pytest.approx((1 + calls.loc[equity.index, "next_return"]).prod())


def test_try_formula(lab):
    out = lab.try_formula("subtract(pct_change(AAPL_Close, 5), pct_change(QQQ_Close, 5))")
    assert out["info"]["max_window"] == 5 and len(out["nearest"]) == 5 and abs(out["stats"]["edge"]) < 0.2
    for bad in ("__import__('os')", "pct_change(AAPL_Close, -1)", "lag(TSLA_Close, 1)", "AAPL_Close +", "sign(sign(AAPL_Close))"):
        with pytest.raises(FormulaError):
            lab.try_formula(bad)


# ---- live prices
def _fake_download(raw):
    wide = raw.pivot(index="Date", columns="Ticker", values=["Open", "High", "Low", "Close", "Volume"])
    wide.columns = wide.columns.swaplevel()
    return wide.sort_index(axis=1)


def test_live_download_is_tidied_and_unfinished_days_are_dropped(raw):
    last = raw["Date"].max()
    during_session = datetime(last.year, last.month, last.day, 11, 0, tzinfo=NY)
    after_close = datetime(last.year, last.month, last.day, 17, 0, tzinfo=NY)
    live = fetch_live(now=after_close, downloader=lambda: _fake_download(raw))
    assert live["Date"].max() == last and len(live) == len(raw)
    live = fetch_live(now=during_session, downloader=lambda: _fake_download(raw))
    assert live["Date"].max() < last                      # today's half-finished bar is not used
    a, b = Lab(live).values, Lab(raw[raw["Date"] < last]).values
    pd.testing.assert_frame_equal(a, b)


def test_both_yfinance_column_layouts_and_time_zones_are_read(raw):
    by_ticker = _fake_download(raw)                                   # columns (ticker, field)
    by_field = by_ticker.swaplevel(axis=1).sort_index(axis=1)         # columns (field, ticker), yfinance's default
    by_field.columns.names = ["Price", "Ticker"]
    zoned = by_ticker.copy()
    zoned.index = zoned.index.tz_localize("America/New_York")
    late = datetime(2030, 1, 1, tzinfo=NY)
    a = fetch_live(now=late, downloader=lambda: by_ticker)
    for other in (by_field, zoned):
        pd.testing.assert_frame_equal(a, fetch_live(now=late, downloader=lambda: other))
    assert a["Date"].max() == raw["Date"].max() and str(a["Date"].dtype).startswith("datetime64")


def test_days_missing_a_ticker_are_dropped(raw):
    last = raw["Date"].max()
    holey = raw[~((raw["Ticker"] == "NVDA") & (raw["Date"] == last))]
    live = fetch_live(now=datetime(2030, 1, 1, tzinfo=NY), downloader=lambda: _fake_download(holey))
    assert live["Date"].max() < last and live.groupby("Date")["Ticker"].nunique().eq(4).all()


def test_bad_prices_are_refused(raw):
    with pytest.raises(ValueError, match="missing"):
        check_prices(raw[raw["Ticker"] != "QQQ"])
    with pytest.raises(ValueError, match="trading days"):
        check_prices(raw[raw["Date"] >= "2026-06-01"])
    broken = raw.copy()
    broken.loc[broken.index[5], "Close"] = 0.0
    with pytest.raises(ValueError, match="zero or negative"):
        check_prices(broken)
    with pytest.raises(ValueError, match="before the saved file"):
        check_prices(raw[raw["Date"] < "2026-08-01"], at_least_until=raw["Date"].max())
    with pytest.raises(Exception):
        fetch_live(downloader=lambda: pd.DataFrame())


def test_calendar():
    assert last_complete_day(datetime(2026, 11, 16, 15, 0, tzinfo=NY)).isoformat() == "2026-11-15"
    assert last_complete_day(datetime(2026, 11, 16, 16, 30, tzinfo=NY)).isoformat() == "2026-11-16"
    assert next_trading_day(pd.Timestamp("2026-11-13")) == pd.Timestamp("2026-11-16")      # Friday -> Monday
    assert next_trading_day(pd.Timestamp("2026-11-25")) == pd.Timestamp("2026-11-27")      # Thanksgiving skipped
    assert next_trading_day(pd.Timestamp("2026-12-24")) == pd.Timestamp("2026-12-28")      # Christmas + weekend
