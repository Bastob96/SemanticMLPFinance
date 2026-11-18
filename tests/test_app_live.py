"""The Live page's logic: the market clock, the intraday reader, and the "if the market closed now" preview."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.app.engine import NEW_YORK, Lab, load_snapshot, pool_members
from src.app.live import Preview, bar_so_far, market_clock, simulated_session, tidy_intraday
from src.stage2.formula import TICKERS


@pytest.fixture(scope="module")
def lab():
    return Lab(load_snapshot())


def test_market_clock():
    open_ = market_clock(datetime(2026, 11, 17, 11, 0, tzinfo=NEW_YORK))
    assert open_.is_open and open_.next_event == "close" and open_.next_time.hour == 16
    evening = market_clock(datetime(2026, 11, 17, 20, 30, tzinfo=NEW_YORK))
    assert not evening.is_open and evening.next_time.date().isoformat() == "2026-11-18" and evening.next_time.hour == 9
    friday = market_clock(datetime(2026, 11, 20, 17, 0, tzinfo=NEW_YORK))
    assert friday.next_time.date().isoformat() == "2026-11-23"                    # skips the weekend
    thanksgiving = market_clock(datetime(2026, 11, 25, 17, 0, tzinfo=NEW_YORK))
    assert thanksgiving.next_time.date().isoformat() == "2026-11-27"              # skips the holiday


def test_preview_at_the_close_equals_the_official_call(lab):
    """The whole point of the Live page: the last step of a session is exactly the call the Today page shows."""
    members = pool_members(lab.cat, "llm_blind")
    calls = lab.calls(lab.signal(members["key"]))
    for day in (lab.index[-1], lab.index[-63], lab.index[-300]):
        exact = {c: lab.prices.loc[day, c] for c in lab.prices.columns}
        r = Preview(lab, members, day).at(exact)
        assert r["signal"] == pytest.approx(calls.loc[day, "signal"], abs=1e-9)
        assert r["p_up"] == pytest.approx(calls.loc[day, "p_up"], abs=1e-9)
        assert r["lean"] == calls.loc[day, "lean"]


def test_preview_uses_nothing_after_the_session(lab):
    """Cutting every later day off the prices changes nothing in the preview."""
    members = pool_members(lab.cat, "llm_blind")
    day = lab.index[-63]
    cut = load_snapshot()
    cut = cut[cut["Date"] <= day]
    session = simulated_session(lab, day)
    a = Preview(lab, members, day).path(session, every=13)
    b = Preview(Lab(cut), members, day).path(session, every=13)
    assert np.allclose(a["p_up"], b["p_up"]) and np.allclose(a["signal"], b["signal"])


def test_simulated_session_rebuilds_the_day(lab):
    day = lab.index[-1]
    session = simulated_session(lab, day)
    bar = bar_so_far(session, session["Time"].max())
    for t in TICKERS:
        for f in ("Open", "High", "Low", "Close", "Volume"):
            assert bar[f"{t}_{f}"] == pytest.approx(lab.prices.loc[day, f"{t}_{f}"])
    assert len(session["Time"].unique()) == 78


@pytest.mark.parametrize("ticker_first", [True, False])
def test_tidy_intraday_reads_both_layouts_and_keeps_regular_hours(ticker_first):
    stamps = pd.date_range("2026-11-16 14:25", periods=6, freq="5min", tz="UTC")   # 09:25 to 09:50 New York
    fields = ["Open", "High", "Low", "Close", "Volume"]
    cols = pd.MultiIndex.from_product([TICKERS, fields] if ticker_first else [fields, TICKERS])
    frame = pd.DataFrame(np.arange(len(stamps) * len(cols), dtype=float).reshape(len(stamps), -1) + 1, index=stamps, columns=cols)
    out = tidy_intraday(frame)
    assert set(out["Ticker"]) == set(TICKERS)
    assert out["Time"].dt.strftime("%H:%M").min() == "09:30"                     # the 09:25 pre-market bar is dropped
    assert (out["Session"].astype(str) == "2026-11-16").all()
