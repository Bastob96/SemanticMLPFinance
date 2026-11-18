"""Every page of the Factor Lab app runs without an error, in the settings a visitor can choose."""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _page(view: str, state: dict):
    def script(view):
        import streamlit as st

        from src.app import theme
        from src.app.state import sidebar
        from src.app.views import factors, formula, live, method, research, today, track

        st.session_state.setdefault("mode", "saved")
        theme.apply()
        {"live": live, "today": today, "track": track, "factors": factors, "research": research, "formula": formula, "method": method}[view].render(sidebar())

    at = AppTest.from_function(script, kwargs={"view": view}, default_timeout=300)
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.mark.parametrize("view", ["live", "today", "track", "factors", "research", "formula", "method"])
def test_page_runs(view):
    at = _page(view, {})
    assert len(at.title) == 1


@pytest.mark.parametrize("state", [
    {"pool": "base"}, {"pool": "templates"}, {"pool": "fable", "stated": True}, {"pool": "llm_all"},
])
def test_today_and_track_with_other_factor_sets(state):
    _page("today", state)
    _page("track", state)


def test_replay_dates():
    import datetime as dt

    at = _page("today", {"as_of": dt.date(2025, 4, 4)})                  # an ordinary past day: the outcome is shown
    assert "What happened" in " ".join(m.value for m in at.markdown)
    at = _page("today", {"as_of": dt.date(2025, 4, 5)})                  # a Saturday: falls back to Friday
    assert any("No trading" in c.value for c in at.sidebar.caption)
    _page("today", {"as_of": dt.date(2022, 7, 1)})                       # the first day of the record: no probability yet
    _page("today", {"as_of": dt.date(2022, 9, 15)})


def test_formula_page_shows_a_clear_error():
    at = _page("formula", {"try_formula": "import os", "try_example": "(write my own)", "try_loaded": "(write my own)"})
    assert len(at.error) == 1 and "cannot be used" in at.error[0].value


def test_live_prices_add_a_live_period_and_a_current_call(monkeypatch):
    """With newer prices than the saved file (as live data gives), the app shows them and scores the new days."""
    import numpy as np
    import pandas as pd

    from src.app import state
    from src.app.engine import load_snapshot

    raw = load_snapshot()
    last = raw["Date"].max()
    days = pd.bdate_range(last + pd.Timedelta(days=1), periods=45)
    rng = np.random.default_rng(3)
    extra = []
    for ticker, g in raw.groupby("Ticker"):
        close = g.sort_values("Date")["Close"].iloc[-1] * np.cumprod(1 + rng.normal(0.0005, 0.015, len(days)))
        extra.append(pd.DataFrame({"Ticker": ticker, "Date": days, "Open": close * 0.995, "High": close * 1.01, "Low": close * 0.985,
                                   "Close": close, "Volume": g["Volume"].iloc[-1]}))
    newer = pd.concat([raw] + extra, ignore_index=True)
    monkeypatch.setattr(state, "fetch_live", lambda: newer)
    state.lab_for.clear()
    try:
        at = _page("today", {"mode": "live"})
        assert not at.sidebar.warning
        assert any(f"{days[-1]:%d %b %Y}" in c.value for c in at.sidebar.caption)
        at = _page("today", {"mode": "saved"})                # a visitor on the newest saved day switches to live prices...
        at.sidebar.radio[0].set_value("live").run()
        assert not at.exception and not any("Replay" in c.value for c in at.sidebar.caption)   # ...and stays on the newest day
        at = _page("track", {"mode": "live"})
        assert any("Live (since" in str(getattr(option, "content", option)) for option in at.button_group[0].options)
    finally:
        state.lab_for.clear()


def test_falls_back_to_the_saved_file_when_live_prices_fail(monkeypatch):
    from src.app import state

    def fail():
        raise ConnectionError("no network")

    monkeypatch.setattr(state, "fetch_live", fail)
    state.lab_for.clear()
    try:
        at = _page("today", {"mode": "live"})
        assert len(at.sidebar.warning) == 1 and "saved price file" in at.sidebar.warning[0].value
    finally:
        state.lab_for.clear()


def test_today_says_when_the_latest_day_is_not_published_yet(monkeypatch):
    """Yahoo can list a finished day with empty prices; the app must say it is waiting, not promise a future call."""
    import datetime as dt

    from src.app.engine import load_snapshot
    from src.app.views import today

    last = load_snapshot()["Date"].max().date()
    for days_later, expected in ((1, "Waiting for"), (2, "Waiting for"), (60, "not a current call"), (0, "The next call appears")):
        monkeypatch.setattr(today, "last_complete_day", lambda d=days_later: last + dt.timedelta(days=d))
        at = _page("today", {})
        assert expected in " ".join(m.value for m in at.markdown), (days_later, expected)
