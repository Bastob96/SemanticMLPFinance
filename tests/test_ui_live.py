"""Offline tests for the Live prediction page. No network access is used."""

import sys
import threading
import time
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from src.ui import live
    from src.ui.live_feed import LiveFeed, append_live_points, provisional_today, session_fraction
except ImportError:  # requirements.txt installs these in normal environments
    AppTest = None

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/aligned_daily_ohlcv_5y.csv"


class FakeSocket:
    """Stands in for yfinance.WebSocket: sends a few ticks, then waits."""

    sent = []

    def __init__(self, *args, **kwargs):
        self.closed = threading.Event()

    def subscribe(self, symbols):
        self.symbols = list(symbols)

    def listen(self, handler):
        for i in range(3):
            for symbol in self.symbols:
                message = {"id": symbol, "price": 200.0 + i, "time": str(int(time.time() * 1000))}
                FakeSocket.sent.append(message)
                handler(message)
        self.closed.wait(2)

    def close(self):
        self.closed.set()


def _fake_yfinance(fail: bool = False, socket_class=FakeSocket):
    saved = pd.read_csv(DATA, parse_dates=["Date"])
    last_close = saved[saved.Date == saved.Date.max()].set_index("Ticker")["Close"]
    module = types.ModuleType("yfinance")

    def download(tickers, period=None, interval=None, **kwargs):
        if fail:
            raise RuntimeError("simulated network failure")
        if interval == "1d":
            d = saved[saved.Date >= saved.Date.max() - pd.Timedelta(days=170)]
            wide = d.pivot(index="Date", columns="Ticker", values=["Open", "High", "Low", "Close", "Volume"])
            wide.columns.names = ["Price", "Ticker"]
            return wide
        rng = np.random.default_rng(1)
        if interval == "1m":
            idx = pd.date_range("2026-10-21 15:55", periods=1, tz="America/New_York")
        else:
            idx = pd.DatetimeIndex(
                list(pd.date_range("2026-10-20 09:30", "2026-10-20 15:55", freq="5min", tz="America/New_York"))
                + list(pd.date_range("2026-10-21 09:30", "2026-10-21 15:55", freq="5min", tz="America/New_York"))
            )
        cols = pd.MultiIndex.from_product(
            [["Open", "High", "Low", "Close", "Volume"], ["AAPL", "QQQ", "MSFT", "NVDA"]],
            names=["Price", "Ticker"],
        )
        frame = pd.DataFrame(index=idx, columns=cols, dtype=float)
        for t in ["AAPL", "QQQ", "MSFT", "NVDA"]:
            price = float(last_close[t]) * np.exp(np.cumsum(rng.normal(0, 0.001, len(idx))))
            for c in ["Open", "High", "Low", "Close"]:
                frame[(c, t)] = price
            frame[("Volume", t)] = 1e5
        return frame

    module.download = download
    module.WebSocket = socket_class
    return module


@unittest.skipIf(AppTest is None, "Streamlit is not installed")
class LivePageTests(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()
        self._saved_module = sys.modules.get("yfinance")

    def tearDown(self):
        if self._saved_module is None:
            sys.modules.pop("yfinance", None)
        else:
            sys.modules["yfinance"] = self._saved_module

    def _open(self):
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=300).run()
        app.radio[0].set_value("Live prediction").run()
        return app

    def test_live_download_shows_calls(self):
        sys.modules["yfinance"] = _fake_yfinance()
        app = self._open()
        self.assertFalse(app.exception)
        self.assertFalse(any("download failed" in w.value for w in app.warning))

    def test_page_shows_the_discovered_factor_and_eight_calls(self):
        sys.modules["yfinance"] = _fake_yfinance()
        app = self._open()
        self.assertFalse(app.exception)
        self.assertTrue(any("discovered factor inside the prediction" in s.value for s in app.subheader))
        text = " ".join(m.value for m in app.markdown)
        self.assertIn("nvda_range_vs_aapl_range", text)
        calls = [df.value for df in app.dataframe if "Feature set" in df.value.columns][0]
        self.assertEqual(len(calls), 8)
        self.assertIn("C · + Claude factor", set(calls["Feature set"]))

    def test_live_stream_mode_is_the_default_and_uses_ticks(self):
        sys.modules["yfinance"] = _fake_yfinance()
        app = self._open()
        self.assertEqual(app.selectbox[0].value, live.LIVE_STREAM)
        time.sleep(0.5)  # let the background thread deliver its ticks
        app.run()
        self.assertFalse(app.exception)
        text = " ".join(m.value for m in app.markdown)
        self.assertIn("Yahoo Finance (live stream)", text)
        self.assertTrue(any("Provisional model calls" in s.value for s in app.subheader))

    def test_timer_modes_and_finished_basis_still_work(self):
        sys.modules["yfinance"] = _fake_yfinance()
        app = self._open()
        for mode in ["Every 30 seconds", "Every minute", "Every 2 minutes", "Every 5 minutes", "Off"]:
            app.selectbox[0].set_value(mode).run()
            self.assertFalse(app.exception, mode)
        basis = next(r for r in app.radio if r.label == "Model call basis")
        basis.set_value(live.BASIS_FINISHED).run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Model calls for the session after" in s.value for s in app.subheader))

    def test_stream_blocked_falls_back_to_polling(self):
        class Dead(FakeSocket):
            def listen(self, handler):
                raise ConnectionError("blocked")

        sys.modules["yfinance"] = _fake_yfinance(socket_class=Dead)
        app = self._open()
        self.assertFalse(app.exception)
        self.assertFalse(any("download failed" in w.value for w in app.warning))

    def test_failed_download_falls_back_without_crashing(self):
        sys.modules["yfinance"] = _fake_yfinance(fail=True)
        app = self._open()
        self.assertFalse(app.exception)
        self.assertTrue(any("download failed" in w.value for w in app.warning))

    def test_missing_package_falls_back(self):
        sys.modules["yfinance"] = None
        app = self._open()
        self.assertFalse(app.exception)
        self.assertTrue(any("download failed" in w.value for w in app.warning))


@unittest.skipIf(AppTest is None, "Streamlit is not installed")
class LiveHelperTests(unittest.TestCase):
    def setUp(self):
        self.saved = pd.read_csv(DATA, parse_dates=["Date"])

    def test_clean_prices_drops_incomplete_dates(self):
        last = self.saved.Date.max()
        broken = self.saved[~((self.saved.Date == last) & (self.saved.Ticker == "NVDA"))]
        self.assertLess(live.clean_prices(broken).Date.max(), last)

    def test_unfinished_session_is_dropped_until_close(self):
        last = self.saved.Date.max()
        extra = self.saved[self.saved.Date == last].copy()
        extra["Date"] = pd.Timestamp("2026-10-21")
        full = pd.concat([self.saved, extra])
        out, partial = live.drop_unfinished_session(full, datetime(2026, 10, 21, 14, 0, tzinfo=timezone.utc))
        self.assertTrue(partial)
        self.assertEqual(out.Date.max(), last)
        out, partial = live.drop_unfinished_session(full, datetime(2026, 10, 21, 22, 0, tzinfo=timezone.utc))
        self.assertFalse(partial)
        self.assertEqual(out.Date.max(), pd.Timestamp("2026-10-21"))

    def test_latest_row_matches_committed_feature_builder(self):
        from src.features import build_stage1_features

        reference = build_stage1_features(self.saved)
        day = pd.Timestamp("2025-12-15")
        row = live.latest_feature_row(self.saved[self.saved.Date <= day])
        ref = reference[reference.Date == day].iloc[0]
        for col in reference.columns:
            if col in ("Date", "target_date", "target_next_day_up"):
                continue
            self.assertAlmostEqual(float(row[col]), float(ref[col]), places=12, msg=col)


@unittest.skipIf(AppTest is None, "Streamlit is not installed")
class ClaudeFactorTests(unittest.TestCase):
    def setUp(self):
        from src.ui.data import load_dashboard_data

        self.data = load_dashboard_data()
        self.saved = pd.read_csv(DATA, parse_dates=["Date"])

    def test_kept_factor_comes_from_the_committed_scoreboard(self):
        factor = live.claude_factor(self.data)
        self.assertEqual(factor["name"], "nvda_range_vs_aapl_range")
        self.assertEqual(factor["years_helped"], 3)

    def test_factor_value_matches_the_formula_by_hand(self):
        factor = live.claude_factor(self.data)
        row = live.latest_feature_row(self.saved, factor)
        last = self.saved[self.saved.Date == self.saved.Date.max()].set_index("Ticker")
        by_hand = ((last.loc["NVDA", "High"] - last.loc["NVDA", "Low"]) / last.loc["NVDA", "Close"]
                   - (last.loc["AAPL", "High"] - last.loc["AAPL", "Low"]) / last.loc["AAPL", "Close"])
        self.assertAlmostEqual(float(row[factor["name"]]), float(by_hand), places=12)

    def test_factor_uses_no_future_data(self):
        factor = live.claude_factor(self.data)
        day = pd.Timestamp("2025-12-15")
        full = live.factor_value(factor["formula"], self.saved)
        cut = live.factor_value(factor["formula"], self.saved[self.saved.Date <= day])
        self.assertAlmostEqual(float(full[day]), float(cut[day]), places=12)

    def test_calls_include_the_claude_feature_set(self):
        factor = live.claude_factor(self.data)
        row = live.latest_feature_row(self.saved, factor)
        calls, n_train, info = live.live_calls(row, str(DATA), factor)
        self.assertEqual(len(calls), 8)
        self.assertTrue(calls["Probability up"].between(0, 1).all())
        self.assertLess(info["low"], info["high"])
        plain, _, none_info = live.live_calls(row, str(DATA))
        self.assertEqual(len(plain), 6)
        self.assertIsNone(none_info)

    def test_summaries_match_the_committed_results(self):
        hits, total = live.noise_summary(self.data)
        self.assertEqual((hits, total), (2, 32))
        gains = live.q4_factor_gain(self.data)
        self.assertAlmostEqual(gains["claude"], 0.037, places=3)
        self.assertAlmostEqual(gains["random"], 0.023, places=3)


class LiveFeedTests(unittest.TestCase):
    def test_messages_are_parsed_and_bad_ones_ignored(self):
        feed = LiveFeed(["AAPL", "QQQ"])
        feed._on_message({"id": "AAPL", "price": 201.5, "time": "1790000000000"})
        feed._on_message({"id": "AAPL", "price": "oops"})
        feed._on_message({"id": "XYZ", "price": 5})
        feed._on_message({"id": "QQQ", "price": -1})
        feed._on_message({"error": "decode"})
        feed._on_message(None)
        ticks = feed.snapshot()
        self.assertEqual(set(ticks), {"AAPL"})
        self.assertEqual(ticks["AAPL"].price, 201.5)
        self.assertEqual(ticks["AAPL"].trade_time, 1790000000.0)

    def test_fresh_prices_needs_every_ticker_and_recent_ticks(self):
        feed = LiveFeed(["AAPL", "QQQ"])
        feed._on_message({"id": "AAPL", "price": 1.0})
        self.assertIsNone(feed.fresh_prices())
        feed._on_message({"id": "QQQ", "price": 2.0})
        self.assertEqual({k: v.price for k, v in feed.fresh_prices().items()}, {"AAPL": 1.0, "QQQ": 2.0})
        self.assertIsNone(feed.fresh_prices(max_age=-1))

    def test_feed_reconnects_after_a_failure(self):
        attempts = []

        class Flaky:
            def subscribe(self, symbols):
                pass

            def listen(self, handler):
                attempts.append(1)
                if len(attempts) == 1:
                    raise ConnectionError("first connection fails")
                handler({"id": "AAPL", "price": 10.0})
                time.sleep(5)

            def close(self):
                pass

        feed = LiveFeed(["AAPL"], ws_factory=Flaky, max_backoff=0.05)
        feed.start()
        deadline = time.time() + 5
        while "AAPL" not in feed.snapshot() and time.time() < deadline:
            time.sleep(0.05)
        feed.stop()
        self.assertGreaterEqual(len(attempts), 2)
        self.assertEqual(feed.snapshot()["AAPL"].price, 10.0)
        self.assertIsNone(feed.error)

    def test_failing_stream_reports_the_error(self):
        class Dead:
            def subscribe(self, symbols):
                pass

            def listen(self, handler):
                raise ConnectionError("blocked")

            def close(self):
                pass

        feed = LiveFeed(["AAPL"], ws_factory=Dead, max_backoff=0.05)
        feed.start()
        time.sleep(0.3)
        feed.stop()
        self.assertIn("blocked", feed.error)
        self.assertIn("blocked", feed.describe())


@unittest.skipIf(AppTest is None, "Streamlit is not installed")
class LiveHelpersTests(unittest.TestCase):
    def _today(self):
        idx = pd.date_range("2026-10-21 09:30", periods=3, freq="5min", tz="America/New_York")
        rows = []
        for t, base in [("AAPL", 100.0), ("QQQ", 200.0)]:
            for i, ts in enumerate(idx):
                rows.append({"Ticker": t, "Date": ts, "Open": base + i, "High": base + i + 1,
                             "Low": base + i - 1, "Close": base + i, "Volume": 1000.0})
        return pd.DataFrame(rows)

    def test_live_point_is_added_only_when_newer(self):
        today = self._today()
        later = pd.Timestamp("2026-10-21 09:58:30", tz="America/New_York")
        out = append_live_points(today, {"AAPL": 111.0, "QQQ": 222.0}, later)
        self.assertEqual(len(out), len(today) + 2)
        self.assertEqual(out[out.Ticker == "AAPL"].sort_values("Date")["Close"].iloc[-1], 111.0)
        earlier = pd.Timestamp("2026-10-21 09:30", tz="America/New_York")
        self.assertEqual(len(append_live_points(today, {"AAPL": 111.0, "QQQ": 222.0}, earlier)), len(today))

    def test_provisional_day_uses_live_close_and_projects_volume(self):
        today = self._today()
        history = pd.DataFrame({
            "Ticker": ["AAPL", "QQQ"], "Date": pd.Timestamp("2026-10-20"),
            "Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5, "Volume": 10.0,
        })
        now_ny = pd.Timestamp("2026-10-21 12:30", tz="America/New_York")  # 180 of 390 minutes
        out = provisional_today(history, today, {"AAPL": 150.0, "QQQ": 250.0}, today["Date"].dt.date.max(), now_ny)
        new = out[out.Date == pd.Timestamp("2026-10-21")].set_index("Ticker")
        self.assertEqual(new.loc["AAPL", "Close"], 150.0)
        self.assertEqual(new.loc["AAPL", "Open"], 100.0)
        self.assertEqual(new.loc["AAPL", "High"], 150.0)
        self.assertEqual(new.loc["AAPL", "Low"], 99.0)
        self.assertAlmostEqual(new.loc["AAPL", "Volume"], 3000.0 / (180 / 390))

    def test_provisional_day_is_skipped_when_the_day_is_already_final(self):
        today = self._today()
        history = pd.DataFrame({
            "Ticker": ["AAPL", "QQQ"], "Date": pd.Timestamp("2026-10-21"),
            "Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5, "Volume": 10.0,
        })
        now_ny = pd.Timestamp("2026-10-21 17:00", tz="America/New_York")
        self.assertIsNone(provisional_today(history, today, {"AAPL": 1.0, "QQQ": 1.0},
                                            today["Date"].dt.date.max(), now_ny))

    def test_market_open_check_uses_new_york_hours(self):
        from datetime import datetime, timezone

        self.assertTrue(live.market_is_open(datetime(2026, 11, 4, 15, 0, tzinfo=timezone.utc)))   # 10:00 New York
        self.assertFalse(live.market_is_open(datetime(2026, 11, 5, 7, 0, tzinfo=timezone.utc)))   # 02:00 New York
        self.assertFalse(live.market_is_open(datetime(2026, 11, 7, 15, 0, tzinfo=timezone.utc)))  # Saturday

    def test_session_fraction_is_bounded(self):
        self.assertEqual(session_fraction(pd.Timestamp("2026-10-21 09:30", tz="America/New_York")), 0.05)
        self.assertEqual(session_fraction(pd.Timestamp("2026-10-21 20:00", tz="America/New_York")), 1.0)


if __name__ == "__main__":
    unittest.main()
