"""Streaming prices for the Live prediction page.

Yahoo Finance pushes a message every time a trade prints. A background thread
keeps one connection open and stores the newest price per ticker, so the page
can read it every second without downloading anything. If the stream cannot
connect, the page falls back to short interval polling, and then to the saved
file. Nothing here touches Q4 results or any model.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable

import pandas as pd

STREAM_FRESH_SECONDS = 60.0
MARKET_MINUTES = 390.0


@dataclass(frozen=True)
class Tick:
    price: float
    trade_time: float  # seconds since the epoch, as reported by the exchange
    received: float  # seconds since the epoch, when this process received it


def _default_ws_factory():
    import yfinance as yf

    return yf.WebSocket(verbose=False)


class LiveFeed:
    """One shared WebSocket connection with automatic reconnection."""

    def __init__(self, symbols: Iterable[str], ws_factory: Callable | None = None,
                 max_backoff: float = 30.0):
        self.symbols = list(symbols)
        self._ws_factory = ws_factory or _default_ws_factory
        self._max_backoff = max_backoff
        self._ticks: dict[str, Tick] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ws = None
        self.error: str | None = None
        self.connections = 0
        self._messages = 0

    # ---------------------------------------------------------- lifecycle
    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="live-feed", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._close_quietly()

    def _close_quietly(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    def _run(self) -> None:
        delay = 1.0
        while not self._stop.is_set():
            before = self._message_count()
            try:
                self._ws = self._ws_factory()
                self._ws.subscribe(self.symbols)
                self.connections += 1
                self._ws.listen(self._on_message)
            except Exception as exc:  # network, protocol or package problem
                self.error = f"{type(exc).__name__}: {exc}"
            finally:
                self._close_quietly()
            if self._stop.is_set():
                break
            if self._message_count() > before:
                delay = 1.0  # the connection worked for a while, so retry quickly
            self._stop.wait(delay)
            delay = min(delay * 2, self._max_backoff)

    # ------------------------------------------------------------- reading
    def _message_count(self) -> int:
        with self._lock:
            return self._messages

    def _on_message(self, message: dict) -> None:
        try:
            symbol = message.get("id")
            price = float(message["price"])
        except (AttributeError, KeyError, TypeError, ValueError):
            return
        if symbol not in self.symbols or not price > 0:
            return
        now = time.time()
        try:
            trade_time = float(message.get("time")) / 1000.0
        except (TypeError, ValueError):
            trade_time = now
        with self._lock:
            self._ticks[symbol] = Tick(price, trade_time, now)
            self._messages += 1
            self.error = None

    def snapshot(self) -> dict[str, Tick]:
        with self._lock:
            return dict(self._ticks)

    def fresh_prices(self, max_age: float = STREAM_FRESH_SECONDS) -> dict[str, Tick] | None:
        """Newest tick per ticker, or None unless every ticker has a recent one."""
        now = time.time()
        ticks = self.snapshot()
        if all(s in ticks and now - ticks[s].received <= max_age for s in self.symbols):
            return ticks
        return None

    def describe(self) -> str:
        ticks = self.snapshot()
        if ticks:
            age = time.time() - max(t.received for t in ticks.values())
            return f"last tick {age:.0f} s ago"
        return self.error or "waiting for the first price"


# ---------------------------------------------------------- pure helpers

def append_live_points(today: pd.DataFrame, prices: dict[str, float], when: pd.Timestamp) -> pd.DataFrame:
    """Add one point per ticker at `when` (New York time) when it is newer than the last bar."""
    rows = []
    for ticker, price in prices.items():
        one = today[today["Ticker"] == ticker]
        if one.empty or when <= one["Date"].max():
            continue
        rows.append({"Ticker": ticker, "Date": when, "Open": price, "High": price,
                     "Low": price, "Close": price, "Volume": 0.0})
    if not rows:
        return today
    return pd.concat([today, pd.DataFrame(rows)], ignore_index=True).sort_values(["Date", "Ticker"])


def session_fraction(now_ny: pd.Timestamp) -> float:
    """Share of the 390 minute US session that has elapsed, kept away from zero."""
    minutes = (now_ny.hour * 60 + now_ny.minute + now_ny.second / 60.0) - (9 * 60 + 30)
    return min(max(minutes / MARKET_MINUTES, 0.05), 1.0)


def provisional_today(prices: pd.DataFrame, today: pd.DataFrame, live: dict[str, float],
                      session_date, now_ny: pd.Timestamp) -> pd.DataFrame | None:
    """Daily history plus today's unfinished session, with the live price as the close.

    Volume is projected to a full session, because the volume feature compares
    against 20 full days. The result is provisional and is replaced by the real
    finished day after the close. Returns None when today cannot be built.
    """
    session_day = pd.Timestamp(session_date).normalize()
    if prices.empty or prices["Date"].max() >= session_day:
        return None
    fraction = session_fraction(now_ny)
    rows = []
    for ticker, price in live.items():
        bars = today[today["Ticker"] == ticker]
        if bars.empty:
            return None
        rows.append({
            "Ticker": ticker,
            "Date": session_day,
            "Open": float(bars["Open"].iloc[0]),
            "High": float(max(bars["High"].max(), price)),
            "Low": float(min(bars["Low"].min(), price)),
            "Close": float(price),
            "Volume": float(bars["Volume"].sum()) / fraction,
        })
    if len(rows) != prices["Ticker"].nunique():
        return None
    return pd.concat([prices, pd.DataFrame(rows)], ignore_index=True).sort_values(["Date", "Ticker"]).reset_index(drop=True)
