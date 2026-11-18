"""Live view logic: the market clock, intraday prices, and "tomorrow's call if the market closed now".

The model is a daily model: it predicts the next trading day from a FINISHED day. During a session the day is not
finished, so this module asks a clear question instead: if the market closed at the latest price, what would the
call be? It builds today's bar so far (open, high, low, latest price, volume so far), adds it after the finished
days, and runs exactly the same steps as the official call:

    factor value today   each formula on the prices up to and including today's bar so far
    standardise          against the factor's own values on EARLIER days (the same mean and spread the app uses)
    direction            the direction measured at the start of the current quarter, on earlier days
    signal               the average of the turned, standardised factors
    chance of up day     the same logistic curve, fitted on earlier days only

At the last bar of a session the preview equals the official call for that day (a test checks this).

Author: Abu Wakkas Bastob (capstone PG-S2-36, supervised by A/Prof Lingqiao Liu).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.stage2.formula import TICKERS, evaluate

from .engine import MIN_CALIBRATION_DAYS, NEW_YORK, NYSE_HOLIDAYS, TILT, Z_CLIP, Z_START, Lab

OPEN, CLOSE = time(9, 30), time(16, 0)
TAIL = 200                     # days of history a formula needs (longest window 60, nested at most a few deep)
FIELDS = ("Open", "High", "Low", "Close", "Volume")


# ------------------------------------------------------------------ market clock
def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in NYSE_HOLIDAYS


@dataclass
class Clock:
    now: datetime               # New York time
    is_open: bool
    next_event: str             # "close" while open, else "open"
    next_time: datetime


def market_clock(now: datetime | None = None) -> Clock:
    ny = (now or datetime.now(tz=NEW_YORK)).astimezone(NEW_YORK)
    if is_trading_day(ny.date()) and OPEN <= ny.time() < CLOSE:
        return Clock(ny, True, "close", datetime.combine(ny.date(), CLOSE, tzinfo=NEW_YORK))
    d = ny.date() if ny.time() < OPEN else ny.date() + timedelta(days=1)
    while not is_trading_day(d):
        d += timedelta(days=1)
    return Clock(ny, False, "open", datetime.combine(d, OPEN, tzinfo=NEW_YORK))


# ------------------------------------------------------------------ intraday prices
def tidy_intraday(frame: pd.DataFrame) -> pd.DataFrame:
    """yfinance intraday download -> long table: Time (New York), Session, Ticker, OHLCV. Regular hours only."""
    if frame is None or len(frame) == 0 or not isinstance(frame.columns, pd.MultiIndex):
        raise ValueError("the intraday download returned no usable table")
    level = next((i for i in range(frame.columns.nlevels) if set(TICKERS) <= set(frame.columns.get_level_values(i))), None)
    if level is None:
        raise ValueError("intraday prices are missing one of AAPL, QQQ, MSFT, NVDA")
    stamps = pd.to_datetime(frame.index)
    stamps = stamps.tz_localize("UTC") if stamps.tz is None else stamps
    stamps = stamps.tz_convert(NEW_YORK)
    parts = []
    for ticker in TICKERS:
        t = frame.xs(ticker, axis=1, level=level)[list(FIELDS)].copy()
        t.columns.name = None
        t.insert(0, "Time", stamps)
        t.insert(0, "Ticker", ticker)
        parts.append(t.reset_index(drop=True))
    out = pd.concat(parts, ignore_index=True).dropna(subset=["Close"])
    clock = out["Time"].dt.time
    out = out[(clock >= OPEN) & (clock < CLOSE)].copy()
    out["Session"] = out["Time"].dt.date
    return out.sort_values(["Time", "Ticker"]).reset_index(drop=True)


def fetch_intraday(downloader=None) -> pd.DataFrame:
    """The last five sessions of 5-minute prices for the four tickers, from Yahoo Finance."""
    if downloader is None:
        import yfinance as yf

        def downloader():
            return yf.download(list(TICKERS), period="5d", interval="5m", auto_adjust=True, group_by="ticker",
                               prepost=False, progress=False, threads=False)
    return tidy_intraday(downloader())


def simulated_session(lab: Lab, day: pd.Timestamp, bars: int = 78) -> pd.DataFrame:
    """A stand-in session built from one finished day's open, high, low and close (used only when Yahoo's
    intraday prices cannot be reached). The path goes open -> low/high -> close; volume builds up evenly."""
    rows, start = [], datetime.combine(day.date(), OPEN, tzinfo=NEW_YORK)
    for ticker in TICKERS:
        o, h, l, c, v = (float(lab.prices.loc[day, f"{ticker}_{f}"]) for f in FIELDS)
        first, second = (l, h) if c >= o else (h, l)
        anchors = np.interp(np.arange(bars + 1), [0, round(bars * 0.3), round(bars * 0.7), bars], [o, first, second, c])
        for i in range(bars):
            a, b = anchors[i], anchors[i + 1]
            rows.append({"Ticker": ticker, "Time": pd.Timestamp(start + timedelta(minutes=5 * i)),
                         "Open": a, "High": max(a, b), "Low": min(a, b), "Close": b, "Volume": v / bars})
    out = pd.DataFrame(rows)
    out["Session"] = out["Time"].dt.date
    return out.sort_values(["Time", "Ticker"]).reset_index(drop=True)


def bar_so_far(session: pd.DataFrame, upto: pd.Timestamp) -> dict:
    """Today's daily bar built from the 5-minute bars up to and including ``upto``: {AAPL_Open: ..., ...}."""
    part = session[session["Time"] <= upto]
    bar = {}
    for ticker, g in part.groupby("Ticker"):
        g = g.sort_values("Time")
        bar.update({f"{ticker}_Open": g["Open"].iloc[0], f"{ticker}_High": g["High"].max(), f"{ticker}_Low": g["Low"].min(),
                    f"{ticker}_Close": g["Close"].iloc[-1], f"{ticker}_Volume": g["Volume"].sum()})
    return bar


# ------------------------------------------------------------------ the preview
class Preview:
    """Everything fixed for one session (earlier days only), so each new price needs only the formulas re-run."""

    def __init__(self, lab: Lab, members: pd.DataFrame, session_day: pd.Timestamp):
        self.lab, self.members = lab, members
        self.day = pd.Timestamp(session_day).normalize()
        self.keys = members["key"].tolist()
        self.formulas = dict(zip(members["key"], members["formula"]))
        idx = lab.index
        earlier = idx < self.day
        self.history = lab.prices[earlier].tail(TAIL)
        past = lab.values.loc[earlier, self.keys]
        enough = past.notna().sum() >= Z_START
        self.mean, self.std = past.mean().where(enough), past.std().where(enough)
        # directions and the probability curve of the quarter the session belongs to
        refits = [p for p in lab.refits if idx[p] <= self.day]
        self.refit_pos = refits[-1]
        self.signs = np.sign(lab.edge_history.loc[idx[self.refit_pos], self.keys])
        signal = lab.signal(self.keys)
        s, y = signal.to_numpy(dtype=float), lab.up.to_numpy(dtype=float)
        known = lab._known_before(self.refit_pos)
        self.base_rate = float(y[known].mean())
        ok = known & ~np.isnan(s)
        self.model, self.scale = None, 1.0
        if ok.sum() >= MIN_CALIBRATION_DAYS:
            self.scale = s[ok].std() or 1.0
            self.model = LogisticRegression(C=1.0).fit((s[ok] / self.scale).reshape(-1, 1), y[ok])

    def at(self, bar: dict) -> dict:
        """The call if the session closed with this bar."""
        row = pd.DataFrame([bar], index=[self.day])[self.history.columns]
        prices = pd.concat([self.history, row])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values = pd.Series({k: evaluate(f, prices).iloc[-1] for k, f in self.formulas.items()}, dtype=float)
        values = values.replace([np.inf, -np.inf], np.nan)
        z = ((values - self.mean) / self.std).replace([np.inf, -np.inf], np.nan).clip(-Z_CLIP, Z_CLIP)
        used = z.notna() & (self.signs != 0)
        parts = (z * self.signs).where(used)
        signal = float(parts.mean()) if used.any() else float("nan")
        p_up = float("nan")
        if self.model is not None and not np.isnan(signal):
            p_up = float(self.model.predict_proba(np.array([[signal / self.scale]]))[0, 1])
        lean = None if np.isnan(signal) else ("UP" if (p_up >= 0.5 if not np.isnan(p_up) else signal >= 0) else "DOWN")
        tilt = p_up - self.base_rate
        stance = None if np.isnan(p_up) else ("above" if tilt >= TILT else "below" if tilt <= -TILT else "usual")
        contrib = (parts / max(int(used.sum()), 1)).dropna()
        return {"signal": signal, "p_up": p_up, "base_rate": self.base_rate, "lean": lean, "stance": stance,
                "price": bar.get("AAPL_Close"), "contributions": contrib, "used": int(used.sum())}

    def path(self, session: pd.DataFrame, every: int = 1) -> pd.DataFrame:
        """The preview after every ``every``-th 5-minute bar of the session (the last bar is always included)."""
        times = sorted(session["Time"].unique())
        picked = times[every - 1::every]
        if times and picked[-1] != times[-1]:
            picked.append(times[-1])
        rows = []
        for t in picked:
            r = self.at(bar_so_far(session, t))
            rows.append({"Time": pd.Timestamp(t), "price": r["price"], "signal": r["signal"], "p_up": r["p_up"],
                         "lean": r["lean"], "stance": r["stance"]})
        return pd.DataFrame(rows)
