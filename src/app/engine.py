"""Engine of the Factor Lab app: data, factors, the past-only signal and its track record.

Everything shown for a day t is computed from information available at the close of day t:

    factor values   formulas in the safe formula language (src.stage2.formula): rolling windows end at t
    standardising   each factor against its OWN PAST values only (expanding mean and std, shifted one day)
    directions      at the start of every calendar quarter, each factor's direction (does a higher value go
                    with up days or down days?) is re-measured on all EARLIER days whose outcome is known
    signal          the average of the standardised factors, each turned so that higher means "up"
    probability     a one-input logistic curve fitted, again at each quarter start, on the signal's own
                    earlier out-of-sample days

So the value shown for any past date is exactly what the app would have shown on that date (tests check
this by cutting the future off the data), and "replay" needs no special code path.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

from src.models.stage1_models import fit_logistic_regression, fit_random_forest
from src.stage2.context import BASE_FORMULAS
from src.stage2.formula import TICKERS, FormulaError, evaluate, validate, wide_prices

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "data/processed/aligned_daily_ohlcv_5y.csv"
APP_DATA = ROOT / "app_data"
DOWNLOAD_START = "2021-08-31"          # first day of the project's price file
FIRST_SIGNAL = "2022-07-01"            # first quarter with enough history to measure directions
RESEARCH_END = "2025-12-31"            # the research (searches and the sealed Q4 tests) used 2022-2025
MIN_BLOCK_DAYS = 20                    # a year counts towards a direction once it has this many days
MIN_CALIBRATION_DAYS = 120             # out-of-sample days needed before a probability is shown
TILT = 0.02                            # P(up) this far from the usual day counts as a tilt
Z_START, Z_CLIP = 60, 3.0              # standardising: at least 60 past days; capped at +-3
NEW_YORK = ZoneInfo("America/New_York")

# New York Stock Exchange full-day closures (for naming the next trading day)
NYSE_HOLIDAYS = {date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3), date(2026, 5, 25),
                 date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26), date(2026, 12, 25),
                 date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26), date(2027, 5, 31),
                 date(2027, 6, 18), date(2027, 7, 5), date(2027, 9, 6), date(2027, 11, 25), date(2027, 12, 24)}

ROLE_LABEL = {"apple_own": "Apple's own behaviour", "market_condition": "Market mood (Nasdaq-100)",
              "peer_lead": "Peers (Microsoft, NVIDIA)", "relative_move": "Apple against the others",
              "template": "Template formulas"}

POOLS = {
    "llm_blind": "LLM factors proposed blind, 3 models",
    "llm_feedback": "LLM factors proposed after feedback, 3 models",
    "llm_all": "All LLM factors",
    "opus": "Opus 5.5 factors", "fable": "Fable 5.2 factors", "haiku": "Haiku 5.5 factors",
    "base": "The 22 existing factors",
    "kept": "The 4 factors kept by the iterative search",
    "templates": "Template formulas (no LLM)",
}
NOT_LLM_BATCH = ("Existing 22", "Templates (no LLM)", "Iterative search (kept)")
DEFAULT_POOL = "llm_blind"


# ------------------------------------------------------------------ data
def load_snapshot() -> pd.DataFrame:
    """The price file saved with the project (long table: Ticker, Date, Open, High, Low, Close, Volume)."""
    raw = pd.read_csv(SNAPSHOT)
    raw["Date"] = pd.to_datetime(raw["Date"])
    return raw


def last_complete_day(now: datetime | None = None) -> date:
    """The latest date whose daily bar is final: today after the New York close (16:00, plus a margin), else yesterday."""
    ny = (now or datetime.now(tz=NEW_YORK)).astimezone(NEW_YORK)
    return ny.date() if ny.time() >= time(16, 20) else ny.date() - timedelta(days=1)


def tidy_download(frame: pd.DataFrame) -> pd.DataFrame:
    """yfinance's wide download (columns ticker x field) -> the project's long table; days missing any ticker are dropped."""
    parts = []
    for ticker in TICKERS:
        t = frame[ticker][["Open", "High", "Low", "Close", "Volume"]].copy()
        t.insert(0, "Ticker", ticker)
        parts.append(t.rename_axis("Date").reset_index())
    raw = pd.concat(parts, ignore_index=True)
    raw["Date"] = pd.to_datetime(raw["Date"]).dt.tz_localize(None).dt.normalize()
    raw = raw.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    complete = raw.groupby("Date")["Ticker"].nunique()
    return raw[raw["Date"].isin(complete[complete == len(TICKERS)].index)].sort_values(["Ticker", "Date"]).reset_index(drop=True)


def fetch_live(now: datetime | None = None, downloader=None) -> pd.DataFrame:
    """Daily prices from Yahoo Finance from the project's start date to the last finished trading day.

    The same settings as the project's price file (adjusted prices). Raises if the download is unusable."""
    if downloader is None:
        import yfinance as yf

        def downloader():
            return yf.download(list(TICKERS), start=DOWNLOAD_START, interval="1d", auto_adjust=True,
                               group_by="ticker", progress=False, threads=False)
    raw = tidy_download(downloader())
    raw = raw[raw["Date"].dt.date <= last_complete_day(now)]
    check_prices(raw)
    return raw


def check_prices(raw: pd.DataFrame, at_least_until: pd.Timestamp | None = None) -> None:
    """Refuse data the app should not predict from."""
    if set(raw["Ticker"].unique()) != set(TICKERS):
        raise ValueError("prices are missing one of AAPL, QQQ, MSFT, NVDA")
    days = raw["Date"].nunique()
    if days < 400:
        raise ValueError(f"only {days} trading days of prices")
    if (raw[["Open", "High", "Low", "Close"]] <= 0).any().any() or (raw["Volume"] < 0).any():
        raise ValueError("prices contain zero or negative values")
    if raw.duplicated(["Ticker", "Date"]).any():
        raise ValueError("prices contain a repeated day")
    if at_least_until is not None and raw["Date"].max() < at_least_until:
        raise ValueError(f"live prices end {raw['Date'].max().date()}, before the saved file ({at_least_until.date()})")


def next_trading_day(day: pd.Timestamp) -> pd.Timestamp:
    d = pd.Timestamp(day).date() + timedelta(days=1)
    while d.weekday() >= 5 or d in NYSE_HOLIDAYS:
        d += timedelta(days=1)
    return pd.Timestamp(d)


def load_catalogue() -> pd.DataFrame:
    """Every factor the app knows: the 22 existing ones, the LLM factors of the direction test, the templates."""
    cat = pd.read_csv(APP_DATA / "factors.csv", keep_default_na=False)
    cat["key"] = cat["formula"].str.replace(" ", "")
    cat["round"] = cat["round"].astype(int)
    return cat


def pool_members(cat: pd.DataFrame, pool: str) -> pd.DataFrame:
    """Catalogue rows of a pool, one row per different formula (the first proposer of a formula keeps it)."""
    llm = ~cat["source"].isin(NOT_LLM_BATCH)
    mask = {"llm_blind": llm & (cat["round"] == 1), "llm_feedback": llm & (cat["round"] == 2), "llm_all": llm,
            "opus": cat["source"] == "Opus 5.5", "fable": cat["source"] == "Fable 5.2", "haiku": cat["source"] == "Haiku 5.5",
            "base": cat["source"] == "Existing 22", "kept": cat["source"] == "Iterative search (kept)", "templates": cat["source"] == "Templates (no LLM)"}[pool]
    return cat[mask].drop_duplicates("key")


# ------------------------------------------------------------------ statistics
def edge_block(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ROC AUC minus 0.5 of every column of x against the 0/1 outcomes y (missing values carry no information)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(x, axis=0)
    x = np.where(np.isnan(x), np.where(np.isfinite(med), med, 0.0), x)
    n1 = y.sum()
    n0 = len(y) - n1
    return (y @ rankdata(x, axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0) - 0.5


def edges(x: np.ndarray, y: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Direct edge of every column: AUC - 0.5 inside each calendar year, averaged with the years' day counts.

    Computing it inside each year means a year that simply had more up days cannot create an edge."""
    x = x.reshape(len(y), -1)
    total, days = np.zeros(x.shape[1]), 0
    for year in np.unique(years):
        m = years == year
        if m.sum() < MIN_BLOCK_DAYS or y[m].min() == y[m].max():
            continue
        total, days = total + edge_block(x[m], y[m]) * m.sum(), days + m.sum()
    return total / days if days else total


def auc_of(signal: pd.Series, up: pd.Series) -> float:
    """Within-year ROC AUC of one signal on the days where both the signal and the outcome exist."""
    ok = signal.notna() & up.notna()
    if ok.sum() < MIN_BLOCK_DAYS or up[ok].nunique() < 2:
        return float("nan")
    return float(0.5 + edges(signal[ok].to_numpy(dtype=float), up[ok].to_numpy(dtype=float), signal.index[ok].year.to_numpy())[0])


def shuffle_p(signal: pd.Series, up: pd.Series, n: int = 2000, seed: int = 0) -> float:
    """How often shuffled outcomes (shuffled inside each year) give an AUC at least this high."""
    ok = signal.notna() & up.notna()
    if ok.sum() < MIN_BLOCK_DAYS or up[ok].nunique() < 2:
        return float("nan")
    x, y, years = signal[ok].to_numpy(dtype=float), up[ok].to_numpy(dtype=float), signal.index[ok].year.to_numpy()
    real = edges(x, y, years)[0]
    rng, hits = np.random.default_rng(seed), 0
    blocks = [np.flatnonzero(years == v) for v in np.unique(years)]
    for _ in range(n):
        shuffled = y.copy()
        for b in blocks:
            shuffled[b] = rng.permutation(y[b])
        hits += edges(x, shuffled, years)[0] >= real
    return float((1 + hits) / (n + 1))


# ------------------------------------------------------------------ the lab
@dataclass
class Call:
    """What the app says at the close of one day about the next trading day."""
    day: pd.Timestamp
    next_day: pd.Timestamp
    signal: float
    p_up: float                 # NaN while there is not enough out-of-sample history
    base_rate: float            # share of up days among earlier days
    lean: str                   # "UP" or "DOWN"
    stance: str | None          # "above", "usual" or "below": P(up) against a usual day
    outcome: float              # 1 up, 0 down, NaN if not known yet
    next_return: float


class Lab:
    """Prices -> factor values -> past-only directions -> signal. One Lab per price table."""

    def __init__(self, raw: pd.DataFrame, catalogue: pd.DataFrame | None = None, source: str = "saved file"):
        self.source = source
        self.memo: dict = {}                                   # results the pages reuse (per Lab)
        self.cat = load_catalogue() if catalogue is None else catalogue
        self.prices = wide_prices(raw)
        self.index = self.prices.index
        self.close = self.prices["AAPL_Close"]
        self.next_return = self.close.shift(-1) / self.close - 1
        self.up = (self.next_return > 0).astype(float).where(self.next_return.notna())
        keys = dict(zip(self.cat["key"], self.cat["formula"]))
        self.values = self._table(keys)
        self.z = self._standardise(self.values)
        self.refits = self._refit_positions()
        self.edge_history = self._measure(self.values)          # rows: refit dates, columns: formula keys

    # ---- building blocks
    def _table(self, formulas: dict) -> pd.DataFrame:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            table = pd.DataFrame({key: evaluate(f, self.prices) for key, f in formulas.items()}, index=self.index)
        return table.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _standardise(values: pd.DataFrame) -> pd.DataFrame:
        past = values.expanding(Z_START)
        return ((values - past.mean().shift(1)) / past.std().shift(1)).replace([np.inf, -np.inf], np.nan).clip(-Z_CLIP, Z_CLIP)

    def _refit_positions(self) -> list:
        """Position of the first trading day of every quarter from FIRST_SIGNAL on."""
        quarter = self.index.to_period("Q")
        first = np.flatnonzero(np.r_[True, quarter[1:] != quarter[:-1]])
        return [int(p) for p in first if self.index[p] >= pd.Timestamp(FIRST_SIGNAL)]

    def _known_before(self, position: int) -> np.ndarray:
        """Days whose outcome (the next close) is already known before the day at this position starts."""
        m = np.zeros(len(self.index), dtype=bool)
        m[:max(position - 1, 0)] = True
        return m & self.up.notna().to_numpy()

    def _measure(self, values: pd.DataFrame) -> pd.DataFrame:
        x, y, years = values.to_numpy(dtype=float), self.up.to_numpy(dtype=float), self.index.year.to_numpy()
        rows = {}
        for p in self.refits:
            m = self._known_before(p)
            rows[self.index[p]] = edges(x[m], y[m], years[m])
        return pd.DataFrame.from_dict(rows, orient="index", columns=values.columns)

    def _segments(self):
        """(refit date, slice of day positions that use the directions measured at that date)."""
        ends = self.refits[1:] + [len(self.index)]
        return [(self.index[p], slice(p, e)) for p, e in zip(self.refits, ends)]

    # ---- the signal
    def signal(self, keys, stated: dict | None = None, z: pd.DataFrame | None = None,
               edge_history: pd.DataFrame | None = None) -> pd.Series:
        """Average of the standardised factors, each turned so that higher means 'up'.

        Directions come from the data (measured on earlier days, re-measured each quarter) unless ``stated``
        gives a fixed direction per key (+1 / -1), which is how the LLM's own directions are tested."""
        z = self.z if z is None else z
        history = self.edge_history if edge_history is None else edge_history
        keys = list(keys)
        zs = z[keys].to_numpy(dtype=float)
        out = np.full(len(self.index), np.nan)
        fixed = None if stated is None else np.array([stated[k] for k in keys], dtype=float)
        for refit, seg in self._segments():
            signs = fixed if fixed is not None else np.sign(history.loc[refit, keys].to_numpy(dtype=float))
            block = zs[seg] * signs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[seg] = np.nanmean(np.where(signs == 0, np.nan, block), axis=1)
        return pd.Series(out, index=self.index, name="signal")

    def signs_on(self, day: pd.Timestamp, keys) -> pd.Series:
        """The directions in force on a day (measured at the start of its quarter)."""
        refit = max(r for r, _ in self._segments() if r <= day)
        return np.sign(self.edge_history.loc[refit, list(keys)])

    def probability(self, signal: pd.Series) -> pd.DataFrame:
        """P(up) for each day from the signal, fitted at each quarter start on earlier out-of-sample days only."""
        s, y = signal.to_numpy(dtype=float), self.up.to_numpy(dtype=float)
        p_up, base = np.full(len(s), np.nan), np.full(len(s), np.nan)
        for _, seg in self._segments():
            known = self._known_before(seg.start)
            base[seg] = y[known].mean()
            past = known & ~np.isnan(s)
            now = ~np.isnan(s[seg])
            if past.sum() >= MIN_CALIBRATION_DAYS and now.any():
                scale = s[past].std() or 1.0
                model = LogisticRegression(C=1.0).fit((s[past] / scale).reshape(-1, 1), y[past])
                block = np.full(seg.stop - seg.start, np.nan)
                block[now] = model.predict_proba((s[seg][now] / scale).reshape(-1, 1))[:, 1]
                p_up[seg] = block
        return pd.DataFrame({"signal": signal, "p_up": p_up, "base_rate": base}, index=self.index)

    def calls(self, signal: pd.Series) -> pd.DataFrame:
        """One row per day: the signal, P(up), the call and what happened next.

        lean    UP when P(up) is at least 50%, else DOWN (before a probability exists: the sign of the signal)
        tilt    P(up) minus the usual share of up days; "above" / "below" usual when it differs by 2 points or more
        """
        t = self.probability(signal)
        t["lean"] = np.where(t["p_up"].notna(), np.where(t["p_up"] >= 0.5, "UP", "DOWN"), np.where(t["signal"] >= 0, "UP", "DOWN"))
        t.loc[t["signal"].isna(), "lean"] = None
        t["tilt"] = t["p_up"] - t["base_rate"]
        t["stance"] = np.select([t["tilt"] >= TILT, t["tilt"] <= -TILT], ["above", "below"], "usual")
        t.loc[t["p_up"].isna(), "stance"] = None
        t["outcome"], t["next_return"] = self.up, self.next_return
        t["hit"] = np.where(t["outcome"].notna() & t["lean"].notna(), (t["lean"] == "UP") == (t["outcome"] == 1), np.nan)
        return t

    def call_on(self, calls: pd.DataFrame, day: pd.Timestamp) -> Call:
        r = calls.loc[day]
        pos = self.index.get_loc(day)
        nxt = self.index[pos + 1] if pos + 1 < len(self.index) else next_trading_day(day)
        return Call(day, nxt, float(r["signal"]), float(r["p_up"]), float(r["base_rate"]), r["lean"], r["stance"],
                    float(r["outcome"]), float(r["next_return"]))

    # ---- explanations
    def contributions(self, day: pd.Timestamp, members: pd.DataFrame, stated: bool = False) -> pd.DataFrame:
        """Each factor's part of the signal on a day (the parts add up to the signal)."""
        keys = members["key"].tolist()
        z = self.z.loc[day, keys]
        measured = self.signs_on(day, keys)
        llm = members["stated"].map({"up": 1.0, "down": -1.0}).to_numpy()
        sign = pd.Series(llm, index=keys) if stated else measured
        used = z.notna() & (sign != 0)
        out = members.set_index("key").assign(z=z, direction=sign, measured=measured, llm=llm)
        out["contribution"] = (z * sign).where(used) / max(int(used.sum()), 1)
        past = self.values.loc[:day, keys]
        out["percentile"] = (past.iloc[:-1] < past.iloc[-1]).sum() / past.iloc[:-1].notna().sum().clip(lower=1)
        out["value"] = self.values.loc[day, keys]
        return out[used.to_numpy()].sort_values("contribution")

    def bucket_table(self, calls: pd.DataFrame, day: pd.Timestamp, buckets: int = 5) -> pd.DataFrame:
        """Earlier out-of-sample days grouped by signal size: how often AAPL closed up next day in each group."""
        pos = self.index.get_loc(day)                     # at the close of this day, the outcomes of all earlier days are known
        past = calls.iloc[:pos].dropna(subset=["signal", "outcome"])
        if len(past) < 100:
            return pd.DataFrame()
        cuts = np.quantile(past["signal"], np.linspace(0, 1, buckets + 1))
        cuts[0], cuts[-1] = -np.inf, np.inf
        group = pd.cut(past["signal"], cuts, labels=False)
        t = past.groupby(group).agg(days=("outcome", "size"), up_rate=("outcome", "mean"), low=("signal", "min"), high=("signal", "max"))
        t["label"] = ["Lowest fifth", "Low", "Middle", "High", "Highest fifth"][:len(t)] if buckets == 5 else [f"Group {i + 1}" for i in range(len(t))]
        today = calls.loc[day, "signal"]
        t["today"] = False
        if not np.isnan(today):
            t.loc[int(np.searchsorted(cuts, today, side="right") - 1), "today"] = True
        return t.reset_index(drop=True)

    # ---- track record
    def periods(self) -> dict:
        """Named sets of days for the track record."""
        idx, first = self.index, pd.Timestamp(FIRST_SIGNAL)
        snap_end = load_snapshot()["Date"].max()
        out = {"All days since July 2022": idx >= first,
               "Sealed quarters (Oct-Dec 2022-2025)": (idx >= first) & (idx.quarter == 4) & (idx <= pd.Timestamp(RESEARCH_END)),
               "Fresh 2026 (never used in the research)": (idx.year == 2026) & (idx <= snap_end)}
        if (idx > snap_end).sum() >= MIN_BLOCK_DAYS:
            out[f"Live (since {(snap_end.date() + timedelta(days=1)):%d %b %Y})"] = idx > snap_end
        for year in sorted(set(idx[idx >= first].year)):
            out[str(year)] = (idx >= first) & (idx.year == year)
        return out

    def score(self, calls: pd.DataFrame, mask, p_value: bool = True) -> dict:
        c = calls[np.asarray(mask)].dropna(subset=["signal", "outcome"])
        if len(c) < MIN_BLOCK_DAYS:
            return {"days": len(c)}
        up, lean_up = c["outcome"] == 1, c["lean"] == "UP"
        out = {"days": len(c), "auc": auc_of(c["signal"], c["outcome"]), "hit_rate": float((lean_up == up).mean()),
               "always_up": float(up.mean()), "share_lean_up": float(lean_up.mean()),
               "up_when_lean_up": float(up[lean_up].mean()) if lean_up.any() else float("nan"),
               "up_when_lean_down": float(up[~lean_up].mean()) if (~lean_up).any() else float("nan"),
               "balanced_accuracy": float((up[lean_up].sum() / max(up.sum(), 1) + (~up[~lean_up]).sum() / max((~up).sum(), 1)) / 2)}
        out["days_with_chance"] = int(c["p_up"].notna().sum())      # the first months have a signal but no probability yet
        for stance in ("above", "usual", "below"):
            m = c["stance"] == stance
            out[f"days_{stance}"], out[f"up_{stance}"] = int(m.sum()), float(up[m].mean()) if m.any() else float("nan")
        if p_value:
            out["p"] = shuffle_p(c["signal"], c["outcome"])
        return out

    def equity(self, calls: pd.DataFrame, mask) -> pd.DataFrame:
        """Growth of 1 unit: holding AAPL except on days the signal says 'below usual' (cash on those days),
        against always holding. No trading costs; an illustration of the signal, not a trading result."""
        c = calls[np.asarray(mask)].dropna(subset=["p_up", "next_return"])
        held = np.where(c["stance"] == "below", 0.0, c["next_return"])
        return pd.DataFrame({"Step aside when below usual": (1 + held).cumprod(), "Always hold AAPL": (1 + c["next_return"]).cumprod()}, index=c.index)

    def random_reference(self, pools: int = 30, size: int = 100, seed: int = 7) -> pd.DataFrame:
        """Signals built the same way from pools of pure random numbers: what 'no information' looks like."""
        rng = np.random.default_rng(seed)
        noise = pd.DataFrame(rng.standard_normal((len(self.index), pools * size)), index=self.index)
        z, history = self._standardise(noise), self._measure(noise)
        return pd.DataFrame({i: self.signal(range(i * size, (i + 1) * size), z=z, edge_history=history) for i in range(pools)})

    def model_opinions(self, day: pd.Timestamp | None = None) -> pd.DataFrame:
        """Second opinions: the project's two classifiers, retrained at each quarter start on all earlier days
        (so also past-only). Columns p_lr, p_rf use the 22 existing factors; p_lr_kept, p_rf_kept add the four
        factors the iterative LLM search kept. With ``day``, only that day's quarter is trained."""
        base = [f.replace(" ", "") for f in BASE_FORMULAS.values()]
        kept = pool_members(self.cat, "kept")["key"].tolist()
        y = self.up.to_numpy(dtype=float)
        out = pd.DataFrame(np.nan, index=self.index, columns=["p_lr", "p_rf", "p_lr_kept", "p_rf_kept"])
        for suffix, keys in (("", base), ("_kept", base + kept)):
            x = self.values[keys].to_numpy(dtype=float)
            complete = ~np.isnan(x).any(axis=1)
            for refit, seg in self._segments():
                if day is not None and not (refit <= day and self.index.get_loc(day) < seg.stop):
                    continue
                train = self._known_before(seg.start) & complete
                rows = np.arange(seg.start, seg.stop)[complete[seg]]
                if train.sum() < MIN_CALIBRATION_DAYS or not len(rows):
                    continue
                out.loc[self.index[rows], f"p_lr{suffix}"] = fit_logistic_regression(x[train], y[train]).predict_proba(x[rows])[:, 1]
                out.loc[self.index[rows], f"p_rf{suffix}"] = fit_random_forest(x[train], y[train], seed=0, max_features=4).predict_proba(x[rows])[:, 1]
        return out

    # ---- single factors
    def factor_stats(self, keys=None, values: pd.DataFrame | None = None) -> pd.DataFrame:
        """Per factor: edge on the research years, on the sealed quarters, on 2026, and by year."""
        values = self.values if values is None else values
        keys = list(values.columns) if keys is None else list(keys)
        x, y, years = values[keys].to_numpy(dtype=float), self.up.to_numpy(dtype=float), self.index.year.to_numpy()
        known = ~np.isnan(y)
        research = known & (years >= 2022) & (self.index <= pd.Timestamp(RESEARCH_END))
        cols = {"edge": research, "edge_search": research & (self.index.quarter < 4), "edge_q4": research & (self.index.quarter == 4),
                "edge_2026": known & (years == 2026)}
        out = pd.DataFrame({name: edges(x[m], y[m], years[m]) for name, m in cols.items()}, index=keys)
        for year in sorted(set(years[known & (years >= 2022)])):
            m = known & (years == year)
            if m.sum() >= MIN_BLOCK_DAYS:
                out[f"y{year}"] = edges(x[m], y[m], years[m])
        return out

    def quintiles(self, series: pd.Series) -> pd.DataFrame:
        """Up rate next day by fifth of a factor's value (fifths formed inside each year)."""
        d = pd.DataFrame({"v": series, "up": self.up}).dropna()
        d = d[d.index >= pd.Timestamp("2022-01-01")]
        d["fifth"] = d.groupby(d.index.year)["v"].transform(lambda v: pd.qcut(v.rank(method="first"), 5, labels=False))
        return d.groupby("fifth").agg(days=("up", "size"), up_rate=("up", "mean")).reset_index()

    def try_formula(self, formula: str) -> dict:
        """Validate and score a formula typed by the user. Raises FormulaError with a readable message."""
        info = validate(formula)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            series = evaluate(formula, self.prices).replace([np.inf, -np.inf], np.nan)
        usable = series[self.index >= pd.Timestamp("2022-01-01")]
        if usable.notna().mean() < 0.9 or usable.nunique() < 5:
            raise FormulaError("the formula gives too few different values to be tested")
        stats = self.factor_stats(values=series.to_frame("x")).iloc[0]
        ok = self.index >= pd.Timestamp("2022-01-01")
        similar = self.values[ok].corrwith(series[ok], method="spearman").abs().dropna().sort_values(ascending=False)
        nearest = self.cat.drop_duplicates("key").set_index("key").loc[similar.index[:5]].assign(similarity=similar.iloc[:5].to_numpy())
        return {"info": info, "series": series, "stats": stats, "nearest": nearest.reset_index()}
