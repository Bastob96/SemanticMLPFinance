"""Load, validate and align the four-asset daily OHLCV data; build the target and splits."""

import hashlib
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["Ticker", "Date", "Open", "High", "Low", "Close", "Volume"]


@dataclass
class MarketData:
    """Wide, date-indexed data plus the AAPL target.

    ``wide`` holds one row per common trading date and columns such as
    ``aapl_close`` or ``qqq_volume`` plus derived ``{asset}_ret1`` (1-day log
    return) and ``{asset}_logvol`` (log volume). ``target`` is the next-day AAPL
    log return and ``target_date`` the date whose close defines it.
    """

    wide: pd.DataFrame
    target: pd.Series
    target_date: pd.Series
    manifest: dict


@dataclass
class Splits:
    train: pd.DatetimeIndex
    validation: pd.DatetimeIndex
    test: pd.DatetimeIndex

    @property
    def development(self):
        return self.train.append(self.validation)

    def boundaries(self, target_date):
        def describe(index):
            return {"first_feature_date": str(index.min().date()),
                    "last_feature_date": str(index.max().date()),
                    "first_target_date": str(target_date.loc[index.min()].date()),
                    "last_target_date": str(target_date.loc[index.max()].date()),
                    "rows": int(len(index))}
        return {"train": describe(self.train), "validation": describe(self.validation),
                "test": describe(self.test),
                "rule": "Phase 1 Experiment A: chronological 70/15/15 by row position over rows "
                        "with complete baseline factors and target; dates frozen for all models."}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw(path):
    """Read the long-format CSV and run integrity checks. Never fills values."""
    raw = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Data file is missing required columns: {missing}")
    raw = raw[REQUIRED_COLUMNS + [c for c in ("Quarter", "Split") if c in raw.columns]].copy()
    raw["Date"] = pd.to_datetime(raw["Date"], format="%Y-%m-%d", errors="raise")
    raw["Ticker"] = raw["Ticker"].astype(str).str.upper().str.strip()
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    unknown = sorted(set(raw["Ticker"]) - set(config.TICKERS))
    absent = sorted(set(config.TICKERS) - set(raw["Ticker"]))
    if absent:
        raise ValueError(f"Data file does not contain tickers: {absent}")
    raw = raw[raw["Ticker"].isin(config.TICKERS)]
    if raw.duplicated(["Ticker", "Date"]).any():
        raise ValueError("Duplicate Ticker-Date rows found")

    price_cols = ["Open", "High", "Low", "Close"]
    invalid = (raw[price_cols + ["Volume"]].isna().any(axis=1)
               | (raw[price_cols] <= 0).any(axis=1)
               | (raw["Volume"] < 0)
               | (raw["High"] < raw[["Open", "Close", "Low"]].max(axis=1))
               | (raw["Low"] > raw[["Open", "Close", "High"]].min(axis=1)))
    checks = {"rows_read": int(len(raw)), "ignored_tickers": unknown,
              "invalid_rows_removed": int(invalid.sum()),
              "invalid_rows": raw.loc[invalid, ["Ticker", "Date"]].astype(str).to_dict("records")}
    if invalid.any():
        # Removing (not imputing) invalid bars; dates lacking any asset are dropped below.
        log.warning("Removing %d invalid OHLCV rows", int(invalid.sum()))
        raw = raw[~invalid]
    return raw.sort_values(["Date", "Ticker"]).reset_index(drop=True), checks


def to_wide(raw):
    """Pivot to one row per date, keeping only dates on which all four assets trade."""
    counts = raw.groupby("Date")["Ticker"].nunique()
    common = counts[counts == len(config.TICKERS)].index
    dropped = counts.index.difference(common)
    frames = []
    for ticker in config.TICKERS:
        part = (raw[raw["Ticker"] == ticker].set_index("Date")
                .loc[common, ["Open", "High", "Low", "Close", "Volume"]])
        part.columns = [f"{ticker.lower()}_{c.lower()}" for c in part.columns]
        frames.append(part)
    wide = pd.concat(frames, axis=1).sort_index()
    wide.index.name = "date"
    for ticker in config.TICKERS:
        prefix = ticker.lower()
        wide[f"{prefix}_ret1"] = np.log(wide[f"{prefix}_close"]).diff()
        wide[f"{prefix}_logvol"] = np.log(wide[f"{prefix}_volume"].replace(0, np.nan))
    return wide, [str(d.date()) for d in dropped]


def load_market_data(path):
    raw, checks = load_raw(path)
    wide, dropped_dates = to_wide(raw)
    close = wide[f"{config.TARGET_TICKER.lower()}_close"]
    target = np.log(close.shift(-1) / close).rename(config.TARGET_NAME)
    target_date = pd.Series(wide.index, index=wide.index).shift(-1).rename("target_date")

    returns = wide[[f"{t.lower()}_ret1" for t in config.TICKERS]]
    rows, cols = np.where((returns.abs() > config.SPLIT_JUMP_THRESHOLD).to_numpy())
    jump_list = [{"date": str(returns.index[r].date()), "column": returns.columns[c],
                  "log_return": float(returns.iat[r, c])} for r, c in zip(rows, cols)]
    if jump_list:
        log.warning("Possible unadjusted corporate actions: %s", jump_list)

    manifest = {
        "source_file": str(path), "sha256": file_sha256(path),
        "tickers": list(config.TICKERS), "target_ticker": config.TARGET_TICKER,
        "first_date": str(wide.index.min().date()), "last_date": str(wide.index.max().date()),
        "common_trading_dates": int(len(wide)), "dates_dropped_incomplete": dropped_dates,
        "integrity_checks": checks, "price_adjustment": config.PRICE_ADJUSTMENT_NOTE,
        "split_jump_check": {"threshold_abs_log_return": config.SPLIT_JUMP_THRESHOLD,
                             "suspicious_moves": jump_list},
        "missing_value_policy": "Invalid bars are removed, never imputed; no forward or backward "
                                "filling. Rolling statistics use trailing windows only.",
        "columns": {
            "{asset}_open/high/low/close": "Adjusted daily prices (USD)",
            "{asset}_volume": "Daily share volume",
            "{asset}_ret1": "ln(close_t / close_{t-1})",
            "{asset}_logvol": "ln(volume_t)",
        },
    }
    return MarketData(wide=wide, target=target, target_date=target_date, manifest=manifest)


def make_splits(baseline_features, data):
    """Reproduce the Phase 1 70/15/15 split and assert the label-date rule."""
    usable = baseline_features.notna().all(axis=1) & data.target.notna()
    index = baseline_features.index[usable]
    n = len(index)
    train_end, validation_end = int(n * config.SPLIT_FRACTIONS[0]), int(n * config.SPLIT_FRACTIONS[1])
    splits = Splits(train=index[:train_end], validation=index[train_end:validation_end],
                    test=index[validation_end:])
    assert_no_label_overlap(splits.train, splits.validation, data.target_date)
    assert_no_label_overlap(splits.development, splits.test, data.target_date)
    return splits


def assert_no_label_overlap(fit_index, eval_index, target_date):
    """Every training label must be observed before the first evaluation target date."""
    last_fit_label = target_date.loc[fit_index].max()
    first_eval_target = target_date.loc[eval_index].min()
    if not last_fit_label < first_eval_target:
        raise AssertionError(f"Label leakage: training label date {last_fit_label.date()} is not "
                             f"before evaluation target date {first_eval_target.date()}")
    if not fit_index.max() < eval_index.min():
        raise AssertionError("Splits are not chronological")
