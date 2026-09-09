"""Chronological 2025 split with labels contained inside each period."""

from __future__ import annotations

import pandas as pd


def quarterly_split(features: pd.DataFrame, year: int = 2025) -> dict[str, pd.DataFrame]:
    data = features.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data["target_date"] = pd.to_datetime(data["target_date"])
    data = data.loc[data["Date"].dt.year.eq(year)].sort_values("Date")
    if data.empty:
        raise ValueError(f"No feature rows available for {year}")

    source_quarter = data["Date"].dt.quarter
    target_quarter = data["target_date"].dt.quarter
    target_year = data["target_date"].dt.year
    source_partition = source_quarter.map({1: "train", 2: "train", 3: "validation", 4: "test"})
    target_partition = target_quarter.map({1: "train", 2: "train", 3: "validation", 4: "test"})
    contained = target_year.eq(year) & source_partition.eq(target_partition)
    data = data.loc[contained].copy()
    source_quarter = data["Date"].dt.quarter

    splits = {
        "train": data.loc[source_quarter.isin([1, 2])].reset_index(drop=True),
        "validation": data.loc[source_quarter.eq(3)].reset_index(drop=True),
        "test": data.loc[source_quarter.eq(4)].reset_index(drop=True),
    }
    if any(frame.empty for frame in splits.values()):
        raise ValueError("One or more required quarterly periods are empty")
    if not (splits["train"].Date.max() < splits["validation"].Date.min() < splits["test"].Date.min()):
        raise AssertionError("Split is not strictly chronological")
    return splits
