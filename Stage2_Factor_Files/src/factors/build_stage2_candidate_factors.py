from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd


TICKERS = ["AAPL", "QQQ", "MSFT", "NVDA"]
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
FACTOR_COLUMNS = [
    "peer_signed_return_confirmation",
    "peer_return_dispersion",
    "aapl_peer_consensus_relative_return",
    "aapl_to_qqq_relative_realized_risk",
    "peer_volume_confirmed_aapl_return",
    "relative_return_market_risk_interaction",
    "aapl_to_qqq_normalized_range_ratio",
    "lagged_qqq_relative_momentum",
]
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
RAW_COLUMNS = [f"{ticker}_{field}" for ticker in TICKERS for field in OHLCV]
FEATURE_SETS = {
    "A": [],
    "B": [
        "aapl_to_qqq_relative_realized_risk",
        "relative_return_market_risk_interaction",
        "aapl_to_qqq_normalized_range_ratio",
        "lagged_qqq_relative_momentum",
    ],
    "C": FACTOR_COLUMNS,
}


def resolve_input(filename: str, explicit: Path | None = None,
                  repo_root: Path | None = None) -> Path:
    """Resolve only the two documented data directories, independent of cwd."""
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Input not found: {path}")
        return path
    root = repo_root.resolve() if repo_root else PACKAGE_ROOT.parent
    candidates = [root / "data" / folder / filename
                  for folder in ["processed", "processed_5y_yahoo"]]
    found = [path for path in candidates if path.is_file()]
    if len(found) > 1:
        raise ValueError(f"Multiple copies of {filename}; use --five-year or --split-2025 explicitly")
    if not found:
        raise ValueError(f"Cannot find {filename} under {root / 'data'}. "
                         "Keep Stage2_Factor_Files at the repository root, "
                         "or pass --repo-root, --five-year and --split-2025.")
    return found[0]


def _load_market_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    required = {"Ticker", "Date", *OHLCV}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")
    if df.duplicated(["Ticker", "Date"]).any():
        raise ValueError(f"Duplicate Ticker-Date rows in {path.name}")
    if set(df["Ticker"].unique()) != set(TICKERS):
        raise ValueError(f"Unexpected ticker set in {path.name}")
    if df[list(OHLCV)].isna().any().any():
        raise ValueError(f"Missing OHLCV values in {path.name}")
    if df["Date"].isna().any() or not df["Date"].eq(df["Date"].dt.normalize()).all():
        raise ValueError(f"Invalid daily dates in {path.name}")
    values = df[OHLCV].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite or non-numeric OHLCV values in {path.name}")
    if (values[["Open", "High", "Low", "Close"]] <= 0).any().any() or (values["Volume"] < 0).any():
        raise ValueError(f"Invalid price or volume in {path.name}")
    if (values["High"] < values[["Open", "Low", "Close"]].max(axis=1) - 1e-8).any() or \
       (values["Low"] > values[["Open", "High", "Close"]].min(axis=1) + 1e-8).any():
        raise ValueError(f"Invalid OHLC relationship in {path.name}")
    df[OHLCV] = values
    if not df.groupby("Date")["Ticker"].nunique().eq(len(TICKERS)).all():
        raise ValueError(f"Unaligned ticker dates in {path.name}")
    return df.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def _wide_ohlcv(history: pd.DataFrame) -> pd.DataFrame:
    wide = pd.DataFrame(index=sorted(history["Date"].unique()))
    wide.index.name = "Date"
    for ticker in TICKERS:
        ticker_df = history.loc[history["Ticker"].eq(ticker)].set_index("Date")
        for field in OHLCV:
            wide[f"{ticker}_{field}"] = ticker_df[field]
    return wide.sort_index()


def calculate_factors(wide: pd.DataFrame) -> pd.DataFrame:
    """Calculate the eight accepted Stage 2 generated factors using only data at or before t."""
    eps = 1e-12

    def price_return(ticker: str, periods: int) -> pd.Series:
        return wide[f"{ticker}_Close"].pct_change(
            periods=periods,
            fill_method=None,
        )

    def volume_return(ticker: str, periods: int) -> pd.Series:
        return wide[f"{ticker}_Volume"].pct_change(
            periods=periods,
            fill_method=None,
        )

    factors = pd.DataFrame(index=wide.index)

    # FCT-01
    msft_ret_3d = price_return("MSFT", 3)
    nvda_ret_3d = price_return("NVDA", 3)

    factors["peer_signed_return_confirmation"] = (
        msft_ret_3d + nvda_ret_3d
    ).rolling(5, min_periods=5).mean()

    # FCT-02
    factors["peer_return_dispersion"] = (
        (msft_ret_3d - nvda_ret_3d)
        .abs()
        .rolling(5, min_periods=5)
        .mean()
    )

    # FCT-03
    aapl_ret_5d = price_return("AAPL", 5)
    msft_ret_5d = price_return("MSFT", 5)
    nvda_ret_5d = price_return("NVDA", 5)

    factors["aapl_peer_consensus_relative_return"] = (
        aapl_ret_5d - ((msft_ret_5d + nvda_ret_5d) / 2.0)
    )

    # FCT-04
    aapl_risk_10d = (
        price_return("AAPL", 2)
        .rolling(10, min_periods=10)
        .std(ddof=1)
    )
    qqq_risk_10d = (
        price_return("QQQ", 2)
        .rolling(10, min_periods=10)
        .std(ddof=1)
    )

    factors["aapl_to_qqq_relative_realized_risk"] = (
        aapl_risk_10d / (qqq_risk_10d + eps)
    )

    # FCT-05
    peer_volume_change = (
        volume_return("MSFT", 3)
        + volume_return("NVDA", 3)
    ).rolling(5, min_periods=5).mean()

    factors["peer_volume_confirmed_aapl_return"] = (
        price_return("AAPL", 3) * peer_volume_change
    )

    # FCT-06
    relative_return_5d = (
        price_return("AAPL", 5)
        - price_return("QQQ", 5)
    )

    factors["relative_return_market_risk_interaction"] = (
        relative_return_5d * qqq_risk_10d
    )

    # FCT-07
    aapl_normalized_range = (
        (wide["AAPL_High"] - wide["AAPL_Low"])
        / (wide["AAPL_Close"] + eps)
    )
    qqq_normalized_range = (
        (wide["QQQ_High"] - wide["QQQ_Low"])
        / (wide["QQQ_Close"] + eps)
    )

    aapl_range_5d = aapl_normalized_range.rolling(
        5, min_periods=5
    ).mean()
    qqq_range_5d = qqq_normalized_range.rolling(
        5, min_periods=5
    ).mean()

    factors["aapl_to_qqq_normalized_range_ratio"] = (
        aapl_range_5d / (qqq_range_5d + eps)
    )

    # FCT-08
    factors["lagged_qqq_relative_momentum"] = (
        price_return("AAPL", 3)
        - price_return("QQQ", 3).shift(2)
    )
    return factors

def build_factor_table(five_year_path: Path, split_2025_path: Path,
                       output_path: Path | None = None) -> pd.DataFrame:
    history = _load_market_data(five_year_path)
    split_data = _load_market_data(split_2025_path)
    if not {"Quarter", "Split"}.issubset(split_data.columns):
        raise ValueError("Missing Quarter or Split columns")
    if not split_data["Date"].dt.year.eq(2025).all():
        raise ValueError("Split input must contain only 2025 dates")
    q = split_data["Date"].dt.quarter
    if not split_data["Quarter"].eq("Q" + q.astype(str)).all() or not split_data["Split"].eq(q.map({1: "train", 2: "train", 3: "validation", 4: "test"})).all():
        raise ValueError("Incorrect chronological Quarter or Split labels")
    shared = history.loc[history["Date"].dt.year.eq(2025)].set_index(["Date", "Ticker"])[OHLCV].sort_index()
    supplied = split_data.set_index(["Date", "Ticker"])[OHLCV].sort_index()
    if not shared.index.equals(supplied.index) or not np.allclose(shared.to_numpy(), supplied.to_numpy(), rtol=1e-10, atol=1e-8):
        raise ValueError("2025 OHLCV values or dates differ between input files")
    date_labels = split_data[["Date", "Quarter", "Split"]].drop_duplicates()
    wide = _wide_ohlcv(history)
    factors = calculate_factors(wide)
    target = (wide["AAPL_Close"].shift(-1) > wide["AAPL_Close"]).astype("Int64")
    target[wide["AAPL_Close"].shift(-1).isna()] = pd.NA

    output = date_labels.set_index("Date").join(wide, how="left")
    output = output.join(factors, how="left")
    output["target_next_day_up"] = target
    output["target_date"] = pd.Series(wide.index, index=wide.index).shift(-1)
    output = output.sort_index().reset_index()

    raw_columns = [f"{ticker}_{field}" for ticker in TICKERS for field in OHLCV]
    output = output[
        ["Date", "Quarter", "Split", *raw_columns, *FACTOR_COLUMNS, "target_next_day_up", "target_date"]
    ]

    if len(output) != 250 or output["Date"].nunique() != 250:
        raise ValueError("Expected exactly 250 unique 2025 trading dates")
    if output[FACTOR_COLUMNS].isna().any().any():
        raise ValueError("Candidate factors contain missing values")
    if not np.isfinite(output[FACTOR_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("Candidate factors contain non-finite values")
    if output["target_next_day_up"].isna().any():
        raise ValueError("The next-day target contains missing values")
    if not set(output["target_next_day_up"].astype(int).unique()).issubset({0, 1}):
        raise ValueError("The target must be binary")

    if output.isna().any().any():
        raise ValueError("Output contains missing values")
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    return output


def partition_tables(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Purge any label observed after the end of its own date-based split."""
    ends = {"train": "2025-06-30", "validation": "2025-09-30", "test": "2025-12-31"}
    return {name: table.loc[table["Split"].eq(name) & table["target_date"].le(pd.Timestamp(end))].copy()
            for name, end in ends.items()}


def write_outputs(table: pd.DataFrame, output_path: Path,
                  five_year: Path, split_2025: Path) -> dict:
    parts = partition_tables(table)
    out_dir = output_path.parent
    reserved = [out_dir / f"stage2_{name}_2025.csv" for name in parts]
    reserved += [out_dir / "build_report.json"]
    if output_path.name in {p.name for p in reserved}:
        raise ValueError("--output filename conflicts with a split/report filename")
    inputs = {five_year.resolve(), split_2025.resolve()}
    if any(p.resolve() in inputs for p in [output_path, *reserved]):
        raise ValueError("Output must not overwrite either source input")
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    for name, part in parts.items():
        part.to_csv(out_dir / f"stage2_{name}_2025.csv", index=False, date_format="%Y-%m-%d")
    kept = pd.concat(parts.values())["Date"]
    excluded = table.loc[~table["Date"].isin(kept), ["Date", "Split", "target_date"]].copy()
    for col in ["Date", "target_date"]:
        excluded[col] = excluded[col].dt.strftime("%Y-%m-%d")
    report = {
        "implementation_version": "v2",
        "rows": len(table), "columns": len(table.columns),
        "date_split_counts": table["Split"].value_counts().to_dict(),
        "usable_split_counts": {k: len(v) for k, v in parts.items()},
        "excluded_boundary_rows": excluded.to_dict("records"),
        "missing_values": int(table.isna().sum().sum()),
        "duplicate_dates": int(table["Date"].duplicated().sum()),
        "factor_columns": FACTOR_COLUMNS,
        "feature_sets": FEATURE_SETS,
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [five_year, split_2025]},
        "runtime": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
        "target": "1 if next trading day's adjusted AAPL Close > current Close, else 0",
        "signal_time": "After all four markets' day-t closing OHLCV is available",
        "evaluation": "No model training, factor selection or predictive-performance evaluation performed",
    }
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the 2025 Stage 2 candidate factor table")
    parser.add_argument("--repo-root", type=Path, help="Repository containing the shared data folder")
    parser.add_argument("--five-year", type=Path, help="Explicit source CSV, if autodetection is ambiguous")
    parser.add_argument("--split-2025", type=Path, help="Explicit 2025 source CSV")
    parser.add_argument("--output", type=Path, default=PACKAGE_ROOT / "data" / "processed" / "stage2_candidate_factor_table_2025.csv")
    args = parser.parse_args()

    try:
        five_year = resolve_input("aligned_daily_ohlcv_5y.csv", args.five_year, args.repo_root)
        split_2025 = resolve_input("aligned_daily_ohlcv_2025_split.csv", args.split_2025, args.repo_root)
        result = build_factor_table(five_year, split_2025)
        report = write_outputs(result, args.output, five_year, split_2025)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Saved {len(result)} rows and {len(result.columns)} columns to {args.output}")
    print("Usable split rows:", report["usable_split_counts"])


if __name__ == "__main__":
    main()
