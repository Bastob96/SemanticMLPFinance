"""Verify the shared dataset and quarterly Stage 2 profile before training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_canonical_data(
    repo_root: Path,
    manifest_path: Path,
    verify_stage2_profile: bool = True,
) -> dict:
    """Fail fast if a model run points at a different data snapshot or split."""
    root = repo_root.resolve()
    manifest = json.loads(manifest_path.resolve().read_text(encoding="utf-8"))
    sources = manifest.get("sources", {})
    if set(sources) != {"five_year", "split_2025"}:
        raise ValueError("Canonical manifest must define five_year and split_2025")

    loaded: dict[str, pd.DataFrame] = {}
    report = {"status": manifest.get("status"), "sources": {}}
    for name, spec in sources.items():
        path = root / spec["path"]
        if not path.is_file():
            raise ValueError(f"Canonical {name} file not found: {path}")
        digest = _sha256(path)
        if digest != spec["sha256"]:
            raise ValueError(
                f"Canonical {name} SHA-256 mismatch: expected {spec['sha256']}, got {digest}"
            )
        frame = pd.read_csv(path, parse_dates=["Date"])
        required = {"Ticker", "Date", "Open", "High", "Low", "Close", "Volume"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Canonical {name} is missing columns: {missing}")
        if len(frame) != spec["rows"] or frame["Date"].nunique() != spec["trading_dates"]:
            raise ValueError(f"Canonical {name} counts do not match the manifest")
        if frame["Date"].min().strftime("%Y-%m-%d") != spec["start_date"] or frame["Date"].max().strftime("%Y-%m-%d") != spec["end_date"]:
            raise ValueError(f"Canonical {name} date range does not match the manifest")
        if set(frame["Ticker"].unique()) != {"AAPL", "QQQ", "MSFT", "NVDA"}:
            raise ValueError(f"Canonical {name} has an unexpected ticker set")
        if frame.duplicated(["Ticker", "Date"]).any() or frame.isna().any().any():
            raise ValueError(f"Canonical {name} contains duplicate or missing values")
        numeric = frame[["Open", "High", "Low", "Close", "Volume"]].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"Canonical {name} contains non-finite OHLCV values")
        loaded[name] = frame.sort_values(["Date", "Ticker"]).reset_index(drop=True)
        report["sources"][name] = {
            "path": spec["path"], "sha256": digest, "rows": len(frame),
            "trading_dates": int(frame["Date"].nunique()),
        }

    split = loaded["split_2025"]
    if not {"Quarter", "Split"}.issubset(split.columns):
        raise ValueError("Canonical 2025 split is missing Quarter or Split")
    quarter = split["Date"].dt.quarter
    expected = quarter.map({1: "train", 2: "train", 3: "validation", 4: "test"})
    if not split["Date"].dt.year.eq(2025).all() or not split["Quarter"].eq("Q" + quarter.astype(str)).all() or not split["Split"].eq(expected).all():
        raise ValueError("Canonical 2025 quarterly labels are incorrect")

    history = loaded["five_year"]
    history_2025 = history.loc[history["Date"].dt.year.eq(2025)].reset_index(drop=True)
    identifiers_match = history_2025[["Ticker", "Date", "Volume"]].equals(
        split[["Ticker", "Date", "Volume"]]
    )
    prices_match = np.allclose(
        history_2025[["Open", "High", "Low", "Close"]],
        split[["Open", "High", "Low", "Close"]],
        rtol=1e-12,
        atol=1e-12,
    )
    if not identifiers_match or not prices_match:
        raise ValueError("Canonical 2025 split is inconsistent with the five-year file")

    if verify_stage2_profile:
        stage2_dir = root / manifest["stage2_profile"]["path"]
        build_report_path = stage2_dir / "build_report.json"
        build = json.loads(build_report_path.read_text(encoding="utf-8"))
        expected_hashes = {Path(v["path"]).name: v["sha256"] for v in sources.values()}
        if build.get("input_sha256") != expected_hashes:
            raise ValueError("Quarterly Stage 2 factors were built from different source files")
        if build.get("usable_split_counts") != {"train": 121, "validation": 63, "test": 63}:
            raise ValueError("Quarterly Stage 2 split counts are not 121/63/63")
        report["stage2_profile"] = {
            "name": "quarterly",
            "path": manifest["stage2_profile"]["path"],
            "usable_split_counts": build["usable_split_counts"],
        }
    return report
