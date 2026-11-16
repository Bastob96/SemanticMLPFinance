"""Directions in the runs already made (src/stage2/direction.py). No API calls, no model training, about a minute.

1. Every factor any search ever proposed (each formula once), grouped by where it came from:
   does the direction it shows on the search days (Q1-Q3) carry over to the sealed Q4 days?
   Random-number factors are the reference (they should show no carry-over).
2. The informed Opus run stated every factor's direction in advance: how often was it right?

Usage: python experiments/analyse_directions.py
Writes results/stage2/direction/existing_runs_persistence.csv, existing_runs_stated.csv and all_candidates.csv.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import build_stage1_features  # noqa: E402
from src.stage2.direction import UP, day_blocks, edges, factor_table, match_test, persistence, usable  # noqa: E402
from src.stage2.formula import wide_prices  # noqa: E402
from src.stage2.scoring import BASE  # noqa: E402

warnings.filterwarnings("ignore")
DATA = ROOT / "data/processed/aligned_daily_ohlcv_5y.csv"
S = ROOT / "results/stage2"
OUT = S / "direction"
SHUFFLES = 5000


def source_of(folder: Path, cfg: dict):
    """Where a run's factors came from; None for replays and random-number runs (no new formulas)."""
    name = folder.name
    if cfg.get("proposer") in ("replay", "noise") or "sk-ant" in name:
        return None
    if cfg.get("proposer") == "template" or "random" in name:
        return "templates (no LLM)"
    if cfg.get("batch"):
        return "Sonnet, batch of 100"
    model = cfg.get("model", "claude-sonnet-5-5")
    if "haiku" in model:
        return "Haiku, 8 rounds"
    if "opus" in model:
        return "Opus, 8 rounds, informed" if cfg.get("informed") else "Opus, 8 rounds"
    return "Sonnet, 8 or 16 rounds"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA)
    features, prices = build_stage1_features(raw), wide_prices(raw)
    search, q4, every = (day_blocks(features, w) for w in ("search", "q4", "all"))

    rows = []
    for folder in [S / "runs/20261015_1635_claude"] + sorted((S / "runs_v2").iterdir()):
        if not (folder / "scoreboard.csv").exists():
            continue
        cfg = json.loads((folder / "config.json").read_text()) if (folder / "config.json").exists() else {}
        source = source_of(folder, cfg)
        if source is None:
            continue
        board = pd.read_csv(folder / "scoreboard.csv")
        for r in board[board["status"] != "invalid"].itertuples():
            rows.append({"run": folder.name, "source": source, "name": r.name, "formula": r.formula,
                         "expected_sign": getattr(r, "expected_sign", "")})
    cand = pd.DataFrame(rows)
    proposed = len(cand)
    cand["key"] = cand["formula"].str.replace(r"\s+", "", regex=True)
    cand = cand.drop_duplicates("key").reset_index(drop=True)             # each formula once (first proposer keeps it)
    table = factor_table(dict(zip(cand["key"], cand["formula"])), prices)
    ok = usable(table, every)
    cand, table = cand[ok[cand["key"]].to_numpy()].reset_index(drop=True), table.loc[:, ok[ok].index]
    cand["edge_search"], cand["edge_q4"] = edges(table, search).to_numpy(), edges(table, q4).to_numpy()
    cand.drop(columns="key").to_csv(OUT / "all_candidates.csv", index=False)
    print(f"{proposed} valid proposals in all searches, {len(cand)} different formulas")

    llm = ~cand["source"].str.startswith("templates")
    noise = factor_table({f"noise_{i}": f"noise({900_000 + i})" for i in range(2000)}, prices)
    base = features.assign(Date=pd.to_datetime(features["Date"])).set_index("Date")[BASE]
    groups = [("all LLM factors", table.loc[:, llm.to_numpy()]), ("templates (no LLM)", table.loc[:, (~llm).to_numpy()]),
              ("the 22 existing factors", base), ("random numbers (2000)", noise)]
    groups += [(s, table.loc[:, (cand["source"] == s).to_numpy()]) for s in sorted(cand.loc[llm, "source"].unique())]
    out = []
    print("\nDoes the direction on the search days carry over to the sealed Q4 days?")
    for name, t in groups:
        r = persistence(t, search, q4, SHUFFLES)
        out.append({"group": name, **r})
        print(f"  {name:28} {r['factors']:4d} factors | correlation {r['correlation']:+.2f} (p {r['p_correlation']:.3f}) "
              f"| same direction {r['same_direction']:.0%} (p {r['p_same_direction']:.3f})")
    pd.DataFrame(out).to_csv(OUT / "existing_runs_persistence.csv", index=False)

    board = pd.read_csv(S / "runs_v2/abl_claude_opus_informed/scoreboard.csv")
    board = board[(board["status"] != "invalid") & board["expected_sign"].notna()].drop_duplicates("name")
    t = factor_table(dict(zip(board["name"], board["formula"])), prices)
    stated = []
    print("\nInformed Opus run: was the direction it stated in advance right? (guessing gives 50%)")
    for label, mask in (("all its factors", np.ones(len(board), bool)), ("said 'higher means up'", (board["expected_sign"] == UP).to_numpy()),
                        ("said 'higher means down'", (board["expected_sign"] != UP).to_numpy())):
        for days_name, blocks in (("all days", every), ("sealed Q4", q4)):
            r = match_test(t.loc[:, mask], board.loc[mask, "expected_sign"], blocks, SHUFFLES)
            stated.append({"factors_group": label, "days_used": days_name, **r})
            print(f"  {label:26} {days_name:10} {r['match_rate']:.0%} right of {r['factors']:2d} "
                  f"(p better than chance {r['p_better_than_chance']:.3f}, worse {r['p_worse_than_chance']:.3f})")
    pd.DataFrame(stated).to_csv(OUT / "existing_runs_stated.csv", index=False)
