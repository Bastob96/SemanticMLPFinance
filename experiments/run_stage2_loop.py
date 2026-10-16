"""Run the Stage 2 LLM factor-discovery loop.

Usage (on your Mac, with ANTHROPIC_API_KEY set):
    python experiments/run_stage2_loop.py --rounds 10
    python experiments/run_stage2_loop.py --rounds 10 --model claude-opus-5-5
Offline test / no-LLM baseline:
    python experiments/run_stage2_loop.py --rounds 2 --proposer random
Continue a stopped run:
    python experiments/run_stage2_loop.py --rounds 10 --out results/stage2/runs/<run> --resume
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import build_stage1_features  # noqa: E402
from src.stage2.formula import evaluate, wide_prices  # noqa: E402
from src.stage2.llm import ClaudeProposer, RandomProposer  # noqa: E402
from src.stage2.loop import Loop  # noqa: E402

warnings.filterwarnings("ignore")
DATA = ROOT / "data/processed/aligned_daily_ohlcv_5y.csv"
IMPORTANCE = ROOT / "results/multi_year/permutation_importance_summary.csv"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--new", type=int, default=3)
    ap.add_argument("--proposer", choices=["claude", "random"], default="claude")
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--seed", type=int, default=0, help="random proposer only")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out = args.out or ROOT / "results/stage2/runs" / f"{datetime.now():%Y%m%d_%H%M}_{args.proposer}"
    proposer = ClaudeProposer(args.model) if args.proposer == "claude" else RandomProposer(args.seed)
    raw = pd.read_csv(DATA)
    loop = Loop(proposer=proposer, raw=raw, features=build_stage1_features(raw), prices=wide_prices(raw),
                out_dir=out, stage1_importance_path=IMPORTANCE, n_variant=args.variants, n_new=args.new)

    start = 1
    if args.resume and (out / "scoreboard.csv").exists():
        board = pd.read_csv(out / "scoreboard.csv").fillna({"reason": "", "parent": "", "closest": ""})
        loop.board = board.to_dict("records")
        loop.kept = json.loads((out / "kept_factors.json").read_text())
        loop._kept_values = {k["name"]: evaluate(k["formula"], loop.prices) for k in loop.kept}
        start = int(board["round"].max()) + 1
        print(f"Resuming {out.name} at round {start} ({len(loop.board)} tried, {len(loop.kept)} kept)")
    else:
        (out / "config.json").write_text(json.dumps({**vars(args), "out": str(out),
                                                     "started": datetime.now().isoformat()}, indent=2, default=str))

    for r in range(start, args.rounds + 1):
        print(f"Round {r}", flush=True)
        for attempt in range(3):
            try:
                secs = loop.run_round(r)
                break
            except Exception as exc:  # API hiccup: wait and retry the same round
                print(f"  round {r} failed ({type(exc).__name__}: {exc}); retrying", flush=True)
                time.sleep(10 * (attempt + 1))
        else:
            sys.exit(f"Stopped at round {r}. Re-run with --resume --out {out}")
        print(f"  {len(loop.kept)} kept so far ({secs:.0f}s)", flush=True)
    print(f"\nDone. Results in {out}")
    print(pd.DataFrame(loop.kept).to_string(index=False) if loop.kept else "No factor met the selection rule.")
