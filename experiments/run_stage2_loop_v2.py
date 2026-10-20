"""Run the interaction-aware Stage 2 loop (version 2: report cards, team trial, pair trials, partners).

Usage (on your Mac, with ANTHROPIC_API_KEY set):
    python experiments/run_stage2_loop_v2.py --rounds 8
Re-judge the clues of an earlier run with the new rules (no LLM, no API key):
    python experiments/run_stage2_loop_v2.py --proposer replay --replay results/stage2/runs/20261015_1635_claude
Offline test with template formulas (no LLM):
    python experiments/run_stage2_loop_v2.py --rounds 2 --proposer template
Continue a stopped run:
    python experiments/run_stage2_loop_v2.py --rounds 8 --out results/stage2/runs_v2/<run> --resume
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
from src.stage2.formula import wide_prices  # noqa: E402
from src.stage2.llm import ClaudeProposer  # noqa: E402
from src.stage2.loop_v2 import LoopV2, ReplayProposer, TemplateProposerV2  # noqa: E402
from src.stage2.prompt_v2 import SYSTEM_V2, TOOL_V2  # noqa: E402
from src.stage2.selection import Rules  # noqa: E402

warnings.filterwarnings("ignore")
DATA = ROOT / "data/processed/aligned_daily_ohlcv_5y.csv"
IMPORTANCE = ROOT / "results/multi_year/permutation_importance_summary.csv"
RULES_FILE = ROOT / "results/stage2/rules_v2.json"     # bars set by experiments/calibrate_noise_v2.py

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--improve", type=int, default=2)
    ap.add_argument("--new", type=int, default=2)
    ap.add_argument("--partner", type=int, default=2)
    ap.add_argument("--proposer", choices=["claude", "template", "replay"], default="claude")
    ap.add_argument("--replay", type=Path, help="run folder whose factors are replayed (--proposer replay)")
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--seed", type=int, default=0, help="template proposer only")
    ap.add_argument("--final-check", action="store_true",
                    help="give the promising list one more team trial at the end (off: lets random numbers in, see calibrate_noise_v2.py)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.proposer == "claude":
        proposer = ClaudeProposer(args.model, tool=TOOL_V2, system=SYSTEM_V2)
    elif args.proposer == "replay":
        proposer = ReplayProposer(args.replay)
        args.rounds = len(list(Path(args.replay).glob("round_*_response.json")))
    else:
        proposer = TemplateProposerV2(args.seed, args.improve, args.new, args.partner)
    rules = Rules(**json.loads(RULES_FILE.read_text())) if RULES_FILE.exists() else Rules()

    tag = args.proposer if args.proposer != "replay" else f"replay_{Path(args.replay).name}"
    out = args.out or ROOT / "results/stage2/runs_v2" / f"{datetime.now():%Y%m%d_%H%M}_{tag}"
    raw = pd.read_csv(DATA)
    loop = LoopV2(proposer=proposer, features=build_stage1_features(raw), prices=wide_prices(raw),
                  out_dir=out, stage1_importance_path=IMPORTANCE, rules=rules,
                  n_improve=args.improve, n_new=args.new, n_partner=args.partner)

    start = 1
    if args.resume and (out / "scoreboard.csv").exists():
        start = loop.load_state()
        print(f"Resuming {out.name} at round {start} ({len(loop.board)} tried, team: {loop.team})")
    else:
        (out / "config.json").write_text(json.dumps({**vars(args), "out": str(out), "rules": rules.as_dict(),
                                                     "started": datetime.now().isoformat()}, indent=2, default=str))
    print(f"Rules: {rules.as_dict()}")

    for r in range(start, args.rounds + 1):
        print(f"Round {r}", flush=True)
        for attempt in range(3):
            try:
                loop.run_round(r)
                break
            except Exception as exc:  # API hiccup: wait and retry the same round
                print(f"  round {r} failed ({type(exc).__name__}: {exc}); retrying", flush=True)
                loop.board = [x for x in loop.board if x["round"] != r]
                loop.pairs = [p for p in loop.pairs if p["round"] != r]
                loop.team = [n for n in loop.team if any(x["name"] == n for x in loop.board)]
                time.sleep(10 * (attempt + 1))
        else:
            sys.exit(f"Stopped at round {r}. Re-run with --resume --out {out}")
        print(f"  team now: {loop.team or '(empty)'} | {loop.log[-1]['seconds']:.0f}s", flush=True)

    if args.final_check:
        print("Final check of the promising list against the final team", flush=True)
        loop.finalize()
    print(f"\nDone. Results in {out}")
    print("Selected team:")
    for name in loop.team:
        rec = loop.rec(name)
        print(f"  {name}  ({rec['how_selected']})\n    {rec['formula']}")
    if not loop.team:
        print("  (empty: nothing met the team or pair rules)")
