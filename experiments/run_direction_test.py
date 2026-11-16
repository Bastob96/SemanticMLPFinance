"""Direction test: does the LLM know which way its own factors point, and can it learn it? (src/stage2/direction.py)

Round 1  The LLM proposes N interpretable factors in one call and states for each, in advance, whether a
         higher value makes an up day tomorrow more or less likely. It sees no results. Its directions are
         checked on every day of 2022-2025 (nothing is selected, so all days are a fair test).
Round 2  The LLM is shown how its round-1 directions turned out on the search days only (January-September
         of each year) and proposes N NEW factors with directions. These are checked on the sealed
         October-December quarters, which it never saw. Near-copies of round-1 factors do not count.
Reference  For the same round-2 factors: how often the direction measured on the search days is the
         direction on the sealed quarters (what the data alone can do, without the LLM's reasoning).

No model is trained, so the results do not depend on scikit-learn or on random seeds, and it runs in seconds.
Every reply is saved; running again continues from the saved replies and makes no new API calls.

Usage (ANTHROPIC_API_KEY set; about US$0.40 with Sonnet, US$0.02 with Haiku, US$0.75 with Opus):
    python experiments/run_direction_test.py --model claude-sonnet-5-5
    python experiments/run_direction_test.py --proposer template     # no LLM: template formulas, coin-flip directions
Writes results/stage2/direction/<model>/: factors.csv, summary.csv, prompts and replies.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import build_stage1_features  # noqa: E402
from src.stage2.context import BASE_FORMULAS, market_summary, stage1_importance  # noqa: E402
from src.stage2.direction import (DOWN, UP, composite, composite_test, day_blocks, edges, factor_table,  # noqa: E402
                                  match_test, persistence, stated_signs, usable)
from src.stage2.formula import FormulaError, validate, wide_prices  # noqa: E402
from src.stage2.llm import ClaudeProposer, dump  # noqa: E402
from src.stage2.loop_v2 import TemplateBatchProposer  # noqa: E402
from src.stage2.prompt_v2 import (SYSTEM_DIRECTION, TOOL_DIRECTION, build_prompt_direction,  # noqa: E402
                                  direction_evidence)

warnings.filterwarnings("ignore")
DATA = ROOT / "data/processed/aligned_daily_ohlcv_5y.csv"
IMPORTANCE = ROOT / "results/multi_year/permutation_importance_summary.csv"
COPY = 0.9                       # the project's near-copy threshold (|Spearman correlation| on the search days)


class CoinFlipTemplates:
    """No-LLM control: template formulas with directions chosen by coin flip (different formulas in round 2)."""

    def __init__(self, seed=0):
        self.seed, self.calls = seed, 0

    def __call__(self, prompt):
        self.calls += 1
        out = TemplateBatchProposer(self.seed + 100 * self.calls)(prompt[prompt.index("BATCH OF"):])
        rng = np.random.default_rng(self.seed + self.calls)
        for f in out["factors"]:
            f["name"], f["expected_sign"] = f"r{self.calls}_{f['name']}", str(rng.choice([UP, DOWN]))
        return out


def ask(proposer, prompt, out, tag):
    """One call, saved at once; a saved reply is reused (no second charge)."""
    (out / f"{tag}_prompt.txt").write_text(prompt)
    path = out / f"{tag}_response.json"
    if path.exists():
        print(f"{tag}: using the saved reply")
        return json.loads(path.read_text())
    reply = proposer(prompt)
    path.write_text(dump(reply))
    print(f"{tag}: {len(reply['factors'])} factors, tokens {reply.get('_usage')}", flush=True)
    return reply


def clean(reply, prices, blocks, seen):
    """Valid, new, testable factors with a stated direction -> (rows, table of daily values)."""
    rows, formulas = [], {}
    for f in reply["factors"]:
        formula, sign = str(f.get("formula", "")), f.get("expected_sign")
        key = formula.replace(" ", "")
        if sign not in (UP, DOWN) or key in seen or f.get("name") in formulas:
            continue
        try:
            validate(formula)
        except FormulaError:
            continue
        seen.add(key)
        formulas[f["name"]] = formula
        rows.append({"name": f["name"], "role": f.get("role", ""), "formula": formula,
                     "meaning": f.get("meaning", ""), "expected_sign": sign})
    table = factor_table(formulas, prices)
    ok = usable(table, blocks)
    return [r for r in rows if ok[r["name"]]], table.loc[:, ok[ok].index]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--n", type=int, default=100, help="factors asked for in each round")
    ap.add_argument("--proposer", choices=["claude", "template"], default="claude")
    ap.add_argument("--seed", type=int, default=0, help="template control only")
    ap.add_argument("--shuffles", type=int, default=5000)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    tag = args.model.replace("claude-", "") if args.proposer == "claude" else f"template_coinflip_{args.seed}"
    out = args.out or ROOT / "results/stage2/direction" / tag
    out.mkdir(parents=True, exist_ok=True)
    proposer = (ClaudeProposer(args.model, max_tokens=32000, tool=TOOL_DIRECTION, system=SYSTEM_DIRECTION)
                if args.proposer == "claude" else CoinFlipTemplates(args.seed))

    raw = pd.read_csv(DATA)
    features, prices = build_stage1_features(raw), wide_prices(raw)
    search, q4, every = (day_blocks(features, w) for w in ("search", "q4", "all"))
    context = dict(n=args.n, base_formulas=BASE_FORMULAS, stage1_top=stage1_importance(IMPORTANCE),
                   market=market_summary(prices))
    seen = {f.replace(" ", "") for f in BASE_FORMULAS.values()}

    # ---- round 1: directions from reasoning alone
    rows1, t1 = clean(ask(proposer, build_prompt_direction(**context), out, "round1"), prices, every, seen)
    e_search, e_q4, e_all = edges(t1, search), edges(t1, q4), edges(t1, every)
    by_year = pd.concat([edges(t1, [b]) for b in search], axis=1)
    for r in rows1:
        n = r["name"]
        r.update(round=1, edge_search=e_search[n], edge_q4=e_q4[n], edge_all=e_all[n],
                 years_same=int((np.sign(by_year.loc[n]) == np.sign(e_search[n])).sum()),
                 right=bool(np.sign(e_search[n]) == stated_signs([r["expected_sign"]])[0]))

    # ---- round 2: new factors after seeing the round-1 directions on the search days
    evidence = direction_evidence(rows1, sum(r["right"] for r in rows1))
    rows2, t2 = clean(ask(proposer, build_prompt_direction(**context, evidence=evidence), out, "round2"),
                      prices, every, seen)
    days = pd.DatetimeIndex(np.concatenate([d for d, _ in search]))
    similar = t2.reindex(days).apply(lambda c: t1.reindex(days).corrwith(c, method="spearman").abs().max())
    e2_search, e2_q4, e2_all = edges(t2, search), edges(t2, q4), edges(t2, every)
    for r in rows2:
        n = r["name"]
        r.update(round=2, edge_search=e2_search[n], edge_q4=e2_q4[n], edge_all=e2_all[n],
                 closest_round1=float(similar[n]), new_idea=bool(similar[n] < COPY))
    factors = pd.DataFrame(rows1 + rows2)
    for col in ("edge_search", "edge_q4", "edge_all"):
        factors[f"right_{col[5:]}"] = np.sign(factors[col]) == stated_signs(factors["expected_sign"])
    factors.drop(columns=["right"]).to_csv(out / "factors.csv", index=False)

    # ---- the tests
    new = [r["name"] for r in rows2 if r["new_idea"]]
    t2n = t2[new]
    exp1 = [r["expected_sign"] for r in rows1]
    exp2 = [r["expected_sign"] for r in rows2 if r["new_idea"]]
    k = dict(n=args.shuffles)
    tests = [("round 1: stated direction, all days", match_test(t1, exp1, every, **k)),
             ("round 1: stated direction, search days", match_test(t1, exp1, search, **k)),
             ("round 1: stated direction, sealed Q4", match_test(t1, exp1, q4, **k)),
             ("round 2 (new ideas): stated direction, sealed Q4", match_test(t2n, exp2, q4, **k)),
             ("round 2 (new ideas): stated direction, search days", match_test(t2n, exp2, search, **k))]
    ref = persistence(t2n, search, q4, **k)
    c1 = composite_test(composite(t1, stated_signs(exp1)), q4, **k)
    c2 = composite_test(composite(t2n, stated_signs(exp2)), q4, **k)
    summary = pd.DataFrame([{"test": name, **res} for name, res in tests])
    summary.to_csv(out / "summary.csv", index=False)
    extra = {"round1_up_share": float(np.mean([e == UP for e in exp1])), "round2_up_share": float(np.mean([e == UP for e in exp2])),
             "round2_factors": len(rows2), "round2_new_ideas": len(new),
             "reference_data_direction_search_to_q4": ref, "signal_round1_q4": c1, "signal_round2_q4": c2,
             "lesson": (json.loads((out / "round2_response.json").read_text()).get("lesson") or "")}
    (out / "summary_extra.json").write_text(json.dumps(extra, indent=2) + "\n")

    print(f"\nDIRECTION TEST ({tag}). Guessing gives 50%.")
    for name, r in tests:
        print(f"  {name:52} {r['match_rate']:.0%} right of {r['factors']:3d} "
              f"(p better than chance {r['p_better_than_chance']:.3f}, worse {r['p_worse_than_chance']:.3f})")
    print(f"  said 'higher means up': round 1 {extra['round1_up_share']:.0%}, round 2 {extra['round2_up_share']:.0%} "
          f"| round 2: {len(new)} new ideas of {len(rows2)}")
    print(f"  reference, data alone (search-day direction -> sealed Q4), same round-2 factors: "
          f"{ref['same_direction']:.0%} (p {ref['p_same_direction']:.3f}), correlation {ref['correlation']:+.2f} (p {ref['p_correlation']:.3f})")
    print(f"  one signal from the stated directions, sealed Q4: round 1 AUC {c1['auc']:.3f} (p {c1['p']:.3f}), "
          f"round 2 AUC {c2['auc']:.3f} (p {c2['p']:.3f})")
    if extra["lesson"]:
        print(f"  the LLM's lesson: {extra['lesson']}")
    print(f"Results in {out}")
