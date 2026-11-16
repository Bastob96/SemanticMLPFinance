"""Re-judge every run's selected factors with the luck-proof checks (src/stage2/familywise.py). No API calls.

For each run with kept factors:
  1. Family-wise bar: a factor that joined the team in a team trial must beat the best of N random-number
     factors (N = factors the run tested); a factor that joined in a pair trial, the best of P random pairs
     (P = pair trials the run made). Each stage at 2.5%, so a run of pure luck keeps anything at most about
     5% of the time.
  2. Direction: the sign of the factor's LR coefficient in each year; and, where the factor came with a
     direction stated in advance (expected_sign, the informed Opus run), how often it matches.
  3. Recurrence: which kept ideas were found by more than one independent LLM search (replays of a run
     count as the same search; templates and random numbers are counted separately).
Then the pooled Q4 test (experiments/run_pooled_test.py) on the survivors of each run and on the recurring ideas.

Usage: python experiments/calibrate_familywise.py      (once, about 30 min)
       python experiments/rejudge_familywise.py        (then this)
Writes results/stage2/familywise/: rejudge_runs.csv, rejudge_factors.csv, sets/<name>/kept_factors.json
and results/stage2/pooled/familywise_summary.csv.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import build_stage1_features  # noqa: E402
from src.stage2.familywise import (bar, check_days, lr_signs, normalise, same_factor, sign_summary,  # noqa: E402
                                   values_of)
from src.stage2.formula import wide_prices  # noqa: E402

warnings.filterwarnings("ignore")
DATA = ROOT / "data/processed/aligned_daily_ohlcv_5y.csv"
S = ROOT / "results/stage2"
OUT = S / "familywise"
LEVEL = 0.975                    # per stage (team trial, pair trial): about 5% for a whole run


def find_runs():
    runs = {"one_at_a_time_A": S / "runs/20261015_1635_claude"}
    for d in sorted((S / "runs_v2").iterdir()):
        if d.is_dir() and (d / "kept_factors.json").exists() and (d / "scoreboard.csv").exists() and "sk-ant" not in d.name:
            runs[d.name] = d
    return runs


def describe(folder: Path):
    """kind (claude / template / noise), model and the independent search it belongs to."""
    cfg = json.loads((folder / "config.json").read_text()) if (folder / "config.json").exists() else {}
    rules = cfg.get("rules", {})
    if rules.get("curve") or len(rules.get("checks", ["q3"])) > 1:
        return None                                          # other rules: not comparable with these bars
    proposer = cfg.get("proposer", "claude")
    source = folder.name
    if proposer == "replay":
        source = Path(cfg["replay"]).name
        proposer = "claude"                                  # every replay in this project replays Claude run A
    kind = {"claude": "claude", "template": "template", "noise": "noise"}.get(proposer, proposer)
    if "random" in folder.name and kind == "claude":
        kind = "template"
    model = cfg.get("model", "claude-sonnet-5-5") if kind == "claude" else "-"
    return {"kind": kind, "model": model, "source": source, "batch": int(cfg.get("batch", 0) or 0)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pooled", action="store_true", help="skip the pooled Q4 test at the end")
    ap.add_argument("--draws", type=int, default=200)
    args = ap.parse_args()
    solo_null = pd.read_csv(OUT / "noise_solo.csv")["gain"]
    pair_null = pd.read_csv(OUT / "noise_pairs.csv")["gain"]
    raw = pd.read_csv(DATA)
    features, prices = build_stage1_features(raw), wide_prices(raw)
    days = check_days(features)

    run_rows, fac_rows, values = [], [], {}
    for name, folder in find_runs().items():
        info = describe(folder)
        if info is None:
            print(f"skip {name}: different rules (8 checks or curve)")
            continue
        board = pd.read_csv(folder / "scoreboard.csv")
        pairs = pd.read_csv(folder / "pairs.csv") if (folder / "pairs.csv").exists() and (folder / "pairs.csv").stat().st_size > 1 else pd.DataFrame()
        kept = json.loads((folder / "kept_factors.json").read_text())
        n_tested = int((board["status"] != "invalid").sum())
        n_pairs = len(pairs)
        solo_bar, pair_bar = bar(solo_null, n_tested, LEVEL), bar(pair_null, max(n_pairs, 1), LEVEL)
        survivors = []
        for k in kept:
            rec = board[board["name"] == k["name"]].iloc[0]
            how = str(rec.get("how_selected", "") or k.get("how_selected", "") or "")
            if how.startswith("pair trial"):
                row = pairs[(pairs["joined"].astype(str) == "True") & ((pairs["a"] == k["name"]) | (pairs["b"] == k["name"]))]
                evidence, need, stage = float(row["gain"].iloc[0]), pair_bar["bar"], "pair"
            else:
                tg = rec.get("team_gain")
                evidence = float(tg) if pd.notna(tg) else float(rec["auc_gain"])
                need, stage = solo_bar["bar"], "team"
            v = values_of(k["formula"], prices)
            values[(name, k["name"])] = v
            sig = sign_summary(lr_signs(v, features), rec.get("expected_sign") if "expected_sign" in rec else None)
            ok = evidence >= need
            if ok:
                survivors.append(k)
            fac_rows.append({"run": name, **info, "factor": k["name"], "formula": k["formula"], "stage": stage,
                             "evidence_gain": evidence, "familywise_bar": need, "survives": ok, **sig})
        run_rows.append({"run": name, **info, "tested": n_tested, "pair_trials": n_pairs, "kept": len(kept),
                         "survive": len(survivors), "solo_bar": solo_bar["bar"], "pair_bar": pair_bar["bar"],
                         "survivors": "|".join(k["name"] for k in survivors)})
        out = OUT / "sets" / f"fw_{name}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "kept_factors.json").write_text(json.dumps(survivors, indent=2, default=float) + "\n")
        print(f"{name:42} {info['kind']:8} tested {n_tested:3}, kept {len(kept)}, survive {len(survivors)} "
              f"(bars: team {solo_bar['bar']:+.4f}, pair {pair_bar['bar']:+.4f})", flush=True)

    # recurrence: group kept ideas that are the same factor (equal formula or almost identical values)
    facs = pd.DataFrame(fac_rows)
    keys = list(zip(facs["run"], facs["factor"]))

    def groups(threshold):
        """Complete linkage: a factor joins a group only if it is that similar to EVERY member (no chains)."""
        out, members = [-1] * len(keys), []
        for i in range(len(keys)):
            for g, mem in enumerate(members):
                if facs.at[i, "kind"] == facs.at[mem[0], "kind"] and all(
                        normalise(facs.at[i, "formula"]) == normalise(facs.at[j, "formula"])
                        or same_factor(values[keys[i]], values[keys[j]], days, threshold) for j in mem):
                    mem.append(i)
                    out[i] = g
                    break
            else:
                members.append([i])
                out[i] = len(members) - 1
        return out
    facs["idea"] = groups(0.9)                 # the same idea (the project's near-copy threshold)
    facs["family"] = groups(0.8)               # close variants of one idea
    facs["independent_searches"] = facs.groupby("idea")["source"].transform("nunique")
    facs["searches_with_variants"] = facs.groupby("family")["source"].transform("nunique")
    facs.to_csv(OUT / "rejudge_factors.csv", index=False)
    pd.DataFrame(run_rows).to_csv(OUT / "rejudge_runs.csv", index=False)

    recurring = facs[(facs["kind"] == "claude") & (facs["independent_searches"] >= 2)].drop_duplicates("idea")
    rec_set = [{"name": r.factor, "formula": r.formula} for r in recurring.itertuples()]
    (OUT / "sets" / "recurring_llm").mkdir(parents=True, exist_ok=True)
    (OUT / "sets" / "recurring_llm" / "kept_factors.json").write_text(json.dumps(rec_set, indent=2) + "\n")
    print("\nIdeas kept by 2+ independent LLM searches (similarity 0.9+):")
    for r in recurring.itertuples():
        print(f"  {r.factor} ({r.independent_searches} searches, signs {r.signs}) = {r.formula}")
    variants = facs[(facs["kind"] == "claude") & (facs["searches_with_variants"] > facs["independent_searches"])]
    print("Ideas with close variants (0.8-0.9) in other searches:")
    for fam, d in variants.groupby("family"):
        print(f"  {' / '.join(sorted(set(d['factor'])))} ({d['source'].nunique()} searches)")
    print("\nPer kind:", pd.DataFrame(run_rows).groupby("kind")[["kept", "survive"]].sum().to_dict("index"))

    if not args.no_pooled:
        sets = [f"{d.name}={d}" for d in sorted((OUT / "sets").iterdir())
                if json.loads((d / "kept_factors.json").read_text())]
        subprocess.run([sys.executable, str(ROOT / "experiments/run_pooled_test.py"), *sets,
                        "--draws", str(args.draws), "--name", "familywise_summary.csv"], check=True)
