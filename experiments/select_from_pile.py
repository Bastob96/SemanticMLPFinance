"""Liu's suggestion (21 Oct): generate a large set of factors with the LLM, then use feature selection to find
a good combination. Here the "pile" is every factor Claude has written so far (15 Oct and 21 Oct runs).

Selection (forward selection, the same rules as the loop's team trial and pair trial, 8 checks):
  1. Every factor alone (from results/stage2/checks8_real.csv). Near-copies of the 22 factors are dropped.
  2. Candidates: factors that helped in at least half the checks with a positive average ("promising").
  3. Team: add the candidate that improves the team most if it passes the team bar; re-test the rest on
     the bigger team; repeat until none passes or the team has 6 factors.
  4. Pairs: the best promising pairs (different kinds first) are tried on top of the team; a pair joins if it
     passes the pair bar and beats the two gains added up.
Bars come from results/stage2/rules_v2_8checks.json (set with random numbers before any real factor was looked at).

Usage: python experiments/select_from_pile.py
Writes results/stage2/pile_8checks/ (selection log, pairs, kept_factors.json).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import build_stage1_features  # noqa: E402
from src.stage2.formula import evaluate, wide_prices  # noqa: E402
from src.stage2.scoring import BASE, Scorer  # noqa: E402
from src.stage2.selection import group_of, summarise_result  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "results/stage2/pile_8checks"
MAX_PAIRS = 12

if __name__ == "__main__":
    rules = json.loads((ROOT / "results/stage2/rules_v2_8checks.json").read_text())
    k, bar, pair_gain, pair_extra = rules["min_years"], rules["team_gain"], rules["pair_gain"], rules["pair_synergy"]
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(ROOT / "data/processed/aligned_daily_ohlcv_5y.csv")
    features, prices = build_stage1_features(raw), wide_prices(raw)
    scorer = Scorer(features, checks=tuple(rules["checks"]))
    real = pd.read_csv(ROOT / "results/stage2/checks8_real.csv").drop_duplicates("formula")
    hyp = {}
    for run in ("runs/20261015_1635_claude", "runs_v2/20261021_1105_claude"):
        b = pd.read_csv(ROOT / "results/stage2" / run / "scoreboard.csv")
        hyp.update(dict(zip(b["name"], b["hypothesis"])))
    values = {r.factor: evaluate(r.formula, prices) for r in real.itertuples()}

    dates = pd.to_datetime(features["Date"])
    idx = pd.DatetimeIndex(dates[(dates.dt.year >= 2022) & (dates.dt.month <= 9)])
    ref = features.set_index(pd.to_datetime(features["Date"]))[BASE]

    def similarity(name, team):
        r = ref.copy()
        for n in team:
            r[n] = values[n].reindex(r.index)
        return float(r.loc[idx].corrwith(values[name].reindex(r.index).loc[idx], method="spearman").abs().max())

    real["similarity"] = [similarity(n, []) for n in real.factor]
    pile = real[real.similarity < rules["max_corr"]]
    cands = list(pile[(pile.gain > 0) & (pile.checks >= 4)].sort_values("gain", ascending=False).factor)
    print(f"Pile: {len(real)} different factors, {len(pile)} not copies, {len(cands)} promising candidates", flush=True)

    team, log, how = [], [], {}
    while len(team) < rules["team_size"]:
        best = None
        for c in cands:
            if c in team:
                continue
            if team:
                res = summarise_result(scorer.gains({c: values[c]}, {n: values[n] for n in team}))
            else:                                   # alone = already measured in checks8_real.csv
                row = pile.set_index("factor").loc[c]
                res = {"gain": float(row.gain), "years": int(row.checks)}
            sim = similarity(c, team)
            log.append({"step": len(team) + 1, "team_size": len(team), "factor": c, "gain": res["gain"],
                        "checks": res["years"], "similarity": sim})
            if res["years"] >= k and res["gain"] >= bar and sim < rules["max_corr"]:
                if best is None or res["gain"] > best[1]:
                    best = (c, res["gain"], res["years"])
        if best is None:
            break
        team.append(best[0])
        how[best[0]] = f"forward selection step {len(team)}: gain on the team {best[1]:+.3f}, helped {best[2]}/8 checks"
        print(f"  joined: {best[0]} ({how[best[0]]})", flush=True)
    pd.DataFrame(log).to_csv(OUT / "selection_log.csv", index=False)

    rest = [c for c in cands if c not in team]
    groups = {c: group_of(pile.set_index("factor").loc[c, "formula"]) for c in rest}
    pairs = sorted(((a, b) for i, a in enumerate(rest) for b in rest[i + 1:]),
                   key=lambda p: (groups[p[0]] == groups[p[1]], rest.index(p[0]) + rest.index(p[1])))
    pair_rows, alone = [], {}
    tv = {n: values[n] for n in team}
    for a, b in pairs:
        if len(pair_rows) >= MAX_PAIRS or len(team) + 2 > rules["team_size"]:
            break
        if a in team or b in team or abs(values[a].reindex(idx).corr(values[b].reindex(idx), method="spearman")) >= rules["pair_max_corr"]:
            continue
        for x in (a, b):
            if x not in alone:
                alone[x] = summarise_result(scorer.gains({x: values[x]}, tv))["gain"]
        res = summarise_result(scorer.gains({a: values[a], b: values[b]}, tv))
        extra = res["gain"] - alone[a] - alone[b]
        joined = res["years"] >= k and res["gain"] >= pair_gain and extra >= pair_extra
        pair_rows.append({"a": a, "b": b, "group_a": groups[a], "group_b": groups[b], "gain": res["gain"],
                          "checks": res["years"], "a_alone": alone[a], "b_alone": alone[b], "extra": extra, "joined": joined})
        print(f"  pair {a} + {b}: {res['gain']:+.4f} ({res['years']}/8), extra {extra:+.4f}{' -> JOINED' if joined else ''}", flush=True)
        if joined:
            team += [a, b]
            how[a] = how[b] = f"pair trial ({a} + {b}): pair gain {res['gain']:+.3f}, {extra:+.3f} above the two added up"
            tv = {n: values[n] for n in team}
            alone = {}
    pd.DataFrame(pair_rows).to_csv(OUT / "pairs.csv", index=False)

    f = pile.set_index("factor")
    kept = [{"name": n, "formula": f.loc[n, "formula"], "hypothesis": hyp.get(n, ""), "round": f.loc[n, "kind"],
             "how_selected": how[n], "auc_gain": float(f.loc[n, "gain"]), "years_helped": int(f.loc[n, "checks"])}
            for n in team]
    (OUT / "kept_factors.json").write_text(json.dumps(kept, indent=2) + "\n")
    pile.assign(name=pile.factor, status=["kept" if n in team else "candidate" for n in pile.factor]).to_csv(
        OUT / "scoreboard.csv", index=False)
    print("\nSelected:", team or "(none)")
