"""Interaction-aware iterative factor search (version 2, after Liu's 20 Oct feedback).

Every round:
  1. Claude gets the full lab notebook (report card for every factor tried, the selected team,
     the promising list and the pair results) and returns lessons for the last round plus
     IMPROVE, NEW and PARTNER factors.
  2. Each formula is checked and calculated (past data only).
  3. Solo test: each factor added to the 22 Stage 1 factors (same as version 1).
  4. Team trial: each factor added to the 22 factors plus the selected team; the best ones that
     clearly improve the team join it (best first, team stays small).
  5. Pair trials: a few promising factors, preferably of different kinds, are tested two at a time;
     a pair that does clearly better than its two members' gains added up joins the team together
     (two lucky factors only add up their luck; real teamwork gives more than the sum).
  6. Factors that are weak alone but show some sign of life stay on the promising list.
  7. Everything is saved and goes back to Claude in the next round.
finalize() gives the promising list one more team trial at the end. It is off by default in
experiments/run_stage2_loop_v2.py because the random-number check showed it lets luck in (1 in 18).
The Q4 test quarters are never used here.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .context import BASE_FORMULAS, market_summary, stage1_importance
from .feedback import board_records, lab_notebook
from .formula import FormulaError, evaluate, validate
from .llm import RandomProposer, dump
from .prompt_v2 import build_prompt_v2
from .scoring import BASE, Scorer
from .selection import Rules, choose_pairs, group_of, is_promising, passes, summarise_result


def best_gain(rec):
    """Best average gain seen for a factor: alone, or on top of the selected team."""
    team = rec.get("team_gain", -1) if rec.get("team_size") else -1
    return max(rec.get("auc_gain", -1), team)


@dataclass
class LoopV2:
    proposer: object
    features: pd.DataFrame
    prices: pd.DataFrame
    out_dir: Path
    stage1_importance_path: Path
    rules: Rules = field(default_factory=Rules)
    n_improve: int = 2
    n_new: int = 2
    n_partner: int = 2
    board: list = field(default_factory=list)     # one record per factor tried
    pairs: list = field(default_factory=list)     # one record per pair trial
    team: list = field(default_factory=list)      # names of the selected team, in the order they joined
    log: list = field(default_factory=list)       # one summary per round

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.scorer = Scorer(self.features)
        self.market = market_summary(self.prices)
        self.stage1_top = stage1_importance(self.stage1_importance_path)
        dates = pd.to_datetime(self.features["Date"])
        self.check_idx = pd.DatetimeIndex(dates[(dates.dt.year >= 2022) & (dates.dt.month <= 9)])   # Q1-Q3 only
        self.reference = self.features.set_index(pd.to_datetime(self.features["Date"]))[BASE]
        self.values, self.solo = {}, {}
        self._team_cache, self._corr_cache = {}, {}

    # ---------------- small helpers ----------------
    def rec(self, name):
        return next(r for r in self.board if r["name"] == name)

    def team_values(self):
        return {n: self.values[n] for n in self.team}

    def similarity(self, name):
        """Largest |Spearman correlation| with the 22 factors and the selected team (Q1-Q3 days)."""
        ref = self.reference.copy()
        for n in self.team:
            if n != name:
                ref[n] = self.values[n].reindex(ref.index)
        x = self.values[name].reindex(ref.index).loc[self.check_idx]
        c = ref.loc[self.check_idx].corrwith(x, method="spearman").abs()
        return float(c.max()), str(c.idxmax())

    def corr(self, a, b):
        key = tuple(sorted((a, b)))
        if key not in self._corr_cache:
            xa, xb = self.values[a].reindex(self.check_idx), self.values[b].reindex(self.check_idx)
            self._corr_cache[key] = float(xa.corr(xb, method="spearman"))
        return self._corr_cache[key]

    def team_result(self, name):
        """The factor added on top of the 22 factors plus the current team (cached per team)."""
        key = (name, tuple(sorted(self.team)))
        if key not in self._team_cache:
            if not self.team:
                self._team_cache[key] = self.solo[name]          # with no team this is the solo test
            else:
                self._team_cache[key] = summarise_result(
                    self.scorer.gains({name: self.values[name]}, self.team_values()))
        return self._team_cache[key]

    # ---------------- one factor ----------------
    def _build(self, cand, round_no):
        rec = {"round": round_no, "name": str(cand.get("name", "?")), "mode": cand.get("mode", "?"),
               "parent": cand.get("parent", "") or "", "role": cand.get("role", "") or "",
               "formula": str(cand.get("formula", "")), "hypothesis": cand.get("hypothesis", ""),
               "status": "invalid", "reason": "", "lesson": "", "how_selected": ""}
        names = {r["name"] for r in self.board} | set(BASE)
        base_name, k = rec["name"], 1
        while rec["name"] in names:
            rec["name"] = f"{base_name}_r{round_no}" + (f"_{k}" if k > 1 else "")
            k += 1
        formulas = {r["formula"].replace(" ", "") for r in self.board} | \
                   {f.replace(" ", "") for f in BASE_FORMULAS.values()}
        if rec["formula"].replace(" ", "") in formulas:
            rec["reason"] = "duplicate formula"
            return rec
        try:
            validate(rec["formula"])
            values = evaluate(rec["formula"], self.prices)
        except (FormulaError, KeyError, ValueError) as exc:
            rec["reason"] = str(exc)[:200]
            return rec
        if values.loc["2022":].isna().any():
            rec["reason"] = "missing values in 2022-2025 (e.g. division by zero)"
            return rec
        self.values[rec["name"]] = values
        rec["group"] = group_of(rec["formula"], rec["role"])
        rec["status"] = "tested"
        return rec

    def _write_solo(self, rec, res):
        rec.update(auc_gain=res["gain"], years_helped=res["years"], auc_gain_lr=res["lr"], auc_gain_rf=res["rf"],
                   **{f"gain_{y}": g for y, g in res["by_year"].items()})

    def _write_team(self, rec, res):
        rec.update(team_size=len(self.team), team_gain=res["gain"], team_years_helped=res["years"],
                   team_lr=res["lr"], team_rf=res["rf"])

    # ---------------- team trial ----------------
    def _grow_team(self, recs, how):
        """Team trial for each factor. The best one that clearly improves the team joins it, then the rest are
        re-tested on the bigger team, until none passes. When the team is full the trial is still run and
        reported, but nothing joins."""
        while True:
            best = None
            for r in recs:
                if r["status"] == "kept":
                    continue
                res = self.team_result(r["name"])
                if not res.get("valid"):
                    continue
                self._write_team(r, res)
                r["max_corr"], r["closest"] = self.similarity(r["name"])
                if passes(res, self.rules, self.rules.team_gain) and r["max_corr"] < self.rules.max_corr:
                    if best is None or res["gain"] > best[1]["gain"]:
                        best = (r, res)
            if best is None:
                return
            r, res = best
            if len(self.team) >= self.rules.team_size:
                r["team_full"] = True
                return
            self.team.append(r["name"])
            r["status"], r["how_selected"] = "kept", how
            r["reason"] = f"joined the team ({how}): gain on the team {res['gain']:+.3f}, helped {res['years']}/4 years"

    # ---------------- pair trials ----------------
    def _pair_trials(self, round_no, new_names):
        """A few pair trials. PARTNER factors are always tried with their parent (they were designed to work
        together), even when weak alone. Other pairs: this round's promising factors with the promising list."""
        pool = [r for r in self.board if r["status"] in ("tested", "promising") and r["name"] in self.solo
                and is_promising(r, self.rules)]
        info = {r["name"]: {"group": r.get("group", ""), "score": best_gain(r)} for r in pool}
        forced = []
        for n in new_names:
            r = self.rec(n)
            p = next((x for x in self.board if x["name"] == r["parent"]), None)
            if (r["mode"] == "partner" and p is not None and p["status"] in ("tested", "promising", "rejected")
                    and p["name"] in self.solo and r.get("max_corr", 1) < self.rules.max_corr
                    and p.get("max_corr", 1) < self.rules.max_corr):
                for x in (r, p):
                    info.setdefault(x["name"], {"group": x.get("group", ""), "score": best_gain(x)})
                forced.append((n, p["name"]))
        new_promising = [n for n in new_names if n in {r["name"] for r in pool}]
        pool_names = [r["name"] for r in pool]
        for a, b in choose_pairs(new_promising, pool_names, info, self.corr, self.rules, forced):
            if a in self.team or b in self.team:
                continue
            res = summarise_result(self.scorer.gains({a: self.values[a], b: self.values[b]}, self.team_values()))
            ra, rb = self.team_result(a), self.team_result(b)
            if not (res.get("valid") and ra.get("valid") and rb.get("valid")):
                continue
            if self.team:                                  # keep each member's team result up to date
                self._write_team(self.rec(a), ra)
                self._write_team(self.rec(b), rb)
            synergy = res["gain"] - (ra["gain"] + rb["gain"])    # extra over the two gains added up
            room = len(self.team) + 2 <= self.rules.team_size
            not_copies = all(self.similarity(x)[0] < self.rules.max_corr for x in (a, b))   # vs the current team
            joined = (passes(res, self.rules, self.rules.pair_gain) and synergy >= self.rules.pair_synergy
                      and room and not_copies)
            self.pairs.append({"round": round_no, "a": a, "b": b,
                               "group_a": info[a]["group"], "group_b": info[b]["group"],
                               "team_size": len(self.team), "gain": res["gain"], "years": res["years"],
                               "lr": res["lr"], "rf": res["rf"],
                               **{f"gain_{y}": g for y, g in res["by_year"].items()},
                               "a_alone_on_team": ra["gain"], "b_alone_on_team": rb["gain"],
                               "synergy": synergy, "joined": joined})
            for x, y in ((a, b), (b, a)):
                rx = self.rec(x)
                if rx.get("best_pair_gain") is None or res["gain"] > rx["best_pair_gain"]:
                    rx.update(best_pair_partner=y, best_pair_gain=res["gain"], best_pair_years=res["years"],
                              best_pair_synergy=synergy)
            if joined:
                for x, y in ((a, b), (b, a)):
                    rx = self.rec(x)
                    self.team.append(x)
                    rx["status"], rx["how_selected"] = "kept", f"pair trial with {y}, round {round_no}"
                    rx["reason"] = (f"joined the team as a pair with {y}: pair gain {res['gain']:+.3f}, helped "
                                    f"{res['years']}/4 years, {synergy:+.3f} more than the two alone added up")

    # ---------------- statuses ----------------
    def _set_status(self, r):
        if r["status"] in ("kept", "invalid"):
            return
        result = f"alone {r.get('auc_gain', float('nan')):+.3f} ({r.get('years_helped', 0)}/4 years)"
        if r.get("team_size"):
            result += f", on the team {r['team_gain']:+.3f} ({r['team_years_helped']}/4 years)"
        if r.pop("team_full", False):
            r["status"] = "promising"
            r["reason"] = f"passed the team trial but the team is already full ({self.rules.team_size}): {result}"
            return
        if is_promising(r, self.rules):
            r["status"] = "promising"
            r["reason"] = f"not enough for the team yet but some sign of life: {result}"
            return
        r["status"] = "rejected"
        if r.get("max_corr", 0) >= self.rules.max_corr:
            r["reason"] = f"near-copy of {r.get('closest', '')} ({r['max_corr']:.2f})"
        else:
            r["reason"] = f"too weak: {result}"

    # ---------------- text for Claude ----------------
    def team_text(self):
        if not self.team:
            return "(empty so far)"
        return "\n".join(f"- {n} [{self.rec(n).get('group', '')}] = {self.rec(n)['formula']} "
                         f"(joined: {self.rec(n)['how_selected']})" for n in self.team)

    def promising_text(self, limit=12):
        pool = [r for r in self.board if r["status"] == "promising"]
        pool.sort(key=lambda r: -best_gain(r))
        if not pool:
            return "(empty so far)"
        lines = []
        for r in pool[:limit]:
            line = f"- {r['name']} [{r.get('group', '')}] alone {r.get('auc_gain', 0):+.3f} ({r.get('years_helped', 0)}/4 years)"
            if r.get("team_size"):
                line += f", on the team {r['team_gain']:+.3f} ({r['team_years_helped']}/4)"
            if r.get("best_pair_partner"):
                line += f", best pair with {r['best_pair_partner']} {r['best_pair_gain']:+.3f}"
            lines.append(line + f" | {r['formula']}")
        return "\n".join(lines)

    def pairs_text(self, limit=12):
        if not self.pairs:
            return "(none yet)"
        lines = []
        for p in self.pairs[-limit:]:
            years = ", ".join(f"{y} {p.get(f'gain_{y}', float('nan')):+.3f}" for y in (2022, 2023, 2024, 2025))
            lines.append(f"- R{p['round']}: {p['a']} [{p['group_a']}] + {p['b']} [{p['group_b']}]: pair {p['gain']:+.3f} "
                         f"({years}; {p['years']}/4 years; LR {p['lr']:+.3f}, RF {p['rf']:+.3f}); alone on the team "
                         f"{p['a_alone_on_team']:+.3f} and {p['b_alone_on_team']:+.3f}, extra over the two added up "
                         f"{p['synergy']:+.3f} -> "
                         f"{'JOINED the team' if p['joined'] else 'not joined'}")
        return "\n".join(lines)

    # ---------------- one round ----------------
    def run_round(self, round_no):
        t0 = time.time()
        previous = [r["name"] for r in self.board if r["round"] == round_no - 1]
        prompt = build_prompt_v2(round_no=round_no, n_improve=self.n_improve, n_new=self.n_new,
                                 n_partner=self.n_partner, base_formulas=BASE_FORMULAS,
                                 stage1_top=self.stage1_top, market=self.market,
                                 notebook=lab_notebook(self.board), team_text=self.team_text(),
                                 promising_text=self.promising_text(), pairs_text=self.pairs_text(),
                                 rules=self.rules.as_dict(), previous_names=previous)
        tag = f"round_{round_no:02d}"
        (self.out_dir / f"{tag}_prompt.txt").write_text(prompt)
        response = self.proposer(prompt)
        (self.out_dir / f"{tag}_response.json").write_text(dump(response))

        for item in response.get("lessons", []) or []:          # Claude's lessons about the last round
            if not isinstance(item, dict):
                continue
            for r in self.board:
                if r["round"] == round_no - 1 and r["name"] == item.get("name"):
                    r["lesson"] = str(item.get("lesson", ""))[:300]

        n_total = self.n_improve + self.n_new + self.n_partner
        new = []
        for cand in response.get("factors", [])[:n_total]:
            r = self._build(cand, round_no)          # added one by one, so repeats within a round are caught too
            self.board.append(r)
            new.append(r)
        tested = [r for r in new if r["status"] == "tested"]
        for r in tested:                                          # 1. solo test
            res = summarise_result(self.scorer.gains({r["name"]: self.values[r["name"]]}, {}))
            if not res["valid"]:
                r["status"], r["reason"] = "invalid", "constant or missing in a training/validation quarter"
                continue
            self.solo[r["name"]] = res
            self._write_solo(r, res)
            r["max_corr"], r["closest"] = self.similarity(r["name"])
        tested = [r for r in tested if r["status"] == "tested"]
        team_before = len(self.team)
        self._grow_team(tested, how=f"team trial, round {round_no}")    # 2. team trial
        pairs_before = len(self.pairs)
        self._pair_trials(round_no, [r["name"] for r in tested if r["status"] != "kept"])   # 3. pair trials
        for r in self.board:                                       # 4. statuses (old promising ones may have joined)
            if r["status"] in ("tested", "promising"):
                self._set_status(r)
        for r in new:
            alone = f"{r['auc_gain']:+.4f} ({r['years_helped']}/4)" if "auc_gain" in r else "-"
            team = f"{r['team_gain']:+.4f} ({r['team_years_helped']}/4)" if r.get("team_size") else "-"
            print(f"  [{r['status']:9}] {r['name']}: alone {alone} | on team {team} "
                  f"| similarity {r.get('max_corr', float('nan')):.2f} | {r['reason']}", flush=True)
        new_pairs = self.pairs[pairs_before:]
        for p in new_pairs:
            print(f"  pair {p['a']} + {p['b']}: {p['gain']:+.4f} ({p['years']}/4), extra {p['synergy']:+.4f}"
                  f"{'  -> JOINED' if p['joined'] else ''}", flush=True)
        self.log.append({"round": round_no, "team_size": len(self.team), "joined": len(self.team) - team_before,
                         "promising": sum(r["status"] == "promising" for r in self.board),
                         "pairs_tried": len(new_pairs), "pairs_joined": sum(p["joined"] for p in new_pairs),
                         "mean_alone_gain": float(np.nanmean([r.get("auc_gain", np.nan) for r in tested])) if tested else np.nan,
                         "team": "|".join(self.team), "seconds": round(time.time() - t0, 1)})
        self.save()

    def finalize(self):
        """One more team trial for everything on the promising list, against the final team."""
        pool = [r for r in self.board if r["status"] == "promising"]
        before = len(self.team)
        self._grow_team(pool, how="final check of the promising list")
        for r in pool:
            self._set_status(r)
        self.log.append({"round": "final", "team_size": len(self.team), "joined": len(self.team) - before,
                         "promising": sum(r["status"] == "promising" for r in self.board),
                         "pairs_tried": 0, "pairs_joined": 0, "mean_alone_gain": np.nan,
                         "team": "|".join(self.team), "seconds": 0})
        self.save()

    # ---------------- saving and continuing ----------------
    def load_state(self) -> int:
        """Continue a stopped run from its saved files. Returns the next round number."""
        self.board = board_records(pd.read_csv(self.out_dir / "scoreboard.csv"))
        for r in self.board:
            if r["status"] == "invalid":
                continue
            self.values[r["name"]] = evaluate(r["formula"], self.prices)
            if "auc_gain" in r:
                self.solo[r["name"]] = {"valid": True, "gain": r["auc_gain"], "years": r["years_helped"],
                                        "lr": r["auc_gain_lr"], "rf": r["auc_gain_rf"],
                                        "by_year": {y: r[f"gain_{y}"] for y in (2022, 2023, 2024, 2025)}}
        try:
            self.pairs = pd.read_csv(self.out_dir / "pairs.csv").to_dict("records")
        except (FileNotFoundError, pd.errors.EmptyDataError):
            self.pairs = []
        self.log = pd.read_csv(self.out_dir / "rounds.csv").to_dict("records")
        self.team = [k["name"] for k in json.loads((self.out_dir / "kept_factors.json").read_text())]
        return int(max(r["round"] for r in self.board)) + 1

    def save(self):
        pd.DataFrame(self.board).to_csv(self.out_dir / "scoreboard.csv", index=False)
        pd.DataFrame(self.pairs).to_csv(self.out_dir / "pairs.csv", index=False)
        pd.DataFrame(self.log).to_csv(self.out_dir / "rounds.csv", index=False)
        kept = [{k: self.rec(n).get(k) for k in ("name", "formula", "hypothesis", "round", "how_selected",
                                                  "auc_gain", "years_helped", "team_gain")} for n in self.team]
        (self.out_dir / "kept_factors.json").write_text(json.dumps(kept, indent=2, default=float) + "\n")
        (self.out_dir / "rules.json").write_text(json.dumps(self.rules.as_dict(), indent=2) + "\n")


class TemplateProposerV2:
    """No-LLM stand-in for testing the loop offline: template formulas, modes and partners filled in."""

    def __init__(self, seed=0, n_improve=2, n_new=2, n_partner=2):
        self.base = RandomProposer(seed)
        self.modes = ["improve"] * n_improve + ["new"] * n_new + ["partner"] * n_partner

    def __call__(self, prompt):
        section = prompt.split("PROMISING LIST", 1)[1].split("PAIR TRIALS SO FAR", 1)[0]
        promising = re.findall(r"^- (\S+) \[", section, flags=re.M)
        out = []
        for i, mode in enumerate(self.modes):
            parent = promising[i % len(promising)] if (mode == "partner" and promising) else ""
            out.append({"name": f"template_{self.base.rng.randrange(10 ** 6)}", "mode": mode, "parent": parent,
                        "role": "", "formula": self.base._formula(), "hypothesis": "template formula (no LLM)"})
        return {"reflection": "template formulas (no LLM)", "lessons": [], "factors": out}


class ReplayProposer:
    """Replays the factors from an earlier run, round by round (for 'old clues, new judging')."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)

    def __call__(self, prompt):
        round_no = int(re.match(r"ROUND (\d+)", prompt).group(1))
        saved = json.loads((self.run_dir / f"round_{round_no:02d}_response.json").read_text())
        factors = [{**f, "mode": "improve" if f.get("mode") == "variant" else f.get("mode", "new"), "role": ""}
                   for f in saved.get("factors", [])]
        return {"reflection": f"replay of {self.run_dir.name}, round {round_no}", "lessons": [], "factors": factors}
