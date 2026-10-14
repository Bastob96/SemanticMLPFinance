"""The automated factor-discovery loop: propose -> build -> score -> select -> feed back.

Every round:
  1. The proposer (an LLM) gets the context and the full scoreboard and returns
     VARIANT factors (improve something that worked) and NEW factors (a different idea).
  2. Each formula is validated and computed with src.stage2.formula (past data only).
  3. Each factor is scored with src.stage2.scoring on the 2022-2025 validation quarters.
  4. Selection: keep it if years_helped >= 3, mean auc_gain >= 0.01 (above noise level) and it is not a near-copy
     (|Spearman| < 0.9) of an existing or already kept factor.
  5. Results (including errors) go on the scoreboard, which the next prompt includes.
Q4 test quarters are never used inside the loop.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .context import BASE_FORMULAS, market_summary, stage1_importance
from .formula import FormulaError, evaluate, validate
from .llm import build_prompt, dump
from .scoring import BASE, Scorer, summarise

# MIN_GAIN set from 32 pure-noise factors (experiments/calibrate_noise.py):
# 6/32 noise factors pass 'years>=3, gain>0', only 1/32 also passes gain >= 0.01.
MIN_YEARS, MIN_GAIN, MAX_CORR = 3, 0.01, 0.9


@dataclass
class Loop:
    proposer: object
    raw: pd.DataFrame               # long OHLCV table
    features: pd.DataFrame          # Stage 1 feature table
    prices: pd.DataFrame            # wide OHLCV table
    out_dir: Path
    stage1_importance_path: Path
    n_variant: int = 3
    n_new: int = 3
    board: list = field(default_factory=list)
    kept: list = field(default_factory=list)

    def __post_init__(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.scorer = Scorer(self.features)
        self.market = market_summary(self.prices)
        self.stage1_top = stage1_importance(self.stage1_importance_path)
        dates = pd.to_datetime(self.features["Date"])
        self._eval_dates = dates[(dates >= "2022-01-01") & (dates.dt.month <= 9)]
        self._reference = self.features.set_index(pd.to_datetime(self.features["Date"]))[BASE]
        self._kept_values = {}

    # ---------- scoreboard text for the prompt ----------
    def scoreboard_text(self, limit=40) -> str:
        if not self.board:
            return "(empty: this is the first round)"
        df = pd.DataFrame(self.board)
        scored = df[df.status.isin(["kept", "rejected"])].sort_values("auc_gain", ascending=False)
        failed = df[df.status.eq("invalid")]
        lines = ["round | name | mode | auc_gain | years_helped | importance | max_corr (with) | status | formula"]
        for r in scored.head(limit).itertuples():
            lines.append(f"{r.round} | {r.name} | {r.mode} | {r.auc_gain:+.4f} | {r.years_helped}/4 | "
                         f"{r.importance:+.4f} | {r.max_corr:.2f} ({r.closest}) | {r.status}"
                         f"{' - ' + r.reason if r.reason else ''} | {r.formula}")
        if len(scored) > limit:
            lines.append(f"... {len(scored) - limit} weaker factors not shown")
        for r in failed.tail(10).itertuples():
            lines.append(f"{r.round} | {r.name} | INVALID: {r.reason} | {r.formula}")
        return "\n".join(lines)

    # ---------- one candidate ----------
    def _redundancy(self, values: pd.Series):
        ref = self._reference.copy()
        for k, v in self._kept_values.items():
            ref[k] = v
        x = values.reindex(ref.index).loc[self._eval_dates]
        corr = ref.loc[self._eval_dates].corrwith(x, method="spearman").abs()
        return float(corr.max()), str(corr.idxmax())

    def _evaluate(self, cand, round_no):
        rec = {"round": round_no, "name": cand.get("name", "?"), "mode": cand.get("mode", "?"),
               "parent": cand.get("parent", ""), "formula": cand.get("formula", ""),
               "hypothesis": cand.get("hypothesis", ""), "status": "invalid", "reason": "",
               "auc_gain": float("nan"), "years_helped": 0, "importance": float("nan"),
               "max_corr": float("nan"), "closest": ""}
        names = {b["name"] for b in self.board} | set(BASE)
        formulas = {b["formula"].replace(" ", "") for b in self.board} | {f.replace(" ", "") for f in BASE_FORMULAS.values()}
        if rec["name"] in names:
            rec["name"] = f"{rec['name']}_r{round_no}"
        if rec["formula"].replace(" ", "") in formulas:
            rec["reason"] = "duplicate formula"
            return rec, None
        try:
            validate(rec["formula"])
            values = evaluate(rec["formula"], self.prices)
        except (FormulaError, KeyError, ValueError) as exc:
            rec["reason"] = str(exc)[:200]
            return rec, None
        if values.loc["2022":].isna().any():
            rec["reason"] = "missing values in 2022-2025 (e.g. division by zero)"
            return rec, None
        rec["max_corr"], rec["closest"] = self._redundancy(values)
        per_year = self.scorer.score(rec["name"], values)
        if not per_year["valid"].all():
            rec["reason"] = "constant or missing in a training/validation quarter"
            return rec, per_year
        s = summarise(per_year).iloc[0]
        rec.update(auc_gain=float(s.auc_gain), years_helped=int(s.years_helped),
                   importance=float(s.importance), auc_gain_lr=float(s.auc_gain_lr),
                   auc_gain_rf=float(s.auc_gain_rf), bal_acc_gain=float(s.bal_acc_gain))
        reasons = []
        if rec["years_helped"] < MIN_YEARS:
            reasons.append(f"helped only {rec['years_helped']}/4 years")
        if rec["auc_gain"] < MIN_GAIN:
            reasons.append(f"mean auc_gain below {MIN_GAIN}")
        if rec["max_corr"] >= MAX_CORR:
            reasons.append(f"near-copy of {rec['closest']}")
        rec["status"] = "rejected" if reasons else "kept"
        rec["reason"] = "; ".join(reasons)
        if rec["status"] == "kept":
            self._kept_values[rec["name"]] = values
            self.kept.append({k: rec[k] for k in ("name", "formula", "hypothesis", "round", "auc_gain", "years_helped")})
        return rec, per_year

    # ---------- rounds ----------
    def run_round(self, round_no: int):
        prompt = build_prompt(round_no=round_no, n_variant=self.n_variant, n_new=self.n_new,
                              base_formulas=BASE_FORMULAS, stage1_top=self.stage1_top,
                              market=self.market, scoreboard=self.scoreboard_text(), kept=self.kept)
        tag = f"round_{round_no:02d}"
        (self.out_dir / f"{tag}_prompt.txt").write_text(prompt)
        t0 = time.time()
        response = self.proposer(prompt)
        (self.out_dir / f"{tag}_response.json").write_text(dump(response))
        per_years = []
        for cand in response.get("factors", [])[: self.n_variant + self.n_new]:
            rec, per_year = self._evaluate(cand, round_no)
            self.board.append(rec)
            if per_year is not None:
                per_years.append(per_year.assign(round=round_no))
            print(f"  [{rec['status']:8}] {rec['name']}: gain {rec['auc_gain']:+.4f}, "
                  f"{rec['years_helped']}/4 yrs, corr {rec['max_corr']:.2f}  {rec['reason']}", flush=True)
        if per_years:
            path = self.out_dir / "scores_by_year.csv"
            pd.concat(per_years).to_csv(path, mode="a", header=not path.exists(), index=False)
        self.save()
        return time.time() - t0

    def save(self):
        pd.DataFrame(self.board).to_csv(self.out_dir / "scoreboard.csv", index=False)
        (self.out_dir / "kept_factors.json").write_text(json.dumps(self.kept, indent=2) + "\n")
