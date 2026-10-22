"""Stage 0 baseline, the automated LLM feedback rounds, final selection and the single final test.

Retention policy (fixed before any run, identical in every round):
  1. Candidates must pass factor_validator (validity, leakage, redundancy).
  2. For each valid candidate the possible moves against the current set S are
     ADD (if fewer than MAX_RETAINED_LLM_FACTORS LLM factors are retained) and
     REPLACE r for every retained LLM factor r. The best move is the one with the
     lowest validation MAE (ties: fewer features).
  3. Candidates are tried in order of their best-move MAE against the
     round-start set. Each candidate's moves are re-evaluated against the
     current S, and the best move is accepted only if validation MAE falls by
     at least MIN_RELATIVE_MAE_IMPROVEMENT and validation RMSE does not rise.
  4. At most MAX_ACCEPTED_PER_ROUND changes (additions or replacements) per round.
  5. Baseline factors are never removed. A replaced factor is retired and cannot return.
Final selection: lowest validation MAE among end-of-round sets that contain at
least one LLM factor (ties: fewer LLM factors, then earlier round). If none
exists, the selected model is the baseline and the result is reported as negative.
"""

import logging

import numpy as np
import pandas as pd

import config
from baseline_factors import BASELINE_FACTOR_DEFINITIONS, BASELINE_FACTOR_NAMES, compute_baseline_factors
from data_loader import load_market_data, make_splits
from evaluator import (FeatureSetEvaluator, compute_permutation_importance, naive_predictions,
                       price_metrics, regression_metrics, relative_improvement)
from factor_engine import LANGUAGE_REFERENCE, compute_expression
from factor_validator import validate_candidate
from feedback_builder import (assert_no_test_dates, build_data_context, build_round_feedback, observations,
                              round_floats, tried_expressions)
from llm_generator import SYSTEM_PROMPT, FatalLLMError, build_user_message
from model_trainer import fit_model, predict, returns_to_prices

log = logging.getLogger("capstone.loop")

PHASE1_REFERENCE = {"validation_mae": 0.011370, "validation_rmse": 0.013050}


class ExperimentRunner:
    def __init__(self, settings, generator, run_logger):
        self.settings = settings
        self.generator = generator
        self.out = run_logger
        self.retained = []          # retained LLM factor specs (dicts)
        self.retired = []
        self.round_records = []
        self.feedback_history = []
        self.set_history = []
        self.usage_rows = []

    # ================================================================ stage 0
    def prepare(self):
        data = load_market_data(self.settings.data_path)
        baseline_all = compute_baseline_factors(data.wide)
        splits = make_splits(baseline_all, data)
        dev_wide = data.wide.loc[:splits.validation.max()]   # the loop never sees test rows
        dev_baseline = compute_baseline_factors(dev_wide)
        if not np.allclose(dev_baseline.loc[splits.development].to_numpy(),
                           baseline_all.loc[splits.development].to_numpy(), equal_nan=True):
            raise AssertionError("Baseline factors depend on data after the development period")
        self.data, self.splits, self.dev_wide = data, splits, dev_wide
        self.target_dev = data.target.loc[splits.development]
        self.evaluator = FeatureSetEvaluator(dev_baseline.loc[splits.development].copy(), self.target_dev,
                                             splits.train, splits.validation)
        self.out.save_json("data_manifest.json", data.manifest)
        self.out.save_json("split_boundaries.json", splits.boundaries(data.target_date))
        self.out.save_json("baseline/factor_definitions.json", BASELINE_FACTOR_DEFINITIONS)
        self.out.save_text("factor_language.txt", LANGUAGE_REFERENCE)
        log.info("Splits: train %s..%s (%d), validation %s..%s (%d), test %s..%s (%d)",
                 splits.train.min().date(), splits.train.max().date(), len(splits.train),
                 splits.validation.min().date(), splits.validation.max().date(), len(splits.validation),
                 splits.test.min().date(), splits.test.max().date(), len(splits.test))

    def stage0(self):
        result = self.evaluator.evaluate(BASELINE_FACTOR_NAMES)
        y_val = self.target_dev.loc[self.splits.validation]
        self.baseline_metrics = result["metrics"]
        self.naive_val_metrics = regression_metrics(y_val, naive_predictions(len(y_val)))
        importance = compute_permutation_importance(result["model"], self.evaluator.features, self.target_dev,
                                                    self.splits.validation, BASELINE_FACTOR_NAMES)
        self.baseline_importance = importance
        reproduction = {
            "phase1_reference": PHASE1_REFERENCE,
            "reproduced_validation_mae": self.baseline_metrics["mae"],
            "reproduced_validation_rmse": self.baseline_metrics["rmse"],
            "matches_phase1": bool(abs(self.baseline_metrics["mae"] - PHASE1_REFERENCE["validation_mae"]) < 5e-7
                                   and abs(self.baseline_metrics["rmse"] - PHASE1_REFERENCE["validation_rmse"]) < 5e-7),
        }
        self.out.save_json("baseline/metrics.json", {
            "validation": self.baseline_metrics, "naive_zero_return_validation": self.naive_val_metrics,
            "phase1_reproduction": reproduction,
            "model": {"imputer_strategy": config.IMPUTER_STRATEGY, "hgbr_params": config.HGBR_PARAMS},
        })
        self.out.save_csv("baseline/permutation_importance_validation.csv", importance)
        self.out.save_csv("baseline/validation_predictions.csv", pd.DataFrame({
            "date": self.splits.validation, "actual_log_return": y_val.to_numpy(),
            "baseline_pred": result["predictions"]}))
        self.set_history.append({"round": 0, "llm_factors": [], "columns": list(BASELINE_FACTOR_NAMES),
                                 "metrics": self.baseline_metrics, "round_evaluated_metrics": None})
        log.info("Stage 0 baseline validation MAE=%.6f RMSE=%.6f (Phase 1: %.6f / %.6f) | naive MAE=%.6f",
                 self.baseline_metrics["mae"], self.baseline_metrics["rmse"],
                 PHASE1_REFERENCE["validation_mae"], PHASE1_REFERENCE["validation_rmse"],
                 self.naive_val_metrics["mae"])
        self.context = build_data_context(self.dev_wide, self.target_dev, self.splits, self.baseline_metrics,
                                          self.naive_val_metrics,
                                          importance[["factor", "importance_mean"]].to_dict("records"))
        self.out.save_json("llm_data_context.json", self.context)
        return reproduction

    # ================================================================ rounds
    def current_columns(self):
        return list(BASELINE_FACTOR_NAMES) + [f["name"] for f in self.retained]

    def build_payload(self, round_id):
        current = self.evaluator.evaluate(self.current_columns())
        return round_floats({
            "round": round_id,
            "candidates_requested": self.settings.candidates,
            "data_context": self.context,
            "current_state": {
                "retained_llm_factors": [{k: f[k] for k in ("name", "expression", "category", "round_added")}
                                         for f in self.retained],
                "retired_llm_factors": self.retired,
                "current_set_validation_metrics": current["metrics"],
                "baseline_validation_metrics": self.baseline_metrics,
            },
            "previous_rounds_feedback": self.feedback_history,
            "already_tried_expressions": tried_expressions(self.round_records),
        })

    def run_rounds(self):
        seen_names = set(BASELINE_FACTOR_NAMES)
        seen_canonicals = {}
        for round_id in range(1, self.settings.rounds + 1):
            log.info("===== Round %d of %d =====", round_id, self.settings.rounds)
            payload = self.build_payload(round_id)
            assert_no_test_dates(payload, self.splits.test)
            self.out.save_json(f"rounds/round_{round_id}/prompt.json", {
                "mode": self.generator.mode, "system": SYSTEM_PROMPT, "user": build_user_message(payload),
                "payload": payload})
            try:
                generation = self.generator.generate(round_id, payload)
            except FatalLLMError:
                self.out.save_json(f"rounds/round_{round_id}/llm_status.json", {"status": "fatal_error"})
                raise
            self.save_generation(round_id, generation)
            record = self.process_round(round_id, generation, seen_names, seen_canonicals)
            self.round_records.append(record)
            feedback = build_round_feedback(record)
            self.feedback_history.append(feedback)
            self.out.save_json(f"rounds/round_{round_id}/feedback.json", feedback)
            log.info("Round %d: %d received, %d valid, changes=%s, current MAE=%.6f",
                     round_id, record["candidates_received"],
                     sum(c["validation"]["status"] == "valid" for c in record["candidates"]),
                     [f"{c['type']}:{c['candidate']}" for c in record["changes"]] or "none",
                     record["metrics_after"]["mae"])

    def save_generation(self, round_id, generation):
        folder = f"rounds/round_{round_id}"
        self.out.save_json(f"{folder}/llm_response.json", {
            "status": generation.status, "error": generation.error,
            "feedback_analysis": generation.feedback_analysis, "candidates": generation.candidates,
            "attempts": [{k: v for k, v in a.items() if k not in ("request", "user_message")}
                         for a in generation.attempts]})
        for attempt in generation.attempts:
            usage = attempt.get("usage") or {}
            self.usage_rows.append({
                "round": round_id, "attempt": attempt.get("attempt"), "from_cache": attempt.get("from_cache"),
                "model_requested": attempt.get("model_requested"), "model_served": attempt.get("model_served"),
                "stop_reason": attempt.get("stop_reason"), "error": attempt.get("error"),
                "seconds": attempt.get("seconds"),
                "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
                "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
                "estimated_cost_usd": attempt.get("estimated_cost_usd")})
        self.out.save_csv("llm_usage.csv", pd.DataFrame(self.usage_rows))

    def best_move(self, name, columns, llm_names):
        """Evaluate ADD / REPLACE moves for one candidate against ``columns``."""
        moves = []
        if len(llm_names) < config.MAX_RETAINED_LLM_FACTORS:
            moves.append({"type": "add", "replaced": None, "columns": columns + [name]})
        if config.ALLOW_REPLACEMENT:
            for old in llm_names:
                moves.append({"type": "replace", "replaced": old,
                              "columns": [c for c in columns if c != old] + [name]})
        for move in moves:
            move["metrics"] = self.evaluator.evaluate(move["columns"])["metrics"]
        if not moves:
            return None, []
        best = min(moves, key=lambda m: (m["metrics"]["mae"], len(m["columns"])))
        return best, moves

    def process_round(self, round_id, generation, seen_names, seen_canonicals):
        folder = f"rounds/round_{round_id}"
        n_requested = self.settings.candidates
        raw = list(generation.candidates)
        extra = raw[n_requested:]
        raw = raw[:n_requested]
        retained_before = [dict(f) for f in self.retained]
        start_columns = self.current_columns()
        start_metrics = self.evaluator.evaluate(start_columns)["metrics"]

        # ---- validation
        candidates, valid_names = [], []
        for spec in raw:
            reference = self.evaluator.features[start_columns + valid_names]
            result = validate_candidate(spec, self.dev_wide, self.splits.development, self.splits.train,
                                        reference, seen_names, seen_canonicals)
            if isinstance(spec, dict) and isinstance(spec.get("name"), str):
                seen_names.add(spec["name"])
            if result.canonical and result.canonical not in seen_canonicals:
                seen_canonicals[result.canonical] = result.name
            item = {"name": result.name, "spec": spec if isinstance(spec, dict) else {"raw": spec},
                    "expression": spec.get("expression") if isinstance(spec, dict) else None,
                    "category": spec.get("category") if isinstance(spec, dict) else None,
                    "relation_to_previous": spec.get("relation_to_previous") if isinstance(spec, dict) else None,
                    "validation": result.to_record(), "decision": "invalid",
                    "decision_reason": result.message}
            if result.is_valid:
                self.evaluator.add_feature(result.name, result.values)
                valid_names.append(result.name)
                y_val = self.target_dev.loc[self.splits.validation]
                item["target_spearman_validation"] = float(
                    result.values.reindex(self.splits.validation).corr(y_val, method="spearman"))
                alone = self.evaluator.evaluate(list(BASELINE_FACTOR_NAMES) + [result.name])["metrics"]
                item["mae_baseline_plus"] = alone["mae"]
                item["rmse_baseline_plus"] = alone["rmse"]
                item["rel_improvement_vs_baseline"] = relative_improvement(self.baseline_metrics["mae"], alone["mae"])
            candidates.append(item)

        # ---- screening against the round-start set
        valid_items = [c for c in candidates if c["validation"]["status"] == "valid"]
        start_llm = [f["name"] for f in self.retained]
        for item in valid_items:
            best, _ = self.best_move(item["name"], start_columns, start_llm)
            item["screening_mae"] = best["metrics"]["mae"] if best else np.inf

        # ---- greedy acceptance
        changes, accepted = [], 0
        columns = list(start_columns)
        current_metrics = start_metrics
        move_rows = []
        for item in sorted(valid_items, key=lambda c: c["screening_mae"]):
            if accepted >= config.MAX_ACCEPTED_PER_ROUND:
                item["decision"] = "not_considered_round_limit"
                item["decision_reason"] = f"{config.MAX_ACCEPTED_PER_ROUND} changes already accepted this round"
                continue
            llm_names = [f["name"] for f in self.retained]
            best, moves = self.best_move(item["name"], columns, llm_names)
            for move in moves:
                move_rows.append({"candidate": item["name"], "move": move["type"], "replaced": move["replaced"],
                                  "n_features": len(move["columns"]), **{f"val_{k}": v for k, v in move["metrics"].items()},
                                  "reference_mae": current_metrics["mae"], "reference_rmse": current_metrics["rmse"]})
            rel = relative_improvement(current_metrics["mae"], best["metrics"]["mae"])
            item.update({"best_move": best["type"] + (f" (replace {best['replaced']})" if best["replaced"] else ""),
                         "best_move_mae": best["metrics"]["mae"], "best_move_rmse": best["metrics"]["rmse"],
                         "best_move_rel_improvement": rel})
            mae_ok = rel >= config.MIN_RELATIVE_MAE_IMPROVEMENT
            rmse_ok = (not config.REQUIRE_RMSE_NOT_WORSE) or best["metrics"]["rmse"] <= current_metrics["rmse"]
            if mae_ok and rmse_ok:
                accepted += 1
                spec = item["spec"]
                new_factor = {"name": item["name"], "expression": spec["expression"],
                              "category": spec.get("category"), "interpretation": spec.get("interpretation"),
                              "math_definition": spec.get("math_definition"), "inputs": spec.get("inputs"),
                              "windows": spec.get("windows"), "round_added": round_id}
                if best["type"] == "replace":
                    old = next(f for f in self.retained if f["name"] == best["replaced"])
                    self.retained.remove(old)
                    self.retired.append({"name": old["name"], "expression": old["expression"],
                                         "round_added": old["round_added"], "round_removed": round_id,
                                         "replaced_by": item["name"]})
                self.retained.append(new_factor)
                changes.append({"type": best["type"], "candidate": item["name"], "replaced": best["replaced"],
                                "mae_before": current_metrics["mae"], "mae_after": best["metrics"]["mae"],
                                "rmse_before": current_metrics["rmse"], "rmse_after": best["metrics"]["rmse"],
                                "relative_mae_improvement": rel})
                columns = best["columns"]
                current_metrics = best["metrics"]
                item["decision"] = f"accepted_{best['type']}"
                item["decision_reason"] = (f"{best['type']} lowered validation MAE by {rel:.2%} "
                                           f"(>= {config.MIN_RELATIVE_MAE_IMPROVEMENT:.1%}) without raising RMSE")
            else:
                item["decision"] = "rejected_no_improvement" if not mae_ok else "rejected_rmse_guard"
                item["decision_reason"] = (
                    f"best move ({item['best_move']}) changed MAE by {rel:+.2%}; needs >= "
                    f"{config.MIN_RELATIVE_MAE_IMPROVEMENT:.1%}" if not mae_ok else
                    f"MAE improved {rel:.2%} but RMSE rose from {current_metrics['rmse']:.6f} "
                    f"to {best['metrics']['rmse']:.6f}")

        # ---- diagnostics on the end-of-round set (validation and training only)
        end = self.evaluator.evaluate(columns)
        assert end["columns"] == self.current_columns(), "retained list and evaluated columns diverged"
        round_evaluated = (self.evaluator.evaluate(start_columns + valid_names)["metrics"]
                           if valid_names else None)
        importance = compute_permutation_importance(end["model"], self.evaluator.features, self.target_dev,
                                                    self.splits.validation, columns)
        corr = self.evaluator.features.loc[self.splits.train, list(dict.fromkeys(columns + valid_names))].corr(
            method="spearman")
        llm_like = set(valid_names) | {f["name"] for f in self.retained}
        pairs = [{"factor_1": a, "factor_2": b, "spearman_train": float(corr.loc[a, b])}
                 for i, a in enumerate(corr.columns) for b in corr.columns[i + 1:]
                 if (a in llm_like or b in llm_like) and abs(corr.loc[a, b]) >= 0.8]

        record = {
            "round": round_id, "llm_status": generation.status, "llm_error": generation.error,
            "feedback_analysis": generation.feedback_analysis,
            "candidates_requested": n_requested, "candidates_received": len(generation.candidates),
            "extra_candidates_ignored": [c.get("name") if isinstance(c, dict) else str(c) for c in extra],
            "candidates": candidates, "changes": changes,
            "retained_before": retained_before, "retained_after": [dict(f) for f in self.retained],
            "baseline_metrics": self.baseline_metrics, "metrics_before": start_metrics,
            "metrics_after": end["metrics"], "round_evaluated_metrics": round_evaluated,
            "importance_after": importance.to_dict("records"), "high_correlation_pairs": pairs,
        }
        record["observations"] = observations(record)
        self.set_history.append({"round": round_id, "llm_factors": [f["name"] for f in self.retained],
                                 "columns": list(columns), "metrics": end["metrics"],
                                 "round_evaluated_metrics": round_evaluated})

        # ---- save
        self.out.save_json(f"{folder}/candidates.json", [c["spec"] for c in candidates])
        self.out.save_csv(f"{folder}/validation_report.csv", pd.DataFrame([c["validation"] for c in candidates]))
        self.out.save_csv(f"{folder}/candidate_evaluations.csv", pd.DataFrame([{
            k: c.get(k) for k in ("name", "expression", "decision", "decision_reason", "mae_baseline_plus",
                                  "rmse_baseline_plus", "rel_improvement_vs_baseline", "best_move", "best_move_mae",
                                  "best_move_rmse", "best_move_rel_improvement", "target_spearman_validation")}
            for c in candidates]))
        self.out.save_csv(f"{folder}/move_evaluations.csv", pd.DataFrame(move_rows))
        self.out.save_json(f"{folder}/retention_decision.json", {
            "policy": __doc__, "changes": changes, "retained_before": retained_before,
            "retained_after": self.retained, "retired": self.retired,
            "metrics_before": start_metrics, "metrics_after": end["metrics"],
            "round_evaluated_model": {"columns": start_columns + valid_names, "metrics": round_evaluated}})
        self.out.save_csv(f"{folder}/diagnostics/permutation_importance_validation.csv", importance)
        self.out.save_csv(f"{folder}/diagnostics/factor_correlations_train.csv", corr, index=True)
        return record

    # ================================================================ selection + test
    def select_final(self):
        eligible = [s for s in self.set_history[1:] if s["llm_factors"]]
        if eligible:
            chosen = min(eligible, key=lambda s: (s["metrics"]["mae"], len(s["llm_factors"]), s["round"]))
            negative = False
        else:
            chosen, negative = self.set_history[0], True
        names = set(chosen["llm_factors"])
        all_specs = {}   # specs come from the round records, so retired factors are covered too
        for record in self.round_records:
            for c in record["candidates"]:
                if c["name"] in names and isinstance(c["spec"], dict):
                    all_specs[c["name"]] = {"name": c["name"], "expression": c["spec"]["expression"],
                                            "category": c["spec"].get("category"),
                                            "interpretation": c["spec"].get("interpretation"),
                                            "math_definition": c["spec"].get("math_definition"),
                                            "round_proposed": record["round"]}
        selection = {
            "rule": "Lowest validation MAE among end-of-round sets containing >= 1 LLM factor; ties -> fewer LLM "
                    "factors, then earlier round. If none, the baseline is selected (negative result).",
            "selected_round": chosen["round"], "negative_result": negative,
            "selected_llm_factors": [all_specs[n] for n in chosen["llm_factors"]],
            "selected_columns": chosen["columns"], "selected_validation_metrics": chosen["metrics"],
            "baseline_validation_metrics": self.baseline_metrics,
            "all_end_of_round_sets": [{"round": s["round"], "llm_factors": s["llm_factors"],
                                       "validation_mae": s["metrics"]["mae"], "validation_rmse": s["metrics"]["rmse"]}
                                      for s in self.set_history],
        }
        self.selection = selection
        self.out.save_json("final/selected_model.json", selection)
        self.out.save_json("final/frozen_factor_definitions.json", {
            "baseline_factors": BASELINE_FACTOR_DEFINITIONS, "llm_factors": selection["selected_llm_factors"],
            "frozen_before_test": True})
        log.info("Selected set from round %d: %s (validation MAE %.6f)%s", chosen["round"],
                 chosen["llm_factors"] or "baseline only", chosen["metrics"]["mae"],
                 " -- NEGATIVE RESULT: no LLM factor met the acceptance rule" if negative else "")
        return selection

    def final_test(self):
        data, splits = self.data, self.splits
        features = compute_baseline_factors(data.wide)
        for spec in self.selection["selected_llm_factors"]:
            features[spec["name"]] = compute_expression(data.wide, spec["expression"]).values
        y = data.target
        base_cols = list(BASELINE_FACTOR_NAMES)
        llm_cols = self.selection["selected_columns"]
        y_test = y.loc[splits.test]
        close_t = data.wide.loc[splits.test, "aapl_close"]

        results, predictions = {}, {}
        for label, fit_index in (("refit_train_validation", splits.development), ("train_only", splits.train)):
            base_model = fit_model(features, y, fit_index, base_cols)
            llm_model = fit_model(features, y, fit_index, llm_cols)
            preds = {"baseline": predict(base_model, features, splits.test, base_cols),
                     "llm_enhanced": predict(llm_model, features, splits.test, llm_cols),
                     "naive": naive_predictions(len(y_test))}
            results[label] = {name: {**regression_metrics(y_test, p), **price_metrics(close_t, y_test, p)}
                              for name, p in preds.items()}
            predictions[label] = preds

        primary = predictions["refit_train_validation"]
        frame = pd.DataFrame({
            "date": splits.test, "target_date": data.target_date.loc[splits.test].to_numpy(),
            "close_t": close_t.to_numpy(), "actual_log_return": y_test.to_numpy(),
            "baseline_pred": primary["baseline"], "llm_enhanced_pred": primary["llm_enhanced"],
            "naive_pred": primary["naive"]})
        frame["actual_next_close"] = returns_to_prices(frame["close_t"], frame["actual_log_return"])
        frame["baseline_next_close"] = returns_to_prices(frame["close_t"], frame["baseline_pred"])
        frame["llm_enhanced_next_close"] = returns_to_prices(frame["close_t"], frame["llm_enhanced_pred"])
        frame["naive_next_close"] = frame["close_t"]
        test_summary = {
            "primary_protocol": "Fit on train + validation rows, evaluate once on the test rows",
            "secondary_protocol": "Fit on train rows only (Phase 1 style), same test rows",
            "fit_rows": {"refit_train_validation": int(len(splits.development)), "train_only": int(len(splits.train))},
            "test_period": {"first_feature_date": str(splits.test.min().date()),
                            "last_feature_date": str(splits.test.max().date()), "rows": int(len(splits.test))},
            "selected_round": self.selection["selected_round"], "negative_result": self.selection["negative_result"],
            "metrics": results,
        }
        self.out.save_json("final/test_metrics.json", test_summary)
        self.out.save_csv("final/test_predictions.csv", frame)
        m = results["refit_train_validation"]
        log.info("FINAL TEST (refit train+val): MAE naive=%.6f baseline=%.6f llm=%.6f | RMSE naive=%.6f "
                 "baseline=%.6f llm=%.6f", m["naive"]["mae"], m["baseline"]["mae"], m["llm_enhanced"]["mae"],
                 m["naive"]["rmse"], m["baseline"]["rmse"], m["llm_enhanced"]["rmse"])
        self.test_summary, self.test_predictions = test_summary, frame
        return test_summary

    def figure_inputs(self):
        """Validation importance and training correlations for the selected and baseline sets."""
        selected = self.selection["selected_columns"]
        sel_model = self.evaluator.evaluate(selected)["model"]
        importance = compute_permutation_importance(sel_model, self.evaluator.features, self.target_dev,
                                                    self.splits.validation, selected)
        llm_names = {f["name"] for f in self.selection["selected_llm_factors"]}
        importance["group"] = ["LLM-generated" if f in llm_names else "Baseline" for f in importance["factor"]]
        corr = self.evaluator.features.loc[self.splits.train, selected].corr(method="spearman")
        candidate_rows = [{"round": r["round"], "name": c["name"], "decision": c["decision"],
                           "rel_improvement_vs_baseline": c.get("rel_improvement_vs_baseline"),
                           "best_move_rel_improvement": c.get("best_move_rel_improvement"),
                           "code": c["validation"]["code"]} for r in self.round_records for c in r["candidates"]]
        return importance, corr, pd.DataFrame(candidate_rows)
