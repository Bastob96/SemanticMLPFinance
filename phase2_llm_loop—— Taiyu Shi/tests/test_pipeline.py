"""Offline tests: data, baseline reproduction, factor language, validation, LLM client handling, loop."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import config
from baseline_factors import BASELINE_FACTOR_NAMES, compute_baseline_factors
from data_loader import load_market_data, make_splits
from evaluator import FeatureSetEvaluator
from factor_engine import FactorExpressionError, compute_expression
from factor_validator import validate_candidate


@pytest.fixture(scope="module")
def prepared():
    data = load_market_data(config.DATA_PATH)
    baseline = compute_baseline_factors(data.wide)
    splits = make_splits(baseline, data)
    dev_wide = data.wide.loc[:splits.validation.max()]
    return data, baseline, splits, dev_wide


def test_data_alignment(prepared):
    data, _, _, _ = prepared
    assert len(data.wide) == 250
    assert data.wide.index.is_monotonic_increasing and not data.wide.index.has_duplicates
    assert data.manifest["split_jump_check"]["suspicious_moves"] == []
    # target is the next-day log return
    close = data.wide["aapl_close"]
    assert data.target.iloc[0] == pytest.approx(np.log(close.iloc[1] / close.iloc[0]))
    assert np.isnan(data.target.iloc[-1])


def test_splits_match_phase1(prepared):
    _, _, splits, _ = prepared
    assert (len(splits.train), len(splits.validation), len(splits.test)) == (160, 34, 35)
    assert str(splits.train.min().date()) == "2025-02-03"
    assert str(splits.validation.min().date()) == "2025-09-23"
    assert str(splits.test.min().date()) == "2025-11-10"
    assert str(splits.test.max().date()) == "2025-12-30"


def test_baseline_reproduces_phase1(prepared):
    data, baseline, splits, _ = prepared
    evaluator = FeatureSetEvaluator(baseline.loc[splits.development].copy(), data.target.loc[splits.development],
                                    splits.train, splits.validation)
    metrics = evaluator.evaluate(BASELINE_FACTOR_NAMES)["metrics"]
    assert metrics["mae"] == pytest.approx(0.011370, abs=5e-7)
    assert metrics["rmse"] == pytest.approx(0.013050, abs=5e-7)
    assert metrics["directional_accuracy"] == pytest.approx(0.5)


def test_baseline_is_causal(prepared):
    data, baseline, splits, dev_wide = prepared
    truncated = compute_baseline_factors(dev_wide)
    np.testing.assert_allclose(truncated.loc[splits.development], baseline.loc[splits.development])


@pytest.mark.parametrize("expression, code", [
    ("lag(aapl_close, -1)", "FUTURE_REFERENCE"),
    ("aapl_close.shift(-1)", "DISALLOWED_FUNCTION"),
    ("__import__('os').system('dir')", "DISALLOWED_FUNCTION"),
    ("open('secrets.txt')", "DISALLOWED_FUNCTION"),
    ("eval('1+1')", "DISALLOWED_FUNCTION"),
    ("aapl_close ** 2", "DISALLOWED_SYNTAX"),
    ("aapl_close[0]", "DISALLOWED_SYNTAX"),
    ("rolling_mean(aapl_close, w=5)", "DISALLOWED_SYNTAX"),
    ("target_log_return", "FUTURE_REFERENCE"),
    ("aapl_vwap", "UNKNOWN_INPUT"),
    ("zscore(diff(aapl_close, 5), 20)", "LOOKBACK_EXCEEDED"),
    ("rolling_mean(aapl_close, 21)", "BAD_ARGUMENT"),
    ("rolling_mean(aapl_close, 5.5)", "BAD_ARGUMENT"),
    ("1 + 2", "NOT_A_SERIES"),
    ("aapl_close +", "SYNTAX_ERROR"),
    ("[x for x in aapl_close]", "DISALLOWED_SYNTAX"),
])
def test_expression_language_rejects(prepared, expression, code):
    _, _, _, dev_wide = prepared
    with pytest.raises(FactorExpressionError) as error:
        compute_expression(dev_wide, expression)
    assert error.value.code == code


def test_lookback_accounting(prepared):
    _, _, _, dev_wide = prepared
    assert compute_expression(dev_wide, "rolling_mean(aapl_ret1, 20)").lookback == 20
    assert compute_expression(dev_wide, "zscore(aapl_logvol, 20)").lookback == 19
    assert compute_expression(dev_wide, "rolling_corr(aapl_ret1, qqq_ret1, 20)").lookback == 20
    assert compute_expression(dev_wide, "diff(log(aapl_close), 5) - lag(qqq_ret1, 3)").lookback == 5


def test_expression_matches_baseline(prepared):
    _, baseline, _, dev_wide = prepared
    values = compute_expression(dev_wide, "diff(log(aapl_close), 5)").values
    np.testing.assert_allclose(values.dropna(), baseline["log_return_5d"].loc[dev_wide.index].dropna())


def _spec(name, expression, inputs):
    return {"name": name, "expression": expression, "inputs": inputs, "interpretation": "x",
            "math_definition": "x", "predictive_rationale": "x", "windows": []}


def _validate(prepared, spec, seen_names=None, seen_canonicals=None):
    _, baseline, splits, dev_wide = prepared
    return validate_candidate(spec, dev_wide, splits.development, splits.train,
                              baseline.loc[splits.development], seen_names or set(BASELINE_FACTOR_NAMES),
                              seen_canonicals or {})


def test_validator_accepts_and_rejects(prepared):
    ok = _validate(prepared, _spec("rel_mom_5", "diff(log(aapl_close), 5) - diff(log(qqq_close), 5)",
                                   ["aapl_close", "qqq_close"]))
    assert ok.is_valid and ok.lookback == 5 and ok.missing_fraction == 0
    assert _validate(prepared, _spec("dup5", "diff(log(aapl_close), 5)", ["aapl_close"])).code == "REDUNDANT"
    assert _validate(prepared, _spec("const1", "aapl_close / aapl_close", ["aapl_close"])).code == "NEAR_CONSTANT"
    assert _validate(prepared, _spec("log_return_1d", "qqq_ret1", ["qqq_ret1"])).code == "DUPLICATE_NAME"
    assert _validate(prepared, _spec("bad name!", "qqq_ret1", ["qqq_ret1"])).code == "INVALID_NAME"
    assert _validate(prepared, _spec("vwap_ratio", "aapl_close", ["aapl_vwap"])).code == "UNKNOWN_INPUT"
    assert _validate(prepared, {"name": "missing_fields"}).code == "MALFORMED_CANDIDATE"
    again = _validate(prepared, _spec("rel_mom_5b", "diff(log(aapl_close),5)-diff(log(qqq_close),5)",
                                      ["aapl_close", "qqq_close"]), seen_canonicals={ok.canonical: "rel_mom_5"})
    assert again.code == "DUPLICATE_EXPRESSION"


# ---------------------------------------------------------------- Claude client with a fake transport
class _FakeStream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.message


class _FakeMessage:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self, mode="json"):
        return self.payload


def _response(text, stop_reason="end_turn"):
    return {"model": "claude-opus-5-5", "stop_reason": stop_reason, "stop_details": None,
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 10000, "output_tokens": 2000, "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 0}}


def _fake_generator(tmp_path, responses):
    import anthropic
    from llm_generator import ClaudeFactorGenerator
    generator = object.__new__(ClaudeFactorGenerator)
    generator.anthropic = anthropic
    generator.settings = config.RunSettings(use_cache=True, use_fallbacks=True)
    generator.cache_dir = tmp_path
    generator.mode = "live"
    queue = list(responses)
    calls = []

    def stream(**params):
        calls.append(params)
        return _FakeStream(_FakeMessage(queue.pop(0)))

    generator.client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=stream)),
                                       messages=SimpleNamespace(stream=stream))
    return generator, calls


def test_claude_generator_parses_logs_and_caches(tmp_path):
    proposal = {"feedback_analysis": "a", "candidates": [_spec(f"f{i}x", "qqq_ret1", ["qqq_ret1"]) for i in range(5)]}
    generator, calls = _fake_generator(tmp_path, [_response(json.dumps(proposal))])
    payload = {"round": 1, "candidates_requested": 5}
    result = generator.generate(1, payload)
    assert result.status == "ok" and len(result.candidates) == 5
    assert result.attempts[0]["estimated_cost_usd"] == pytest.approx(0.08)  # 10k*$4 + 2k*$20 per MTok
    assert calls[0]["fallbacks"] == "default" and calls[0]["output_config"]["format"]["type"] == "json_schema"
    assert "api_key" not in json.dumps(calls[0]).lower()
    cached = generator.generate(1, payload)          # second call is served from the cache
    assert cached.attempts[0]["from_cache"] and len(calls) == 1


def test_claude_generator_bounded_repair(tmp_path):
    good = {"feedback_analysis": "a", "candidates": [_spec("abc", "qqq_ret1", ["qqq_ret1"])]}
    generator, calls = _fake_generator(tmp_path, [_response("{not json"), _response(json.dumps(good))])
    result = generator.generate(1, {"round": 1, "candidates_requested": 5})
    assert result.status == "partial" and len(calls) == 2
    generator, calls = _fake_generator(tmp_path / "b", [_response("", "refusal"), _response("", "max_tokens")])
    result = generator.generate(1, {"round": 1, "candidates_requested": 5})
    assert result.status == "failed" and len(calls) == 1 + config.LLM_MAX_REPAIR_ATTEMPTS


# ---------------------------------------------------------------- full loop with the mock generator
def test_loop_feedback_propagation_and_test_protection(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "figures")
    from iteration_loop import ExperimentRunner
    from llm_generator import MockFactorGenerator
    from run_logger import RunLogger

    settings = config.RunSettings(rounds=3, candidates=5, dry_run=True)
    out = RunLogger("test")
    try:
        runner = ExperimentRunner(settings, MockFactorGenerator(), out)
        runner.prepare()
        runner.stage0()
        runner.run_rounds()
        selection = runner.select_final()
        runner.final_test()
    finally:
        out.close()

    prompt2 = json.loads((out.run_dir / "rounds/round_2/prompt.json").read_text(encoding="utf-8"))
    feedback = prompt2["payload"]["previous_rounds_feedback"]
    assert len(feedback) == 1 and feedback[0]["round"] == 1
    assert {c["name"] for c in feedback[0]["candidates"]} >= {"aapl_qqq_rel_mom_5", "future_close_leak"}
    assert any(c.get("rejection_code") == "FUTURE_REFERENCE" for c in feedback[0]["candidates"])
    prompt3 = json.loads((out.run_dir / "rounds/round_3/prompt.json").read_text(encoding="utf-8"))
    assert len(prompt3["payload"]["previous_rounds_feedback"]) == 2

    # each round requested exactly the configured number of candidates
    for r in (1, 2, 3):
        p = json.loads((out.run_dir / f"rounds/round_{r}/prompt.json").read_text(encoding="utf-8"))
        assert p["payload"]["candidates_requested"] == 5
        for d in runner.splits.test:
            assert str(d.date()) not in p["user"]

    # policy invariants
    for record in runner.round_records:
        assert len(record["changes"]) <= config.MAX_ACCEPTED_PER_ROUND
        assert len(record["retained_after"]) <= config.MAX_RETAINED_LLM_FACTORS
        for change in record["changes"]:
            assert change["relative_mae_improvement"] >= config.MIN_RELATIVE_MAE_IMPROVEMENT
            assert change["rmse_after"] <= change["rmse_before"]
    assert any(c["type"] == "replace" for r in runner.round_records for c in r["changes"])
    assert runner.set_history[0]["columns"] == BASELINE_FACTOR_NAMES   # baseline unchanged
    best = min(s["metrics"]["mae"] for s in runner.set_history[1:] if s["llm_factors"])
    assert selection["selected_validation_metrics"]["mae"] == best
    assert (out.run_dir / "final/test_metrics.json").exists()
