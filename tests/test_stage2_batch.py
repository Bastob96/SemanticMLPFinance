"""Batch mode (Liu, 12 Nov) and the pooled Q4 test: fast checks, no model fitting except one tiny case."""

import numpy as np
import pandas as pd
import pytest

from src.stage2.formula import FormulaError
from src.stage2.loop_v2 import LoopV2, NoiseBatchProposer, TemplateBatchProposer
from src.stage2.pooled_test import factor_values, p_value, pooled_gain
from src.stage2.prompt_v2 import TOOL_BATCH, build_prompt_batch

RULES = {"min_years": 3, "team_gain": 0.01, "pair_gain": 0.01, "pair_synergy": 0.01, "max_corr": 0.9, "team_size": 6}


def test_batch_prompt_asks_for_meaning_and_count():
    text = build_prompt_batch(n=100, base_formulas={"ret_1": "pct_change(AAPL_Close, 1)"},
                              stage1_top="x", market="m", rules=RULES)
    assert text.startswith("BATCH OF 100")
    assert "Propose exactly 100 factors" in text and "5 minutes" in text
    assert "meaning" in TOOL_BATCH["input_schema"]["properties"]["factors"]["items"]["required"]


def test_template_batch_gives_n_different_formulas():
    out = TemplateBatchProposer(seed=1)("BATCH OF 50\n...")["factors"]
    assert len(out) == 50 and len({f["formula"] for f in out}) == 50


def test_noise_only_when_allowed():
    prices = pd.DataFrame({"AAPL_Close": np.linspace(100, 110, 30)},
                          index=pd.date_range("2022-01-03", periods=30, freq="B"))
    formula = NoiseBatchProposer(seed=2)("BATCH OF 3")["factors"][0]["formula"]
    loop = LoopV2.__new__(LoopV2)                     # only compute() is needed here
    loop.prices, loop.allow_noise = prices, True
    a, b = loop.compute(formula), factor_values(formula, prices)
    assert len(a) == 30 and np.allclose(a, b)         # the pooled test rebuilds the same random numbers
    loop.allow_noise = False
    with pytest.raises(FormulaError):
        loop.compute(formula)


def test_pooled_gain_and_p_value():
    rng = np.random.default_rng(0)
    days = pd.date_range("2022-10-03", periods=40, freq="B")
    rows = [pd.DataFrame({"Date": days, "model": m, "actual": rng.integers(0, 2, 40), "prob": rng.random(40)})
            for m in ("logistic_regression", "random_forest")]
    base = pd.concat(rows, ignore_index=True)
    g = pooled_gain(base, base.copy())
    assert all(abs(g[m]["auc_gain"]) < 1e-12 and g[m]["days"] == 40 for m in g)
    assert p_value(0.05, pd.Series([0.0, 0.01, 0.1])) == pytest.approx(2 / 4)
