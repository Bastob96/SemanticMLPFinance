import numpy as np
import pandas as pd
import pytest

from src.features import build_stage1_features
from src.stage2.context import BASE_FORMULAS
from src.stage2.formula import FormulaError, evaluate, validate, wide_prices

DATA = "data/processed/aligned_daily_ohlcv_5y.csv"


@pytest.fixture(scope="module")
def raw():
    return pd.read_csv(DATA)


def test_base_formulas_reproduce_stage1_features(raw):
    """The formula language computes the 22 Stage 1 factors exactly."""
    prices = wide_prices(raw)
    feats = build_stage1_features(raw).set_index("Date")
    feats.index = pd.to_datetime(feats.index)
    for name, formula in BASE_FORMULAS.items():
        ours = evaluate(formula, prices).reindex(feats.index)
        assert np.allclose(ours, feats[name], rtol=1e-9, atol=1e-12), name


def test_no_lookahead(raw):
    """A value at day t is identical when every later day is deleted."""
    prices = wide_prices(raw)
    formulas = list(BASE_FORMULAS.values()) + [
        "zscore(AAPL_Volume, 20)",
        "safe_divide(subtract(AAPL_Close, AAPL_Open), subtract(AAPL_High, AAPL_Low))",
        "subtract(safe_divide(AAPL_Close, rolling_max(AAPL_High, 60)), 1)",
        "rolling_sum(multiply(sign(diff(MSFT_Close, 1)), log(NVDA_Volume)), 10)",
        "lag(pct_change(QQQ_Close, 3), 2)",
    ]
    full = {f: evaluate(f, prices) for f in formulas}
    for t in prices.index[[150, 400, 700, 1000]]:
        cut = prices.loc[:t]
        for f in formulas:
            assert np.isclose(evaluate(f, cut).iloc[-1], full[f].loc[t], equal_nan=True), (f, t)


@pytest.mark.parametrize("bad", [
    "AAPL_Close + QQQ_Close",
    "lag(AAPL_Close, -1)",
    "shift(AAPL_Close, -1)",
    "rolling_mean(AAPL_Close, 7)",
    "pct_change(TSLA_Close, 1)",
    "rolling_std(AAPL_Close, 1)",
    "add(1, 2)",
    "__import__('os')",
    "pct_change(AAPL_Close, n=5)",
])
def test_rejects_bad_formulas(bad):
    with pytest.raises(FormulaError):
        validate(bad)
