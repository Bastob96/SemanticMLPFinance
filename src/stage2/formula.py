"""A small formula language for LLM-proposed factors, with a safe validator and executor.

The LLM writes a factor as nested function calls, for example
    subtract(pct_change(AAPL_Close, 5), pct_change(QQQ_Close, 5))

Every operator only looks backwards in time (rolling windows end at day t,
lags are positive), so any formula that passes ``validate`` can only use
information available at the close of day t.
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd

TICKERS = ("AAPL", "QQQ", "MSFT", "NVDA")
FIELDS = ("Open", "High", "Low", "Close", "Volume")
INPUTS = tuple(f"{t}_{f}" for t in TICKERS for f in FIELDS)
WINDOWS = (1, 2, 3, 5, 10, 20, 60)
MAX_DEPTH = 6

# name -> (number of series arguments, takes a window argument, description shown to the LLM)
OPERATORS = {
    "pct_change":   (1, True,  "x(t) / x(t-n) - 1"),
    "lag":          (1, True,  "x(t-n)"),
    "diff":         (1, True,  "x(t) - x(t-n)"),
    "rolling_mean": (1, True,  "mean of the last n values, ending at t"),
    "rolling_std":  (1, True,  "sample std (ddof=1) of the last n values, ending at t (n >= 2)"),
    "rolling_min":  (1, True,  "min of the last n values, ending at t"),
    "rolling_max":  (1, True,  "max of the last n values, ending at t"),
    "rolling_sum":  (1, True,  "sum of the last n values, ending at t"),
    "zscore":       (1, True,  "(x(t) - rolling_mean(x, n)) / rolling_std(x, n) (n >= 2)"),
    "add":          (2, False, "a + b"),
    "subtract":     (2, False, "a - b"),
    "multiply":     (2, False, "a * b"),
    "safe_divide":  (2, False, "a / b, missing when |b| < 1e-12"),
    "abs":          (1, False, "|x|"),
    "sign":         (1, False, "-1, 0 or +1"),
    "log":          (1, False, "natural log, missing when x <= 0"),
}


class FormulaError(ValueError):
    pass


def _parse(formula: str) -> ast.expr:
    try:
        return ast.parse(formula.strip(), mode="eval").body
    except SyntaxError as exc:
        raise FormulaError(f"not valid function syntax: {exc.msg}") from None


def validate(formula: str) -> dict:
    """Check the formula and return {'inputs': [...], 'max_window': n, 'depth': d}."""
    inputs, windows = set(), []

    def walk(node, depth):
        if depth > MAX_DEPTH:
            raise FormulaError(f"nested deeper than {MAX_DEPTH}")
        if isinstance(node, ast.Name):
            if node.id not in INPUTS:
                raise FormulaError(f"unknown input {node.id!r}")
            inputs.add(node.id)
            return "series"
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return "number"
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            raise FormulaError("negative constants are not allowed; use subtract")
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise FormulaError("only function calls, input names and numbers are allowed")
        name = node.func.id
        if name not in OPERATORS:
            raise FormulaError(f"unknown operator {name!r}")
        if node.keywords:
            raise FormulaError(f"{name}: use positional arguments only")
        n_series, windowed, _ = OPERATORS[name]
        expected = n_series + (1 if windowed else 0)
        if len(node.args) != expected:
            raise FormulaError(f"{name} takes {expected} arguments")
        kinds = [walk(arg, depth + 1) for arg in node.args[:n_series]]
        if name in ("add", "subtract", "multiply", "safe_divide"):
            if "series" not in kinds:
                raise FormulaError(f"{name}: at least one argument must be a series")
        elif kinds[0] != "series":
            raise FormulaError(f"{name}: first argument must be a series")
        if windowed:
            w = node.args[-1]
            if not (isinstance(w, ast.Constant) and type(w.value) is int and w.value in WINDOWS):
                raise FormulaError(f"{name}: window must be one of {WINDOWS}")
            if name in ("rolling_std", "zscore") and w.value < 2:
                raise FormulaError(f"{name}: window must be at least 2")
            windows.append(w.value)
        return "series"

    if walk(_parse(formula), 1) != "series":
        raise FormulaError("formula must produce a series")
    if not inputs:
        raise FormulaError("formula uses no market data")
    return {"inputs": sorted(inputs), "max_window": max(windows, default=0)}


def evaluate(formula: str, prices: pd.DataFrame) -> pd.Series:
    """Compute the factor. ``prices`` is wide: one row per date, columns like AAPL_Close."""
    validate(formula)

    def ev(node):
        if isinstance(node, ast.Name):
            return prices[node.id].astype(float)
        if isinstance(node, ast.Constant):
            return float(node.value)
        name = node.func.id
        n_series, windowed, _ = OPERATORS[name]
        args = [ev(a) for a in node.args[:n_series]]
        n = node.args[-1].value if windowed else None
        x = args[0]
        if name == "pct_change":
            return x / x.shift(n) - 1
        if name == "lag":
            return x.shift(n)
        if name == "diff":
            return x - x.shift(n)
        if name == "rolling_mean":
            return x.rolling(n, min_periods=n).mean()
        if name == "rolling_std":
            return x.rolling(n, min_periods=n).std(ddof=1)
        if name == "rolling_min":
            return x.rolling(n, min_periods=n).min()
        if name == "rolling_max":
            return x.rolling(n, min_periods=n).max()
        if name == "rolling_sum":
            return x.rolling(n, min_periods=n).sum()
        if name == "zscore":
            sd = x.rolling(n, min_periods=n).std(ddof=1)
            return (x - x.rolling(n, min_periods=n).mean()) / sd.where(sd.abs() >= 1e-12)
        a, b = (args + [None])[:2]
        if name == "add":
            return a + b
        if name == "subtract":
            return a - b
        if name == "multiply":
            return a * b
        if name == "safe_divide":
            if isinstance(b, float):
                return a / b if abs(b) >= 1e-12 else a * np.nan
            return a / b.where(b.abs() >= 1e-12)
        if name == "abs":
            return x.abs()
        if name == "sign":
            return np.sign(x)
        if name == "log":
            return np.log(x.where(x > 0))
        raise FormulaError(name)

    out = ev(_parse(formula))
    if isinstance(out, float):
        raise FormulaError("formula must produce a series")
    return out.replace([np.inf, -np.inf], np.nan)


def wide_prices(raw: pd.DataFrame) -> pd.DataFrame:
    """Long Ticker/Date/OHLCV table -> wide table with AAPL_Close style columns."""
    raw = raw.copy()
    raw["Date"] = pd.to_datetime(raw["Date"])
    wide = raw.pivot(index="Date", columns="Ticker", values=list(FIELDS))
    wide.columns = [f"{ticker}_{field}" for field, ticker in wide.columns]
    return wide.sort_index()


def operator_guide() -> str:
    lines = [f"- {name}({'x, n' if w else ', '.join('ab'[:k]) if k == 2 else 'x'}): {desc}"
             for name, (k, w, desc) in OPERATORS.items()]
    return "\n".join(lines)
