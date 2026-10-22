"""Ordered quality checks for LLM-proposed factors.

The first failing check determines the rejection code. Rejections are recorded
with a short explanation and passed back to the LLM in the next round's feedback.
"""

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
from factor_engine import ALLOWED_INPUTS, FactorExpressionError, compute_expression

NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{2,59}$")
REQUIRED_TEXT_FIELDS = ("name", "expression", "interpretation", "math_definition", "predictive_rationale")

CHECK_DESCRIPTIONS = {
    "MALFORMED_CANDIDATE": "Required fields are missing or have the wrong type",
    "INVALID_NAME": "Name must start with a letter and use 3-60 letters, digits or underscores",
    "DUPLICATE_NAME": "Name already used by a baseline, retained or previously proposed factor",
    "UNKNOWN_INPUT": "Declared or referenced input column does not exist",
    "DUPLICATE_EXPRESSION": "Expression is identical to a previously proposed expression",
    "FUTURE_REFERENCE": "Expression uses or could use information after day t",
    "LOOKBACK_EXCEEDED": f"Expression needs more than {config.MAX_LOOKBACK} trading days of history",
    "TOO_MANY_MISSING": f"More than {config.MAX_MISSING_FRACTION:.0%} missing or non-finite values on modelling rows",
    "NEAR_CONSTANT": "Factor is constant or nearly constant on the training rows",
    "REDUNDANT": f"|Spearman| >= {config.REDUNDANCY_THRESHOLD} with an existing factor on training rows",
    "MISALIGNED": "Output is not a numeric series aligned with the trading dates",
}


@dataclass
class ValidationResult:
    name: str
    status: str                       # "valid" or "invalid"
    code: str = "OK"
    message: str = ""
    warnings: list = field(default_factory=list)
    lookback: int = None
    inputs_used: tuple = ()
    canonical: str = None
    missing_fraction: float = None
    max_abs_correlation: float = None
    most_correlated_with: str = None
    values: pd.Series = None

    @property
    def is_valid(self):
        return self.status == "valid"

    def to_record(self):
        return {"name": self.name, "status": self.status, "code": self.code, "message": self.message,
                "warnings": "; ".join(self.warnings), "lookback": self.lookback,
                "inputs_used": ",".join(self.inputs_used), "missing_fraction": self.missing_fraction,
                "max_abs_spearman_train": self.max_abs_correlation,
                "most_correlated_with": self.most_correlated_with}


def _reject(result, code, message):
    result.status, result.code, result.message = "invalid", code, message
    return result


def validate_candidate(candidate, dev_wide, model_index, train_index, reference_features,
                       seen_names, seen_canonicals):
    """Validate one candidate dict.

    dev_wide: development inputs (no test rows). model_index: train+validation
    rows. reference_features: factors already in use (baseline, retained, and
    valid candidates accepted earlier in this round's validation pass).
    seen_names / seen_canonicals: every name / expression proposed so far.
    """
    name = candidate.get("name") if isinstance(candidate, dict) else None
    result = ValidationResult(name=str(name) if name is not None else "<missing>", status="valid")

    # 1. schema
    if not isinstance(candidate, dict):
        return _reject(result, "MALFORMED_CANDIDATE", "Candidate is not a JSON object")
    missing = [f for f in REQUIRED_TEXT_FIELDS if not isinstance(candidate.get(f), str) or not candidate[f].strip()]
    inputs = candidate.get("inputs")
    if missing or not isinstance(inputs, list) or not all(isinstance(i, str) for i in inputs):
        detail = f"missing/empty fields: {missing}" if missing else "inputs must be a list of strings"
        return _reject(result, "MALFORMED_CANDIDATE", detail)

    # 2. name
    if not NAME_PATTERN.match(name):
        return _reject(result, "INVALID_NAME", CHECK_DESCRIPTIONS["INVALID_NAME"])
    if name.lower() in {n.lower() for n in seen_names}:
        return _reject(result, "DUPLICATE_NAME", f"Name '{name}' was already used")

    # 3. declared inputs exist
    unknown = [i for i in inputs if i not in ALLOWED_INPUTS]
    if unknown:
        return _reject(result, "UNKNOWN_INPUT", f"Declared inputs do not exist: {unknown}")

    # 4-6. syntax, allowlist, lookback, computation
    expression = candidate["expression"]
    try:
        computed = compute_expression(dev_wide, expression)
    except FactorExpressionError as exc:
        return _reject(result, exc.code, str(exc))
    except (ValueError, TypeError, OverflowError, RecursionError, KeyError) as exc:
        return _reject(result, "MISALIGNED", f"Computation failed: {type(exc).__name__}: {exc}")
    result.lookback, result.inputs_used, result.canonical = computed.lookback, computed.inputs_used, computed.canonical
    if computed.canonical in seen_canonicals:
        return _reject(result, "DUPLICATE_EXPRESSION",
                       f"Same expression as previously proposed '{seen_canonicals[computed.canonical]}'")
    if set(computed.inputs_used) - set(inputs):
        result.warnings.append(f"inputs used but not declared: {sorted(set(computed.inputs_used) - set(inputs))}")
    windows = candidate.get("windows") or []
    if any(not isinstance(w, int) or w > config.MAX_LOOKBACK for w in windows):
        result.warnings.append(f"declared windows {windows} differ from the allowed range")

    values = computed.values
    if not values.index.equals(dev_wide.index) or not pd.api.types.is_float_dtype(values):
        return _reject(result, "MISALIGNED", "Output is not a float series on the trading-date index")

    # 7. truncation (leakage) test: values up to date T must not change when later data is removed
    for fraction in config.TRUNCATION_CHECK_POINTS:
        cut = max(config.MAX_LOOKBACK + 2, int(len(dev_wide) * fraction))
        truncated = compute_expression(dev_wide.iloc[:cut], expression).values
        if not np.allclose(truncated.to_numpy(), values.iloc[:cut].to_numpy(),
                           rtol=1e-9, atol=1e-12, equal_nan=True):
            return _reject(result, "FUTURE_REFERENCE",
                           f"Values change when data after {dev_wide.index[cut - 1].date()} is removed")

    # 8. missing / non-finite values on modelling rows
    modelling = values.reindex(model_index)
    result.missing_fraction = float(modelling.isna().mean())
    if result.missing_fraction > config.MAX_MISSING_FRACTION:
        return _reject(result, "TOO_MANY_MISSING",
                       f"{result.missing_fraction:.1%} of modelling rows are missing or non-finite")

    # 9. near-constant on training rows
    train_values = values.reindex(train_index).dropna()
    unique_fraction = train_values.nunique() / max(len(train_values), 1)
    if train_values.std() <= config.MIN_TRAIN_STD or unique_fraction < config.MIN_UNIQUE_FRACTION:
        return _reject(result, "NEAR_CONSTANT",
                       f"Training std={train_values.std():.3g}, distinct-value share={unique_fraction:.1%}")

    # 10. redundancy against factors already in use
    if reference_features is not None and reference_features.shape[1]:
        reference = reference_features.reindex(train_index)
        correlations = reference.corrwith(values.reindex(train_index), method="spearman").abs().dropna()
        if len(correlations):
            result.max_abs_correlation = float(correlations.max())
            result.most_correlated_with = str(correlations.idxmax())
            if result.max_abs_correlation >= config.REDUNDANCY_THRESHOLD:
                return _reject(result, "REDUNDANT",
                               f"|Spearman|={result.max_abs_correlation:.3f} with '{result.most_correlated_with}' "
                               f"on training rows")

    result.values = values
    result.message = "Passed all checks"
    return result
