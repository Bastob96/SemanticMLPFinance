"""Aggregate independent RF seeds with sample SD (ddof=1)."""

import numpy as np
import pandas as pd

from .metrics import METRICS


def aggregate_runs(detailed):
    required = {"split", "experiment", "model", "seed", "n_observations", *METRICS}
    if detailed.empty or not required.issubset(detailed.columns):
        raise ValueError("Missing or empty detailed metrics")
    if detailed.duplicated(["split", "experiment", "model", "seed"]).any():
        raise ValueError("Duplicate metric runs")
    rows = []
    for key, group in detailed.groupby(["split", "experiment", "model"], sort=True):
        if key[2] == "random_forest":
            if len(group) != 3 or set(group.seed) != {0, 1, 2}:
                raise ValueError("Random Forest requires exactly seeds 0, 1 and 2")
        elif len(group) != 1:
            raise ValueError("Non-RF configurations require exactly one run")
        if group.n_observations.nunique() != 1:
            raise ValueError("Seed observation counts differ")
        row = dict(zip(["split", "experiment", "model"], key))
        row.update(n_runs=len(group), n_observations=int(group.n_observations.iloc[0]))
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            # Propagate undefined metrics instead of silently dropping a seed.
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
