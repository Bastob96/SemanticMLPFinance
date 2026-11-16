"""Direction test (src/stage2/direction.py): fast checks on made-up data, no model fitting and no API."""

import numpy as np
import pandas as pd
import pytest

from src.stage2.direction import DOWN, UP, composite, edges, match_test, persistence, shuffled_edges, stated_signs
from src.stage2.prompt_v2 import TOOL_BATCH, TOOL_DIRECTION, build_prompt_direction, direction_evidence


def blocks(seed=0, years=2, days=120):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(years):
        d = pd.date_range(f"{2022 + i}-01-03", periods=days, freq="B")
        out.append((d, rng.integers(0, 2, days).astype(float)))
    return out


def table_for(blks, seed=1):
    rng = np.random.default_rng(seed)
    idx = blks[0][0].append([b[0] for b in blks[1:]])
    y = np.concatenate([b[1] for b in blks])
    return pd.DataFrame({"perfect_up": y + rng.random(len(y)) * 0.1, "perfect_down": -y + rng.random(len(y)) * 0.1,
                         "noise": rng.standard_normal(len(y))}, index=idx)


def test_edge_sign_and_size():
    blks = blocks()
    e = edges(table_for(blks), blks)
    assert e["perfect_up"] == pytest.approx(0.5) and e["perfect_down"] == pytest.approx(-0.5)
    assert abs(e["noise"]) < 0.15


def test_edge_ignores_a_years_overall_up_rate():
    """A factor that only says which YEAR it is has no edge, even if one year has more up days."""
    d1, d2 = pd.date_range("2022-01-03", periods=100, freq="B"), pd.date_range("2023-01-03", periods=100, freq="B")
    blks = [(d1, np.r_[np.ones(20), np.zeros(80)]), (d2, np.r_[np.ones(80), np.zeros(20)])]
    year_flag = pd.DataFrame({"year_flag": np.r_[np.zeros(100), np.ones(100)]}, index=d1.append(d2))
    assert edges(year_flag, blks)["year_flag"] == pytest.approx(0.0)


def test_match_test_right_wrong_and_shuffles():
    blks = blocks()
    t = table_for(blks)[["perfect_up", "perfect_down"]]
    right = match_test(t, [UP, DOWN], blks, n=200)
    wrong = match_test(t, [DOWN, UP], blks, n=200)
    assert right["match_rate"] == 1.0 and right["p_signed_edge"] < 0.02
    assert wrong["match_rate"] == 0.0 and wrong["mean_signed_edge"] == pytest.approx(-0.5)
    assert shuffled_edges(t, blks, n=50).shape == (50, 2)
    assert list(stated_signs([UP, DOWN])) == [1.0, -1.0]


def test_persistence_and_composite_use_only_the_past():
    search, q4 = blocks(seed=0, days=100), blocks(seed=5, days=100)
    q4 = [(d + pd.Timedelta(days=200), y) for d, y in q4]
    both = pd.concat([table_for(search), table_for(q4)])
    r = persistence(both, search, q4, n=200)
    assert r["factors"] == 3 and r["same_direction"] >= 2 / 3
    sig = composite(both[["perfect_up"]], [1.0])
    changed = both[["perfect_up"]].copy()
    changed.iloc[-1] = 99.0                                   # a change on the last day cannot change earlier days
    assert composite(changed, [1.0]).iloc[:-1].equals(sig.iloc[:-1])


def test_direction_prompt_and_tool():
    kw = dict(n=100, base_formulas={"ret_1": "pct_change(AAPL_Close, 1)"}, stage1_top="x", market="m")
    first = build_prompt_direction(**kw)
    ev = direction_evidence([dict(name="f", formula="lag(AAPL_Close, 1)", expected_sign=UP, edge_search=-0.03,
                                  years_same=3, right=False)], 0)
    second = build_prompt_direction(**kw, evidence=ev)
    assert "ROUND 1" in first and "EVIDENCE" not in first
    assert "ROUND 2" in second and "WRONG" in second and "sealed" in second
    assert "expected_sign" in TOOL_DIRECTION["input_schema"]["properties"]["factors"]["items"]["required"]
    assert "expected_sign" not in TOOL_BATCH["input_schema"]["properties"]["factors"]["items"]["required"]
