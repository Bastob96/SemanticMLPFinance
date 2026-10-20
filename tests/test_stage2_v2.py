"""Fast checks for the version 2 helpers (report cards, keep rules, pair choice, prompt). No model fitting."""

import pandas as pd

from src.stage2.feedback import lab_notebook, report_card, why_line
from src.stage2.prompt_v2 import build_prompt_v2
from src.stage2.selection import Rules, choose_pairs, group_of, is_promising, passes, summarise_result


def test_group_of():
    assert group_of("pct_change(QQQ_Close, 1)") == "market"
    assert group_of("subtract(pct_change(AAPL_Close, 1), pct_change(QQQ_Close, 1))") == "relative"
    assert group_of("zscore(AAPL_Volume, 20)") == "apple"
    assert group_of("zscore(AAPL_Volume, 20)", "market_condition") == "market"   # Claude's label wins


def test_summarise_and_passes():
    per_year = pd.DataFrame({"year": [2022, 2023, 2024, 2025], "valid": True,
                             "auc_gain": [0.02, 0.01, -0.005, 0.03],
                             "auc_gain_lr": [0.01, 0.0, 0.0, 0.02], "auc_gain_rf": [0.03, 0.02, -0.01, 0.04]})
    res = summarise_result(per_year)
    assert res["years"] == 3 and abs(res["gain"] - 0.01375) < 1e-12
    assert res["by_year"][2024] == -0.005
    assert passes(res, Rules(), 0.01) and not passes(res, Rules(), 0.015)
    assert summarise_result(per_year.assign(valid=[True, False, True, True])) == {"valid": False}


def test_promising_list_rule():
    r = Rules()
    assert is_promising({"status": "tested", "max_corr": 0.5, "auc_gain": 0.003, "years_helped": 2}, r)
    assert not is_promising({"status": "tested", "max_corr": 0.95, "auc_gain": 0.02, "years_helped": 4}, r)
    assert not is_promising({"status": "tested", "max_corr": 0.5, "auc_gain": -0.003, "years_helped": 2}, r)
    # weak alone, but helped on top of the selected team: stays on the promising list
    assert is_promising({"status": "tested", "max_corr": 0.5, "auc_gain": -0.002, "years_helped": 1,
                         "team_size": 2, "team_gain": 0.004, "team_years_helped": 3}, r)


def test_choose_pairs():
    info = {"a": {"group": "market", "score": 0.004}, "b": {"group": "relative", "score": 0.003},
            "c": {"group": "market", "score": 0.006}, "d": {"group": "apple", "score": 0.001}}

    def corr(x, y):
        return 0.9 if {x, y} == {"a", "c"} else 0.1

    pairs = choose_pairs(["a"], list(info), info, corr, Rules(pairs_per_round=2), forced=[("d", "b")])
    assert pairs[0] == ("d", "b")                                  # Claude's partner pair is tried first
    assert ("a", "c") not in pairs and ("c", "a") not in pairs      # too similar to each other
    assert pairs[1] == ("a", "b") and len(pairs) == 2              # different kinds preferred; cap respected


def test_report_card_and_prompt():
    rec = {"round": 2, "name": "x", "mode": "partner", "parent": "y", "role": "relative_move",
           "status": "promising", "reason": "weak alone but some sign of life", "formula": "zscore(AAPL_Volume, 20)",
           "gain_2022": 0.01, "gain_2023": -0.004, "gain_2024": 0.002, "gain_2025": 0.003,
           "auc_gain": 0.00275, "auc_gain_lr": 0.004, "auc_gain_rf": -0.001, "max_corr": 0.2, "closest": "z",
           "best_pair_partner": "y", "best_pair_gain": 0.02, "best_pair_years": 3, "best_pair_synergy": 0.01,
           "lesson": "volume only matters on down days"}
    why = why_line(rec)
    assert "not in 2023" in why and "Logistic Regression only" in why and "too small" in why
    card = report_card(rec)
    assert "2023 -0.004" in card and "LR +0.004, RF -0.001" in card and "(parent: y)" in card
    assert "worked better together with y" in card and "your lesson: volume only matters" in card
    prompt = build_prompt_v2(round_no=3, n_improve=2, n_new=2, n_partner=2,
                             base_formulas={"f": "pct_change(AAPL_Close, 1)"}, stage1_top="-", market="-",
                             notebook=lab_notebook([rec]), team_text="(empty so far)",
                             promising_text="- x [relative] alone +0.003", pairs_text="(none yet)",
                             rules=Rules().as_dict(), previous_names=["x"])
    assert prompt.startswith("ROUND 3")
    for part in ("THE SELECTED TEAM", "PROMISING LIST", "PAIR TRIALS SO FAR", "YOUR LAB NOTEBOOK",
                 "2 PARTNER factors", "lesson for each of these factors from the previous round: x",
                 "Q4 is never used"):
        assert part in prompt, part
