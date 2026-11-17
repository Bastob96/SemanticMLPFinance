"""Collect what the Factor Lab app needs into app_data/ (small files that are committed with the code).

The app computes every live number itself from prices. It only needs two things from the research runs:
    app_data/factors.csv     every factor with its formula, plain-English meaning and the direction the LLM stated
    app_data/research.json   the headline numbers of the direction test and its checks (Research page)

Sources: results/stage2/direction/<model>/ (experiments/run_direction_test.py) and
results/stage2/direction/time_order.csv, fresh_2026.csv (experiments/check_direction_time_order.py).

Usage:  python experiments/build_app_data.py

Author: Abu Wakkas Bastob (capstone PG-S2-36, supervised by A/Prof Lingqiao Liu).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.stage2.context import BASE_FORMULAS  # noqa: E402

DIRECTION = ROOT / "results/stage2/direction"
OUT = ROOT / "app_data"
KEPT = ROOT / "results/stage2/runs_v2/20261021_1110_replay_20261015_1635_claude/kept_factors.json"   # Stage 2 loop v2, run B
KEPT_ROLE = {"nvda_range_vs_aapl_range": "relative_move", "msft_nvda_close_position_diff": "peer_lead",
             "nvda_vol10_minus_aapl_vol10": "relative_move", "nvda_range_vs_qqq_range_zscore": "peer_lead"}
MODELS = {"opus-5-5": "Opus 5.5", "fable-5-2": "Fable 5.2", "haiku-5-5": "Haiku 5.5"}

BASE_ROLE = {"AAPL": "apple_own", "QQQ": "market_condition", "MSFT": "peer_lead", "NVDA": "peer_lead"}
BASE_MEANING = {
    "AAPL_return_1d": "Apple's return today.",
    "AAPL_mean_return_3d": "Apple's average daily return over the last 3 days.",
    "AAPL_mean_return_5d": "Apple's average daily return over the last week.",
    "AAPL_mean_return_10d": "Apple's average daily return over the last two weeks.",
    "AAPL_mean_return_20d": "Apple's average daily return over the last month.",
    "AAPL_cum_return_5d": "Apple's total return over the last week.",
    "AAPL_cum_return_10d": "Apple's total return over the last two weeks.",
    "AAPL_cum_return_20d": "Apple's total return over the last month.",
    "AAPL_volatility_5d": "How much Apple's daily returns have swung over the last week.",
    "AAPL_volatility_10d": "How much Apple's daily returns have swung over the last two weeks.",
    "AAPL_volatility_20d": "How much Apple's daily returns have swung over the last month.",
    "AAPL_close_to_ma20": "How far Apple's price is above or below its 20-day average.",
    "AAPL_volume_to_ma20": "How heavy Apple's trading volume is compared with its 20-day average.",
    "AAPL_daily_range": "The size of Apple's high-to-low range today, as a share of its price.",
    "QQQ_return_1d": "The Nasdaq-100 fund's return today.",
    "QQQ_cum_return_5d": "The Nasdaq-100 fund's total return over the last week.",
    "MSFT_return_1d": "Microsoft's return today.",
    "MSFT_cum_return_5d": "Microsoft's total return over the last week.",
    "NVDA_return_1d": "NVIDIA's return today.",
    "NVDA_cum_return_5d": "NVIDIA's total return over the last week.",
    "AAPL_minus_QQQ_1d": "Apple's return today minus the Nasdaq-100 fund's return.",
    "AAPL_minus_peer_mean_1d": "Apple's return today minus the average return of Microsoft and NVIDIA.",
}


def catalogue() -> pd.DataFrame:
    rows = []
    for name, formula in BASE_FORMULAS.items():
        role = "relative_move" if "minus" in name else BASE_ROLE[name.split("_")[0]]
        rows.append({"id": f"base:{name}", "name": name, "source": "Existing 22", "round": 0, "role": role,
                     "formula": formula, "meaning": BASE_MEANING[name], "stated": "", "new_idea": ""})
    for k in json.loads(KEPT.read_text()):
        rows.append({"id": f"kept:{k['name']}", "name": k["name"], "source": "Iterative search (kept)", "round": 0,
                     "role": KEPT_ROLE[k["name"]], "formula": k["formula"],
                     "meaning": k["hypothesis"] + f" Kept by the iterative search ({k['how_selected']}).", "stated": "", "new_idea": ""})
    for tag, label in MODELS.items():
        f = pd.read_csv(DIRECTION / tag / "factors.csv")
        for _, r in f.iterrows():
            rows.append({"id": f"{tag}:r{int(r['round'])}:{r['name']}", "name": r["name"], "source": label,
                         "round": int(r["round"]), "role": r["role"], "formula": r["formula"], "meaning": r["meaning"],
                         "stated": "up" if r["expected_sign"] == "higher_means_up" else "down",
                         "new_idea": "" if int(r["round"]) == 1 else str(r["new_idea"]) == "True"})
    for folder in sorted(DIRECTION.glob("template_coinflip_*")):
        f = pd.read_csv(folder / "factors.csv")
        for _, r in f[f["round"] == 1].iterrows():
            rows.append({"id": f"template{folder.name[-1]}:{r['name']}", "name": f"template_{folder.name[-1]}_{r['name'][-3:]}",
                         "source": "Templates (no LLM)", "round": 0, "role": "template", "formula": r["formula"],
                         "meaning": "A formula filled in at random from six hand-written templates. No LLM and no reasoning: the control group.",
                         "stated": "", "new_idea": ""})
    return pd.DataFrame(rows)


def research() -> dict:
    out = {"built_from": "experiments/run_direction_test.py and experiments/check_direction_time_order.py, 16 Nov 2026",
           "models": {}}
    for tag, label in MODELS.items():
        s = pd.read_csv(DIRECTION / tag / "summary.csv").set_index("test")
        extra = json.loads((DIRECTION / tag / "summary_extra.json").read_text())
        pick = lambda key: {"rate": float(s.loc[key, "match_rate"]), "n": int(s.loc[key, "factors"]),  # noqa: E731
                            "p_better": float(s.loc[key, "p_better_than_chance"]),
                            "p_worse": float(s.loc[key, "p_worse_than_chance"])}
        out["models"][label] = {
            "blind_all_days": pick("round 1: stated direction, all days"),
            "blind_q4": pick("round 1: stated direction, sealed Q4"),
            "feedback_q4": pick("round 2 (new ideas): stated direction, sealed Q4"),
            "data_alone_q4": extra["reference_data_direction_search_to_q4"]["same_direction"],
            "signal_blind_q4": extra["signal_round1_q4"], "signal_feedback_q4": extra["signal_round2_q4"],
            "lesson": extra["lesson"]}
    out["time_order"] = json.loads(pd.read_csv(DIRECTION / "time_order.csv").to_json(orient="records"))
    out["fresh_2026"] = json.loads(pd.read_csv(DIRECTION / "fresh_2026.csv").to_json(orient="records"))
    # the old judge (keep a factor if it clears pass bars on validation data), batches of 100, 12-16 Nov
    out["old_judge"] = {"note": "Factors kept by the search in the 100-factor batches (three batches each)",
                        "kept": {"LLM (Sonnet 5.5)": [0, 0, 0], "Templates (no LLM)": [5, 6, 0], "Random numbers": [5, 3, 3]}}
    return out


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    cat = catalogue()
    cat.to_csv(OUT / "factors.csv", index=False)
    (OUT / "research.json").write_text(json.dumps(research(), indent=1) + "\n")
    print(cat.groupby(["source", "round"]).size().to_string())
    print(f"Wrote {OUT / 'factors.csv'} ({len(cat)} factors) and research.json")
