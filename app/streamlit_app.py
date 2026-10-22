"""Results display for PG-S2-36: interpretable financial factors for AAPL direction.

A thin, read-only display layer. It loads result files that already exist in the
repository and recomputes nothing about the research pipeline.

Run from the repository root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STAGE2 = ROOT / "Stage2_Factor_Files"

SOURCES = {
    "Original committed run": RESULTS / "stage1_results_summary.csv",
    "Re-verified run (Abu, 16 Sep 2026)": RESULTS / "abu_stage1_2025" / "evaluation_comparison.csv",
}
PREDICTIONS = {
    "Original committed run": RESULTS / "stage1_predictions.csv",
    "Re-verified run (Abu, 16 Sep 2026)": RESULTS / "abu_stage1_2025" / "stage1_predictions.csv",
}
BACKTEST_DAILY = RESULTS / "abu_stage1_2025" / "backtest_daily.csv"
BACKTEST_SUMMARY = RESULTS / "abu_stage1_2025" / "backtest_summary.csv"
CANDIDATES = STAGE2 / "data" / "generated" / "run_20260913T011756Z" / "validated_candidates.json"
CANDIDATE_META = STAGE2 / "data" / "generated" / "run_20260913T011756Z" / "generation_metadata.json"
PHASE2_RESULTS = RESULTS / "phase2_abc_validation_test_results(taiyu).csv"
PHASE2_IMPORTANCE = RESULTS / "phase2_experiment_c_factor_importance(Taiyu).csv"

EXPERIMENT_LABELS = {
    "baseline": "Baseline",
    "A": "A: AAPL only",
    "B": "B: AAPL + QQQ",
    "C": "C: AAPL + QQQ + MSFT + NVDA",
}
MODEL_LABELS = {
    "majority_class": "Majority class",
    "previous_direction": "Previous direction",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}
METRICS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "auc": "AUC",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
}

DISCLAIMER = (
    "Academic research only. Nothing here is financial advice, a trading signal or a "
    "forecast of future prices. Results come from one 63 day test period (Q4 2025) and "
    "are close to chance."
)


@st.cache_data
def load_csv(path: str) -> pd.DataFrame | None:
    p = Path(path)
    return pd.read_csv(p) if p.is_file() else None


@st.cache_data
def load_json(path: str) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def missing(path: Path) -> None:
    st.warning(f"File not found: `{path.relative_to(ROOT)}`. Pull the latest repository files.")


def fmt(mean: float, std: float | None) -> str:
    if pd.isna(mean):
        return ""
    if std is None or pd.isna(std) or std == 0:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"


def results_table(df: pd.DataFrame, split: str) -> pd.DataFrame:
    sub = df[df["split"] == split].copy()
    order = {"baseline": 0, "A": 1, "B": 2, "C": 3}
    model_order = {"majority_class": 0, "previous_direction": 1, "logistic_regression": 2, "random_forest": 3}
    sub["_o"] = sub["experiment"].map(order)
    sub["_m"] = sub["model"].map(model_order)
    sub = sub.sort_values(["_o", "_m"])
    out = pd.DataFrame(
        {
            "Experiment": sub["experiment"].map(EXPERIMENT_LABELS),
            "Model": sub["model"].map(MODEL_LABELS),
        }
    )
    for key, label in METRICS.items():
        std_col = f"{key}_std"
        out[label] = [
            fmt(m, s if std_col in sub.columns else None)
            for m, s in zip(sub[f"{key}_mean"], sub[std_col] if std_col in sub.columns else [None] * len(sub))
        ]
    return out.reset_index(drop=True)


def side_note() -> None:
    st.sidebar.markdown("---")
    st.sidebar.caption(DISCLAIMER)


# ---------------------------------------------------------------- pages


def page_overview() -> None:
    st.title("Interpretable financial factors for stock prediction")
    st.caption("PG-S2-36 · Capstone · Supervisor: Associate Professor Lingqiao Liu, AIML, University of Adelaide")
    st.info(DISCLAIMER)

    st.subheader("Research question")
    st.markdown(
        "Can a language model discover financial factors that improve next day stock "
        "direction prediction **and** can be read and checked by a person?"
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("What is predicted")
        st.markdown(
            "- Target: will **AAPL** close higher the next trading day (up or down)\n"
            "- Related inputs: **MSFT**, **NVDA** and the Nasdaq 100 ETF **QQQ**\n"
            "- Data: public daily prices, 2025 split by time into train (Q1 to Q2, 121 days), "
            "validation (Q3, 63 days) and an untouched test period (Q4, 63 days)"
        )
    with c2:
        st.subheader("How the project is staged")
        st.markdown(
            "- **Stage 1:** a reliable pipeline, two naive baselines, Logistic Regression and "
            "Random Forest, and three experiments (A, B, C) that add related stocks step by step\n"
            "- **Stage 2:** a language model proposes readable factors; they are validated, "
            "implemented and compared with the Stage 1 features\n"
            "- The language model never predicts prices. It only proposes formulas that people can read."
        )

    summary = load_csv(str(SOURCES["Original committed run"]))
    if summary is not None:
        t = summary[summary["split"] == "test"]
        ml = t[t["experiment"].isin(["A", "B", "C"])]
        best = ml.loc[ml["balanced_accuracy_mean"].idxmax()]
        prev = t[t["model"] == "previous_direction"].iloc[0]
        maj = t[t["model"] == "majority_class"].iloc[0]
        st.subheader("Headline result on the untouched test period")
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Best model, balanced accuracy",
            f"{best['balanced_accuracy_mean']:.3f}",
            help=f"{EXPERIMENT_LABELS[best['experiment']]}, {MODEL_LABELS[best['model']]}",
        )
        m2.metric("Previous direction baseline", f"{prev['balanced_accuracy_mean']:.3f}")
        m3.metric("Majority class baseline (accuracy)", f"{maj['accuracy_mean']:.3f}")
        st.markdown(
            f"The best model was **{EXPERIMENT_LABELS[best['experiment']]}, {MODEL_LABELS[best['model']]}**. "
            "It did not beat the simple previous direction baseline, and adding related stocks did not help "
            "on this test period. A balanced accuracy of 0.5 is the level of a coin toss. "
            "This is a valid research outcome and is reported as found."
        )

    st.subheader("What is in this display")
    st.markdown(
        "Use the menu on the left: the eight language model factors, the Stage 1 results, "
        "day by day predictions for the test period, the Stage 2 comparison, and the limitations."
    )


def page_factors() -> None:
    st.title("Language model factors")
    st.info(DISCLAIMER)
    data = load_json(str(CANDIDATES))
    meta = load_json(str(CANDIDATE_META))
    if data is None:
        missing(CANDIDATES)
        return

    if meta:
        st.markdown(
            f"One structured batch of **{len(data['candidates'])} candidate factors** was generated on "
            f"{meta['generated_at_utc'][:10]} (prompt version `{meta['prompt_version']}`, "
            f"model `{meta['model']}`). It passed every local check: allowed inputs and operators only, "
            "formula syntax, matching inputs and lookbacks, and no duplicates. No factor was chosen "
            "using test period results."
        )

    rows = pd.DataFrame(data["candidates"])
    overview = rows[["factor_id", "name", "family", "inputs", "lookback_days", "expected_direction"]].copy()
    overview["inputs"] = overview["inputs"].apply(lambda x: ", ".join(x))
    overview.columns = ["ID", "Name", "Family", "Inputs", "Lookback (days)", "Expected direction"]
    st.dataframe(overview, hide_index=True, width="stretch")

    st.subheader("Each factor in detail")
    for cand in data["candidates"]:
        with st.expander(f"{cand['factor_id']} · {cand['name']}"):
            st.markdown("**Hypothesis**")
            st.write(cand["hypothesis"])
            st.markdown("**Formula**")
            st.code(cand["formula"], language="text")
            st.markdown("**Leakage self check (written by the language model, then tested locally)**")
            st.write(cand["leakage_self_check"])
            st.markdown("**How it differs from the baseline features**")
            st.write(cand["difference_from_baseline"])


def page_stage1() -> None:
    st.title("Stage 1 results")
    st.info(DISCLAIMER)

    source = st.sidebar.radio("Result source", list(SOURCES), key="stage1_source")
    df = load_csv(str(SOURCES[source]))
    if df is None:
        missing(SOURCES[source])
        return
    st.caption(
        "Random Forest rows are the mean ± standard deviation over three random seeds. "
        "Logistic Regression and baselines are single deterministic runs. Random Forest numbers "
        "shift slightly with the scikit-learn version, which is why two runs are offered."
    )

    split = st.radio("Period", ["test", "validation"], horizontal=True,
                     format_func=lambda s: "Test (Q4 2025, untouched)" if s == "test" else "Validation (Q3 2025)")
    st.dataframe(results_table(df, split), hide_index=True, width="stretch")

    st.subheader("Compare models")
    metric = st.selectbox("Metric", list(METRICS), index=1, format_func=lambda k: METRICS[k])
    sub = df[df["split"] == split].copy()
    sub["Configuration"] = sub["experiment"].map(EXPERIMENT_LABELS) + " · " + sub["model"].map(MODEL_LABELS)
    sub["value"] = sub[f"{metric}_mean"]
    sub["Kind"] = sub["experiment"].map(lambda e: "Baseline" if e == "baseline" else "Machine learning")
    chart = (
        alt.Chart(sub)
        .mark_bar()
        .encode(
            y=alt.Y("Configuration:N", sort="-x", title=None),
            x=alt.X("value:Q", title=METRICS[metric], scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Kind:N", legend=alt.Legend(title=None)),
            tooltip=["Configuration", alt.Tooltip("value:Q", format=".3f", title=METRICS[metric])],
        )
    )
    rule = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
    st.altair_chart(chart + rule, width="stretch")
    st.caption("The dashed line at 0.5 marks coin toss level for balanced accuracy and AUC.")

    st.subheader("How to read this")
    st.markdown(
        "- **Accuracy** is the share of days predicted correctly. The majority class baseline scores well "
        "on it only because up days were more common in the test period.\n"
        "- **Balanced accuracy** and **AUC** correct for that, so they are the fairer comparison.\n"
        "- Adding QQQ, MSFT and NVDA (experiments B and C) did not improve these scores on the test period."
    )


def page_predictions() -> None:
    st.title("Test period predictions")
    st.info(DISCLAIMER)
    st.caption("Day by day predicted against actual AAPL direction for the 63 held out days of Q4 2025.")

    source = st.sidebar.radio("Result source", list(PREDICTIONS), key="pred_source")
    preds = load_csv(str(PREDICTIONS[source]))
    if preds is None:
        missing(PREDICTIONS[source])
        return
    preds = preds[preds["split"] == "test"].copy()

    c1, c2 = st.columns(2)
    exp = c1.selectbox("Experiment", ["baseline", "A", "B", "C"], index=1, format_func=EXPERIMENT_LABELS.get)
    models = sorted(preds[preds["experiment"] == exp]["model"].unique(), key=list(MODEL_LABELS).index)
    model = c2.selectbox("Model", models, format_func=MODEL_LABELS.get)

    sel = preds[(preds["experiment"] == exp) & (preds["model"] == model)]
    if sel.empty:
        st.warning("No predictions for this selection.")
        return
    if model == "random_forest":
        day = sel.groupby(["Date", "target_date", "actual"], as_index=False)["probability_up"].mean()
        day["predicted"] = (day["probability_up"] >= 0.5).astype(int)
        st.caption(
            "Random Forest shows the average probability across three seeds, thresholded at 0.5. "
            "The count can differ by a day or two from the mean per seed accuracy on the Stage 1 page."
        )
    else:
        day = sel[["Date", "target_date", "actual", "predicted", "probability_up"]].copy()
    day["Date"] = pd.to_datetime(day["Date"])
    day["correct"] = day["actual"] == day["predicted"]

    k1, k2, k3 = st.columns(3)
    k1.metric("Days", len(day))
    k2.metric("Correct", f"{int(day['correct'].sum())} ({day['correct'].mean():.1%})")
    k3.metric("Actual up days", f"{int(day['actual'].sum())} ({day['actual'].mean():.1%})")

    plot = day.copy()
    plot["Outcome"] = plot["correct"].map({True: "Correct", False: "Wrong"})
    plot["Predicted"] = plot["predicted"].map({1: "Up", 0: "Down"})
    plot["Actual"] = plot["actual"].map({1: "Up", 0: "Down"})
    strip = (
        alt.Chart(plot)
        .mark_rect()
        .encode(
            x=alt.X("Date:T", title=None),
            color=alt.Color("Outcome:N", scale=alt.Scale(domain=["Correct", "Wrong"], range=["#2a9d8f", "#e76f51"])),
            tooltip=[alt.Tooltip("Date:T"), "Predicted", "Actual", "Outcome"],
        )
        .properties(height=60)
    )
    st.altair_chart(strip, width="stretch")

    plot["cumulative_accuracy"] = plot["correct"].expanding().mean()
    line = (
        alt.Chart(plot)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("cumulative_accuracy:Q", title="Cumulative accuracy", scale=alt.Scale(domain=[0, 1])),
        )
    )
    rule = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
    st.altair_chart(line + rule, width="stretch")

    table = plot[["Date", "Predicted", "Actual", "Outcome", "probability_up"]].copy()
    table["Date"] = table["Date"].dt.strftime("%Y-%m-%d")
    table = table.rename(columns={"probability_up": "Probability of up"})
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={"Probability of up": st.column_config.NumberColumn(format="%.3f")})

    st.subheader("A simple trading check")
    st.caption(
        "A basic back test (go long when the model says up, otherwise stay out) was run as infrastructure, "
        "not as a claim that the model can be traded. The strategy was chosen on validation results only."
    )
    daily = load_csv(str(BACKTEST_DAILY))
    summ = load_csv(str(BACKTEST_SUMMARY))
    if daily is not None and summ is not None:
        d = daily.copy()
        d["Date"] = pd.to_datetime(d["Date"])
        long = d.melt("Date", ["strategy_equity", "buy_hold_equity"], var_name="Series", value_name="Equity")
        long["Series"] = long["Series"].map({"strategy_equity": "Model long/flat", "buy_hold_equity": "Buy and hold AAPL"})
        st.altair_chart(
            alt.Chart(long).mark_line().encode(
                x=alt.X("Date:T", title=None),
                y=alt.Y("Equity:Q", scale=alt.Scale(zero=False), title="Growth of 1"),
                color=alt.Color("Series:N", legend=alt.Legend(title=None)),
            ),
            width="stretch",
        )
        s = summ.copy()
        s["strategy"] = s["strategy"].map({"long_flat": "Model long/flat", "buy_and_hold_aapl": "Buy and hold AAPL"})
        s = s.rename(columns={"strategy": "Strategy", "cumulative_return": "Return", "max_drawdown": "Max drawdown", "sharpe": "Sharpe"})
        shown = s.assign(**{
            "Return": (s["Return"] * 100).round(2).astype(str) + "%",
            "Max drawdown": (s["Max drawdown"] * 100).round(2).astype(str) + "%",
            "Sharpe": s["Sharpe"].round(2),
        })
        st.dataframe(shown, hide_index=True, width="stretch")
        st.caption("The model driven strategy lost money and trailed simply holding AAPL over the same quarter.")
    else:
        st.info("Back test files were not found in this checkout.")


def page_stage2() -> None:
    st.title("Stage 2: do the language model factors help?")
    st.info(DISCLAIMER)
    res = load_csv(str(PHASE2_RESULTS))
    imp = load_csv(str(PHASE2_IMPORTANCE))
    if res is None:
        missing(PHASE2_RESULTS)
        return

    st.markdown(
        "This comparison comes from a separate regression track run by a team member. It predicts the "
        "next day **return** with a HistGradientBoosting model and then reads off the direction. "
        "Its experiment labels differ from Stage 1: here **A** uses 10 traditional factors, **B** adds the "
        "4 language model factors that compare AAPL with QQQ, and **C** adds all 8."
    )
    t = res.copy()
    t = t.rename(columns={
        "Experiment": "Experiment", "Factors": "Factors",
        "Val_Direction": "Validation direction accuracy", "Test_Direction": "Test direction accuracy",
        "Val_R2": "Validation R²", "Test_R2": "Test R²",
    })
    st.dataframe(
        t[["Experiment", "Factors", "Validation direction accuracy", "Test direction accuracy", "Validation R²", "Test R²"]],
        hide_index=True, width="stretch",
        column_config={c: st.column_config.NumberColumn(format="%.3f") for c in
                       ["Validation direction accuracy", "Test direction accuracy", "Validation R²", "Test R²"]},
    )

    long = res.melt("Experiment", ["Val_Direction", "Test_Direction"], var_name="Period", value_name="Direction accuracy")
    long["Period"] = long["Period"].map({"Val_Direction": "Validation", "Test_Direction": "Test"})
    bars = alt.Chart(long).mark_bar().encode(
        x=alt.X("Experiment:N", title=None, axis=alt.Axis(labelAngle=0, labelLimit=200)),
        xOffset="Period:N",
        y=alt.Y("Direction accuracy:Q", scale=alt.Scale(domain=[0, 1])),
        color="Period:N",
        tooltip=["Experiment", "Period", alt.Tooltip("Direction accuracy:Q", format=".3f")],
    )
    rule = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
    st.altair_chart(bars + rule, width="stretch")
    st.markdown(
        "On the test period, direction accuracy **fell** as language model factors were added, and R² was "
        "negative throughout, meaning the models predicted returns worse than a constant guess. "
        "So in this run the language model factors did not improve prediction."
    )

    if imp is not None:
        st.subheader("Which factors the model leaned on (experiment C)")
        st.caption(
            "Permutation importance on the validation data. Values are tiny and several are negative, "
            "so this shows weak, unstable signals rather than a clear ranking."
        )
        chart = alt.Chart(imp).mark_bar().encode(
            y=alt.Y("Factor:N", sort="-x", title=None),
            x=alt.X("Importance:Q", title="Permutation importance"),
            color=alt.Color("Type:N", legend=alt.Legend(title=None)),
            tooltip=["Factor", "Type", alt.Tooltip("Importance:Q", format=".5f")],
        )
        st.altair_chart(chart, width="stretch")

    st.subheader("Still to come")
    st.markdown(
        "The Stage 2 factor table and its A, B, C feature sets are built and tested. "
        "A classification comparison on the same split as Stage 1 is not part of this display yet."
    )


def page_limits() -> None:
    st.title("Limitations and responsible use")
    st.info(DISCLAIMER)
    st.markdown(
        "- **Small sample.** The test period has 63 labelled days. A few days either way changes accuracy by "
        "several percentage points.\n"
        "- **One market, one year.** All experiments use AAPL, MSFT, NVDA and QQQ in 2025. Findings should not "
        "be generalised to other stocks, periods or markets.\n"
        "- **Markets change.** A pattern that works in one quarter can disappear in the next.\n"
        "- **Prices are adjusted history.** The data is a fixed download, not a point in time archive.\n"
        "- **Language model formulas are hypotheses.** Each was checked locally for format and leakage, and "
        "none was kept or dropped using test period results.\n"
        "- **Negative results count.** The project aims to find out whether the approach works, not to show that it does.\n"
        "- **No investment use.** This is not financial advice and not a system for trading."
    )
    st.subheader("Reproducibility")
    st.markdown(
        "Every number on these pages is read from files in the repository `results` and `Stage2_Factor_Files` "
        "folders. The display recomputes nothing."
    )


PAGES = {
    "Overview": page_overview,
    "Language model factors": page_factors,
    "Stage 1 results": page_stage1,
    "Test period predictions": page_predictions,
    "Stage 2 comparison": page_stage2,
    "Limitations": page_limits,
}


def main() -> None:
    st.set_page_config(page_title="PG-S2-36 Results", page_icon="📈", layout="wide")
    st.sidebar.title("PG-S2-36")
    page = st.sidebar.radio("Page", list(PAGES), label_visibility="collapsed")
    PAGES[page]()
    side_note()


if __name__ == "__main__":
    main()
    