"""Dashboard page renderers. All pages are read-only views of frozen outputs."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.features import FEATURE_SETS

from .charts import (
    CONFIG_LABELS,
    candidate_scatter,
    equity_chart,
    factor_year_bars,
    final_test_bars,
    importance_bars,
    stage1_heatmap,
    stage1_metric_bars,
)
from .components import callout, card, decimal, hero, percent
from .data import DashboardData

PLOT_CONFIG = {"displayModeBar": False, "responsive": True}
METRICS = {
    "Balanced accuracy": "balanced_accuracy_mean",
    "ROC AUC": "auc_mean",
    "Accuracy": "accuracy_mean",
    "F1 score": "f1_mean",
}
MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "majority_class": "Majority baseline",
    "previous_direction": "Previous direction",
}


def _plot(fig) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)


def _stage2_variants(frame: pd.DataFrame) -> pd.DataFrame:
    baseline = frame.loc[frame.features.eq("stage1_22") & frame.proposer.eq("claude")].copy()
    baseline["variant"] = "Stage 1 · 22 factors"
    added = frame.loc[frame.features.eq("stage1_plus_llm")].copy()
    added["variant"] = added.proposer.map(
        {"claude": "Stage 1 + Claude factor", "random": "Stage 1 + random factor"}
    )
    return pd.concat([baseline, added], ignore_index=True)


def overview(data: DashboardData) -> None:
    hero(
        "SemanticMLPFinance · Research dashboard",
        "Can interpretable factors improve next-day AAPL prediction?",
        "Explore the Stage 1 baselines, validation-selected backtests, and the Stage 2 LLM factor-discovery experiment. Every number below comes from versioned repository artifacts—not a live or silently changing model.",
    )

    stage1_2025 = data.stage1.loc[data.stage1.year.eq(2025) & data.stage1.split.eq("test")]
    model_2025 = stage1_2025.loc[~stage1_2025.experiment.eq("baseline")]
    best_model = model_2025.sort_values("balanced_accuracy_mean", ascending=False).iloc[0]
    previous = stage1_2025.loc[stage1_2025.model.eq("previous_direction")].iloc[0]
    claude = data.stage2_scoreboard.loc[data.stage2_scoreboard.proposer.eq("claude")]
    final = _stage2_variants(data.stage2_final_metrics)
    lr = final.loc[final.model.eq("logistic_regression")]
    pivot = lr.pivot(index="year", columns="variant", values="balanced_accuracy")
    improved = int(
        (pivot["Stage 1 + Claude factor"] > pivot["Stage 1 · 22 factors"]).sum()
    )

    cols = st.columns(4)
    with cols[0]:
        card(
            "Best observed 2025 model",
            decimal(best_model.balanced_accuracy_mean),
            f"{best_model.experiment} · {MODEL_NAMES[best_model.model]}; previous-direction baseline {previous.balanced_accuracy_mean:.3f}",
        )
    with cols[1]:
        card(
            "Stage 2 candidates",
            f"{len(claude)}",
            f"{int(claude.status.eq('kept').sum())} survived all pre-registered validation checks",
        )
    with cols[2]:
        kept = claude.loc[claude.status.eq("kept")].iloc[0]
        card(
            "Kept factor Q3 gain",
            f"{kept.auc_gain:+.4f}",
            f"Improved AUC in {int(kept.years_helped)}/4 validation years",
        )
    with cols[3]:
        card(
            "LR Q4 improvement",
            f"{improved}/4 years",
            "Claude factor improved balanced accuracy over the same 22 inputs",
        )

    st.subheader("The result at a glance")
    left, right = st.columns([1.55, 1])
    with left:
        _plot(
            final_test_bars(
                data.stage2_final_metrics,
                "logistic_regression",
                "balanced_accuracy",
                "Balanced accuracy",
            )
        )
    with right:
        callout(
            "<strong>What improved:</strong> adding the selected Claude factor raised Logistic Regression balanced accuracy in every Q4 year.",
            "success",
        )
        callout(
            "<strong>What did not:</strong> results are still small-sample and inconsistent by model. This is evidence for further testing, not evidence of a tradable system.",
            "warning",
        )
        st.markdown(
            """
            **Research design**

            - Train: Q1–Q2 of each year
            - Select and score candidates: Q3 only
            - Final evaluation: frozen Q4
            - Years: 2022–2025
            - Target: AAPL direction on the next trading day
            """
        )

    st.subheader("What each stage contributes")
    cols = st.columns(3)
    content = [
        ("01 · Baseline", "Stage 1", "Kevin's leakage-safe feature, model and seed pipeline establishes honest reference performance."),
        ("02 · Discovery", "Stage 2", "Claude proposes readable formulas; the same four Q3 validation quarters accept or reject them."),
        ("03 · Evidence", "Frozen test", "Abu's evaluation and backtesting code measures Q4 classification and strategy outcomes."),
    ]
    for col, (eyebrow, title, text) in zip(cols, content):
        with col:
            st.markdown(
                f'<div class="method-step"><div class="eyebrow">{eyebrow}</div><strong>{title}</strong>{text}</div>',
                unsafe_allow_html=True,
            )


def stage1(data: DashboardData) -> None:
    hero(
        "Stage 1 · Baseline modelling",
        "A fair, chronological benchmark",
        "Compare both baselines, Logistic Regression, and Random Forest across the three feature ablations. Random Forest values are means across seeds 0, 1 and 2; its error bars show sample standard deviation.",
    )
    split_col, metric_col, year_col = st.columns([1, 1.3, 1])
    split = split_col.selectbox("Evaluation split", ["test", "validation"], format_func=str.title)
    metric_label = metric_col.selectbox("Metric", list(METRICS))
    metric = METRICS[metric_label]
    year = year_col.selectbox("Detail year", sorted(data.stage1.year.unique(), reverse=True))

    if split == "test":
        callout(
            "Q4 is displayed for reporting only. It was not used for model, feature, seed, or threshold selection.",
            "warning",
        )
    _plot(stage1_heatmap(data.stage1, split, metric, metric_label))
    _plot(stage1_metric_bars(data.stage1, int(year), split, metric, metric_label))

    st.subheader("Exact values")
    table = data.stage1.loc[data.stage1.year.eq(year) & data.stage1.split.eq(split)].copy()
    table["configuration"] = [CONFIG_LABELS[(e, m)] for e, m in zip(table.experiment, table.model)]
    columns = ["configuration", "accuracy_mean", "balanced_accuracy_mean", "auc_mean", "f1_mean"]
    display = table[columns].rename(
        columns={
            "configuration": "Configuration",
            "accuracy_mean": "Accuracy",
            "balanced_accuracy_mean": "Balanced accuracy",
            "auc_mean": "AUC",
            "f1_mean": "F1",
        }
    )
    st.dataframe(
        display.style.format({key: "{:.3f}" for key in display.columns if key != "Configuration"}),
        width="stretch",
        hide_index=True,
    )

    st.divider()
    st.subheader("Feature importance uses validation only")
    model_col, n_col = st.columns([1, 1])
    importance_model = model_col.selectbox(
        "Model",
        ["logistic_regression", "random_forest"],
        format_func=lambda value: MODEL_NAMES[value],
    )
    top_n = n_col.slider("Factors shown", 6, 22, 12)
    _plot(importance_bars(data.importance, importance_model, top_n))
    st.caption(
        "Importance is the mean change in Q3 ROC AUC when a factor is shuffled. Positive does not prove causality; it only shows that the fitted model relied on that input in validation."
    )


def backtests(data: DashboardData) -> None:
    hero(
        "Stage 1 · Backtesting",
        "What happened when predictions became positions?",
        "For each year, the configuration was chosen from Q3 balanced accuracy and then simulated once on Q4. A prediction of UP holds AAPL; DOWN stays in cash.",
    )
    year = st.selectbox("Q4 year", sorted(data.stage1_backtest_daily.year.unique(), reverse=True))
    daily = data.stage1_backtest_daily.loc[data.stage1_backtest_daily.year.eq(year)]
    summary = data.stage1_backtest_summary.loc[data.stage1_backtest_summary.year.eq(year)]
    selection = next(row for row in data.stage1_selections if int(row["year"]) == int(year))
    strategy = summary.loc[summary.strategy.eq("long_flat")].iloc[0]
    hold = summary.loc[summary.strategy.eq("buy_and_hold_aapl")].iloc[0]

    cols = st.columns(4)
    with cols[0]:
        card("Selected on Q3", f"{selection['experiment']} · {MODEL_NAMES[selection['model']]}", f"Validation BA {selection['selection_score']:.3f}")
    with cols[1]:
        card("Q4 strategy return", percent(strategy.cumulative_return), "Long when UP, otherwise cash")
    with cols[2]:
        card("Q4 buy & hold", percent(hold.cumulative_return), "Fully invested in AAPL")
    with cols[3]:
        card("Time in market", f"{selection['long_intervals']}/{selection['n_intervals']} days", "No short positions or leverage")

    _plot(equity_chart(daily))

    exact = summary[["strategy", "cumulative_return", "max_drawdown", "sharpe"]].copy()
    exact["strategy"] = exact.strategy.map({"long_flat": "Selected model", "buy_and_hold_aapl": "Buy & hold AAPL"})
    st.dataframe(
        exact.rename(
            columns={
                "strategy": "Strategy",
                "cumulative_return": "Cumulative return",
                "max_drawdown": "Max drawdown",
                "sharpe": "Sharpe",
            }
        ).style.format({"Cumulative return": "{:+.2%}", "Max drawdown": "{:+.2%}", "Sharpe": "{:.2f}"}),
        width="stretch",
        hide_index=True,
    )
    callout(
        "<strong>Research limitation:</strong> this is a frictionless same-close-to-next-close diagnostic. Close-based features and same-close execution make it optimistic; transaction costs, slippage, and an extra execution lag are not included.",
        "warning",
    )


def stage2(data: DashboardData) -> None:
    hero(
        "Stage 2 · LLM factor discovery",
        "Readable ideas, hard validation gates",
        "Claude and a random-control proposer received the same candidate budget. Formulas were validated, scored across four Q3 quarters, checked for redundancy, and only then allowed into the frozen Q4 comparison.",
    )
    scoreboard = data.stage2_scoreboard
    claude = scoreboard.loc[scoreboard.proposer.eq("claude")]
    random = scoreboard.loc[scoreboard.proposer.eq("random")]
    noise_pass = int(
        ((data.noise_null.years_helped >= 3) & (data.noise_null.auc_gain >= 0.01)).sum()
    )
    cols = st.columns(4)
    with cols[0]:
        card("Claude candidates", str(len(claude)), "8 rounds × 6 proposals")
    with cols[1]:
        card("Claude kept", f"{int(claude.status.eq('kept').sum())}/{len(claude)}", "Passed consistency, gain, and correlation gates")
    with cols[2]:
        card("Mean valid-candidate gain", f"{claude.auc_gain.mean():+.4f}", f"Random control {random.auc_gain.mean():+.4f}")
    with cols[3]:
        card("Strict noise passes", f"{noise_pass}/{len(data.noise_null)}", "Empirical guard against chance discoveries")

    st.subheader("Candidate scoreboard")
    proposer_col, status_col, round_col = st.columns(3)
    proposer_choice = proposer_col.multiselect(
        "Proposer", ["claude", "random"], default=["claude", "random"], format_func=lambda x: "Claude" if x == "claude" else "Random control"
    )
    status_choice = status_col.multiselect(
        "Status", sorted(scoreboard.status.unique()), default=sorted(scoreboard.status.unique()), format_func=str.title
    )
    round_choice = round_col.slider("Rounds", 1, int(scoreboard["round"].max()), (1, int(scoreboard["round"].max())))
    filtered = scoreboard.loc[
        scoreboard.proposer.isin(proposer_choice)
        & scoreboard.status.isin(status_choice)
        & scoreboard["round"].between(*round_choice)
    ]
    if filtered.empty:
        st.info("No candidates match these filters.")
    else:
        _plot(candidate_scatter(filtered))
        candidate_table = filtered[
            ["proposer", "round", "name", "status", "auc_gain", "years_helped", "max_corr", "reason"]
        ].sort_values(["status", "auc_gain"], ascending=[True, False])
        st.dataframe(
            candidate_table,
            width="stretch",
            hide_index=True,
            column_config={
                "auc_gain": st.column_config.NumberColumn("Mean AUC gain", format="%+.4f"),
                "max_corr": st.column_config.NumberColumn("Max |correlation|", format="%.3f"),
                "years_helped": st.column_config.NumberColumn("Years helped", format="%d/4"),
            },
        )

    st.divider()
    st.subheader("Inspect a candidate")
    factor_names = sorted(scoreboard.name.unique())
    default_factor = "nvda_range_vs_aapl_range" if "nvda_range_vs_aapl_range" in factor_names else factor_names[0]
    selected_factor = st.selectbox("Candidate", factor_names, index=factor_names.index(default_factor))
    candidate = scoreboard.loc[scoreboard.name.eq(selected_factor)].iloc[0]
    callout(
        f"<strong>{candidate['status'].upper()}</strong> · {candidate['reason']}",
        "success" if candidate.status == "kept" else "info",
    )
    detail_left, detail_right = st.columns([1.35, 1])
    with detail_left:
        st.markdown("**Hypothesis**")
        st.write(candidate.hypothesis)
        st.markdown("**Formula**")
        st.code(candidate.formula, language=None)
    with detail_right:
        st.metric("Mean Q3 AUC gain", f"{candidate.auc_gain:+.4f}")
        st.metric("Years helped", f"{int(candidate.years_helped)}/4")
        st.metric("Closest-factor correlation", f"{candidate.max_corr:.3f}")
    candidate_years = data.stage2_scores_by_year.loc[
        data.stage2_scores_by_year.factor.eq(selected_factor)
        & data.stage2_scores_by_year.proposer.eq(candidate.proposer)
    ]
    if candidate_years.empty:
        st.info("This formula was rejected before scoring, so it has no per-year validation results.")
    else:
        _plot(factor_year_bars(data.stage2_scores_by_year, selected_factor, candidate.proposer))

    st.divider()
    st.subheader("Frozen Q4 evaluation")
    model_col, metric_col = st.columns(2)
    final_model = model_col.selectbox(
        "Final-test model", ["logistic_regression", "random_forest"], format_func=lambda value: MODEL_NAMES[value]
    )
    final_metrics = {
        "Balanced accuracy": "balanced_accuracy",
        "ROC AUC": "auc",
        "Strategy return": "strategy_return",
    }
    final_label = metric_col.selectbox("Final-test metric", list(final_metrics))
    _plot(final_test_bars(data.stage2_final_metrics, final_model, final_metrics[final_label], final_label))
    callout(
        "The Q4 panel is read-only by design. Candidate formulas, thresholds, and keep decisions were fixed before these outcomes were generated.",
        "warning",
    )

    equity_col, year_col, proposer_col = st.columns(3)
    show_equity = equity_col.toggle("Show final-test equity curve", value=False)
    final_year = year_col.selectbox("Equity year", [2025, 2024, 2023, 2022], disabled=not show_equity)
    final_proposer = proposer_col.selectbox(
        "Added factor", ["claude", "random"], format_func=lambda x: "Claude" if x == "claude" else "Random control", disabled=not show_equity
    )
    if show_equity:
        equity = data.stage2_final_equity.loc[
            data.stage2_final_equity.year.eq(final_year)
            & data.stage2_final_equity.model.eq(final_model)
            & data.stage2_final_equity.proposer.eq(final_proposer)
            & data.stage2_final_equity.features.eq("stage1_plus_llm")
        ]
        _plot(equity_chart(equity, strategy_label=f"Stage 1 + {final_proposer.title()} factor"))


def methodology(data: DashboardData) -> None:
    hero(
        "Methods · Reproducibility",
        "How to read—and reproduce—the dashboard",
        "The interface is a presentation layer over committed files. It does not download prices, call an LLM, train models, or choose configurations when a visitor opens it.",
    )
    callout(
        f"Pinned source commit: <span class='mono'>{data.manifest['source_commit']}</span><br>Canonical input: <span class='mono'>{data.manifest['canonical_data']}</span>",
        "info",
    )

    st.subheader("Chronological protocol")
    cols = st.columns(3)
    protocol = [
        ("Q1–Q2", "Train", "Fit scaling and model parameters. Rolling features only use information available by day t."),
        ("Q3", "Validate", "Compare Stage 1 configurations and score Stage 2 factors. All keep rules are applied here."),
        ("Q4", "Test once", "Report classification and long/flat outcomes. No tuning, threshold choice, or factor selection."),
    ]
    for col, (period, label, text) in zip(cols, protocol):
        with col:
            st.markdown(
                f'<div class="method-step"><div class="eyebrow">{period}</div><strong>{label}</strong>{text}</div>',
                unsafe_allow_html=True,
            )

    st.subheader("Stage 1 ablations")
    ablations = pd.DataFrame(
        [
            {"Experiment": "A", "Inputs": "AAPL only", "Feature count": len(FEATURE_SETS["A"])},
            {"Experiment": "B", "Inputs": "AAPL + QQQ", "Feature count": len(FEATURE_SETS["B"])},
            {"Experiment": "C", "Inputs": "AAPL + QQQ + MSFT + NVDA", "Feature count": len(FEATURE_SETS["C"])},
        ]
    )
    st.dataframe(ablations, width="stretch", hide_index=True)

    left, right = st.columns(2)
    with left:
        st.markdown("**Models and seeds**")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Model": "Logistic Regression", "Seed(s)": "42", "Reporting": "Single deterministic run"},
                    {"Model": "Random Forest", "Seed(s)": "0, 1, 2", "Reporting": "Mean ± sample SD"},
                    {"Model": "Baselines", "Seed(s)": "—", "Reporting": "One deterministic run"},
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    with right:
        rule = data.manifest["stage2"]["selection_rule"]
        st.markdown("**Stage 2 keep rule**")
        st.dataframe(
            pd.DataFrame(
                [
                    {"Gate": "Consistency", "Requirement": f"Helps at least {rule['minimum_years_helped']}/4 Q3 years"},
                    {"Gate": "Effect size", "Requirement": f"Mean AUC gain ≥ {rule['minimum_mean_auc_gain']:.2f}"},
                    {"Gate": "Novelty", "Requirement": f"Maximum |correlation| < {rule['maximum_absolute_correlation']:.1f}"},
                ]
            ),
            width="stretch",
            hide_index=True,
        )

    st.subheader("Important limitations")
    st.markdown(
        """
        - Each quarter contains only about 63 trading observations, so estimates are noisy.
        - Multiple candidate testing creates false-discovery risk; the noise calibration measures but does not eliminate it.
        - A single kept Claude factor is not broad evidence that LLM factors generalize.
        - Backtests omit costs, slippage, and a realistic next-session execution delay.
        - The output is research evidence, not investment advice or a live trading signal.
        """
    )

    st.subheader("Download the evidence")
    cols = st.columns(3)
    files = [
        ("Stage 1 metrics", data.stage1, "stage1_comparison.csv"),
        ("Stage 2 scoreboard", data.stage2_scoreboard, "stage2_scoreboard.csv"),
        ("Stage 2 final metrics", data.stage2_final_metrics, "stage2_final_metrics.csv"),
    ]
    for col, (label, frame, filename) in zip(cols, files):
        with col:
            st.download_button(label, frame.to_csv(index=False), filename, "text/csv", width="stretch")
    with st.expander("Artifact manifest"):
        st.code(json.dumps(data.manifest, indent=2), language="json")
