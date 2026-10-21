"""Plotly figures used by the research dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

INK = "#12212f"
MUTED = "#64748b"
BLUE = "#2563eb"
TEAL = "#0f9f8f"
AMBER = "#f59e0b"
RED = "#dc5a5a"
GRID = "#e7edf3"
SURFACE = "#ffffff"

CONFIG_LABELS = {
    ("baseline", "majority_class"): "Majority baseline",
    ("baseline", "previous_direction"): "Previous direction",
    ("A", "logistic_regression"): "A · AAPL · LR",
    ("A", "random_forest"): "A · AAPL · RF",
    ("B", "logistic_regression"): "B · + QQQ · LR",
    ("B", "random_forest"): "B · + QQQ · RF",
    ("C", "logistic_regression"): "C · + peers · LR",
    ("C", "random_forest"): "C · + peers · RF",
}
CONFIG_ORDER = list(CONFIG_LABELS)


def _finish(fig: go.Figure, *, height: int = 420, y_title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=16, r=16, t=46, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=SURFACE,
        font=dict(family="Inter, ui-sans-serif, system-ui", color=INK, size=12),
        hoverlabel=dict(bgcolor=INK, font_color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont_color=MUTED)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, tickfont_color=MUTED, title=y_title)
    return fig


def stage1_heatmap(frame: pd.DataFrame, split: str, metric: str, title: str) -> go.Figure:
    data = frame.loc[frame["split"].eq(split)]
    years = sorted(data["year"].unique())
    z, labels = [], []
    for experiment, model in CONFIG_ORDER:
        rows = data.loc[data.experiment.eq(experiment) & data.model.eq(model)].set_index("year")
        z.append([rows.loc[year, metric] for year in years])
        labels.append(CONFIG_LABELS[(experiment, model)])
    values = np.asarray(z, dtype=float)
    center = 0.5 if metric != "f1_mean" else float(np.nanmedian(values))
    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=[str(year) for year in years],
            y=labels,
            zmid=center,
            colorscale=[[0, "#f6c5c5"], [0.5, "#f2f5f8"], [1, "#a9d8d2"]],
            text=np.vectorize(lambda value: f"{value:.3f}")(values),
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}: %{z:.3f}<extra></extra>",
            colorbar=dict(title=title, thickness=12),
        )
    )
    fig.update_layout(title=f"{title} by year", yaxis_autorange="reversed")
    return _finish(fig, height=455)


def stage1_metric_bars(frame: pd.DataFrame, year: int, split: str, metric: str, title: str) -> go.Figure:
    data = frame.loc[frame.year.eq(year) & frame.split.eq(split)].copy()
    data["configuration"] = [CONFIG_LABELS[(e, m)] for e, m in zip(data.experiment, data.model)]
    data["error"] = data.get(metric.replace("_mean", "_std"), pd.Series(index=data.index, dtype=float)).fillna(0)
    data["kind"] = np.where(data.experiment.eq("baseline"), "Baseline", "ML model")
    order = [CONFIG_LABELS[key] for key in CONFIG_ORDER]
    fig = px.bar(
        data,
        x="configuration",
        y=metric,
        error_y="error",
        color="kind",
        category_orders={"configuration": order},
        color_discrete_map={"Baseline": AMBER, "ML model": BLUE},
        labels={metric: title, "configuration": "", "kind": "Type"},
        text_auto=".3f",
    )
    if metric in {"balanced_accuracy_mean", "auc_mean", "accuracy_mean"}:
        fig.add_hline(y=0.5, line_dash="dot", line_color=MUTED, annotation_text="0.50 reference")
    fig.update_layout(title=f"{year} {split} comparison")
    fig.update_traces(textposition="outside", cliponaxis=False)
    return _finish(fig, height=445, y_title=title)


def importance_bars(frame: pd.DataFrame, model: str, top_n: int) -> go.Figure:
    data = frame.loc[frame.model.eq(model)].copy()
    data = data.reindex(data.mean_importance.abs().sort_values(ascending=False).index).head(top_n)
    data = data.sort_values("mean_importance")
    data["direction"] = np.where(data.mean_importance >= 0, "Helpful", "Not helpful")
    fig = px.bar(
        data,
        x="mean_importance",
        y="factor",
        orientation="h",
        color="direction",
        color_discrete_map={"Helpful": TEAL, "Not helpful": RED},
        custom_data=["positive_years"],
        labels={"mean_importance": "Mean validation AUC drop when shuffled", "factor": "", "direction": ""},
    )
    fig.update_traces(
        hovertemplate="%{y}<br>Mean importance: %{x:+.4f}<br>Helpful years: %{customdata[0]}/4<extra></extra>"
    )
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    fig.update_layout(title="Which Stage 1 inputs carried repeatable validation signal?")
    return _finish(fig, height=max(390, 28 * top_n + 120))


def equity_chart(frame: pd.DataFrame, *, strategy_label: str = "Selected model") -> go.Figure:
    data = frame.sort_values("target_date")
    start = pd.to_datetime(data["Date"].iloc[0])
    dates = pd.concat([pd.Series([start]), data["target_date"].reset_index(drop=True)], ignore_index=True)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=np.r_[1.0, data.strategy_equity.to_numpy()],
            mode="lines",
            name=strategy_label,
            line=dict(color=BLUE, width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=np.r_[1.0, data.buy_hold_equity.to_numpy()],
            mode="lines",
            name="Buy & hold AAPL",
            line=dict(color=AMBER, width=2.5),
        )
    )
    fig.add_hline(y=1, line_color=GRID)
    fig.update_layout(title="Q4 growth of $1", hovermode="x unified")
    return _finish(fig, height=430, y_title="Portfolio value")


def candidate_scatter(frame: pd.DataFrame) -> go.Figure:
    data = frame.copy()
    data["Proposer"] = data.proposer.map({"claude": "Claude", "random": "Random control"})
    data["Status"] = data.status.str.title()
    data["size"] = data.importance.abs().fillna(0) + 0.002
    fig = px.scatter(
        data,
        x="auc_gain",
        y="years_helped",
        color="Proposer",
        symbol="Status",
        size="size",
        size_max=14,
        hover_name="name",
        hover_data={"round": True, "importance": ":+.4f", "size": False},
        color_discrete_map={"Claude": BLUE, "Random control": AMBER},
        labels={"auc_gain": "Mean Q3 AUC gain", "years_helped": "Years helped (of 4)"},
    )
    fig.add_vline(x=0.01, line_dash="dot", line_color=MUTED, annotation_text="+0.01 keep threshold")
    fig.add_hline(y=3, line_dash="dot", line_color=MUTED)
    fig.update_layout(title="Every candidate is judged on the same four validation quarters")
    fig.update_yaxes(dtick=1, range=[-0.25, 4.35])
    return _finish(fig, height=440)


def factor_year_bars(frame: pd.DataFrame, factor: str, proposer: str) -> go.Figure:
    data = frame.loc[frame.factor.eq(factor) & frame.proposer.eq(proposer)].copy()
    data["direction"] = np.where(data.auc_gain > 0, "Helped", "Hurt")
    fig = px.bar(
        data,
        x="year",
        y="auc_gain",
        color="direction",
        color_discrete_map={"Helped": TEAL, "Hurt": RED},
        text_auto="+.4f",
        labels={"auc_gain": "Q3 AUC gain", "year": "Validation year", "direction": ""},
    )
    fig.add_hline(y=0, line_color=MUTED)
    fig.update_layout(title=f"{factor}: gain is not uniform across market years")
    return _finish(fig, height=380, y_title="AUC gain vs 22 features")


def final_test_bars(frame: pd.DataFrame, model: str, metric: str, title: str) -> go.Figure:
    baseline = frame.loc[
        frame.model.eq(model) & frame.features.eq("stage1_22") & frame.proposer.eq("claude")
    ].copy()
    baseline["Feature set"] = "Stage 1 · 22 factors"
    added = frame.loc[frame.model.eq(model) & frame.features.eq("stage1_plus_llm")].copy()
    added["Feature set"] = added.proposer.map(
        {"claude": "Stage 1 + Claude factor", "random": "Stage 1 + random factor"}
    )
    data = pd.concat([baseline, added], ignore_index=True)
    fig = px.bar(
        data,
        x="year",
        y=metric,
        color="Feature set",
        barmode="group",
        color_discrete_map={
            "Stage 1 · 22 factors": "#94a3b8",
            "Stage 1 + Claude factor": BLUE,
            "Stage 1 + random factor": AMBER,
        },
        labels={metric: title, "year": "Q4 test year"},
        text_auto=".3f" if metric in {"balanced_accuracy", "auc"} else ".1%",
    )
    if metric in {"balanced_accuracy", "auc"}:
        fig.add_hline(y=0.5, line_dash="dot", line_color=MUTED)
    else:
        fig.add_hline(y=0, line_color=MUTED)
    fig.update_layout(title=f"Frozen Q4 comparison · {model.replace('_', ' ').title()}")
    fig.update_traces(textposition="outside", cliponaxis=False)
    return _finish(fig, height=430, y_title=title)
