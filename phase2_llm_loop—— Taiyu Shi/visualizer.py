"""Automatic experiment figures. Every PNG is saved next to the CSV/JSON it was drawn from.

Colours follow the entity, never its rank: baseline = blue, LLM-enhanced = orange,
naive = neutral grey, actual = near-black. Status colours (accepted / rejected)
are used only in the candidate-outcome chart.
"""

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

log = logging.getLogger("capstone.figures")

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLORS = {"Naive (zero return)": "#8a8985", "HGBR baseline": "#2a78d6", "LLM-enhanced HGBR": "#eb6834",
          "Actual": INK}
STATUS = {"accepted": "#0ca30c", "not accepted": "#8a8985", "invalid": "#d03b3b"}
DIVERGING = LinearSegmentedColormap.from_list("blue_grey_orange", ["#2a78d6", "#f0efec", "#eb6834"])
FOOTNOTE_SIZE = 8

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "legend.frameon": False,
})


def _save(fig, directory, stem, data, note=None):
    if note:
        # Placed just below the figure; bbox_inches="tight" extends the canvas to include it.
        fig.text(0.01, -0.01, note, fontsize=FOOTNOTE_SIZE, color=INK_2, ha="left", va="top")
    png = directory / f"{stem}.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    if isinstance(data, pd.DataFrame):
        data.to_csv(directory / f"{stem}.csv", index=False)
    else:
        (directory / f"{stem}.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    log.info("Saved figure %s", png.name)
    return png


def fig1_final_comparison(test_metrics, directory, mode_label):
    m = test_metrics["metrics"]["refit_train_validation"]
    labels = {"naive": "Naive (zero return)", "baseline": "HGBR baseline", "llm_enhanced": "LLM-enhanced HGBR"}
    rows = [{"model": labels[k], "mae": m[k]["mae"], "rmse": m[k]["rmse"],
             "price_mae_usd": m[k]["price_mae_usd"], "price_rmse_usd": m[k]["price_rmse_usd"]} for k in labels]
    data = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    for ax, metric, title in ((axes[0], "mae", "MAE"), (axes[1], "rmse", "RMSE")):
        values = data[metric] * 100
        bars = ax.bar(data["model"], values, color=[COLORS[x] for x in data["model"]], width=0.6, edgecolor=SURFACE,
                      linewidth=2)
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=3, color=INK, fontsize=9)
        ax.set_title(f"Test {title}")
        ax.set_ylabel(f"{title} of next-day log return (%)")
        ax.set_ylim(0, values.max() * 1.18)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", labelsize=9)
    period = test_metrics["test_period"]
    fig.suptitle(f"Figure 1. Final test comparison, AAPL next-day log return ({mode_label})", fontweight="bold")
    note = (f"Test period {period['first_feature_date']} to {period['last_feature_date']} ({period['rows']} days). "
            "Models refit on train + validation, evaluated once. Lower is better.")
    return _save(fig, directory, "fig1_final_model_comparison", data, note)


def fig2_iteration_performance(set_history, directory, mode_label):
    rows = []
    for s in set_history:
        evaluated = s.get("round_evaluated_metrics")
        rows.append({"stage": "Baseline" if s["round"] == 0 else f"Round {s['round']}", "round": s["round"],
                     "retained_mae": s["metrics"]["mae"], "retained_rmse": s["metrics"]["rmse"],
                     "evaluated_mae": evaluated["mae"] if evaluated else None,
                     "evaluated_rmse": evaluated["rmse"] if evaluated else None,
                     "llm_factors": ", ".join(s["llm_factors"])})
    data = pd.DataFrame(rows)
    x = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, metric, title in ((axes[0], "mae", "MAE"), (axes[1], "rmse", "RMSE")):
        ax.axhline(data.loc[0, f"retained_{metric}"] * 100, color=COLORS["HGBR baseline"], linestyle="--",
                   linewidth=1.5, label="Baseline")
        ax.plot(x, data[f"retained_{metric}"] * 100, color=COLORS["LLM-enhanced HGBR"], linewidth=2, marker="o",
                markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, label="Best retained set (end of round)")
        evaluated = data[f"evaluated_{metric}"].astype(float) * 100
        ax.scatter(x, evaluated, s=70, facecolors="none", edgecolors="#6250d6", linewidths=2, zorder=3,
                   label="Round's evaluated model (all valid candidates added)")
        for xi, value in zip(x, evaluated):
            if np.isnan(value) and xi > 0:
                ax.annotate("no valid\ncandidates", (xi, data.loc[xi, f"retained_{metric}"] * 100),
                            textcoords="offset points", xytext=(0, 12), ha="center", fontsize=8, color=INK_2)
        ax.set_xticks(x, data["stage"])
        ax.set_title(f"Validation {title}")
        ax.set_ylabel(f"{title} of next-day log return (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9,
                        title="All rounds shown as evaluated, including rounds that did not improve. Lower is better.")
    legend.get_title().set_fontsize(FOOTNOTE_SIZE)
    legend.get_title().set_color(INK_2)
    fig.suptitle(f"Figure 2. Validation performance by iteration ({mode_label})", fontweight="bold")
    return _save(fig, directory, "fig2_iteration_performance", data)


def fig3_feature_importance(importance, directory, validation_period, mode_label):
    data = importance.sort_values("importance_mean").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.38 * len(data) + 1.5)), layout="constrained")
    colors = [COLORS["LLM-enhanced HGBR"] if g == "LLM-generated" else COLORS["HGBR baseline"] for g in data["group"]]
    ax.barh(data["factor"], data["importance_mean"] * 1e4, xerr=data["importance_std"] * 1e4, color=colors,
            height=0.65, edgecolor=SURFACE, linewidth=2, error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2})
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_xlabel("Increase in validation MAE when the factor is shuffled (log return, x1e-4)")
    ax.set_title(f"Figure 3. Permutation importance, selected model ({mode_label})")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=COLORS["HGBR baseline"], label="Baseline factor"),
                       Patch(color=COLORS["LLM-enhanced HGBR"], label="LLM-generated factor")],
              loc="lower right")
    note = (f"Model fit on training rows; importance on validation {validation_period}; 30 shuffles, seed 42. "
            "Bars: mean; whiskers: std. Negative = shuffling lowered MAE. Not computed on the test set.")
    return _save(fig, directory, "fig3_feature_importance", data, note)


def fig4_actual_vs_predicted(predictions, directory, mode_label):
    data = predictions.copy()
    dates = pd.to_datetime(data["target_date"])
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    ax.plot(dates, data["actual_next_close"], color=COLORS["Actual"], linewidth=2, label="Actual AAPL close")
    ax.plot(dates, data["baseline_next_close"], color=COLORS["HGBR baseline"], linewidth=2, label="HGBR baseline")
    ax.plot(dates, data["llm_enhanced_next_close"], color=COLORS["LLM-enhanced HGBR"], linewidth=2,
            label="LLM-enhanced HGBR")
    ax.plot(dates, data["naive_next_close"], color=COLORS["Naive (zero return)"], linewidth=1.5, linestyle=":",
            label="Naive (previous close)")
    ax.set_ylabel("AAPL close (USD, adjusted)")
    ax.set_xlabel("Target date (t+1)")
    ax.set_title(f"Figure 4. Actual vs predicted AAPL closing price, final test period ({mode_label})")
    ax.legend(ncol=4, loc="upper left", fontsize=9)
    fig.autofmt_xdate()
    return _save(fig, directory, "fig4_actual_vs_predicted", data,
                 "Predicted close = Close(t) x exp(predicted log return). Models refit on train + validation.")


def fig5_correlation_heatmap(corr, llm_names, directory, train_period, mode_label):
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(7, 0.6 * n + 3), max(6, 0.55 * n + 2.5)), layout="constrained")
    image = ax.imshow(corr.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(n), corr.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(n), corr.index, fontsize=8)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        if tick.get_text() in llm_names:
            tick.set_fontweight("bold")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=7, color=INK)
    ax.grid(False)
    fig.colorbar(image, ax=ax, shrink=0.8, label="Spearman correlation")
    ax.set_title(f"Figure 5. Factor correlations, training rows ({mode_label})")
    data = corr.reset_index().rename(columns={"index": "factor"})
    return _save(fig, directory, "fig5_factor_correlation_heatmap", data,
                 f"Spearman rank correlation on training rows {train_period}. Bold labels: LLM-generated factors.")


def fig6_candidate_outcomes(candidates, directory, mode_label):
    if candidates.empty:
        log.warning("No candidates to plot for figure 6")
        return None
    data = candidates.copy()
    data["status"] = np.where(data["decision"].str.startswith("accepted"), "accepted",
                              np.where(data["decision"] == "invalid", "invalid", "not accepted"))
    data["plot_value"] = data["rel_improvement_vs_baseline"].fillna(0.0) * 100
    data["label"] = "R" + data["round"].astype(str) + "  " + data["name"]
    data = data.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.36 * len(data) + 1.5)), layout="constrained")
    ax.barh(data["label"], data["plot_value"], color=[STATUS[s] for s in data["status"]], height=0.65,
            edgecolor=SURFACE, linewidth=2)
    invalid_rows = data.index[data["status"] == "invalid"]
    ax.scatter(np.zeros(len(invalid_rows)), invalid_rows, marker="X", s=70, color=STATUS["invalid"],
               edgecolors=SURFACE, linewidths=1, zorder=3)
    for i in invalid_rows:
        ax.text(0, i, f"   invalid: {data.loc[i, 'code']}", va="center", fontsize=8, color=INK_2)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_xlabel("Validation MAE improvement when added alone to the baseline (%)")
    ax.set_title(f"Figure 6. Candidate outcomes by round ({mode_label})")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(handles=[Patch(color=STATUS["accepted"], label="accepted"),
                       Patch(color=STATUS["not accepted"], label="valid, not accepted"),
                       Line2D([], [], marker="X", linestyle="", markersize=9, color=STATUS["invalid"],
                              label="invalid (failed checks)")], loc="lower right")
    return _save(fig, directory, "fig6_candidate_outcomes", data,
                 "Acceptance is decided against the current set, so 'added alone' gains are a screening view only.")
