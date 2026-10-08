"""Charts for Stage 1 multi-year evaluation, backtesting and factor importance."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"     # strategy / above chance / helps
ORANGE = "#eb6834"   # buy-and-hold benchmark
RED = "#e34948"      # below chance / hurts
NEUTRAL = "#f0efec"

ROW_ORDER = [
    ("baseline", "majority_class", "Always up (majority)"),
    ("baseline", "previous_direction", "Tomorrow = today"),
    ("A", "logistic_regression", "A  AAPL only · LR"),
    ("A", "random_forest", "A  AAPL only · RF"),
    ("B", "logistic_regression", "B  + QQQ · LR"),
    ("B", "random_forest", "B  + QQQ · RF"),
    ("C", "logistic_regression", "C  + QQQ, MSFT, NVDA · LR"),
    ("C", "random_forest", "C  + QQQ, MSFT, NVDA · RF"),
]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_accuracy_by_year(comparison, split="test", metric="balanced_accuracy_mean"):
    """Heatmap: rows = baselines and model/feature-set configs, columns = years.

    Blue cells beat chance (0.5), red cells are worse than chance, gray is 0.5.
    """
    data = comparison[comparison.split.eq(split)]
    years = sorted(data.year.unique())
    grid = np.array([[data.loc[data.experiment.eq(e) & data.model.eq(m) & data.year.eq(y), metric].iloc[0]
                      for y in years] for e, m, _ in ROW_ORDER])
    cmap = LinearSegmentedColormap.from_list("div", [RED, NEUTRAL, BLUE])
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    ax.imshow(grid, cmap=cmap, norm=TwoSlopeNorm(vcenter=0.5, vmin=0.35, vmax=0.65), aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=9, color=TEXT)
    ax.set_xticks(range(len(years)), [str(y) for y in years])
    ax.set_yticks(range(len(ROW_ORDER)), [label for *_, label in ROW_ORDER])
    ax.set_xticks(np.arange(-0.5, len(years)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ROW_ORDER)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0, colors=MUTED, labelsize=9)
    ax.axhline(1.5, color=TEXT, linewidth=1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"{split.capitalize()} balanced accuracy by year (Q4 of each year)\n"
                 "blue = better than a coin flip (0.5), red = worse", loc="left", fontsize=11, color=TEXT)
    fig.tight_layout()
    return fig


def plot_equity_curves(daily):
    """Small multiples: growth of $1 for the long/flat strategy vs buy-and-hold, one panel per year."""
    years = sorted(daily.year.unique())
    fig, axes = plt.subplots(1, len(years), figsize=(3.1 * len(years), 3.6), sharey=True, facecolor=SURFACE)
    for ax, year in zip(np.atleast_1d(axes), years):
        d = daily[daily.year.eq(year)].sort_values("target_date")
        x = pd.DatetimeIndex([d.Date.iloc[0], *d.target_date])
        strat = np.concatenate(([1.0], d.strategy_equity.to_numpy()))
        hold = np.concatenate(([1.0], d.buy_hold_equity.to_numpy()))
        _style(ax)
        ax.axhline(1.0, color=GRID, linewidth=1)
        ax.plot(x, hold, color=ORANGE, linewidth=2)
        ax.plot(x, strat, color=BLUE, linewidth=2)
        ax.set_title(f"Q4 {year}\nModel {strat[-1] - 1:+.1%}  ·  Hold {hold[-1] - 1:+.1%}", loc="left", fontsize=10, color=TEXT)
        ax.set_xticks([x[0], x[-1]], [x[0].strftime("%d %b"), x[-1].strftime("%d %b")])
    np.atleast_1d(axes)[0].set_ylabel("Value of $1", color=MUTED, fontsize=9)
    handles = [plt.Line2D([], [], color=BLUE, linewidth=2), plt.Line2D([], [], color=ORANGE, linewidth=2)]
    fig.legend(handles, ["Model (long when it predicts up, else cash)", "Buy & hold AAPL"],
               loc="upper right", frameon=False, fontsize=9, ncol=2)
    fig.suptitle("Backtest: does trading on the predictions beat simply holding Apple?",
                 x=0.01, ha="left", fontsize=11, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


def plot_factor_importance(summary):
    """Two panels (LR, RF): mean drop in validation AUC when each factor is shuffled, across years.

    Bars right of zero (blue) mean the model relied on the factor; left of zero (red)
    means shuffling it did not hurt, i.e. it carried no useful signal on validation.
    The label shows in how many of the years the factor helped.
    """
    models = [("logistic_regression", "Logistic Regression"), ("random_forest", "Random Forest")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 7), facecolor=SURFACE)
    n_years = None
    for ax, (key, title) in zip(axes, models):
        d = summary[summary.model.eq(key)].sort_values("mean_importance")
        n_years = n_years or int(d.positive_years.max())
        colors = [BLUE if v > 0 else RED for v in d.mean_importance]
        _style(ax)
        ax.barh(d.factor, d.mean_importance, color=colors, height=0.7)
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        span = d.mean_importance.abs().max()
        for y, (v, k) in enumerate(zip(d.mean_importance, d.positive_years)):
            ax.text(v + (span * 0.03 if v >= 0 else -span * 0.03), y, f"{k}/4",
                    va="center", ha="left" if v >= 0 else "right", fontsize=8, color=MUTED)
        ax.set_xlim(-span * 1.3, span * 1.3)
        ax.set_title(title, loc="left", fontsize=10, color=TEXT)
        ax.set_xlabel("Mean drop in validation AUC when shuffled", color=MUTED, fontsize=9)
    fig.suptitle("Factor importance (permutation, feature set C, average of 2022-2025)\n"
                 "x/4 = number of years the factor helped", x=0.01, ha="left", fontsize=11, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
