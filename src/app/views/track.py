"""Track record: every past call scored against what happened, next to the alternatives.

Author: Abu Wakkas Bastob (capstone PG-S2-36, supervised by A/Prof Lingqiao Liu).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import theme as T
from ..engine import POOLS, auc_of
from ..state import FIRST_SIGNAL_TEXT, Context, calls_for, classifier_history, random_reference


def _verdict(score: dict, period: str) -> str:
    auc, p = score["auc"], score.get("p", float("nan"))
    if p < 0.05 and auc > 0.5:
        skill = (f"the signal ranked up days above down days <b>better than chance</b> (AUC {auc:.3f}; shuffled outcomes reach this "
                 f"only {p:.1%} of the time)")
    elif auc > 0.5:
        skill = (f"the signal ranked up days slightly above down days (AUC {auc:.3f}), but <b>luck alone gives this often</b> "
                 f"(shuffled outcomes reach it {p:.0%} of the time)")
    else:
        skill = f"the signal showed <b>no skill</b> (AUC {auc:.3f}, where 0.50 is none)"
    gap = score["hit_rate"] - score["always_up"]
    call = ("about the same as" if abs(gap) < 0.015 else "better than" if gap > 0 else "worse than")
    return (f"<b>{period}, {score['days']} days.</b> Here {skill}. As a plain UP or DOWN call it was right {T.pct(score['hit_rate'])} "
            f"of the time, {call} always saying UP ({T.pct(score['always_up'])}). It is evidence that the factors hold "
            f"information, not a way to trade.")


def _alternatives(ctx: Context, mask, period: str) -> None:
    lab = ctx.lab
    rows = []
    listed = {"llm_blind": T.BLUE, "llm_feedback": T.BLUE, "base": T.ORANGE, "templates": T.AQUA}
    if ctx.stated or ctx.pool not in listed:
        colour = T.ORANGE if ctx.pool == "base" else T.AQUA if ctx.pool == "templates" else T.BLUE
        name = POOLS[ctx.pool] + (", LLM's own directions" if ctx.stated else "")
        rows.append((f"<b>{name}</b><br>(selected)", auc_of(ctx.calls["signal"][mask], lab.up[mask]), colour))
    for pool, colour in listed.items():
        chosen = pool == ctx.pool and not ctx.stated
        name = f"<b>{POOLS[pool]}</b><br>(selected)" if chosen else POOLS[pool]
        rows.append((name, auc_of(calls_for(lab, pool)["signal"][mask], lab.up[mask]), colour))
    with st.spinner("Training the classifiers on every past quarter and building the random-number reference (up to half a minute, first time only)..."):
        models, noise = classifier_history(lab), random_reference(lab)
    for name, column in (("Logistic regression, 22 factors", "p_lr"), ("Random forest, 22 factors", "p_rf"),
                         ("Logistic regression, 22 + 4 kept by the search", "p_lr_kept"), ("Random forest, 22 + 4 kept by the search", "p_rf_kept")):
        rows.append((name, auc_of(models[column][mask], lab.up[mask]), T.INK2))
    luck = np.array([auc_of(noise[c][mask], lab.up[mask]) for c in noise])
    low, high = np.quantile(luck, [0.05, 0.95])

    names, values, colours = zip(*rows)
    fig = go.Figure()
    fig.add_vrect(x0=low, x1=high, fillcolor=T.GREY, opacity=0.28, line_width=0,
                  annotation_text="random numbers<br>(9 in 10 pools)", annotation_position="bottom left", annotation_font=dict(color=T.INK2, size=12))
    fig.add_trace(go.Bar(x=[v - 0.5 for v in values], y=list(names), base=0.5, orientation="h",
                         marker=dict(color=list(colours), cornerradius=4), text=[f"{v:.3f}" for v in values],
                         textposition="outside", textfont=dict(color=T.INK2), customdata=list(values),
                         hovertemplate="%{y}<br>AUC %{customdata:.3f}<extra></extra>"))
    T.reference_line(fig, 0.5, "no skill", vertical=True)
    reach = max(0.08, max(abs(v - 0.5) for v in values) + 0.03, high - 0.5 + 0.02)
    fig.update_xaxes(range=[0.5 - reach, 0.5 + reach], title_text=f"ranking skill (AUC) on: {period.lower()}")
    fig.update_yaxes(autorange="reversed")
    T.chart(fig, height=90 + 44 * len(rows), legend=False, key="alternatives")
    st.caption("Blue: factors proposed by LLMs. Orange: the 22 existing hand-made factors. Green: template formulas with no LLM. "
               "Dark grey: the project's two classifiers, retrained each quarter on earlier days, with the 22 factors and with the "
               "4 factors the iterative LLM search kept. Every signal is built the same way (directions from earlier days only). "
               "A bar inside the light grey band is no better than random numbers treated the same way.")


def _by_year(ctx: Context) -> None:
    lab, periods = ctx.lab, ctx.lab.periods()
    years = [k for k in periods if k.isdigit()]
    base = calls_for(lab, "base")
    mine = [auc_of(ctx.calls["signal"][periods[y]], lab.up[periods[y]]) for y in years]
    theirs = [auc_of(base["signal"][periods[y]], lab.up[periods[y]]) for y in years]
    labels = [f"{y}<br>(from Jul)" if y == "2022" else y for y in years]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[v - 0.5 for v in mine], base=0.5, name="This selection", marker=dict(color=T.BLUE, cornerradius=4),
                         text=[f"{v:.2f}" for v in mine], textposition="outside", textfont=dict(color=T.INK2),
                         customdata=mine, hovertemplate="%{x}: AUC %{customdata:.3f}<extra>This selection</extra>"))
    if ctx.pool != "base":
        fig.add_trace(go.Bar(x=labels, y=[v - 0.5 for v in theirs], base=0.5, name="The 22 existing factors",
                             marker=dict(color=T.ORANGE, cornerradius=4), customdata=theirs,
                             hovertemplate="%{x}: AUC %{customdata:.3f}<extra>The 22 existing factors</extra>"))
    T.reference_line(fig, 0.5, "no skill")
    fig.update_yaxes(range=[0.38, 0.70], title_text="ranking skill (AUC)")
    fig.update_xaxes(type="category", tickangle=0)
    fig.update_traces(textfont_size=12, cliponaxis=False)
    fig.update_layout(barmode="group", bargroupgap=0.08, uniformtext=dict(minsize=11, mode="show"))
    T.chart(fig, height=300, key="by_year")


def _reliability(calls: pd.DataFrame) -> None:
    c = calls.dropna(subset=["p_up", "outcome"])
    if len(c) < 100:
        st.info("Too few days in this period for this chart.")
        return
    group = pd.qcut(c["p_up"].rank(method="first"), 5, labels=False)
    t = c.groupby(group).agg(said=("p_up", "mean"), happened=("outcome", "mean"), days=("outcome", "size"))
    lo, hi = min(t["said"].min(), t["happened"].min()) - 0.04, max(t["said"].max(), t["happened"].max()) + 0.04
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=T.AXIS, width=1, dash="dot"), name="perfect", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t["said"], y=t["happened"], mode="lines+markers", line=dict(color=T.BLUE, width=2),
                             marker=dict(size=10, color=T.BLUE, line=dict(color="#fff", width=2)), name="calls", customdata=t["days"],
                             hovertemplate="said %{x:.0%} → happened %{y:.0%} (%{customdata} days)<extra></extra>"))
    fig.update_xaxes(tickformat=".0%", title_text="chance of an up day the app gave")
    fig.update_yaxes(tickformat=".0%", title_text="share of those days AAPL closed up")
    T.chart(fig, height=300, legend=False, key="reliability")


def render(ctx: Context) -> None:
    lab = ctx.lab
    T.header("Track record",
             f"Every call since {FIRST_SIGNAL_TEXT}, scored against what AAPL did the next day. Each call used only earlier data, "
             "so this is what you would have seen had the app been running.")
    periods = lab.periods()
    period = st.segmented_control("Days", list(periods), default=list(periods)[0], key="period", label_visibility="collapsed") or list(periods)[0]
    mask = periods[period]
    score = lab.score(ctx.calls, mask)
    if score["days"] < 20 or "auc" not in score:
        st.info("Too few scored days in this period.")
        return

    def after(stance: str) -> str:
        n = score[f"days_{stance}"]
        return f"{T.pct(score[f'up_{stance}'])} up on {n} day{'s' if n != 1 else ''}" if n else "no such calls"
    T.tiles([("Days scored", f"{score['days']}", f"AAPL closed up on {T.pct(score['always_up'])}"),
             ("Ranking skill (AUC)", f"{score['auc']:.3f}", f"0.50 is none · luck reaches this {score['p']:.1%} of the time"),
             ("After 'above usual' calls", T.pct(score["up_above"]) + " up" if score["days_above"] else "none made",
              f"{after('above')} · after 'below usual' calls: {after('below')}"),
             ("UP / DOWN call right", T.pct(score["hit_rate"]), f"always saying UP: {T.pct(score['always_up'])}")])
    st.write("")
    T.verdict(_verdict(score, period))

    st.subheader("Against the alternatives")
    _alternatives(ctx, mask, period)

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Year by year")
        _by_year(ctx)
        st.caption("The skill is not steady: it is clear in some years and absent in others (2024 in particular).")
    with right:
        st.subheader("Do the stated chances hold?")
        _reliability(ctx.calls[mask])
        st.caption("Calls in the selected period, split into five equal groups by the chance the app gave. On the dotted line, "
                   "a stated chance of 55% would mean AAPL closed up on 55% of those days.")

    st.subheader("Would acting on it have paid?")
    equity = lab.equity(ctx.calls, mask)
    if len(equity) > 20:
        fig = go.Figure()
        for name, colour, width in (("Always hold AAPL", T.INK2, 1.5), ("Step aside when below usual", T.BLUE, 2)):
            fig.add_trace(go.Scatter(x=equity.index, y=equity[name], mode="lines", name=name, line=dict(color=colour, width=width),
                                     hovertemplate="%{x|%d %b %Y}: %{y:.2f}<extra>" + name + "</extra>"))
        fig.update_yaxes(title_text="value of 1 invested at the start")
        fig.update_layout(hovermode="x unified")
        T.chart(fig, height=300, key="equity")
        end = equity.iloc[-1]
        worse = end["Step aside when below usual"] < end["Always hold AAPL"]
        st.caption(f"Holding AAPL except on 'below usual' days turned 1 into {end['Step aside when below usual']:.2f}; always holding "
                   f"turned it into {end['Always hold AAPL']:.2f}. " +
                   ("Stepping aside cost money: AAPL still rose on most of those days. " if worse else "") +
                   "No trading costs are included. The project measures prediction, not profit.")

    with st.expander("All calls in this period"):
        table = ctx.calls[mask].dropna(subset=["signal"])[["signal", "p_up", "base_rate", "lean", "stance", "next_return", "hit"]].copy()
        table.index = table.index.strftime("%Y-%m-%d")
        table.columns = ["Signal", "Chance of up", "Usual day", "Call", "Against usual", "AAPL next-day return", "Call right (1) or wrong (0)"]
        st.dataframe(table.iloc[::-1], width="stretch", height=320)
        st.download_button("Download as CSV", table.to_csv().encode(), file_name="factor_lab_calls.csv", mime="text/csv")
