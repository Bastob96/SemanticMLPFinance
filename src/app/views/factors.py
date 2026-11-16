"""Factors: browse every factor, read what it means, and see whether the data agrees with the LLM."""

from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .. import theme as T
from ..engine import ROLE_LABEL
from ..state import Context, factor_stats

ROUND_LABEL = {0: "", 1: "Blind", 2: "After feedback"}
ARROW = {1.0: "▲ up", -1.0: "▼ down", 0.0: "none"}


def catalogue_table(ctx: Context) -> pd.DataFrame:
    stats = factor_stats(ctx.lab)
    t = ctx.lab.cat.join(stats, on="key")
    t["llm_sign"] = t["stated"].map({"up": 1.0, "down": -1.0})
    t["data_sign"] = np.sign(t["edge"])
    t["match"] = np.where(t["llm_sign"].isna(), "", np.where(t["llm_sign"] == t["data_sign"], "✓", "✗"))
    return t


def _detail(ctx: Context, row: pd.Series) -> None:
    lab = ctx.lab
    series = lab.values[row["key"]]
    chips = T.chip(row["source"]) + (T.chip(ROUND_LABEL[row["round"]], "blue") if row["round"] else "") + T.chip(ROLE_LABEL.get(row["role"], row["role"]))
    T.html(f'<div class="fl-card"><div class="fl-eyebrow">Selected factor</div>'
           f'<h3 style="margin:0.2rem 0 0.3rem 0">{escape(row["name"])}</h3><div>{chips}</div>'
           f'<p class="fl-note" style="margin-top:0.5rem">{escape(row["meaning"])}</p></div>')
    st.code(row["formula"], language="python", wrap_lines=True)

    data_word = "up" if row["data_sign"] > 0 else "down"
    if row["stated"]:
        agree = row["llm_sign"] == row["data_sign"]
        text = (f"The LLM said a higher value makes an up day <b>{'more' if row['stated'] == 'up' else 'less'} likely</b>. "
                f"In 2022 to 2025 a higher value went with <b>{data_word}</b> days, so the data "
                f"<b>{'agrees' if agree else 'disagrees'}</b> with the LLM.")
    else:
        text = f"In 2022 to 2025 a higher value went with <b>{data_word}</b> days."
    year_cols = [c for c in row.index if c.startswith("y2")]
    same = sum(np.sign(row[c]) == row["data_sign"] for c in year_cols if c <= "y2025")
    T.verdict(text + f" The direction was the same in {same} of {sum(c <= 'y2025' for c in year_cols)} of those years. "
              f"Alone, the factor is worth {abs(row['edge']) * 100:.1f} AUC points (50 = nothing).")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Edge by year**")
        vals = [row[c] * 100 for c in year_cols]
        fig = go.Figure(go.Bar(x=[c[1:] + (" (fresh)" if c == "y2026" else "") for c in year_cols], y=vals, marker=dict(color=[T.UP if v > 0 else T.DOWN for v in vals], cornerradius=4),
                               text=[f"{v:+.1f}" for v in vals], textposition="outside", textfont=dict(color=T.INK2),
                               hovertemplate="%{x}: %{y:+.1f} AUC points<extra></extra>"))
        fig.update_yaxes(title_text="AUC points above 50", zeroline=True, zerolinecolor=T.AXIS)
        fig.update_xaxes(type="category")
        T.chart(fig, height=280, legend=False, key="factor_years")
        st.caption("Green: a higher value went with up days that year. Red: with down days. 2026 is fresh: it was never used in the research.")
    with right:
        st.markdown("**Next-day up rate by level of the factor**")
        q = lab.quintiles(series)
        fig = go.Figure(go.Bar(x=["Lowest fifth", "Low", "Middle", "High", "Highest fifth"][:len(q)], y=q["up_rate"], customdata=q["days"],
                               marker=dict(color=T.BLUE, cornerradius=4), text=[T.pct(v) for v in q["up_rate"]], textposition="outside",
                               textfont=dict(color=T.INK2), hovertemplate="%{x}: up on %{y:.0%} of %{customdata} days<extra></extra>"))
        T.reference_line(fig, float(lab.up[lab.index >= "2022-01-01"].mean()), "all days")
        fig.update_yaxes(range=[0.3, 0.75], tickformat=".0%", title_text="share of days AAPL closed up next day")
        T.chart(fig, height=280, legend=False, key="factor_fifths")
        st.caption("All days since 2022, split into fifths by the factor's value inside each year. A clear slope means the factor matters.")

    st.markdown("**The factor over time**")
    recent = series.loc[series.index[-1] - pd.DateOffset(years=2):]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.07)
    fig.add_trace(go.Scatter(x=recent.index, y=lab.close.loc[recent.index], mode="lines", line=dict(color=T.INK, width=1.5),
                             hovertemplate="%{x|%d %b %Y}: $%{y:,.2f}<extra>AAPL close</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=recent.index, y=recent, mode="lines", line=dict(color=T.BLUE, width=1.5),
                             hovertemplate="%{x|%d %b %Y}: %{y:.4g}<extra>factor</extra>"), row=2, col=1)
    fig.update_yaxes(title_text="AAPL close ($)", row=1, col=1)
    fig.update_yaxes(title_text="factor value", row=2, col=1)
    T.chart(fig, height=360, legend=False, key="factor_series")


def render(ctx: Context) -> None:
    T.header("Factors",
             "Every factor is a short formula on daily prices and volumes, with a meaning a person can check in a few minutes. "
             "Pick one to see what the LLM claimed and what the data says.")
    table = catalogue_table(ctx)
    sources = list(table["source"].unique())
    chosen = st.pills("Proposed by", sources, selection_mode="multi", default=[s for s in sources if s != "Templates (no LLM)"], key="f_sources")
    c2, c3, c4 = st.columns([2, 2, 4])
    with c2:
        rounds = st.selectbox("LLM round", ["All", "Blind", "After feedback"], key="f_round",
                              help="Blind: proposed before seeing any result. After feedback: proposed after seeing how the blind round's directions turned out.")
    with c3:
        check = st.selectbox("LLM's direction", ["All", "Data agrees ✓", "Data disagrees ✗"], key="f_match")
    with c4:
        search = st.text_input("Search name, meaning or formula", key="f_search", placeholder="e.g. volatility, NVDA, gap")
    view = table[table["source"].isin(chosen)]
    if rounds != "All":
        view = view[view["round"] == {"Blind": 1, "After feedback": 2}[rounds]]
    if check != "All":
        view = view[view["match"] == ("✓" if "agrees ✓" in check else "✗")]
    if search.strip():
        s = search.strip().lower()
        view = view[view["name"].str.lower().str.contains(s, regex=False) | view["meaning"].str.lower().str.contains(s, regex=False)
                    | view["formula"].str.lower().str.contains(s, regex=False)]
    if view.empty:
        st.info("No factor matches these filters.")
        return

    judged = view[view["match"] != ""]
    share = (judged["match"] == "✓").mean() if len(judged) else float("nan")
    T.tiles([("Factors shown", f"{len(view)}", f"{view['key'].nunique()} different formulas"),
             ("LLM direction confirmed by data", T.pct(share), f"of {len(judged)} factors with a stated direction · guessing gives 50%"),
             ("Strongest single factor", f"{view['edge'].abs().max() * 100:.1f} points", "AUC points above 50, on 2022 to 2025")])
    st.write("")

    view = view.reindex(view["edge"].abs().sort_values(ascending=False).index)
    shown = pd.DataFrame({
        "Factor": view["name"], "Proposed by": view["source"], "Round": view["round"].map(ROUND_LABEL),
        "Describes": view["role"].map(ROLE_LABEL).fillna(view["role"]),
        "LLM said higher means": view["llm_sign"].map(ARROW).fillna(""), "Data says higher means": view["data_sign"].map(ARROW),
        "Match": view["match"], "Edge 2022-25": view["edge"] * 100, "Edge sealed quarters": view["edge_q4"] * 100,
        "Edge 2026": view["edge_2026"] * 100, "Meaning": view["meaning"]})
    number = lambda title: st.column_config.NumberColumn(title, format="%+.1f", help="AUC points above 50 when the factor is used alone. Positive: higher values go with up days.")  # noqa: E731
    event = st.dataframe(shown, hide_index=True, width="stretch", height=330, on_select="rerun", selection_mode="single-row", key="factor_table",
                         column_config={"Edge 2022-25": number("Edge 2022-25"), "Edge sealed quarters": number("Edge sealed quarters"),
                                        "Edge 2026": number("Edge 2026"), "Meaning": st.column_config.TextColumn("Meaning", width="large")})
    picked = event.selection.rows[0] if event.selection.rows else 0
    st.caption("Click a row to open it below. Sorted by the size of the edge.")
    _detail(ctx, view.iloc[picked])
