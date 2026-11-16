"""Try a formula: write a factor in the project's formula language and test it the same way."""

from __future__ import annotations

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import theme as T
from ..engine import ROLE_LABEL
from ..state import Context
from src.stage2.formula import FIELDS, OPERATORS, TICKERS, WINDOWS, FormulaError

EXAMPLES = {
    "NVIDIA's range minus Apple's range (the first factor the search kept)":
        "subtract(safe_divide(subtract(NVDA_High, NVDA_Low), NVDA_Close), safe_divide(subtract(AAPL_High, AAPL_Low), AAPL_Close))",
    "Market turbulence over the last month": "rolling_std(pct_change(QQQ_Close, 1), 20)",
    "Apple's distance from its 60-day high": "subtract(safe_divide(AAPL_Close, rolling_max(AAPL_High, 60)), 1)",
    "Apple's week against the Nasdaq-100's week": "subtract(pct_change(AAPL_Close, 5), pct_change(QQQ_Close, 5))",
    "Unusual Apple volume": "zscore(AAPL_Volume, 20)",
}


def _result(ctx: Context, formula: str) -> None:
    lab = ctx.lab
    try:
        out = lab.try_formula(formula)
    except FormulaError as exc:
        st.error(f"This formula cannot be used: {exc}", icon=":material/error:")
        return
    except Exception as exc:  # anything unexpected must not break the page
        st.error(f"This formula could not be computed ({type(exc).__name__}).", icon=":material/error:")
        return
    stats, info = out["stats"], out["info"]
    direction = "up" if stats["edge"] > 0 else "down"
    near = out["nearest"].iloc[0]
    T.tiles([("Edge, 2022 to 2025", f"{stats['edge'] * 100:+.1f} points", f"higher values go with {direction} days"),
             ("On the sealed quarters", f"{stats['edge_q4'] * 100:+.1f} points", "October to December, 2022 to 2025"),
             ("On fresh 2026 days", f"{stats['edge_2026'] * 100:+.1f} points", "never used in the research"),
             ("Closest known factor", f"{near['similarity']:.2f}", f"similarity to {escape(near['name'])}")])
    st.write("")
    years = [c for c in stats.index if c.startswith("y2")]
    same = sum((stats[c] > 0) == (stats["edge"] > 0) for c in years if c <= "y2025")
    copy = " It is nearly a copy of an existing factor (similarity 0.9 or more), so it adds little that is new." if near["similarity"] >= 0.9 else ""
    weak = abs(stats["edge"]) < 0.02
    window = f"longest window {info['max_window']} days" if info["max_window"] else "today's values only"
    T.verdict(f"Uses {', '.join(info['inputs'])}; {window}. "
              f"Its direction was the same in {same} of {sum(c <= 'y2025' for c in years)} research years. "
              + ("An edge under 2 points is what random numbers often show, so treat this as no evidence." if weak else
                 "An edge of 2 points or more is uncommon for random numbers, but one formula tried by hand can still be luck.") + copy)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Edge by year**")
        vals = [stats[c] * 100 for c in years]
        fig = go.Figure(go.Bar(x=[c[1:] + (" (fresh)" if c == "y2026" else "") for c in years], y=vals, marker=dict(color=[T.UP if v > 0 else T.DOWN for v in vals], cornerradius=4),
                               text=[f"{v:+.1f}" for v in vals], textposition="outside", textfont=dict(color=T.INK2),
                               hovertemplate="%{x}: %{y:+.1f} AUC points<extra></extra>"))
        fig.update_yaxes(title_text="AUC points above 50", zeroline=True, zerolinecolor=T.AXIS)
        fig.update_xaxes(type="category")
        T.chart(fig, height=280, legend=False, key="try_years")
    with right:
        st.markdown("**Next-day up rate by level of the factor**")
        q = lab.quintiles(out["series"])
        fig = go.Figure(go.Bar(x=["Lowest fifth", "Low", "Middle", "High", "Highest fifth"][:len(q)], y=q["up_rate"], customdata=q["days"],
                               marker=dict(color=T.BLUE, cornerradius=4), text=[T.pct(v) for v in q["up_rate"]], textposition="outside",
                               textfont=dict(color=T.INK2), hovertemplate="%{x}: up on %{y:.0%} of %{customdata} days<extra></extra>"))
        T.reference_line(fig, float(lab.up[lab.index >= "2022-01-01"].mean()), "all days")
        fig.update_yaxes(range=[0.3, 0.75], tickformat=".0%", title_text="share of days AAPL closed up next day")
        T.chart(fig, height=280, legend=False, key="try_fifths")

    st.markdown("**Most similar factors already in the lab**")
    nearest = out["nearest"]
    st.dataframe(pd.DataFrame({"Factor": nearest["name"], "Proposed by": nearest["source"],
                               "Describes": nearest["role"].map(ROLE_LABEL).fillna(nearest["role"]),
                               "Similarity": nearest["similarity"], "Meaning": nearest["meaning"]}),
                 hide_index=True, width="stretch",
                 column_config={"Similarity": st.column_config.ProgressColumn("Similarity", min_value=0, max_value=1, format="%.2f")})


def render(ctx: Context) -> None:
    T.header("Try a formula",
             "Write your own factor in the same small formula language the LLMs use and test it the same way. "
             "The language can only look backwards in time and cannot run code, so a formula cannot cheat.")
    pick = st.selectbox("Start from an example", ["(write my own)"] + list(EXAMPLES), key="try_example")
    if pick != "(write my own)" and st.session_state.get("try_loaded") != pick:
        st.session_state["try_formula"], st.session_state["try_loaded"] = EXAMPLES[pick], pick
    st.session_state.setdefault("try_formula", next(iter(EXAMPLES.values())))
    formula = st.text_area("Formula", key="try_formula", height=90,
                           help="Nested function calls on the inputs below. Every window ends on the day itself.")
    with st.expander("What can I write?"):
        st.markdown(f"**Inputs** ({len(TICKERS) * len(FIELDS)}): a ticker and a field joined by an underscore, for example `AAPL_Close`, `NVDA_Volume`. "
                    f"Tickers: {', '.join(f'`{t}`' for t in TICKERS)}. Fields: {', '.join(f'`{f}`' for f in FIELDS)}.")
        st.markdown(f"**Windows** (the number n, in trading days): {', '.join(str(w) for w in WINDOWS)}.")
        ops = pd.DataFrame([{"Operator": f"{name}({'a, b' if n == 2 else 'x'}{', n' if windowed else ''})", "Meaning": text}
                            for name, (n, windowed, text) in OPERATORS.items()])
        st.dataframe(ops, hide_index=True, width="stretch")
    if formula.strip():
        _result(ctx, formula.strip())
