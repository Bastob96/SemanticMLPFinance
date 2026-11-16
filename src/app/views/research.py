"""Research findings: what the project learned, in five charts. Numbers come from the saved research runs."""

from __future__ import annotations

import json
from html import escape

import plotly.graph_objects as go
import streamlit as st

from .. import theme as T
from ..engine import APP_DATA
from ..state import Context


@st.cache_data
def load_research() -> dict:
    return json.loads((APP_DATA / "research.json").read_text())


def _finding(number: int, title: str, text: str) -> None:
    T.html(f'<div class="fl-step" style="margin-top:1.8rem"><div class="k">{number}</div><div><h3 style="margin:0">{title}</h3>'
           f'<p class="fl-note" style="margin:0.15rem 0 0 0; max-width:60rem">{text}</p></div></div>')


def _old_judge(r: dict) -> None:
    kept = r["old_judge"]["kept"]
    T.tiles([(name, f"{sum(v)} kept", "in three batches of 100: " + ", ".join(map(str, v))) for name, v in kept.items()])


def _directions(r: dict) -> None:
    models = list(r["models"])
    series = [("Blind (no results seen)", "blind_q4", T.ORANGE), ("After seeing earlier results", "feedback_q4", T.BLUE)]
    fig = go.Figure()
    for name, key, colour in series:
        vals = [r["models"][m][key]["rate"] for m in models]
        ns = [r["models"][m][key]["n"] for m in models]
        fig.add_trace(go.Bar(x=models, y=vals, name=name, marker=dict(color=colour, cornerradius=4), text=[T.pct(v) for v in vals],
                             textposition="outside", textfont=dict(color=T.INK2), customdata=ns,
                             hovertemplate="%{x}: right for %{y:.0%} of %{customdata} factors<extra>" + name + "</extra>"))
    data = [r["models"][m]["data_alone_q4"] for m in models]
    fig.add_trace(go.Scatter(x=models, y=data, mode="markers", name="Reading the direction off earlier data, no LLM",
                             marker=dict(symbol="diamond", size=12, color=T.INK, line=dict(color="#fff", width=2)),
                             hovertemplate="%{x}: %{y:.0%}<extra>data alone</extra>"))
    T.reference_line(fig, 0.5, "guessing 50%")
    fig.update_yaxes(range=[0, 0.95], tickformat=".0%", title_text="stated direction right on the sealed quarters")
    fig.update_layout(barmode="group", bargroupgap=0.08)
    T.chart(fig, height=340, key="r_directions")


def _pools(r: dict) -> None:
    rows = [x for x in r["time_order"] if x["directions"] == "past only"]
    names, values, colours = [], [], []
    for x in rows:
        pool = x["pool"]
        if pool.startswith("random"):
            continue
        if "round 1" in pool:
            names.append({"opus-5-5": "Opus 5.5", "fable-5-2": "Fable 5.2", "haiku-5-5": "Haiku 5.5"}[pool.split(",")[0]] + ", blind factors")
            colours.append(T.BLUE)
        elif pool.startswith("the 22"):
            names.append("The 22 existing factors")
            colours.append(T.ORANGE)
        elif pool.startswith("templates"):
            names.append(f"Template formulas, no LLM (set {pool[-2]})")
            colours.append(T.AQUA)
        else:
            continue
        values.append(x["auc_q4"])
    random = next(x for x in rows if x["pool"].startswith("random"))
    order = sorted(range(len(names)), key=lambda i: (colours[i] != T.BLUE, colours[i] != T.ORANGE, names[i]))
    names, values, colours = [names[i] for i in order], [values[i] for i in order], [colours[i] for i in order]
    fig = go.Figure(go.Bar(x=[v - 0.5 for v in values], y=names, base=0.5, orientation="h", marker=dict(color=colours, cornerradius=4),
                           text=[f"{v:.3f}" for v in values], textposition="outside", textfont=dict(color=T.INK2), customdata=values,
                           hovertemplate="%{y}: AUC %{customdata:.3f}<extra></extra>"))
    T.reference_line(fig, 0.5, "no skill", vertical=True)
    fig.add_vline(x=random["random_95th"], line=dict(color=T.GREY, width=2), annotation_text="95% of random-number pools fall below",
                  annotation_position="bottom right", annotation_font=dict(color=T.INK2, size=12))
    fig.update_xaxes(range=[0.44, 0.66], title_text="ranking skill (AUC) on the sealed quarters, directions from earlier days only")
    fig.update_yaxes(autorange="reversed")
    T.chart(fig, height=330, legend=False, key="r_pools")


def _flip(r: dict) -> None:
    rows = {(x["pool"], x["directions"]): x["auc_q4"] for x in r["time_order"]}
    models = [("Opus 5.5", "opus-5-5"), ("Fable 5.2", "fable-5-2"), ("Haiku 5.5", "haiku-5-5")]
    stated = [rows[(f"{tag}, round 1 (blind)", "stated")] for _, tag in models]
    measured = [rows[(f"{tag}, round 1 (blind)", "past only")] for _, tag in models]
    fig = go.Figure()
    for name, vals, colour in (("Directions as the LLM stated", stated, T.ORANGE), ("Directions measured on earlier data", measured, T.BLUE)):
        fig.add_trace(go.Bar(x=[m for m, _ in models], y=[v - 0.5 for v in vals], base=0.5, name=name, marker=dict(color=colour, cornerradius=4),
                             text=[f"{v:.2f}" for v in vals], textposition="outside", textfont=dict(color=T.INK2), customdata=vals,
                             hovertemplate="%{x}: AUC %{customdata:.3f}<extra>" + name + "</extra>"))
    T.reference_line(fig, 0.5, "no skill")
    fig.update_yaxes(range=[0.38, 0.66], title_text="ranking skill (AUC), same blind factors")
    fig.update_layout(barmode="group", bargroupgap=0.08)
    T.chart(fig, height=320, key="r_flip")


def render(ctx: Context) -> None:
    r = load_research()
    T.header("Research findings",
             "Can an LLM discover factors that predict AAPL's next-day direction and that a person can understand? "
             "Five results from the project, each with the number behind it, then the limits.")

    _finding(1, "Keeping the factors that pass a validation test rewards luck",
             "The first judge kept a factor when it improved the model on validation data. Given 100 candidates at a time, it kept "
             "none of the LLM's factors and eleven columns of pure random numbers. So \"kept factors\" proves nothing, and a different test was needed.")
    _old_judge(r)

    _finding(2, "LLM-proposed factors hold real information; template formulas do not",
             "New test, with no model to fool: each pool of factors becomes one signal, every direction is taken from earlier days only, "
             "and the signal is scored on sealed quarters the search never used. All three LLM pools clear the random-number line. "
             "Templates filled in at random do not.")
    _pools(r)

    _finding(3, "The LLM proposes well but cannot say which way a factor points",
             "Each LLM wrote down in advance whether a higher value makes an up day more or less likely. Using its own directions, "
             "the strongest model's signal is worse than chance. The same factors work once earlier data sets the direction. "
             "That is how this app is built: the LLM proposes, the data decides the direction.")
    _flip(r)

    _finding(4, "Shown how its first guesses turned out, the LLM corrects itself",
             "After seeing the blind round's results on the search days, the two larger models stated the right direction for about "
             "three in four new factors on quarters they had never seen. They reached, but did not beat, simply reading the direction off earlier data (diamonds).")
    _directions(r)

    _finding(5, "Three models, working separately, wrote down the same lesson",
             "Each model summarised what it learned in its own words. They agree: turbulence tends to be followed by up days, "
             "strength over one to three months tends to reverse, and Apple's own excess moves tend to reverse.")
    for col, (model, m) in zip(st.columns(3, gap="medium"), r["models"].items()):
        with col:
            T.html(f'<div class="fl-eyebrow" style="margin-bottom:0.4rem">{escape(model)}</div><div class="fl-quote">{escape(m["lesson"])}</div>')

    st.subheader("Limits")
    fresh = {x["pool"]: x for x in r["fresh_2026"] if x["directions"].startswith("measured")}
    llm = [v["auc_2026"] for k, v in fresh.items() if "round 1" in k]
    past = {x["pool"]: x for x in r["time_order"] if x["directions"] == "past only"}
    gains = [x["p_vs_22"] for k, x in past.items() if "round" in k]
    T.verdict(
        f"<b>Not shown to beat the 22 existing factors.</b> LLM pools score higher on the sealed quarters, but the gap could be luck "
        f"(best p-value {min(gains):.2f}).<br>"
        f"<b>Smaller on fresh data.</b> On January to August 2026, which no search or test had touched, LLM pools score "
        f"{min(llm):.2f} to {max(llm):.2f} and the 22 existing factors {fresh['the 22 existing factors']['auc_2026']:.2f}; "
        f"with 163 days none of these is statistically clear.<br>"
        f"<b>The edge is mostly known structure</b> (volatility and reversal), and it was absent in 2024.<br>"
        f"<b>The LLMs may have read about these years.</b> Their training data can include 2022 to 2025 market history, "
        f"which is one more reason the app lets the data, not the LLM, set each direction.")
    st.caption(f"Source: {r['built_from']}. The Track record page recomputes the live versions of these numbers from prices.")
