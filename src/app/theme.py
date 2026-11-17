"""Look of the Factor Lab app: colours, page styles, chart defaults and small HTML building blocks.

Author: Abu Wakkas Bastob (capstone PG-S2-36, supervised by A/Prof Lingqiao Liu).
"""

from __future__ import annotations

from html import escape

import plotly.graph_objects as go
import streamlit as st

# Colours. Series colours are a colour-blind-checked order (blue, orange, aqua); up/down always carry a
# symbol and a word as well, never colour alone.
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a897f"
SURFACE, CARD, LINE, GRID, AXIS = "#fcfcfb", "#ffffff", "rgba(11,11,11,0.10)", "#e9e8e2", "#c3c2b7"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8ae"
UP, UP_TEXT, DOWN, DOWN_TEXT = "#0ca30c", "#006300", "#d03b3b", "#a82525"
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"       # one family for every chart (measured and drawn alike)
POOL_COLOUR = {"llm": BLUE, "base": ORANGE, "templates": AQUA, "random": GREY}

CSS = f"""
<style>
.block-container {{ padding-top: 3.4rem; padding-bottom: 3rem; max-width: 1240px; }}
h1 {{ font-size: 1.9rem !important; letter-spacing: -0.02em; margin-bottom: 0.1rem !important; }}
h2 {{ font-size: 1.25rem !important; letter-spacing: -0.01em; margin-top: 1.6rem !important; }}
h3 {{ font-size: 1.02rem !important; }}
.fl-sub {{ color: {INK2}; font-size: 1.0rem; margin: 0 0 1.1rem 0; max-width: 58rem; }}
.fl-card {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 14px; padding: 1.1rem 1.25rem; height: 100%; }}
.fl-eyebrow {{ color: {INK2}; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; }}
.fl-hero-row {{ display: flex; align-items: baseline; gap: 0.9rem; margin: 0.35rem 0 0.2rem 0; flex-wrap: wrap; }}
.fl-call {{ font-size: 3.3rem; font-weight: 700; letter-spacing: -0.03em; line-height: 1; }}
.fl-pct {{ font-size: 3.3rem; font-weight: 300; letter-spacing: -0.03em; line-height: 1; color: {INK}; }}
.fl-note {{ color: {INK2}; font-size: 0.93rem; line-height: 1.45; }}
.fl-up {{ color: {UP_TEXT}; }} .fl-down {{ color: {DOWN_TEXT}; }}
.fl-chip {{ display: inline-block; border-radius: 999px; padding: 0.16rem 0.62rem; font-size: 0.8rem; font-weight: 600;
           border: 1px solid {LINE}; color: {INK2}; background: #f4f3ee; margin: 0.15rem 0.3rem 0.15rem 0; white-space: nowrap; }}
.fl-chip.up {{ background: #e7f5e7; color: {UP_TEXT}; border-color: #bfe3bf; }}
.fl-chip.down {{ background: #fbe9e9; color: {DOWN_TEXT}; border-color: #f0c4c4; }}
.fl-chip.blue {{ background: #e6f0fc; color: #184f95; border-color: #b7d3f6; }}
.fl-track {{ position: relative; height: 10px; border-radius: 999px; margin: 1.5rem 0 1.7rem 0;
            background: linear-gradient(90deg, #f3c9c9 0%, #f0efec 50%, #c9e8c9 100%); }}
.fl-pin {{ position: absolute; top: -5px; width: 4px; height: 20px; border-radius: 2px; background: {INK}; transform: translateX(-2px); }}
.fl-pin.usual {{ background: {MUTED}; height: 14px; top: -2px; width: 2px; }}
.fl-pinlabel {{ position: absolute; font-size: 0.74rem; color: {INK2}; transform: translateX(-50%); white-space: nowrap; }}
.fl-tile {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 12px; padding: 0.7rem 0.9rem; }}
.fl-tile .v {{ font-size: 1.45rem; font-weight: 650; letter-spacing: -0.02em; line-height: 1.2; }}
.fl-tile .l {{ color: {INK2}; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; }}
.fl-tile .s {{ color: {INK2}; font-size: 0.84rem; }}
.fl-factor {{ border-top: 1px solid {LINE}; padding: 0.6rem 0 0.55rem 0; }}
.fl-factor:first-child {{ border-top: none; }}
.fl-factor .n {{ font-weight: 650; font-size: 0.95rem; }}
.fl-factor .m {{ color: {INK2}; font-size: 0.9rem; line-height: 1.4; margin: 0.1rem 0 0.25rem 0; }}
.fl-quote {{ border-left: 3px solid {BLUE}; padding: 0.1rem 0 0.1rem 0.9rem; color: {INK}; font-size: 0.93rem; line-height: 1.5; }}
.fl-verdict {{ background: #f4f3ee; border-radius: 12px; padding: 0.85rem 1.05rem; font-size: 0.97rem; line-height: 1.5; }}
.fl-step {{ display: flex; gap: 0.8rem; align-items: flex-start; margin-bottom: 0.7rem; }}
.fl-step .k {{ flex: none; width: 1.7rem; height: 1.7rem; border-radius: 50%; background: {INK}; color: #fff; font-weight: 700;
              font-size: 0.85rem; display: flex; align-items: center; justify-content: center; }}
code {{ font-size: 0.86em !important; }}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.2rem; }}
[data-testid="stMetricValue"] {{ font-size: 1.5rem; }}
</style>
"""


def apply() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def html(text: str) -> None:
    st.markdown(text, unsafe_allow_html=True)


def header(title: str, sub: str) -> None:
    st.title(title)
    html(f'<p class="fl-sub">{sub}</p>')


def chip(text: str, kind: str = "") -> str:
    return f'<span class="fl-chip {kind}">{escape(text)}</span>'


def tile(label: str, value: str, sub: str = "", kind: str = "") -> str:
    cls = {"up": "fl-up", "down": "fl-down"}.get(kind, "")
    return (f'<div class="fl-tile"><div class="l">{escape(label)}</div><div class="v {cls}">{escape(value)}</div>'
            f'<div class="s">{sub}</div></div>')


def tiles(items: list) -> None:
    for col, item in zip(st.columns(len(items)), items):
        with col:
            html(tile(*item))


def verdict(text: str) -> None:
    html(f'<div class="fl-verdict">{text}</div>')


def pct(x: float, digits: int = 0) -> str:
    return "n/a" if x != x else f"{x * 100:.{digits}f}%"


def signed_pct(x: float, digits: int = 2) -> str:
    return "n/a" if x != x else f"{x * 100:+.{digits}f}%"


def chart(fig: go.Figure, height: int = 320, legend: bool = True, key: str | None = None) -> None:
    """One look for every chart: thin marks, quiet grid, hover on, no toolbar."""
    right = (fig.layout.meta or {}).get("right", 12)
    fig.update_layout(
        height=height, margin=dict(l=8, r=right, t=14, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=13, color=INK2),
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=INK2), title_text=""),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=AXIS, font=dict(color=INK, size=13)), bargap=0.35)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS, tickfont=dict(color=INK2), title_font=dict(color=INK2))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS, tickfont=dict(color=INK2), title_font=dict(color=INK2), automargin=True)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)


def reference_line(fig: go.Figure, value: float, text: str, vertical: bool = False) -> None:
    """A dotted reference line. Horizontal lines are labelled in the right margin so the label never covers a bar."""
    line = dict(color=INK2, width=1, dash="dot")
    if vertical:
        fig.add_vline(x=value, line=line, annotation_text=text, annotation_position="top", annotation_font=dict(color=INK2, size=12))
        return
    fig.add_hline(y=value, line=line)
    fig.add_annotation(xref="paper", x=1.0, xanchor="left", xshift=6, y=value, yref="y", text=text.replace(" ", "<br>", 1) if len(text) > 12 else text,
                       showarrow=False, align="left", font=dict(color=INK2, size=12))
    fig.update_layout(meta={"right": 92})
