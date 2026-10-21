"""Small Streamlit presentation helpers."""

from __future__ import annotations

import html

import streamlit as st


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500&display=swap');
        :root { --ink:#12212f; --muted:#64748b; --blue:#2563eb; --teal:#0f9f8f; --line:#e7edf3; --soft:#f5f8fb; }
        html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
        .stApp { background:linear-gradient(180deg,#f7fafc 0,#ffffff 240px); }
        [data-testid="stSidebar"] { background:#102331; }
        [data-testid="stSidebar"] * { color:#e8f0f5; }
        [data-testid="stSidebar"] [data-baseweb="radio"] label { padding:.35rem .55rem; border-radius:.55rem; }
        [data-testid="stSidebar"] [aria-checked="true"] { background:#1d3b4f; }
        .block-container { max-width:1220px; padding-top:2rem; padding-bottom:3rem; }
        h1,h2,h3 { letter-spacing:-.025em; color:var(--ink); }
        h1 { font-size:2.35rem !important; }
        .eyebrow { font-family:'IBM Plex Mono',monospace; color:var(--blue); font-size:.76rem; letter-spacing:.09em; text-transform:uppercase; font-weight:600; }
        .hero-copy { color:var(--muted); max-width:760px; font-size:1.05rem; line-height:1.65; margin:.25rem 0 1.1rem; }
        .research-card { background:#fff; border:1px solid var(--line); border-radius:14px; padding:1rem 1.1rem; box-shadow:0 8px 24px rgba(18,33,47,.045); min-height:124px; }
        .research-card .label { color:var(--muted); font-size:.79rem; font-weight:600; text-transform:uppercase; letter-spacing:.045em; }
        .research-card .value { font-size:1.7rem; font-weight:700; line-height:1.2; margin:.35rem 0; }
        .research-card .note { color:var(--muted); font-size:.84rem; line-height:1.35; }
        .callout { border-left:4px solid var(--blue); background:#eef5ff; border-radius:0 10px 10px 0; padding:.85rem 1rem; color:#23405b; margin:.7rem 0 1rem; }
        .warning-callout { border-left-color:#f59e0b; background:#fff8e9; }
        .success-callout { border-left-color:#0f9f8f; background:#ebfaf7; }
        .method-step { background:#fff; border:1px solid var(--line); border-radius:12px; padding:1rem; height:100%; }
        .method-step strong { display:block; margin-bottom:.3rem; }
        .mono { font-family:'IBM Plex Mono',monospace; font-size:.82rem; word-break:break-word; }
        [data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:12px; padding:.85rem 1rem; }
        [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:10px; overflow:hidden; }
        .stPlotlyChart { background:#fff; border:1px solid var(--line); border-radius:14px; padding:.25rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(eyebrow: str, title: str, copy: str) -> None:
    st.markdown(f'<div class="eyebrow">{html.escape(eyebrow)}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="hero-copy">{html.escape(copy)}</div>', unsafe_allow_html=True)


def card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""<div class="research-card"><div class="label">{html.escape(label)}</div>
        <div class="value">{html.escape(value)}</div><div class="note">{html.escape(note)}</div></div>""",
        unsafe_allow_html=True,
    )


def callout(text: str, kind: str = "info") -> None:
    css = {"info": "", "warning": " warning-callout", "success": " success-callout"}[kind]
    st.markdown(f'<div class="callout{css}">{text}</div>', unsafe_allow_html=True)


def percent(value: float, digits: int = 1) -> str:
    return f"{value:+.{digits}%}"


def decimal(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"
