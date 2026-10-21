"""Interactive, read-only presentation of SemanticMLPFinance results."""

from __future__ import annotations

import streamlit as st

from src.ui import load_dashboard_data
from src.ui.components import inject_theme
from src.ui.pages import backtests, methodology, overview, stage1, stage2

st.set_page_config(
    page_title="Semantic Factor Lab",
    page_icon="◫",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_theme()


@st.cache_data(show_spinner=False)
def _load():
    return load_dashboard_data()


try:
    dashboard = _load()
except (FileNotFoundError, ValueError, KeyError) as exc:
    st.error("The dashboard artifact bundle is unavailable or invalid.")
    st.code(str(exc))
    st.info("From the repository root, run: python experiments/export_ui_artifacts.py")
    st.stop()

with st.sidebar:
    st.markdown("## Semantic Factor Lab")
    st.caption("AAPL next-day direction research")
    page = st.radio(
        "Navigate",
        ["Overview", "Stage 1", "Backtests", "Stage 2", "Methodology"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Artifact status**")
    st.success("Frozen & reproducible", icon="✅")
    st.caption("2022–2025 · daily OHLCV · Q4 locked")
    st.caption(f"Source `{dashboard.manifest['source_commit'][:10]}`")
    st.divider()
    st.caption("Research use only · Not investment advice")

PAGES = {
    "Overview": overview,
    "Stage 1": stage1,
    "Backtests": backtests,
    "Stage 2": stage2,
    "Methodology": methodology,
}
PAGES[page](dashboard)
