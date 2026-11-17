"""Factor Lab: the project's interface. Live AAPL next-day call from LLM-proposed factors, with the evidence.

Run:   streamlit run app.py
Pages: Today (the call and why), Track record, Factors, Research findings, Try a formula, How it works.
The code is in src/app/ (engine.py holds every calculation and is covered by tests/test_app_*.py).

Author: Abu Wakkas Bastob (capstone PG-S2-36, supervised by A/Prof Lingqiao Liu).
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Factor Lab · AAPL next-day call", page_icon=":material/experiment:", layout="wide")

from src.app import theme  # noqa: E402
from src.app.state import sidebar  # noqa: E402
from src.app.views import factors, formula, method, research, today, track  # noqa: E402

theme.apply()
ctx = sidebar()

pages = [
    st.Page(lambda: today.render(ctx), title="Today", icon=":material/target:", default=True),
    st.Page(lambda: track.render(ctx), title="Track record", icon=":material/monitoring:", url_path="track-record"),
    st.Page(lambda: factors.render(ctx), title="Factors", icon=":material/category:", url_path="factors"),
    st.Page(lambda: research.render(ctx), title="Research findings", icon=":material/science:", url_path="research"),
    st.Page(lambda: formula.render(ctx), title="Try a formula", icon=":material/function:", url_path="try-a-formula"),
    st.Page(lambda: method.render(ctx), title="How it works", icon=":material/menu_book:", url_path="how-it-works"),
]
st.navigation(pages, position="top").run()
