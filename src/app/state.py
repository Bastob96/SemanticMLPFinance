"""What every page shares: the Lab for the chosen price source, the sidebar choices, and cached results."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from .engine import (DEFAULT_POOL, FIRST_SIGNAL, POOLS, Lab, check_prices, fetch_live, load_snapshot, pool_members)

LLM_POOLS = ("llm_blind", "llm_feedback", "llm_all", "opus", "fable", "haiku")


@st.cache_resource(show_spinner="Loading prices and computing every factor...", ttl=1800)
def lab_for(mode: str) -> tuple:
    """(Lab, note). Live prices when asked for and available, else the saved file with a note saying why."""
    if mode == "live":
        try:
            raw = fetch_live()
            check_prices(raw, at_least_until=load_snapshot()["Date"].max())
            return Lab(raw, source="Yahoo Finance, live"), ""
        except Exception as exc:  # no network, Yahoo unavailable, bad data: the app must still work
            note = ("Live prices are not reachable right now, so the saved price file is shown. "
                    f"Check the internet connection and press Refresh prices. ({type(exc).__name__})")
            return Lab(load_snapshot(), source="saved price file"), note
    return Lab(load_snapshot(), source="saved price file"), ""


def memo(lab: Lab, key, build):
    if key not in lab.memo:
        lab.memo[key] = build()
    return lab.memo[key]


def members_of(lab: Lab, pool: str) -> pd.DataFrame:
    return memo(lab, ("members", pool), lambda: pool_members(lab.cat, pool))


def calls_for(lab: Lab, pool: str, stated: bool = False) -> pd.DataFrame:
    def build():
        m = members_of(lab, pool)
        fixed = dict(zip(m["key"], m["stated"].map({"up": 1.0, "down": -1.0}))) if stated else None
        return lab.calls(lab.signal(m["key"], stated=fixed))
    return memo(lab, ("calls", pool, stated), build)


def random_reference(lab: Lab) -> pd.DataFrame:
    return memo(lab, "random", lab.random_reference)


def classifier_history(lab: Lab) -> pd.DataFrame:
    return memo(lab, "classifiers", lab.model_opinions)


def classifier_day(lab: Lab, day: pd.Timestamp) -> pd.Series:
    if "classifiers" in lab.memo:
        return lab.memo["classifiers"].loc[day]
    quarter = day.to_period("Q")
    return memo(lab, ("classifiers", quarter), lambda: lab.model_opinions(day)).loc[day]


def factor_stats(lab: Lab) -> pd.DataFrame:
    return memo(lab, "stats", lab.factor_stats)


@dataclass
class Context:
    lab: Lab
    note: str
    day: pd.Timestamp          # the day whose closing prices the call is made from
    latest: bool               # True when ``day`` is the last day of prices
    pool: str
    stated: bool               # True: use the directions the LLM stated instead of the measured ones
    members: pd.DataFrame
    calls: pd.DataFrame


def sidebar() -> Context:
    with st.sidebar:
        st.markdown("### Factor Lab")
        st.caption("AAPL next-day direction from LLM-proposed, human-readable factors. Capstone PG-S2-36.")
        mode = st.radio("Prices", ["live", "saved"], horizontal=True, key="mode",
                        format_func={"live": "Live", "saved": "Saved file"}.get,
                        help="Live downloads daily prices from Yahoo Finance up to the last finished trading day. "
                             "Saved file is the project's price file (to 27 Aug 2026) and needs no internet.")
        lab, note = lab_for(mode)
        last, first = lab.index[-1], lab.index[lab.refits[0]]
        st.caption(f"Prices through **{last:%a %d %b %Y}** · {lab.source}")
        if note:
            st.warning(note, icon=":material/cloud_off:")
        if st.button("Refresh prices", width="stretch", help="Download again (live prices are otherwise kept for 30 minutes)"):
            lab_for.clear()
            st.rerun()

        st.divider()
        # stay on the newest day when the prices change (live <-> saved, or a refresh), unless an earlier day was picked
        was_latest = st.session_state.get("as_of") == st.session_state.get("_latest_day")
        if "as_of" in st.session_state and (was_latest or not (first.date() <= st.session_state["as_of"] <= last.date())):
            st.session_state["as_of"] = last.date()
        st.session_state["_latest_day"] = last.date()
        picked = st.date_input("Call made at the close of", value=last.date(), min_value=first.date(), max_value=last.date(),
                               key="as_of", help="Pick an earlier date to replay exactly what the app would have shown then. "
                                                 "Nothing after that date is used.")
        day = lab.index[lab.index <= pd.Timestamp(picked)][-1]
        if day.date() != picked:
            st.caption(f"No trading on {picked:%a %d %b}: showing {day:%a %d %b %Y}.")
        latest = day == last
        if not latest:
            st.caption(":material/history: Replay mode")

        st.divider()
        pool = st.selectbox("Factors behind the call", list(POOLS), index=list(POOLS).index(DEFAULT_POOL), format_func=POOLS.get, key="pool",
                            help="Which set of factors is combined into the signal. The default is the cleanest test: "
                                 "factors the three LLMs proposed before seeing any result.")
        stated = False
        if pool in LLM_POOLS:
            stated = st.radio("Direction of each factor", [False, True], key="stated",
                              format_func={False: "Measured on past data", True: "As the LLM stated"}.get,
                              help="A factor's direction says whether a higher value goes with up days or down days. "
                                   "Measured: taken from earlier days only, re-measured each quarter. "
                                   "As the LLM stated: the direction the LLM wrote down before seeing any data.")
        st.divider()
        st.caption("Research demo. Not financial advice.")
    return Context(lab, note, day, latest, pool, stated, members_of(lab, pool), calls_for(lab, pool, stated))


FIRST_SIGNAL_TEXT = f"{pd.Timestamp(FIRST_SIGNAL):%B %Y}"
