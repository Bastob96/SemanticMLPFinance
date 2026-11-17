"""How it works: the pipeline, the guarantees and the limits, in plain language."""

from __future__ import annotations

import streamlit as st

from .. import theme as T
from ..engine import MIN_CALIBRATION_DAYS, TILT, Z_START
from ..state import FIRST_SIGNAL_TEXT, Context

STEPS = [
    ("An LLM proposes factors",
     "Three LLMs (Opus 5.5, Fable 5.2, Haiku 5.5) were each asked for 100 factors that a person can understand in five minutes. "
     "Each factor is a short formula on the daily prices and volumes of Apple, Microsoft, NVIDIA and the Nasdaq-100 fund (QQQ), "
     "plus a one-line meaning."),
    ("A safe formula language runs them",
     "The LLM never writes code. It writes formulas such as <code>subtract(pct_change(AAPL_Close, 5), pct_change(QQQ_Close, 5))</code>. "
     "Every operator only looks backwards, so a factor's value on a day uses that day's close and earlier, never the future."),
    ("The data decides each factor's direction",
     "For every factor the app measures, on earlier days only, whether a higher value went with up days or down days. "
     "This is re-measured at the start of every quarter. The LLM's own guess of the direction is shown but not used, "
     "because the research found it unreliable."),
    ("The factors are combined into one signal",
     f"Each factor is compared with its own past (at least {Z_START} earlier days), turned so that higher means up, and averaged. "
     "No factor is picked for having worked: every proposed factor stays in, which is what protects the signal from luck."),
    ("The signal becomes a chance",
     f"A one-input curve, fitted on at least {MIN_CALIBRATION_DAYS} earlier calls, turns the signal into the chance of an up day. "
     f"A call is described as above or below usual only when it is {TILT * 100:.0f} points or more from the usual share of up days."),
]

CHECKS = [
    ("No look-ahead", "Cutting all later days off the price file leaves every factor value, signal and call for earlier dates unchanged. A test in the code checks this."),
    ("Past-only directions", f"Directions and the chance curve are fitted only on days whose outcome was already known. The record starts in {FIRST_SIGNAL_TEXT}, once enough history exists."),
    ("A luck reference", "The same recipe is run on pools of pure random numbers. The Track record page shows where random numbers land."),
    ("A fresh test", "January to August 2026 was never touched by any search or test in the research before it was scored, once."),
    ("Finished days only", "Live prices are used up to the last finished trading day in New York, so a half-finished day never enters a call."),
]

GLOSSARY = [
    ("Factor", "A number computed each day from prices and volumes, for example Apple's return over the last week."),
    ("Direction", "Whether a higher value of a factor goes with up days or with down days."),
    ("Signal", "The average of all factors after each is turned so that higher means up. Zero is a usual day."),
    ("AUC", "Ranking skill. Take a random up day and a random down day: AUC is how often the signal was higher on the up day. 0.50 is no skill, 1.00 is perfect."),
    ("Edge", "A single factor's AUC minus 0.50, shown in points (0.03 = 3 points)."),
    ("Sealed quarters", "October to December of 2022 to 2025. The factor searches never used these days."),
    ("Template formulas", "Formulas filled in at random from six hand-written patterns. No LLM: the control group."),
]


def render(ctx: Context) -> None:
    T.header("How it works", "From an LLM's idea to a call for tomorrow, and what stops the app from fooling itself.")
    left, right = st.columns([7, 5], gap="large")
    with left:
        st.subheader("Five steps")
        T.html("".join(f'<div class="fl-step"><div class="k">{i}</div><div><b>{title}</b><div class="fl-note">{text}</div></div></div>'
                       for i, (title, text) in enumerate(STEPS, 1)))
    with right:
        st.subheader("What keeps it honest")
        T.html("".join(f'<div class="fl-factor"><div class="n">{title}</div><div class="m">{text}</div></div>' for title, text in CHECKS))

    st.subheader("What this app is not")
    T.verdict("It is a research demonstration, not financial advice and not a trading system. The edge is small, it comes and goes "
              "between years, and after trading costs it would not pay. The project's question is whether LLMs can discover "
              "understandable factors that hold information, and how to test that without being fooled by luck.")

    st.subheader("Words used in the app")
    T.html("".join(f'<div class="fl-factor"><span class="n">{word}</span> <span class="m">{text}</span></div>' for word, text in GLOSSARY))

    st.subheader("Data and credits")
    lab = ctx.lab
    st.markdown(
        f"- **Prices:** daily open, high, low, close and volume from Yahoo Finance (adjusted), {lab.index[0]:%d %b %Y} to {lab.index[-1]:%d %b %Y}; "
        f"currently using {'live prices from Yahoo Finance' if 'live' in lab.source else 'the saved price file'}.\n"
        f"- **Factors:** {len(lab.cat)} in the catalogue ({lab.cat['key'].nunique()} different formulas): the 22 existing factors, "
        "the 4 factors kept by the iterative search, the LLM factors of the direction test, and the template control.\n"
        "- **Classifiers shown as second opinions:** logistic regression and random forest (scikit-learn) on the 22 existing factors, "
        "and on those plus the 4 factors kept by the iterative LLM search (Stage 2 loop).\n"
        "- **Project:** Automatic Discovery of Interpretable Financial Factors for Stock Price Prediction, capstone group PG-S2-36, "
        "supervised by A/Prof Lingqiao Liu. Team: Abu Wakkas Bastob (lead; evaluation, LLM factor search, this app), Taiyu Shi, "
        "Leyao Chang, Kevin Varghese, Nikhil Nakade.\n"
        "- **LLM use:** the factors were proposed by LLMs, as the project requires; an LLM coding assistant also helped write the code (see the Git history).\n"
        "- **Reproduce:** `experiments/run_direction_test.py`, `experiments/check_direction_time_order.py`, "
        "`experiments/build_app_data.py`; the app's calculations are in `src/app/engine.py`.")
