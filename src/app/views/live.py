"""Live: watch the model work, price by price.

Market open:   real 5-minute prices refresh every 30 seconds and the call is recomputed each time
               ("tomorrow's call if the market closed now").
Market closed: replay the last finished session from its real 5-minute prices. The replay ends exactly on the
               official call for that day, which the Today page shows.

Author: Abu Wakkas Bastob (capstone PG-S2-36, supervised by A/Prof Lingqiao Liu).
"""

from __future__ import annotations

import time as clock_time
from datetime import datetime, time
from html import escape
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .. import theme as T
from ..engine import NEW_YORK, POOLS, ROLE_LABEL, next_trading_day
from ..live import Preview, bar_so_far, fetch_intraday, market_clock, simulated_session
from ..state import Context, memo

ADELAIDE = ZoneInfo("Australia/Adelaide")
REFRESH_SECONDS = 30
STEP_PAUSE = 0.15              # seconds per 5-minute bar in the replay (78 bars: about 30 seconds with drawing)


@st.cache_data(ttl=25, show_spinner=False)
def _intraday() -> tuple:
    """(5-minute prices or None, note)."""
    try:
        return fetch_intraday(), ""
    except ImportError:
        return None, "Intraday prices need the yfinance package (pip install yfinance)."
    except Exception as exc:  # no network, Yahoo unavailable, empty answer
        return None, f"Yahoo's 5-minute prices are not reachable right now ({type(exc).__name__})."


def _preview(ctx: Context, day: pd.Timestamp) -> Preview:
    return memo(ctx.lab, ("preview", ctx.pool, day), lambda: Preview(ctx.lab, ctx.members, day))


def _path(ctx: Context, day: pd.Timestamp, session: pd.DataFrame, tag: str) -> pd.DataFrame:
    return memo(ctx.lab, ("path", ctx.pool, day, tag), lambda: _preview(ctx, day).path(session))


def _countdown(target: datetime, now: datetime) -> str:
    mins = int((target - now).total_seconds() // 60)
    hours, mins = divmod(max(mins, 0), 60)
    return f"{hours} h {mins:02d} min" if hours else f"{mins} min"


def _status_bar(clock) -> None:
    now_adl = clock.now.astimezone(ADELAIDE)
    if clock.is_open:
        state = T.chip("● Market open", "up")
        nxt = f"Closes in <b>{_countdown(clock.next_time, clock.now)}</b>: then the call locks in for the next trading day."
    else:
        state = T.chip("Market closed")
        nxt = (f"Next open: <b>{clock.next_time:%a %d %b, %H:%M}</b> New York "
               f"({clock.next_time.astimezone(ADELAIDE):%a %H:%M} Adelaide), in {_countdown(clock.next_time, clock.now)}.")
    T.html(f'<div class="fl-card" style="padding:0.75rem 1.1rem">{state}'
           f'<span class="fl-note" style="margin-left:0.6rem">New York <b>{clock.now:%a %H:%M}</b> · Adelaide <b>{now_adl:%a %H:%M}</b> · {nxt}</span></div>')


def _pipeline(ctx: Context, bar: dict, prev_close: pd.Series, r: dict, label: str) -> str:
    def move(t):
        return bar[f"{t}_Close"] / prev_close[f"{t}_Close"] - 1
    up = r["lean"] == "UP"
    tilt = (r["p_up"] - r["base_rate"]) * 100
    stance = {"above": T.chip(f"{tilt:+.0f} points vs a usual day", "up"), "below": T.chip(f"{tilt:+.0f} points vs a usual day", "down")}.get(
        r["stance"], T.chip("close to a usual day"))
    word = f'<span class="fl-call {"fl-up" if up else "fl-down"}" style="font-size:2.6rem">{"▲ UP" if up else "▼ DOWN"}</span>'
    steps = [
        ("1", "Prices now", f"AAPL <b>${bar['AAPL_Close']:,.2f}</b> ({T.signed_pct(move('AAPL'))} today) · "
                            f"Nasdaq-100 {T.signed_pct(move('QQQ'))} · MSFT {T.signed_pct(move('MSFT'))} · NVDA {T.signed_pct(move('NVDA'))}"),
        ("2", f"{r['used']} LLM factors recomputed", "Each readable formula is re-run on today's prices so far and compared with its own past."),
        ("3", "One signal", f"Average push of all factors: <b>{r['signal']:+.2f}</b> (0 = a usual day, above 0 leans up)."),
        ("4", "Chance of an up day", f"<b>{T.pct(r['p_up'], 1)}</b> against a usual day's {T.pct(r['base_rate'], 1)}."),
    ]
    rows = "".join(f'<div class="fl-step"><span class="fl-chip blue" style="min-width:1.6rem;text-align:center">{n}</span>'
                   f'<div><b>{escape(t)}</b><div class="fl-note">{d}</div></div></div>' for n, t, d in steps)
    return (f'<div class="fl-card"><div class="fl-eyebrow">{label}</div>'
            f'<div class="fl-hero-row">{word}<span class="fl-pct" style="font-size:2.6rem">{T.pct(r["p_up"], 1)}</span></div>'
            f'<div style="margin-bottom:0.8rem">{stance}</div>{rows}</div>')


def _session_chart(path: pd.DataFrame, upto: int, day: pd.Timestamp, base_rate: float, key: str) -> None:
    shown = path.iloc[:upto + 1]
    start = datetime.combine(day.date(), time(9, 30), tzinfo=NEW_YORK)
    end = datetime.combine(day.date(), time(16, 0), tzinfo=NEW_YORK)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.09,
                        subplot_titles=("AAPL price", "Chance tomorrow is an up day, if the market closed then"))
    fig.add_trace(go.Scatter(x=shown["Time"], y=shown["price"], mode="lines", line=dict(color=T.INK, width=2),
                             hovertemplate="%{x|%H:%M} · $%{y:,.2f}<extra></extra>"), row=1, col=1)
    colour = [T.UP if p >= 0.5 else T.DOWN for p in shown["p_up"]]
    fig.add_trace(go.Scatter(x=shown["Time"], y=shown["p_up"], mode="lines+markers", line=dict(color=T.BLUE, width=2),
                             marker=dict(size=5, color=colour), hovertemplate="%{x|%H:%M} · %{y:.1%}<extra></extra>"), row=2, col=1)
    fig.add_hline(y=base_rate, line=dict(color=T.INK2, width=1, dash="dot"), row=2, col=1,
                  annotation_text=f"usual day {base_rate:.0%}", annotation_position="bottom right", annotation_font=dict(size=11, color=T.INK2))
    fig.add_hline(y=0.5, line=dict(color=T.AXIS, width=1), row=2, col=1)
    lo = min(path["price"].min(), shown["price"].min())
    hi = max(path["price"].max(), shown["price"].max())
    pad = (hi - lo) * 0.08 or 1
    fig.update_yaxes(range=[lo - pad, hi + pad], tickprefix="$", row=1, col=1)
    span = max(abs(path["p_up"] - 0.5).max(), abs(base_rate - 0.5)) + 0.01
    fig.update_yaxes(range=[0.5 - span, 0.5 + span], tickformat=".0%", row=2, col=1)
    fig.update_xaxes(range=[start, end], tickformat="%H:%M")
    fig.update_annotations(font=dict(size=13, color=T.INK2))
    T.chart(fig, height=430, legend=False, key=key)


def _pushes(ctx: Context, r: dict) -> str:
    contrib = r["contributions"]
    info = ctx.members.set_index("key")
    roles = contrib.groupby(info.loc[contrib.index, "role"]).sum()
    lines = []
    for role, v in roles.reindex([k for k in ROLE_LABEL if k in roles.index]).items():
        width = min(abs(v) / max(abs(roles).max(), 1e-9) * 100, 100)
        colour = T.UP if v > 0 else T.DOWN
        lines.append(f'<div style="display:flex;align-items:center;gap:0.6rem;margin:0.3rem 0">'
                     f'<div style="width:12.5rem" class="fl-note">{escape(ROLE_LABEL[role])}</div>'
                     f'<div style="flex:1;background:#f1f0eb;border-radius:4px;height:12px"><div style="width:{width:.0f}%;height:12px;'
                     f'border-radius:4px;background:{colour}"></div></div>'
                     f'<div style="width:4.5rem;text-align:right" class="fl-note">{"▲" if v > 0 else "▼"} {v:+.2f}</div></div>')
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(3)
    names = "".join(f'<div class="fl-factor"><div class="n">{"▲" if v > 0 else "▼"} {escape(info.loc[k, "name"])}</div>'
                    f'<div class="m">{escape(info.loc[k, "meaning"])}</div></div>' for k, v in top.items())
    return (f'<div class="fl-card"><div class="fl-eyebrow">What is pushing the call right now</div>{"".join(lines)}'
            f'<div class="fl-eyebrow" style="margin-top:0.9rem">Strongest single factors</div>{names}</div>')


def _frame(ctx, slots, path, session, preview, day, i, label, key):
    t = path["Time"].iloc[i]
    bar = bar_so_far(session, t)
    r = preview.at(bar)
    prev_close = preview.history.iloc[-1]
    with slots[0].container():
        _session_chart(path, i, day, preview.base_rate, key=f"{key}_chart_{i}")
    with slots[1].container():
        T.html(_pipeline(ctx, bar, prev_close, r, label.format(t=t)))
    with slots[2].container():
        T.html(_pushes(ctx, r))
    return r


def _layout():
    left, right = st.columns([7, 5], gap="large")
    with left:
        a = st.empty()
    with right:
        b = st.empty()
    c = st.empty()
    return [a, b, c]


def _live(ctx: Context) -> None:
    @st.fragment(run_every=REFRESH_SECONDS)
    def panel():
        intra, note = _intraday()
        today = pd.Timestamp(datetime.now(tz=NEW_YORK).date())
        if intra is None or today.date() not in set(intra["Session"]):
            st.warning((note or "Today's prices have not arrived yet.") + " Showing a replay instead.", icon=":material/cloud_off:")
            return
        session = intra[intra["Session"] == today.date()]
        preview = _preview(ctx, today)
        done = memo(ctx.lab, ("live_steps", ctx.pool, today), dict)       # only new 5-minute bars are computed
        for t in sorted(session["Time"].unique()):
            if t not in done:
                r = preview.at(bar_so_far(session, t))
                done[t] = {"Time": pd.Timestamp(t), "price": r["price"], "signal": r["signal"], "p_up": r["p_up"],
                           "lean": r["lean"], "stance": r["stance"]}
        path = pd.DataFrame([done[t] for t in sorted(done)])
        slots = _layout()
        _frame(ctx, slots, path, session, preview, today, len(path) - 1,
               "Tomorrow's call if the market closed now ({t:%H:%M} New York)", "live")
        st.caption(f"Updates every {REFRESH_SECONDS} seconds from Yahoo Finance 5-minute prices. "
                   f"Last update {datetime.now(tz=ADELAIDE):%H:%M:%S} Adelaide.")
    panel()


def _replay(ctx: Context) -> None:
    intra, note = _intraday()
    finished = set(ctx.lab.index.date)
    sessions = [] if intra is None else sorted(d for d in intra["Session"].unique() if d in finished)
    if sessions:
        day = pd.Timestamp(sessions[-1])
        session, tag, source = intra[intra["Session"] == sessions[-1]], "real", "real 5-minute prices from Yahoo Finance"
    else:
        day = ctx.lab.index[-1]
        session, tag = simulated_session(ctx.lab, day), "simulated"
        source = "a stand-in path built from that day's real open, high, low and close - 5-minute prices not reachable"
        if note:
            st.caption(note)
    preview = _preview(ctx, day)
    path = _path(ctx, day, session, tag)
    next_day = next_trading_day(day)
    T.html(f'<p class="fl-sub">Replaying <b>{day:%A %d %B %Y}</b> ({len(path)} five-minute steps, {source}). '
           f'At every step the model asks: if the market closed now, what would the call for {next_day:%a %d %b} be?</p>')

    play = st.button("▶ Replay this session", type="primary", key="replay_play")
    pick = st.select_slider("Or drag through the day", options=list(range(len(path))), value=len(path) - 1,
                            format_func=lambda i: f"{path['Time'].iloc[i]:%H:%M}", key="replay_pick")
    slots = _layout()
    label = "If the market closed at {t:%H:%M} New York"
    if play:
        for i in range(len(path)):
            _frame(ctx, slots, path, session, preview, day, i, label, "replay")
            clock_time.sleep(STEP_PAUSE)
        pick = len(path) - 1
    r = _frame(ctx, slots, path, session, preview, day, pick, label, "replay_end")
    if pick == len(path) - 1:
        official = ctx.lab.calls(ctx.lab.signal(ctx.members["key"])).loc[day] if day in ctx.lab.index else None
        same = official is not None and abs(official["p_up"] - r["p_up"]) < 1e-6
        T.verdict(f"<b>16:00, the session closes.</b> The last step is the official call for {next_day:%A %d %B}: "
                  f"<b>{r['lean']} at {T.pct(r['p_up'], 1)}</b>"
                  + (", the same number the Today page shows. ✓" if same and tag == "real" else
                     ". It matches the Today page exactly when real 5-minute prices are used." if tag != "real" else "."))


def render(ctx: Context) -> None:
    clock = market_clock()
    T.header("Watch the model work, live",
             f"Every few minutes of trading changes Apple's price, so the model recomputes its <b>{len(ctx.members)} readable factors</b> "
             f"({escape(POOLS[ctx.pool])}) and the chance of an up day tomorrow. When the session closes, that number becomes the official call.")
    _status_bar(clock)
    st.write("")
    if clock.is_open:
        _live(ctx)
        with st.expander("Replay the last finished session"):
            _replay(ctx)
    else:
        _replay(ctx)
    st.caption("A daily model: it predicts the next day from a finished day. The live numbers answer \"what if the market closed now\". "
               "Research demo, not financial advice.")
