"""Today: the call for the next trading day, and the reasons behind it."""

from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .. import theme as T
from ..engine import POOLS, ROLE_LABEL, TILT, last_complete_day
from ..state import LLM_POOLS, Context, classifier_day

TRACK_LOW, TRACK_HIGH = 0.35, 0.70      # range of the probability bar under the call


def _position(p: float) -> float:
    return float(np.clip((p - TRACK_LOW) / (TRACK_HIGH - TRACK_LOW), 0.01, 0.99) * 100)


def _hero(ctx: Context, call) -> str:
    when = f"{call.next_day:%A %d %B %Y}"
    replay = "" if ctx.latest else T.chip("Replay", "blue")
    # the day being called has already finished in New York: either the source has not published it yet
    # (Yahoo often lists a new day with empty prices for some hours), or the prices are simply old
    behind = (last_complete_day() - call.next_day.date()).days if ctx.latest else -1
    waiting, stale = 0 <= behind <= 3, behind > 3
    if waiting:
        replay = T.chip(f"Waiting for {call.next_day:%a %d %b} prices: not published yet")
    if stale:
        replay = T.chip(f"Prices end {call.day:%d %b %Y}: not a current call", "down")
    head = f'<div class="fl-eyebrow">AAPL · call for {when} {replay}</div>'
    if np.isnan(call.signal):
        return f'<div class="fl-card">{head}<p class="fl-note">No call: not enough history on this date.</p></div>'
    up = call.lean == "UP"
    word = f'<span class="fl-call {"fl-up" if up else "fl-down"}">{"▲ UP" if up else "▼ DOWN"}</span>'
    if np.isnan(call.p_up):
        body = (f'<div class="fl-hero-row">{word}</div><p class="fl-note">The signal points {"up" if up else "down"}, but there are '
                f'not yet enough earlier calls to turn it into a probability (the first few months of the record).</p>')
        return f'<div class="fl-card">{head}{body}</div>'
    points = round((call.p_up - call.base_rate) * 100)
    if call.stance == "above":
        tilt = T.chip(f"{points} points above a usual day", "up")
    elif call.stance == "below":
        tilt = T.chip(f"{abs(points)} points below a usual day", "down")
    else:
        tilt = T.chip("Close to a usual day")
    track = (f'<div class="fl-track">'
             f'<div class="fl-pin usual" style="left:{_position(call.base_rate):.1f}%"></div>'
             f'<div class="fl-pinlabel" style="left:{_position(call.base_rate):.1f}%; top:-22px">usual day {T.pct(call.base_rate)}</div>'
             f'<div class="fl-pin" style="left:{_position(call.p_up):.1f}%"></div>'
             f'<div class="fl-pinlabel" style="left:{_position(call.p_up):.1f}%; top:18px"><b>this call {T.pct(call.p_up)}</b></div></div>')
    body = (f'<div class="fl-hero-row">{word}<span class="fl-pct">{T.pct(call.p_up)}</span></div>'
            f'<div class="fl-note">Estimated chance that AAPL closes higher on {call.next_day:%a %d %b} than its close on '
            f'{call.day:%a %d %b} (${ctx.lab.close.loc[call.day]:,.2f}).</div>{track}<div>{tilt}')
    if not np.isnan(call.outcome):
        went_up = call.outcome == 1
        right = went_up == up
        body += T.chip(f"What happened: AAPL closed {'up' if went_up else 'down'} {T.signed_pct(call.next_return)}", "up" if went_up else "down")
        body += T.chip("Call was right ✓" if right else "Call was wrong ✗")
    body += "</div>"
    if waiting:
        body += (f'<div class="fl-note" style="margin-top:0.6rem">The {call.next_day:%a %d %b} session has ended, but the price source '
                 'has not published its final prices yet. Press Refresh prices later to score this call and see the next one.</div>')
    elif ctx.latest and not stale:
        body += ('<div class="fl-note" style="margin-top:0.6rem">The next call appears once the New York session of '
                 f'{call.next_day:%a %d %b} has closed (press Refresh prices).</div>')
    return f'<div class="fl-card">{head}{body}</div>'


def _market_tiles(ctx: Context) -> None:
    prices, day = ctx.lab.prices, ctx.day
    pos = ctx.lab.index.get_loc(day)
    items = []
    for ticker, name in (("AAPL", "Apple"), ("QQQ", "Nasdaq-100 fund"), ("MSFT", "Microsoft"), ("NVDA", "NVIDIA")):
        close = prices[f"{ticker}_Close"]
        change = close.iloc[pos] / close.iloc[pos - 1] - 1
        items.append((f"{ticker} · {name}", f"${close.iloc[pos]:,.2f}",
                      f'<span class="{"fl-up" if change >= 0 else "fl-down"}">{"▲" if change >= 0 else "▼"} {T.signed_pct(change)}</span> on the day'))
    for row in (items[:2], items[2:]):
        T.tiles(row)
        st.write("")


def _factor_block(row, stated_mode: bool) -> str:
    pushes_up = row["contribution"] > 0
    pct_rank = row["percentile"]
    level = "high" if pct_rank >= 0.67 else "low" if pct_rank <= 0.33 else "middling"
    rank = ("the highest it has been" if pct_rank >= 0.995 else "the lowest it has been" if pct_rank <= 0.005
            else f"above {pct_rank:.0%} of its own past values")
    goes = "up" if row["direction"] > 0 else "down"
    basis = "The LLM stated that" if stated_mode else "On earlier days,"
    chips = T.chip(row["source"]) + T.chip(f"pushes {'▲ up' if pushes_up else '▼ down'}", "up" if pushes_up else "down")
    if row["llm"] == row["llm"] and not stated_mode:        # an LLM factor: did its stated direction match the data?
        chips += T.chip("LLM expected the same direction", "blue") if row["llm"] == row["measured"] else T.chip("LLM expected the opposite direction")
    return (f'<div class="fl-factor"><div class="n">{escape(row["name"])}</div>'
            f'<div class="m">{escape(row["meaning"])}</div>'
            f'<div class="fl-note">Now <b>{level}</b>: {rank}. '
            f'{basis} a higher value goes with <b>{goes}</b> days.</div><div>{chips}</div></div>')


def _why(ctx: Context, call) -> None:
    contrib = ctx.lab.contributions(ctx.day, ctx.members, stated=ctx.stated)
    if contrib.empty:
        return
    st.subheader("Why")
    by_role = contrib.groupby("role")["contribution"].sum().reindex(list(ROLE_LABEL)).dropna()
    strongest_up, strongest_down = by_role.idxmax(), by_role.idxmin()
    parts = []
    if by_role.max() > 0:
        parts.append(f"<b>{ROLE_LABEL[strongest_up]}</b> pushes the call up most")
    if by_role.min() < 0:
        parts.append(f"<b>{ROLE_LABEL[strongest_down]}</b> pushes it down most")
    T.html(f'<p class="fl-sub">{len(contrib)} factors each add a small push. Grouped by what they describe: {" and ".join(parts)}.</p>')

    left, right = st.columns([5, 7], gap="large")
    with left:
        fig = go.Figure(go.Bar(
            x=by_role.values, y=[ROLE_LABEL[r].replace(" (", "<br>(") for r in by_role.index], orientation="h",
            marker=dict(color=[T.UP if v > 0 else T.DOWN for v in by_role.values], cornerradius=4),
            text=["about 0" if abs(v) < 0.005 else f"{'▲' if v > 0 else '▼'} {v:+.2f}" for v in by_role.values], textposition="outside", textfont=dict(color=T.INK2),
            customdata=contrib.groupby("role").size().reindex(by_role.index).values,
            hovertemplate="%{y}<br>net push %{x:+.3f} from %{customdata} factors<extra></extra>"))
        span = max(abs(by_role.values).max() * 2.1, 0.05)
        fig.update_xaxes(range=[-span, span], title_text="net push (◀ down · up ▶)", zeroline=True, zerolinecolor=T.AXIS)
        fig.update_yaxes(autorange="reversed", automargin=True)
        T.chart(fig, height=250, legend=False, key="why_roles")
        st.caption(f"The four bars add up to the signal ({call.signal:+.2f}). A signal of 0 is a usual day.")
    with right:
        up_tab, down_tab = st.tabs(["Pushing up most", "Pushing down most"])
        with up_tab:
            top = contrib[contrib["contribution"] > 0].iloc[::-1].head(4)
            T.html("".join(_factor_block(r, ctx.stated) for _, r in top.iterrows()) or '<p class="fl-note">No factor pushes up.</p>')
        with down_tab:
            top = contrib[contrib["contribution"] < 0].head(4)
            T.html("".join(_factor_block(r, ctx.stated) for _, r in top.iterrows()) or '<p class="fl-note">No factor pushes down.</p>')


def _history(ctx: Context) -> None:
    st.subheader("How has a signal like this done before?")
    left, right = st.columns([6, 6], gap="large")
    buckets = ctx.lab.bucket_table(ctx.calls, ctx.day)
    with left:
        if buckets.empty:
            st.info("Not enough earlier calls on this date to compare with.")
        else:
            now = buckets[buckets["today"]]
            fig = go.Figure(go.Bar(
                x=buckets["label"], y=buckets["up_rate"], customdata=buckets["days"],
                marker=dict(color=[T.BLUE if t else T.GREY for t in buckets["today"]], cornerradius=4),
                text=[T.pct(v) for v in buckets["up_rate"]], textposition="outside", textfont=dict(color=T.INK2),
                hovertemplate="%{x}: AAPL closed up on %{y:.0%} of %{customdata} days<extra></extra>"))
            base = float(ctx.calls.loc[ctx.day, "base_rate"])
            T.reference_line(fig, base, f"usual {T.pct(base)}")
            fig.update_yaxes(range=[0.3, 0.75], tickformat=".0%", title_text="share of days AAPL closed up next day")
            fig.update_xaxes(title_text="earlier days, grouped by how high the signal was")
            T.chart(fig, height=300, legend=False, key="buckets")
            if len(now):
                st.caption(f"Blue marks the group this call falls in: on those {int(now['days'].iloc[0])} earlier days AAPL closed up "
                           f"{T.pct(float(now['up_rate'].iloc[0]))} of the time. Only days before this call are counted.")
    with right:
        st.markdown("**Second opinions**")
        with st.spinner("Training the classifiers on earlier days..."):
            opinion = classifier_day(ctx.lab, ctx.day)
        p_now = float(ctx.calls.loc[ctx.day, "p_up"])
        rows = [("This signal", p_now, T.BLUE),
                ("Logistic regression, 22 factors", float(opinion["p_lr"]), T.INK2),
                ("Random forest, 22 factors", float(opinion["p_rf"]), T.INK2),
                ("Logistic regression, 22 + 4 kept", float(opinion["p_lr_kept"]), T.INK2),
                ("Random forest, 22 + 4 kept", float(opinion["p_rf_kept"]), T.INK2)]
        names, values, colours = zip(*rows)
        fig = go.Figure(go.Bar(x=[v - 0.5 for v in values], y=list(names), base=0.5, orientation="h", customdata=list(values),
                               marker=dict(color=list(colours), cornerradius=4), text=[T.pct(v) for v in values], textposition="outside",
                               textfont=dict(color=T.INK2), hovertemplate="%{y}: %{customdata:.1%} chance of an up day<extra></extra>"))
        T.reference_line(fig, 0.5, "50%", vertical=True)
        reach = max(0.12, max(abs(v - 0.5) for v in values) + 0.05)
        fig.update_xaxes(range=[0.5 - reach, 0.5 + reach], tickformat=".0%", title_text="chance of an up day")
        fig.update_yaxes(autorange="reversed")
        T.chart(fig, height=300, legend=False, key="opinions")
        st.caption("The project's two classifiers, retrained on all earlier days, with the 22 existing factors and with the "
                   "4 factors the iterative LLM search kept. Which opinion has been most reliable is on the Track record page.")


def _recent(ctx: Context) -> None:
    st.subheader("Recent calls")
    lab = ctx.lab
    pos = lab.index.get_loc(ctx.day)
    window = ctx.calls.iloc[max(pos - 119, 0):pos + 1]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=window.index, y=lab.close.loc[window.index], mode="lines", line=dict(color=T.INK, width=2),
                             name="AAPL close", hovertemplate="%{x|%a %d %b %Y}: $%{y:,.2f}<extra></extra>"), row=1, col=1)
    tilt = (window["tilt"] * 100).fillna(0)
    fig.add_trace(go.Bar(x=window.index, y=tilt, marker=dict(color=[T.UP if v >= 0 else T.DOWN for v in tilt]), name="call",
                         hovertemplate="%{x|%a %d %b %Y}: %{y:+.1f} points against a usual day<extra></extra>"), row=2, col=1)
    fig.update_yaxes(title_text="AAPL close ($)", row=1, col=1)
    fig.update_yaxes(title_text="call vs usual (points)", row=2, col=1, zeroline=True, zerolinecolor=T.AXIS)
    fig.update_layout(bargap=0.15)
    T.chart(fig, height=380, legend=False, key="recent_chart")

    rows = ctx.calls.iloc[max(pos - 9, 0):pos + 1].iloc[::-1]
    table = pd.DataFrame({
        "Close of": rows.index.strftime("%a %d %b %Y"),
        "Call": ["▲ UP" if v == "UP" else "▼ DOWN" for v in rows["lean"]],
        "Chance of up": [T.pct(v) for v in rows["p_up"]],
        "Against a usual day": [("same" if round(v * 100) == 0 else f"{round(v * 100):+d} point{'s' if abs(round(v * 100)) != 1 else ''}") if v == v else "n/a" for v in rows["tilt"]],
        "AAPL next day": [T.signed_pct(v) if v == v else "not known yet" for v in rows["next_return"]],
        "Result": ["✓ right" if v == 1 else "✗ wrong" if v == 0 else "" for v in rows["hit"]]})
    st.dataframe(table, hide_index=True, width="stretch")


def render(ctx: Context) -> None:
    call = ctx.lab.call_on(ctx.calls, ctx.day)
    directions = "the directions the LLM stated" if ctx.stated else "directions measured on earlier days"
    T.header("Will Apple close higher on the next trading day?",
             f"One call per day, built from <b>{len(ctx.members)} factors</b> ({escape(POOLS[ctx.pool])}), each a plain formula with a "
             f"one-line meaning, combined using {directions}. Every number on this page uses only prices up to the close of {ctx.day:%a %d %b %Y}.")
    left, right = st.columns([7, 5], gap="large")
    with left:
        T.html(_hero(ctx, call))
    with right:
        _market_tiles(ctx)
    if ctx.stated and ctx.pool in LLM_POOLS:
        st.info("You are using the directions the LLMs stated before seeing data. The research found these unreliable "
                "(see Research findings); switch to \"Measured on past data\" in the sidebar for the app's real call.", icon=":material/info:")
    T.verdict(f"<b>How much to trust this.</b> The edge is small. A call counts as tilted only when it is at least {TILT * 100:.0f} points away "
              "from a usual day, and even then AAPL moves the other way on many days. The Track record page shows exactly how often.")
    if np.isnan(call.signal):
        return
    _why(ctx, call)
    if not np.isnan(call.p_up):
        _history(ctx)
    _recent(ctx)
