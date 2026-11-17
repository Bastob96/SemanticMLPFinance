"""Check that the Factor Lab app can use real, live prices from Yahoo Finance. Run it on a machine with internet.

It does exactly what the app does in "Live" mode (src.app.engine.fetch_live), then checks the result:

    1. the download works and has all four tickers up to the last finished trading day
    2. on the days it shares with the project's saved price file, daily returns agree
       (prices themselves may differ slightly: Yahoo re-adjusts old prices after each dividend)
    3. the app's calculations run on the live prices and give a call for the next trading day

Usage:  python experiments/check_live_prices.py
Prints PASS or FAIL for each check and "ALL CHECKS PASSED" at the end when everything is fine.

Author: Abu Wakkas Bastob (capstone PG-S2-36, supervised by A/Prof Lingqiao Liu).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.app.engine import (DEFAULT_POOL, Lab, check_prices, fetch_live, last_complete_day, load_snapshot,  # noqa: E402
                            pool_members)

warnings.filterwarnings("ignore")


def main(downloader=None, now=None) -> bool:
    ok = True

    def report(passed: bool, text: str) -> None:
        nonlocal ok
        ok = ok and passed
        print(f"  {'PASS' if passed else 'FAIL'}  {text}")

    print("1. Download from Yahoo Finance")
    try:
        import yfinance
        print(f"  yfinance {yfinance.__version__}")
    except ImportError:
        if downloader is None:
            print("  FAIL  yfinance is not installed. Run:  pip install yfinance")
            return False
    saved = load_snapshot()
    try:
        live = fetch_live(now=now, downloader=downloader)
        check_prices(live, at_least_until=saved["Date"].max())
    except Exception as exc:
        print(f"  FAIL  the download could not be used: {type(exc).__name__}: {exc}")
        return False
    days = live["Date"].nunique()
    report(True, f"{days} trading days for {sorted(live['Ticker'].unique())}, {live['Date'].min():%d %b %Y} to {live['Date'].max():%d %b %Y}")
    expected = last_complete_day(now)
    gap = (expected - live["Date"].max().date()).days
    report(gap <= 4, f"last day is {live['Date'].max():%a %d %b %Y}; the last finished trading day can be at most {expected:%a %d %b %Y}")
    new_days = int((live["Date"] > saved["Date"].max()).groupby(live["Ticker"]).sum().min())
    print(f"  {new_days} new trading days since the saved file ({saved['Date'].max():%d %b %Y})")

    print("2. Agreement with the saved price file on shared days")
    worst = 0.0
    for ticker in sorted(saved["Ticker"].unique()):
        a = saved[saved["Ticker"] == ticker].set_index("Date")["Close"].pct_change()
        b = live[live["Ticker"] == ticker].set_index("Date")["Close"].pct_change()
        shared = a.index.intersection(b.index)
        diff = float((a[shared] - b[shared]).abs().max())
        worst = max(worst, diff)
        report(len(shared) >= 0.99 * len(a) and diff < 0.005,
               f"{ticker}: {len(shared)} of {len(a)} saved days found; largest difference in a daily return {diff * 100:.3f} points")
    print(f"  (differences come from Yahoo re-adjusting old prices after dividends; largest here {worst * 100:.3f} points)")

    print("3. The app's calculations on live prices")
    lab = Lab(live, source="Yahoo Finance, live")
    members = pool_members(lab.cat, DEFAULT_POOL)
    calls = lab.calls(lab.signal(members["key"]))
    call = lab.call_on(calls, lab.index[-1])
    report(not lab.values.iloc[-1].isna().all(), f"{lab.values.shape[1]} factor formulas computed up to {lab.index[-1]:%a %d %b %Y}")
    report(not np.isnan(call.p_up),
           f"call for {call.next_day:%a %d %b %Y}: {call.lean}, chance of an up day {call.p_up:.0%} (usual day {call.base_rate:.0%})")
    old = Lab(saved)
    old_calls = old.calls(old.signal(members["key"]))
    shared = old_calls.index.intersection(calls.index)
    both = old_calls.loc[shared, "p_up"].notna() & calls.loc[shared, "p_up"].notna()
    drift = float((old_calls.loc[shared, "p_up"][both] - calls.loc[shared, "p_up"][both]).abs().max())
    report(drift < 0.03, f"past calls agree with the saved-file version: largest difference in the chance of an up day {drift * 100:.2f} points")

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED: send this output to Claude")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
