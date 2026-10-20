"""The report card Claude sees for every factor it has tried (Liu, 20 Oct).

For each factor:
  - the formula
  - the change in validation AUC in each year (2022-2025) and the average
  - whether it helped Logistic Regression and/or Random Forest
  - how similar it is to the closest existing factor
  - kept, promising or rejected, and the reason
  - a short automatic "why" line written from the numbers
  - Claude's own one-line lesson about it (written in the following round)
  - its results in the team trial and pair trials (when it had them)

Everything here only formats numbers that were already measured on the
validation quarters (Q3). Nothing from the Q4 test quarters is ever shown.
"""

from __future__ import annotations

import math

YEARS = (2022, 2023, 2024, 2025)


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def year_gains(rec) -> dict:
    """{2022: gain, ...} from a scoreboard record (keys gain_2022 ... gain_2025)."""
    out = {}
    for y in YEARS:
        g = _num(rec.get(f"gain_{y}"))
        if g is not None:
            out[y] = g
    return out


def _years_text(gains: dict) -> str:
    if not gains:
        return "no yearly results"
    up = [str(y) for y, g in gains.items() if g > 0]
    down = [str(y) for y, g in gains.items() if g <= 0]
    if not down:
        return "helped in all 4 years"
    if not up:
        return "did not help in any year"
    worst = min(gains, key=gains.get)
    return f"helped in {', '.join(up)} but not in {', '.join(down)} (worst {worst}: {gains[worst]:+.3f})"


def _models_text(lr, rf) -> str:
    if lr is None or rf is None:
        return ""
    if lr > 0 and rf > 0:
        return "helped both models"
    if lr > 0:
        return "helped Logistic Regression only"
    if rf > 0:
        return "helped Random Forest only"
    return "helped neither model"


def why_line(rec, min_gain: float = 0.01) -> str:
    """One plain sentence explaining the result, written only from the measured numbers."""
    if rec.get("status") == "invalid":
        return f"could not be used: {rec.get('reason', '')}"
    parts = []
    gains = year_gains(rec)
    parts.append(_years_text(gains))
    m = _models_text(_num(rec.get("auc_gain_lr")), _num(rec.get("auc_gain_rf")))
    if m:
        parts.append(m)
    avg = _num(rec.get("auc_gain"))
    if avg is not None and 0 < avg < min_gain:
        parts.append(f"average gain {avg:+.3f} is too small to tell apart from luck")
    corr, closest = _num(rec.get("max_corr")), rec.get("closest", "")
    if corr is not None:
        if corr >= 0.9:
            parts.append(f"near-copy of {closest} ({corr:.2f})")
        elif corr >= 0.7:
            parts.append(f"quite similar to {closest} ({corr:.2f}), so little new information")
        elif corr < 0.3:
            parts.append("new information (not similar to any existing factor)")
    team = _num(rec.get("team_gain"))
    if team is not None and avg is not None and rec.get("team_size", 0):
        if avg > 0 >= team:
            parts.append("helped alone but added nothing on top of the selected team")
        elif team > 0 >= avg:
            parts.append("weak alone but helped on top of the selected team")
    if rec.get("best_pair_partner"):
        pg = _num(rec.get("best_pair_gain"))
        syn = _num(rec.get("best_pair_synergy"))
        if pg is not None and syn is not None and syn > 0:
            parts.append(f"worked better together with {rec['best_pair_partner']} "
                         f"(pair {pg:+.3f}, {syn:+.3f} more than the better one alone)")
    return "; ".join(parts)


def report_card(rec, min_gain: float = 0.01) -> str:
    """A few lines about one factor, as shown to Claude."""
    status = str(rec.get("status", "")).upper()
    role = rec.get("role") or "-"
    mode = str(rec.get("mode")) + (f" (parent: {rec['parent']})" if rec.get("parent") else "")
    head = (f"[R{rec.get('round')}] {rec.get('name')} | {mode} | role: {role} | {status}"
            + (f" ({rec['reason']})" if rec.get("reason") else ""))
    lines = [head, f"  formula: {rec.get('formula')}"]
    if status != "INVALID":
        gains = year_gains(rec)
        years = ", ".join(f"{y} {g:+.3f}" for y, g in gains.items())
        avg, lr, rf = (_num(rec.get(k)) for k in ("auc_gain", "auc_gain_lr", "auc_gain_rf"))
        corr = _num(rec.get("max_corr"))
        perf = f"  alone (added to the 22): {years}"
        if avg is not None:
            perf += f" | avg {avg:+.3f}"
        if lr is not None and rf is not None:
            perf += f" | LR {lr:+.3f}, RF {rf:+.3f}"
        if corr is not None:
            perf += f" | similarity {corr:.2f} ({rec.get('closest', '')})"
        lines.append(perf)
        team = _num(rec.get("team_gain"))
        if team is not None and rec.get("team_size", 0):
            lines.append(f"  on top of the selected team ({rec['team_size']} factors): avg {team:+.3f}, "
                         f"helped {rec.get('team_years_helped', 0)}/4 years")
        if rec.get("best_pair_partner"):
            syn = _num(rec.get("best_pair_synergy"))
            lines.append(f"  best pair: with {rec['best_pair_partner']}: avg {_num(rec.get('best_pair_gain')):+.3f}, "
                         f"helped {rec.get('best_pair_years', 0)}/4 years"
                         + (f", {syn:+.3f} compared with the better one alone" if syn is not None else ""))
    lines.append(f"  why: {why_line(rec, min_gain)}")
    if rec.get("lesson"):
        lines.append(f"  your lesson: {rec['lesson']}")
    return "\n".join(lines)


def one_liner(rec) -> str:
    avg = _num(rec.get("auc_gain"))
    avg_txt = f"{avg:+.3f}" if avg is not None else "n/a"
    return (f"[R{rec.get('round')}] {rec.get('name')} | {str(rec.get('status', '')).upper()} | avg {avg_txt} | "
            f"{why_line(rec)} | {rec.get('formula')}")


def lab_notebook(board: list, full_cards: int = 24) -> str:
    """Everything tried so far. Full report cards for the most useful entries, one line for the rest.

    Order: kept factors, then promising ones, then the latest round, then the best of the rest.
    """
    if not board:
        return "(empty: this is the first round)"
    last_round = max(int(r.get("round", 0)) for r in board)
    rank = {"kept": 0, "promising": 1}

    def key(r):
        avg = _num(r.get("auc_gain"))
        return (rank.get(r.get("status"), 2), 0 if int(r.get("round", 0)) == last_round else 1,
                -(avg if avg is not None else -9))

    ordered = sorted(board, key=key)
    full = ordered[:full_cards]
    rest = ordered[full_cards:]
    text = "\n\n".join(report_card(r) for r in full)
    if rest:
        text += "\n\nOLDER OR WEAKER ATTEMPTS (one line each):\n" + "\n".join(one_liner(r) for r in rest)
    return text
