"""Self-optimization — degradation detection + feature evaluation (Phase 9).

Reads everything accumulated in Phases 3-7 and surfaces two things:

* **Degradation** — is the bot's own live performance decaying? (recent window
  vs an older baseline window, on expectancy).
* **Feature evaluation** — does any stored advisory context (sentiment, the
  indicator families, the risk verdicts) actually predict resolved outcomes?

THIS LAYER FLAGS AND REPORTS ONLY. It never auto-tunes a threshold, mutes a
pair, benches a ticker, or changes any behavior — acting on a finding is a
deliberate, later, sample-gated step.

MULTIPLE-COMPARISONS DISCIPLINE. Every statistic carries its sample size ``n``.
Anything below :data:`config.SO_MIN_SAMPLE` is labeled "insufficient sample - not
actionable", never hidden and never acted on, because testing many features
against a still-small outcome set manufactures false positives. Degradation
requires BOTH a minimum sample AND a meaningful-change threshold to fire, so
normal variance (a 20-trade cold streak) does not trip an alarm.

The win_rate / expectancy math is REUSED verbatim from
:func:`trading_bot.signal_pairs._windowed_stats` so the definitions are identical
to the live gating subsystems — no divergent math.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, NamedTuple

from trading_bot import db
from trading_bot.signal_pairs import _windowed_stats


class Stat(NamedTuple):
    """Resolved-trade stats for a slice. ``n`` is the resolved count
    (wins + losses); win_rate / expectancy are ``None`` for an empty slice."""

    n: int
    win_rate: float | None       # wins / (wins + losses) * 100
    expectancy: float | None     # mean resolved pnl %


def stat_for(rows: Sequence[Mapping[str, Any]]) -> Stat:
    """Canonical resolved stats for a set of trade rows.

    Delegates to :func:`trading_bot.signal_pairs._windowed_stats` so the
    win_rate (wins / (wins + losses)) and expectancy (mean resolved pnl %)
    definitions are IDENTICAL to the gating subsystems. ``n`` is the resolved
    count (wins + losses); expired/open rows never reach here.
    """
    pairs: list[tuple[str, float | None]] = [
        (str(r["outcome"]), r["pnl_pct"]) for r in rows
    ]
    n, win_rate, expectancy = _windowed_stats(pairs)
    return Stat(n, win_rate, expectancy)


def resolved_context_rows(
    since: datetime | None = None, until: datetime | None = None
) -> list[dict[str, Any]]:
    """All resolved ACTIVE trades (with stored context) in ``[since, until)``.

    Thin pass-through to the Phase 9 db query — win/loss only, track_mode
    active, ordered by ``closed_at``. ``since=None`` means no lower bound (every
    resolved active trade, used by feature evaluation).
    """
    return db.get_resolved_context_outcomes(since=since, until=until)
