"""Daily evaluator scheduling + transition notifications — Phase 15.5.

Root-cause fix for a real incident. ``signal_pairs.evaluate_signal_pairs`` and
``watchlist_state.evaluate_watchlist`` are correct, tested, and already
authorized to auto-act on their deterministic capabilities (Phase 10) — but
NOTHING invoked them automatically. Three crypto pairs ran negative-expectancy
for weeks, uncaught, until an operator manually ran the evaluators by hand.

This module is the missing DAILY call site (wired into scanner's existing
daily-task mechanism — no new scheduler) plus a PER-EVENT Pushover notification
that fires whenever a daily run produces ANY transition. It changes NO evaluator
logic and NO threshold — it only adds the call and the notification.

The transitions reported are the evaluators' own decisions (mute / enable /
demote / recover): the deterministic capabilities auto-ACT on them per Phase 10,
so a reported transition is one the evaluator applied this run.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from trading_bot import signal_pairs, watchlist_state

# A notifier sends (title, message) and returns True on a confirmed send —
# dependency-injected so tests capture it; the default posts to Pushover.
Notifier = Callable[[str, str], bool]


@dataclass(frozen=True)
class EvaluatorRunResult:
    """The outcome of one daily evaluator cycle."""

    pair_transitions: int
    watchlist_transitions: int
    lines: list[str] = field(default_factory=list)
    notified: bool = False
    message: str = ""

    @property
    def total_transitions(self) -> int:
        return self.pair_transitions + self.watchlist_transitions


def _fmt_expectancy(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _pair_transition_lines(run: signal_pairs.PairRun) -> list[str]:
    """One line per applied pair transition (mute / enable), newest concerns
    first: the muted losers, then the recovered enables."""
    lines: list[str] = []
    for e in run.mutes + run.enables:
        lines.append(
            f"PAIR {e.ticker}/{e.signal_type}: {e.status} -> {e.new_status} "
            f"(expectancy {_fmt_expectancy(e.expectancy)}, n={e.closed_count}) "
            f"- {e.reason}"
        )
    return lines


def _watchlist_transition_lines(run: watchlist_state.StateRun) -> list[str]:
    """One line per applied watchlist transition (demote / recover)."""
    lines: list[str] = []
    for e in run.demotions + run.recoveries:
        lines.append(
            f"WATCHLIST {e.ticker}: {e.status} -> {e.new_status} "
            f"(expectancy {_fmt_expectancy(e.expectancy)}, n={e.closed_count}) "
            f"- {e.reason}"
        )
    return lines


def run_daily_evaluators(*, now: datetime | None = None) -> EvaluatorRunResult:
    """Run BOTH evaluators in real (non-dry-run) mode and collect any transitions.

    This is the missing daily call site. The evaluators apply their own
    transitions (subject to the Phase 10 readiness gate they already enforce);
    this function only observes and summarizes them. The notification is added in
    :func:`notify_transitions` (called by the scanner hook)."""
    pair_run = signal_pairs.evaluate_signal_pairs(dry_run=False, now=now)
    state_run = watchlist_state.evaluate_watchlist(dry_run=False, now=now)

    lines = _pair_transition_lines(pair_run) + _watchlist_transition_lines(state_run)
    n_pair = len(pair_run.mutes) + len(pair_run.enables)
    n_wl = len(state_run.demotions) + len(state_run.recoveries)

    if not lines:
        print(
            "  evaluators: daily run complete - no transitions (no notification)",
            file=sys.stderr,
        )

    return EvaluatorRunResult(
        pair_transitions=n_pair,
        watchlist_transitions=n_wl,
        lines=lines,
        message="\n".join(lines),
    )
