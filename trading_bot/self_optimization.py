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
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, NamedTuple

from trading_bot import config, db
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


# ══════════════════════════════════════════════════════════════════════════════
# DEGRADATION DETECTION — recent window vs older baseline, on expectancy
# ══════════════════════════════════════════════════════════════════════════════

# Tolerance for the meaningful-delta threshold comparison, so a drop landing
# exactly on the threshold is not flipped to "stable" by float rounding.
_DELTA_EPSILON = 1e-9


@dataclass(frozen=True)
class DegradationFinding:
    """One scope's recent-vs-baseline expectancy comparison.

    ``verdict`` is one of ``degraded`` / ``stable`` / ``insufficient_sample`` /
    ``insufficient_baseline``. ``delta`` is ``baseline - recent`` (positive == a
    drop) when both are measurable, else ``None``. Degradation fires ONLY when
    the recent sample clears the floor AND the drop clears the meaningful delta.
    """

    scope: str                          # "overall" | "setup:<type>" | "ticker:<sym>"
    baseline_expectancy: float | None
    baseline_n: int
    recent_expectancy: float | None
    recent_n: int
    delta: float | None
    verdict: str
    note: str


def _degradation_finding(
    scope: str,
    baseline_rows: Sequence[Mapping[str, Any]],
    recent_rows: Sequence[Mapping[str, Any]],
    *,
    min_sample: int,
    meaningful_delta: float,
) -> DegradationFinding:
    base = stat_for(baseline_rows)
    rec = stat_for(recent_rows)

    if rec.n < min_sample:
        return DegradationFinding(
            scope, base.expectancy, base.n, rec.expectancy, rec.n, None,
            "insufficient_sample",
            f"recent n={rec.n} < {min_sample} - not actionable",
        )
    if base.expectancy is None:
        return DegradationFinding(
            scope, None, base.n, rec.expectancy, rec.n, None,
            "insufficient_baseline",
            f"no baseline expectancy (baseline n={base.n}) to compare against",
        )
    if rec.expectancy is None:
        return DegradationFinding(
            scope, base.expectancy, base.n, None, rec.n, None,
            "insufficient_sample",
            "recent window has no measurable expectancy",
        )

    delta = base.expectancy - rec.expectancy
    # Epsilon so a drop sitting exactly on the threshold isn't flipped by float
    # noise (e.g. 2.0 - 1.85 != 0.15 exactly) — a hair under still counts.
    if delta >= meaningful_delta - _DELTA_EPSILON:
        verdict = "degraded"
        note = (
            f"recent expectancy {rec.expectancy:+.2f}% is {delta:.2f} pts below "
            f"baseline {base.expectancy:+.2f}% (recent n={rec.n})"
        )
    else:
        verdict = "stable"
        note = (
            f"recent expectancy {rec.expectancy:+.2f}% within {meaningful_delta} "
            f"of baseline {base.expectancy:+.2f}% (recent n={rec.n})"
        )
    return DegradationFinding(
        scope, base.expectancy, base.n, rec.expectancy, rec.n, delta, verdict, note,
    )


def compute_degradation(
    baseline_rows: Sequence[Mapping[str, Any]],
    recent_rows: Sequence[Mapping[str, Any]],
    *,
    min_sample: int = config.SO_MIN_SAMPLE,
    meaningful_delta: float = config.SO_MEANINGFUL_DELTA,
) -> list[DegradationFinding]:
    """Recent-vs-baseline expectancy findings for overall, per-setup, per-ticker.

    Pure and deterministic — the caller supplies the already-split row sets.
    Scopes are emitted in a stable order (overall, then setups sorted, then
    tickers sorted) so the report and persisted runs are reproducible.
    """
    findings = [
        _degradation_finding(
            "overall", baseline_rows, recent_rows,
            min_sample=min_sample, meaningful_delta=meaningful_delta,
        )
    ]
    everything = [*baseline_rows, *recent_rows]

    for setup in sorted({str(r["signal_type"]) for r in everything}):
        findings.append(_degradation_finding(
            f"setup:{setup}",
            [r for r in baseline_rows if r["signal_type"] == setup],
            [r for r in recent_rows if r["signal_type"] == setup],
            min_sample=min_sample, meaningful_delta=meaningful_delta,
        ))

    for ticker in sorted({str(r["ticker"]) for r in everything}):
        findings.append(_degradation_finding(
            f"ticker:{ticker}",
            [r for r in baseline_rows if r["ticker"] == ticker],
            [r for r in recent_rows if r["ticker"] == ticker],
            min_sample=min_sample, meaningful_delta=meaningful_delta,
        ))

    return findings


def detect_degradation(
    *,
    now: datetime | None = None,
    degrade_window_days: int | None = None,
    baseline_window_days: int | None = None,
) -> list[DegradationFinding]:
    """Fetch the two windows from the db and compute degradation findings.

    Recent window is ``[now - degrade_window_days, now]``; baseline is the older,
    disjoint ``[now - baseline_window_days, now - degrade_window_days)``. The
    split is done by the db query (SQL ``closed_at`` comparison) so there is no
    timezone arithmetic on the Python side.
    """
    moment = now if now is not None else datetime.now(UTC)
    dwd = degrade_window_days if degrade_window_days is not None else config.SO_DEGRADE_WINDOW_DAYS
    bwd = baseline_window_days if baseline_window_days is not None else config.SO_BASELINE_WINDOW_DAYS
    recent_cut = moment - timedelta(days=dwd)
    baseline_cut = moment - timedelta(days=bwd)

    recent_rows = resolved_context_rows(since=recent_cut)
    baseline_rows = resolved_context_rows(since=baseline_cut, until=recent_cut)
    return compute_degradation(baseline_rows, recent_rows)
