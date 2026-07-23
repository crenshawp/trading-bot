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

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
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


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EVALUATION — does any stored advisory context predict outcomes?
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class FeatureBucket:
    """One bucket of a feature with its resolved-outcome stats."""

    label: str
    n: int
    win_rate: float | None
    expectancy: float | None
    actionable: bool             # n >= the sample floor


@dataclass(frozen=True)
class FeatureEvaluation:
    """One feature's buckets + a plain-language read.

    ``actionable`` is True only when at least two buckets each clear the sample
    floor (so a comparison between them is not just noise). The ``note`` always
    spells out the actionability caveat.
    """

    feature: str
    buckets: list[FeatureBucket]
    actionable: bool
    note: str


def _bucket_label(row: Mapping[str, Any], column: str, allowed: set[str]) -> str | None:
    value = row.get(column)
    return str(value) if value in allowed else None


def _bucket_sentiment(row: Mapping[str, Any]) -> str | None:
    # Neutral is meaningful only when it came from a successful LLM score.
    # Fail-soft neutral and provenance-unknown legacy rows are not evidence.
    if row.get("sentiment_ok") is not True:
        return None
    return _bucket_label(row, "sentiment_label", {"bearish", "neutral", "bullish"})


def _indicator_succeeded(row: Mapping[str, Any]) -> bool:
    return row.get("ind_ok") is True


def _bucket_vol_regime(row: Mapping[str, Any]) -> str | None:
    if not _indicator_succeeded(row):
        return None
    return _bucket_label(row, "ind_vol_regime", {"low", "normal", "high"})


def _bucket_concentration(row: Mapping[str, Any]) -> str | None:
    if not _indicator_succeeded(row):
        return None
    return _bucket_label(
        row, "ind_concentration", {"concentrated", "moderate", "diversified"}
    )


def _bucket_portfolio_verdict(row: Mapping[str, Any]) -> str | None:
    return _bucket_label(
        row, "risk_portfolio_verdict", {"ok", "would-exceed-portfolio"}
    )


def _bucket_rsi(row: Mapping[str, Any]) -> str | None:
    if not _indicator_succeeded(row):
        return None
    value = row.get("ind_rsi")
    if value is None:
        return None
    if value < 40.0:
        return "low(<40)"
    if value <= 60.0:
        return "mid(40-60)"
    return "high(>60)"


def _bucket_adx(row: Mapping[str, Any]) -> str | None:
    if not _indicator_succeeded(row):
        return None
    value = row.get("ind_adx")
    if value is None:
        return None
    if value < 20.0:
        return "weak(<20)"
    if value <= 40.0:
        return "moderate(20-40)"
    return "strong(>40)"


def _bucket_obv(row: Mapping[str, Any]) -> str | None:
    if not _indicator_succeeded(row):
        return None
    value = row.get("ind_obv")
    if value is None:
        return None
    if value > 0.0:
        return "positive"
    if value < 0.0:
        return "negative"
    return "zero"


# (feature name, bucketer, stable display order). RSI/ADX use textbook bands and
# OBV its sign — interpretable, deterministic buckets rather than data-dependent
# terciles, so the read is reproducible and hand-checkable.
_FEATURES: list[tuple[str, Any, list[str]]] = [
    ("sentiment", _bucket_sentiment, ["bearish", "neutral", "bullish"]),
    ("vol_regime", _bucket_vol_regime, ["low", "normal", "high"]),
    ("rsi", _bucket_rsi, ["low(<40)", "mid(40-60)", "high(>60)"]),
    ("adx", _bucket_adx, ["weak(<20)", "moderate(20-40)", "strong(>40)"]),
    ("obv", _bucket_obv, ["negative", "zero", "positive"]),
    ("concentration", _bucket_concentration,
     ["concentrated", "moderate", "diversified"]),
    ("portfolio_verdict", _bucket_portfolio_verdict,
     ["ok", "would-exceed-portfolio"]),
]


def _win_rate_key(bucket: FeatureBucket) -> float:
    return bucket.win_rate if bucket.win_rate is not None else 0.0


def _feature_note(buckets: list[FeatureBucket], *, min_sample: int) -> str:
    rated = [b for b in buckets if b.win_rate is not None]
    if len(rated) < 2:
        return "insufficient sample - not actionable (need >= 2 buckets with outcomes)"
    best = max(rated, key=_win_rate_key)
    worst = min(rated, key=_win_rate_key)
    spread = (
        f"{best.label} {best.win_rate:.0f}% (n={best.n}) vs "
        f"{worst.label} {worst.win_rate:.0f}% (n={worst.n})"
    )
    if best.actionable and worst.actionable:
        return f"{spread} - both buckets clear n>={min_sample}"
    return f"{spread} - suggestive, NOT actionable at n<{min_sample}"


def _evaluate_feature(
    feature: str,
    rows: Sequence[Mapping[str, Any]],
    bucketer: Any,
    order: Sequence[str],
    *,
    min_sample: int,
) -> FeatureEvaluation:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        label = bucketer(row)
        if label is not None:
            grouped.setdefault(label, []).append(row)

    buckets: list[FeatureBucket] = []
    for label in order:
        if label not in grouped:
            continue
        stat = stat_for(grouped[label])
        buckets.append(FeatureBucket(
            label, stat.n, stat.win_rate, stat.expectancy, stat.n >= min_sample,
        ))

    actionable = sum(1 for b in buckets if b.actionable) >= 2
    return FeatureEvaluation(
        feature, buckets, actionable, _feature_note(buckets, min_sample=min_sample),
    )


def compute_feature_evaluations(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_sample: int = config.SO_MIN_SAMPLE,
) -> list[FeatureEvaluation]:
    """Bucket resolved trades by each stored feature and compare outcome rates.

    Pure — the caller supplies the rows. Buckets with no resolved trades are
    omitted; any feature lacking two floor-clearing buckets is reported but
    flagged not-actionable.
    """
    return [
        _evaluate_feature(name, rows, bucketer, order, min_sample=min_sample)
        for name, bucketer, order in _FEATURES
    ]


def evaluate_features() -> list[FeatureEvaluation]:
    """Evaluate every feature over ALL resolved active trades (max sample)."""
    return compute_feature_evaluations(resolved_context_rows())


# ══════════════════════════════════════════════════════════════════════════════
# PAYLOAD + REPORT — serialize findings, persist a run, render it
# ══════════════════════════════════════════════════════════════════════════════


def build_payload(
    degradations: Sequence[DegradationFinding],
    features: Sequence[FeatureEvaluation],
    *,
    run_timestamp: str,
    degrade_window_days: int,
    baseline_window_days: int,
) -> dict[str, Any]:
    """A JSON-serializable payload of one run's findings (for persist + report)."""
    return {
        "run_timestamp": run_timestamp,
        "degrade_window_days": degrade_window_days,
        "baseline_window_days": baseline_window_days,
        "degradations": [asdict(f) for f in degradations],
        "features": [asdict(f) for f in features],
    }


def persist_run(payload: Mapping[str, Any]) -> int:
    """Write a run payload to the optimization_runs history table. Returns the id."""
    return db.insert_optimization_run(
        str(payload["run_timestamp"]),
        int(payload["degrade_window_days"]),
        int(payload["baseline_window_days"]),
        json.dumps(payload),
    )


def _fmt(value: object, spec: str) -> str:
    return format(value, spec) if isinstance(value, (int, float)) else "-"


def render_report(payload: Mapping[str, Any]) -> str:
    """Render a run payload (degradation then feature evaluations) to text.

    Reads the plain dict so a persisted run replays verbatim. Every line carries
    its sample size and actionability label; the header restates that this is
    flags-only.
    """
    lines = [
        f"SELF-OPTIMIZATION REPORT  {payload['run_timestamp']}",
        f"  windows: recent {payload['degrade_window_days']}d vs "
        f"baseline {payload['baseline_window_days']}d  "
        f"(actionable floor n>={config.SO_MIN_SAMPLE})",
        "  FLAGS ONLY - nothing here changes the bot's behavior.",
        "",
        "DEGRADATION",
    ]
    degradations = payload.get("degradations", [])
    if not degradations:
        lines.append("  (no resolved trades in the window)")
    for d in degradations:
        lines.append(
            f"  {d['scope']:<26} {d['verdict']:<22} "
            f"delta={_fmt(d['delta'], '+.2f'):<6}  {d['note']}"
        )

    lines += ["", "FEATURE EVALUATION"]
    for feat in payload.get("features", []):
        flag = "actionable" if feat["actionable"] else "not actionable"
        lines.append(f"  {feat['feature']}  [{flag}]  {feat['note']}")
        if not feat["buckets"]:
            lines.append("    (no resolved trades carry this feature)")
        for b in feat["buckets"]:
            tag = "" if b["actionable"] else "  (n<floor)"
            lines.append(
                f"    {b['label']:<18} "
                f"win_rate={_fmt(b['win_rate'], '.0f') + '%':<5} "
                f"expectancy={_fmt(b['expectancy'], '+.2f'):<7} n={b['n']}{tag}"
            )
    return "\n".join(lines)


def run_optimization(
    *,
    now: datetime | None = None,
    degrade_window_days: int | None = None,
    baseline_window_days: int | None = None,
) -> dict[str, Any]:
    """Run degradation detection + feature evaluation and build a run payload.

    Reads the db (resolved active trades) but does NOT persist — the caller
    persists and renders. Flags only.
    """
    moment = now if now is not None else datetime.now(UTC)
    dwd = degrade_window_days if degrade_window_days is not None else config.SO_DEGRADE_WINDOW_DAYS
    bwd = baseline_window_days if baseline_window_days is not None else config.SO_BASELINE_WINDOW_DAYS

    degradations = detect_degradation(
        now=moment, degrade_window_days=dwd, baseline_window_days=bwd,
    )
    features = evaluate_features()
    return build_payload(
        degradations, features, run_timestamp=moment.isoformat(),
        degrade_window_days=dwd, baseline_window_days=bwd,
    )
