"""Unified readiness gate — the capstone authority (Phase 10).

ONE authority over EVERY capability. It tracks accumulated resolved-trade counts
per capability against per-capability data thresholds and, when a capability
first crosses its threshold, fires a ONE-TIME notification and takes the
capability-appropriate action:

* DETERMINISTIC capabilities (watchlist rotation, pair gating, shadow promotion):
  auto-ACTIVATE — once ready they are simply allowed to act, no human involved.
  ``self_optimization`` is tracked here too (its Phase 9 actionability floor),
  but it takes no auto-action — it self-labels per finding.
* ML capabilities (pattern recognition, predictive sizing): auto-flag READY TO
  BUILD only. They NEVER self-train or self-deploy — crossing a threshold only
  reaches ``ready`` and summons a human build. (Registered in Section 4.)

This module centralizes the scattered Phase 3/4/9 sample-minimums: deterministic
thresholds are the existing constants (``config.SM_MIN_CLOSED_SIGNALS`` etc.),
UNCHANGED. The registry is the single source of truth.

``is_ready`` is the gate the deterministic evaluators consult. Readiness is
MONOTONIC — resolved counts only grow — so the live count check equals the
persisted readiness status after any :func:`evaluate_readiness` pass, and is also
correct before the first pass.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import requests

from trading_bot import config, db, secrets

# A notifier sends (title, message) and returns True on a confirmed send. It is
# dependency-injected so tests mock it; the default posts to Pushover.
Notifier = Callable[[str, str], bool]

_HTTP_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class Capability:
    """A registered capability and its data-sufficiency gate.

    ``scope`` selects which resolved-trade count measures it: ``all`` (every
    track_mode), ``active`` (the live alerting book), or ``shadow``.
    """

    name: str
    kind: str            # 'deterministic' | 'ml'
    threshold: int
    scope: str           # 'all' | 'active' | 'shadow'
    description: str


# The registry — single source of truth. Deterministic thresholds are the
# existing Phase 3/4/9 minimums, centralized UNCHANGED. ML capabilities are
# appended in Section 4.
REGISTRY: tuple[Capability, ...] = (
    Capability(
        "watchlist_rotation", "deterministic", config.SM_MIN_CLOSED_SIGNALS,
        "all", "active<->benched ticker rotation (Phase 3.3)",
    ),
    Capability(
        "pair_gating", "deterministic", config.SP_MIN_CLOSED_SIGNALS,
        "all", "per-(ticker, signal) mute/enable (Phase 4)",
    ),
    Capability(
        "shadow_promotion", "deterministic", config.MIN_SHADOW_SIGNALS,
        "shadow", "live-shadow promotion into the watchlist (Phase 3.1-LIVE)",
    ),
    Capability(
        "self_optimization", "deterministic", config.SO_MIN_SAMPLE,
        "active", "degradation + feature-evaluation actionability (Phase 9)",
    ),
)


def _by_name() -> dict[str, Capability]:
    return {c.name: c for c in REGISTRY}


def capability(name: str) -> Capability:
    """Look up a registered capability by name. Raises KeyError if unknown."""
    try:
        return _by_name()[name]
    except KeyError:
        raise KeyError(f"unknown capability {name!r}") from None


def _scope_count(scope: str) -> int:
    if scope == "all":
        return db.count_resolved_trades()
    if scope == "active":
        return db.count_resolved_trades(track_mode="active")
    if scope == "shadow":
        return db.count_resolved_trades(track_mode="shadow")
    raise ValueError(f"unknown readiness scope {scope!r}")


def resolved_count(name: str) -> int:
    """Cumulative resolved-trade count for a capability's scope."""
    return _scope_count(capability(name).scope)


def is_ready(name: str) -> bool:
    """True iff the capability has accumulated enough resolved data to act.

    Live count vs threshold. Because readiness is monotonic this equals the
    persisted status after any evaluate_readiness pass; the deterministic
    evaluators consult this as their auto-activation gate (ready == active).
    """
    cap = capability(name)
    return _scope_count(cap.scope) >= cap.threshold


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION — flip warming->ready, fire the one-time crossing notification
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ReadinessResult:
    """One capability's standing after an evaluation pass."""

    capability: str
    kind: str
    count: int
    threshold: int
    status: str              # 'warming' | 'ready'
    announced: bool
    newly_announced: bool    # this pass sent the (one-time) crossing notification


def _pushover_notify(title: str, message: str) -> bool:
    """Default notifier — post to Pushover. Returns True only on a 2xx send.

    Fail-soft: missing creds, a non-2xx status, or any exception returns False
    (logged), so the caller leaves ``announced`` False and retries next cycle.
    Never raises.
    """
    try:
        token = secrets.get_secret("PUSHOVER_APP_TOKEN")
        user = secrets.get_secret("PUSHOVER_USER_KEY")
        if not token or not user:
            print(
                "  readiness: Pushover creds unset - notification deferred",
                file=sys.stderr,
            )
            return False
        resp = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={"token": token, "user": user, "title": title, "message": message},
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
        status = getattr(resp, "status_code", None)
        if status is not None and 200 <= status < 300:
            return True
        print(f"  readiness: Pushover HTTP {status} - will retry", file=sys.stderr)
        return False
    except Exception as exc:  # noqa: BLE001 - notification must never raise into a scan
        print(f"  readiness: Pushover error ({exc}) - will retry", file=sys.stderr)
        return False


def _announce(cap: Capability, n: int, send: Notifier) -> bool:
    """Send the capability-appropriate crossing message. Returns send success."""
    title = f"Readiness: {cap.name}"
    if cap.kind == "ml":
        message = (
            f"Enough data has accumulated for {cap.name} to be built "
            f"(n={n} >= {cap.threshold}). Awaiting the build."
        )
    else:
        message = (
            f"{cap.name} has enough data to activate (n={n} >= {cap.threshold}) "
            "-> switched on automatically."
        )
    try:
        return send(title, message)
    except Exception as exc:  # noqa: BLE001 - a misbehaving notifier must not crash the pass
        print(f"  readiness: notifier error for {cap.name} ({exc})", file=sys.stderr)
        return False


def evaluate_readiness(
    *, notifier: Notifier | None = None, now: datetime | None = None
) -> list[ReadinessResult]:
    """Evaluate every capability; flip warming->ready and notify on FIRST crossing.

    A threshold crossing is a ONE-TIME edge event: the persisted ``announced``
    flag makes each notification fire EXACTLY ONCE and never re-fire. A failed
    send leaves ``announced`` False so a confirmed crossing retries next cycle; a
    successful send sets it True permanently. Deterministic capabilities announce
    auto-activation; ML capabilities announce a build summons (they NEVER train
    or deploy — only reach 'ready').
    """
    send = notifier if notifier is not None else _pushover_notify
    moment = now if now is not None else datetime.now(UTC)
    results: list[ReadinessResult] = []

    for cap in REGISTRY:
        count = _scope_count(cap.scope)
        row = db.get_readiness_state(cap.name)
        status = row["status"] if row else "warming"
        announced = bool(row["announced"]) if row else False
        n_at_crossing = row["n_at_crossing"] if row else None
        crossed_at = row["crossed_at"] if row else None

        # First crossing: latch to ready and record n + time (once).
        if count >= cap.threshold and status != "ready":
            status = "ready"
            n_at_crossing = count
            crossed_at = moment.isoformat()

        newly_announced = False
        # One-time notification: ready but not yet announced.
        if status == "ready" and not announced:
            crossing_n = n_at_crossing if n_at_crossing is not None else count
            if _announce(cap, crossing_n, send):
                announced = True
                newly_announced = True
            # else announced stays False -> retried on a later pass

        db.upsert_readiness_state(
            cap.name, status=status, n_at_crossing=n_at_crossing,
            crossed_at=crossed_at, announced=announced,
        )
        results.append(ReadinessResult(
            cap.name, cap.kind, count, cap.threshold, status, announced,
            newly_announced,
        ))

    return results
