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

from dataclasses import dataclass

from trading_bot import config, db


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
