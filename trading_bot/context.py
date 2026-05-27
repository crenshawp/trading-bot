"""Market Context — composite (regime x VIX) score.

Phase 2.3. The single-number consolidation of the two Phase 2 axes:
:mod:`trading_bot.regime` (direction) and :mod:`trading_bot.vix` (volatility).
Pure composition logic — no yfinance calls of its own. Both
``get_current_context()`` delegates pull from the underlying modules'
existing caches, so this layer is cheap and never blocks.

The score is **long-bias**. PUT / SHORT signals fired in a low-score
environment are technically *benefiting* from that context (e.g. a bear
market is great for puts) but we intentionally do NOT invert the score at
this layer. Phase 4 (Strategy Evolution) will handle the signal-direction
x context interaction. For Phase 2.3, the score is pure metadata —
something to filter on, sort by, and threshold against.

Score matrix (rows = regime, columns = VIX band):

::

                  Low VIX    Elevated    High    Extreme
    Bull            5           4          3        2
    Sideways        4           3          2        1
    Bear            2           2          1        1

Any ``unknown`` on either axis collapses the score to 0. The asymmetry is
deliberate: bear markets cap at 2 regardless of volatility, extreme VIX
docks every regime by at least one point.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from trading_bot import db, regime, vix

# ────────────────────────────────────────────────────────────────────────────
# Score matrix
# ────────────────────────────────────────────────────────────────────────────

_VALID_REGIMES: frozenset[str] = frozenset(
    {"bull", "sideways", "bear", "unknown"}
)
_VALID_VIX_BANDS: frozenset[str] = frozenset(
    {"low", "elevated", "high", "extreme", "unknown"}
)

# Keyed (regime, vix_band) → score. Lookup is O(1) and the matrix is
# canonical (any caller can read this table and confirm the rule).
_SCORE_MATRIX: dict[tuple[str, str], int] = {
    ("bull",     "low"):      5,
    ("bull",     "elevated"): 4,
    ("bull",     "high"):     3,
    ("bull",     "extreme"):  2,
    ("sideways", "low"):      4,
    ("sideways", "elevated"): 3,
    ("sideways", "high"):     2,
    ("sideways", "extreme"):  1,
    ("bear",     "low"):      2,
    ("bear",     "elevated"): 2,
    ("bear",     "high"):     1,
    ("bear",     "extreme"):  1,
}

_LABELS: dict[int, str] = {
    0: "unknown",
    1: "hostile",
    2: "unfavorable",
    3: "neutral",
    4: "favorable",
    5: "ideal",
}


@dataclass(frozen=True)
class ContextSnapshot:
    regime: str
    vix_band: str
    vix_level: float | None
    score: int
    label: str
    captured_at: str   # ISO timestamp


# ────────────────────────────────────────────────────────────────────────────
# Pure functions
# ────────────────────────────────────────────────────────────────────────────


def score(regime_value: str, vix_band: str) -> int:
    """Return the 0-5 score for a (regime, vix_band) pair.

    Either axis being ``'unknown'`` collapses the score to 0. Invalid input
    strings raise ``ValueError`` — a typo should fail loud, not silently
    return 0.
    """
    if regime_value not in _VALID_REGIMES:
        raise ValueError(
            f"Invalid regime {regime_value!r}; expected one of {sorted(_VALID_REGIMES)}"
        )
    if vix_band not in _VALID_VIX_BANDS:
        raise ValueError(
            f"Invalid vix_band {vix_band!r}; expected one of {sorted(_VALID_VIX_BANDS)}"
        )
    if regime_value == "unknown" or vix_band == "unknown":
        return 0
    return _SCORE_MATRIX[(regime_value, vix_band)]


def label(score_value: int) -> str:
    """Map a numeric score (0-5) to its human-readable label."""
    if score_value not in _LABELS:
        raise ValueError(
            f"score out of range: {score_value} (expected 0-5)"
        )
    return _LABELS[score_value]


# ────────────────────────────────────────────────────────────────────────────
# Composite snapshot
# ────────────────────────────────────────────────────────────────────────────


def get_current_context() -> ContextSnapshot:
    """Pull current regime + VIX (via their caches) and build a snapshot.

    Does NOT force-refresh either underlying cache — that's what the
    per-axis CLIs are for. Falls back to ``'unknown'`` on either axis if
    its fetch raises, so the context dashboard never crashes on a transient
    yfinance hiccup.
    """
    try:
        regime_snap = regime.get_current_regime()
        regime_tag = regime_snap.regime
    except regime.RegimeFetchError:
        regime_tag = "unknown"

    try:
        vix_snap = vix.get_current_vix()
        vix_band = vix_snap.vix_band
        vix_level: float | None = vix_snap.vix_level
    except vix.VixFetchError:
        vix_band = "unknown"
        vix_level = None

    value = score(regime_tag, vix_band)
    return ContextSnapshot(
        regime=regime_tag,
        vix_band=vix_band,
        vix_level=vix_level,
        score=value,
        label=label(value),
        captured_at=datetime.now(UTC).isoformat(),
    )


# ────────────────────────────────────────────────────────────────────────────
# Backfill (Phase 2.3 one-shot)
# ────────────────────────────────────────────────────────────────────────────


def backfill_context_scores() -> dict[str, int]:
    """Derive context_score on trades + predictions that have both axes
    populated but a NULL score. Pure derivation — no yfinance, no SPY/VIX
    fetches. Idempotent: a second run finds zero candidates."""
    trades_updated = 0
    for trade in db.get_trades_missing_context_score():
        if (
            trade.id is None
            or trade.market_regime is None
            or trade.vix_band is None
        ):
            continue
        s = score(trade.market_regime, trade.vix_band)
        db.update_trade(trade.id, context_score=s)
        trades_updated += 1

    preds_updated = 0
    for pred in db.get_predictions_missing_context_score():
        if (
            pred.id is None
            or pred.market_regime is None
            or pred.vix_band is None
        ):
            continue
        s = score(pred.market_regime, pred.vix_band)
        db.update_prediction(pred.id, context_score=s)
        preds_updated += 1

    return {
        "trades_updated": trades_updated,
        "predictions_updated": preds_updated,
    }
