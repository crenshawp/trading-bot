"""Advisory risk framework — Phase 7.

Computes a recommended, VOLATILITY-NORMALIZED position size and portfolio-level
risk verdicts for a fired signal, against a configurable NOTIONAL account
(:data:`trading_bot.config.NOTIONAL_ACCOUNT`). ADVISORY ONLY — the operator is
not trading, there is no capital to size or protect. Every number is a
recommendation recorded next to the eventual resolved outcome so later phases
can judge whether the rules were sound. Nothing here sizes, blocks, or shrinks a
real position.

Sizing is normalized by volatility: the stop distance derives from the Phase 6
ATR (``ATR × RISK_ATR_STOP_MULTIPLE``), so a noisier name with a wider ATR gets a
SMALLER size for the same dollar risk. Risk per trade is a fixed fraction of the
notional account; the recommendation is capped at a per-position notional
ceiling. Portfolio verdicts sum the recommended risk across currently-open
positions and reuse the Phase 6 correlation/concentration label for
correlated-cluster exposure.

Everything is pure (open positions are injected, not queried) and FAIL-SOFT:
any missing input (no ATR, no correlation) yields an "unavailable"/"unknown"
recommendation with a logged reason; it never raises, so risk work can never
block a signal.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from trading_bot import config

# ══════════════════════════════════════════════════════════════════════════════
# POSITION SIZING — volatility-normalized, capped at a per-position ceiling
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SizeRecommendation:
    """A recommended position size for one signal. Advisory only.

    ``risk_pct`` is the fraction of the notional account actually put at risk
    (≈ the requested per-trade risk, but LOWER when the per-position cap binds).
    ``ok`` is False whenever sizing fell through to an unavailable result — the
    caller persists it either way and never gates on it.
    """

    recommended_size: float | None = None   # units (shares/contracts)
    stop_distance: float | None = None       # ATR × stop multiple
    dollar_risk: float | None = None         # notional $ at risk if stop hit
    risk_pct: float | None = None            # dollar_risk as % of the account
    position_value: float | None = None      # recommended_size × entry
    position_pct: float | None = None        # position_value as % of the account
    capped: bool = False                     # per-position ceiling bound the size
    ok: bool = False
    reason: str = "not computed"


def position_size(
    entry: float | None,
    atr: float | None,
    account: float = config.NOTIONAL_ACCOUNT,
    risk_per_trade_pct: float = config.RISK_PER_TRADE_PCT,
    *,
    stop_multiple: float = config.RISK_ATR_STOP_MULTIPLE,
    max_position_pct: float = config.MAX_POSITION_PCT,
) -> SizeRecommendation:
    """Volatility-normalized size: ``dollar_risk / (ATR × stop_multiple)``.

    Dollar risk is ``account × risk_per_trade_pct%``; the stop distance is an
    ATR multiple, so a wider-ATR (noisier) name is sized SMALLER for the same
    dollar risk. The position value is capped at ``max_position_pct%`` of the
    account — when that binds, the actual risk is lower than requested and
    ``capped`` is set.

    FAIL-SOFT: a missing/zero ATR (or non-positive entry/account) returns an
    unavailable recommendation with a logged reason; it never raises.
    """
    try:
        if atr is None or atr <= 0.0:
            return _unavailable("missing/zero ATR")
        if entry is None or entry <= 0.0:
            return _unavailable("non-positive entry price")
        if account <= 0.0:
            return _unavailable("non-positive account size")

        stop_distance = atr * stop_multiple
        dollar_risk = account * risk_per_trade_pct / 100.0
        raw_size = dollar_risk / stop_distance
        position_value = raw_size * entry
        max_position_value = account * max_position_pct / 100.0

        capped = position_value > max_position_value
        if capped:
            size = max_position_value / entry
            position_value = max_position_value
            dollar_risk = size * stop_distance
        else:
            size = raw_size

        return SizeRecommendation(
            recommended_size=size,
            stop_distance=stop_distance,
            dollar_risk=dollar_risk,
            risk_pct=dollar_risk / account * 100.0,
            position_value=position_value,
            position_pct=position_value / account * 100.0,
            capped=capped,
            ok=True,
            reason="ok",
        )
    except Exception as exc:  # noqa: BLE001 - sizing must never block a signal
        print(f"  risk: position sizing failed, recommending unavailable: {exc}",
              file=sys.stderr)
        return _unavailable(str(exc))


def _unavailable(why: str) -> SizeRecommendation:
    """A fail-soft 'no size' recommendation, logged."""
    print(f"  risk: size unavailable ({why})", file=sys.stderr)
    return SizeRecommendation(ok=False, reason=f"size unavailable: {why}")
