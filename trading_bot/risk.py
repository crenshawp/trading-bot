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
from collections.abc import Sequence
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


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO RISK — advisory verdicts across currently-open positions
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class OpenPosition:
    """A currently-open trade's contribution to portfolio risk.

    ``risk_pct`` is the recommended risk recorded on that trade at its own fire
    time (``None`` for trades opened before Phase 7 — skipped from the sums).
    ``concentration`` is its persisted Phase 6 label.
    """

    risk_pct: float | None
    concentration: str = "unknown"


@dataclass(frozen=True)
class PortfolioRisk:
    """Advisory portfolio-level verdicts for a candidate against the open book.

    Each verdict is ``ok`` / ``would-exceed-X`` / ``unknown`` and is RECORDED,
    never enforced. ``ok`` is True only when all three verdicts are ``ok``.
    """

    total_risk_pct: float | None = None       # open + candidate risk, % of notional
    portfolio_verdict: str = "unknown"        # ok | would-exceed-portfolio | unknown
    position_pct: float | None = None         # candidate position size, % of notional
    position_verdict: str = "unknown"         # ok | would-exceed-position | unknown
    cluster_risk_pct: float | None = None     # concentrated-cluster risk, % of notional
    cluster_verdict: str = "unknown"          # ok | would-exceed-cluster | unknown
    ok: bool = False
    reason: str = "not computed"


def portfolio_risk(
    candidate: SizeRecommendation,
    candidate_concentration: str,
    open_positions: Sequence[OpenPosition],
    *,
    max_portfolio_pct: float = config.MAX_PORTFOLIO_RISK_PCT,
    max_position_pct: float = config.MAX_POSITION_PCT,
    max_cluster_pct: float = config.MAX_CORRELATED_CLUSTER_PCT,
) -> PortfolioRisk:
    """Advisory verdicts: summed open risk vs the portfolio cap, the candidate's
    per-position size vs the position cap, and correlated-cluster exposure vs the
    cluster cap.

    Cluster exposure reuses the Phase 6 concentration label: a ``concentrated``
    candidate joins the cluster of all currently-open ``concentrated`` positions
    (names that move with the pack are a hidden single bet), and their combined
    risk is compared to ``max_cluster_pct``. A non-concentrated candidate is not
    adding to a cluster (verdict ``ok``); an ``unknown`` concentration yields an
    ``unknown`` cluster verdict.

    FAIL-SOFT: if the candidate has no usable size, or any input is malformed,
    the verdicts are ``unknown`` (logged); it never raises.
    """
    try:
        if not candidate.ok or candidate.risk_pct is None:
            print("  risk: portfolio verdicts unknown (candidate size unavailable)",
                  file=sys.stderr)
            return PortfolioRisk(reason="portfolio risk unknown: candidate size unavailable")

        open_risk = sum(
            p.risk_pct for p in open_positions if p.risk_pct is not None
        )
        total = open_risk + candidate.risk_pct
        portfolio_verdict = (
            "would-exceed-portfolio" if total > max_portfolio_pct else "ok"
        )

        position_pct = candidate.position_pct
        position_verdict = (
            "would-exceed-position"
            if position_pct is not None and position_pct >= max_position_pct
            else "ok"
        )

        cluster_open = sum(
            p.risk_pct for p in open_positions
            if p.risk_pct is not None and p.concentration == "concentrated"
        )
        if candidate_concentration == "unknown":
            cluster_risk: float | None = None
            cluster_verdict = "unknown"
        elif candidate_concentration == "concentrated":
            cluster_risk = cluster_open + candidate.risk_pct
            cluster_verdict = (
                "would-exceed-cluster" if cluster_risk > max_cluster_pct else "ok"
            )
        else:  # moderate / diversified — not joining a correlated cluster
            cluster_risk = cluster_open
            cluster_verdict = "ok"

        ok = (
            portfolio_verdict == "ok"
            and position_verdict == "ok"
            and cluster_verdict == "ok"
        )
        return PortfolioRisk(
            total_risk_pct=total,
            portfolio_verdict=portfolio_verdict,
            position_pct=position_pct,
            position_verdict=position_verdict,
            cluster_risk_pct=cluster_risk,
            cluster_verdict=cluster_verdict,
            ok=ok,
            reason="ok",
        )
    except Exception as exc:  # noqa: BLE001 - portfolio risk must never block a signal
        print(f"  risk: portfolio verdicts failed, returning unknown: {exc}",
              file=sys.stderr)
        return PortfolioRisk(reason=f"portfolio risk unknown: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# ASSESSMENT — the per-signal bundle attached to a fired trade
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RiskAssessment:
    """Flat sizing + portfolio bundle snapshotted for one fired signal.

    Combines :class:`SizeRecommendation` and :class:`PortfolioRisk` into one
    object the scanner persists on the trade row. Advisory only.
    """

    recommended_size: float | None = None
    stop_distance: float | None = None
    dollar_risk: float | None = None
    risk_pct: float | None = None
    position_pct: float | None = None
    capped: bool = False
    total_risk_pct: float | None = None
    portfolio_verdict: str = "unknown"
    position_verdict: str = "unknown"
    cluster_risk_pct: float | None = None
    cluster_verdict: str = "unknown"
    ok: bool = False
    reason: str = "not computed"


def assess(
    entry: float | None,
    atr: float | None,
    concentration: str,
    open_positions: Sequence[OpenPosition],
    *,
    account: float = config.NOTIONAL_ACCOUNT,
    risk_per_trade_pct: float = config.RISK_PER_TRADE_PCT,
    stop_multiple: float = config.RISK_ATR_STOP_MULTIPLE,
    max_position_pct: float = config.MAX_POSITION_PCT,
    max_portfolio_pct: float = config.MAX_PORTFOLIO_RISK_PCT,
    max_cluster_pct: float = config.MAX_CORRELATED_CLUSTER_PCT,
) -> RiskAssessment:
    """Size the candidate, then judge it against the open book. Pure — the open
    positions are injected. Fail-soft throughout (sizing/portfolio each swallow
    their own errors)."""
    size = position_size(
        entry, atr, account, risk_per_trade_pct,
        stop_multiple=stop_multiple, max_position_pct=max_position_pct,
    )
    pf = portfolio_risk(
        size, concentration, open_positions,
        max_portfolio_pct=max_portfolio_pct, max_position_pct=max_position_pct,
        max_cluster_pct=max_cluster_pct,
    )
    return RiskAssessment(
        recommended_size=size.recommended_size,
        stop_distance=size.stop_distance,
        dollar_risk=size.dollar_risk,
        risk_pct=size.risk_pct,
        position_pct=size.position_pct,
        capped=size.capped,
        total_risk_pct=pf.total_risk_pct,
        portfolio_verdict=pf.portfolio_verdict,
        position_verdict=pf.position_verdict,
        cluster_risk_pct=pf.cluster_risk_pct,
        cluster_verdict=pf.cluster_verdict,
        ok=size.ok and pf.ok,
        reason=size.reason if not size.ok else pf.reason,
    )
