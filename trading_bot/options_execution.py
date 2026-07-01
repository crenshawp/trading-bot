"""Options execution — selection, cost model, hierarchy, exit watcher (Phase 13).

Consumes a Phase 12 allocated SWING candidate (side + allocated capital +
underlying) and turns it into an actual single-leg option order (buy call / buy
put), with a fractional-share fallback. Still PAPER ONLY.

Execution hierarchy per allocated swing signal (Section 4):
  1. a FULL-sized option at target delta (0.65-0.75), or
  2. an UNDERSIZED option (delta band widened down to 0.50, still liquidity-
     gated, still a WHOLE contract) that fits the allocated capital, or
  3. fall back to FRACTIONAL SHARES via the existing Phase 11 equity order path.

Greeks (theta/vega/gamma) are stored for audit but are NOT hard gates this phase.
No spreads / multi-leg. No naive market orders (LIMIT default). Alpaca has no
bracket/OTO for options, so exits are managed explicitly (Section 6).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from trading_bot import config
from trading_bot.broker.options import (
    OPTION_TYPE_CALL,
    OPTION_TYPE_PUT,
    OptionContract,
)


def _dte(expiry: str, ref_date: date) -> int | None:
    """Days-to-expiration from ``ref_date`` to the ``'YYYY-MM-DD'`` expiry, or
    None if the date is malformed (never raises)."""
    try:
        exp = date.fromisoformat(expiry)
    except ValueError:
        return None
    return (exp - ref_date).days


def _passes_liquidity(
    contract: OptionContract, min_open_interest: int, max_spread_pct: float,
) -> bool:
    """Liquidity floor: enough open interest AND a tight enough bid-ask spread.
    A contract missing either datum fails (we do not trade what we cannot price)."""
    if contract.open_interest is None or contract.open_interest < min_open_interest:
        return False
    spread = contract.spread_pct
    return spread is not None and spread <= max_spread_pct


def select_contract(
    side: str,
    contracts: Sequence[OptionContract],
    *,
    ref_date: date,
    delta_low: float = config.TARGET_DELTA_LOW,
    delta_high: float = config.TARGET_DELTA_HIGH,
    min_dte: int = config.MIN_DTE,
    min_open_interest: int = config.MIN_OPEN_INTEREST,
    max_spread_pct: float = config.MAX_SPREAD_PCT,
) -> OptionContract | None:
    """Pick the best contract in the delta band that clears DTE + liquidity.

    ``side`` maps call/long → a CALL and put/short → a PUT. A contract qualifies
    when: its type matches the side, ``|delta|`` is within ``[delta_low,
    delta_high]``, its DTE ≥ ``min_dte``, and it clears the liquidity floor.
    Among qualifiers, the one whose ``|delta|`` is CLOSEST to the band centre is
    chosen. Returns None if nothing qualifies (the caller widens the band, then
    falls back to shares). Pure — no I/O, ``ref_date`` is injected.
    """
    want_type = OPTION_TYPE_CALL if side in ("call", "long") else OPTION_TYPE_PUT
    centre = (delta_low + delta_high) / 2.0

    qualifying: list[tuple[float, OptionContract]] = []
    for c in contracts:
        if c.option_type != want_type:
            continue
        d = c.delta
        if d is None or not (delta_low <= abs(d) <= delta_high):
            continue
        dte = _dte(c.expiry, ref_date)
        if dte is None or dte < min_dte:
            continue
        if not _passes_liquidity(c, min_open_interest, max_spread_pct):
            continue
        qualifying.append((abs(d), c))

    if not qualifying:
        return None
    return min(qualifying, key=lambda t: abs(t[0] - centre))[1]
