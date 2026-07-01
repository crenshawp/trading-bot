"""Options broker layer — Phase 13. Neutral contract type + OCC encoding.

The neutral :class:`OptionContract` is the only options shape higher layers see;
the companion ``AlpacaOptionsClient`` (added in Section 2) translates Alpaca's
raw contract + snapshot JSON into it. Single-leg only (buy call / buy put) — no
spreads or multi-leg this phase.

OCC symbol format (Alpaca form, no space-padding of the root):

    {ROOT}{YYMMDD}{C|P}{strike × 1000, zero-padded to 8 digits}

e.g. AAPL 2026-01-16 call $150.00 → ``AAPL260116C00150000``. Alpaca returns this
as each contract's ``symbol``; the existing Phase 11 order path submits it
verbatim (an option order is just an order whose ``symbol`` is an OCC string).
"""

from __future__ import annotations

from dataclasses import dataclass

OPTION_TYPE_CALL = "call"
OPTION_TYPE_PUT = "put"
VALID_OPTION_TYPES: frozenset[str] = frozenset({OPTION_TYPE_CALL, OPTION_TYPE_PUT})


@dataclass(frozen=True)
class OptionContract:
    """A single option contract in neutral form. Greeks/quotes are ``None`` when
    the chain snapshot did not supply them (such a contract fails the liquidity /
    delta gates and is excluded from selection)."""

    symbol: str                    # OCC symbol, directly submittable
    underlying: str
    option_type: str               # 'call' | 'put'
    strike: float
    expiry: str                    # 'YYYY-MM-DD'
    delta: float | None = None
    theta: float | None = None
    vega: float | None = None
    gamma: float | None = None
    open_interest: int | None = None
    bid: float | None = None
    ask: float | None = None
    mid: float | None = None

    @property
    def spread_pct(self) -> float | None:
        """Bid-ask spread as a percentage of mid, or None if not computable."""
        if (
            self.bid is None or self.ask is None
            or self.mid is None or self.mid <= 0.0
        ):
            return None
        return (self.ask - self.bid) / self.mid * 100.0


def occ_symbol(underlying: str, expiry: str, option_type: str, strike: float) -> str:
    """Build the Alpaca OCC option symbol. ``expiry`` is ``'YYYY-MM-DD'``.

    The strike is encoded as ``strike × 1000`` zero-padded to 8 digits, so a
    $150.00 strike → ``00150000`` and a $7.50 strike → ``00007500``.
    """
    if option_type not in VALID_OPTION_TYPES:
        raise ValueError(f"invalid option_type {option_type!r}")
    yymmdd = f"{expiry[2:4]}{expiry[5:7]}{expiry[8:10]}"
    cp = "C" if option_type == OPTION_TYPE_CALL else "P"
    strike_thousandths = int(round(strike * 1000))
    return f"{underlying.upper()}{yymmdd}{cp}{strike_thousandths:08d}"


def parse_occ_symbol(symbol: str) -> tuple[str, str, str, float] | None:
    """Reverse :func:`occ_symbol` → ``(root, 'YYYY-MM-DD', option_type, strike)``.

    Returns None on a malformed symbol (too short, bad C/P marker, or a
    non-numeric date/strike) — parsing must never raise on garbage input.
    """
    if len(symbol) < 16:   # need root(>=1) + 6 date + 1 cp + 8 strike
        return None
    tail = symbol[-15:]
    root = symbol[:-15]
    cp = tail[6]
    if not root or cp not in ("C", "P"):
        return None
    date_part, strike_part = tail[:6], tail[7:]
    if not (date_part.isdigit() and strike_part.isdigit()):
        return None
    expiry = f"20{date_part[0:2]}-{date_part[2:4]}-{date_part[4:6]}"
    option_type = OPTION_TYPE_CALL if cp == "C" else OPTION_TYPE_PUT
    strike = int(strike_part) / 1000.0
    return root, expiry, option_type, strike
