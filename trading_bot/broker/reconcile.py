"""Broker-authoritative reconciliation — Phase 11, scope fixed in Phase 18.

Compares the bot's BROKER-TRACKED open positions against what the broker
actually holds and REPORTS the divergences. The broker is the source of truth
for what positions / orders really exist; when the two disagree the broker
wins. This layer REPORTS divergences only — it never mutates a record.
Auto-healing internal history from the broker is wired carefully in a later
phase, because a transient broker read must never be allowed to corrupt the
outcome history.

SCOPE (Phase 18): the internal side is ONLY what was actually routed through
execution — open option positions and open long-term positions, both recorded
exclusively on a broker-accepted order. The ``trades`` table is SIGNAL
tracking: yfinance-resolved signal rows that were never sent to a broker and
vastly outnumber real positions. Comparing them against broker positions
produced a flood of false ``internal_only`` divergences in completely normal
operation — noise that would falsely trip Phase 15's Tier 2
reconcile-divergence streak the moment reconciliation ran automatically. Those
rows are out of reconciliation's scope entirely: not compared, not reported.

Three divergence kinds are detected:

* ``internal_only``  — the bot has an open broker-tracked position but the
  broker holds nothing under that symbol.
* ``broker_only``    — the broker holds a position the bot does not track
  (e.g. a Phase 16 SWING shares fallback, which has no internal lifecycle
  book — an honest gap worth surfacing).
* ``qty_mismatch``   — both sides have the symbol but the quantities differ.
  The internal side now carries REAL quantities (contracts / fractional
  shares from the position tables), so this check is live.

If the broker is UNREACHABLE the report is ``ok=False`` and NO internal
position is flagged — an outage is never treated as "the broker has no
positions".
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from trading_bot import db
from trading_bot.broker.base import Broker

_QTY_TOLERANCE = 1e-9
_ALPACA_CRYPTO_SYMBOL_RE = re.compile(r"^(?P<base>[A-Z0-9]+)/USD$")


@dataclass(frozen=True)
class Divergence:
    """One disagreement between internal records and the broker."""

    kind: str            # 'internal_only' | 'broker_only' | 'qty_mismatch'
    symbol: str
    detail: str
    internal_qty: float | None = None
    broker_qty: float | None = None


@dataclass(frozen=True)
class ReconciliationReport:
    """The outcome of a reconciliation pass.

    ``ok=False`` means the broker could not be read, so nothing could be
    concluded — divergences then carries a single ``broker_unavailable`` note
    and no internal trade is flagged.
    """

    ok: bool
    divergences: list[Divergence] = field(default_factory=list)
    internal_symbols: list[str] = field(default_factory=list)
    broker_symbols: list[str] = field(default_factory=list)
    broker_open_orders: int | None = None
    note: str = ""


def _qty_equal(a: float, b: float) -> bool:
    return abs(a - b) <= _QTY_TOLERANCE


def _comparison_symbol(symbol: str) -> str:
    """Return the existing internal form for exact Alpaca crypto wire symbols.

    This adapter is deliberately local to reconciliation. It is not a general
    symbol canonicalizer and leaves stocks, dual-class shares, OCC options, and
    non-exact spellings unchanged.
    """
    match = _ALPACA_CRYPTO_SYMBOL_RE.match(symbol)
    if match is None:
        return symbol
    return f"{match.group('base')}-USD"


def _normalized_internal_positions(
    positions: Mapping[str, float | None],
) -> dict[str, float | None]:
    """Normalize internal keys and merge aliases without inventing quantity.

    Numeric aliases sum. If any alias has unknown quantity, the collapsed
    quantity remains unknown so presence-only reconciliation semantics survive.
    """
    normalized: dict[str, float | None] = {}
    for symbol, qty in positions.items():
        key = _comparison_symbol(symbol)
        if key not in normalized:
            normalized[key] = qty
            continue
        current = normalized[key]
        normalized[key] = (
            None if current is None or qty is None else current + qty
        )
    return normalized


def _normalized_broker_positions(
    positions: Iterable[tuple[str, float]],
) -> dict[str, float]:
    """Normalize broker keys and sum every exact/alias collision safely."""
    normalized: dict[str, float] = {}
    for symbol, qty in positions:
        key = _comparison_symbol(symbol)
        normalized[key] = normalized.get(key, 0.0) + qty
    return normalized


def compare_positions(
    internal: Mapping[str, float | None], broker: Mapping[str, float],
) -> list[Divergence]:
    """Pure comparison of two symbol→qty maps. The broker map is authoritative.

    ``internal`` values may be ``None`` (presence known, quantity not) — in that
    case only presence is reconciled, never quantity.
    """
    normalized_internal = _normalized_internal_positions(internal)
    normalized_broker = _normalized_broker_positions(broker.items())
    divergences: list[Divergence] = []
    for symbol in sorted(normalized_internal):
        iqty = normalized_internal[symbol]
        if symbol not in normalized_broker:
            divergences.append(Divergence(
                "internal_only", symbol,
                "internal open trade but the broker holds no position",
                internal_qty=iqty, broker_qty=None,
            ))
        elif iqty is not None and not _qty_equal(iqty, normalized_broker[symbol]):
            divergences.append(Divergence(
                "qty_mismatch", symbol,
                f"internal qty {iqty} != broker qty {normalized_broker[symbol]}",
                internal_qty=iqty, broker_qty=normalized_broker[symbol],
            ))
    for symbol in sorted(normalized_broker):
        if symbol not in normalized_internal:
            divergences.append(Divergence(
                "broker_only", symbol,
                "broker holds a position the bot does not track",
                internal_qty=None, broker_qty=normalized_broker[symbol],
            ))
    return divergences


def internal_open_positions() -> dict[str, float | None]:
    """Symbols the bot believes are open AT THE BROKER, with real quantities.

    Phase 18 scope: ONLY broker-tracked positions —

    * open ``option_positions`` (recorded exclusively when the broker accepted
      the order; each row carries its broker ``order_id``), keyed by OCC
      symbol with quantity in contracts;
    * open ``long_term_positions`` (inserted exclusively by the Phase 16
      entry path after a broker-accepted order), keyed by ticker with the
      fractional-share quantity.

    Multiple open rows on one symbol (entries across cycles) sum. The
    ``trades`` table is deliberately NOT read: those are signal-tracking rows
    resolved against yfinance data, never routed to a broker — reconciling
    them produced false ``internal_only`` floods in normal operation.
    """
    result: dict[str, float | None] = {}
    for option in db.get_open_option_positions():
        current = result.get(option.symbol) or 0.0
        result[option.symbol] = current + option.contracts
    for position in db.get_open_long_term_positions():
        current = result.get(position.ticker) or 0.0
        result[position.ticker] = current + position.qty
    return result


def reconcile(broker_client: Broker) -> ReconciliationReport:
    """Reconcile broker-tracked open positions against the broker (which is
    authoritative). Signal-tracking trades are out of scope — see
    :func:`internal_open_positions`.

    Fail-soft: if the broker positions read fails the report is ``ok=False`` and
    no internal position is flagged. Divergences are logged to stderr and
    returned; nothing is mutated.
    """
    positions = broker_client.get_positions()
    if not positions.ok:
        print(
            "  reconcile: broker positions unavailable - cannot reconcile "
            f"({positions.reason})",
            file=sys.stderr,
        )
        return ReconciliationReport(
            ok=False,
            divergences=[Divergence(
                "broker_unavailable", "",
                f"broker positions unavailable: {positions.reason}",
            )],
            note=positions.reason,
        )

    broker_map = _normalized_broker_positions(
        (position.symbol, position.qty) for position in positions.positions
    )
    internal_map = _normalized_internal_positions(internal_open_positions())
    divergences = compare_positions(internal_map, broker_map)

    # Open broker orders are surfaced as context (a resting order is not a
    # position, but it is worth seeing alongside the reconciliation).
    orders = broker_client.list_orders("open")
    open_order_count = len(orders.orders) if orders.ok else None

    for d in divergences:
        print(
            f"  reconcile divergence: [{d.kind}] {d.symbol or '-'} - {d.detail}",
            file=sys.stderr,
        )

    return ReconciliationReport(
        ok=True,
        divergences=divergences,
        internal_symbols=sorted(internal_map),
        broker_symbols=sorted(broker_map),
        broker_open_orders=open_order_count,
    )
