"""Broker-authoritative reconciliation — Phase 11.

Compares the bot's internal open trades against what the broker actually holds
and REPORTS the divergences. The broker is the source of truth for what
positions / orders really exist; when the two disagree the broker wins. This
phase REPORTS divergences only — it never mutates a trade record. Auto-healing
internal history from the broker is wired carefully in a later phase, because a
transient broker read must never be allowed to corrupt the outcome history.

Three divergence kinds are detected:

* ``internal_only``  — the bot has an open trade but the broker holds nothing.
* ``broker_only``    — the broker holds a position the bot does not track.
* ``qty_mismatch``   — both sides have the symbol but the quantities differ.

The ``qty_mismatch`` check only fires when the internal side carries a known
quantity. Phase 11 does not yet attach real order quantities to trades (signal→
order wiring is a later phase), so today internal quantities are ``None`` and
``reconcile`` surfaces presence divergences; the comparison is already in place
for when trades start carrying fills.

If the broker is UNREACHABLE the report is ``ok=False`` and NO internal trade is
flagged — an outage is never treated as "the broker has no positions".
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass, field

from trading_bot import db
from trading_bot.broker.base import Broker

_QTY_TOLERANCE = 1e-9


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


def compare_positions(
    internal: Mapping[str, float | None], broker: Mapping[str, float],
) -> list[Divergence]:
    """Pure comparison of two symbol→qty maps. The broker map is authoritative.

    ``internal`` values may be ``None`` (presence known, quantity not) — in that
    case only presence is reconciled, never quantity.
    """
    divergences: list[Divergence] = []
    for symbol in sorted(internal):
        iqty = internal[symbol]
        if symbol not in broker:
            divergences.append(Divergence(
                "internal_only", symbol,
                "internal open trade but the broker holds no position",
                internal_qty=iqty, broker_qty=None,
            ))
        elif iqty is not None and not _qty_equal(iqty, broker[symbol]):
            divergences.append(Divergence(
                "qty_mismatch", symbol,
                f"internal qty {iqty} != broker qty {broker[symbol]}",
                internal_qty=iqty, broker_qty=broker[symbol],
            ))
    for symbol in sorted(broker):
        if symbol not in internal:
            divergences.append(Divergence(
                "broker_only", symbol,
                "broker holds a position the bot does not track",
                internal_qty=None, broker_qty=broker[symbol],
            ))
    return divergences


def internal_open_positions() -> dict[str, float | None]:
    """Symbols the bot believes are open, from its ACTIVE open trades.

    Quantity is ``None``: Phase 11 trades do not carry a real order quantity yet
    (no signal→order wiring). Shadow trades are excluded — they never become
    real broker positions.
    """
    result: dict[str, float | None] = {}
    for trade in db.get_open_trades():
        if trade.track_mode != "active":
            continue
        signal = db.get_signal_by_id(trade.signal_id)
        if signal is None:
            continue
        result.setdefault(signal.ticker, None)
    return result


def reconcile(broker_client: Broker) -> ReconciliationReport:
    """Reconcile internal open trades against the broker. Broker is authoritative.

    Fail-soft: if the broker positions read fails the report is ``ok=False`` and
    no internal trade is flagged. Divergences are logged to stderr and returned;
    nothing is mutated.
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

    broker_map: dict[str, float] = {p.symbol: p.qty for p in positions.positions}
    internal_map = internal_open_positions()
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
