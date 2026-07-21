"""In-memory fake :class:`Broker` — Phase 11 test / demo double.

NOT a real broker and NOT a network client: it simulates a brokerage entirely
in memory so higher layers (and the offline ``demo_broker.py``) can exercise the
abstract interface without touching Alpaca. It honours the same fail-soft
contract — reads return ``ok=False`` when ``fail`` is set, ``submit_order``
returns structured rejections — so tests can drive every code path the real
adapter exposes.
"""

from __future__ import annotations

from datetime import UTC, datetime

from trading_bot.broker.base import (
    ORDER_LOOKUP_FOUND,
    ORDER_LOOKUP_NOT_FOUND,
    ORDER_LOOKUP_UNAVAILABLE,
    STATUS_CANCELED,
    STATUS_ERROR,
    STATUS_FILLED,
    STATUS_NEW,
    STATUS_REJECTED,
    VALID_ORDER_TYPES,
    VALID_SIDES,
    VALID_TIF,
    AccountInfo,
    Broker,
    OrderLookupResult,
    OrderResult,
    OrdersResult,
    Position,
    PositionsResult,
)


class FakeBroker(Broker):
    """A controllable in-memory broker for tests and the offline demo.

    Controls:
      * ``fail`` — when True every read returns an ``ok=False`` result and every
        submission returns a transport ``error`` (simulates an outage).
      * ``reject_reason`` — when set, ``submit_order`` returns a structured
        rejection with this reason (simulates e.g. insufficient buying power).
      * ``auto_fill`` — when True a submitted order is immediately ``filled``,
        otherwise it rests as ``new``.
    """

    def __init__(
        self,
        *,
        buying_power: float = 100_000.0,
        cash: float = 100_000.0,
        equity: float = 100_000.0,
        fail: bool = False,
        reject_reason: str | None = None,
        auto_fill: bool = False,
    ) -> None:
        self.fail = fail
        self.reject_reason = reject_reason
        self.auto_fill = auto_fill
        self._account = AccountInfo(
            ok=True, account_number="FAKE-PAPER", buying_power=buying_power,
            cash=cash, equity=equity, status="ACTIVE",
        )
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, OrderResult] = {}
        self._seq = 0

    # ── test helpers (not part of the Broker interface) ──────────────────────

    def set_position(
        self, symbol: str, qty: float, *, side: str = "long",
        avg_entry_price: float | None = None,
    ) -> None:
        """Seed a held position (as if a prior order filled)."""
        self._positions[symbol] = Position(
            symbol=symbol, qty=qty, side=side, avg_entry_price=avg_entry_price,
        )

    # ── Broker interface ─────────────────────────────────────────────────────

    def get_account(self) -> AccountInfo:
        if self.fail:
            return AccountInfo(ok=False, reason="fake broker: simulated outage")
        return self._account

    def get_positions(self) -> PositionsResult:
        if self.fail:
            return PositionsResult(ok=False, reason="fake broker: simulated outage")
        return PositionsResult(ok=True, positions=list(self._positions.values()))

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        *,
        order_type: str = "limit",
        limit_price: float | None = None,
        time_in_force: str = "day",
        client_order_id: str | None = None,
    ) -> OrderResult:
        if self.fail:
            return OrderResult(
                ok=False, status=STATUS_ERROR, symbol=symbol,
                reason="fake broker: simulated outage",
            )
        if side not in VALID_SIDES:
            return OrderResult(
                ok=False, status=STATUS_REJECTED, symbol=symbol,
                reason=f"invalid side {side!r}",
            )
        if order_type not in VALID_ORDER_TYPES:
            return OrderResult(
                ok=False, status=STATUS_REJECTED, symbol=symbol,
                reason=f"invalid order_type {order_type!r}",
            )
        if time_in_force not in VALID_TIF:
            return OrderResult(
                ok=False, status=STATUS_REJECTED, symbol=symbol,
                reason=f"invalid time_in_force {time_in_force!r}",
            )
        if self.reject_reason is not None:
            return OrderResult(
                ok=False, status=STATUS_REJECTED, symbol=symbol,
                reason=self.reject_reason,
            )

        self._seq += 1
        order_id = f"fake-{self._seq}"
        status = STATUS_FILLED if self.auto_fill else STATUS_NEW
        order = OrderResult(
            ok=True, status=status, order_id=order_id,
            client_order_id=client_order_id, symbol=symbol, qty=qty,
            filled_qty=qty if self.auto_fill else 0.0,
            filled_avg_price=limit_price if self.auto_fill else None,
            side=side, order_type=order_type, time_in_force=time_in_force,
            limit_price=limit_price, submitted_at=datetime.now(UTC).isoformat(),
            raw_status=status,
        )
        self._orders[order_id] = order
        if self.auto_fill:
            self._positions[symbol] = Position(
                symbol=symbol, qty=qty, side="long" if side == "buy" else "short",
                avg_entry_price=limit_price,
            )
        return order

    def get_order(self, order_id: str) -> OrderResult:
        if self.fail:
            return OrderResult(
                ok=False, status=STATUS_ERROR, order_id=order_id,
                reason="fake broker: simulated outage",
            )
        order = self._orders.get(order_id)
        if order is None:
            return OrderResult(
                ok=False, status=STATUS_ERROR, order_id=order_id,
                reason="order not found",
            )
        return order

    def get_order_by_client_order_id(
        self, client_order_id: str,
    ) -> OrderLookupResult:
        if client_order_id.startswith("legacy-"):
            return OrderLookupResult(
                outcome=ORDER_LOOKUP_UNAVAILABLE,
                reason="synthetic legacy client order IDs are not provider identities",
            )
        if self.fail:
            return OrderLookupResult(
                outcome=ORDER_LOOKUP_UNAVAILABLE,
                reason="fake broker: simulated outage",
            )
        for order in self._orders.values():
            if order.client_order_id == client_order_id:
                return OrderLookupResult(
                    outcome=ORDER_LOOKUP_FOUND,
                    order=order,
                )
        return OrderLookupResult(
            outcome=ORDER_LOOKUP_NOT_FOUND,
            reason="order not found",
        )

    def cancel_order(self, order_id: str) -> OrderResult:
        if self.fail:
            return OrderResult(
                ok=False, status=STATUS_ERROR, order_id=order_id,
                reason="fake broker: simulated outage",
            )
        order = self._orders.get(order_id)
        if order is None:
            return OrderResult(
                ok=False, status=STATUS_ERROR, order_id=order_id,
                reason="order not found",
            )
        canceled = OrderResult(
            ok=True, status=STATUS_CANCELED, order_id=order_id,
            client_order_id=order.client_order_id, symbol=order.symbol,
            qty=order.qty, filled_qty=order.filled_qty,
            filled_avg_price=order.filled_avg_price, side=order.side,
            order_type=order.order_type, time_in_force=order.time_in_force,
            limit_price=order.limit_price, submitted_at=order.submitted_at,
            raw_status=STATUS_CANCELED,
        )
        self._orders[order_id] = canceled
        return canceled

    def list_orders(self, status: str = "open") -> OrdersResult:
        if self.fail:
            return OrdersResult(ok=False, reason="fake broker: simulated outage")
        orders = list(self._orders.values())
        if status == "open":
            orders = [o for o in orders if o.status in (STATUS_NEW,)]
        elif status == "closed":
            orders = [o for o in orders if o.status not in (STATUS_NEW,)]
        # status == "all" (or anything else) returns everything
        return OrdersResult(ok=True, orders=orders)
