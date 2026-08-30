"""Broker-backed exposure snapshot tests."""

from datetime import UTC, datetime

from trading_bot import config, execution_exposure
from trading_bot.broker.base import AccountInfo, Position, PositionsResult
from trading_bot.models import LongTermPosition, OptionPosition, PendingOrder

_NOW = datetime(2026, 8, 14, 15, 0, tzinfo=UTC)


def _account() -> AccountInfo:
    return AccountInfo(ok=True, equity=100_000.0, cash=50_000.0)


def _option() -> OptionPosition:
    return OptionPosition(
        symbol="META260918C00500000", underlying="META", option_type="call",
        strike=500.0, expiry="2026-09-18", contracts=2.0, opened_at=_NOW,
    )


def _long_term() -> LongTermPosition:
    return LongTermPosition(
        ticker="AAPL", asset_class="stock", entry_price=200.0,
        entry_date=_NOW, qty=10.0,
    )


def _pending() -> PendingOrder:
    return PendingOrder(
        broker_order_id="order-1", client_order_id="client-1", ticker="MSFT",
        broker_symbol="MSFT", asset_class="stock", vehicle="shares",
        target_position_kind="long_term", side="buy", requested_qty=20.0,
        requested_limit_price=100.0, submitted_at=_NOW,
        intent_payload_json='{"intent_kind":"long_term"}', filled_qty=5.0,
    )


def test_build_existing_exposure_maps_underlyings_pools_and_working_remainder() -> None:
    positions = PositionsResult(ok=True, positions=[
        Position("META260918C00500000", 2.0, market_value=2_400.0),
        Position("AAPL", 10.0, market_value=2_100.0),
    ])
    result = execution_exposure.build_existing_exposure(
        _account(), positions, [_option()], [_long_term()], [_pending()],
    )

    assert result.ok is True
    assert result.exposure is not None
    assert result.exposure.gross_value == 6_000.0
    assert result.exposure.pool_values == {
        config.POOL_SWING: 2_400.0,
        config.POOL_LONG_TERM: 3_600.0,
    }
    assert result.exposure.ticker_values == {
        "META": 2_400.0, "AAPL": 2_100.0, "MSFT": 1_500.0,
    }


def test_build_existing_exposure_fails_closed_when_positions_unavailable() -> None:
    result = execution_exposure.build_existing_exposure(
        _account(), PositionsResult(ok=False, reason="timeout"), [], [], [],
    )
    assert result.ok is False
    assert result.exposure is None
    assert result.reason == "timeout"


def test_build_existing_exposure_fails_closed_when_holding_cannot_be_valued() -> None:
    result = execution_exposure.build_existing_exposure(
        _account(), PositionsResult(ok=True, positions=[Position("MYSTERY", 1.0)]),
        [], [], [],
    )
    assert result.ok is False
    assert "cannot value" in result.reason


def test_build_existing_exposure_uses_entry_cost_fallback() -> None:
    result = execution_exposure.build_existing_exposure(
        _account(),
        PositionsResult(
            ok=True,
            positions=[Position("MYSTERY", 2.0, avg_entry_price=125.0)],
        ),
        [], [], [],
    )
    assert result.exposure is not None
    assert result.exposure.gross_value == 250.0
    assert result.exposure.pool_values[config.POOL_SWING] == 250.0


def _swing_fallback_pending(payload: str | None = None) -> PendingOrder:
    """A swing shares-fallback entry: long_term target, but SWING capital.

    This is the options hierarchy's third tier (options_execution.py sets
    target_position_kind="long_term" with source="swing_fallback").
    """
    return PendingOrder(
        broker_order_id="order-2", client_order_id="client-2", ticker="META",
        broker_symbol="META", asset_class="stock", vehicle="shares",
        target_position_kind="long_term", side="buy", requested_qty=10.0,
        requested_limit_price=500.0, submitted_at=_NOW,
        intent_payload_json=(
            payload
            if payload is not None
            else '{"intent_kind":"shares_fallback","source":"swing_fallback"}'
        ),
        filled_qty=0.0,
    )


def test_working_swing_fallback_order_counts_against_the_swing_pool() -> None:
    """The pool must not flip at fill time.

    _pool_for_long_term already routes a FILLED swing-fallback position to
    POOL_SWING on source == "swing_fallback". Routing the working order on
    target_position_kind alone charged the same dollars to LONG_TERM until it
    filled, then moved them to SWING.
    """
    result = execution_exposure.build_existing_exposure(
        _account(), PositionsResult(ok=True, positions=[]), [], [],
        [_swing_fallback_pending()],
    )

    assert result.ok is True
    assert result.exposure is not None
    assert result.exposure.pool_values == {config.POOL_SWING: 5_000.0}


def test_filled_swing_fallback_position_agrees_with_the_working_order() -> None:
    """The same entry must sit in the same pool before and after it fills."""
    filled = LongTermPosition(
        ticker="META", asset_class="stock", entry_price=500.0,
        entry_date=_NOW, qty=10.0, source="swing_fallback",
    )
    positions = PositionsResult(ok=True, positions=[Position("META", 10.0, market_value=5_000.0)])

    result = execution_exposure.build_existing_exposure(
        _account(), positions, [], [filled], [],
    )

    assert result.ok is True
    assert result.exposure is not None
    assert result.exposure.pool_values == {config.POOL_SWING: 5_000.0}


def test_unreadable_intent_payload_falls_back_to_structural_routing() -> None:
    """A malformed payload must not break the snapshot (fail-soft contract)."""
    result = execution_exposure.build_existing_exposure(
        _account(), PositionsResult(ok=True, positions=[]), [], [],
        [_swing_fallback_pending(payload="not json at all")],
    )

    assert result.ok is True
    assert result.exposure is not None
    assert result.exposure.pool_values == {config.POOL_LONG_TERM: 5_000.0}


def test_option_entry_cost_fallback_applies_the_contract_multiplier() -> None:
    """An option position with no ``market_value`` must still be valued in
    dollars, not in per-share premium.

    ``avg_entry_price`` is the premium per share and ``qty`` is contracts, so
    the outlay is ``premium x OPTION_MULTIPLIER x contracts`` — exactly what the
    working-order branch below already computes for a pending option entry, and
    what ``options_execution._premium_cost`` computes at fire time. Without the
    multiplier a $4,000 option book was reported as $40, so the allocator's
    gross- and ticker-exposure caps saw essentially unlimited room.

    ``_position_body_error`` explicitly permits a missing ``market_value``
    (``if row.get(field) is not None``), so this branch is reachable.
    """
    positions = PositionsResult(ok=True, positions=[
        Position("META260918C00500000", 2.0, avg_entry_price=20.0),
    ])
    result = execution_exposure.build_existing_exposure(
        _account(), positions, [_option()], [], [],
    )

    assert result.ok is True
    assert result.exposure is not None
    # 20.00 premium x 100 x 2 contracts = $4,000 (was $40 before the fix).
    assert result.exposure.gross_value == 4_000.0
    assert result.exposure.ticker_values == {"META": 4_000.0}
    assert result.exposure.pool_values == {config.POOL_SWING: 4_000.0}


def test_share_entry_cost_fallback_is_unchanged() -> None:
    """Control: a plain equity holding must NOT gain a 100x multiplier."""
    positions = PositionsResult(ok=True, positions=[
        Position("AAPL", 10.0, avg_entry_price=200.0),
    ])
    result = execution_exposure.build_existing_exposure(
        _account(), positions, [], [_long_term()], [],
    )

    assert result.ok is True
    assert result.exposure is not None
    assert result.exposure.gross_value == 2_000.0


def test_option_market_value_is_not_multiplied_again() -> None:
    """The broker reports an option's market value with the multiplier already
    applied, so the primary branch must pass it through untouched."""
    positions = PositionsResult(ok=True, positions=[
        Position("META260918C00500000", 2.0, market_value=4_000.0,
                 avg_entry_price=20.0),
    ])
    result = execution_exposure.build_existing_exposure(
        _account(), positions, [_option()], [], [],
    )

    assert result.ok is True
    assert result.exposure is not None
    assert result.exposure.gross_value == 4_000.0


def test_partially_filled_option_entry_uses_the_multiplier_before_its_book_row() -> None:
    """A partially filled option entry has a broker position before the
    option_positions row exists — the working order names the kind."""
    pending_option = PendingOrder(
        broker_order_id="o-9", client_order_id="c-9", ticker="NVDA",
        broker_symbol="NVDA260918C00900000", asset_class="stock",
        vehicle="option", target_position_kind="option", side="buy",
        requested_qty=3.0, requested_limit_price=5.0, submitted_at=_NOW,
        intent_payload_json='{"intent_kind":"option"}', filled_qty=1.0,
    )
    positions = PositionsResult(ok=True, positions=[
        Position("NVDA260918C00900000", 1.0, avg_entry_price=5.0),
    ])
    result = execution_exposure.build_existing_exposure(
        _account(), positions, [], [], [pending_option],
    )

    assert result.ok is True
    assert result.exposure is not None
    # Filled contract: 5.00 x 100 x 1 = $500. Working remainder: 2 x 5.00 x 100
    # = $1,000 (the working-order branch already applied the multiplier).
    assert result.exposure.ticker_values == {"NVDA": 1_500.0}


def test_untracked_option_contract_still_gets_the_multiplier() -> None:
    """A broker-held OCC contract that no internal book claims must still be
    valued in dollars.

    Option-ness used to be decided ONLY from the internal books, so a contract
    with no ``option_positions`` row and no working entry order fell through to
    ``multiplier = 1.0`` and was valued at 1/100th of its dollars. The allocator
    then saw essentially unlimited gross- and ticker-exposure room and approved
    entries it should have skipped — failing OPEN, which this module's docstring
    says it never does.

    Reachable whenever the broker still shows a contract the books do not: an
    internal row closed ahead of the broker, a materialisation failure after a
    terminal fill, or a residual contract in the same paper account. The symbol
    itself is sufficient to identify it, which is what
    ``risk_of_ruin._generic_close_limit_price`` already does.
    """
    positions = PositionsResult(ok=True, positions=[
        Position("AMZN260320C00250000", 3.0, avg_entry_price=6.00),
    ])
    result = execution_exposure.build_existing_exposure(
        _account(), positions, [], [], [],
    )

    assert result.ok is True
    assert result.exposure is not None
    # 6.00 premium x 100 x 3 contracts = $1,800 (was $18 before the fix).
    assert result.exposure.gross_value == 1_800.0


def test_untracked_option_matches_the_same_position_when_tracked() -> None:
    """The snapshot must not depend on whether the internal book happens to
    carry the row: the dollars at risk are the same either way."""
    positions = PositionsResult(ok=True, positions=[
        Position("META260918C00500000", 2.0, avg_entry_price=20.0),
    ])
    untracked = execution_exposure.build_existing_exposure(
        _account(), positions, [], [], [],
    )
    tracked = execution_exposure.build_existing_exposure(
        _account(), positions, [_option()], [], [],
    )

    assert untracked.exposure is not None and tracked.exposure is not None
    assert untracked.exposure.gross_value == tracked.exposure.gross_value == 4_000.0


def test_untracked_equity_symbol_gets_no_multiplier() -> None:
    """Control: an unclaimed PLAIN symbol must not be mistaken for a contract."""
    positions = PositionsResult(ok=True, positions=[
        Position("AAPL", 10.0, avg_entry_price=200.0),
    ])
    result = execution_exposure.build_existing_exposure(
        _account(), positions, [], [], [],
    )

    assert result.ok is True
    assert result.exposure is not None
    assert result.exposure.gross_value == 2_000.0


def test_a_market_value_is_never_multiplied_for_an_untracked_contract() -> None:
    """Control: the broker reports an option's ``market_value`` with the
    multiplier ALREADY applied, so the symbol-based fallback must not double it."""
    positions = PositionsResult(ok=True, positions=[
        Position("AMZN260320C00250000", 3.0, avg_entry_price=6.00,
                 market_value=1_800.0),
    ])
    result = execution_exposure.build_existing_exposure(
        _account(), positions, [], [], [],
    )

    assert result.ok is True
    assert result.exposure is not None
    assert result.exposure.gross_value == 1_800.0
