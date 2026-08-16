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
