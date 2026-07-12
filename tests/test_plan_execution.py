"""Tests for trading_bot.plan_execution (Phase 16 — plan → broker bridge)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from trading_bot import db
from trading_bot import plan_execution as pe
from trading_bot.models import PlanExecution

# 11:00 ET on a July (EDT, UTC-4) trading day.
_NOW = datetime(2026, 7, 10, 15, 0, tzinfo=UTC)


def _execution(
    *,
    ticker: str = "META",
    pool: str = "SWING",
    status: str = pe.STATUS_SUBMITTED,
    executed_at: datetime = _NOW,
    order_ref: str | None = "fake-1",
) -> PlanExecution:
    return PlanExecution(
        plan_id=pe.make_plan_id(executed_at),
        executed_at=executed_at,
        ticker=ticker,
        pool=pool,
        status=status,
        signal_type="ema21_pullback",
        side="buy",
        qty=2.0,
        vehicle="option_full",
        order_ref=order_ref,
        reason="ok",
    )


# ───────────────────────── audit-table roundtrip ─────────────────────────────


def test_insert_and_read_roundtrip(tmp_db: Path) -> None:
    row_id = db.insert_plan_execution(_execution())
    rows = db.get_plan_executions()
    assert len(rows) == 1
    got = rows[0]
    assert got.id == row_id
    assert got.ticker == "META"
    assert got.pool == "SWING"
    assert got.status == "submitted"
    assert got.signal_type == "ema21_pullback"
    assert got.side == "buy"
    assert got.qty == 2.0
    assert got.vehicle == "option_full"
    assert got.order_ref == "fake-1"
    assert got.reason == "ok"
    assert got.executed_at == _NOW
    assert got.plan_id == pe.make_plan_id(_NOW)


def test_insert_invalid_status_raises(tmp_db: Path) -> None:
    with pytest.raises(ValueError, match="Invalid plan-execution status"):
        db.insert_plan_execution(_execution(status="filled"))


def test_get_plan_executions_newest_first(tmp_db: Path) -> None:
    earlier = datetime(2026, 7, 10, 14, 0, tzinfo=UTC)
    db.insert_plan_execution(_execution(ticker="AAA", executed_at=earlier))
    db.insert_plan_execution(_execution(ticker="BBB", executed_at=_NOW))
    rows = db.get_plan_executions()
    assert [r.ticker for r in rows] == ["BBB", "AAA"]


# ───────────────────────── idempotency guard ─────────────────────────────────


def test_already_executed_blocks_same_cycle(tmp_db: Path) -> None:
    db.insert_plan_execution(_execution())
    assert pe.already_executed("META", "SWING", _NOW) is True


def test_already_executed_is_per_ticker_and_pool(tmp_db: Path) -> None:
    db.insert_plan_execution(_execution())
    assert pe.already_executed("GOOGL", "SWING", _NOW) is False
    assert pe.already_executed("META", "LONG_TERM", _NOW) is False


def test_already_executed_ignores_non_submitted(tmp_db: Path) -> None:
    for status in (pe.STATUS_REJECTED, pe.STATUS_ERROR, pe.STATUS_SKIPPED):
        db.insert_plan_execution(_execution(status=status, order_ref=None))
    assert pe.already_executed("META", "SWING", _NOW) is False


def test_already_executed_prior_cycle_does_not_block(tmp_db: Path) -> None:
    # 22:00 ET the previous day — a prior cycle, so today may execute.
    yesterday_et = datetime(2026, 7, 10, 2, 0, tzinfo=UTC)
    db.insert_plan_execution(_execution(executed_at=yesterday_et))
    assert pe.already_executed("META", "SWING", _NOW) is False


def test_cycle_start_is_midnight_et() -> None:
    start = pe.cycle_start(_NOW)
    # Midnight ET on 2026-07-10 is 04:00 UTC (EDT).
    assert start == datetime(2026, 7, 10, 4, 0, tzinfo=UTC)


def test_make_plan_id_derives_from_timestamp() -> None:
    assert pe.make_plan_id(_NOW) == "plan-20260710T150000Z"
