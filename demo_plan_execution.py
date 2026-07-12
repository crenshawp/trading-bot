"""Offline demo of the Phase 16 plan-execution bridge (operator-gated, PAPER).

Run with: ``python demo_plan_execution.py``

No network whatsoever: a hand-built mixed SWING/LONG_TERM plan + a
:class:`FakeBroker` + a throwaway SQLite database. It shows an AUTHORIZED run
routing each planned order to its pool's existing submission path (SWING →
the Phase 13 option hierarchy, LONG_TERM → the Phase 14 fractional-share
entry), the idempotency guard skipping a re-run, and a REFUSED run when the
Phase 15 ``new_position_entry`` capability is revoked. The real path is
``python -m trading_bot allocate execute --confirm`` against Alpaca PAPER.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from trading_bot import config

config.DB_PATH = Path(tempfile.mkdtemp()) / "demo_plan_execution.db"

from trading_bot import db, risk_of_ruin  # noqa: E402
from trading_bot import plan_execution as pe  # noqa: E402
from trading_bot.allocation import ExecutionPlan, PlannedOrder  # noqa: E402
from trading_bot.broker.fake import FakeBroker  # noqa: E402
from trading_bot.broker.options import OptionChainResult, OptionContract  # noqa: E402

_NOW = datetime(2026, 7, 10, 15, 0, tzinfo=UTC)


def _chain(_underlying: str) -> OptionChainResult:
    expiry = (_NOW + timedelta(days=30)).date().isoformat()
    return OptionChainResult(ok=True, contracts=[OptionContract(
        symbol="META-call-480", underlying="META", option_type="call",
        strike=480.0, expiry=expiry, delta=0.70, theta=-0.05, vega=0.1,
        gamma=0.01, open_interest=500, bid=4.9, ask=5.1, mid=5.0,
    )])


def _mixed_plan() -> ExecutionPlan:
    return ExecutionPlan(orders=[
        PlannedOrder(
            rank=1, pool=config.POOL_SWING, tier="HIGH", ticker="META",
            signal_type="ema21_pullback", side="buy", qty=10.0, entry=480.0,
            est_cost=600.0, dollar_risk=150.0, score=0.82,
        ),
        PlannedOrder(
            rank=2, pool=config.POOL_LONG_TERM, tier="NORMAL", ticker="AAPL",
            signal_type="long_term_stock", side="buy", qty=45.0, entry=100.0,
            est_cost=4_500.0, dollar_risk=4_500.0, score=0.41,
        ),
    ])


def _show(run: pe.ExecutionRunResult) -> None:
    if not run.ok:
        print(f"  REFUSED - {run.note}")
        return
    for e in run.executions:
        print(
            f"  {e.order.ticker:<8} {e.order.pool:<10} -> {e.status:<10} "
            f"vehicle={e.vehicle or '-':<12} ref={e.order_ref or '-':<8} "
            f"({e.reason})"
        )


def main() -> None:
    db.init_db()
    broker = FakeBroker()
    plan = _mixed_plan()

    print("=== Authorized run: mixed SWING/LONG_TERM plan submits by pool ===")
    _show(pe.execute_plan(broker, plan, now=_NOW, option_chain_fetch=_chain))
    print(f"  open option positions:    {len(db.get_open_option_positions())}")
    print(f"  open long-term positions: {len(db.get_open_long_term_positions())}")

    print("\n=== Re-run same cycle: idempotency guard skips everything ===")
    _show(pe.execute_plan(
        broker, plan, now=_NOW + timedelta(minutes=5), option_chain_fetch=_chain,
    ))

    print("\n=== Unauthorized run: Phase 15 revocation refuses the batch ===")
    risk_of_ruin.revoke(config.ENTRY_CAPABILITY, "tier1: 7 consecutive losses")
    _show(pe.execute_plan(
        broker, plan, now=_NOW + timedelta(days=1), option_chain_fetch=_chain,
    ))
    risk_of_ruin.authorize(config.ENTRY_CAPABILITY)

    print(f"\n  audit rows in plan_executions: {len(db.get_plan_executions())}")


if __name__ == "__main__":
    main()
