"""Offline demo of the Phase 12 capital allocation engine (PLAN-ONLY).

Run with: ``python demo_allocate.py``

No network whatsoever: a :class:`FakeBroker` supplies paper account state, and the
four-stage pipeline (FILTER → RANK → ALLOCATE → PLAN) runs against an
illustrative candidate set. It EXECUTES NOTHING — the plan is printed and the
broker shows zero submitted orders. The real ``allocate plan`` CLI runs the same
pipeline against the live PAPER account; only the account source changes here.
"""

from __future__ import annotations

from trading_bot import allocation
from trading_bot.broker.fake import FakeBroker


def main() -> None:
    broker = FakeBroker(buying_power=50_000.0, cash=50_000.0, equity=50_000.0)
    account = broker.get_account()
    candidates = allocation.sample_candidates()
    result = allocation.build_plan(candidates, account)

    print("=== Pools (capital split of the paper account) ===")
    for p in result.pools:
        print(
            f"  {p.pool:<10} capital=${p.capital:>10,.0f}  cash=${p.cash:>10,.0f}  "
            f"deployed=${p.deployed:>10,.0f}  orders={p.orders}"
        )

    print("\n=== Execution plan (ordered by rank) - NOT executed ===")
    if not result.plan.orders:
        print("  (no orders)")
    for o in result.plan.orders:
        print(
            f"  #{o.rank} {o.pool:<8} {o.tier:<6} {o.ticker:<8} {o.side:<4} "
            f"qty={o.qty:>10.4f}  est=${o.est_cost:>9,.0f}  "
            f"$risk=${o.dollar_risk:>7,.0f}  score={o.score:.3f}"
        )
    print(
        f"  TOTAL est cost=${result.plan.total_est_cost:,.0f}  "
        f"total $risk=${result.plan.total_dollar_risk:,.0f}"
    )

    print("\n=== Skipped (with stage + reason) ===")
    for s in result.skipped:
        print(f"  {s.ticker:<8} [{s.stage}] {s.reason}")

    print("\n=== Proof of NO execution ===")
    print(f"  broker orders submitted: {len(broker._orders)}  (PLAN ONLY)")


if __name__ == "__main__":
    main()
