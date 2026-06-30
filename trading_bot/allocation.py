"""Capital allocation engine — Phase 12. PLAN-ONLY.

Turns a set of simultaneously-firing signals into an ordered EXECUTION PLAN,
respecting capital limits, confidence tiers, and pool boundaries. It PRODUCES A
PLAN — it never executes. The plan is a data structure handed back to the
caller; wiring it to the Phase 11 broker's ``submit_order`` is deliberately
deferred until the allocation logic is proven (a test asserts zero broker
submissions occur anywhere in the allocate path).

FOUR LEGIBLE STAGES (kept distinct, NOT one blended score):

1. FILTER  — hard eligibility gates: ticker active AND pair enabled AND no
   earnings blackout AND the volatility-normalized size clears the minimum
   dollar-risk floor. Ineligible candidates are dropped with a logged reason.
2. RANK    — eligible candidates ranked by a documented COMPOSITE CONFIDENCE
   score (see :func:`composite_confidence`), each tiered HIGH or NORMAL.
3. ALLOCATE— assign capital top-down by rank WITHIN each pool until the pool's
   tier deployment cap (50% HIGH / 30% NORMAL) or its cash is exhausted.
   Volatility-normalized Phase 7 sizing sets each position's size; a candidate
   that cannot be sized at/above the floor in the remaining capital is SKIPPED,
   never silently undersized.
4. PLAN    — emit the ordered :class:`ExecutionPlan` plus the :class:`SkippedSignal`
   list (each with a reason). Nothing executes.

Everything here is PURE: the firing signals, their context, and the paper
account snapshot are injected, never queried. Fail-soft: an unreadable account
yields an empty plan with a logged reason; it never raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trading_bot import config
from trading_bot.broker.base import AccountInfo


@dataclass(frozen=True)
class Candidate:
    """One firing signal plus the precomputed context the pipeline reads.

    Gate inputs (``ticker_active`` / ``pair_enabled`` / ``earnings_blackout``)
    and the scoring inputs (expectancy, the Phase 6 indicator readings,
    sentiment, vol_regime) are snapshotted by the caller so the pipeline stays
    pure and hand-testable.
    """

    ticker: str
    signal_type: str
    direction: str                 # 'call' | 'put' | 'long' | 'short'
    asset_class: str               # 'stock' | 'crypto' (pool routing fallback)
    entry: float
    atr: float | None
    # Stage 1 gate inputs
    ticker_active: bool = True
    pair_enabled: bool = True
    earnings_blackout: bool = False
    # Stage 2 scoring inputs
    expectancy: float | None = None    # per-pair windowed expectancy (% per trade)
    rsi: float | None = None
    adx: float | None = None
    obv: float | None = None
    vol_regime: str = "unknown"        # low | normal | high | unknown
    sentiment_score: float | None = None   # -1..+1 (advisory)
    concentration: str = "unknown"     # Phase 6 label, carried for completeness


@dataclass(frozen=True)
class ConfidenceScore:
    """The composite confidence for one candidate and its HIGH/NORMAL tier.

    The component terms are exposed so the score is fully auditable — no magic
    number, every contribution is inspectable.
    """

    score: float
    tier: str                      # 'HIGH' | 'NORMAL'
    expectancy_term: float
    agreement: int                 # 0-3 indicator-family agreement count
    agreement_term: float
    sentiment_term: float
    vol_term: float


@dataclass(frozen=True)
class RankedCandidate:
    """An eligible candidate with its score, tier, pool, and global rank."""

    candidate: Candidate
    pool: str
    score: float
    tier: str                      # 'HIGH' | 'NORMAL'
    rank: int                      # 1-based, across all eligible candidates


@dataclass(frozen=True)
class PlannedOrder:
    """One line of the execution plan. A PLAN entry — nothing is submitted."""

    rank: int
    pool: str
    tier: str
    ticker: str
    signal_type: str
    side: str                      # 'buy' | 'sell'
    qty: float                     # volatility-normalized size (units)
    entry: float
    est_cost: float                # position value (qty × entry)
    dollar_risk: float             # $ at risk if the stop is hit
    score: float


@dataclass(frozen=True)
class SkippedSignal:
    """A candidate that did not make the plan, with the stage and reason."""

    ticker: str
    signal_type: str
    pool: str | None
    stage: str                     # 'filter' | 'allocate' | 'account'
    reason: str


@dataclass(frozen=True)
class PoolAccounting:
    """One pool's capital accounting after allocation."""

    pool: str
    capital: float                 # cap base (for the % deployment ceilings)
    cash: float                    # spend base at the start
    deployed: float                # capital placed into the plan
    cash_remaining: float
    orders: int


@dataclass(frozen=True)
class ExecutionPlan:
    """The ordered plan: a list of :class:`PlannedOrder`. NOT executed."""

    orders: list[PlannedOrder] = field(default_factory=list)

    @property
    def total_est_cost(self) -> float:
        return sum(o.est_cost for o in self.orders)

    @property
    def total_dollar_risk(self) -> float:
        return sum(o.dollar_risk for o in self.orders)


@dataclass(frozen=True)
class AllocationResult:
    """The full result of an allocation pass: the plan, the skipped list, the
    per-pool accounting, and an ``ok`` flag (False when the account was
    unavailable, in which case the plan is empty and ``note`` says why)."""

    ok: bool
    plan: ExecutionPlan
    skipped: list[SkippedSignal] = field(default_factory=list)
    pools: list[PoolAccounting] = field(default_factory=list)
    note: str = ""


# ── pool model ───────────────────────────────────────────────────────────────


def route_pool(candidate: Candidate) -> str:
    """Route a candidate to its pool by ``signal_type`` (fallback ``asset_class``).

    Unknown signal types fall back to the asset class: crypto → CRYPTO, anything
    else → SWING. LONG_TERM has no automatic routing yet (buy-hold; later phase).
    """
    pool = config.SIGNAL_TYPE_TO_POOL.get(candidate.signal_type)
    if pool is not None:
        return pool
    return config.POOL_CRYPTO if candidate.asset_class == "crypto" else config.POOL_SWING


def available_capital(account: AccountInfo) -> tuple[float, float] | None:
    """Return ``(cap_base, cash_base)`` from a paper account, or None if unusable.

    ``cap_base`` (preferring equity) is what the % deployment ceilings are taken
    against; ``cash_base`` (preferring cash) is what positions are actually spent
    from. A non-ok account, or one with no usable numbers, returns None so the
    caller can fail soft to an empty plan.
    """
    if not account.ok:
        return None
    cap_base = (
        account.equity if account.equity is not None
        else account.cash if account.cash is not None
        else account.buying_power
    )
    cash_base = (
        account.cash if account.cash is not None
        else account.buying_power if account.buying_power is not None
        else account.equity
    )
    if cap_base is None or cash_base is None or cap_base <= 0.0 or cash_base <= 0.0:
        return None
    return cap_base, cash_base


def pool_capitals(
    account: AccountInfo, *, split: dict[str, float] | None = None,
) -> dict[str, tuple[float, float]] | None:
    """Per-pool ``(capital, cash)`` from the account split, or None if unusable."""
    bases = available_capital(account)
    if bases is None:
        return None
    cap_base, cash_base = bases
    weights = split if split is not None else config.POOL_CAPITAL_SPLIT
    return {
        pool: (cap_base * weights.get(pool, 0.0), cash_base * weights.get(pool, 0.0))
        for pool in config.POOLS
    }
