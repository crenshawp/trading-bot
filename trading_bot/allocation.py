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

import sys
from dataclasses import dataclass, field
from datetime import datetime

from trading_bot import config, risk
from trading_bot.broker.base import AccountInfo

TIER_HIGH = "HIGH"
TIER_NORMAL = "NORMAL"


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
    # Phase 20: the ORIGINAL swing signal's resolver settlement deadline
    # (signal timestamp + the Phase 1 hold window) — the same moment the paper
    # trade on this signal expires. Populated by the live candidate source for
    # SWING-pool candidates only; LONG_TERM/CRYPTO have no time stop by design
    # and stay None. Carried, never computed here.
    hold_deadline: datetime | None = None


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
    # Phase 20: the candidate's hold_deadline, passed through UNCHANGED so the
    # shares-fallback record inherits the original swing time stop.
    hold_deadline: datetime | None = None


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


# ── Stage 2 inputs: composite confidence (documented, auditable, tunable) ────


def _is_long(direction: str) -> bool:
    return direction in ("call", "long")


# vol_regime → confidence contribution: low vol is a cleaner signal than high.
_VOL_TERMS: dict[str, float] = {"low": 1.0, "normal": 0.0, "high": -1.0, "unknown": 0.0}


def indicator_agreement(candidate: Candidate) -> int:
    """Phase 6 indicator-family AGREEMENT count in ``{0,1,2,3}``.

    Counts how many of three independent directional families CONFIRM the trade,
    oriented by side (vol_regime is excluded — it is its own composite factor, so
    counting it here would double-weight it):

    * trend strength — ADX ≥ ``CONF_ADX_TREND_MIN`` (a trustworthy trend exists;
      direction-agnostic);
    * momentum — RSI is not stretched against the trade (long: below overbought;
      short: above oversold);
    * volume flow — OBV confirms the side (long: positive; short: negative).

    A family whose reading is missing simply does not contribute.
    """
    agree = 0
    long = _is_long(candidate.direction)
    if candidate.adx is not None and candidate.adx >= config.CONF_ADX_TREND_MIN:
        agree += 1
    if candidate.rsi is not None and (
        (long and candidate.rsi < config.CONF_RSI_OVERBOUGHT)
        or (not long and candidate.rsi > config.CONF_RSI_OVERSOLD)
    ):
        agree += 1
    if candidate.obv is not None and (
        (long and candidate.obv > 0.0) or (not long and candidate.obv < 0.0)
    ):
        agree += 1
    return agree


def composite_confidence(candidate: Candidate) -> ConfidenceScore:
    """The single documented composite-confidence function.

    ``score = W_E·e + W_A·a + W_S·s + W_V·v`` with documented weights summing to
    1.0 (per-pair expectancy PRIMARY):

    * ``e`` — expectancy term: ``clamp(expectancy / CONF_EXPECTANCY_REF, -1, 1)``
      (None → 0). Primary factor, weight ``CONF_WEIGHT_EXPECTANCY``.
    * ``a`` — indicator agreement (0-3) scaled to ``[0,1]`` (``agreement/3``),
      weight ``CONF_WEIGHT_AGREEMENT``.
    * ``s`` — sentiment in ``[-1,1]`` (None → 0), SIGN-FLIPPED for short signals
      (bullish news is unfavorable to a short), weight ``CONF_WEIGHT_SENTIMENT``.
    * ``v`` — vol_regime term (low +1 / normal 0 / high -1 / unknown 0), weight
      ``CONF_WEIGHT_VOL``.

    Tier is HIGH only when the STRONGEST factors clear their bars AND the total
    clears the score gate: ``expectancy ≥ CONF_HIGH_EXPECTANCY`` AND
    ``agreement ≥ CONF_HIGH_AGREEMENT`` AND ``score ≥ CONF_HIGH_SCORE``; else
    NORMAL. Pure and fully hand-computable — no magic.
    """
    expectancy = candidate.expectancy
    if expectancy is None:
        e_term = 0.0
    else:
        e_term = max(-1.0, min(1.0, expectancy / config.CONF_EXPECTANCY_REF))

    agreement = indicator_agreement(candidate)
    a_term = agreement / 3.0

    sentiment = 0.0 if candidate.sentiment_score is None else candidate.sentiment_score
    s_term = sentiment if _is_long(candidate.direction) else -sentiment

    v_term = _VOL_TERMS.get(candidate.vol_regime, 0.0)

    score = (
        config.CONF_WEIGHT_EXPECTANCY * e_term
        + config.CONF_WEIGHT_AGREEMENT * a_term
        + config.CONF_WEIGHT_SENTIMENT * s_term
        + config.CONF_WEIGHT_VOL * v_term
    )

    is_high = (
        expectancy is not None
        and expectancy >= config.CONF_HIGH_EXPECTANCY
        and agreement >= config.CONF_HIGH_AGREEMENT
        and score >= config.CONF_HIGH_SCORE
    )
    return ConfidenceScore(
        score=score,
        tier=TIER_HIGH if is_high else TIER_NORMAL,
        expectancy_term=e_term,
        agreement=agreement,
        agreement_term=a_term,
        sentiment_term=s_term,
        vol_term=v_term,
    )


# ── Stage 1: hard eligibility filter ─────────────────────────────────────────


def filter_reason(
    candidate: Candidate, pool_capital: float, *, pool: str = config.POOL_SWING,
) -> str | None:
    """Return a drop reason if ``candidate`` is ineligible, else None.

    Hard gates, in order: ticker must be active, its (ticker, signal) pair must
    be enabled, no earnings blackout, and the pool-appropriate size must put at
    least ``MIN_DOLLAR_RISK`` at risk against the pool's capital (a signal that
    cannot be sized is ``unsizeable``; one whose risk falls below the floor is
    ``below_floor``). ``pool`` selects the sizing model (default SWING = Phase 7).
    Never raises — sizing is fail-soft.
    """
    if not candidate.ticker_active:
        return "ticker_inactive"
    if not candidate.pair_enabled:
        return "pair_muted"
    if candidate.earnings_blackout:
        return "earnings_blackout"
    ok, _pv, dollar_risk, _qty = _size_candidate(candidate, pool, pool_capital)
    if not ok or dollar_risk is None:
        return "unsizeable"
    if dollar_risk < config.MIN_DOLLAR_RISK:
        return "below_floor"
    return None


def _size_candidate(
    candidate: Candidate, pool: str, pool_capital: float,
) -> tuple[bool, float | None, float | None, float | None]:
    """Size a candidate for its pool: ``(ok, position_value, dollar_risk, qty)``.

    SWING uses the Phase 7 ATR/stop model. LONG_TERM / CRYPTO use the Phase 14
    diversification-weighted model (no tight stop → the notional weight IS both
    the position value and the amount at risk). Keeping the two sizing methods
    behind one dispatcher is what lets the pools coexist in one allocator.
    """
    if pool in (config.POOL_LONG_TERM, config.POOL_CRYPTO):
        lt = risk.position_size_long_term(pool_capital, candidate.entry)
        if not lt.ok or lt.dollars is None or lt.qty is None:
            return False, None, None, None
        return True, lt.dollars, lt.dollars, lt.qty
    size = risk.position_size(candidate.entry, candidate.atr, account=pool_capital)
    return size.ok, size.position_value, size.dollar_risk, size.recommended_size


# ── Stage 2/3: rank, then allocate top-down within each pool ─────────────────

_ALLOC_EPS = 1e-9   # float tolerance so an exact-fit position still places


def rank_candidates(eligible: list[tuple[Candidate, str]]) -> list[RankedCandidate]:
    """Score every eligible ``(candidate, pool)`` and rank globally by composite
    confidence (desc), tie-broken by expectancy (desc) then ticker (asc). Ranks
    are 1-based and assigned across ALL pools; allocation then walks each pool in
    this order."""
    scored = [(c, pool, composite_confidence(c)) for c, pool in eligible]
    scored.sort(
        key=lambda t: (
            -t[2].score,
            -(t[0].expectancy if t[0].expectancy is not None else -1e9),
            t[0].ticker,
        )
    )
    return [
        RankedCandidate(candidate=c, pool=pool, score=cs.score, tier=cs.tier, rank=i)
        for i, (c, pool, cs) in enumerate(scored, start=1)
    ]


def allocate(
    ranked: list[RankedCandidate],
    pool_caps: dict[str, tuple[float, float]],
) -> tuple[list[PlannedOrder], list[SkippedSignal], list[PoolAccounting]]:
    """Assign capital top-down by rank WITHIN each pool. Returns the planned
    orders (rank-ordered), the skipped list, and per-pool accounting.

    Each pool starts with ``(capital, cash)``. A candidate is sized via Phase 7
    against the pool's CAPITAL (so risk fraction and the per-position cap scale to
    the pool). It is placed only if both hold:

    * the tier deployment ceiling — ``capital × 50%`` for HIGH, ``× 30%`` for
      NORMAL — is not exceeded by the running deployed total, AND
    * the position cost fits the pool's remaining cash.

    A candidate that cannot be placed is SKIPPED (``capital-exhausted: …``),
    NEVER undersized — shrinking a position would break the Phase 7 risk math.
    """
    orders: list[PlannedOrder] = []
    skipped: list[SkippedSignal] = []
    pools_acct: list[PoolAccounting] = []

    for pool in config.POOLS:
        capital, cash = pool_caps.get(pool, (0.0, 0.0))
        deployed = 0.0
        cash_remaining = cash
        placed = 0

        for rc in [r for r in ranked if r.pool == pool]:   # already rank-ordered
            c = rc.candidate
            ok, pv, dollar_risk, qty = _size_candidate(c, pool, capital)
            if not ok or pv is None or dollar_risk is None or qty is None:
                skipped.append(SkippedSignal(
                    c.ticker, c.signal_type, pool, "allocate", "unsizeable",
                ))
                continue

            cap_frac = (
                config.POOL_DEPLOY_CAP_HIGH if rc.tier == TIER_HIGH
                else config.POOL_DEPLOY_CAP_NORMAL
            )
            ceiling = capital * cap_frac
            if deployed + pv > ceiling + _ALLOC_EPS:
                skipped.append(SkippedSignal(
                    c.ticker, c.signal_type, pool, "allocate",
                    f"capital-exhausted: {rc.tier} deploy cap "
                    f"({cap_frac:.0%} of pool)",
                ))
                continue
            if pv > cash_remaining + _ALLOC_EPS:
                skipped.append(SkippedSignal(
                    c.ticker, c.signal_type, pool, "allocate",
                    "capital-exhausted: pool cash",
                ))
                continue

            orders.append(PlannedOrder(
                rank=rc.rank, pool=pool, tier=rc.tier, ticker=c.ticker,
                signal_type=c.signal_type,
                side="buy" if _is_long(c.direction) else "sell",
                qty=qty, entry=c.entry, est_cost=pv, dollar_risk=dollar_risk,
                score=rc.score, hold_deadline=c.hold_deadline,
            ))
            deployed += pv
            cash_remaining -= pv
            placed += 1

        pools_acct.append(PoolAccounting(
            pool=pool, capital=capital, cash=cash, deployed=deployed,
            cash_remaining=cash_remaining, orders=placed,
        ))

    orders.sort(key=lambda o: o.rank)
    return orders, skipped, pools_acct


# ── Stage 4: emit the plan (NO EXECUTION) ────────────────────────────────────


def build_plan(
    candidates: list[Candidate],
    account: AccountInfo,
    *,
    split: dict[str, float] | None = None,
    entry_authorized: bool = True,
) -> AllocationResult:
    """Run the four-stage pipeline and EMIT an :class:`AllocationResult`.

    FILTER → RANK → ALLOCATE → PLAN. The returned plan is a pure data structure;
    this function NEVER calls ``broker.submit_order`` or otherwise executes —
    wiring the plan to the broker is deliberately deferred to a later phase until
    the allocation logic is proven (a test asserts zero submissions occur in this
    path).

    ``entry_authorized`` is the Phase 15 risk-of-ruin gate (callers pass
    ``risk_of_ruin.is_entry_authorized()``): when the ``new_position_entry``
    capability is revoked the plan is ENTRIES-EMPTY — every candidate is skipped
    with a logged reason. Existing positions are unaffected (their watchers run
    regardless of this gate).

    Fail-soft: if the paper account cannot be read (``account.ok`` False or no
    usable balances), an empty plan is returned with ``ok=False`` and a logged
    reason; every candidate is recorded as skipped (``account-unavailable``) so
    nothing is silently lost.
    """
    if not entry_authorized:
        print(
            "  allocate: new-position entry REVOKED (risk-of-ruin) - "
            "entries-empty plan",
            file=sys.stderr,
        )
        return AllocationResult(
            ok=True,
            plan=ExecutionPlan(),
            skipped=[
                SkippedSignal(
                    c.ticker, c.signal_type, None, "risk",
                    "entries-paused (risk-of-ruin)",
                )
                for c in candidates
            ],
            pools=[],
            note="entries paused (risk-of-ruin)",
        )

    caps = pool_capitals(account, split=split)
    if caps is None:
        reason = account.reason or "account unavailable"
        print(
            f"  allocate: account unavailable - empty plan ({reason})",
            file=sys.stderr,
        )
        skipped = [
            SkippedSignal(
                c.ticker, c.signal_type, route_pool(c), "account",
                "account-unavailable",
            )
            for c in candidates
        ]
        return AllocationResult(
            ok=False, plan=ExecutionPlan(), skipped=skipped, pools=[],
            note=f"account unavailable: {reason}",
        )

    # Stage 1 — FILTER (hard eligibility gates, with logged drop reasons).
    eligible: list[tuple[Candidate, str]] = []
    skipped = []
    for c in candidates:
        # Data-only signal types (the crypto SWING signals) can NEVER execute —
        # drop them before routing so they never reach a pool (Phase 14).
        if c.signal_type in config.DATA_ONLY_SIGNAL_TYPES:
            print(
                f"  allocate: drop {c.ticker}/{c.signal_type} "
                "(data-only - never executes)",
                file=sys.stderr,
            )
            skipped.append(SkippedSignal(
                c.ticker, c.signal_type, None, "filter",
                "data-only signal (never executes)",
            ))
            continue
        pool = route_pool(c)
        drop_reason = filter_reason(c, caps[pool][0], pool=pool)
        if drop_reason is not None:
            print(
                f"  allocate: drop {c.ticker}/{c.signal_type} ({drop_reason})",
                file=sys.stderr,
            )
            skipped.append(
                SkippedSignal(c.ticker, c.signal_type, pool, "filter", drop_reason)
            )
        else:
            eligible.append((c, pool))

    # Stage 2 — RANK by composite confidence.
    ranked = rank_candidates(eligible)

    # Stage 3 — ALLOCATE top-down within each pool.
    orders, alloc_skipped, pools_acct = allocate(ranked, caps)
    skipped.extend(alloc_skipped)

    # Stage 4 — PLAN. Return the data structure; execute NOTHING.
    return AllocationResult(
        ok=True, plan=ExecutionPlan(orders=orders), skipped=skipped,
        pools=pools_acct, note="",
    )


def sample_candidates() -> list[Candidate]:
    """A small, DOCUMENTED illustrative candidate set for plan inspection.

    Phase 12 does not persist the live "currently firing" signal set, so the
    ``allocate plan`` CLI runs the pipeline against this representative sample
    (clearly labelled in the output) while reading the REAL paper account. It
    spans both pools, several tiers, and an ineligible name so the plan and the
    skipped list both have content. Replacing this with a live signal gatherer is
    a later phase.
    """
    return [
        Candidate(
            ticker="META", signal_type="ema21_pullback", direction="call",
            asset_class="stock", entry=480.0, atr=8.0, expectancy=0.9,
            adx=27.0, rsi=52.0, obv=1_200_000.0, vol_regime="low",
            sentiment_score=0.4,
        ),
        Candidate(
            ticker="GOOGL", signal_type="ema21_pullback", direction="call",
            asset_class="stock", entry=175.0, atr=3.0, expectancy=0.3,
            adx=22.0, rsi=55.0, obv=500_000.0, vol_regime="normal",
            sentiment_score=0.1,
        ),
        Candidate(
            ticker="BLK", signal_type="ema21_pullback", direction="call",
            asset_class="stock", entry=820.0, atr=12.0, expectancy=-0.2,
            adx=15.0, rsi=68.0, obv=-200_000.0, vol_regime="high",
            sentiment_score=-0.3,
        ),
        Candidate(
            ticker="NVDA", signal_type="ema21_pullback", direction="call",
            asset_class="stock", entry=120.0, atr=4.0, ticker_active=False,
        ),
        Candidate(
            ticker="BTC-USD", signal_type="momentum_breakout", direction="long",
            asset_class="crypto", entry=64_000.0, atr=1_500.0, expectancy=0.6,
            adx=24.0, rsi=58.0, obv=30_000.0, vol_regime="normal",
            sentiment_score=0.2,
        ),
        Candidate(
            ticker="ETH-USD", signal_type="oversold_reversal", direction="long",
            asset_class="crypto", entry=3_400.0, atr=90.0, expectancy=0.5,
            adx=21.0, rsi=35.0, obv=10_000.0, vol_regime="low",
            sentiment_score=0.0,
        ),
    ]
