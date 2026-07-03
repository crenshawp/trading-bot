"""Tests for the capital allocation engine (Phase 12, PLAN-ONLY).

Pure pipeline, mocked data only. The composite score and tiering are checked
against hand-computed values; allocation is checked for pool caps, cash
exhaustion, and skip-not-undersize; and a dedicated test asserts the allocate
path submits ZERO broker orders.
"""

from __future__ import annotations

import pytest

from trading_bot import allocation, config
from trading_bot.allocation import Candidate, ExecutionPlan, PlannedOrder
from trading_bot.broker.base import AccountInfo
from trading_bot.broker.fake import FakeBroker


def _candidate(**kw: object) -> Candidate:
    base: dict[str, object] = dict(
        ticker="AAPL", signal_type="ema21_pullback", direction="call",
        asset_class="stock", entry=100.0, atr=2.0,
    )
    base.update(kw)
    return Candidate(**base)  # type: ignore[arg-type]


# ───────────────────────── pool routing ─────────────────────────────────────


def test_route_pool_by_signal_type() -> None:
    assert allocation.route_pool(_candidate(signal_type="ema21_pullback")) == config.POOL_SWING
    assert allocation.route_pool(
        _candidate(signal_type="oversold_reversal", asset_class="crypto")
    ) == config.POOL_CRYPTO
    assert allocation.route_pool(
        _candidate(signal_type="momentum_breakout", asset_class="crypto")
    ) == config.POOL_CRYPTO


def test_route_pool_falls_back_to_asset_class() -> None:
    assert allocation.route_pool(
        _candidate(signal_type="mystery", asset_class="crypto")
    ) == config.POOL_CRYPTO
    assert allocation.route_pool(
        _candidate(signal_type="mystery", asset_class="stock")
    ) == config.POOL_SWING


# ───────────────────────── account capital ──────────────────────────────────


def test_available_capital_prefers_equity_and_cash() -> None:
    acct = AccountInfo(ok=True, equity=10_000.0, cash=8_000.0, buying_power=16_000.0)
    bases = allocation.available_capital(acct)
    assert bases == (10_000.0, 8_000.0)   # equity for cap base, cash for spend


def test_available_capital_falls_back_when_fields_missing() -> None:
    acct = AccountInfo(ok=True, equity=None, cash=None, buying_power=5_000.0)
    assert allocation.available_capital(acct) == (5_000.0, 5_000.0)


def test_available_capital_none_when_unavailable_or_zero() -> None:
    assert allocation.available_capital(AccountInfo(ok=False)) is None
    assert allocation.available_capital(
        AccountInfo(ok=True, equity=0.0, cash=0.0, buying_power=0.0)
    ) is None


def test_pool_capitals_split() -> None:
    acct = AccountInfo(ok=True, equity=10_000.0, cash=10_000.0)
    caps = allocation.pool_capitals(acct)
    assert caps is not None
    assert caps[config.POOL_SWING] == (5_000.0, 5_000.0)
    assert caps[config.POOL_LONG_TERM] == (3_000.0, 3_000.0)
    assert caps[config.POOL_CRYPTO] == (2_000.0, 2_000.0)


def test_pool_capitals_none_when_account_unavailable() -> None:
    assert allocation.pool_capitals(AccountInfo(ok=False)) is None


# ───────────────────────── plan dataclasses ─────────────────────────────────


def test_execution_plan_totals() -> None:
    plan = ExecutionPlan(orders=[
        PlannedOrder(1, "SWING", "HIGH", "AAPL", "ema21_pullback", "buy",
                     10.0, 100.0, 1000.0, 50.0, 0.8),
        PlannedOrder(2, "SWING", "NORMAL", "MSFT", "ema21_pullback", "buy",
                     5.0, 200.0, 1000.0, 30.0, 0.4),
    ])
    assert plan.total_est_cost == 2000.0
    assert plan.total_dollar_risk == 80.0


def test_execution_plan_empty_defaults() -> None:
    plan = ExecutionPlan()
    assert plan.orders == []
    assert plan.total_est_cost == 0.0


# ───────────────────────── indicator agreement ──────────────────────────────


def test_indicator_agreement_long_all_three() -> None:
    c = _candidate(direction="call", adx=25.0, rsi=50.0, obv=100.0)
    assert allocation.indicator_agreement(c) == 3


def test_indicator_agreement_short_orientation() -> None:
    # For a short: RSI above oversold confirms, OBV negative confirms.
    c = _candidate(direction="short", adx=25.0, rsi=60.0, obv=-50.0)
    assert allocation.indicator_agreement(c) == 3
    # A long with the same readings: OBV negative does NOT confirm a long.
    c2 = _candidate(direction="call", adx=25.0, rsi=60.0, obv=-50.0)
    assert allocation.indicator_agreement(c2) == 2


def test_indicator_agreement_missing_and_weak() -> None:
    c = _candidate(direction="call", adx=10.0, rsi=80.0, obv=None)
    assert allocation.indicator_agreement(c) == 0   # weak ADX, overbought RSI, no OBV


# ───────────────────────── composite confidence (hand values) ───────────────


def test_composite_confidence_hand_value_high() -> None:
    c = _candidate(
        direction="call", expectancy=1.0, adx=25.0, rsi=50.0, obv=100.0,
        sentiment_score=0.5, vol_regime="low",
    )
    cs = allocation.composite_confidence(c)
    # e=1.0, a=3/3=1.0, s=0.5, v=1.0 ->
    # 0.50*1 + 0.25*1 + 0.15*0.5 + 0.10*1 = 0.925
    assert cs.score == pytest.approx(0.925)
    assert cs.agreement == 3
    assert cs.tier == allocation.TIER_HIGH


def test_composite_confidence_high_requires_strong_expectancy() -> None:
    # Strong everything else but weak expectancy -> score may clear, tier NORMAL.
    c = _candidate(
        direction="call", expectancy=0.2, adx=25.0, rsi=50.0, obv=100.0,
        sentiment_score=1.0, vol_regime="low",
    )
    cs = allocation.composite_confidence(c)
    # 0.50*0.2 + 0.25*1 + 0.15*1 + 0.10*1 = 0.60 (>= score gate) but expectancy < 0.5
    assert cs.score == pytest.approx(0.60)
    assert cs.tier == allocation.TIER_NORMAL


def test_composite_confidence_short_inverts_sentiment() -> None:
    long = _candidate(direction="call", sentiment_score=0.8, vol_regime="normal")
    short = _candidate(direction="short", sentiment_score=0.8, vol_regime="normal")
    assert allocation.composite_confidence(long).sentiment_term == 0.8
    assert allocation.composite_confidence(short).sentiment_term == -0.8


def test_composite_confidence_none_inputs_are_neutral() -> None:
    c = _candidate(
        direction="call", expectancy=None, adx=None, rsi=None, obv=None,
        sentiment_score=None, vol_regime="unknown",
    )
    cs = allocation.composite_confidence(c)
    assert cs.score == 0.0
    assert cs.agreement == 0
    assert cs.tier == allocation.TIER_NORMAL


def test_composite_confidence_expectancy_clamped() -> None:
    c = _candidate(direction="call", expectancy=5.0, vol_regime="normal")
    # expectancy term clamps at 1.0 -> contributes 0.50 only.
    assert allocation.composite_confidence(c).expectancy_term == 1.0


# ───────────────────────── Stage 1 eligibility filter ───────────────────────


def test_filter_drops_each_ineligibility_reason() -> None:
    cap = 5_000.0
    assert allocation.filter_reason(_candidate(ticker_active=False), cap) == "ticker_inactive"
    assert allocation.filter_reason(_candidate(pair_enabled=False), cap) == "pair_muted"
    assert allocation.filter_reason(_candidate(earnings_blackout=True), cap) == "earnings_blackout"
    assert allocation.filter_reason(_candidate(atr=None), cap) == "unsizeable"


def test_filter_below_floor_when_pool_capital_tiny() -> None:
    # 1% risk of a $500 pool == $5 dollar-risk, below the $10 floor.
    assert allocation.filter_reason(_candidate(), 500.0) == "below_floor"


def test_filter_eligible_returns_none() -> None:
    assert allocation.filter_reason(_candidate(), 5_000.0) is None


# ───────────────────────── ranking ──────────────────────────────────────────


def test_rank_candidates_orders_by_score_desc() -> None:
    strong = _candidate(ticker="STR5", expectancy=1.0, adx=25.0, rsi=50.0,
                        obv=100.0, vol_regime="low")
    weak = _candidate(ticker="WEAK", expectancy=0.0, vol_regime="high")
    ranked = allocation.rank_candidates([(weak, "SWING"), (strong, "SWING")])
    assert [r.candidate.ticker for r in ranked] == ["STR5", "WEAK"]
    assert [r.rank for r in ranked] == [1, 2]


# ───────────────────────── allocation ───────────────────────────────────────


def _ranked(
    ticker: str, pool: str, tier: str, rank: int, *,
    entry: float = 100.0, atr: float = 2.0, direction: str = "call",
    signal_type: str = "ema21_pullback",
) -> allocation.RankedCandidate:
    c = _candidate(ticker=ticker, entry=entry, atr=atr, direction=direction,
                   signal_type=signal_type)
    return allocation.RankedCandidate(
        candidate=c, pool=pool, score=0.5, tier=tier, rank=rank,
    )


def test_allocate_normal_tier_stops_at_30_percent_cap() -> None:
    # Each position caps at 20% of pool capital ($2000 of $10k). NORMAL ceiling
    # is 30% ($3000) -> the first fits, the second is skipped (tier cap).
    ranked = [
        _ranked("AAA", "SWING", allocation.TIER_NORMAL, 1),
        _ranked("BBB", "SWING", allocation.TIER_NORMAL, 2),
    ]
    orders, skipped, pools = allocation.allocate(ranked, {"SWING": (10_000.0, 10_000.0)})
    assert [o.ticker for o in orders] == ["AAA"]
    assert orders[0].est_cost == 2_000.0
    assert [(s.ticker, s.reason.split(":")[0]) for s in skipped] == [
        ("BBB", "capital-exhausted"),
    ]
    swing = next(p for p in pools if p.pool == "SWING")
    assert swing.orders == 1 and swing.deployed == 2_000.0


def test_allocate_high_tier_uses_50_percent_cap() -> None:
    # HIGH ceiling is 50% ($5000) -> two $2000 positions fit, the third is skipped.
    ranked = [
        _ranked("AAA", "SWING", allocation.TIER_HIGH, 1),
        _ranked("BBB", "SWING", allocation.TIER_HIGH, 2),
        _ranked("CCC", "SWING", allocation.TIER_HIGH, 3),
    ]
    orders, skipped, _ = allocation.allocate(ranked, {"SWING": (10_000.0, 10_000.0)})
    assert [o.ticker for o in orders] == ["AAA", "BBB"]
    assert [s.ticker for s in skipped] == ["CCC"]
    assert "HIGH deploy cap" in skipped[0].reason


def test_allocate_stops_at_cash_exhaustion() -> None:
    # Capital high (cap not binding for HIGH), but pool cash only $2500 -> the
    # first $2000 position fits, the second is skipped for CASH (not the cap).
    ranked = [
        _ranked("AAA", "SWING", allocation.TIER_HIGH, 1),
        _ranked("BBB", "SWING", allocation.TIER_HIGH, 2),
    ]
    orders, skipped, _ = allocation.allocate(ranked, {"SWING": (10_000.0, 2_500.0)})
    assert [o.ticker for o in orders] == ["AAA"]
    assert skipped[0].ticker == "BBB"
    assert "pool cash" in skipped[0].reason


def test_allocate_skips_not_undersizes() -> None:
    # The skipped candidate is NOT placed at a reduced size; every placed order
    # carries the full volatility-normalized cost.
    ranked = [
        _ranked("AAA", "SWING", allocation.TIER_NORMAL, 1),
        _ranked("BBB", "SWING", allocation.TIER_NORMAL, 2),
    ]
    orders, skipped, _ = allocation.allocate(ranked, {"SWING": (10_000.0, 10_000.0)})
    assert all(o.est_cost == 2_000.0 for o in orders)     # full size, never shrunk
    assert "BBB" not in [o.ticker for o in orders]
    assert "BBB" in [s.ticker for s in skipped]


def test_allocate_pools_are_independent() -> None:
    # SWING exhausting its cap does not affect CRYPTO's allocation.
    ranked = [
        _ranked("AAA", "SWING", allocation.TIER_NORMAL, 1),
        _ranked("BBB", "SWING", allocation.TIER_NORMAL, 2),
        _ranked("ETH", "CRYPTO", allocation.TIER_NORMAL, 3,
                signal_type="oversold_reversal"),
    ]
    orders, _, pools = allocation.allocate(
        ranked, {"SWING": (10_000.0, 10_000.0), "CRYPTO": (10_000.0, 10_000.0)},
    )
    assert {o.ticker for o in orders} == {"AAA", "ETH"}   # one per pool
    crypto = next(p for p in pools if p.pool == "CRYPTO")
    assert crypto.orders == 1


def test_allocate_unsizeable_is_skipped() -> None:
    ranked = [_ranked("AAA", "SWING", allocation.TIER_NORMAL, 1, atr=0.0)]
    orders, skipped, _ = allocation.allocate(ranked, {"SWING": (10_000.0, 10_000.0)})
    assert orders == []
    assert skipped[0].reason == "unsizeable"


# ───────────────────────── Stage 4: build_plan end-to-end ───────────────────


def test_build_plan_end_to_end() -> None:
    acct = AccountInfo(ok=True, equity=100_000.0, cash=100_000.0)
    strong = _candidate(
        ticker="STRONG", expectancy=1.0, adx=25.0, rsi=50.0, obv=100.0,
        vol_regime="low", atr=10.0,
    )
    normal = _candidate(
        ticker="NORMAL", expectancy=0.3, adx=25.0, rsi=50.0, obv=100.0,
        vol_regime="normal", atr=10.0,
    )
    muted = _candidate(ticker="MUTED", pair_enabled=False, atr=10.0)
    crypto = _candidate(
        ticker="ETHX", signal_type="long_term_crypto", asset_class="crypto",
        direction="long", expectancy=0.4, atr=10.0,
    )

    res = allocation.build_plan([normal, strong, muted, crypto], acct)
    assert res.ok is True
    # Highest composite score ranks first.
    assert res.plan.orders[0].ticker == "STRONG"
    # Crypto routed to the CRYPTO pool.
    eth = next(o for o in res.plan.orders if o.ticker == "ETHX")
    assert eth.pool == config.POOL_CRYPTO
    # The muted pair dropped at the FILTER stage.
    muted_skip = next(s for s in res.skipped if s.ticker == "MUTED")
    assert muted_skip.stage == "filter" and muted_skip.reason == "pair_muted"
    # All three pools are accounted for.
    assert {p.pool for p in res.pools} == set(config.POOLS)


def test_build_plan_drops_data_only_crypto_swing_signals() -> None:
    # Crypto SWING signals must NEVER execute — dropped before routing.
    acct = AccountInfo(ok=True, equity=100_000.0, cash=100_000.0)
    swing = _candidate(
        ticker="ETH-USD", signal_type="oversold_reversal", asset_class="crypto",
        direction="long", expectancy=0.9, atr=10.0,
    )
    res = allocation.build_plan([swing], acct)
    assert res.plan.orders == []                       # nothing planned
    drop = next(s for s in res.skipped if s.ticker == "ETH-USD")
    assert "data-only" in drop.reason


def test_build_plan_long_term_uses_diversification_sizing() -> None:
    # LONG_TERM pool capital = 30% of $100k = $30k; a 15% weight = $4500 (NOT
    # ATR/stop sizing — the candidate has no ATR).
    acct = AccountInfo(ok=True, equity=100_000.0, cash=100_000.0)
    lt = Candidate(
        ticker="AAPL", signal_type="long_term_stock", direction="long",
        asset_class="stock", entry=100.0, atr=None,
    )
    res = allocation.build_plan([lt], acct)
    order = next(o for o in res.plan.orders if o.ticker == "AAPL")
    assert order.pool == config.POOL_LONG_TERM
    assert order.est_cost == 4_500.0            # 15% of $30k, diversification-weighted
    assert order.qty == 45.0                    # $4500 / $100


def test_build_plan_account_unavailable_is_empty_plan() -> None:
    res = allocation.build_plan(
        [_candidate(ticker="AAA")], AccountInfo(ok=False, reason="creds unset"),
    )
    assert res.ok is False
    assert res.plan.orders == []
    assert res.skipped[0].reason == "account-unavailable"
    assert res.skipped[0].stage == "account"
    assert "account unavailable" in res.note


def test_sample_candidates_shape() -> None:
    cands = allocation.sample_candidates()
    assert len(cands) >= 4
    assert any(c.asset_class == "crypto" for c in cands)
    assert any(not c.ticker_active for c in cands)   # an ineligible one for skips


# ───────────────────────── allocate plan CLI ────────────────────────────────


def test_cli_allocate_plan_ok(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import __main__ as m
    monkeypatch.setattr(m.broker, "AlpacaBroker", lambda: FakeBroker())
    m.cmd_allocate_plan()
    out = capsys.readouterr().out
    assert "ALLOCATION PLAN" in out
    assert "PLAN only" in out
    assert "META" in out          # an eligible candidate appears in the plan
    assert "NVDA" in out          # the inactive candidate appears in skipped


def test_cli_allocate_plan_account_unavailable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import __main__ as m
    monkeypatch.setattr(m.broker, "AlpacaBroker", lambda: FakeBroker(fail=True))
    m.cmd_allocate_plan()
    assert "account unavailable" in capsys.readouterr().out


# ───────────────────────── NO EXECUTION (the core safety property) ──────────


def test_allocate_path_submits_zero_orders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The allocate path produces a plan and submits NOTHING. Assert the broker's
    submit_order is never called anywhere in the CLI allocate path."""
    from trading_bot import __main__ as m
    b = FakeBroker()
    calls: list[tuple[object, ...]] = []
    original = b.submit_order

    def spy(*args: object, **kwargs: object) -> object:
        calls.append(args)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(b, "submit_order", spy)
    monkeypatch.setattr(m.broker, "AlpacaBroker", lambda: b)
    m.cmd_allocate_plan()
    assert calls == []            # zero submissions
    assert b._orders == {}        # nothing recorded on the broker either
