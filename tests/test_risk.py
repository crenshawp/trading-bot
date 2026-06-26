"""Tests for trading_bot.risk — advisory sizing + portfolio verdicts (Phase 7).

Every value is hand-computed; no network. Risk is advisory only, so the point
is that the recommendation matches the documented math and degrades gracefully.
"""

import pytest

from trading_bot import risk

# ───────────────────────── position sizing: known values ────────────────────────


def test_position_size_uncapped_hand_computed() -> None:
    # stop = 10*1.5 = 15; dollar_risk = 10000*1% = 100; size = 100/15 = 6.6667;
    # position_value = 666.67 < 2000 cap -> not capped.
    rec = risk.position_size(
        entry=100.0, atr=10.0, account=10_000.0, risk_per_trade_pct=1.0,
        stop_multiple=1.5, max_position_pct=20.0,
    )
    assert rec.ok is True
    assert rec.capped is False
    assert rec.stop_distance == pytest.approx(15.0)
    assert rec.dollar_risk == pytest.approx(100.0)
    assert rec.recommended_size == pytest.approx(6.66667, abs=1e-4)
    assert rec.risk_pct == pytest.approx(1.0)
    assert rec.position_pct == pytest.approx(6.66667, abs=1e-4)


def test_position_size_capped_at_max_position_pct() -> None:
    # stop = 2*1.5 = 3; raw size = 100/3 = 33.33 -> value 3333 > 2000 cap.
    # capped size = 2000/100 = 20; dollar_risk = 20*3 = 60; risk_pct = 0.6.
    rec = risk.position_size(
        entry=100.0, atr=2.0, account=10_000.0, risk_per_trade_pct=1.0,
        stop_multiple=1.5, max_position_pct=20.0,
    )
    assert rec.ok is True
    assert rec.capped is True
    assert rec.recommended_size == pytest.approx(20.0)
    assert rec.position_value == pytest.approx(2000.0)
    assert rec.position_pct == pytest.approx(20.0)
    assert rec.dollar_risk == pytest.approx(60.0)
    assert rec.risk_pct == pytest.approx(0.6)   # capped -> LESS than requested 1%


def test_higher_atr_gets_proportionally_smaller_size() -> None:
    # Same dollar risk, no cap (max 100%): size is inversely proportional to ATR.
    low_vol = risk.position_size(
        entry=100.0, atr=2.0, account=10_000.0, risk_per_trade_pct=1.0,
        stop_multiple=1.5, max_position_pct=100.0,
    )
    high_vol = risk.position_size(
        entry=100.0, atr=4.0, account=10_000.0, risk_per_trade_pct=1.0,
        stop_multiple=1.5, max_position_pct=100.0,
    )
    assert low_vol.recommended_size is not None
    assert high_vol.recommended_size is not None
    # ATR doubled -> size halved.
    assert low_vol.recommended_size == pytest.approx(2.0 * high_vol.recommended_size)


@pytest.mark.parametrize("atr", [None, 0.0, -1.0])
def test_position_size_fail_soft_on_bad_atr(
    atr: float | None, capsys: pytest.CaptureFixture[str],
) -> None:
    rec = risk.position_size(entry=100.0, atr=atr, account=10_000.0,
                             risk_per_trade_pct=1.0)
    assert rec.ok is False
    assert rec.recommended_size is None
    assert "ATR" in rec.reason
    assert "size unavailable" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("entry", "account", "needle"),
    [(0.0, 10_000.0, "entry"), (-5.0, 10_000.0, "entry"), (100.0, 0.0, "account")],
)
def test_position_size_fail_soft_on_bad_entry_or_account(
    entry: float, account: float, needle: str,
) -> None:
    rec = risk.position_size(entry=entry, atr=2.0, account=account,
                             risk_per_trade_pct=1.0)
    assert rec.ok is False
    assert needle in rec.reason


def test_position_size_outer_guard_swallows_garbage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A non-numeric entry slips past the None/<=0 guards and raises inside the
    # arithmetic; the outer guard must still return unavailable, never propagate.
    rec = risk.position_size(entry="oops", atr=2.0, account=10_000.0,  # type: ignore[arg-type]
                             risk_per_trade_pct=1.0)
    assert rec.ok is False
    assert "size unavailable" in rec.reason
    assert "size unavailable" in capsys.readouterr().err
