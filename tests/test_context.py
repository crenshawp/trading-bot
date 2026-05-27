"""Tests for trading_bot.context — Phase 2.3 market context score.

Pure score matrix + label table + scanner integration + backfill. Regime
and VIX modules are mocked at the boundary so we never hit yfinance.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from trading_bot import context, db, regime, vix
from trading_bot.models import Prediction, Signal, Trade

# ────────────────────── score matrix ──────────────────────


@pytest.mark.parametrize(
    "regime_val,vix_val,expected",
    [
        ("bull",     "low",      5),
        ("bull",     "elevated", 4),
        ("bull",     "high",     3),
        ("bull",     "extreme",  2),
        ("sideways", "low",      4),
        ("sideways", "elevated", 3),
        ("sideways", "high",     2),
        ("sideways", "extreme",  1),
        ("bear",     "low",      2),
        ("bear",     "elevated", 2),
        ("bear",     "high",     1),
        ("bear",     "extreme",  1),
    ],
)
def test_score_matrix_all_valid_cells(
    regime_val: str, vix_val: str, expected: int,
) -> None:
    assert context.score(regime_val, vix_val) == expected


@pytest.mark.parametrize("vix_val", ["low", "elevated", "high", "extreme", "unknown"])
def test_unknown_regime_yields_zero(vix_val: str) -> None:
    assert context.score("unknown", vix_val) == 0


@pytest.mark.parametrize("regime_val", ["bull", "sideways", "bear", "unknown"])
def test_unknown_vix_yields_zero(regime_val: str) -> None:
    assert context.score(regime_val, "unknown") == 0


def test_both_unknown_yields_zero() -> None:
    assert context.score("unknown", "unknown") == 0


def test_invalid_regime_raises() -> None:
    with pytest.raises(ValueError, match="Invalid regime"):
        context.score("bullish", "low")


def test_invalid_vix_band_raises() -> None:
    with pytest.raises(ValueError, match="Invalid vix_band"):
        context.score("bull", "very_low")


# ────────────────────── labels ──────────────────────


@pytest.mark.parametrize(
    "score_val,expected",
    [
        (5, "ideal"),
        (4, "favorable"),
        (3, "neutral"),
        (2, "unfavorable"),
        (1, "hostile"),
        (0, "unknown"),
    ],
)
def test_label_for_each_score(score_val: int, expected: str) -> None:
    assert context.label(score_val) == expected


@pytest.mark.parametrize("bad", [-1, 6, 99])
def test_label_out_of_range_raises(bad: int) -> None:
    with pytest.raises(ValueError, match="out of range"):
        context.label(bad)


# ────────────────────── get_current_context ──────────────────────


def test_get_current_context_composes_axes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_regime = regime.RegimeSnapshot(
        date="2026-05-26", regime="bull", spy_close=600.0,
        ema50=590.0, ema200=550.0, ema50_slope=1.0,
    )
    fake_vix = vix.VixSnapshot(
        date="2026-05-26", vix_level=18.4, vix_band="low",
        captured_at=datetime.now(UTC).isoformat(),
    )
    monkeypatch.setattr(
        "trading_bot.context.regime.get_current_regime", lambda **_: fake_regime,
    )
    monkeypatch.setattr(
        "trading_bot.context.vix.get_current_vix", lambda **_: fake_vix,
    )
    snap = context.get_current_context()
    assert snap.regime == "bull"
    assert snap.vix_band == "low"
    assert snap.vix_level == pytest.approx(18.4)
    assert snap.score == 5
    assert snap.label == "ideal"


def test_get_current_context_unknown_regime_on_fetch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(**_: object) -> regime.RegimeSnapshot:
        raise regime.RegimeFetchError("net")

    fake_vix = vix.VixSnapshot(
        date="2026-05-26", vix_level=18.0, vix_band="low",
        captured_at=datetime.now(UTC).isoformat(),
    )
    monkeypatch.setattr(
        "trading_bot.context.regime.get_current_regime", boom,
    )
    monkeypatch.setattr(
        "trading_bot.context.vix.get_current_vix", lambda **_: fake_vix,
    )
    snap = context.get_current_context()
    assert snap.regime == "unknown"
    assert snap.score == 0
    assert snap.label == "unknown"


def test_get_current_context_unknown_vix_on_fetch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_regime = regime.RegimeSnapshot(
        date="2026-05-26", regime="bull", spy_close=600.0,
        ema50=590.0, ema200=550.0, ema50_slope=1.0,
    )

    def boom(**_: object) -> vix.VixSnapshot:
        raise vix.VixFetchError("net")

    monkeypatch.setattr(
        "trading_bot.context.regime.get_current_regime", lambda **_: fake_regime,
    )
    monkeypatch.setattr("trading_bot.context.vix.get_current_vix", boom)
    snap = context.get_current_context()
    assert snap.vix_band == "unknown"
    assert snap.vix_level is None
    assert snap.score == 0


def test_get_current_context_does_not_force_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_current_context() must NOT pass force_refresh — that's what the
    per-axis CLIs are for."""
    calls: dict[str, list[bool]] = {"regime": [], "vix": []}

    def regime_spy(force_refresh: bool = False) -> regime.RegimeSnapshot:
        calls["regime"].append(force_refresh)
        return regime.RegimeSnapshot(
            date="2026-05-26", regime="bull", spy_close=600.0,
            ema50=590.0, ema200=550.0, ema50_slope=1.0,
        )

    def vix_spy(force_refresh: bool = False) -> vix.VixSnapshot:
        calls["vix"].append(force_refresh)
        return vix.VixSnapshot(
            date="2026-05-26", vix_level=18.0, vix_band="low",
            captured_at=datetime.now(UTC).isoformat(),
        )

    monkeypatch.setattr(
        "trading_bot.context.regime.get_current_regime", regime_spy,
    )
    monkeypatch.setattr(
        "trading_bot.context.vix.get_current_vix", vix_spy,
    )
    context.get_current_context()
    assert calls["regime"] == [False]
    assert calls["vix"] == [False]


# ────────────────────── scanner integration ──────────────────────


def _stub_scanner_axes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    regime_tag: str = "bull",
    vix_band: str = "low",
    vix_level: float | None = 18.0,
    regime_raises: bool = False,
    vix_raises: bool = False,
) -> None:
    from trading_bot import scanner

    if regime_raises:
        def regime_fn(**_: object) -> regime.RegimeSnapshot:
            raise regime.RegimeFetchError("net")
    else:
        def regime_fn(**_: object) -> regime.RegimeSnapshot:
            return regime.RegimeSnapshot(
                date="2026-05-26", regime=regime_tag, spy_close=600.0,
                ema50=590.0, ema200=550.0, ema50_slope=1.0,
            )

    if vix_raises:
        def vix_fn(**_: object) -> vix.VixSnapshot:
            raise vix.VixFetchError("net")
    else:
        def vix_fn(**_: object) -> vix.VixSnapshot:
            return vix.VixSnapshot(
                date="2026-05-26",
                vix_level=vix_level if vix_level is not None else 0.0,
                vix_band=vix_band,
                captured_at=datetime.now(UTC).isoformat(),
            )

    monkeypatch.setattr(scanner.regime, "get_current_regime", regime_fn)
    monkeypatch.setattr(scanner.vix, "get_current_vix", vix_fn)


def test_scanner_tags_context_score_on_new_trade(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading_bot import scanner

    _stub_scanner_axes(
        monkeypatch, regime_tag="bull", vix_band="low", vix_level=18.0,
    )
    sid = scanner.log_signal({
        "ticker": "GOOGL", "asset_type": "stock", "direction": "CALL",
        "setup": "EMA21 Pullback", "price": 180.0,
        "take_profit": 185.0, "stop_loss": 178.0,
    })
    assert sid is not None
    trade = db.get_trade_by_signal_id(sid)
    assert trade is not None
    assert trade.context_score == 5  # bull + low = ideal


def test_scanner_unknown_regime_yields_score_zero(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading_bot import scanner

    _stub_scanner_axes(monkeypatch, regime_raises=True, vix_band="low")
    sid = scanner.log_signal({
        "ticker": "GOOGL", "asset_type": "stock", "direction": "CALL",
        "setup": "EMA21 Pullback", "price": 180.0,
        "take_profit": 185.0, "stop_loss": 178.0,
    })
    assert sid is not None
    trade = db.get_trade_by_signal_id(sid)
    assert trade is not None
    assert trade.context_score == 0


def test_scanner_unknown_vix_yields_score_zero(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading_bot import scanner

    _stub_scanner_axes(monkeypatch, regime_tag="bull", vix_raises=True)
    sid = scanner.log_signal({
        "ticker": "GOOGL", "asset_type": "stock", "direction": "CALL",
        "setup": "EMA21 Pullback", "price": 180.0,
        "take_profit": 185.0, "stop_loss": 178.0,
    })
    assert sid is not None
    trade = db.get_trade_by_signal_id(sid)
    assert trade is not None
    assert trade.context_score == 0


def test_scanner_tags_context_score_on_new_prediction(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prediction sweep should tag context_score on the inserted row."""
    import json

    from trading_bot import scanner, settings
    from trading_bot.models import Prediction as P

    settings.set_bool("predictions.enabled", True)
    settings.set("predictions.window_start", "00:00")
    settings.set("predictions.window_end", "23:59")
    settings.set("predictions.tickers", "BTC-USD")
    monkeypatch.setattr(scanner, "DRY_RUN", True)

    def fake_predict(ticker: str, **_k: Any) -> P:
        return P(
            ticker=ticker, direction="HIGHER", confidence=72.0,
            entry_price=94000.0,
            target_window_end=datetime.now(UTC) + timedelta(minutes=15),
            signals_used=json.dumps({"x": 1}),
            created_at=datetime.now(UTC),
            market_regime="bull", vix_band="low", vix_level=18.0,
        )

    monkeypatch.setattr(scanner.predictions, "predict_direction", fake_predict)
    scanner._run_prediction_sweep()

    rows = db.get_predictions(limit=5)
    assert len(rows) == 1
    assert rows[0].context_score == 5


# ────────────────────── backfill ──────────────────────


def _seed_trade(
    *,
    regime_tag: str = "bull",
    vix_band: str = "low",
    context_score: int | None = None,
    outcome: str = "win",
) -> int:
    ts = datetime(2026, 4, 1, tzinfo=UTC) + timedelta(seconds=_seed_trade.counter)
    _seed_trade.counter += 1  # type: ignore[attr-defined]
    sid = db.insert_signal(Signal(
        timestamp=ts, ticker="GOOGL", asset_class="stock",
        signal_type="ema21_pullback", direction="call",
        entry_price=100.0, take_profit=110.0, stop_loss=95.0,
    ))
    return db.insert_trade(Trade(
        signal_id=sid, opened_at=ts,
        closed_at=ts + timedelta(days=2),
        outcome=outcome, exit_price=105.0, pnl_pct=5.0,
        market_regime=regime_tag, vix_band=vix_band, vix_level=18.0,
        context_score=context_score,
    ))


_seed_trade.counter = 0  # type: ignore[attr-defined]


def _seed_prediction(
    *,
    regime_tag: str = "bull",
    vix_band: str = "low",
    context_score: int | None = None,
) -> int:
    ts = datetime(2026, 5, 26, 14, 0, tzinfo=UTC) + timedelta(
        seconds=_seed_prediction.counter,
    )
    _seed_prediction.counter += 1  # type: ignore[attr-defined]
    return db.insert_prediction(Prediction(
        ticker="BTC-USD", direction="HIGHER", confidence=72.0,
        entry_price=94000.0,
        target_window_end=ts + timedelta(minutes=15),
        signals_used="{}", created_at=ts,
        market_regime=regime_tag, vix_band=vix_band, vix_level=18.0,
        context_score=context_score,
    ))


_seed_prediction.counter = 0  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def _reset_seed_counters() -> None:
    _seed_trade.counter = 0  # type: ignore[attr-defined]
    _seed_prediction.counter = 0  # type: ignore[attr-defined]


def test_backfill_populates_trade_context_scores(tmp_db: Path) -> None:
    _seed_trade(regime_tag="bull", vix_band="low")
    _seed_trade(regime_tag="sideways", vix_band="elevated")
    _seed_trade(regime_tag="bear", vix_band="extreme")

    summary = context.backfill_context_scores()
    assert summary["trades_updated"] == 3
    assert summary["predictions_updated"] == 0

    # Verify the actual values
    sql = "SELECT context_score FROM trades ORDER BY id"
    conn = db.get_connection()
    try:
        scores = [r["context_score"] for r in conn.execute(sql).fetchall()]
    finally:
        conn.close()
    assert scores == [5, 3, 1]


def test_backfill_populates_prediction_context_scores(tmp_db: Path) -> None:
    _seed_prediction(regime_tag="bull", vix_band="low")
    _seed_prediction(regime_tag="bear", vix_band="high")

    summary = context.backfill_context_scores()
    assert summary["predictions_updated"] == 2

    sql = "SELECT context_score FROM predictions ORDER BY id"
    conn = db.get_connection()
    try:
        scores = [r["context_score"] for r in conn.execute(sql).fetchall()]
    finally:
        conn.close()
    assert scores == [5, 1]


def test_backfill_is_idempotent(tmp_db: Path) -> None:
    _seed_trade(regime_tag="bull", vix_band="low")
    first = context.backfill_context_scores()
    second = context.backfill_context_scores()
    assert first["trades_updated"] == 1
    assert second["trades_updated"] == 0


def test_backfill_skips_trades_already_scored(tmp_db: Path) -> None:
    _seed_trade(regime_tag="bull", vix_band="low", context_score=5)
    summary = context.backfill_context_scores()
    assert summary["trades_updated"] == 0


def test_backfill_unknown_regime_writes_zero(tmp_db: Path) -> None:
    _seed_trade(regime_tag="unknown", vix_band="low")
    context.backfill_context_scores()
    sql = "SELECT context_score FROM trades"
    conn = db.get_connection()
    try:
        row = conn.execute(sql).fetchone()
    finally:
        conn.close()
    assert row["context_score"] == 0
