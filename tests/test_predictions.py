"""Tests for trading_bot.predictions — 15-min direction engine.

Indicators are exercised against synthetic candles. yfinance is mocked
throughout. Regime + VIX modules are mocked at the boundary so we don't
hit their disk caches in these tests.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from trading_bot import db, predictions, regime, vix
from trading_bot.models import Prediction

# ────────────────────── fixtures ──────────────────────


@pytest.fixture(autouse=True)
def stub_context(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Stub regime + VIX so predictions don't try real yfinance calls."""
    monkeypatch.setattr(
        "trading_bot.predictions.regime.get_current_regime",
        lambda **_: regime.RegimeSnapshot(
            date="2026-05-26", regime="bull", spy_close=600.0,
            ema50=590.0, ema200=550.0, ema50_slope=1.0,
        ),
    )
    monkeypatch.setattr(
        "trading_bot.predictions.vix.get_current_vix",
        lambda **_: vix.VixSnapshot(
            date="2026-05-26", vix_level=18.4, vix_band="low",
            captured_at=datetime.now(UTC).isoformat(),
        ),
    )
    yield


def _build_candles(
    *, closes: list[float], opens: list[float] | None = None,
    volumes: list[int] | None = None, start: datetime | None = None,
) -> pd.DataFrame:
    n = len(closes)
    opens = opens if opens is not None else closes.copy()
    volumes = volumes if volumes is not None else [1_000_000] * n
    anchor = start if start is not None else datetime(
        2026, 5, 26, 14, 0, tzinfo=UTC,
    )
    index = [anchor + timedelta(minutes=15 * i) for i in range(n)]
    return pd.DataFrame(
        {
            "Open":   opens,
            "High":   [max(o, c) + 1 for o, c in zip(opens, closes, strict=True)],
            "Low":    [min(o, c) - 1 for o, c in zip(opens, closes, strict=True)],
            "Close":  closes,
            "Volume": volumes,
        },
        index=pd.DatetimeIndex(index),
    )


def _strong_uptrend(n: int = 35) -> pd.DataFrame:
    """All indicators should vote HIGHER."""
    closes = [100.0 + i * 0.5 for i in range(n)]
    opens = [closes[i] - 0.3 for i in range(n)]  # green candles
    # Last 3 candles get a clean uptrend close; volume spikes on last
    volumes = [1_000_000] * (n - 1) + [3_000_000]
    return _build_candles(closes=closes, opens=opens, volumes=volumes)


def _strong_downtrend(n: int = 35) -> pd.DataFrame:
    closes = [200.0 - i * 0.5 for i in range(n)]
    opens = [closes[i] + 0.3 for i in range(n)]  # red candles
    volumes = [1_000_000] * (n - 1) + [3_000_000]
    return _build_candles(closes=closes, opens=opens, volumes=volumes)


def _flat_candles(n: int = 35) -> pd.DataFrame:
    """Perfectly flat market — every indicator votes NEUTRAL."""
    closes = [150.0] * n
    opens = [150.0] * n
    return _build_candles(closes=closes, opens=opens, volumes=[1_000_000] * n)


def _patch_fetch(
    monkeypatch: pytest.MonkeyPatch, df: pd.DataFrame | None,
) -> None:
    monkeypatch.setattr(
        "trading_bot.predictions._fetch_15m_candles", lambda _t: df,
    )


# ────────────────────── tally_votes (pure) ──────────────────────


def test_tally_all_higher() -> None:
    votes = dict.fromkeys(["a", "b", "c", "d", "e", "f"], 1)
    direction, confidence = predictions.tally_votes(votes)
    assert direction == "HIGHER"
    assert confidence == 100.0


def test_tally_all_lower() -> None:
    votes = dict.fromkeys(["a", "b", "c", "d", "e", "f"], -1)
    direction, confidence = predictions.tally_votes(votes)
    assert direction == "LOWER"
    assert confidence == 100.0


def test_tally_balanced_returns_none() -> None:
    # 3 HIGHER, 3 LOWER → confidence 50%, below tolerance
    votes = {"a": 1, "b": 1, "c": 1, "d": -1, "e": -1, "f": -1}
    direction, _ = predictions.tally_votes(votes)
    assert direction is None


def test_tally_within_tolerance_returns_none() -> None:
    # 4 HIGHER, 3 LOWER → 4/7 = 57.1% which is > 55%, BUT
    # 3 HIGHER, 2 LOWER → 3/5 = 60% > 55% (decisive)
    # Need a sub-55% case: 1 vote difference at low totals is decisive,
    # but a 5-4 split: 5/9 = 55.5% (just over). Let's use a 5-5 split = 50%.
    votes = {"a": 1, "b": 1, "c": 1, "d": 1, "e": 1,
             "f": -1, "g": -1, "h": -1, "i": -1, "j": -1}
    direction, _ = predictions.tally_votes(votes)
    assert direction is None


def test_tally_all_neutral_returns_none() -> None:
    votes = dict.fromkeys(["a", "b", "c", "d", "e", "f"], 0)
    direction, confidence = predictions.tally_votes(votes)
    assert direction is None
    assert confidence == 0.0


def test_tally_decisive_higher() -> None:
    # 4 HIGHER, 1 LOWER, 1 NEUTRAL → 4/5 = 80%
    votes = {"a": 1, "b": 1, "c": 1, "d": 1, "e": -1, "f": 0}
    direction, confidence = predictions.tally_votes(votes)
    assert direction == "HIGHER"
    assert confidence == pytest.approx(80.0)


# ────────────────────── predict_direction (with full stack) ──────────────────────


def test_predict_strong_uptrend_returns_higher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fetch(monkeypatch, _strong_uptrend())
    pred = predictions.predict_direction("BTC-USD")
    assert pred is not None
    assert pred.direction == "HIGHER"
    assert pred.confidence >= 55.0


def test_predict_strong_downtrend_returns_lower(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fetch(monkeypatch, _strong_downtrend())
    pred = predictions.predict_direction("BTC-USD")
    assert pred is not None
    assert pred.direction == "LOWER"


def test_predict_flat_candles_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All indicators NEUTRAL → no prediction."""
    _patch_fetch(monkeypatch, _flat_candles())
    assert predictions.predict_direction("BTC-USD") is None


def test_predict_no_data_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fetch(monkeypatch, None)
    assert predictions.predict_direction("BTC-USD") is None


def test_predict_short_history_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Bypass fetch — feed too few rows into the underlying engine via fetch stub.
    # _fetch_15m_candles returns None when df.shape[0] < 30. We patch it to
    # return None directly to simulate that.
    _patch_fetch(monkeypatch, None)
    assert predictions.predict_direction("BTC-USD") is None


def test_predict_tags_regime_and_vix(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fetch(monkeypatch, _strong_uptrend())
    pred = predictions.predict_direction("BTC-USD")
    assert pred is not None
    assert pred.market_regime == "bull"
    assert pred.vix_band == "low"
    assert pred.vix_level == pytest.approx(18.4)


def test_predict_unknown_regime_on_fetch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fetch(monkeypatch, _strong_uptrend())

    def boom(**_: object) -> regime.RegimeSnapshot:
        raise regime.RegimeFetchError("net down")

    monkeypatch.setattr(
        "trading_bot.predictions.regime.get_current_regime", boom,
    )
    pred = predictions.predict_direction("BTC-USD")
    assert pred is not None
    assert pred.market_regime == "unknown"


def test_predict_unknown_vix_on_fetch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fetch(monkeypatch, _strong_uptrend())

    def boom(**_: object) -> vix.VixSnapshot:
        raise vix.VixFetchError("net down")

    monkeypatch.setattr(
        "trading_bot.predictions.vix.get_current_vix", boom,
    )
    pred = predictions.predict_direction("BTC-USD")
    assert pred is not None
    assert pred.vix_band == "unknown"
    assert pred.vix_level is None


def test_predict_signals_used_is_valid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fetch(monkeypatch, _strong_uptrend())
    pred = predictions.predict_direction("BTC-USD")
    assert pred is not None
    parsed = json.loads(pred.signals_used)
    assert set(parsed) == {
        "candle_streak", "rsi_slope", "macd_hist", "vwap",
        "volume_conf", "bb_position",
    }


def test_predict_target_window_is_15_minutes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fetch(monkeypatch, _strong_uptrend())
    now = datetime(2026, 5, 26, 14, 30, tzinfo=UTC)
    pred = predictions.predict_direction("BTC-USD", now=now)
    assert pred is not None
    assert pred.target_window_end == now + timedelta(minutes=15)


# ────────────────────── resolution ──────────────────────


def _make_prediction(
    *,
    direction: str = "HIGHER",
    entry: float = 100.0,
    target_offset_min: int = -15,  # default: target ended 15 min ago
    ticker: str = "BTC-USD",
    now: datetime | None = None,
) -> int:
    """Insert a stub Prediction. Returns the new id."""
    now_dt = now if now is not None else datetime(2026, 5, 26, 14, 0, tzinfo=UTC)
    target = now_dt + timedelta(minutes=target_offset_min)
    return db.insert_prediction(Prediction(
        ticker=ticker,
        direction=direction,
        confidence=70.0,
        entry_price=entry,
        target_window_end=target,
        signals_used=json.dumps({"x": 1}),
        created_at=target - timedelta(minutes=15),
        market_regime="bull",
        vix_band="low",
        vix_level=18.0,
    ))


def _resolution_df(
    target_window_end: datetime, close_at_target: float, ticker_close_now: float,
) -> pd.DataFrame:
    """Build a 15-min DataFrame containing the target candle.

    The candle indexed at (target - 15min) closes AT target_window_end and
    has close = close_at_target.
    """
    n = 30
    closes = [ticker_close_now] * n
    # Make one candle at the target_open position carry close_at_target.
    target_open = target_window_end - timedelta(minutes=15)
    closes[-1] = close_at_target  # most recent candle = the target candle
    index = [target_open - timedelta(minutes=15 * i) for i in range(n)]
    index.reverse()
    return pd.DataFrame(
        {
            "Open":   closes,
            "High":   [c + 1 for c in closes],
            "Low":    [c - 1 for c in closes],
            "Close":  closes,
            "Volume": [1_000_000] * n,
        },
        index=pd.DatetimeIndex(index, tz="UTC"),
    )


def test_resolve_higher_correct(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 26, 14, 0, tzinfo=UTC)
    target = now - timedelta(minutes=15)
    pid = _make_prediction(direction="HIGHER", entry=100.0, now=now)
    _patch_fetch(monkeypatch, _resolution_df(target, close_at_target=105.0,
                                             ticker_close_now=104.0))

    result = predictions.resolve_prediction(pid, now=now)
    assert result.status == "resolved"
    assert result.outcome == "correct"
    assert result.exit_price == pytest.approx(105.0)


def test_resolve_higher_incorrect(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 26, 14, 0, tzinfo=UTC)
    target = now - timedelta(minutes=15)
    pid = _make_prediction(direction="HIGHER", entry=100.0, now=now)
    _patch_fetch(monkeypatch, _resolution_df(target, close_at_target=95.0,
                                             ticker_close_now=94.0))
    result = predictions.resolve_prediction(pid, now=now)
    assert result.outcome == "incorrect"


def test_resolve_lower_correct(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 26, 14, 0, tzinfo=UTC)
    target = now - timedelta(minutes=15)
    pid = _make_prediction(direction="LOWER", entry=100.0, now=now)
    _patch_fetch(monkeypatch, _resolution_df(target, close_at_target=95.0,
                                             ticker_close_now=94.0))
    assert predictions.resolve_prediction(pid, now=now).outcome == "correct"


def test_resolve_lower_incorrect(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 26, 14, 0, tzinfo=UTC)
    target = now - timedelta(minutes=15)
    pid = _make_prediction(direction="LOWER", entry=100.0, now=now)
    _patch_fetch(monkeypatch, _resolution_df(target, close_at_target=105.0,
                                             ticker_close_now=104.0))
    assert predictions.resolve_prediction(pid, now=now).outcome == "incorrect"


def test_resolve_push(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 26, 14, 0, tzinfo=UTC)
    target = now - timedelta(minutes=15)
    pid = _make_prediction(direction="HIGHER", entry=100.0, now=now)
    _patch_fetch(monkeypatch, _resolution_df(target, close_at_target=100.0,
                                             ticker_close_now=100.0))
    assert predictions.resolve_prediction(pid, now=now).outcome == "push"


def test_resolve_unresolved_when_candle_missing(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 26, 14, 0, tzinfo=UTC)
    target_far_future = now + timedelta(hours=2)  # target_open not in df
    pid = _make_prediction(direction="HIGHER", entry=100.0,
                           target_offset_min=120, now=now)
    _patch_fetch(monkeypatch, _resolution_df(
        target_far_future + timedelta(minutes=1), 105.0, 104.0,
    ))
    result = predictions.resolve_prediction(pid, now=now)
    assert result.status == "unresolved"


def test_resolve_returns_resolved_for_already_resolved(
    tmp_db: Path,
) -> None:
    pid = _make_prediction()
    db.update_prediction(
        pid, resolved_at=datetime(2026, 5, 26, 14, 0, tzinfo=UTC),
        exit_price=105.0, outcome="correct",
    )
    result = predictions.resolve_prediction(pid)
    assert result.status == "resolved"
    assert result.outcome == "correct"


def test_resolve_due_only_touches_past_window(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 5, 26, 14, 0, tzinfo=UTC)
    # Past: should resolve. Future: should be left alone.
    pid_past = _make_prediction(target_offset_min=-15, now=now)
    pid_future = _make_prediction(target_offset_min=60, now=now)

    _patch_fetch(monkeypatch, _resolution_df(
        now - timedelta(minutes=15), 105.0, 104.0,
    ))
    results = predictions.resolve_due_predictions(now=now)
    ids = {r.prediction_id for r in results}
    assert pid_past in ids
    assert pid_future not in ids


def test_resolve_prediction_not_found(tmp_db: Path) -> None:
    result = predictions.resolve_prediction(99999)
    assert result.status == "error"


def test_resolve_unresolved_when_no_data(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid = _make_prediction()
    _patch_fetch(monkeypatch, None)
    result = predictions.resolve_prediction(pid)
    assert result.status == "unresolved"


# ────────────────────── fetch failure path ──────────────────────


def test_fetch_15m_returns_none_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_a: Any, **_k: Any) -> pd.DataFrame:
        raise RuntimeError("rate limited")

    monkeypatch.setattr("trading_bot.predictions.yf.download", boom)
    assert predictions._fetch_15m_candles("BTC-USD") is None  # noqa: SLF001


def test_fetch_15m_returns_none_when_too_few_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    small = _build_candles(closes=[100.0] * 5)
    monkeypatch.setattr(
        "trading_bot.predictions.yf.download",
        lambda *_a, **_k: small,
    )
    assert predictions._fetch_15m_candles("BTC-USD") is None  # noqa: SLF001


def test_fetch_15m_returns_none_on_empty_df(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "trading_bot.predictions.yf.download",
        lambda *_a, **_k: pd.DataFrame(),
    )
    assert predictions._fetch_15m_candles("BTC-USD") is None  # noqa: SLF001


# ────────────────────── scanner gating ──────────────────────


def test_scanner_skips_when_disabled(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import scanner, settings

    settings.set_bool("predictions.enabled", False)
    settings.set("predictions.window_start", "00:00")
    settings.set("predictions.window_end", "23:59")

    calls: list[str] = []

    def spy(_t: str, **_k: Any) -> Prediction | None:
        calls.append(_t)
        return None

    monkeypatch.setattr(scanner.predictions, "predict_direction", spy)
    scanner._run_prediction_sweep()
    out = capsys.readouterr().out
    assert calls == []
    assert "skipped: disabled" in out


def test_scanner_skips_outside_window(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import scanner, settings

    settings.set_bool("predictions.enabled", True)
    # Set an impossible-to-hit window (1 minute) at midnight ET.
    settings.set("predictions.window_start", "00:00")
    settings.set("predictions.window_end", "00:01")

    calls: list[str] = []
    monkeypatch.setattr(
        scanner.predictions, "predict_direction",
        lambda t, **_k: calls.append(t) or None,
    )
    scanner._run_prediction_sweep()
    out = capsys.readouterr().out
    assert "outside_window" in out
    assert calls == []


def test_scanner_skips_when_paused(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import scanner, settings

    settings.set_bool("predictions.enabled", True)
    settings.set("predictions.window_start", "00:00")
    settings.set("predictions.window_end", "23:59")
    until = (datetime.now(UTC) + timedelta(minutes=60)).isoformat()
    settings.set("predictions.pause_until", until)

    calls: list[str] = []
    monkeypatch.setattr(
        scanner.predictions, "predict_direction",
        lambda t, **_k: calls.append(t) or None,
    )
    scanner._run_prediction_sweep()
    assert "paused" in capsys.readouterr().out
    assert calls == []


def test_scanner_clears_expired_pause(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading_bot import scanner, settings

    settings.set_bool("predictions.enabled", True)
    settings.set("predictions.window_start", "00:00")
    settings.set("predictions.window_end", "23:59")
    settings.set("predictions.tickers", "BTC-USD")
    past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    settings.set("predictions.pause_until", past)

    monkeypatch.setattr(
        scanner.predictions, "predict_direction", lambda _t, **_k: None,
    )
    scanner._run_prediction_sweep()
    # pause_until should be cleared after detecting it's expired
    assert settings.get("predictions.pause_until") is None


def test_scanner_runs_when_enabled_inside_window(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading_bot import scanner, settings

    settings.set_bool("predictions.enabled", True)
    settings.set("predictions.window_start", "00:00")
    settings.set("predictions.window_end", "23:59")
    settings.set("predictions.tickers", "BTC-USD,ETH-USD")
    monkeypatch.setattr(scanner, "DRY_RUN", True)  # suppress notifications

    fake = Prediction(
        ticker="BTC-USD", direction="HIGHER", confidence=72.0,
        entry_price=100.0,
        target_window_end=datetime.now(UTC) + timedelta(minutes=15),
        signals_used=json.dumps({"x": 1}),
        created_at=datetime.now(UTC),
        market_regime="bull", vix_band="low", vix_level=18.0,
    )
    calls: list[str] = []

    def fake_predict(ticker: str, **_k: Any) -> Prediction | None:
        calls.append(ticker)
        # Return a fresh prediction for each ticker (the ticker field is
        # used for db distinction; reuse the dataclass for brevity).
        return Prediction(
            ticker=ticker, direction=fake.direction, confidence=fake.confidence,
            entry_price=fake.entry_price,
            target_window_end=fake.target_window_end,
            signals_used=fake.signals_used,
            created_at=fake.created_at,
            market_regime=fake.market_regime, vix_band=fake.vix_band,
            vix_level=fake.vix_level,
        )

    monkeypatch.setattr(scanner.predictions, "predict_direction", fake_predict)
    scanner._run_prediction_sweep()
    assert calls == ["BTC-USD", "ETH-USD"]
    # Both predictions persisted
    rows = db.get_predictions(limit=10)
    assert len(rows) == 2
    assert {r.ticker for r in rows} == {"BTC-USD", "ETH-USD"}


# ────────────────────── notification format ──────────────────────


def test_prediction_notification_is_plain_ascii(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import scanner

    monkeypatch.setattr(scanner, "DRY_RUN", True)
    pred = Prediction(
        ticker="BTC-USD", direction="HIGHER", confidence=72.0,
        entry_price=94250.0,
        target_window_end=datetime(2026, 5, 26, 14, 45, tzinfo=UTC),
        signals_used=json.dumps({"x": 1}),
        created_at=datetime(2026, 5, 26, 14, 30, tzinfo=UTC),
        market_regime="bull", vix_band="low", vix_level=18.0,
    )
    scanner._send_prediction_notification(pred)
    out = capsys.readouterr().out
    out.encode("cp1252")  # raises if any non-ASCII slipped in
    assert "[PREDICTION] BTC-USD -> HIGHER" in out


def test_resolution_notification_format(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import scanner

    monkeypatch.setattr(scanner, "DRY_RUN", True)
    pred = Prediction(
        ticker="BTC-USD", direction="HIGHER", confidence=71.0,
        entry_price=94250.0,
        target_window_end=datetime(2026, 5, 26, 14, 45, tzinfo=UTC),
        signals_used=json.dumps({"x": 1}),
        created_at=datetime(2026, 5, 26, 14, 30, tzinfo=UTC),
        market_regime="bull", vix_band="low", vix_level=18.0,
    )
    result = predictions.ResolutionResult(
        prediction_id=1, status="resolved", outcome="correct",
        entry_price=94250.0, exit_price=94318.0,
    )
    scanner._send_resolution_notification(pred, result)
    out = capsys.readouterr().out
    out.encode("cp1252")
    assert "[PREDICTION RESOLVED] BTC-USD -> HIGHER [WIN]" in out
    assert "Exit: $94,318" in out
