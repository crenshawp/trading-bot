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
from zoneinfo import ZoneInfo

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


def _strong_uptrend(n: int = 40) -> pd.DataFrame:
    """Accelerating uptrend that yields >= 5 of 6 HIGHER votes.

    A perfectly linear ramp saturates RSI (slope 0) and inverts the MACD
    histogram, so it only scores 4/6. A choppy base followed by an
    acceleration keeps RSI unsaturated and the MACD histogram rising, which
    clears the 5-of-6 agreement gate.
    """
    closes: list[float] = []
    price = 100.0
    for i in range(n):
        if i < 30:
            price += 0.2 + (0.15 if i % 2 == 0 else -0.1)  # choppy mild rise
        else:
            price += 1.0 + (i - 29) * 0.3                  # accelerate
        closes.append(round(price, 2))
    opens = [closes[i] - 0.2 for i in range(n)]            # green candles
    volumes = [1_000_000] * (n - 1) + [3_000_000]          # volume spike last
    return _build_candles(closes=closes, opens=opens, volumes=volumes)


def _strong_downtrend(n: int = 40) -> pd.DataFrame:
    """Accelerating downtrend that yields >= 5 of 6 LOWER votes."""
    closes: list[float] = []
    price = 300.0
    for i in range(n):
        if i < 30:
            price += -0.2 + (0.1 if i % 2 == 0 else -0.15)
        else:
            price -= 1.0 + (i - 29) * 0.3
        closes.append(round(price, 2))
    opens = [closes[i] + 0.2 for i in range(n)]            # red candles
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


def _with_forming_reversal(closed: pd.DataFrame) -> pd.DataFrame:
    """Append a still-forming row that reverses the real indicator verdict."""
    last_close = float(closed["Close"].iloc[-1])
    forming_start = closed.index[-1] + timedelta(minutes=15)
    forming = pd.DataFrame(
        {
            "Open": [last_close],
            "High": [last_close + 1.0],
            "Low": [0.0],
            "Close": [1.0],
            "Volume": [1_000_000_000],
        },
        index=pd.DatetimeIndex([forming_start]),
    )
    return pd.concat([closed, forming])


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


def test_tally_four_of_six_returns_none() -> None:
    # 4 HIGHER, 1 LOWER, 1 NEUTRAL → 4/5 = 80% confidence, but only 4
    # indicators agree. Hardening requires >= 5 of 6 → None.
    votes = {"a": 1, "b": 1, "c": 1, "d": 1, "e": -1, "f": 0}
    direction, confidence = predictions.tally_votes(votes)
    assert direction is None
    # confidence is still computed/reported even when the gate rejects it
    assert confidence == pytest.approx(80.0)


def test_tally_five_of_six_agreement_fires() -> None:
    # 5 HIGHER, 1 LOWER → 5/6 = 83.3% confidence, 5 agree → fires
    votes = {"a": 1, "b": 1, "c": 1, "d": 1, "e": 1, "f": -1}
    direction, confidence = predictions.tally_votes(votes)
    assert direction == "HIGHER"
    assert confidence == pytest.approx(83.33, rel=1e-3)


def test_tally_five_with_one_neutral_fires() -> None:
    # 5 HIGHER, 1 NEUTRAL → 5/5 = 100% confidence, 5 agree → fires
    votes = {"a": 1, "b": 1, "c": 1, "d": 1, "e": 1, "f": 0}
    direction, confidence = predictions.tally_votes(votes)
    assert direction == "HIGHER"
    assert confidence == pytest.approx(100.0)


# ────────────────────── predict_direction (with full stack) ──────────────────────


def test_predict_strong_uptrend_returns_higher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fetch(monkeypatch, _strong_uptrend())
    pred = predictions.predict_direction("BTC-USD")
    assert pred is not None
    assert pred.direction == "HIGHER"
    assert pred.confidence >= predictions.MIN_CONFIDENCE


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


def test_predict_four_of_six_agreement_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only 4 of 6 indicators agree → None, even though 4/4 = high confidence.
    The 5-of-6 agreement gate (in tally_votes) rejects it."""
    _patch_fetch(monkeypatch, _strong_uptrend())
    monkeypatch.setattr(
        predictions, "_gather_votes",
        lambda _df: {
            "candle_streak": 1, "rsi_slope": 1, "macd_hist": 1, "vwap": 1,
            "volume_conf": -1, "bb_position": 0,
        },
    )
    assert predictions.predict_direction("BTC-USD") is None


def test_predict_confidence_below_floor_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even with agreement, confidence below MIN_CONFIDENCE (70) → None.
    Exercised by mocking tally_votes to report a low confidence."""
    _patch_fetch(monkeypatch, _strong_uptrend())
    monkeypatch.setattr(
        predictions, "tally_votes", lambda _v: ("HIGHER", 65.0),
    )
    assert predictions.predict_direction("BTC-USD") is None


def test_predict_at_confidence_floor_fires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confidence exactly at the floor (70) with agreement → fires."""
    _patch_fetch(monkeypatch, _strong_uptrend())
    monkeypatch.setattr(
        predictions, "tally_votes", lambda _v: ("HIGHER", 70.0),
    )
    pred = predictions.predict_direction("BTC-USD")
    assert pred is not None
    assert pred.confidence == pytest.approx(70.0)


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
    now = datetime(2026, 5, 27, 0, 0, tzinfo=UTC)
    pred = predictions.predict_direction("BTC-USD", now=now)
    assert pred is not None
    assert pred.target_window_end == now + timedelta(minutes=15)


def test_predict_excludes_forming_row_from_votes_and_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = _strong_uptrend()
    candles = _with_forming_reversal(closed)
    now = candles.index[-1].to_pydatetime() + timedelta(minutes=7)
    assert predictions.tally_votes(predictions._gather_votes(candles))[0] == "LOWER"

    _patch_fetch(monkeypatch, candles)
    pred = predictions.predict_direction("BTC-USD", now=now)

    assert pred is not None
    assert pred.direction == "HIGHER"
    assert pred.entry_price == pytest.approx(float(closed["Close"].iloc[-1]))
    assert json.loads(pred.signals_used) == predictions._gather_votes(closed)


def test_predict_keeps_latest_row_when_provider_omits_forming_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = _strong_uptrend()
    now = closed.index[-1].to_pydatetime() + timedelta(minutes=22)
    _patch_fetch(monkeypatch, closed)

    pred = predictions.predict_direction("BTC-USD", now=now)

    assert pred is not None
    assert pred.direction == "HIGHER"
    assert pred.entry_price == pytest.approx(float(closed["Close"].iloc[-1]))


def test_predict_includes_row_at_exact_close_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candles = _with_forming_reversal(_strong_uptrend())
    now = candles.index[-1].to_pydatetime() + timedelta(minutes=15)
    _patch_fetch(monkeypatch, candles)

    pred = predictions.predict_direction("BTC-USD", now=now)

    assert pred is not None
    assert pred.direction == "LOWER"
    assert pred.entry_price == pytest.approx(1.0)


def test_predict_compares_aware_index_in_its_timezone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candles = _with_forming_reversal(_strong_uptrend())
    candles = candles.tz_convert("America/New_York")
    now_utc = (
        candles.index[-1].tz_convert("UTC").to_pydatetime()
        + timedelta(minutes=7)
    )
    _patch_fetch(monkeypatch, candles)

    pred = predictions.predict_direction("BTC-USD", now=now_utc)

    assert pred is not None
    assert pred.direction == "HIGHER"
    assert pred.entry_price != pytest.approx(1.0)


def test_predict_treats_naive_index_as_exchange_local_wall_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candles = _with_forming_reversal(_strong_uptrend())
    local_zone = ZoneInfo("America/New_York")
    local_candles = candles.tz_convert(local_zone)
    now_local = (
        local_candles.index[-1].to_pydatetime() + timedelta(minutes=7)
    )
    local_candles.index = local_candles.index.tz_localize(None)
    _patch_fetch(monkeypatch, local_candles)

    pred = predictions.predict_direction("BTC-USD", now=now_local)

    assert pred is not None
    assert pred.direction == "HIGHER"
    assert pred.entry_price != pytest.approx(1.0)


def test_predict_fails_clearly_when_closed_history_is_insufficient(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candles = _strong_uptrend(n=30)
    now = candles.index[-1].to_pydatetime() + timedelta(minutes=7)
    _patch_fetch(monkeypatch, candles)

    assert predictions.predict_direction("BTC-USD", now=now) is None
    assert (
        "prediction insufficient closed candles for BTC-USD: "
        "29 available, 30 required"
    ) in capsys.readouterr().err


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


# ────────────────────── PREDICTIONS_ENABLED env override (FIX 1) ──────────────────────


def test_env_override_true_forces_enabled(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading_bot import scanner, settings

    settings.set_bool("predictions.enabled", False)  # DB says off
    monkeypatch.setenv("PREDICTIONS_ENABLED", "true")
    scanner._seed_prediction_defaults()
    assert settings.get_bool("predictions.enabled") is True


def test_env_override_false_forces_disabled(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading_bot import scanner, settings

    settings.set_bool("predictions.enabled", True)  # DB says on
    monkeypatch.setenv("PREDICTIONS_ENABLED", "FALSE")  # case-insensitive
    scanner._seed_prediction_defaults()
    assert settings.get_bool("predictions.enabled") is False


def test_env_override_unset_preserves_db(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading_bot import scanner, settings

    settings.set_bool("predictions.enabled", True)
    monkeypatch.delenv("PREDICTIONS_ENABLED", raising=False)
    scanner._seed_prediction_defaults()
    assert settings.get_bool("predictions.enabled") is True


def test_env_override_garbage_preserves_db(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading_bot import scanner, settings

    settings.set_bool("predictions.enabled", True)
    monkeypatch.setenv("PREDICTIONS_ENABLED", "maybe")
    scanner._seed_prediction_defaults()
    # unrecognized value → DB value preserved
    assert settings.get_bool("predictions.enabled") is True


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


class _FakeResp:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


def test_prediction_notification_warns_on_non_2xx(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A 429/401 from Pushover/Discord must surface — requests doesn't raise
    on 4xx, so without the status check the alert would silently vanish."""
    from trading_bot import scanner

    monkeypatch.setattr(scanner, "DRY_RUN", False)
    monkeypatch.setattr(scanner, "DISCORD_WEBHOOK_URL", "http://x")
    monkeypatch.setattr(scanner, "PUSHOVER_APP_TOKEN", "t")
    monkeypatch.setattr(scanner, "PUSHOVER_USER_KEY", "u")
    monkeypatch.setattr(
        scanner.requests, "post", lambda *_a, **_k: _FakeResp(429),
    )

    pred = Prediction(
        ticker="BTC-USD", direction="HIGHER", confidence=72.0,
        entry_price=94000.0,
        target_window_end=datetime(2026, 5, 26, 14, 45, tzinfo=UTC),
        signals_used=json.dumps({"x": 1}),
        created_at=datetime(2026, 5, 26, 14, 30, tzinfo=UTC),
        market_regime="bull", vix_band="low", vix_level=18.0,
    )
    scanner._send_prediction_notification(pred)
    err = capsys.readouterr().err
    assert "HTTP 429" in err
    assert "Discord" in err
    assert "Pushover" in err


def test_swing_notification_warns_on_non_2xx(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import scanner

    monkeypatch.setattr(scanner, "DRY_RUN", False)
    monkeypatch.setattr(scanner, "DISCORD_WEBHOOK_URL", "http://x")
    monkeypatch.setattr(scanner, "PUSHOVER_APP_TOKEN", "t")
    monkeypatch.setattr(scanner, "PUSHOVER_USER_KEY", "u")
    monkeypatch.setattr(
        scanner.requests, "post", lambda *_a, **_k: _FakeResp(401),
    )
    # Avoid a real DB write path complicating the test.
    monkeypatch.setattr(scanner, "log_signal", lambda _s: 1)

    scanner.send_notification({
        "ticker": "BTC-USD", "asset_type": "crypto",
        "trade_type": "CRYPTO", "direction": "LONG", "setup": "Oversold",
        "detail": "x", "price": 50000.0, "take_profit": 51000.0,
        "stop_loss": 49500.0, "confidence": "High", "hold_days": "2-8h",
    })
    err = capsys.readouterr().err
    assert "HTTP 401" in err


def test_notification_2xx_does_not_warn(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot import scanner

    monkeypatch.setattr(scanner, "DRY_RUN", False)
    monkeypatch.setattr(scanner, "DISCORD_WEBHOOK_URL", "http://x")
    monkeypatch.setattr(scanner, "PUSHOVER_APP_TOKEN", "t")
    monkeypatch.setattr(scanner, "PUSHOVER_USER_KEY", "u")
    monkeypatch.setattr(
        scanner.requests, "post", lambda *_a, **_k: _FakeResp(200),
    )
    pred = Prediction(
        ticker="ETH-USD", direction="LOWER", confidence=80.0,
        entry_price=3000.0,
        target_window_end=datetime(2026, 5, 26, 14, 45, tzinfo=UTC),
        signals_used=json.dumps({"x": -1}),
        created_at=datetime(2026, 5, 26, 14, 30, tzinfo=UTC),
        market_regime="bear", vix_band="high", vix_level=32.0,
    )
    scanner._send_prediction_notification(pred)
    err = capsys.readouterr().err
    assert "HTTP" not in err


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
