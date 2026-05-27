"""15-minute direction prediction engine for crypto event markets.

Phase 2.2b. Predicts HIGHER / LOWER for a 15-minute window using six
short-term indicators that vote on direction. Tagged with regime + VIX
at creation time so accuracy can later be broken down by macro context.

**This module does NOT place bets.** Robinhood has no public API for
event contracts; the bot only notifies the user, who places bets
manually. The prediction subsystem is also entirely separate from the
swing trade scanner — different table, different reports, different
lifecycle.

The six voting indicators:

1. **Candle streak** — 3 consecutive green/red candles (last 3 closed)
2. **RSI(14) slope** — sign of (rsi_now - rsi_3_back)
3. **MACD histogram direction** — sign of (hist_now - hist_prev)
4. **Price vs VWAP** — above/below session VWAP
5. **Volume confirmation** — current volume vs 20-candle MA, sided by candle color
6. **Bollinger position** — upper/lower half of BB(20, 2)

Each indicator returns +1 / -1 / 0 (HIGHER / LOWER / NEUTRAL). The net
score's sign sets direction; magnitude sets confidence. A confidence
below 55% (i.e. tied within 5 points of 50/50) returns ``None`` — we
skip the prediction rather than guess. Skipping is honest; guessing
poisons the accuracy stats.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf

from trading_bot import db, regime, vix
from trading_bot.models import Prediction

# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────

_INTERVAL = "15m"
_FETCH_PERIOD = "5d"
_WINDOW_MINUTES = 15
_MIN_CANDLES = 30                # need at least 30 closed 15-min candles
_TIE_TOLERANCE_PCT = 5.0         # confidence must beat 50% by > 5 points
_MIN_CONFIDENCE = 50.0 + _TIE_TOLERANCE_PCT

_INDICATOR_NAMES = (
    "candle_streak", "rsi_slope", "macd_hist", "vwap", "volume_conf", "bb_position",
)


# ────────────────────────────────────────────────────────────────────────────
# Result types
# ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ResolutionResult:
    prediction_id: int
    status: str        # 'resolved' | 'unresolved' | 'error'
    outcome: str | None = None    # 'correct' | 'incorrect' | 'push' | None
    entry_price: float | None = None
    exit_price: float | None = None
    detail: str | None = None     # human-readable summary


# ────────────────────────────────────────────────────────────────────────────
# Indicator math (pure)
# ────────────────────────────────────────────────────────────────────────────


def _candle_streak_vote(df: pd.DataFrame) -> int:
    last3 = df.iloc[-3:]
    greens = (last3["Close"] > last3["Open"]).sum()
    reds = (last3["Close"] < last3["Open"]).sum()
    if greens == 3:
        return 1
    if reds == 3:
        return -1
    return 0


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _rsi_slope_vote(df: pd.DataFrame, lookback: int = 3) -> int:
    rsi_series = _rsi(df["Close"])
    if rsi_series.dropna().shape[0] < lookback + 1:
        return 0
    rsi_now = float(rsi_series.iloc[-1])
    rsi_then = float(rsi_series.iloc[-1 - lookback])
    if pd.isna(rsi_now) or pd.isna(rsi_then):
        return 0
    if rsi_now > rsi_then:
        return 1
    if rsi_now < rsi_then:
        return -1
    return 0


def _macd_hist_vote(df: pd.DataFrame) -> int:
    close = df["Close"]
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    if hist.shape[0] < 2:
        return 0
    hist_now = float(hist.iloc[-1])
    hist_prev = float(hist.iloc[-2])
    if hist_now > hist_prev:
        return 1
    if hist_now < hist_prev:
        return -1
    return 0


def _vwap_vote(df: pd.DataFrame) -> int:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    cum_vp = (typical * df["Volume"]).cumsum()
    cum_v = df["Volume"].cumsum()
    # Avoid divide-by-zero on synthetic zero-volume inputs.
    vwap = cum_vp / cum_v.replace(0, pd.NA)
    if pd.isna(vwap.iloc[-1]):
        return 0
    price = float(df["Close"].iloc[-1])
    vw = float(vwap.iloc[-1])
    if price > vw:
        return 1
    if price < vw:
        return -1
    return 0


def _volume_conf_vote(df: pd.DataFrame, ma_period: int = 20) -> int:
    vol_ma = df["Volume"].rolling(ma_period).mean()
    if pd.isna(vol_ma.iloc[-1]):
        return 0
    vol_now = float(df["Volume"].iloc[-1])
    vol_ma_now = float(vol_ma.iloc[-1])
    if vol_now <= vol_ma_now:
        return 0
    # Volume above MA confirms whichever direction the candle went.
    last_open = float(df["Open"].iloc[-1])
    last_close = float(df["Close"].iloc[-1])
    if last_close > last_open:
        return 1
    if last_close < last_open:
        return -1
    return 0


def _bb_position_vote(df: pd.DataFrame, period: int = 20) -> int:
    close = df["Close"]
    middle = close.rolling(period).mean()
    if pd.isna(middle.iloc[-1]):
        return 0
    price = float(close.iloc[-1])
    mid = float(middle.iloc[-1])
    if price > mid:
        return 1
    if price < mid:
        return -1
    return 0


def _gather_votes(df: pd.DataFrame) -> dict[str, int]:
    return {
        "candle_streak": _candle_streak_vote(df),
        "rsi_slope":     _rsi_slope_vote(df),
        "macd_hist":     _macd_hist_vote(df),
        "vwap":          _vwap_vote(df),
        "volume_conf":   _volume_conf_vote(df),
        "bb_position":   _bb_position_vote(df),
    }


def tally_votes(votes: dict[str, int]) -> tuple[str | None, float]:
    """Pure tally: turn 6 indicator votes into (direction, confidence).

    Returns ``(None, confidence)`` when the verdict is too close to call
    (confidence within ``_TIE_TOLERANCE_PCT`` of 50%, or all-neutral).
    Confidence is winning_side / decided * 100. Exposed for testability.
    """
    higher = sum(1 for v in votes.values() if v == 1)
    lower = sum(1 for v in votes.values() if v == -1)
    decided = higher + lower
    if decided == 0:
        return (None, 0.0)
    if higher >= lower:
        direction = "HIGHER"
        confidence = higher / decided * 100.0
    else:
        direction = "LOWER"
        confidence = lower / decided * 100.0
    if confidence < _MIN_CONFIDENCE:
        return (None, confidence)
    return (direction, confidence)


# ────────────────────────────────────────────────────────────────────────────
# Fetch
# ────────────────────────────────────────────────────────────────────────────


def _fetch_15m_candles(ticker: str) -> pd.DataFrame | None:
    """Pull recent 15-min candles. Returns None on any failure or empty."""
    try:
        df = yf.download(
            ticker, period=_FETCH_PERIOD, interval=_INTERVAL,
            progress=False, auto_adjust=False,
        )
    except Exception as exc:  # noqa: BLE001 - yfinance raises anything
        print(f"  prediction fetch error for {ticker}: {exc}", file=sys.stderr)
        return None
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.dropna(subset=["Close", "Open", "High", "Low", "Volume"])
    if df.shape[0] < _MIN_CANDLES:
        return None
    return df


# ────────────────────────────────────────────────────────────────────────────
# Public API — prediction
# ────────────────────────────────────────────────────────────────────────────


def predict_direction(
    ticker: str, now: datetime | None = None,
) -> Prediction | None:
    """Compute a direction prediction. Does NOT log or notify.

    Returns ``None`` when:

    * yfinance data is unavailable or insufficient
    * all six indicators voted neutral
    * the winning side's confidence is within the tie tolerance of 50%

    Otherwise returns a fully-tagged :class:`Prediction` ready to insert.
    Regime + VIX are tagged at call time; their fetch errors fall back to
    ``'unknown'`` rather than blocking the prediction.
    """
    df = _fetch_15m_candles(ticker)
    if df is None:
        return None

    votes = _gather_votes(df)
    direction, confidence = tally_votes(votes)
    if direction is None:
        return None

    entry_price = float(df["Close"].iloc[-1])
    created_at = now if now is not None else datetime.now(UTC)
    target_window_end = created_at + timedelta(minutes=_WINDOW_MINUTES)

    # Context tags. Same fail-soft contract as the swing trade scanner.
    try:
        market_regime = regime.get_current_regime().regime
    except regime.RegimeFetchError:
        market_regime = "unknown"

    vix_level: float | None
    vix_band: str
    try:
        snap = vix.get_current_vix()
        vix_level = snap.vix_level
        vix_band = snap.vix_band
    except vix.VixFetchError:
        vix_level = None
        vix_band = "unknown"

    return Prediction(
        ticker=ticker,
        direction=direction,
        confidence=round(confidence, 2),
        entry_price=entry_price,
        target_window_end=target_window_end,
        signals_used=json.dumps(votes),
        created_at=created_at,
        market_regime=market_regime,
        vix_band=vix_band,
        vix_level=vix_level,
    )


# ────────────────────────────────────────────────────────────────────────────
# Public API — resolution
# ────────────────────────────────────────────────────────────────────────────


def _outcome_from_prices(
    direction: str, entry: float, exit_price: float,
) -> str:
    if exit_price == entry:
        return "push"
    if direction == "HIGHER":
        return "correct" if exit_price > entry else "incorrect"
    # direction == "LOWER"
    return "correct" if exit_price < entry else "incorrect"


def _find_close_at(df: pd.DataFrame, target: datetime) -> float | None:
    """Find the 15-min candle close timestamp matching ``target``.

    The candle that closes at ``T`` is indexed at ``T - 15min`` in
    yfinance (the index represents the candle's open). We look for an
    index entry whose start + 15min equals ``target``. Returns the close
    price or None if no such candle exists yet.
    """
    target_open = target - timedelta(minutes=_WINDOW_MINUTES)
    # Normalize to UTC for comparison; yfinance returns tz-aware UTC for crypto.
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC")
    target_ts = pd.Timestamp(target_open).tz_convert("UTC") \
        if pd.Timestamp(target_open).tzinfo is not None \
        else pd.Timestamp(target_open, tz="UTC")
    matches = idx == target_ts
    if not bool(matches.any()):
        return None
    pos = int(matches.argmax())
    return float(df["Close"].iloc[pos])


def resolve_prediction(
    prediction_id: int, now: datetime | None = None,
) -> ResolutionResult:
    """Resolve a single prediction by ID.

    If the target candle hasn't closed yet (yfinance lag, recent
    prediction), returns ``status='unresolved'`` without touching the row.
    The next resolution sweep will try again.
    """
    pred = db.get_prediction(prediction_id)
    if pred is None:
        return ResolutionResult(
            prediction_id=prediction_id, status="error",
            detail="prediction not found",
        )
    if pred.resolved_at is not None:
        return ResolutionResult(
            prediction_id=prediction_id, status="resolved",
            outcome=pred.outcome, entry_price=pred.entry_price,
            exit_price=pred.exit_price, detail="already resolved",
        )

    df = _fetch_15m_candles(pred.ticker)
    if df is None:
        return ResolutionResult(
            prediction_id=prediction_id, status="unresolved",
            detail="no candle data available",
        )

    exit_price = _find_close_at(df, pred.target_window_end)
    if exit_price is None:
        return ResolutionResult(
            prediction_id=prediction_id, status="unresolved",
            detail="target candle has not closed yet",
        )

    outcome = _outcome_from_prices(pred.direction, pred.entry_price, exit_price)
    resolved_at = now if now is not None else datetime.now(UTC)
    db.update_prediction(
        prediction_id,
        resolved_at=resolved_at,
        exit_price=exit_price,
        outcome=outcome,
    )
    return ResolutionResult(
        prediction_id=prediction_id, status="resolved",
        outcome=outcome, entry_price=pred.entry_price, exit_price=exit_price,
    )


def resolve_due_predictions(
    now: datetime | None = None,
) -> list[ResolutionResult]:
    """Resolve every prediction whose target window has passed and which
    is still unresolved. Idempotent — running mid-sweep is safe."""
    cutoff = now if now is not None else datetime.now(UTC)
    candidates = db.get_unresolved_predictions(now=cutoff)
    results: list[ResolutionResult] = []
    for pred in candidates:
        if pred.id is None:
            continue
        try:
            results.append(resolve_prediction(pred.id, now=cutoff))
        except Exception as exc:  # noqa: BLE001 - resolution must never crash the loop
            results.append(ResolutionResult(
                prediction_id=pred.id, status="error", detail=str(exc),
            ))
    return results


# ────────────────────────────────────────────────────────────────────────────
# Reporting helpers (used by performance + CLI)
# ────────────────────────────────────────────────────────────────────────────


def _decode_signals(prediction: Prediction) -> dict[str, Any]:
    """Decode the signals_used JSON. Empty dict on parse failure."""
    try:
        loaded = json.loads(prediction.signals_used)
    except (json.JSONDecodeError, TypeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}
