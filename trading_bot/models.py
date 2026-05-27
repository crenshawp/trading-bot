"""Dataclass models for database records — pure data, no business logic."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Signal:
    timestamp: datetime
    ticker: str
    asset_class: str            # "stock" or "crypto"
    signal_type: str            # "ema21_pullback", "oversold_reversal", etc.
    direction: str              # "call", "put", "long", "short"
    entry_price: float
    stop_loss: float | None = None
    take_profit: float | None = None
    atr: float | None = None
    rsi: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    ema21: float | None = None
    bb_upper: float | None = None
    bb_lower: float | None = None
    hold_estimate_days: int | None = None
    earnings_risk: bool = False
    news_risk: bool = False
    raw_indicators_json: str | None = None
    id: int | None = None       # populated after insert


@dataclass(frozen=True)
class Trade:
    signal_id: int
    opened_at: datetime
    closed_at: datetime | None = None
    exit_price: float | None = None
    outcome: str | None = None  # "win", "loss", "breakeven", "open", "expired"
    pnl_pct: float | None = None
    pnl_dollars: float | None = None
    notes: str | None = None
    market_regime: str | None = None  # "bull" | "bear" | "sideways" | "unknown" (Phase 2.1)
    vix_level: float | None = None    # close-of-day VIX at fire time (Phase 2.2)
    vix_band: str | None = None       # "low" | "elevated" | "high" | "extreme" | "unknown"
    context_score: int | None = None  # 0-5 composite (Phase 2.3); NULL = un-backfilled
    id: int | None = None


@dataclass(frozen=True)
class DailyPerf:
    date: str                   # YYYY-MM-DD
    signals_fired: int = 0
    trades_opened: int = 0
    trades_closed: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float | None = None
    total_pnl_pct: float | None = None
    id: int | None = None


VALID_ASSET_CLASSES: frozenset[str] = frozenset({"stock", "crypto"})
VALID_DIRECTIONS: frozenset[str] = frozenset({"call", "put", "long", "short"})
VALID_OUTCOMES: frozenset[str] = frozenset(
    {"win", "loss", "breakeven", "open", "expired"}
)
# Phase 2.1 macro regime tags. "unknown" is the fallback when SPY data is
# unfetchable at signal fire time — we still log the trade rather than skip it.
VALID_MARKET_REGIMES: frozenset[str] = frozenset(
    {"bull", "bear", "sideways", "unknown"}
)
# Phase 2.2 VIX bands. "unknown" is the fallback when VIX is unfetchable
# at fire time — we still log the trade rather than skipping it.
VALID_VIX_BANDS: frozenset[str] = frozenset(
    {"low", "elevated", "high", "extreme", "unknown"}
)
# Phase 2.2b 15-min prediction outcomes. "push" handles exit == entry
# (rare on crypto but possible). NULL means not yet resolved.
VALID_PREDICTION_OUTCOMES: frozenset[str] = frozenset(
    {"correct", "incorrect", "push"}
)
VALID_PREDICTION_DIRECTIONS: frozenset[str] = frozenset({"HIGHER", "LOWER"})


@dataclass(frozen=True)
class Prediction:
    """A 15-min direction prediction (Phase 2.2b).

    Tracked entirely separately from trades — different table, different
    reports. Created at fire time, resolved 15 minutes later when the
    target candle closes. ``signals_used`` is a JSON-encoded string
    persisted verbatim into the DB so we can audit per-prediction what
    each indicator contributed.
    """
    ticker: str
    direction: str                # 'HIGHER' | 'LOWER'
    confidence: float             # 0-100
    entry_price: float
    target_window_end: datetime
    signals_used: str             # JSON-encoded dict
    created_at: datetime
    market_regime: str | None = None
    vix_band: str | None = None
    vix_level: float | None = None
    resolved_at: datetime | None = None
    exit_price: float | None = None
    outcome: str | None = None    # 'correct' | 'incorrect' | 'push' | None
    notified: bool = False
    context_score: int | None = None  # 0-5 composite (Phase 2.3); NULL = un-backfilled
    id: int | None = None
