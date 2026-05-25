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
