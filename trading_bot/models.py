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
    track_mode: str = "active"        # "active" | "shadow" (Phase 3.1-LIVE)
    # Phase 5 advisory sentiment, captured at fire time (NULL = not scored).
    sentiment_score: float | None = None   # -1.0..+1.0
    sentiment_label: str | None = None      # bullish | neutral | bearish
    heavy_news: bool = False                # headline volume above threshold
    headline_count: int | None = None       # headlines the score was based on
    # Phase 6 advisory indicator families, captured at fire time (NULL = not
    # computed — shadow/crypto trades, or a fail-soft empty bundle).
    ind_atr: float | None = None            # ATR (absolute price distance)
    ind_realized_vol: float | None = None   # realized vol (fractional)
    ind_vol_regime: str | None = None       # low | normal | high | unknown
    ind_rsi: float | None = None            # momentum oscillator
    ind_adx: float | None = None            # trend strength (direction-agnostic)
    ind_obv: float | None = None            # cumulative volume flow
    ind_correlation: float | None = None    # avg pairwise corr vs active set
    ind_concentration: str | None = None    # concentrated | moderate | diversified | unknown
    # Phase 7 advisory risk recommendation, captured at fire time (NULL = not
    # computed — shadow/crypto trades, or a fail-soft unavailable assessment).
    risk_recommended_size: float | None = None  # units (shares/contracts)
    risk_stop_distance: float | None = None      # ATR * stop multiple
    risk_dollar_risk: float | None = None        # notional $ at risk
    risk_pct: float | None = None                # dollar_risk as % of notional
    risk_position_pct: float | None = None       # position value as % of notional
    risk_capped: bool = False                    # per-position cap bound the size
    risk_total_pct: float | None = None          # portfolio open + candidate risk %
    risk_portfolio_verdict: str | None = None    # ok | would-exceed-portfolio | unknown
    risk_position_verdict: str | None = None     # ok | would-exceed-position | unknown
    risk_cluster_pct: float | None = None        # concentrated-cluster risk %
    risk_cluster_verdict: str | None = None      # ok | would-exceed-cluster | unknown
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
# Phase 3.1-LIVE: how a trade is tracked. 'active' trades alert and feed the
# headline reports; 'shadow' trades are opened silently on the shadow universe
# (no alert) and drive live-shadow promotion. Existing rows backfill to 'active'.
VALID_TRACK_MODES: frozenset[str] = frozenset({"active", "shadow"})
# Phase 3.3: active_watchlist status. 'active' tickers alert; 'benched' tickers
# are still scanned/resolved (data never stops) but their alerts are suppressed.
VALID_WATCHLIST_STATUSES: frozenset[str] = frozenset({"active", "benched"})
# Phase 4: (ticker, signal_type) pair gate. 'enabled' pairs alert; 'muted'
# pairs fire as shadow (no alert) but keep collecting data. A pair with no row
# is treated as 'enabled' (default-enabled).
VALID_PAIR_STATUSES: frozenset[str] = frozenset({"enabled", "muted"})


@dataclass(frozen=True)
class LongTermCandidate:
    """A buy-and-hold entry candidate to feed the Phase 12 allocation plan (Phase 14).

    ``signal_type`` is one of ``config.LONGTERM_STOCK_SIGNAL`` /
    ``LONGTERM_CRYPTO_SIGNAL`` — the ONLY long-term types that route to execution.
    ``entry_rationale`` records which gates it cleared (audit)."""

    ticker: str
    asset_class: str               # 'stock' | 'crypto'
    signal_type: str               # long_term_stock | long_term_crypto
    entry_price: float
    entry_rationale: str = ""


@dataclass(frozen=True)
class LongTermPosition:
    """An open/closed buy-and-hold position (Phase 14; no tight stop — protective
    exit only)."""

    ticker: str
    asset_class: str
    entry_price: float
    entry_date: datetime
    qty: float
    status: str = "open"           # 'open' | 'closed'
    exit_price: float | None = None
    exit_date: datetime | None = None
    exit_reason: str | None = None  # trend_breakdown | drawdown_stop
    id: int | None = None


# Phase 16: outcome of one planned order in an execute run. 'submitted' means
# the broker accepted the order (this is what the idempotency guard blocks on);
# 'rejected' is a structured broker refusal; 'error' is a transport/unexpected
# failure; 'skipped' records an order the run declined to submit (e.g. already
# executed this cycle).
VALID_PLAN_EXECUTION_STATUSES: frozenset[str] = frozenset(
    {"submitted", "rejected", "error", "skipped"}
)


@dataclass(frozen=True)
class PlanExecution:
    """One plan_executions audit row (Phase 16): what the execute command did
    with one planned order, linking the plan run (``plan_id``) to the broker
    order (``order_ref``)."""

    plan_id: str
    executed_at: datetime
    ticker: str
    pool: str
    status: str                     # submitted | rejected | error | skipped
    signal_type: str | None = None
    side: str | None = None
    qty: float | None = None
    vehicle: str | None = None      # option_full | option_undersized | shares
    order_ref: str | None = None    # broker order id when submitted
    reason: str = ""
    id: int | None = None


@dataclass(frozen=True)
class OptionPosition:
    """An open/closed single-leg option position (Phase 13).

    Distinct from ``Trade`` (which is equity, ``pnl_dollars`` always None): an
    option carries a contract-aware dollar cost/PnL through the 100-share
    ``multiplier``. ``tp``/``sl`` are the UNDERLYING price levels (from the
    originating signal, derived exactly as the equity resolver's are); the manual
    exit watcher closes the contract when the underlying crosses them or the
    ``deadline`` passes. Greeks are advisory context, never gates.
    """

    symbol: str                     # OCC symbol
    underlying: str
    option_type: str                # 'call' | 'put'
    strike: float
    expiry: str                     # 'YYYY-MM-DD'
    contracts: float                # whole contracts
    opened_at: datetime
    multiplier: int = 100
    signal_id: int | None = None
    order_id: str | None = None
    premium_entry: float | None = None
    delta_entry: float | None = None
    theta: float | None = None
    vega: float | None = None
    gamma: float | None = None
    tp: float | None = None          # underlying take-profit level
    sl: float | None = None          # underlying stop-loss level
    deadline: datetime | None = None
    closed_at: datetime | None = None
    exit_price: float | None = None
    outcome: str | None = None       # 'win' | 'loss' | 'expired' | 'open'
    pnl_dollars: float | None = None
    vehicle: str = "option_full"     # option_full | option_undersized
    id: int | None = None


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
