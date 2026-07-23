"""Dataclass models for database records — pure data, no business logic."""

from dataclasses import dataclass
from datetime import datetime

RISK_GRADE_HIGH = "HIGH"
RISK_GRADE_MEDIUM = "MEDIUM"
RISK_GRADE_LOW = "LOW"
RISK_GRADE_UNKNOWN = "UNKNOWN"
VALID_RISK_GRADES: tuple[str, ...] = (
    RISK_GRADE_HIGH,
    RISK_GRADE_MEDIUM,
    RISK_GRADE_LOW,
    RISK_GRADE_UNKNOWN,
)


def risk_grade(value: object) -> str:
    """Return the normalized leading grade for a stored risk value.

    Phase 24 stores the full computed text (for example ``"MEDIUM — ..."``).
    Legacy rows are integer booleans: ``0`` remains non-blocking UNKNOWN and
    ``1`` retains its historical hard-risk meaning as HIGH. Unknown/malformed
    values fail open to UNKNOWN.
    """
    if isinstance(value, str):
        upper = value.strip().upper()
        for grade in VALID_RISK_GRADES:
            if upper == grade or upper.startswith(f"{grade} "):
                return grade
        return RISK_GRADE_UNKNOWN
    return RISK_GRADE_HIGH if value is True or value == 1 else RISK_GRADE_UNKNOWN


def normalize_risk_text(value: object) -> str:
    """Return full persisted risk text with a validated leading grade."""
    if isinstance(value, str) and risk_grade(value) != RISK_GRADE_UNKNOWN:
        return value.strip()
    if isinstance(value, str) and value.strip().upper() == RISK_GRADE_UNKNOWN:
        return RISK_GRADE_UNKNOWN
    return risk_grade(value)


def is_hard_risk(value: object) -> bool:
    """True only for the existing hard-blackout-equivalent HIGH grade."""
    return risk_grade(value) == RISK_GRADE_HIGH


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
    earnings_risk: str = RISK_GRADE_UNKNOWN
    news_risk: str = RISK_GRADE_UNKNOWN
    raw_indicators_json: str | None = None
    # Phase 17: set (via db.mark_signals_considered) when the live candidate
    # source pulls this signal into an execute-bound plan — at PULL time, not
    # execution time. NULL = never considered. insert_signal does not write it.
    considered_at: datetime | None = None
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
    # Phase 5 advisory sentiment, captured at fire time. ``sentiment_ok`` is
    # deliberately nullable: True = a genuine LLM result, False = fail-soft
    # neutral, and None = legacy/not evaluated provenance.
    sentiment_score: float | None = None   # -1.0..+1.0
    sentiment_label: str | None = None      # bullish | neutral | bearish
    sentiment_ok: bool | None = None
    sentiment_rationale: str | None = None
    heavy_news: bool = False                # headline volume above threshold
    headline_count: int | None = None       # headlines the score was based on
    # Phase 6 advisory indicator families, captured at fire time. ``ind_ok``
    # is nullable: True = successful computation, False = fail-soft bundle,
    # and None = legacy/not evaluated provenance.
    ind_ok: bool | None = None
    ind_atr: float | None = None            # ATR (absolute price distance)
    ind_realized_vol: float | None = None   # realized vol (fractional)
    ind_vol_regime: str | None = None       # low | normal | high | unknown
    ind_rsi: float | None = None            # momentum oscillator
    ind_adx: float | None = None            # trend strength (direction-agnostic)
    ind_obv: float | None = None            # cumulative volume flow
    ind_correlation: float | None = None    # avg pairwise corr vs active set
    ind_concentration: str | None = None    # concentrated | moderate | diversified | unknown
    # Phase 7 advisory risk recommendation, captured at fire time. ``risk_ok``
    # is nullable: True = valid assessment, False = fail-soft failure, and
    # None = legacy/not evaluated provenance.
    risk_ok: bool | None = None
    risk_reason: str | None = None
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


# Phase 19: where a long_term_positions row came from. 'long_term' is the
# genuine Phase 14 buy-and-hold entry; 'swing_fallback' is a Phase 13 options
# hierarchy third-rung shares position — it carries SWING intent and keeps its
# ORIGINAL tp/sl/deadline exit rules, never the trend/drawdown rules.
VALID_POSITION_SOURCES: frozenset[str] = frozenset({"long_term", "swing_fallback"})
VALID_POSITION_DIRECTIONS: frozenset[str] = frozenset({"long", "short"})


@dataclass(frozen=True)
class LongTermPosition:
    """An open/closed position in the long-term lifecycle book.

    Phase 14 rows (``source='long_term'``) are buy-and-hold: no tight stop,
    protective trend-breakdown / drawdown exits only. Phase 19 rows
    (``source='swing_fallback'``) are Phase 13 shares-fallback SWING positions
    tracked in this table for lifecycle/reconciliation — they carry their
    ORIGINAL swing ``tp``/``sl``/``deadline`` and the watcher applies THOSE,
    never the long-term rules. ``direction`` is 'short' when a put-signal
    fallback sold shares (its close must BUY)."""

    ticker: str
    asset_class: str
    entry_price: float
    entry_date: datetime
    qty: float
    status: str = "open"           # 'open' | 'closed'
    exit_price: float | None = None
    exit_date: datetime | None = None
    # long_term: trend_breakdown | drawdown_stop | emergency_shutdown
    # swing_fallback: take_profit | stop_loss | hold_deadline | emergency_shutdown
    exit_reason: str | None = None
    source: str = "long_term"      # 'long_term' | 'swing_fallback' (Phase 19)
    direction: str = "long"        # 'long' | 'short' (Phase 19)
    tp: float | None = None        # swing_fallback only: original swing levels
    sl: float | None = None
    deadline: datetime | None = None
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


# Cross-cutting order-lifecycle audit: durable accepted-order ledger.  The
# normalized status deliberately mirrors the broker-neutral vocabulary except
# that transport ``error`` is not durable broker truth.  ``expired`` remains
# distinct here even though the current Alpaca adapter maps it to ``canceled``;
# the raw broker status is retained alongside it for the lifecycle materializer.
PENDING_ORDER_INTENT_VERSION = 1
VALID_PENDING_ORDER_INTENT_KINDS: frozenset[str] = frozenset(
    {"option", "long_term", "shares_fallback"}
)
VALID_PENDING_ORDER_STATUSES: frozenset[str] = frozenset(
    {
        "prepared", "new", "partially_filled", "filled", "canceled",
        "rejected", "expired", "unknown", "abandoned",
    }
)
TERMINAL_PENDING_ORDER_STATUSES: frozenset[str] = frozenset(
    {"filled", "canceled", "rejected", "expired", "abandoned"}
)
VALID_PENDING_ORDER_VEHICLES: frozenset[str] = frozenset(
    {"option_full", "option_undersized", "shares"}
)
VALID_PENDING_ORDER_POSITION_KINDS: frozenset[str] = frozenset(
    {"option", "long_term"}
)
VALID_PENDING_ORDER_SIDES: frozenset[str] = frozenset({"buy", "sell"})

# Reserved for rows that predate the atomic submit handoff.  These values are
# durable uniqueness keys only: they were never sent to Alpaca and therefore
# must never be used with Alpaca's order-by-client-ID recovery endpoint.
LEGACY_PENDING_ORDER_CLIENT_ID_PREFIX = "legacy-"


def is_recoverable_pending_order_client_id(client_order_id: str | None) -> bool:
    """Whether a client ID is eligible for provider-side recovery lookup."""
    return bool(
        client_order_id
        and not client_order_id.startswith(LEGACY_PENDING_ORDER_CLIENT_ID_PREFIX)
    )


@dataclass(frozen=True)
class PendingOrder:
    """Prepared or accepted broker order awaiting fill materialization.

    Common submission intent is normalized for restart-safe lookup.  Target-
    specific materialization data lives in immutable ``intent_payload_json``;
    ``intent_payload_version`` selects its decoder.  Version 1 payloads use an
    ``intent_kind`` of ``option``, ``long_term``, or ``shares_fallback``.

    Lifecycle fields always represent the latest *usable* cumulative broker
    snapshot.  Broker read errors are not records and must leave these values
    unchanged.  ``position_kind`` plus ``position_id`` is the typed, one-time
    link to the eventual option_positions or long_term_positions row.
    """

    broker_order_id: str | None
    ticker: str
    broker_symbol: str
    asset_class: str
    vehicle: str
    target_position_kind: str
    side: str
    requested_qty: float
    requested_limit_price: float | None
    submitted_at: datetime
    intent_payload_json: str
    client_order_id: str | None = None
    signal_id: int | None = None
    intent_payload_version: int = PENDING_ORDER_INTENT_VERSION
    lifecycle_status: str = "new"
    broker_status: str | None = None
    filled_qty: float = 0.0
    filled_avg_price: float | None = None
    last_refreshed_at: datetime | None = None
    terminal_reason: str | None = None
    terminal_at: datetime | None = None
    position_kind: str | None = None
    position_id: int | None = None
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
