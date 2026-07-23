"""SQLite database layer — schema, CRUD, and helpers.

Only stdlib sqlite3 is used. Foreign keys ON, WAL journal mode, ISO 8601
datetime strings stored as TEXT (we manage adapters ourselves instead of
relying on sqlite3's deprecated defaults).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from trading_bot import config
from trading_bot.models import (
    LEGACY_PENDING_ORDER_CLIENT_ID_PREFIX,
    PENDING_ORDER_INTENT_VERSION,
    PENDING_ORDER_POSITION_EXIT_INTENT_VERSION,
    TERMINAL_PENDING_ORDER_STATUSES,
    VALID_ASSET_CLASSES,
    VALID_DIRECTIONS,
    VALID_MARKET_REGIMES,
    VALID_OUTCOMES,
    VALID_PAIR_STATUSES,
    VALID_PENDING_ORDER_FILL_TIME_SOURCES,
    VALID_PENDING_ORDER_INTENT_KINDS,
    VALID_PENDING_ORDER_POSITION_KINDS,
    VALID_PENDING_ORDER_ROLES,
    VALID_PENDING_ORDER_SIDES,
    VALID_PENDING_ORDER_STATUSES,
    VALID_PENDING_ORDER_VEHICLES,
    VALID_PLAN_EXECUTION_STATUSES,
    VALID_POSITION_DIRECTIONS,
    VALID_POSITION_SOURCES,
    VALID_PREDICTION_DIRECTIONS,
    VALID_PREDICTION_OUTCOMES,
    VALID_TRACK_MODES,
    VALID_VIX_BANDS,
    VALID_WATCHLIST_STATUSES,
    DailyPerf,
    LongTermPosition,
    OptionPosition,
    PendingOrder,
    PlanExecution,
    Prediction,
    Signal,
    Trade,
    is_recoverable_pending_order_client_id,
    normalize_risk_text,
)

# Version 12 (Phase 5) — adds trades sentiment columns (advisory news context).
# Version 11 (Phase 4) — adds signal_pair_status + signal_pair_transitions.
# Version 10 (Phase 3.3) — adds active_watchlist.status + status_changed_at
#                          and the watchlist_transitions table.
# Version 9 (Phase 3.1-LIVE) — adds the shadow_evaluations table.
# Version 8 (Phase 3.1-LIVE) — adds trades.track_mode (active vs shadow).
# Version 7 (Phase 3.1) — adds the discovery_results table (per-run scores).
# Version 6 (Phase 3.1) — adds the active_watchlist table (DB-driven watchlist).
# Version 5 (Phase 2.3) — adds trades.context_score + predictions.context_score.
# Version 4 (Phase 2.2b) — adds predictions + settings tables.
# Version 29 (cross-cutting audit) — adds exit-role order lifecycle truth.
# Version 28 (cross-cutting audit) — persists risk assessment status + reason.
# Version 27 (cross-cutting audit) — persists indicator computation status.
# Version 26 (cross-cutting audit) — persists sentiment success + rationale.
# Version 25 (cross-cutting audit) — adds pre-submit pending-order identity.
# Version 24 (cross-cutting audit) — adds the durable pending_orders ledger.
# Version 23 (Phase 24) — signals earnings/news risk store full text grades.
# Version 22 (Phase 19) — adds long_term_positions.source/direction/tp/sl/
#                          deadline (shares-fallback lifecycle tracking).
# Version 21 (Phase 17) — adds signals.considered_at (live-candidate sourcing).
# Version 20 (Phase 16) — adds the plan_executions table (execution audit trail).
# Version 19 (Phase 15) — adds the equity_snapshots table (drawdown tracking).
# Version 18 (Phase 14) — adds the long_term_positions table (buy-and-hold).
# Version 17 (Phase 13) — adds the option_positions table (single-leg options).
# Version 16 (Phase 10) — adds the readiness_state ledger.
# Version 15 (Phase 9) — adds the optimization_runs history table.
# Version 14 (Phase 7) — adds trades.risk_* advisory risk-recommendation columns.
# Version 13 (Phase 6) — adds trades.ind_* advisory indicator-family columns.
# Version 3 (Phase 2.2) — adds trades.vix_level + trades.vix_band + vix_snapshots.
# Version 2 (Phase 2.1) — adds trades.market_regime + regime_snapshots table.
SCHEMA_VERSION = 29

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    ticker TEXT NOT NULL,
    asset_class TEXT NOT NULL CHECK (asset_class IN ('stock','crypto')),
    signal_type TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('call','put','long','short')),
    entry_price REAL NOT NULL,
    stop_loss REAL,
    take_profit REAL,
    atr REAL,
    rsi REAL,
    macd REAL,
    macd_signal REAL,
    ema21 REAL,
    bb_upper REAL,
    bb_lower REAL,
    hold_estimate_days INTEGER,
    earnings_risk TEXT NOT NULL DEFAULT 'UNKNOWN',
    news_risk TEXT NOT NULL DEFAULT 'UNKNOWN',
    raw_indicators_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_signals_ticker_ts ON signals (ticker, timestamp);
CREATE INDEX IF NOT EXISTS idx_signals_type ON signals (signal_type);
CREATE UNIQUE INDEX IF NOT EXISTS idx_signals_dedupe
    ON signals (timestamp, ticker, signal_type);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER NOT NULL REFERENCES signals(id) ON DELETE CASCADE,
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    exit_price REAL,
    exit_reason TEXT,
    outcome TEXT CHECK (outcome IN ('win','loss','breakeven','open','expired')),
    pnl_pct REAL,
    pnl_dollars REAL,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_trades_signal ON trades (signal_id);
CREATE INDEX IF NOT EXISTS idx_trades_outcome ON trades (outcome);

CREATE TABLE IF NOT EXISTS daily_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL UNIQUE,
    signals_fired INTEGER NOT NULL DEFAULT 0,
    trades_opened INTEGER NOT NULL DEFAULT 0,
    trades_closed INTEGER NOT NULL DEFAULT 0,
    wins INTEGER NOT NULL DEFAULT 0,
    losses INTEGER NOT NULL DEFAULT 0,
    win_rate REAL,
    total_pnl_pct REAL
);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

-- Phase 2.1: historical macro regime snapshots, one row per date.
CREATE TABLE IF NOT EXISTS regime_snapshots (
    date TEXT PRIMARY KEY,
    regime TEXT NOT NULL,
    spy_close REAL NOT NULL,
    ema50 REAL NOT NULL,
    ema200 REAL NOT NULL,
    ema50_slope REAL NOT NULL,
    captured_at TEXT NOT NULL
);

-- Phase 2.2: historical VIX snapshots, one row per date. VIX does not trade
-- weekends, so this table is sparse on Sat/Sun and market holidays.
CREATE TABLE IF NOT EXISTS vix_snapshots (
    date TEXT PRIMARY KEY,
    vix_level REAL NOT NULL,
    vix_band TEXT NOT NULL,
    captured_at TEXT NOT NULL
);

-- Phase 2.2b: 15-min direction predictions for crypto event markets.
-- Tracked separately from trades — different lifecycle (15-min window
-- vs days/weeks), different outcome shape (correct/incorrect/push vs
-- win/loss/expired). Resolved by comparing the close of the candle
-- that ends at target_window_end against entry_price.
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('HIGHER','LOWER')),
    confidence REAL NOT NULL,
    entry_price REAL NOT NULL,
    target_window_end TEXT NOT NULL,
    signals_used TEXT NOT NULL,
    market_regime TEXT,
    vix_band TEXT,
    vix_level REAL,
    resolved_at TEXT,
    exit_price REAL,
    outcome TEXT CHECK (outcome IN ('correct','incorrect','push')),
    notified INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_predictions_unresolved
    ON predictions(target_window_end) WHERE resolved_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_predictions_ticker ON predictions(ticker);

-- Phase 2.2b: generic key/value settings.
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Phase 3.1: the live stock watchlist the scanner reads at runtime. Seeded
-- on first run from the hardcoded STOCK_WATCHLIST (source='seed'); discovery
-- auto-promotes qualifiers here (source='discovery'). The hardcoded literal
-- stays in code as the seed + fallback source.
CREATE TABLE IF NOT EXISTS active_watchlist (
    ticker TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    added_at TEXT NOT NULL
);

-- Phase 3.1: full audit trail of every discovery sweep. One row per scored
-- ticker per run, qualified or not, so we can see what discovery found and
-- why. win_rate/avg_return_pct/expectancy are NULL when a ticker had no
-- decided backtest trades.
CREATE TABLE IF NOT EXISTS discovery_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp TEXT NOT NULL,
    ticker TEXT NOT NULL,
    win_rate REAL,
    trade_count INTEGER NOT NULL,
    avg_return_pct REAL,
    expectancy REAL,
    qualified INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_discovery_results_run
    ON discovery_results (run_timestamp);

-- Phase 3.1-LIVE: audit trail of every live-shadow promotion evaluation. One
-- row per evaluated candidate per run: how many resolved shadow trades it had
-- in the recent window, its win rate + expectancy, whether it met the bar
-- (eligible), and whether it was actually promoted this run (False on dry-run
-- or when already on the watchlist).
CREATE TABLE IF NOT EXISTS shadow_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp TEXT NOT NULL,
    ticker TEXT NOT NULL,
    closed_count INTEGER NOT NULL,
    win_rate REAL,
    expectancy REAL,
    eligible INTEGER NOT NULL,
    promoted INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_shadow_evaluations_run
    ON shadow_evaluations (run_timestamp);

-- Phase 3.3: audit trail of every active<->benched transition the state
-- evaluator makes. One row per actual status change, with the windowed stats
-- and the reason that drove it.
CREATE TABLE IF NOT EXISTS watchlist_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    from_status TEXT NOT NULL,
    to_status TEXT NOT NULL,
    evaluated_at TEXT NOT NULL,
    win_rate REAL,
    closed_count INTEGER NOT NULL,
    expectancy REAL,
    reason TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_watchlist_transitions_ticker
    ON watchlist_transitions (ticker);

-- Phase 4: per-(ticker, signal_type) gate. A pair with NO row is treated as
-- ENABLED (default-enabled — innocent until proven losing); a row only exists
-- once the evaluator has muted/enabled it. status_changed_at is NULL until the
-- first transition.
CREATE TABLE IF NOT EXISTS signal_pair_status (
    ticker TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'enabled'
        CHECK (status IN ('enabled','muted')),
    status_changed_at TEXT,
    PRIMARY KEY (ticker, signal_type)
);

-- Phase 4: audit trail of every enabled<->muted transition the per-pair
-- evaluator makes. One row per actual status change.
CREATE TABLE IF NOT EXISTS signal_pair_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    from_status TEXT NOT NULL,
    to_status TEXT NOT NULL,
    evaluated_at TEXT NOT NULL,
    win_rate REAL,
    closed_count INTEGER NOT NULL,
    expectancy REAL,
    reason TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_signal_pair_transitions_pair
    ON signal_pair_transitions (ticker, signal_type);

-- Phase 9: history of self-optimization runs. Each row is one operator-triggered
-- run; findings_json holds the full degradation + feature-evaluation payload so
-- the report can replay a past run verbatim. FLAGS ONLY — nothing acts on these.
CREATE TABLE IF NOT EXISTS optimization_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp TEXT NOT NULL,
    degrade_window_days INTEGER NOT NULL,
    baseline_window_days INTEGER NOT NULL,
    findings_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_optimization_runs_ts
    ON optimization_runs (run_timestamp);

-- Phase 10: the unified readiness ledger. One row per capability; 'announced'
-- makes the first-crossing notification a ONE-TIME edge event (set True only on
-- a successful send, so a failed send retries next cycle).
CREATE TABLE IF NOT EXISTS readiness_state (
    capability TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'warming' CHECK (status IN ('warming','ready')),
    n_at_crossing INTEGER,
    crossed_at TEXT,
    announced INTEGER NOT NULL DEFAULT 0
);

-- Phase 13: single-leg option positions. Distinct from trades (equity): carries
-- the OCC contract, entry greeks (advisory), the underlying TP/SL levels the
-- manual exit watcher uses (no broker bracket exists for options), the hold
-- deadline, and a contract-aware realized pnl_dollars (premium * multiplier).
CREATE TABLE IF NOT EXISTS option_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER,
    order_id TEXT,
    symbol TEXT NOT NULL,
    underlying TEXT NOT NULL,
    option_type TEXT NOT NULL CHECK (option_type IN ('call','put')),
    strike REAL NOT NULL,
    expiry TEXT NOT NULL,
    contracts REAL NOT NULL,
    multiplier INTEGER NOT NULL DEFAULT 100,
    premium_entry REAL,
    delta_entry REAL,
    theta REAL,
    vega REAL,
    gamma REAL,
    tp REAL,
    sl REAL,
    deadline TEXT,
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    exit_price REAL,
    outcome TEXT CHECK (outcome IN ('win','loss','breakeven','open','expired')),
    pnl_dollars REAL,
    vehicle TEXT NOT NULL DEFAULT 'option_full'
);

CREATE INDEX IF NOT EXISTS idx_option_positions_open
    ON option_positions(outcome);

-- Phase 14: long-term buy-and-hold positions (fractional shares, no tight stop).
-- The protective exit watcher closes these on trend-breakdown or drawdown-stop.
CREATE TABLE IF NOT EXISTS long_term_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    entry_price REAL NOT NULL,
    entry_date TEXT NOT NULL,
    qty REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open','closed')),
    exit_price REAL,
    exit_date TEXT,
    exit_reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_long_term_positions_status
    ON long_term_positions(status);

-- Phase 15: account-equity snapshots, one per scan cycle. Drawdown is measured
-- peak-to-trough on EQUITY (so unrealized losses on open positions count), not
-- on closed-trade PnL alone.
CREATE TABLE IF NOT EXISTS equity_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_at TEXT NOT NULL,
    equity REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_equity_snapshots_at
    ON equity_snapshots(captured_at);

-- Phase 16: plan-execution audit trail. One row per planned order the execute
-- command acted on (submitted / rejected / error / skipped), linking the plan
-- run (plan_id) to the broker order (order_ref). The idempotency guard reads
-- this table: a 'submitted' row for the same (ticker, pool) within the current
-- cycle blocks a re-submission, so running execute twice never double-submits.
CREATE TABLE IF NOT EXISTS plan_executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id TEXT NOT NULL,
    executed_at TEXT NOT NULL,
    ticker TEXT NOT NULL,
    pool TEXT NOT NULL,
    signal_type TEXT,
    side TEXT,
    qty REAL,
    vehicle TEXT,
    status TEXT NOT NULL CHECK (status IN ('submitted','rejected','error','skipped')),
    order_ref TEXT,
    reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_plan_executions_ticker_pool
    ON plan_executions(ticker, pool, executed_at);

-- Cross-cutting order-lifecycle audit: every submission intent is durable
-- before network I/O. Queryable common intent is normalized; vehicle-specific
-- intent remains an immutable, versioned JSON object.
CREATE TABLE IF NOT EXISTS pending_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_order_id TEXT NOT NULL UNIQUE CHECK (TRIM(client_order_id) <> ''),
    broker_order_id TEXT UNIQUE CHECK (
        broker_order_id IS NULL OR TRIM(broker_order_id) <> ''
    ),
    order_role TEXT NOT NULL DEFAULT 'entry'
        CHECK (order_role IN ('entry','exit')),
    ticker TEXT NOT NULL,
    broker_symbol TEXT NOT NULL,
    asset_class TEXT NOT NULL CHECK (asset_class IN ('stock','crypto')),
    vehicle TEXT NOT NULL
        CHECK (vehicle IN ('option_full','option_undersized','shares')),
    target_position_kind TEXT NOT NULL
        CHECK (target_position_kind IN ('option','long_term')),
    closes_position_kind TEXT CHECK (
        closes_position_kind IN ('option','long_term')
    ),
    closes_position_id INTEGER,
    side TEXT NOT NULL CHECK (side IN ('buy','sell')),
    requested_qty REAL NOT NULL CHECK (requested_qty > 0),
    requested_limit_price REAL CHECK (
        requested_limit_price IS NULL OR requested_limit_price > 0
    ),
    submitted_at TEXT NOT NULL,
    signal_id INTEGER,
    intent_payload_version INTEGER NOT NULL DEFAULT 1
        CHECK (intent_payload_version >= 1),
    intent_payload_json TEXT NOT NULL,
    lifecycle_status TEXT NOT NULL DEFAULT 'new' CHECK (
        lifecycle_status IN (
            'prepared','new','partially_filled','filled','canceled','rejected',
            'expired','unknown','abandoned'
        )
    ),
    broker_status TEXT,
    filled_qty REAL NOT NULL DEFAULT 0 CHECK (filled_qty >= 0),
    filled_avg_price REAL,
    last_fill_at TEXT,
    last_fill_time_source TEXT CHECK (
        last_fill_time_source IN ('broker','observed')
    ),
    fees_dollars REAL CHECK (fees_dollars IS NULL OR fees_dollars >= 0),
    last_refreshed_at TEXT,
    terminal_reason TEXT,
    terminal_at TEXT,
    position_kind TEXT CHECK (position_kind IN ('option','long_term')),
    position_id INTEGER,
    CHECK (
        (order_role = 'entry'
         AND closes_position_kind IS NULL AND closes_position_id IS NULL)
        OR (
            order_role = 'exit'
            AND closes_position_kind = target_position_kind
            AND closes_position_id > 0
        )
    ),
    CHECK (
        (target_position_kind = 'option'
         AND vehicle IN ('option_full','option_undersized')
         AND (
             (order_role = 'entry' AND side = 'buy')
             OR (order_role = 'exit' AND side = 'sell')
         ))
        OR (target_position_kind = 'long_term' AND vehicle = 'shares')
    ),
    CHECK (
        (filled_qty = 0 AND filled_avg_price IS NULL
         AND last_fill_at IS NULL AND last_fill_time_source IS NULL
         AND fees_dollars IS NULL)
        OR (filled_qty > 0 AND filled_avg_price > 0
            AND (
                (last_fill_at IS NULL AND last_fill_time_source IS NULL)
                OR (
                    TRIM(COALESCE(last_fill_at, '')) <> ''
                    AND last_fill_time_source IN ('broker','observed')
                )
            ))
    ),
    CHECK (
        (lifecycle_status IN ('filled','canceled','rejected','expired','abandoned')
         AND terminal_at IS NOT NULL
         AND TRIM(COALESCE(terminal_reason, '')) <> '')
        OR (lifecycle_status IN ('prepared','new','partially_filled','unknown')
            AND terminal_at IS NULL AND terminal_reason IS NULL)
    ),
    CHECK (
        (lifecycle_status IN ('prepared','abandoned') AND broker_order_id IS NULL)
        OR (
            lifecycle_status NOT IN ('prepared','abandoned')
            AND broker_order_id IS NOT NULL
        )
    ),
    CHECK (
        lifecycle_status <> 'prepared'
        OR (
            broker_status IS NULL AND filled_qty = 0
            AND filled_avg_price IS NULL AND last_refreshed_at IS NULL
            AND position_kind IS NULL AND position_id IS NULL
        )
    ),
    CHECK (
        lifecycle_status <> 'abandoned'
        OR (
            broker_status IS NULL AND filled_qty = 0
            AND filled_avg_price IS NULL
            AND position_kind IS NULL AND position_id IS NULL
        )
    ),
    CHECK (
        (position_kind IS NULL AND position_id IS NULL)
        OR (position_kind IS NOT NULL AND position_id > 0)
    ),
    CHECK (
        order_role = 'entry'
        OR (position_kind IS NULL AND position_id IS NULL)
    ),
    CHECK (position_kind IS NULL OR position_kind = target_position_kind),
    CHECK (
        position_id IS NULL OR (filled_qty > 0 AND filled_avg_price > 0)
    )
);

CREATE INDEX IF NOT EXISTS idx_pending_orders_nonterminal
    ON pending_orders(submitted_at) WHERE terminal_at IS NULL;

-- Materialization intent is write-once even for SQL callers that bypass the
-- whitelisted Python update helper.
CREATE TRIGGER IF NOT EXISTS trg_pending_orders_immutable_intent
BEFORE UPDATE OF
    client_order_id, order_role, ticker, broker_symbol, asset_class, vehicle,
    target_position_kind, closes_position_kind, closes_position_id, side,
    requested_qty, requested_limit_price, submitted_at, signal_id,
    intent_payload_version, intent_payload_json
ON pending_orders
BEGIN
    SELECT RAISE(ABORT, 'pending order materialization intent is immutable');
END;

-- A prepared row may bind its broker identity exactly once.  Rebinding or
-- removing an already-bound ID is forbidden even for direct SQL callers.
CREATE TRIGGER IF NOT EXISTS trg_pending_orders_broker_id_bind_once
BEFORE UPDATE OF broker_order_id ON pending_orders
WHEN NOT (
    OLD.broker_order_id IS NULL
    AND NEW.broker_order_id IS NOT NULL
)
BEGIN
    SELECT RAISE(ABORT, 'pending order broker_order_id may only bind once');
END;

-- Broker fill snapshots are cumulative.  An older snapshot may be ignored by
-- the future lifecycle service, but it can never regress durable truth.
CREATE TRIGGER IF NOT EXISTS trg_pending_orders_filled_qty_monotonic
BEFORE UPDATE OF filled_qty ON pending_orders
WHEN NEW.filled_qty < OLD.filled_qty
BEGIN
    SELECT RAISE(ABORT, 'pending order filled_qty cannot decrease');
END;

-- Once a pending order points at its materialized position, every later fill
-- must update that same row rather than relinking or inserting another one.
CREATE TRIGGER IF NOT EXISTS trg_pending_orders_position_link_once
BEFORE UPDATE OF position_kind, position_id ON pending_orders
WHEN OLD.position_id IS NOT NULL AND (
    NEW.position_id IS NOT OLD.position_id
    OR NEW.position_kind IS NOT OLD.position_kind
)
BEGIN
    SELECT RAISE(ABORT, 'pending order position link is immutable once set');
END;
"""

# Whitelist for update_trade — never interpolate user input into SQL column names.
# market_regime was added in Phase 2.1 + vix_level/vix_band in Phase 2.2 + the
# Phase 2.3 composite context_score, all so their respective backfills can
# patch historical rows.
_TRADE_UPDATABLE_FIELDS: frozenset[str] = frozenset(
    {
        "opened_at", "closed_at", "exit_price", "outcome",
        "pnl_pct", "pnl_dollars", "notes", "market_regime",
        "vix_level", "vix_band", "context_score",
    }
)

_TABLE_NAMES: tuple[str, ...] = (
    "signals", "trades", "daily_performance",
    "regime_snapshots", "vix_snapshots",
    "predictions", "settings", "active_watchlist", "discovery_results",
    "shadow_evaluations", "watchlist_transitions",
    "signal_pair_status", "signal_pair_transitions", "optimization_runs",
    "readiness_state", "option_positions", "long_term_positions",
    "equity_snapshots", "plan_executions", "pending_orders",
)

_OPTION_POSITION_UPDATABLE_FIELDS: frozenset[str] = frozenset(
    {
        "order_id", "closed_at", "exit_price", "exit_reason", "outcome",
        "pnl_dollars",
    }
)

_LONG_TERM_UPDATABLE_FIELDS: frozenset[str] = frozenset(
    {"status", "exit_price", "exit_date", "exit_reason"}
)

_PENDING_ORDER_UPDATABLE_FIELDS: frozenset[str] = frozenset(
    {
        "broker_order_id", "lifecycle_status", "broker_status", "filled_qty",
        "filled_avg_price", "last_fill_at", "last_fill_time_source",
        "fees_dollars", "last_refreshed_at", "terminal_reason", "terminal_at",
        "position_kind", "position_id",
    }
)

# Whitelist for update_prediction — resolution flow + Phase 2.3 context backfill.
_PREDICTION_UPDATABLE_FIELDS: frozenset[str] = frozenset(
    {"resolved_at", "exit_price", "outcome", "notified", "context_score"}
)


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    """Return a connection with FK enforcement and WAL journal mode."""
    _ensure_parent_dir(config.DB_PATH)
    conn = sqlite3.connect(
        str(config.DB_PATH),
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _migrate_to_v2(conn: sqlite3.Connection) -> None:
    """Phase 2.1 migration step: add ``trades.market_regime`` if absent.

    ``CREATE TABLE IF NOT EXISTS`` covers the new ``regime_snapshots`` table,
    but column additions on existing tables need an explicit ALTER. Safe to
    run on a fresh schema (just becomes a no-op).
    """
    cur = conn.execute("PRAGMA table_info(trades)")
    cols = {str(row[1]) for row in cur.fetchall()}
    if "market_regime" not in cols:
        conn.execute("ALTER TABLE trades ADD COLUMN market_regime TEXT")


def _migrate_to_v3(conn: sqlite3.Connection) -> None:
    """Phase 2.2 migration step: add ``trades.vix_level`` + ``trades.vix_band``.

    Both nullable — existing rows (including the regime-backfilled ones from
    2.1) stay NULL until ``vix backfill`` runs. ``vix_snapshots`` itself is
    created by the ``CREATE TABLE IF NOT EXISTS`` block above.
    """
    cur = conn.execute("PRAGMA table_info(trades)")
    cols = {str(row[1]) for row in cur.fetchall()}
    if "vix_level" not in cols:
        conn.execute("ALTER TABLE trades ADD COLUMN vix_level REAL")
    if "vix_band" not in cols:
        conn.execute("ALTER TABLE trades ADD COLUMN vix_band TEXT")


def _migrate_to_v4(_conn: sqlite3.Connection) -> None:
    """Phase 2.2b migration step.

    Nothing to ALTER — both new tables (``predictions``, ``settings``) are
    created by the ``CREATE TABLE IF NOT EXISTS`` block above. This step
    exists as the documented hook for future column adds without
    re-numbering the existing migration sequence.
    """
    return None


def _migrate_to_v5(conn: sqlite3.Connection) -> None:
    """Phase 2.3 migration step: add ``context_score`` to trades + predictions.

    Composite (regime x VIX) score in 0-5, NULL until backfilled. Cheap
    integer column on both tables — no indexes needed (filtering is by
    range, low cardinality).
    """
    cur = conn.execute("PRAGMA table_info(trades)")
    cols = {str(row[1]) for row in cur.fetchall()}
    if "context_score" not in cols:
        conn.execute("ALTER TABLE trades ADD COLUMN context_score INTEGER")

    cur = conn.execute("PRAGMA table_info(predictions)")
    pred_cols = {str(row[1]) for row in cur.fetchall()}
    if "context_score" not in pred_cols:
        conn.execute("ALTER TABLE predictions ADD COLUMN context_score INTEGER")


def _migrate_to_v6(_conn: sqlite3.Connection) -> None:
    """Phase 3.1 migration step.

    Nothing to ALTER — the new ``active_watchlist`` table is created by the
    ``CREATE TABLE IF NOT EXISTS`` block above. This step exists as the
    documented hook for future column adds without re-numbering the sequence.
    """
    return None


def _migrate_to_v7(_conn: sqlite3.Connection) -> None:
    """Phase 3.1 migration step.

    Nothing to ALTER — the new ``discovery_results`` table is created by the
    ``CREATE TABLE IF NOT EXISTS`` block above. Documented hook only.
    """
    return None


def _migrate_to_v8(conn: sqlite3.Connection) -> None:
    """Phase 3.1-LIVE migration step: add ``trades.track_mode``.

    ``NOT NULL DEFAULT 'active'`` backfills every existing row to 'active'
    in one shot — there is no pre-3.1-LIVE shadow trade, so 'active' is the
    correct historical value. The CHECK keeps the column to the two valid
    modes. Safe on a fresh schema (becomes a no-op once the column exists).
    """
    cur = conn.execute("PRAGMA table_info(trades)")
    cols = {str(row[1]) for row in cur.fetchall()}
    if "track_mode" not in cols:
        conn.execute(
            "ALTER TABLE trades ADD COLUMN track_mode TEXT NOT NULL "
            "DEFAULT 'active' CHECK (track_mode IN ('active','shadow'))"
        )


def _migrate_to_v9(_conn: sqlite3.Connection) -> None:
    """Phase 3.1-LIVE migration step.

    Nothing to ALTER — the new ``shadow_evaluations`` table is created by the
    ``CREATE TABLE IF NOT EXISTS`` block above. Documented hook only.
    """
    return None


def _migrate_to_v10(conn: sqlite3.Connection) -> None:
    """Phase 3.3 migration step: add active_watchlist.status + status_changed_at.

    ``status NOT NULL DEFAULT 'active'`` backfills every existing watchlist row
    to 'active' (there is no pre-3.3 benched ticker, so 'active' is correct).
    ``status_changed_at`` is nullable — NULL until the state evaluator flips a
    ticker. The CHECK keeps status to the two valid states. The new
    ``watchlist_transitions`` table is created by the schema block above.
    """
    cur = conn.execute("PRAGMA table_info(active_watchlist)")
    cols = {str(row[1]) for row in cur.fetchall()}
    if "status" not in cols:
        conn.execute(
            "ALTER TABLE active_watchlist ADD COLUMN status TEXT NOT NULL "
            "DEFAULT 'active' CHECK (status IN ('active','benched'))"
        )
    if "status_changed_at" not in cols:
        conn.execute(
            "ALTER TABLE active_watchlist ADD COLUMN status_changed_at TEXT"
        )


def _migrate_to_v11(_conn: sqlite3.Connection) -> None:
    """Phase 4 migration step.

    Nothing to ALTER — the new ``signal_pair_status`` and
    ``signal_pair_transitions`` tables are created by the
    ``CREATE TABLE IF NOT EXISTS`` block above. Documented hook only.
    """
    return None


def _migrate_to_v12(conn: sqlite3.Connection) -> None:
    """Phase 5 migration step: add advisory sentiment columns to trades.

    All nullable (``heavy_news`` defaults 0): a trade opened before Phase 5,
    or any shadow/crypto trade that isn't sentiment-scored, simply leaves them
    NULL/0. Sentiment sits next to the eventual resolved outcome for later
    evaluation.
    """
    cur = conn.execute("PRAGMA table_info(trades)")
    cols = {str(row[1]) for row in cur.fetchall()}
    if "sentiment_score" not in cols:
        conn.execute("ALTER TABLE trades ADD COLUMN sentiment_score REAL")
    if "sentiment_label" not in cols:
        conn.execute("ALTER TABLE trades ADD COLUMN sentiment_label TEXT")
    if "heavy_news" not in cols:
        conn.execute(
            "ALTER TABLE trades ADD COLUMN heavy_news INTEGER NOT NULL DEFAULT 0"
        )
    if "headline_count" not in cols:
        conn.execute("ALTER TABLE trades ADD COLUMN headline_count INTEGER")


def _migrate_to_v13(conn: sqlite3.Connection) -> None:
    """Phase 6 migration step: add advisory indicator-family columns to trades.

    All nullable: a trade opened before Phase 6, or any shadow/crypto trade that
    isn't indicator-scored, leaves them NULL. The five families sit next to the
    eventual resolved outcome for later (Phase 9) evaluation — they are advisory
    metadata, never a gate. Mirrors the Phase 5 sentiment-column add (v12).
    """
    cur = conn.execute("PRAGMA table_info(trades)")
    cols = {str(row[1]) for row in cur.fetchall()}
    for column, coltype in (
        ("ind_atr", "REAL"),
        ("ind_realized_vol", "REAL"),
        ("ind_vol_regime", "TEXT"),
        ("ind_rsi", "REAL"),
        ("ind_adx", "REAL"),
        ("ind_obv", "REAL"),
        ("ind_correlation", "REAL"),
        ("ind_concentration", "TEXT"),
    ):
        if column not in cols:
            conn.execute(f"ALTER TABLE trades ADD COLUMN {column} {coltype}")


def _migrate_to_v14(conn: sqlite3.Connection) -> None:
    """Phase 7 migration step: add advisory risk-recommendation columns to trades.

    All nullable (``risk_capped`` defaults 0): a trade opened before Phase 7, or
    any shadow/crypto trade that isn't risk-assessed, leaves them NULL/0. The
    sizing + portfolio verdicts sit next to the eventual resolved outcome for
    later (Phase 9) evaluation — advisory metadata, never a gate. Mirrors the
    Phase 5/6 column adds (v12/v13).
    """
    cur = conn.execute("PRAGMA table_info(trades)")
    cols = {str(row[1]) for row in cur.fetchall()}
    for column, coltype in (
        ("risk_recommended_size", "REAL"),
        ("risk_stop_distance", "REAL"),
        ("risk_dollar_risk", "REAL"),
        ("risk_pct", "REAL"),
        ("risk_position_pct", "REAL"),
        ("risk_capped", "INTEGER NOT NULL DEFAULT 0"),
        ("risk_total_pct", "REAL"),
        ("risk_portfolio_verdict", "TEXT"),
        ("risk_position_verdict", "TEXT"),
        ("risk_cluster_pct", "REAL"),
        ("risk_cluster_verdict", "TEXT"),
    ):
        if column not in cols:
            conn.execute(f"ALTER TABLE trades ADD COLUMN {column} {coltype}")


def _migrate_to_v15(_conn: sqlite3.Connection) -> None:
    """Phase 9 migration step.

    Nothing to ALTER — the new ``optimization_runs`` table is created by the
    ``CREATE TABLE IF NOT EXISTS`` block above. Documented hook only.
    """
    return None


def _migrate_to_v16(_conn: sqlite3.Connection) -> None:
    """Phase 10 migration step.

    Nothing to ALTER — the new ``readiness_state`` table is created by the
    ``CREATE TABLE IF NOT EXISTS`` block above. Documented hook only.
    """
    return None


def _migrate_to_v17(_conn: sqlite3.Connection) -> None:
    """Phase 13 migration step.

    Nothing to ALTER — the new ``option_positions`` table is created by the
    ``CREATE TABLE IF NOT EXISTS`` block above. Documented hook only.
    """
    return None


def _migrate_to_v18(_conn: sqlite3.Connection) -> None:
    """Phase 14 migration step.

    Nothing to ALTER — the new ``long_term_positions`` table is created by the
    ``CREATE TABLE IF NOT EXISTS`` block above. Documented hook only.
    """
    return None


def _migrate_to_v19(_conn: sqlite3.Connection) -> None:
    """Phase 15 migration step.

    Nothing to ALTER — the new ``equity_snapshots`` table is created by the
    ``CREATE TABLE IF NOT EXISTS`` block above. Documented hook only.
    """
    return None


def _migrate_to_v20(_conn: sqlite3.Connection) -> None:
    """Phase 16 migration step.

    Nothing to ALTER — the new ``plan_executions`` table is created by the
    ``CREATE TABLE IF NOT EXISTS`` block above. Documented hook only.
    """
    return None


def _migrate_to_v21(conn: sqlite3.Connection) -> None:
    """Phase 17 migration step: ``signals.considered_at``.

    NULL until the signal is pulled into an execute-bound plan by the live
    candidate source — set at PULL time (not execution time) so a signal is
    considered exactly once, regardless of what the plan later did with it.
    Existing rows stay NULL; the live query's recency window keeps historical
    signals from flooding the first post-migration plan.
    """
    cur = conn.execute("PRAGMA table_info(signals)")
    cols = {str(row[1]) for row in cur.fetchall()}
    if "considered_at" not in cols:
        conn.execute("ALTER TABLE signals ADD COLUMN considered_at TEXT")


def _migrate_to_v22(conn: sqlite3.Connection) -> None:
    """Phase 19 migration step: shares-fallback tracking on long_term_positions.

    ``source`` distinguishes a genuine buy-and-hold entry ('long_term', the
    default — existing rows backfill to it) from a Phase 13 shares-fallback
    position ('swing_fallback'), which carries SWING intent and must keep its
    ORIGINAL exit rules. ``tp``/``sl``/``deadline`` persist those original
    swing exit levels (NULL on genuine long-term rows); ``direction`` records
    the position side ('long' default; a put-signal fallback shorts, so its
    close must BUY). SQLite has no ALTER ADD CHECK — the valid sets are
    enforced by ``insert_long_term_position``.
    """
    cur = conn.execute("PRAGMA table_info(long_term_positions)")
    cols = {str(row[1]) for row in cur.fetchall()}
    for column, coltype in (
        ("source", "TEXT NOT NULL DEFAULT 'long_term'"),
        ("direction", "TEXT NOT NULL DEFAULT 'long'"),
        ("tp", "REAL"),
        ("sl", "REAL"),
        ("deadline", "TEXT"),
    ):
        if column not in cols:
            conn.execute(
                f"ALTER TABLE long_term_positions ADD COLUMN {column} {coltype}"
            )


def _migrate_to_v23(_conn: sqlite3.Connection) -> None:
    """Phase 24: earnings/news risk columns now store full text grades.

    Existing columns have SQLite INTEGER affinity, which still accepts TEXT;
    changing affinity would require rebuilding the referenced ``signals``
    table. Deliberately do not rebuild or backfill it: legacy 0/1 rows remain
    byte-for-byte unchanged and are normalized on read. Fresh databases use
    the TEXT declaration in ``_SCHEMA_SQL``.
    """
    return None


def _migrate_to_v24(_conn: sqlite3.Connection) -> None:
    """Cross-cutting audit: add the durable ``pending_orders`` ledger.

    Nothing to ALTER.  The table, indexes, constraints, and immutability
    triggers are created by the idempotent ``_SCHEMA_SQL`` block before this
    documented migration hook runs.  Existing tables and rows are untouched.
    """
    return None


def _migrate_to_v25(conn: sqlite3.Connection) -> None:
    """Add pre-submit identity while preserving every v24 ledger row.

    SQLite cannot add the required UNIQUE/NULLability/CHECK contract with
    ``ALTER COLUMN``, so the dormant v24 ledger is rebuilt transactionally.
    Bound legacy rows receive deterministic synthetic client IDs.  Those IDs
    are audit keys only and are explicitly recovery-ineligible in ``models``.
    """
    columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(pending_orders)")
    }
    if "client_order_id" in columns:
        return

    conn.execute("DROP INDEX IF EXISTS idx_pending_orders_nonterminal")
    for trigger in (
        "trg_pending_orders_immutable_intent",
        "trg_pending_orders_filled_qty_monotonic",
        "trg_pending_orders_position_link_once",
        "trg_pending_orders_broker_id_bind_once",
    ):
        conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")

    conn.execute(
        """
        CREATE TABLE pending_orders_v25 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_order_id TEXT NOT NULL UNIQUE
                CHECK (TRIM(client_order_id) <> ''),
            broker_order_id TEXT UNIQUE CHECK (
                broker_order_id IS NULL OR TRIM(broker_order_id) <> ''
            ),
            ticker TEXT NOT NULL,
            broker_symbol TEXT NOT NULL,
            asset_class TEXT NOT NULL CHECK (asset_class IN ('stock','crypto')),
            vehicle TEXT NOT NULL
                CHECK (vehicle IN ('option_full','option_undersized','shares')),
            target_position_kind TEXT NOT NULL
                CHECK (target_position_kind IN ('option','long_term')),
            side TEXT NOT NULL CHECK (side IN ('buy','sell')),
            requested_qty REAL NOT NULL CHECK (requested_qty > 0),
            requested_limit_price REAL CHECK (
                requested_limit_price IS NULL OR requested_limit_price > 0
            ),
            submitted_at TEXT NOT NULL,
            signal_id INTEGER,
            intent_payload_version INTEGER NOT NULL DEFAULT 1
                CHECK (intent_payload_version >= 1),
            intent_payload_json TEXT NOT NULL,
            lifecycle_status TEXT NOT NULL DEFAULT 'new' CHECK (
                lifecycle_status IN (
                    'prepared','new','partially_filled','filled','canceled',
                    'rejected','expired','unknown','abandoned'
                )
            ),
            broker_status TEXT,
            filled_qty REAL NOT NULL DEFAULT 0 CHECK (filled_qty >= 0),
            filled_avg_price REAL,
            last_refreshed_at TEXT,
            terminal_reason TEXT,
            terminal_at TEXT,
            position_kind TEXT CHECK (position_kind IN ('option','long_term')),
            position_id INTEGER,
            CHECK (
                (target_position_kind = 'option'
                 AND vehicle IN ('option_full','option_undersized')
                 AND side = 'buy')
                OR (target_position_kind = 'long_term' AND vehicle = 'shares')
            ),
            CHECK (
                (filled_qty = 0 AND filled_avg_price IS NULL)
                OR (filled_qty > 0 AND filled_avg_price > 0)
            ),
            CHECK (
                (lifecycle_status IN (
                    'filled','canceled','rejected','expired','abandoned'
                 )
                 AND terminal_at IS NOT NULL
                 AND TRIM(COALESCE(terminal_reason, '')) <> '')
                OR (lifecycle_status IN (
                    'prepared','new','partially_filled','unknown'
                ) AND terminal_at IS NULL AND terminal_reason IS NULL)
            ),
            CHECK (
                (lifecycle_status IN ('prepared','abandoned')
                 AND broker_order_id IS NULL)
                OR (
                    lifecycle_status NOT IN ('prepared','abandoned')
                    AND broker_order_id IS NOT NULL
                )
            ),
            CHECK (
                lifecycle_status <> 'prepared'
                OR (
                    broker_status IS NULL AND filled_qty = 0
                    AND filled_avg_price IS NULL AND last_refreshed_at IS NULL
                    AND position_kind IS NULL AND position_id IS NULL
                )
            ),
            CHECK (
                lifecycle_status <> 'abandoned'
                OR (
                    broker_status IS NULL AND filled_qty = 0
                    AND filled_avg_price IS NULL
                    AND position_kind IS NULL AND position_id IS NULL
                )
            ),
            CHECK (
                (position_kind IS NULL AND position_id IS NULL)
                OR (position_kind IS NOT NULL AND position_id > 0)
            ),
            CHECK (position_kind IS NULL OR position_kind = target_position_kind),
            CHECK (
                position_id IS NULL OR (filled_qty > 0 AND filled_avg_price > 0)
            )
        )
        """
    )
    conn.execute(
        """
        INSERT INTO pending_orders_v25 (
            id, client_order_id, broker_order_id, ticker, broker_symbol,
            asset_class, vehicle, target_position_kind, side, requested_qty,
            requested_limit_price, submitted_at, signal_id,
            intent_payload_version, intent_payload_json, lifecycle_status,
            broker_status, filled_qty, filled_avg_price, last_refreshed_at,
            terminal_reason, terminal_at, position_kind, position_id
        )
        SELECT
            id, ? || id, broker_order_id, ticker, broker_symbol, asset_class,
            vehicle, target_position_kind, side, requested_qty,
            requested_limit_price, submitted_at, signal_id,
            intent_payload_version, intent_payload_json, lifecycle_status,
            broker_status, filled_qty, filled_avg_price, last_refreshed_at,
            terminal_reason, terminal_at, position_kind, position_id
        FROM pending_orders
        """,
        (f"{LEGACY_PENDING_ORDER_CLIENT_ID_PREFIX}v24-",),
    )
    conn.execute("DROP TABLE pending_orders")
    conn.execute("ALTER TABLE pending_orders_v25 RENAME TO pending_orders")
    conn.execute(
        "CREATE INDEX idx_pending_orders_nonterminal "
        "ON pending_orders(submitted_at) WHERE terminal_at IS NULL"
    )
    conn.execute(
        """
        CREATE TRIGGER trg_pending_orders_immutable_intent
        BEFORE UPDATE OF
            client_order_id, ticker, broker_symbol, asset_class, vehicle,
            target_position_kind, side, requested_qty, requested_limit_price,
            submitted_at, signal_id, intent_payload_version, intent_payload_json
        ON pending_orders
        BEGIN
            SELECT RAISE(
                ABORT, 'pending order materialization intent is immutable'
            );
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER trg_pending_orders_broker_id_bind_once
        BEFORE UPDATE OF broker_order_id ON pending_orders
        WHEN NOT (
            OLD.broker_order_id IS NULL
            AND NEW.broker_order_id IS NOT NULL
        )
        BEGIN
            SELECT RAISE(
                ABORT, 'pending order broker_order_id may only bind once'
            );
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER trg_pending_orders_filled_qty_monotonic
        BEFORE UPDATE OF filled_qty ON pending_orders
        WHEN NEW.filled_qty < OLD.filled_qty
        BEGIN
            SELECT RAISE(ABORT, 'pending order filled_qty cannot decrease');
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER trg_pending_orders_position_link_once
        BEFORE UPDATE OF position_kind, position_id ON pending_orders
        WHEN OLD.position_id IS NOT NULL AND (
            NEW.position_id IS NOT OLD.position_id
            OR NEW.position_kind IS NOT OLD.position_kind
        )
        BEGIN
            SELECT RAISE(
                ABORT, 'pending order position link is immutable once set'
            );
        END
        """
    )


def _migrate_to_v26(conn: sqlite3.Connection) -> None:
    """Persist advisory sentiment provenance without rewriting legacy rows.

    Both columns are nullable by design. Existing sentiment labels cannot tell
    us whether neutral was a real score or a fail-soft fallback, so migration
    leaves their provenance unknown rather than inventing success or failure.
    """
    columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(trades)")
    }
    if "sentiment_ok" not in columns:
        conn.execute(
            "ALTER TABLE trades ADD COLUMN sentiment_ok INTEGER "
            "CHECK (sentiment_ok IN (0,1))"
        )
    if "sentiment_rationale" not in columns:
        conn.execute("ALTER TABLE trades ADD COLUMN sentiment_rationale TEXT")


def _migrate_to_v27(conn: sqlite3.Connection) -> None:
    """Persist indicator success without inventing provenance for legacy rows."""
    columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(trades)")
    }
    if "ind_ok" not in columns:
        conn.execute(
            "ALTER TABLE trades ADD COLUMN ind_ok INTEGER "
            "CHECK (ind_ok IN (0,1))"
        )


def _migrate_to_v28(conn: sqlite3.Connection) -> None:
    """Persist risk provenance without rewriting legacy assessment rows."""
    columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(trades)")
    }
    if "risk_ok" not in columns:
        conn.execute(
            "ALTER TABLE trades ADD COLUMN risk_ok INTEGER "
            "CHECK (risk_ok IN (0,1))"
        )
    if "risk_reason" not in columns:
        conn.execute("ALTER TABLE trades ADD COLUMN risk_reason TEXT")


def _create_v29_pending_order_objects(conn: sqlite3.Connection) -> None:
    """Create the v29 index/triggers after fresh creation or table rebuild."""
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_pending_orders_exit_target "
        "ON pending_orders(closes_position_kind, closes_position_id, "
        "submitted_at, id) WHERE order_role = 'exit'"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_pending_orders_nonterminal "
        "ON pending_orders(submitted_at) WHERE terminal_at IS NULL"
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_pending_orders_fill_time_on_insert
        AFTER INSERT ON pending_orders
        WHEN NEW.filled_qty > 0
             AND NEW.last_fill_at IS NULL
             AND NEW.last_fill_time_source IS NULL
        BEGIN
            UPDATE pending_orders
            SET last_fill_at = COALESCE(NEW.last_refreshed_at, NEW.submitted_at),
                last_fill_time_source = 'observed'
            WHERE id = NEW.id;
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_pending_orders_fill_time_on_update
        BEFORE UPDATE OF filled_qty, last_fill_at, last_fill_time_source
        ON pending_orders
        WHEN NEW.filled_qty > 0
             AND (
                 NEW.last_fill_at IS NULL
                 OR NEW.last_fill_time_source IS NULL
             )
        BEGIN
            SELECT RAISE(
                ABORT, 'positive pending order fill requires fill-time provenance'
            );
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_pending_orders_immutable_intent
        BEFORE UPDATE OF
            client_order_id, order_role, ticker, broker_symbol, asset_class,
            vehicle, target_position_kind, closes_position_kind,
            closes_position_id, side, requested_qty, requested_limit_price,
            submitted_at, signal_id, intent_payload_version, intent_payload_json
        ON pending_orders
        BEGIN
            SELECT RAISE(
                ABORT, 'pending order materialization intent is immutable'
            );
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_pending_orders_broker_id_bind_once
        BEFORE UPDATE OF broker_order_id ON pending_orders
        WHEN NOT (
            OLD.broker_order_id IS NULL
            AND NEW.broker_order_id IS NOT NULL
        )
        BEGIN
            SELECT RAISE(
                ABORT, 'pending order broker_order_id may only bind once'
            );
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_pending_orders_filled_qty_monotonic
        BEFORE UPDATE OF filled_qty ON pending_orders
        WHEN NEW.filled_qty < OLD.filled_qty
        BEGIN
            SELECT RAISE(ABORT, 'pending order filled_qty cannot decrease');
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_pending_orders_position_link_once
        BEFORE UPDATE OF position_kind, position_id ON pending_orders
        WHEN OLD.position_id IS NOT NULL AND (
            NEW.position_id IS NOT OLD.position_id
            OR NEW.position_kind IS NOT OLD.position_kind
        )
        BEGIN
            SELECT RAISE(
                ABORT, 'pending order position link is immutable once set'
            );
        END
        """
    )


def _migrate_to_v29(conn: sqlite3.Connection) -> None:
    """Add dormant exit-role lifecycle truth without rewriting legacy intent.

    SQLite cannot add the role-dependent close-target and fill-time CHECKs via
    ``ALTER TABLE``.  Rebuild the ledger transactionally, preserving every v28
    value and row id.  All historical rows are entry intent; positive fills
    receive explicitly observed provenance from the durable refresh time, or
    submission time when no refresh was ever persisted.  Broker time and fees
    remain unknown rather than being invented.
    """
    option_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(option_positions)")
    }
    if "exit_reason" not in option_columns:
        conn.execute("ALTER TABLE option_positions ADD COLUMN exit_reason TEXT")

    columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(pending_orders)")
    }
    v29_columns = {
        "order_role", "closes_position_kind", "closes_position_id",
        "last_fill_at", "last_fill_time_source", "fees_dollars",
    }
    if "order_role" in columns:
        if not v29_columns.issubset(columns):
            missing = sorted(v29_columns - columns)
            raise RuntimeError(
                f"partial pending_orders v29 schema; missing columns: {missing}"
            )
        _create_v29_pending_order_objects(conn)
        return

    conn.execute("DROP INDEX IF EXISTS idx_pending_orders_nonterminal")
    conn.execute("DROP INDEX IF EXISTS idx_pending_orders_exit_target")
    for trigger in (
        "trg_pending_orders_immutable_intent",
        "trg_pending_orders_broker_id_bind_once",
        "trg_pending_orders_filled_qty_monotonic",
        "trg_pending_orders_position_link_once",
        "trg_pending_orders_fill_time_on_insert",
        "trg_pending_orders_fill_time_on_update",
    ):
        conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")

    conn.execute(
        """
        CREATE TABLE pending_orders_v29 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_order_id TEXT NOT NULL UNIQUE
                CHECK (TRIM(client_order_id) <> ''),
            broker_order_id TEXT UNIQUE CHECK (
                broker_order_id IS NULL OR TRIM(broker_order_id) <> ''
            ),
            order_role TEXT NOT NULL DEFAULT 'entry'
                CHECK (order_role IN ('entry','exit')),
            ticker TEXT NOT NULL,
            broker_symbol TEXT NOT NULL,
            asset_class TEXT NOT NULL CHECK (asset_class IN ('stock','crypto')),
            vehicle TEXT NOT NULL CHECK (
                vehicle IN ('option_full','option_undersized','shares')
            ),
            target_position_kind TEXT NOT NULL CHECK (
                target_position_kind IN ('option','long_term')
            ),
            closes_position_kind TEXT CHECK (
                closes_position_kind IN ('option','long_term')
            ),
            closes_position_id INTEGER,
            side TEXT NOT NULL CHECK (side IN ('buy','sell')),
            requested_qty REAL NOT NULL CHECK (requested_qty > 0),
            requested_limit_price REAL CHECK (
                requested_limit_price IS NULL OR requested_limit_price > 0
            ),
            submitted_at TEXT NOT NULL,
            signal_id INTEGER,
            intent_payload_version INTEGER NOT NULL DEFAULT 1
                CHECK (intent_payload_version >= 1),
            intent_payload_json TEXT NOT NULL,
            lifecycle_status TEXT NOT NULL DEFAULT 'new' CHECK (
                lifecycle_status IN (
                    'prepared','new','partially_filled','filled','canceled',
                    'rejected','expired','unknown','abandoned'
                )
            ),
            broker_status TEXT,
            filled_qty REAL NOT NULL DEFAULT 0 CHECK (filled_qty >= 0),
            filled_avg_price REAL,
            last_fill_at TEXT,
            last_fill_time_source TEXT CHECK (
                last_fill_time_source IN ('broker','observed')
            ),
            fees_dollars REAL CHECK (
                fees_dollars IS NULL OR fees_dollars >= 0
            ),
            last_refreshed_at TEXT,
            terminal_reason TEXT,
            terminal_at TEXT,
            position_kind TEXT CHECK (position_kind IN ('option','long_term')),
            position_id INTEGER,
            CHECK (
                (order_role = 'entry'
                 AND closes_position_kind IS NULL
                 AND closes_position_id IS NULL)
                OR (
                    order_role = 'exit'
                    AND closes_position_kind = target_position_kind
                    AND closes_position_id > 0
                )
            ),
            CHECK (
                (target_position_kind = 'option'
                 AND vehicle IN ('option_full','option_undersized')
                 AND (
                     (order_role = 'entry' AND side = 'buy')
                     OR (order_role = 'exit' AND side = 'sell')
                 ))
                OR (
                    target_position_kind = 'long_term'
                    AND vehicle = 'shares'
                )
            ),
            CHECK (
                (filled_qty = 0 AND filled_avg_price IS NULL
                 AND last_fill_at IS NULL AND last_fill_time_source IS NULL
                 AND fees_dollars IS NULL)
                OR (
                    filled_qty > 0 AND filled_avg_price > 0
                    AND (
                        (last_fill_at IS NULL
                         AND last_fill_time_source IS NULL)
                        OR (
                            TRIM(COALESCE(last_fill_at, '')) <> ''
                            AND last_fill_time_source IN ('broker','observed')
                        )
                    )
                )
            ),
            CHECK (
                (lifecycle_status IN (
                    'filled','canceled','rejected','expired','abandoned'
                 )
                 AND terminal_at IS NOT NULL
                 AND TRIM(COALESCE(terminal_reason, '')) <> '')
                OR (
                    lifecycle_status IN (
                        'prepared','new','partially_filled','unknown'
                    )
                    AND terminal_at IS NULL AND terminal_reason IS NULL
                )
            ),
            CHECK (
                (lifecycle_status IN ('prepared','abandoned')
                 AND broker_order_id IS NULL)
                OR (
                    lifecycle_status NOT IN ('prepared','abandoned')
                    AND broker_order_id IS NOT NULL
                )
            ),
            CHECK (
                lifecycle_status <> 'prepared'
                OR (
                    broker_status IS NULL AND filled_qty = 0
                    AND filled_avg_price IS NULL AND last_refreshed_at IS NULL
                    AND position_kind IS NULL AND position_id IS NULL
                )
            ),
            CHECK (
                lifecycle_status <> 'abandoned'
                OR (
                    broker_status IS NULL AND filled_qty = 0
                    AND filled_avg_price IS NULL
                    AND position_kind IS NULL AND position_id IS NULL
                )
            ),
            CHECK (
                (position_kind IS NULL AND position_id IS NULL)
                OR (position_kind IS NOT NULL AND position_id > 0)
            ),
            CHECK (
                order_role = 'entry'
                OR (position_kind IS NULL AND position_id IS NULL)
            ),
            CHECK (
                position_kind IS NULL
                OR position_kind = target_position_kind
            ),
            CHECK (
                position_id IS NULL
                OR (filled_qty > 0 AND filled_avg_price > 0)
            )
        )
        """
    )
    conn.execute(
        """
        INSERT INTO pending_orders_v29 (
            id, client_order_id, broker_order_id, order_role, ticker,
            broker_symbol, asset_class, vehicle, target_position_kind,
            closes_position_kind, closes_position_id, side, requested_qty,
            requested_limit_price, submitted_at, signal_id,
            intent_payload_version, intent_payload_json, lifecycle_status,
            broker_status, filled_qty, filled_avg_price, last_fill_at,
            last_fill_time_source, fees_dollars, last_refreshed_at,
            terminal_reason, terminal_at, position_kind, position_id
        )
        SELECT
            id, client_order_id, broker_order_id, 'entry', ticker,
            broker_symbol, asset_class, vehicle, target_position_kind,
            NULL, NULL, side, requested_qty, requested_limit_price,
            submitted_at, signal_id, intent_payload_version,
            intent_payload_json, lifecycle_status, broker_status, filled_qty,
            filled_avg_price,
            CASE
                WHEN filled_qty > 0
                THEN COALESCE(last_refreshed_at, submitted_at)
                ELSE NULL
            END,
            CASE WHEN filled_qty > 0 THEN 'observed' ELSE NULL END,
            NULL, last_refreshed_at, terminal_reason, terminal_at,
            position_kind, position_id
        FROM pending_orders
        """
    )
    conn.execute("DROP TABLE pending_orders")
    conn.execute("ALTER TABLE pending_orders_v29 RENAME TO pending_orders")
    _create_v29_pending_order_objects(conn)


def init_db() -> None:
    """Create the schema if absent and apply any pending migrations. Idempotent."""
    conn = get_connection()
    try:
        conn.executescript(_SCHEMA_SQL)
        _migrate_to_v2(conn)
        _migrate_to_v3(conn)
        _migrate_to_v4(conn)
        _migrate_to_v5(conn)
        _migrate_to_v6(conn)
        _migrate_to_v7(conn)
        _migrate_to_v8(conn)
        _migrate_to_v9(conn)
        _migrate_to_v10(conn)
        _migrate_to_v11(conn)
        _migrate_to_v12(conn)
        _migrate_to_v13(conn)
        _migrate_to_v14(conn)
        _migrate_to_v15(conn)
        _migrate_to_v16(conn)
        _migrate_to_v17(conn)
        _migrate_to_v18(conn)
        _migrate_to_v19(conn)
        _migrate_to_v20(conn)
        _migrate_to_v21(conn)
        _migrate_to_v22(conn)
        _migrate_to_v23(conn)
        _migrate_to_v24(conn)
        _migrate_to_v25(conn)
        _migrate_to_v26(conn)
        _migrate_to_v27(conn)
        _migrate_to_v28(conn)
        _migrate_to_v29(conn)
        cur = conn.execute("SELECT version FROM schema_version LIMIT 1")
        row = cur.fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
            )
        elif int(row["version"]) < SCHEMA_VERSION:
            conn.execute(
                "UPDATE schema_version SET version = ?", (SCHEMA_VERSION,)
            )
        conn.commit()
    finally:
        conn.close()


def schema_version() -> int:
    """Return the current schema version (0 if uninitialized or unreadable)."""
    try:
        conn = get_connection()
    except sqlite3.OperationalError:
        return 0
    try:
        try:
            cur = conn.execute("SELECT version FROM schema_version LIMIT 1")
        except sqlite3.OperationalError:
            return 0
        row = cur.fetchone()
        return int(row["version"]) if row is not None else 0
    finally:
        conn.close()


# ---- signals ----


def insert_signal(signal: Signal) -> int:
    """Insert and return the new id. On dedupe conflict, return the existing id."""
    if signal.asset_class not in VALID_ASSET_CLASSES:
        raise ValueError(
            f"Invalid asset_class '{signal.asset_class}' "
            f"(expected one of {sorted(VALID_ASSET_CLASSES)})"
        )
    if signal.direction not in VALID_DIRECTIONS:
        raise ValueError(
            f"Invalid direction '{signal.direction}' "
            f"(expected one of {sorted(VALID_DIRECTIONS)})"
        )

    sql = """
        INSERT INTO signals (
            timestamp, ticker, asset_class, signal_type, direction,
            entry_price, stop_loss, take_profit, atr, rsi, macd,
            macd_signal, ema21, bb_upper, bb_lower, hold_estimate_days,
            earnings_risk, news_risk, raw_indicators_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = (
        signal.timestamp.isoformat(),
        signal.ticker,
        signal.asset_class,
        signal.signal_type,
        signal.direction,
        signal.entry_price,
        signal.stop_loss,
        signal.take_profit,
        signal.atr,
        signal.rsi,
        signal.macd,
        signal.macd_signal,
        signal.ema21,
        signal.bb_upper,
        signal.bb_lower,
        signal.hold_estimate_days,
        normalize_risk_text(signal.earnings_risk),
        normalize_risk_text(signal.news_risk),
        signal.raw_indicators_json,
    )
    conn = get_connection()
    try:
        try:
            cur = conn.execute(sql, params)
            conn.commit()
            new_id = cur.lastrowid
            if new_id is None:
                raise RuntimeError("INSERT did not return a row id")
            return new_id
        except sqlite3.IntegrityError:
            # Dedupe via UNIQUE (timestamp, ticker, signal_type) — return existing id.
            cur = conn.execute(
                "SELECT id FROM signals WHERE timestamp=? AND ticker=? AND signal_type=?",
                (signal.timestamp.isoformat(), signal.ticker, signal.signal_type),
            )
            row = cur.fetchone()
            if row is not None:
                return int(row["id"])
            raise
    finally:
        conn.close()


def _row_to_signal(row: sqlite3.Row) -> Signal:
    return Signal(
        timestamp=datetime.fromisoformat(row["timestamp"]),
        ticker=row["ticker"],
        asset_class=row["asset_class"],
        signal_type=row["signal_type"],
        direction=row["direction"],
        entry_price=float(row["entry_price"]),
        stop_loss=row["stop_loss"],
        take_profit=row["take_profit"],
        atr=row["atr"],
        rsi=row["rsi"],
        macd=row["macd"],
        macd_signal=row["macd_signal"],
        ema21=row["ema21"],
        bb_upper=row["bb_upper"],
        bb_lower=row["bb_lower"],
        hold_estimate_days=row["hold_estimate_days"],
        earnings_risk=normalize_risk_text(row["earnings_risk"]),
        news_risk=normalize_risk_text(row["news_risk"]),
        raw_indicators_json=row["raw_indicators_json"],
        considered_at=(
            datetime.fromisoformat(row["considered_at"])
            if row["considered_at"] else None
        ),
        id=int(row["id"]),
    )


def get_signal_by_id(signal_id: int) -> Signal | None:
    """Fetch a single signal by id, or None if absent."""
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM signals WHERE id = ?", (signal_id,)).fetchone()
    finally:
        conn.close()
    return _row_to_signal(row) if row is not None else None


def get_signals_without_trades() -> list[Signal]:
    """Return every signal that has no corresponding Trade row.

    Used by the Phase 1.3 backfill to seed Trade records for the historical
    CSV-imported signals (and any future stragglers).
    """
    sql = (
        "SELECT signals.* FROM signals "
        "LEFT JOIN trades ON trades.signal_id = signals.id "
        "WHERE trades.id IS NULL "
        "ORDER BY signals.timestamp ASC"
    )
    conn = get_connection()
    try:
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()
    return [_row_to_signal(r) for r in rows]


def get_signals(
    ticker: str | None = None,
    since: datetime | None = None,
    limit: int = 100,
) -> list[Signal]:
    """Query signals with optional filters, newest first."""
    clauses: list[str] = []
    params: list[Any] = []
    if ticker is not None:
        clauses.append("ticker = ?")
        params.append(ticker)
    if since is not None:
        clauses.append("timestamp >= ?")
        params.append(since.isoformat())
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT * FROM signals {where} ORDER BY timestamp DESC LIMIT ?"  # noqa: S608 - whitelisted fragments only
    params.append(limit)
    conn = get_connection()
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [_row_to_signal(r) for r in rows]


# ---- trades ----


def insert_trade(trade: Trade) -> int:
    """Insert and return the new id. Raises IntegrityError on bad signal_id."""
    if trade.outcome is not None and trade.outcome not in VALID_OUTCOMES:
        raise ValueError(
            f"Invalid outcome '{trade.outcome}' "
            f"(expected one of {sorted(VALID_OUTCOMES)})"
        )
    if (
        trade.market_regime is not None
        and trade.market_regime not in VALID_MARKET_REGIMES
    ):
        raise ValueError(
            f"Invalid market_regime '{trade.market_regime}' "
            f"(expected one of {sorted(VALID_MARKET_REGIMES)})"
        )
    if (
        trade.vix_band is not None
        and trade.vix_band not in VALID_VIX_BANDS
    ):
        raise ValueError(
            f"Invalid vix_band '{trade.vix_band}' "
            f"(expected one of {sorted(VALID_VIX_BANDS)})"
        )
    if trade.track_mode not in VALID_TRACK_MODES:
        raise ValueError(
            f"Invalid track_mode '{trade.track_mode}' "
            f"(expected one of {sorted(VALID_TRACK_MODES)})"
        )
    sql = """
        INSERT INTO trades (
            signal_id, opened_at, closed_at, exit_price,
            outcome, pnl_pct, pnl_dollars, notes, market_regime,
            vix_level, vix_band, context_score, track_mode,
            sentiment_score, sentiment_label, sentiment_ok,
            sentiment_rationale, heavy_news, headline_count,
            ind_ok, ind_atr, ind_realized_vol, ind_vol_regime, ind_rsi, ind_adx,
            ind_obv, ind_correlation, ind_concentration,
            risk_ok, risk_reason, risk_recommended_size, risk_stop_distance,
            risk_dollar_risk,
            risk_pct, risk_position_pct, risk_capped, risk_total_pct,
            risk_portfolio_verdict, risk_position_verdict, risk_cluster_pct,
            risk_cluster_verdict
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                  ?, ?, ?, ?, ?, ?, ?, ?, ?,
                  ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = (
        trade.signal_id,
        trade.opened_at.isoformat(),
        trade.closed_at.isoformat() if trade.closed_at is not None else None,
        trade.exit_price,
        trade.outcome,
        trade.pnl_pct,
        trade.pnl_dollars,
        trade.notes,
        trade.market_regime,
        trade.vix_level,
        trade.vix_band,
        trade.context_score,
        trade.track_mode,
        trade.sentiment_score,
        trade.sentiment_label,
        int(trade.sentiment_ok) if trade.sentiment_ok is not None else None,
        trade.sentiment_rationale,
        int(trade.heavy_news),
        trade.headline_count,
        int(trade.ind_ok) if trade.ind_ok is not None else None,
        trade.ind_atr,
        trade.ind_realized_vol,
        trade.ind_vol_regime,
        trade.ind_rsi,
        trade.ind_adx,
        trade.ind_obv,
        trade.ind_correlation,
        trade.ind_concentration,
        int(trade.risk_ok) if trade.risk_ok is not None else None,
        trade.risk_reason,
        trade.risk_recommended_size,
        trade.risk_stop_distance,
        trade.risk_dollar_risk,
        trade.risk_pct,
        trade.risk_position_pct,
        int(trade.risk_capped),
        trade.risk_total_pct,
        trade.risk_portfolio_verdict,
        trade.risk_position_verdict,
        trade.risk_cluster_pct,
        trade.risk_cluster_verdict,
    )
    conn = get_connection()
    try:
        cur = conn.execute(sql, params)
        conn.commit()
        new_id = cur.lastrowid
        if new_id is None:
            raise RuntimeError("INSERT did not return a row id")
        return new_id
    finally:
        conn.close()


def update_trade(trade_id: int, **fields: Any) -> None:
    """Update specified fields only. Whitelist enforced — column names never user-driven."""
    unknown = set(fields) - _TRADE_UPDATABLE_FIELDS
    if unknown:
        raise ValueError(
            f"Unknown trade field(s): {sorted(unknown)} "
            f"(allowed: {sorted(_TRADE_UPDATABLE_FIELDS)})"
        )
    if not fields:
        return

    normalized: dict[str, Any] = {}
    for key, value in fields.items():
        if isinstance(value, datetime):
            normalized[key] = value.isoformat()
        else:
            normalized[key] = value

    if (
        "outcome" in normalized
        and normalized["outcome"] is not None
        and normalized["outcome"] not in VALID_OUTCOMES
    ):
        raise ValueError(
            f"Invalid outcome '{normalized['outcome']}' "
            f"(expected one of {sorted(VALID_OUTCOMES)})"
        )
    if (
        "market_regime" in normalized
        and normalized["market_regime"] is not None
        and normalized["market_regime"] not in VALID_MARKET_REGIMES
    ):
        raise ValueError(
            f"Invalid market_regime '{normalized['market_regime']}' "
            f"(expected one of {sorted(VALID_MARKET_REGIMES)})"
        )
    if (
        "vix_band" in normalized
        and normalized["vix_band"] is not None
        and normalized["vix_band"] not in VALID_VIX_BANDS
    ):
        raise ValueError(
            f"Invalid vix_band '{normalized['vix_band']}' "
            f"(expected one of {sorted(VALID_VIX_BANDS)})"
        )

    set_clause = ", ".join(f"{k} = ?" for k in normalized)  # keys are whitelisted
    sql = f"UPDATE trades SET {set_clause} WHERE id = ?"  # noqa: S608 - whitelisted columns
    params = [*normalized.values(), trade_id]
    conn = get_connection()
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def _row_to_trade(row: sqlite3.Row) -> Trade:
    # Schema-version-aware: rows from a pre-v2 connection won't have
    # market_regime, pre-v3 won't have vix_*. Use row.keys() so this stays
    # robust mid-migration.
    keys = set(row.keys())
    return Trade(
        signal_id=int(row["signal_id"]),
        opened_at=datetime.fromisoformat(row["opened_at"]),
        closed_at=(
            datetime.fromisoformat(row["closed_at"]) if row["closed_at"] else None
        ),
        exit_price=row["exit_price"],
        outcome=row["outcome"],
        pnl_pct=row["pnl_pct"],
        pnl_dollars=row["pnl_dollars"],
        notes=row["notes"],
        market_regime=row["market_regime"] if "market_regime" in keys else None,
        vix_level=row["vix_level"] if "vix_level" in keys else None,
        vix_band=row["vix_band"] if "vix_band" in keys else None,
        context_score=row["context_score"] if "context_score" in keys else None,
        track_mode=row["track_mode"] if "track_mode" in keys else "active",
        sentiment_score=row["sentiment_score"] if "sentiment_score" in keys else None,
        sentiment_label=row["sentiment_label"] if "sentiment_label" in keys else None,
        sentiment_ok=(
            bool(row["sentiment_ok"])
            if "sentiment_ok" in keys and row["sentiment_ok"] is not None
            else None
        ),
        sentiment_rationale=(
            row["sentiment_rationale"] if "sentiment_rationale" in keys else None
        ),
        heavy_news=bool(row["heavy_news"]) if "heavy_news" in keys else False,
        headline_count=row["headline_count"] if "headline_count" in keys else None,
        ind_ok=(
            bool(row["ind_ok"])
            if "ind_ok" in keys and row["ind_ok"] is not None
            else None
        ),
        ind_atr=row["ind_atr"] if "ind_atr" in keys else None,
        ind_realized_vol=row["ind_realized_vol"] if "ind_realized_vol" in keys else None,
        ind_vol_regime=row["ind_vol_regime"] if "ind_vol_regime" in keys else None,
        ind_rsi=row["ind_rsi"] if "ind_rsi" in keys else None,
        ind_adx=row["ind_adx"] if "ind_adx" in keys else None,
        ind_obv=row["ind_obv"] if "ind_obv" in keys else None,
        ind_correlation=row["ind_correlation"] if "ind_correlation" in keys else None,
        ind_concentration=(
            row["ind_concentration"] if "ind_concentration" in keys else None
        ),
        risk_ok=(
            bool(row["risk_ok"])
            if "risk_ok" in keys and row["risk_ok"] is not None
            else None
        ),
        risk_reason=row["risk_reason"] if "risk_reason" in keys else None,
        risk_recommended_size=(
            row["risk_recommended_size"] if "risk_recommended_size" in keys else None
        ),
        risk_stop_distance=(
            row["risk_stop_distance"] if "risk_stop_distance" in keys else None
        ),
        risk_dollar_risk=(
            row["risk_dollar_risk"] if "risk_dollar_risk" in keys else None
        ),
        risk_pct=row["risk_pct"] if "risk_pct" in keys else None,
        risk_position_pct=(
            row["risk_position_pct"] if "risk_position_pct" in keys else None
        ),
        risk_capped=(
            bool(row["risk_capped"]) if "risk_capped" in keys else False
        ),
        risk_total_pct=row["risk_total_pct"] if "risk_total_pct" in keys else None,
        risk_portfolio_verdict=(
            row["risk_portfolio_verdict"] if "risk_portfolio_verdict" in keys else None
        ),
        risk_position_verdict=(
            row["risk_position_verdict"] if "risk_position_verdict" in keys else None
        ),
        risk_cluster_pct=(
            row["risk_cluster_pct"] if "risk_cluster_pct" in keys else None
        ),
        risk_cluster_verdict=(
            row["risk_cluster_verdict"] if "risk_cluster_verdict" in keys else None
        ),
        id=int(row["id"]),
    )


def get_open_trades() -> list[Trade]:
    """Return trades where outcome IS NULL or outcome = 'open'."""
    sql = (
        "SELECT * FROM trades "
        "WHERE outcome IS NULL OR outcome = 'open' "
        "ORDER BY opened_at DESC"
    )
    conn = get_connection()
    try:
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()
    return [_row_to_trade(r) for r in rows]


def get_trade_by_signal_id(signal_id: int) -> Trade | None:
    """Return the Trade for a given signal_id, or None if no trade exists yet.

    Schema doesn't enforce 1:1 (no UNIQUE constraint on trades.signal_id)
    so callers that rely on 1:1 should treat any result as "trade exists".
    """
    sql = "SELECT * FROM trades WHERE signal_id = ? ORDER BY id ASC LIMIT 1"
    conn = get_connection()
    try:
        row = conn.execute(sql, (signal_id,)).fetchone()
    finally:
        conn.close()
    return _row_to_trade(row) if row is not None else None


# ---- daily performance ----


def upsert_daily_performance(perf: DailyPerf) -> None:
    """Insert or update by date."""
    sql = """
        INSERT INTO daily_performance (
            date, signals_fired, trades_opened, trades_closed,
            wins, losses, win_rate, total_pnl_pct
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(date) DO UPDATE SET
            signals_fired = excluded.signals_fired,
            trades_opened = excluded.trades_opened,
            trades_closed = excluded.trades_closed,
            wins = excluded.wins,
            losses = excluded.losses,
            win_rate = excluded.win_rate,
            total_pnl_pct = excluded.total_pnl_pct
    """
    params = (
        perf.date,
        perf.signals_fired,
        perf.trades_opened,
        perf.trades_closed,
        perf.wins,
        perf.losses,
        perf.win_rate,
        perf.total_pnl_pct,
    )
    conn = get_connection()
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def get_daily_performance(target_date: date) -> DailyPerf | None:
    """Fetch the perf row for a given date, or None if absent."""
    sql = "SELECT * FROM daily_performance WHERE date = ?"
    conn = get_connection()
    try:
        row = conn.execute(sql, (target_date.isoformat(),)).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return DailyPerf(
        date=row["date"],
        signals_fired=int(row["signals_fired"]),
        trades_opened=int(row["trades_opened"]),
        trades_closed=int(row["trades_closed"]),
        wins=int(row["wins"]),
        losses=int(row["losses"]),
        win_rate=row["win_rate"],
        total_pnl_pct=row["total_pnl_pct"],
        id=int(row["id"]),
    )


# ---- regime snapshots (Phase 2.1) ----


def upsert_regime_snapshot(
    *,
    snapshot_date: str,
    regime: str,
    spy_close: float,
    ema50: float,
    ema200: float,
    ema50_slope: float,
    captured_at: datetime,
) -> None:
    """Insert or replace the regime_snapshots row for ``snapshot_date``.

    Idempotent: re-running with the same date overwrites the prior row. The
    daily scheduler relies on this — a second run inside the same UTC day
    must not create a duplicate.
    """
    if regime not in VALID_MARKET_REGIMES:
        raise ValueError(
            f"Invalid regime '{regime}' "
            f"(expected one of {sorted(VALID_MARKET_REGIMES)})"
        )
    sql = """
        INSERT INTO regime_snapshots (
            date, regime, spy_close, ema50, ema200, ema50_slope, captured_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(date) DO UPDATE SET
            regime = excluded.regime,
            spy_close = excluded.spy_close,
            ema50 = excluded.ema50,
            ema200 = excluded.ema200,
            ema50_slope = excluded.ema50_slope,
            captured_at = excluded.captured_at
    """
    params = (
        snapshot_date,
        regime,
        spy_close,
        ema50,
        ema200,
        ema50_slope,
        captured_at.isoformat(),
    )
    conn = get_connection()
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def get_regime_snapshot(snapshot_date: str) -> dict[str, Any] | None:
    """Fetch a single snapshot by date, or None if absent."""
    sql = "SELECT * FROM regime_snapshots WHERE date = ?"
    conn = get_connection()
    try:
        row = conn.execute(sql, (snapshot_date,)).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return _regime_row_to_dict(row)


def get_regime_snapshots(limit: int = 30) -> list[dict[str, Any]]:
    """Return the most recent ``limit`` regime_snapshots, newest first."""
    sql = "SELECT * FROM regime_snapshots ORDER BY date DESC LIMIT ?"
    conn = get_connection()
    try:
        rows = conn.execute(sql, (limit,)).fetchall()
    finally:
        conn.close()
    return [_regime_row_to_dict(r) for r in rows]


def _regime_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "date": str(row["date"]),
        "regime": str(row["regime"]),
        "spy_close": float(row["spy_close"]),
        "ema50": float(row["ema50"]),
        "ema200": float(row["ema200"]),
        "ema50_slope": float(row["ema50_slope"]),
        "captured_at": str(row["captured_at"]),
    }


# ---- vix snapshots (Phase 2.2) ----


def upsert_vix_snapshot(
    *,
    snapshot_date: str,
    vix_level: float,
    vix_band: str,
    captured_at: datetime,
) -> None:
    """Insert or replace the vix_snapshots row for ``snapshot_date``.

    Idempotent — same date overwrites. The daily scheduler relies on this
    so a second run in the same UTC day does not create a duplicate row.
    """
    if vix_band not in VALID_VIX_BANDS:
        raise ValueError(
            f"Invalid vix_band '{vix_band}' "
            f"(expected one of {sorted(VALID_VIX_BANDS)})"
        )
    sql = """
        INSERT INTO vix_snapshots (date, vix_level, vix_band, captured_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(date) DO UPDATE SET
            vix_level = excluded.vix_level,
            vix_band = excluded.vix_band,
            captured_at = excluded.captured_at
    """
    params = (snapshot_date, vix_level, vix_band, captured_at.isoformat())
    conn = get_connection()
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def get_vix_snapshot(snapshot_date: str) -> dict[str, Any] | None:
    sql = "SELECT * FROM vix_snapshots WHERE date = ?"
    conn = get_connection()
    try:
        row = conn.execute(sql, (snapshot_date,)).fetchone()
    finally:
        conn.close()
    return _vix_row_to_dict(row) if row is not None else None


def get_vix_snapshots(limit: int = 30) -> list[dict[str, Any]]:
    """Return the most recent ``limit`` vix_snapshots rows, newest first."""
    sql = "SELECT * FROM vix_snapshots ORDER BY date DESC LIMIT ?"
    conn = get_connection()
    try:
        rows = conn.execute(sql, (limit,)).fetchall()
    finally:
        conn.close()
    return [_vix_row_to_dict(r) for r in rows]


def _vix_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "date": str(row["date"]),
        "vix_level": float(row["vix_level"]),
        "vix_band": str(row["vix_band"]),
        "captured_at": str(row["captured_at"]),
    }


def get_closed_trades_missing_vix() -> list[Trade]:
    """Closed trades (win/loss/expired) where vix_band IS NULL.

    Used by the Phase 2.2 backfill subcommand. Independent of the regime
    backfill — a trade that already has a regime tag but no VIX tag is a
    legitimate candidate. Open trades are excluded.
    """
    sql = (
        "SELECT * FROM trades "
        "WHERE outcome IN ('win','loss','expired') AND vix_band IS NULL "
        "ORDER BY opened_at ASC"
    )
    conn = get_connection()
    try:
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()
    return [_row_to_trade(r) for r in rows]


def get_trades_missing_context_score() -> list[Trade]:
    """Closed trades where market_regime and vix_band are populated but
    context_score is NULL. Phase 2.3 backfill target — pure derivation
    from already-stored columns, no yfinance involvement."""
    sql = (
        "SELECT * FROM trades "
        "WHERE outcome IN ('win','loss','expired') "
        "  AND context_score IS NULL "
        "  AND market_regime IS NOT NULL "
        "  AND vix_band IS NOT NULL "
        "ORDER BY opened_at ASC"
    )
    conn = get_connection()
    try:
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()
    return [_row_to_trade(r) for r in rows]


def get_predictions_missing_context_score() -> list[Prediction]:
    """Predictions where market_regime + vix_band are populated but
    context_score is NULL. Phase 2.3 backfill target."""
    sql = (
        "SELECT * FROM predictions "
        "WHERE context_score IS NULL "
        "  AND market_regime IS NOT NULL "
        "  AND vix_band IS NOT NULL "
        "ORDER BY created_at ASC"
    )
    conn = get_connection()
    try:
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()
    return [_row_to_prediction(r) for r in rows]


def get_combined_context_history(limit: int = 30) -> list[dict[str, Any]]:
    """Join regime_snapshots + vix_snapshots by date, newest first.

    Returns one dict per date that has BOTH a regime and a VIX snapshot.
    Dates with only one axis present are omitted — that's the spec for
    ``context history``. Score is computed in Python rather than baked
    into SQL so the matrix table remains the single source of truth.
    """
    sql = (
        "SELECT r.date AS date, "
        "       r.regime AS regime, "
        "       v.vix_band AS vix_band, "
        "       v.vix_level AS vix_level "
        "FROM regime_snapshots r "
        "INNER JOIN vix_snapshots v ON v.date = r.date "
        "ORDER BY r.date DESC LIMIT ?"
    )
    conn = get_connection()
    try:
        rows = conn.execute(sql, (limit,)).fetchall()
    finally:
        conn.close()
    return [
        {
            "date": str(r["date"]),
            "regime": str(r["regime"]),
            "vix_band": str(r["vix_band"]),
            "vix_level": float(r["vix_level"]),
        }
        for r in rows
    ]


def get_closed_trades_missing_regime() -> list[Trade]:
    """Return every closed trade (win/loss/expired) where market_regime IS NULL.

    Used by the Phase 2.1 backfill subcommand. Open trades are intentionally
    excluded — only closed trades have a historical regime worth backfilling,
    and live trades get tagged at fire time going forward.
    """
    sql = (
        "SELECT * FROM trades "
        "WHERE outcome IN ('win','loss','expired') AND market_regime IS NULL "
        "ORDER BY opened_at ASC"
    )
    conn = get_connection()
    try:
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()
    return [_row_to_trade(r) for r in rows]


# ---- predictions (Phase 2.2b) ----


def insert_prediction(prediction: Prediction) -> int:
    """Insert a new Prediction and return its id."""
    if prediction.direction not in VALID_PREDICTION_DIRECTIONS:
        raise ValueError(
            f"Invalid direction '{prediction.direction}' "
            f"(expected one of {sorted(VALID_PREDICTION_DIRECTIONS)})"
        )
    if (
        prediction.outcome is not None
        and prediction.outcome not in VALID_PREDICTION_OUTCOMES
    ):
        raise ValueError(
            f"Invalid outcome '{prediction.outcome}' "
            f"(expected one of {sorted(VALID_PREDICTION_OUTCOMES)})"
        )
    sql = """
        INSERT INTO predictions (
            created_at, ticker, direction, confidence, entry_price,
            target_window_end, signals_used, market_regime, vix_band,
            vix_level, resolved_at, exit_price, outcome, notified,
            context_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = (
        prediction.created_at.isoformat(),
        prediction.ticker,
        prediction.direction,
        prediction.confidence,
        prediction.entry_price,
        prediction.target_window_end.isoformat(),
        prediction.signals_used,
        prediction.market_regime,
        prediction.vix_band,
        prediction.vix_level,
        prediction.resolved_at.isoformat() if prediction.resolved_at else None,
        prediction.exit_price,
        prediction.outcome,
        int(prediction.notified),
        prediction.context_score,
    )
    conn = get_connection()
    try:
        cur = conn.execute(sql, params)
        conn.commit()
        new_id = cur.lastrowid
        if new_id is None:
            raise RuntimeError("INSERT did not return a row id")
        return new_id
    finally:
        conn.close()


def update_prediction(prediction_id: int, **fields: Any) -> None:
    """Whitelisted partial update — used by the resolution flow."""
    unknown = set(fields) - _PREDICTION_UPDATABLE_FIELDS
    if unknown:
        raise ValueError(
            f"Unknown prediction field(s): {sorted(unknown)} "
            f"(allowed: {sorted(_PREDICTION_UPDATABLE_FIELDS)})"
        )
    if not fields:
        return

    normalized: dict[str, Any] = {}
    for key, value in fields.items():
        if isinstance(value, datetime):
            normalized[key] = value.isoformat()
        elif key == "notified" and isinstance(value, bool):
            normalized[key] = int(value)
        else:
            normalized[key] = value

    if (
        "outcome" in normalized
        and normalized["outcome"] is not None
        and normalized["outcome"] not in VALID_PREDICTION_OUTCOMES
    ):
        raise ValueError(
            f"Invalid outcome '{normalized['outcome']}' "
            f"(expected one of {sorted(VALID_PREDICTION_OUTCOMES)})"
        )

    set_clause = ", ".join(f"{k} = ?" for k in normalized)
    sql = f"UPDATE predictions SET {set_clause} WHERE id = ?"  # noqa: S608 - whitelist
    params = [*normalized.values(), prediction_id]
    conn = get_connection()
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def _row_to_prediction(row: sqlite3.Row) -> Prediction:
    keys = set(row.keys())
    return Prediction(
        ticker=str(row["ticker"]),
        direction=str(row["direction"]),
        confidence=float(row["confidence"]),
        entry_price=float(row["entry_price"]),
        target_window_end=datetime.fromisoformat(row["target_window_end"]),
        signals_used=str(row["signals_used"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        market_regime=row["market_regime"],
        vix_band=row["vix_band"],
        vix_level=row["vix_level"],
        resolved_at=(
            datetime.fromisoformat(row["resolved_at"])
            if row["resolved_at"] else None
        ),
        exit_price=row["exit_price"],
        outcome=row["outcome"],
        notified=bool(row["notified"]),
        context_score=row["context_score"] if "context_score" in keys else None,
        id=int(row["id"]),
    )


def get_prediction(prediction_id: int) -> Prediction | None:
    sql = "SELECT * FROM predictions WHERE id = ?"
    conn = get_connection()
    try:
        row = conn.execute(sql, (prediction_id,)).fetchone()
    finally:
        conn.close()
    return _row_to_prediction(row) if row is not None else None


def get_unresolved_predictions(now: datetime | None = None) -> list[Prediction]:
    """Predictions where the 15-min target window has passed but no
    resolution row has been written yet. Sorted oldest first so the
    resolver tackles them in order."""
    cutoff = (now if now is not None else datetime.now(UTC)).isoformat()
    sql = (
        "SELECT * FROM predictions "
        "WHERE resolved_at IS NULL AND target_window_end <= ? "
        "ORDER BY target_window_end ASC"
    )
    conn = get_connection()
    try:
        rows = conn.execute(sql, (cutoff,)).fetchall()
    finally:
        conn.close()
    return [_row_to_prediction(r) for r in rows]


def get_predictions(
    ticker: str | None = None,
    since: datetime | None = None,
    limit: int = 100,
) -> list[Prediction]:
    clauses: list[str] = []
    params: list[Any] = []
    if ticker is not None:
        clauses.append("ticker = ?")
        params.append(ticker)
    if since is not None:
        clauses.append("created_at >= ?")
        params.append(since.isoformat())
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = (
        f"SELECT * FROM predictions {where} "  # noqa: S608 - whitelisted fragments
        f"ORDER BY created_at DESC LIMIT ?"
    )
    params.append(limit)
    conn = get_connection()
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [_row_to_prediction(r) for r in rows]


# ---- diagnostics ----


def get_table_counts() -> dict[str, int]:
    """Return a row count for each managed table."""
    counts: dict[str, int] = {}
    conn = get_connection()
    try:
        for table in _TABLE_NAMES:
            cur = conn.execute(f"SELECT COUNT(*) AS c FROM {table}")  # noqa: S608 - hardcoded names
            row = cur.fetchone()
            counts[table] = int(row["c"]) if row is not None else 0
    finally:
        conn.close()
    return counts


# ---- active watchlist (Phase 3.1) ----


def seed_active_watchlist(tickers: Sequence[str]) -> int:
    """Seed ``active_watchlist`` from ``tickers`` IF the table is empty.

    First-run bootstrap: copies the hardcoded seed list into the DB tagged
    ``source='seed'``. Idempotent — once any row exists (seed or discovery)
    this is a no-op returning 0, so it never clobbers promotions or a
    deliberately-pruned watchlist. Returns the number of tickers inserted.
    """
    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT COUNT(*) AS c FROM active_watchlist"
        ).fetchone()
        if existing is not None and int(existing["c"]) > 0:
            return 0
        now_iso = datetime.now(UTC).isoformat()
        inserted = 0
        seen: set[str] = set()
        for ticker in tickers:
            if ticker in seen:
                continue
            seen.add(ticker)
            conn.execute(
                "INSERT OR IGNORE INTO active_watchlist (ticker, source, added_at) "
                "VALUES (?, ?, ?)",
                (ticker, "seed", now_iso),
            )
            inserted += 1
        conn.commit()
        return inserted
    finally:
        conn.close()


def get_active_watchlist() -> list[str]:
    """Return the live watchlist tickers, oldest-added first.

    Empty list if the table has no rows. Propagates
    ``sqlite3.OperationalError`` if the table is missing/unreadable — the
    scanner catches that and falls back to its hardcoded seed list.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT ticker FROM active_watchlist "
            "ORDER BY added_at ASC, ticker ASC"
        ).fetchall()
    finally:
        conn.close()
    return [str(r["ticker"]) for r in rows]


def add_to_active_watchlist(ticker: str, source: str) -> bool:
    """Insert ``ticker`` tagged with ``source`` if not already present.

    Returns ``True`` if newly added, ``False`` if it was already on the
    watchlist (no duplicate inserted). Used by discovery auto-promotion with
    ``source='discovery'``.
    """
    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT 1 FROM active_watchlist WHERE ticker = ?", (ticker,)
        ).fetchone()
        if existing is not None:
            return False
        conn.execute(
            "INSERT INTO active_watchlist (ticker, source, added_at) "
            "VALUES (?, ?, ?)",
            (ticker, source, datetime.now(UTC).isoformat()),
        )
        conn.commit()
        return True
    finally:
        conn.close()


# ---- watchlist status / transitions (Phase 3.3) ----


def get_watchlist_entries() -> list[dict[str, Any]]:
    """Return every watchlist row with its status, oldest-added first.

    Includes BOTH active and benched tickers. ``status`` defaults to 'active'
    for rows that pre-date the v10 column add (the migration backfills them).
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT ticker, source, status, added_at, status_changed_at "
            "FROM active_watchlist ORDER BY added_at ASC, ticker ASC"
        ).fetchall()
    finally:
        conn.close()
    keys: set[str] = set(rows[0].keys()) if rows else set()
    return [
        {
            "ticker": str(r["ticker"]),
            "source": str(r["source"]),
            "status": (
                str(r["status"]) if "status" in keys and r["status"] is not None
                else "active"
            ),
            "added_at": str(r["added_at"]),
            "status_changed_at": (
                r["status_changed_at"] if "status_changed_at" in keys else None
            ),
        }
        for r in rows
    ]


def set_watchlist_status(
    ticker: str, status: str, *, changed_at: datetime | None = None
) -> None:
    """Set a ticker's active/benched status and stamp status_changed_at.

    Validates ``status`` against the allowed set so a typo fails loud rather
    than silently writing a bad state. A no-op if the ticker isn't present.
    """
    if status not in VALID_WATCHLIST_STATUSES:
        raise ValueError(
            f"Invalid status '{status}' "
            f"(expected one of {sorted(VALID_WATCHLIST_STATUSES)})"
        )
    when = (changed_at if changed_at is not None else datetime.now(UTC)).isoformat()
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE active_watchlist SET status = ?, status_changed_at = ? "
            "WHERE ticker = ?",
            (status, when, ticker),
        )
        conn.commit()
    finally:
        conn.close()


def insert_watchlist_transition(
    *,
    ticker: str,
    from_status: str,
    to_status: str,
    evaluated_at: datetime,
    win_rate: float | None,
    closed_count: int,
    expectancy: float | None,
    reason: str,
) -> None:
    """Record one active<->benched transition in the audit table."""
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO watchlist_transitions ("
            "  ticker, from_status, to_status, evaluated_at, "
            "  win_rate, closed_count, expectancy, reason"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ticker,
                from_status,
                to_status,
                evaluated_at.isoformat(),
                win_rate,
                int(closed_count),
                expectancy,
                reason,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_watchlist_transitions(limit: int = 100) -> list[dict[str, Any]]:
    """Return recent watchlist transitions, newest first."""
    sql = (
        "SELECT * FROM watchlist_transitions "
        "ORDER BY evaluated_at DESC, id DESC LIMIT ?"
    )
    conn = get_connection()
    try:
        rows = conn.execute(sql, (limit,)).fetchall()
    finally:
        conn.close()
    return [
        {
            "ticker": str(r["ticker"]),
            "from_status": str(r["from_status"]),
            "to_status": str(r["to_status"]),
            "evaluated_at": str(r["evaluated_at"]),
            "win_rate": r["win_rate"],
            "closed_count": int(r["closed_count"]),
            "expectancy": r["expectancy"],
            "reason": str(r["reason"]),
        }
        for r in rows
    ]


# ---- signal pair status / transitions (Phase 4) ----


def get_signal_pair_status(ticker: str, signal_type: str) -> str:
    """Return a pair's gate status, or ``'enabled'`` when no row exists.

    Default-enabled (innocent until proven losing): a (ticker, signal_type)
    pair only has a row once the evaluator has muted/enabled it.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT status FROM signal_pair_status "
            "WHERE ticker = ? AND signal_type = ?",
            (ticker, signal_type),
        ).fetchone()
    finally:
        conn.close()
    return str(row["status"]) if row is not None else "enabled"


def set_signal_pair_status(
    ticker: str,
    signal_type: str,
    status: str,
    *,
    changed_at: datetime | None = None,
) -> None:
    """Upsert a pair's gate status and stamp status_changed_at.

    Validates ``status`` so a typo fails loud rather than writing a bad gate.
    """
    if status not in VALID_PAIR_STATUSES:
        raise ValueError(
            f"Invalid status '{status}' "
            f"(expected one of {sorted(VALID_PAIR_STATUSES)})"
        )
    when = (changed_at if changed_at is not None else datetime.now(UTC)).isoformat()
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO signal_pair_status "
            "  (ticker, signal_type, status, status_changed_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(ticker, signal_type) DO UPDATE SET "
            "  status = excluded.status, "
            "  status_changed_at = excluded.status_changed_at",
            (ticker, signal_type, status, when),
        )
        conn.commit()
    finally:
        conn.close()


def get_signal_pair_statuses() -> dict[tuple[str, str], str]:
    """Return every EXPLICIT (ticker, signal_type) -> status row.

    Pairs absent from this mapping are enabled by default — callers that need
    a pair's effective status should use :func:`get_signal_pair_status`.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT ticker, signal_type, status FROM signal_pair_status"
        ).fetchall()
    finally:
        conn.close()
    return {
        (str(r["ticker"]), str(r["signal_type"])): str(r["status"]) for r in rows
    }


def insert_signal_pair_transition(
    *,
    ticker: str,
    signal_type: str,
    from_status: str,
    to_status: str,
    evaluated_at: datetime,
    win_rate: float | None,
    closed_count: int,
    expectancy: float | None,
    reason: str,
) -> None:
    """Record one enabled<->muted pair transition in the audit table."""
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO signal_pair_transitions ("
            "  ticker, signal_type, from_status, to_status, evaluated_at, "
            "  win_rate, closed_count, expectancy, reason"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ticker,
                signal_type,
                from_status,
                to_status,
                evaluated_at.isoformat(),
                win_rate,
                int(closed_count),
                expectancy,
                reason,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_signal_pair_transitions(limit: int = 100) -> list[dict[str, Any]]:
    """Return recent signal-pair transitions, newest first."""
    sql = (
        "SELECT * FROM signal_pair_transitions "
        "ORDER BY evaluated_at DESC, id DESC LIMIT ?"
    )
    conn = get_connection()
    try:
        rows = conn.execute(sql, (limit,)).fetchall()
    finally:
        conn.close()
    return [
        {
            "ticker": str(r["ticker"]),
            "signal_type": str(r["signal_type"]),
            "from_status": str(r["from_status"]),
            "to_status": str(r["to_status"]),
            "evaluated_at": str(r["evaluated_at"]),
            "win_rate": r["win_rate"],
            "closed_count": int(r["closed_count"]),
            "expectancy": r["expectancy"],
            "reason": str(r["reason"]),
        }
        for r in rows
    ]


def get_recent_trade_sentiment(limit: int = 20) -> list[dict[str, Any]]:
    """Recent fired signals that carry advisory sentiment context (Phase 5).

    Joins trades -> signals, newest first, including explicit success/failure
    and legacy labels whose provenance is unknown. Each row carries the signal
    identity, sentiment rationale, and eventual outcome beside the result.
    """
    sql = (
        "SELECT signals.ticker AS ticker, "
        "       signals.signal_type AS signal_type, "
        "       signals.direction AS direction, "
        "       trades.opened_at AS opened_at, "
        "       trades.outcome AS outcome, "
        "       trades.track_mode AS track_mode, "
        "       trades.sentiment_score AS sentiment_score, "
        "       trades.sentiment_label AS sentiment_label, "
        "       trades.sentiment_ok AS sentiment_ok, "
        "       trades.sentiment_rationale AS sentiment_rationale, "
        "       trades.heavy_news AS heavy_news, "
        "       trades.headline_count AS headline_count "
        "FROM trades JOIN signals ON signals.id = trades.signal_id "
        "WHERE trades.sentiment_ok IS NOT NULL "
        "   OR trades.sentiment_label IS NOT NULL "
        "ORDER BY trades.opened_at DESC LIMIT ?"
    )
    conn = get_connection()
    try:
        rows = conn.execute(sql, (limit,)).fetchall()
    finally:
        conn.close()
    return [
        {
            "ticker": str(r["ticker"]),
            "signal_type": str(r["signal_type"]),
            "direction": str(r["direction"]),
            "opened_at": str(r["opened_at"]),
            "outcome": r["outcome"],
            "track_mode": str(r["track_mode"]),
            "sentiment_score": r["sentiment_score"],
            "sentiment_label": r["sentiment_label"],
            "sentiment_ok": (
                bool(r["sentiment_ok"])
                if r["sentiment_ok"] is not None
                else None
            ),
            "sentiment_rationale": r["sentiment_rationale"],
            "heavy_news": bool(r["heavy_news"]),
            "headline_count": r["headline_count"],
        }
        for r in rows
    ]


def get_recent_trade_indicators(limit: int = 20) -> list[dict[str, Any]]:
    """Recent fired signals that carry advisory indicator-family context (Phase 6).

    Explicit success/failure is returned separately from the family values;
    legacy rows with context but no provenance remain visible as unknown.
    """
    sql = (
        "SELECT signals.ticker AS ticker, "
        "       signals.signal_type AS signal_type, "
        "       signals.direction AS direction, "
        "       trades.opened_at AS opened_at, "
        "       trades.outcome AS outcome, "
        "       trades.track_mode AS track_mode, "
        "       trades.ind_ok AS ind_ok, "
        "       trades.ind_atr AS ind_atr, "
        "       trades.ind_realized_vol AS ind_realized_vol, "
        "       trades.ind_vol_regime AS ind_vol_regime, "
        "       trades.ind_rsi AS ind_rsi, "
        "       trades.ind_adx AS ind_adx, "
        "       trades.ind_obv AS ind_obv, "
        "       trades.ind_correlation AS ind_correlation, "
        "       trades.ind_concentration AS ind_concentration "
        "FROM trades JOIN signals ON signals.id = trades.signal_id "
        "WHERE trades.ind_ok IS NOT NULL "
        "   OR trades.ind_vol_regime IS NOT NULL "
        "ORDER BY trades.opened_at DESC LIMIT ?"
    )
    conn = get_connection()
    try:
        rows = conn.execute(sql, (limit,)).fetchall()
    finally:
        conn.close()
    return [
        {
            "ticker": str(r["ticker"]),
            "signal_type": str(r["signal_type"]),
            "direction": str(r["direction"]),
            "opened_at": str(r["opened_at"]),
            "outcome": r["outcome"],
            "track_mode": str(r["track_mode"]),
            "ind_ok": (
                bool(r["ind_ok"])
                if r["ind_ok"] is not None
                else None
            ),
            "ind_atr": r["ind_atr"],
            "ind_realized_vol": r["ind_realized_vol"],
            "ind_vol_regime": r["ind_vol_regime"],
            "ind_rsi": r["ind_rsi"],
            "ind_adx": r["ind_adx"],
            "ind_obv": r["ind_obv"],
            "ind_correlation": r["ind_correlation"],
            "ind_concentration": (
                str(r["ind_concentration"])
                if r["ind_concentration"] is not None
                else None
            ),
        }
        for r in rows
    ]


def get_recent_trade_risk(limit: int = 20) -> list[dict[str, Any]]:
    """Recent fired signals that carry an advisory risk recommendation (Phase 7).

    Explicit success/failure is returned separately from the verdicts; legacy
    rows with risk context but no provenance remain visible as unknown.
    """
    sql = (
        "SELECT signals.ticker AS ticker, "
        "       signals.signal_type AS signal_type, "
        "       signals.direction AS direction, "
        "       trades.opened_at AS opened_at, "
        "       trades.outcome AS outcome, "
        "       trades.track_mode AS track_mode, "
        "       trades.risk_ok AS risk_ok, "
        "       trades.risk_reason AS risk_reason, "
        "       trades.risk_recommended_size AS risk_recommended_size, "
        "       trades.risk_pct AS risk_pct, "
        "       trades.risk_position_pct AS risk_position_pct, "
        "       trades.risk_capped AS risk_capped, "
        "       trades.risk_total_pct AS risk_total_pct, "
        "       trades.risk_portfolio_verdict AS risk_portfolio_verdict, "
        "       trades.risk_position_verdict AS risk_position_verdict, "
        "       trades.risk_cluster_pct AS risk_cluster_pct, "
        "       trades.risk_cluster_verdict AS risk_cluster_verdict "
        "FROM trades JOIN signals ON signals.id = trades.signal_id "
        "WHERE trades.risk_ok IS NOT NULL "
        "   OR trades.risk_portfolio_verdict IS NOT NULL "
        "ORDER BY trades.opened_at DESC LIMIT ?"
    )
    conn = get_connection()
    try:
        rows = conn.execute(sql, (limit,)).fetchall()
    finally:
        conn.close()
    return [
        {
            "ticker": str(r["ticker"]),
            "signal_type": str(r["signal_type"]),
            "direction": str(r["direction"]),
            "opened_at": str(r["opened_at"]),
            "outcome": r["outcome"],
            "track_mode": str(r["track_mode"]),
            "risk_ok": (
                bool(r["risk_ok"])
                if r["risk_ok"] is not None
                else None
            ),
            "risk_reason": r["risk_reason"],
            "risk_recommended_size": r["risk_recommended_size"],
            "risk_pct": r["risk_pct"],
            "risk_position_pct": r["risk_position_pct"],
            "risk_capped": bool(r["risk_capped"]),
            "risk_total_pct": r["risk_total_pct"],
            "risk_portfolio_verdict": r["risk_portfolio_verdict"],
            "risk_position_verdict": (
                str(r["risk_position_verdict"])
                if r["risk_position_verdict"] is not None
                else None
            ),
            "risk_cluster_pct": r["risk_cluster_pct"],
            "risk_cluster_verdict": (
                str(r["risk_cluster_verdict"])
                if r["risk_cluster_verdict"] is not None
                else None
            ),
        }
        for r in rows
    ]


def get_recent_signal_risk_grades(limit: int = 20) -> list[dict[str, str]]:
    """Newest fired signals with their persisted earnings/news risk grades.

    This view deliberately reads ``signals`` directly rather than requiring a
    joined trade or advisory-risk row: the Phase 24 values describe the fired
    signal itself and should remain visible even when another context pipeline
    failed soft. Legacy integer 0/1 values are normalized only in the returned
    representation; the stored row is never rewritten.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT timestamp, ticker, signal_type, earnings_risk, news_risk "
            "FROM signals ORDER BY timestamp DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "timestamp": str(row["timestamp"]),
            "ticker": str(row["ticker"]),
            "signal_type": str(row["signal_type"]),
            "earnings_risk": normalize_risk_text(row["earnings_risk"]),
            "news_risk": normalize_risk_text(row["news_risk"]),
        }
        for row in rows
    ]


def get_traded_signal_pairs() -> list[tuple[str, str]]:
    """Distinct (ticker, signal_type) pairs that have at least one trade.

    Joins trades -> signals so only pairs that have actually fired are
    returned. Drives the per-pair evaluator and the per-pair report.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT DISTINCT signals.ticker AS ticker, "
            "       signals.signal_type AS signal_type "
            "FROM trades JOIN signals ON signals.id = trades.signal_id "
            "ORDER BY signals.ticker ASC, signals.signal_type ASC"
        ).fetchall()
    finally:
        conn.close()
    return [(str(r["ticker"]), str(r["signal_type"])) for r in rows]


# ---- discovery results (Phase 3.1) ----


def insert_discovery_results(
    run_timestamp: str, rows: Sequence[Mapping[str, Any]]
) -> int:
    """Persist one discovery sweep's per-ticker scores. Returns rows inserted.

    Each row mapping must carry ``ticker``, ``trade_count``, ``win_rate``,
    ``avg_return_pct``, ``expectancy``, and ``qualified``. The nullable score
    fields may be ``None`` (a ticker with no decided trades).
    """
    sql = """
        INSERT INTO discovery_results (
            run_timestamp, ticker, win_rate, trade_count,
            avg_return_pct, expectancy, qualified
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    conn = get_connection()
    try:
        count = 0
        for row in rows:
            conn.execute(
                sql,
                (
                    run_timestamp,
                    str(row["ticker"]),
                    row["win_rate"],
                    int(row["trade_count"]),
                    row["avg_return_pct"],
                    row["expectancy"],
                    1 if row["qualified"] else 0,
                ),
            )
            count += 1
        conn.commit()
        return count
    finally:
        conn.close()


def get_discovery_results(limit: int = 100) -> list[dict[str, Any]]:
    """Return recent discovery_results rows, newest run first."""
    sql = (
        "SELECT * FROM discovery_results "
        "ORDER BY run_timestamp DESC, expectancy DESC LIMIT ?"
    )
    conn = get_connection()
    try:
        rows = conn.execute(sql, (limit,)).fetchall()
    finally:
        conn.close()
    return [
        {
            "run_timestamp": str(r["run_timestamp"]),
            "ticker": str(r["ticker"]),
            "win_rate": r["win_rate"],
            "trade_count": int(r["trade_count"]),
            "avg_return_pct": r["avg_return_pct"],
            "expectancy": r["expectancy"],
            "qualified": bool(r["qualified"]),
        }
        for r in rows
    ]


# ---- shadow trades + evaluations (Phase 3.1-LIVE) ----


def get_resolved_outcomes(
    ticker: str,
    since: datetime,
    track_mode: str | None = None,
    signal_type: str | None = None,
) -> list[tuple[str, float | None]]:
    """Return ``(outcome, pnl_pct)`` for a ticker's RESOLVED trades.

    Win/loss trades with ``closed_at >= since`` (the recent window). Expired
    and open trades are excluded. ``track_mode`` filters to one mode, or
    ``None`` (default) spans ALL modes — the state evaluator needs every real
    outcome regardless of how the ticker was tracked (a benched ticker's
    recovery trades are tagged 'shadow' but are still its outcomes).
    ``signal_type`` narrows to a single setup so the same query serves the
    per-(ticker, signal_type) pair evaluator (Phase 4).
    """
    clauses = ""
    params: list[Any] = [ticker, since.isoformat()]
    if track_mode is not None:
        clauses += " AND trades.track_mode = ?"
        params.append(track_mode)
    if signal_type is not None:
        clauses += " AND signals.signal_type = ?"
        params.append(signal_type)
    sql = (
        "SELECT trades.outcome AS outcome, trades.pnl_pct AS pnl_pct "
        "FROM trades JOIN signals ON signals.id = trades.signal_id "
        "WHERE trades.outcome IN ('win','loss') "
        "  AND signals.ticker = ? "
        "  AND trades.closed_at IS NOT NULL "
        "  AND trades.closed_at >= ? "
        f"{clauses} "  # noqa: S608 - whitelisted fragments
        "ORDER BY trades.closed_at ASC"
    )
    conn = get_connection()
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [(str(r["outcome"]), r["pnl_pct"]) for r in rows]


def get_resolved_shadow_outcomes(
    ticker: str, since: datetime
) -> list[tuple[str, float | None]]:
    """Resolved shadow trades only — drives live-shadow promotion.

    Thin wrapper over :func:`get_resolved_outcomes` with ``track_mode='shadow'``.
    """
    return get_resolved_outcomes(ticker, since, track_mode="shadow")


def get_resolved_context_outcomes(
    since: datetime | None = None, until: datetime | None = None
) -> list[dict[str, Any]]:
    """Resolved ACTIVE trades with their stored advisory context (Phase 9).

    Win/loss only (expired/open excluded — the same resolved definition the
    gating subsystems use) and ``track_mode='active'`` (the live book that
    carries the Phase 5-7 context). Optional ``closed_at`` window ``[since,
    until)``. Each row carries the outcome + pnl plus the signal identity and
    every stored-context column self-optimization buckets against, so degradation
    slicing and feature evaluation both read from one query.
    """
    clauses = ""
    params: list[Any] = []
    if since is not None:
        clauses += " AND trades.closed_at >= ?"
        params.append(since.isoformat())
    if until is not None:
        clauses += " AND trades.closed_at < ?"
        params.append(until.isoformat())
    sql = (
        "SELECT trades.outcome AS outcome, trades.pnl_pct AS pnl_pct, "
        "       trades.closed_at AS closed_at, "
        "       signals.ticker AS ticker, signals.signal_type AS signal_type, "
        "       trades.sentiment_label AS sentiment_label, "
        "       trades.sentiment_ok AS sentiment_ok, "
        "       trades.sentiment_rationale AS sentiment_rationale, "
        "       trades.ind_ok AS ind_ok, "
        "       trades.ind_vol_regime AS ind_vol_regime, "
        "       trades.ind_rsi AS ind_rsi, trades.ind_adx AS ind_adx, "
        "       trades.ind_obv AS ind_obv, "
        "       trades.ind_concentration AS ind_concentration, "
        "       trades.risk_ok AS risk_ok, "
        "       trades.risk_reason AS risk_reason, "
        "       trades.risk_portfolio_verdict AS risk_portfolio_verdict "
        "FROM trades JOIN signals ON signals.id = trades.signal_id "
        "WHERE trades.track_mode = 'active' "
        "  AND trades.outcome IN ('win','loss') "
        "  AND trades.closed_at IS NOT NULL "
        f"{clauses} "  # noqa: S608 - whitelisted fragments, params bound
        "ORDER BY trades.closed_at ASC"
    )
    conn = get_connection()
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [
        {
            "outcome": str(r["outcome"]),
            "pnl_pct": r["pnl_pct"],
            "closed_at": str(r["closed_at"]),
            "ticker": str(r["ticker"]),
            "signal_type": str(r["signal_type"]),
            "sentiment_label": r["sentiment_label"],
            "sentiment_ok": (
                bool(r["sentiment_ok"])
                if r["sentiment_ok"] is not None
                else None
            ),
            "sentiment_rationale": r["sentiment_rationale"],
            "ind_ok": (
                bool(r["ind_ok"])
                if r["ind_ok"] is not None
                else None
            ),
            "ind_vol_regime": r["ind_vol_regime"],
            "ind_rsi": r["ind_rsi"],
            "ind_adx": r["ind_adx"],
            "ind_obv": r["ind_obv"],
            "ind_concentration": r["ind_concentration"],
            "risk_ok": (
                bool(r["risk_ok"])
                if r["risk_ok"] is not None
                else None
            ),
            "risk_reason": r["risk_reason"],
            "risk_portfolio_verdict": r["risk_portfolio_verdict"],
        }
        for r in rows
    ]


def insert_optimization_run(
    run_timestamp: str,
    degrade_window_days: int,
    baseline_window_days: int,
    findings_json: str,
) -> int:
    """Persist one self-optimization run's findings (Phase 9). Returns the id."""
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO optimization_runs "
            "(run_timestamp, degrade_window_days, baseline_window_days, findings_json) "
            "VALUES (?, ?, ?, ?)",
            (run_timestamp, degrade_window_days, baseline_window_days, findings_json),
        )
        conn.commit()
        new_id = cur.lastrowid
        if new_id is None:
            raise RuntimeError("INSERT did not return a row id")
        return new_id
    finally:
        conn.close()


def get_optimization_runs(limit: int = 10) -> list[dict[str, Any]]:
    """Recent self-optimization runs, newest first."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, run_timestamp, degrade_window_days, baseline_window_days, "
            "       findings_json "
            "FROM optimization_runs ORDER BY run_timestamp DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "id": int(r["id"]),
            "run_timestamp": str(r["run_timestamp"]),
            "degrade_window_days": int(r["degrade_window_days"]),
            "baseline_window_days": int(r["baseline_window_days"]),
            "findings_json": str(r["findings_json"]),
        }
        for r in rows
    ]


def get_latest_optimization_run() -> dict[str, Any] | None:
    """The most recent self-optimization run, or None if none have run."""
    runs = get_optimization_runs(limit=1)
    return runs[0] if runs else None


def count_resolved_trades(track_mode: str | None = None) -> int:
    """Cumulative count of RESOLVED (win/loss) trades (Phase 10 readiness).

    The canonical resolved definition (``outcome IN ('win','loss')`` — expired
    and open excluded). ``track_mode`` narrows to one mode, or ``None`` spans
    all. Not windowed: readiness measures ACCUMULATED data against a threshold.
    """
    clause = ""
    params: list[Any] = []
    if track_mode is not None:
        clause = " AND track_mode = ?"
        params.append(track_mode)
    sql = (  # noqa: S608 - fixed fragment + bound param
        f"SELECT COUNT(*) AS c FROM trades WHERE outcome IN ('win','loss'){clause}"
    )
    conn = get_connection()
    try:
        row = conn.execute(sql, params).fetchone()
    finally:
        conn.close()
    return int(row["c"]) if row is not None else 0


def get_readiness_state(capability: str) -> dict[str, Any] | None:
    """Return one capability's readiness ledger row, or None if never recorded."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT capability, status, n_at_crossing, crossed_at, announced "
            "FROM readiness_state WHERE capability = ?",
            (capability,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {
        "capability": str(row["capability"]),
        "status": str(row["status"]),
        "n_at_crossing": row["n_at_crossing"],
        "crossed_at": row["crossed_at"],
        "announced": bool(row["announced"]),
    }


def upsert_readiness_state(
    capability: str,
    *,
    status: str,
    n_at_crossing: int | None,
    crossed_at: str | None,
    announced: bool,
) -> None:
    """Insert or update a capability's readiness ledger row."""
    if status not in ("warming", "ready"):
        raise ValueError(f"Invalid readiness status '{status}'")
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO readiness_state "
            "(capability, status, n_at_crossing, crossed_at, announced) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(capability) DO UPDATE SET "
            "  status = excluded.status, "
            "  n_at_crossing = excluded.n_at_crossing, "
            "  crossed_at = excluded.crossed_at, "
            "  announced = excluded.announced",
            (capability, status, n_at_crossing, crossed_at, int(announced)),
        )
        conn.commit()
    finally:
        conn.close()


def insert_shadow_evaluations(
    run_timestamp: str, rows: Sequence[Mapping[str, Any]]
) -> int:
    """Persist one promotion-evaluation run. Returns rows inserted.

    Each row mapping must carry ``ticker``, ``closed_count``, ``win_rate``,
    ``expectancy``, ``eligible``, and ``promoted``.
    """
    sql = """
        INSERT INTO shadow_evaluations (
            run_timestamp, ticker, closed_count, win_rate,
            expectancy, eligible, promoted
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    conn = get_connection()
    try:
        count = 0
        for row in rows:
            conn.execute(
                sql,
                (
                    run_timestamp,
                    str(row["ticker"]),
                    int(row["closed_count"]),
                    row["win_rate"],
                    row["expectancy"],
                    1 if row["eligible"] else 0,
                    1 if row["promoted"] else 0,
                ),
            )
            count += 1
        conn.commit()
        return count
    finally:
        conn.close()


def get_shadow_evaluations(limit: int = 100) -> list[dict[str, Any]]:
    """Return recent shadow_evaluations rows, newest run first."""
    sql = (
        "SELECT * FROM shadow_evaluations "
        "ORDER BY run_timestamp DESC, expectancy DESC LIMIT ?"
    )
    conn = get_connection()
    try:
        rows = conn.execute(sql, (limit,)).fetchall()
    finally:
        conn.close()
    return [
        {
            "run_timestamp": str(r["run_timestamp"]),
            "ticker": str(r["ticker"]),
            "closed_count": int(r["closed_count"]),
            "win_rate": r["win_rate"],
            "expectancy": r["expectancy"],
            "eligible": bool(r["eligible"]),
            "promoted": bool(r["promoted"]),
        }
        for r in rows
    ]


# ---- option positions (Phase 13) ----


def _insert_option_position_row(conn: sqlite3.Connection, pos: OptionPosition) -> int:
    """Insert one option position on an OPEN connection; the caller commits.

    Split out so the eager writer and the atomic fill materializer share one
    column list and never drift apart.
    """
    if pos.option_type not in ("call", "put"):
        raise ValueError(f"Invalid option_type '{pos.option_type}'")
    if pos.outcome is not None and pos.outcome not in VALID_OUTCOMES:
        raise ValueError(
            f"Invalid outcome '{pos.outcome}' "
            f"(expected one of {sorted(VALID_OUTCOMES)})"
        )
    sql = """
        INSERT INTO option_positions (
            signal_id, order_id, symbol, underlying, option_type, strike, expiry,
            contracts, multiplier, premium_entry, delta_entry, theta, vega, gamma,
            tp, sl, deadline, opened_at, closed_at, exit_price, exit_reason,
            outcome, pnl_dollars, vehicle
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = (
        pos.signal_id,
        pos.order_id,
        pos.symbol,
        pos.underlying,
        pos.option_type,
        pos.strike,
        pos.expiry,
        pos.contracts,
        pos.multiplier,
        pos.premium_entry,
        pos.delta_entry,
        pos.theta,
        pos.vega,
        pos.gamma,
        pos.tp,
        pos.sl,
        pos.deadline.isoformat() if pos.deadline is not None else None,
        pos.opened_at.isoformat(),
        pos.closed_at.isoformat() if pos.closed_at is not None else None,
        pos.exit_price,
        pos.exit_reason,
        pos.outcome,
        pos.pnl_dollars,
        pos.vehicle,
    )
    cur = conn.execute(sql, params)
    new_id = cur.lastrowid
    if new_id is None:
        raise RuntimeError("INSERT did not return a row id")
    return new_id


def insert_option_position(pos: OptionPosition) -> int:
    """Insert a single-leg option position and return its id."""
    conn = get_connection()
    try:
        new_id = _insert_option_position_row(conn, pos)
        conn.commit()
        return new_id
    finally:
        conn.close()


def _row_to_option_position(row: sqlite3.Row) -> OptionPosition:
    return OptionPosition(
        symbol=str(row["symbol"]),
        underlying=str(row["underlying"]),
        option_type=str(row["option_type"]),
        strike=float(row["strike"]),
        expiry=str(row["expiry"]),
        contracts=float(row["contracts"]),
        opened_at=datetime.fromisoformat(row["opened_at"]),
        multiplier=int(row["multiplier"]),
        signal_id=row["signal_id"],
        order_id=row["order_id"],
        premium_entry=row["premium_entry"],
        delta_entry=row["delta_entry"],
        theta=row["theta"],
        vega=row["vega"],
        gamma=row["gamma"],
        tp=row["tp"],
        sl=row["sl"],
        deadline=(
            datetime.fromisoformat(row["deadline"]) if row["deadline"] else None
        ),
        closed_at=(
            datetime.fromisoformat(row["closed_at"]) if row["closed_at"] else None
        ),
        exit_price=row["exit_price"],
        exit_reason=row["exit_reason"],
        outcome=row["outcome"],
        pnl_dollars=row["pnl_dollars"],
        vehicle=str(row["vehicle"]),
        id=int(row["id"]),
    )


def get_open_option_positions() -> list[OptionPosition]:
    """Option positions with outcome NULL or 'open' — the exit watcher's book."""
    sql = (
        "SELECT * FROM option_positions "
        "WHERE outcome IS NULL OR outcome = 'open' "
        "ORDER BY opened_at ASC"
    )
    conn = get_connection()
    try:
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()
    return [_row_to_option_position(r) for r in rows]


def get_option_positions(limit: int = 100) -> list[OptionPosition]:
    """Recent option positions, newest first (reporting / CLI)."""
    sql = "SELECT * FROM option_positions ORDER BY opened_at DESC, id DESC LIMIT ?"
    conn = get_connection()
    try:
        rows = conn.execute(sql, (limit,)).fetchall()
    finally:
        conn.close()
    return [_row_to_option_position(r) for r in rows]


def update_option_position(position_id: int, **fields: Any) -> None:
    """Whitelisted partial update — used by the exit watcher when closing."""
    unknown = set(fields) - _OPTION_POSITION_UPDATABLE_FIELDS
    if unknown:
        raise ValueError(
            f"Unknown option_position field(s): {sorted(unknown)} "
            f"(allowed: {sorted(_OPTION_POSITION_UPDATABLE_FIELDS)})"
        )
    if not fields:
        return
    normalized: dict[str, Any] = {}
    for key, value in fields.items():
        normalized[key] = value.isoformat() if isinstance(value, datetime) else value
    if (
        "outcome" in normalized
        and normalized["outcome"] is not None
        and normalized["outcome"] not in VALID_OUTCOMES
    ):
        raise ValueError(f"Invalid outcome '{normalized['outcome']}'")
    set_clause = ", ".join(f"{k} = ?" for k in normalized)  # keys are whitelisted
    sql = f"UPDATE option_positions SET {set_clause} WHERE id = ?"  # noqa: S608
    params = [*normalized.values(), position_id]
    conn = get_connection()
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def get_option_position(position_id: int) -> OptionPosition | None:
    """Return one option position by row id, or None."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM option_positions WHERE id = ?", (position_id,)
        ).fetchone()
    finally:
        conn.close()
    return _row_to_option_position(row) if row is not None else None


def update_option_position_fill(
    pending_order_id: int,
    position_id: int,
    *,
    contracts: float,
    premium_entry: float,
) -> bool:
    """Advance a materialized option position to a larger cumulative fill.

    Separate from :func:`update_option_position` (the exit-watcher whitelist):
    only the fill materializer calls this, when a later cumulative broker
    snapshot raises contract count / weighted-average premium on an already
    linked position. Entry premium mirrors the broker average, never a quote.

    The compare-and-update runs under one write transaction so an older worker
    cannot overwrite a newer materialized quantity. Returns ``True`` only when
    the position advanced; an exact replay returns ``False``.
    """
    if contracts <= 0:
        raise ValueError("option position contracts must be positive")
    if premium_entry <= 0:
        raise ValueError("option position premium_entry must be positive")
    conn = get_connection()
    conn.isolation_level = None
    try:
        conn.execute("BEGIN IMMEDIATE")
        _guard_linked_pending_order_fill(
            conn,
            pending_order_id,
            "option",
            position_id,
            filled_qty=contracts,
            filled_avg_price=premium_entry,
        )
        row = conn.execute(
            "SELECT contracts, premium_entry FROM option_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        if row is None:
            raise ValueError("linked option position row is missing")
        current_contracts = float(row["contracts"])
        current_premium = row["premium_entry"]
        if contracts < current_contracts:
            raise ValueError(
                "cumulative fill regressed below the materialized holding "
                f"({contracts} < {current_contracts})"
            )
        if contracts == current_contracts:
            if current_premium is None or float(current_premium) != premium_entry:
                raise ValueError(
                    "cumulative quantity unchanged but average fill price diverged"
                )
            conn.commit()
            return False
        conn.execute(
            "UPDATE option_positions SET contracts = ?, premium_entry = ? WHERE id = ?",
            (contracts, premium_entry, position_id),
        )
        conn.commit()
        return True
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---- long-term positions (Phase 14) ----


def _insert_long_term_position_row(
    conn: sqlite3.Connection, pos: LongTermPosition,
) -> int:
    """Insert one long-term position on an OPEN connection; the caller commits.

    Split out so the eager writer and the atomic fill materializer share one
    column list and never drift apart.
    """
    if pos.status not in ("open", "closed"):
        raise ValueError(f"Invalid status '{pos.status}'")
    if pos.source not in VALID_POSITION_SOURCES:
        raise ValueError(
            f"Invalid source '{pos.source}' "
            f"(expected one of {sorted(VALID_POSITION_SOURCES)})"
        )
    if pos.direction not in VALID_POSITION_DIRECTIONS:
        raise ValueError(
            f"Invalid direction '{pos.direction}' "
            f"(expected one of {sorted(VALID_POSITION_DIRECTIONS)})"
        )
    sql = """
        INSERT INTO long_term_positions (
            ticker, asset_class, entry_price, entry_date, qty, status,
            exit_price, exit_date, exit_reason, source, direction, tp, sl,
            deadline
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = (
        pos.ticker,
        pos.asset_class,
        pos.entry_price,
        pos.entry_date.isoformat(),
        pos.qty,
        pos.status,
        pos.exit_price,
        pos.exit_date.isoformat() if pos.exit_date is not None else None,
        pos.exit_reason,
        pos.source,
        pos.direction,
        pos.tp,
        pos.sl,
        pos.deadline.isoformat() if pos.deadline is not None else None,
    )
    cur = conn.execute(sql, params)
    new_id = cur.lastrowid
    if new_id is None:
        raise RuntimeError("INSERT did not return a row id")
    return new_id


def insert_long_term_position(pos: LongTermPosition) -> int:
    """Insert a position into the long-term lifecycle book and return its id."""
    conn = get_connection()
    try:
        new_id = _insert_long_term_position_row(conn, pos)
        conn.commit()
        return new_id
    finally:
        conn.close()


def _row_to_long_term_position(row: sqlite3.Row) -> LongTermPosition:
    return LongTermPosition(
        ticker=str(row["ticker"]),
        asset_class=str(row["asset_class"]),
        entry_price=float(row["entry_price"]),
        entry_date=datetime.fromisoformat(row["entry_date"]),
        qty=float(row["qty"]),
        status=str(row["status"]),
        exit_price=row["exit_price"],
        exit_date=(
            datetime.fromisoformat(row["exit_date"]) if row["exit_date"] else None
        ),
        exit_reason=row["exit_reason"],
        source=str(row["source"]),
        direction=str(row["direction"]),
        tp=row["tp"],
        sl=row["sl"],
        deadline=(
            datetime.fromisoformat(row["deadline"]) if row["deadline"] else None
        ),
        id=int(row["id"]),
    )


def get_open_long_term_positions() -> list[LongTermPosition]:
    """Open buy-and-hold positions — the protective exit watcher's book."""
    sql = (
        "SELECT * FROM long_term_positions WHERE status = 'open' "
        "ORDER BY entry_date ASC"
    )
    conn = get_connection()
    try:
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()
    return [_row_to_long_term_position(r) for r in rows]


def get_long_term_positions(limit: int = 100) -> list[LongTermPosition]:
    """Recent buy-and-hold positions, newest first (reporting / CLI)."""
    sql = (
        "SELECT * FROM long_term_positions "
        "ORDER BY entry_date DESC, id DESC LIMIT ?"
    )
    conn = get_connection()
    try:
        rows = conn.execute(sql, (limit,)).fetchall()
    finally:
        conn.close()
    return [_row_to_long_term_position(r) for r in rows]


def update_long_term_position(position_id: int, **fields: Any) -> None:
    """Whitelisted partial update — used by the protective exit watcher."""
    unknown = set(fields) - _LONG_TERM_UPDATABLE_FIELDS
    if unknown:
        raise ValueError(
            f"Unknown long_term_position field(s): {sorted(unknown)} "
            f"(allowed: {sorted(_LONG_TERM_UPDATABLE_FIELDS)})"
        )
    if not fields:
        return
    normalized: dict[str, Any] = {}
    for key, value in fields.items():
        normalized[key] = value.isoformat() if isinstance(value, datetime) else value
    if (
        "status" in normalized
        and normalized["status"] not in ("open", "closed")
    ):
        raise ValueError(f"Invalid status '{normalized['status']}'")
    set_clause = ", ".join(f"{k} = ?" for k in normalized)  # keys are whitelisted
    sql = f"UPDATE long_term_positions SET {set_clause} WHERE id = ?"  # noqa: S608
    params = [*normalized.values(), position_id]
    conn = get_connection()
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def get_long_term_position(position_id: int) -> LongTermPosition | None:
    """Return one long-term position by row id, or None."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM long_term_positions WHERE id = ?", (position_id,)
        ).fetchone()
    finally:
        conn.close()
    return _row_to_long_term_position(row) if row is not None else None


def update_long_term_position_fill(
    pending_order_id: int,
    position_id: int,
    *,
    qty: float,
    entry_price: float,
) -> bool:
    """Advance a materialized long-term position to a larger cumulative fill.

    Separate from :func:`update_long_term_position` (the exit-watcher
    whitelist): only the fill materializer calls this, when a later cumulative
    broker snapshot raises share quantity / weighted-average entry on an already
    linked position. Entry price mirrors the broker average, never a quote.

    The compare-and-update runs under one write transaction so an older worker
    cannot overwrite a newer materialized quantity. Returns ``True`` only when
    the position advanced; an exact replay returns ``False``.
    """
    if qty <= 0:
        raise ValueError("long-term position qty must be positive")
    if entry_price <= 0:
        raise ValueError("long-term position entry_price must be positive")
    conn = get_connection()
    conn.isolation_level = None
    try:
        conn.execute("BEGIN IMMEDIATE")
        _guard_linked_pending_order_fill(
            conn,
            pending_order_id,
            "long_term",
            position_id,
            filled_qty=qty,
            filled_avg_price=entry_price,
        )
        row = conn.execute(
            "SELECT qty, entry_price FROM long_term_positions WHERE id = ?",
            (position_id,),
        ).fetchone()
        if row is None:
            raise ValueError("linked long-term position row is missing")
        current_qty = float(row["qty"])
        current_entry = float(row["entry_price"])
        if qty < current_qty:
            raise ValueError(
                "cumulative fill regressed below the materialized holding "
                f"({qty} < {current_qty})"
            )
        if qty == current_qty:
            if current_entry != entry_price:
                raise ValueError(
                    "cumulative quantity unchanged but average fill price diverged"
                )
            conn.commit()
            return False
        conn.execute(
            "UPDATE long_term_positions SET qty = ?, entry_price = ? WHERE id = ?",
            (qty, entry_price, position_id),
        )
        conn.commit()
        return True
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---- equity snapshots (Phase 15) ----


def insert_equity_snapshot(equity: float, captured_at: datetime) -> int:
    """Record one account-equity observation. Returns the row id."""
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO equity_snapshots (captured_at, equity) VALUES (?, ?)",
            (captured_at.isoformat(), equity),
        )
        conn.commit()
        new_id = cur.lastrowid
        if new_id is None:
            raise RuntimeError("INSERT did not return a row id")
        return new_id
    finally:
        conn.close()


def get_equity_peak() -> float | None:
    """The running maximum of recorded equity (the drawdown peak), or None."""
    conn = get_connection()
    try:
        row = conn.execute("SELECT MAX(equity) AS peak FROM equity_snapshots").fetchone()
    finally:
        conn.close()
    if row is None or row["peak"] is None:
        return None
    return float(row["peak"])


def get_latest_equity() -> float | None:
    """The most recently recorded equity, or None if never recorded."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT equity FROM equity_snapshots "
            "ORDER BY captured_at DESC, id DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    return float(row["equity"]) if row is not None else None


def get_recent_resolved_outcomes(limit: int = 50) -> list[str]:
    """The most recent REAL-EXECUTION resolved outcomes, newest first.

    The same canonical resolved definition every gating subsystem uses
    (``outcome IN ('win','loss')`` — expired/open excluded), ordered by
    ``closed_at`` so the Phase 15 consecutive-loss streak reads the true
    trailing run. Scope (Phase 23, mirroring Phase 18's reconciliation fix):
    active track only — shadow losses are not real losses — AND the data-only
    crypto swing signal types are excluded entirely. Those fire with
    ``track_mode='active'`` but can NEVER route to execution (the Phase 14
    permanent boundary), so their hourly paper losses must not pause real
    swing/long-term entry via the Tier 1 breaker.
    """
    # The interpolation is literal '?' placeholders only; values bind below.
    data_only = sorted(config.DATA_ONLY_SIGNAL_TYPES)
    placeholders = ", ".join("?" for _ in data_only)
    sql = (
        "SELECT trades.outcome FROM trades "
        "JOIN signals ON signals.id = trades.signal_id "
        "WHERE trades.outcome IN ('win','loss') "
        "  AND trades.track_mode = 'active' "
        "  AND trades.closed_at IS NOT NULL "
        f"  AND signals.signal_type NOT IN ({placeholders}) "  # noqa: S608
        "ORDER BY trades.closed_at DESC, trades.id DESC LIMIT ?"
    )
    conn = get_connection()
    try:
        rows = conn.execute(sql, [*data_only, limit]).fetchall()
    finally:
        conn.close()
    return [str(r["outcome"]) for r in rows]


# ---- plan executions (Phase 16) ----


def insert_plan_execution(execution: PlanExecution) -> int:
    """Insert one plan-execution audit row and return its id."""
    if execution.status not in VALID_PLAN_EXECUTION_STATUSES:
        raise ValueError(
            f"Invalid plan-execution status '{execution.status}' "
            f"(expected one of {sorted(VALID_PLAN_EXECUTION_STATUSES)})"
        )
    sql = """
        INSERT INTO plan_executions (
            plan_id, executed_at, ticker, pool, signal_type, side, qty,
            vehicle, status, order_ref, reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = (
        execution.plan_id,
        execution.executed_at.isoformat(),
        execution.ticker,
        execution.pool,
        execution.signal_type,
        execution.side,
        execution.qty,
        execution.vehicle,
        execution.status,
        execution.order_ref,
        execution.reason,
    )
    conn = get_connection()
    try:
        cur = conn.execute(sql, params)
        conn.commit()
        new_id = cur.lastrowid
        if new_id is None:
            raise RuntimeError("INSERT did not return a row id")
        return new_id
    finally:
        conn.close()


def _row_to_plan_execution(row: sqlite3.Row) -> PlanExecution:
    return PlanExecution(
        plan_id=str(row["plan_id"]),
        executed_at=datetime.fromisoformat(row["executed_at"]),
        ticker=str(row["ticker"]),
        pool=str(row["pool"]),
        status=str(row["status"]),
        signal_type=row["signal_type"],
        side=row["side"],
        qty=row["qty"],
        vehicle=row["vehicle"],
        order_ref=row["order_ref"],
        reason=str(row["reason"] or ""),
        id=int(row["id"]),
    )


def get_plan_executions(limit: int = 100) -> list[PlanExecution]:
    """Recent plan-execution audit rows, newest first (reporting / CLI)."""
    sql = "SELECT * FROM plan_executions ORDER BY executed_at DESC, id DESC LIMIT ?"
    conn = get_connection()
    try:
        rows = conn.execute(sql, (limit,)).fetchall()
    finally:
        conn.close()
    return [_row_to_plan_execution(r) for r in rows]


def has_submitted_plan_execution(ticker: str, pool: str, since: datetime) -> bool:
    """True if a SUBMITTED plan-execution row exists for (ticker, pool) at/after
    ``since`` — the Phase 16 idempotency guard's question. Only 'submitted'
    blocks: a rejected/error/skipped attempt may be retried."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT 1 FROM plan_executions "
            "WHERE ticker = ? AND pool = ? AND status = 'submitted' "
            "  AND executed_at >= ? LIMIT 1",
            (ticker, pool, since.isoformat()),
        ).fetchone()
    finally:
        conn.close()
    return row is not None


# ---- live candidate sourcing (Phase 17) ----


def get_actionable_signals(since: datetime) -> list[dict[str, Any]]:
    """Fired signals that are still plannable, newest first (Phase 17).

    The three boundaries are enforced BY THE QUERY, not left to downstream
    checks: (1) fired at/after ``since`` (the recency window); (2) never
    considered (``considered_at IS NULL``); (3) not shadow-tracked, and the
    paper-tracking trade — when one exists — is still open (a resolved trade
    means the move already played out). Signals with NO trade row (the
    long-term entries, which are tracked in ``long_term_positions`` instead)
    are included via the LEFT JOIN.

    Each dict carries the signal columns plus the joined trade's fire-time
    advisory context (``track_mode``, ``sentiment_score``, ``ind_*``) so the
    candidate mapper needs no second query.
    """
    sql = (
        "SELECT signals.*, trades.track_mode AS trade_track_mode, "
        "       trades.sentiment_score AS trade_sentiment_score, "
        "       trades.ind_rsi AS trade_ind_rsi, "
        "       trades.ind_adx AS trade_ind_adx, "
        "       trades.ind_obv AS trade_ind_obv, "
        "       trades.ind_vol_regime AS trade_ind_vol_regime, "
        "       trades.ind_concentration AS trade_ind_concentration "
        "FROM signals "
        "LEFT JOIN trades ON trades.signal_id = signals.id "
        "WHERE signals.timestamp >= ? "
        "  AND signals.considered_at IS NULL "
        "  AND (trades.id IS NULL "
        "       OR (trades.track_mode = 'active' "
        "           AND (trades.outcome IS NULL OR trades.outcome = 'open'))) "
        "ORDER BY signals.timestamp DESC, signals.id DESC"
    )
    conn = get_connection()
    try:
        rows = conn.execute(sql, (since.isoformat(),)).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def mark_signals_considered(
    signal_ids: Sequence[int], considered_at: datetime,
) -> int:
    """Stamp ``considered_at`` on the given signals; returns the rows updated.

    Only NULL rows are stamped (the first consideration wins — the timestamp
    records when the signal entered a plan, never when it was re-seen).
    """
    ids = [int(i) for i in signal_ids]
    if not ids:
        return 0
    # The interpolation is literal '?' placeholders only; values bind below.
    placeholders = ", ".join("?" for _ in ids)
    sql = (
        "UPDATE signals SET considered_at = ? "
        f"WHERE id IN ({placeholders}) AND considered_at IS NULL"  # noqa: S608
    )
    conn = get_connection()
    try:
        cur = conn.execute(sql, [considered_at.isoformat(), *ids])
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


# ---- durable pending orders (cross-cutting order-lifecycle audit) ----


def _is_json_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_optional_json_number(
    payload: Mapping[str, Any], field: str,
) -> None:
    value = payload[field]
    if value is not None and not _is_json_number(value):
        raise ValueError(f"pending-order intent field '{field}' must be numeric or null")


def _validate_pending_order_payload(order: PendingOrder) -> None:
    """Validate the supported immutable materialization-payload contract."""
    expected_version = (
        PENDING_ORDER_POSITION_EXIT_INTENT_VERSION
        if order.order_role == "exit"
        else PENDING_ORDER_INTENT_VERSION
    )
    if order.intent_payload_version != expected_version:
        raise ValueError(
            "Unsupported pending-order intent payload version "
            f"{order.intent_payload_version}; expected {expected_version}"
        )
    try:
        payload = json.loads(order.intent_payload_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("pending-order intent payload must be valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("pending-order intent payload must be a JSON object")

    intent_kind = payload.get("intent_kind")
    if intent_kind not in VALID_PENDING_ORDER_INTENT_KINDS:
        raise ValueError(
            f"Invalid pending-order intent_kind '{intent_kind}' "
            f"(expected one of {sorted(VALID_PENDING_ORDER_INTENT_KINDS)})"
        )

    if order.order_role == "exit":
        if intent_kind != "position_exit":
            raise ValueError("exit-role pending order requires position_exit intent")
        exit_reason = payload.get("exit_reason")
        if not isinstance(exit_reason, str) or not exit_reason.strip():
            raise ValueError(
                "position_exit intent requires a non-empty exit_reason"
            )
        return
    if intent_kind == "position_exit":
        raise ValueError("position_exit intent requires an exit-role pending order")

    if intent_kind == "option":
        if order.target_position_kind != "option" or order.vehicle not in (
            "option_full", "option_undersized",
        ):
            raise ValueError("option intent requires an option target and option vehicle")
        if order.side != "buy":
            raise ValueError("option intent must be a buy order")
        required = {
            "option_type", "strike", "expiry", "multiplier", "delta_entry",
            "theta", "vega", "gamma", "tp", "sl", "deadline",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"option intent payload missing fields: {missing}")
        if payload["option_type"] not in ("call", "put"):
            raise ValueError("option intent option_type must be 'call' or 'put'")
        if not _is_json_number(payload["strike"]) or payload["strike"] <= 0:
            raise ValueError("option intent strike must be positive")
        if (
            not isinstance(payload["multiplier"], int)
            or isinstance(payload["multiplier"], bool)
            or payload["multiplier"] <= 0
        ):
            raise ValueError("option intent multiplier must be a positive integer")
        try:
            date.fromisoformat(str(payload["expiry"]))
        except ValueError as exc:
            raise ValueError("option intent expiry must be an ISO date") from exc
        for field in ("delta_entry", "theta", "vega", "gamma", "tp", "sl"):
            _validate_optional_json_number(payload, field)
    else:
        if order.target_position_kind != "long_term" or order.vehicle != "shares":
            raise ValueError("share intent requires a long_term target and shares vehicle")
        required = {"source", "direction"}
        if intent_kind == "shares_fallback":
            required.update({"tp", "sl", "deadline"})
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"share intent payload missing fields: {missing}")
        expected_source = (
            "long_term" if intent_kind == "long_term" else "swing_fallback"
        )
        if payload["source"] != expected_source:
            raise ValueError(
                f"{intent_kind} intent source must be '{expected_source}'"
            )
        if payload["direction"] not in ("long", "short"):
            raise ValueError("share intent direction must be 'long' or 'short'")
        expected_direction = "long" if order.side == "buy" else "short"
        if payload["direction"] != expected_direction:
            raise ValueError("share intent direction must match the submitted side")
        if intent_kind == "long_term" and order.side != "buy":
            raise ValueError("long_term intent must be a buy/long order")
        if intent_kind == "shares_fallback":
            for field in ("tp", "sl"):
                _validate_optional_json_number(payload, field)

    deadline = payload.get("deadline")
    if deadline is not None:
        if not isinstance(deadline, str):
            raise ValueError("pending-order intent deadline must be an ISO datetime or null")
        try:
            datetime.fromisoformat(deadline)
        except ValueError as exc:
            raise ValueError(
                "pending-order intent deadline must be an ISO datetime or null"
            ) from exc


def _resolved_pending_order_client_id(order: PendingOrder) -> str:
    """Return durable identity, synthesizing only for the legacy bound path."""
    if order.client_order_id is not None:
        if not isinstance(order.client_order_id, str) or not order.client_order_id.strip():
            raise ValueError("pending-order client_order_id must not be empty")
        if not is_recoverable_pending_order_client_id(order.client_order_id):
            raise ValueError("pending-order legacy client_order_id namespace is reserved")
        return order.client_order_id
    if (
        order.broker_order_id is not None
        and order.lifecycle_status not in {"prepared", "abandoned"}
    ):
        return (
            f"{LEGACY_PENDING_ORDER_CLIENT_ID_PREFIX}broker-"
            f"{order.broker_order_id}"
        )
    raise ValueError("pending-order client_order_id must be explicit before submit")


def _validate_pending_order(
    order: PendingOrder,
) -> tuple[str, datetime | None, str | None]:
    for field, value in (("ticker", order.ticker), ("broker_symbol", order.broker_symbol)):
        if not value.strip():
            raise ValueError(f"pending-order {field} must not be empty")
    if order.asset_class not in VALID_ASSET_CLASSES:
        raise ValueError(f"Invalid pending-order asset_class '{order.asset_class}'")
    if order.vehicle not in VALID_PENDING_ORDER_VEHICLES:
        raise ValueError(f"Invalid pending-order vehicle '{order.vehicle}'")
    if order.target_position_kind not in VALID_PENDING_ORDER_POSITION_KINDS:
        raise ValueError(
            f"Invalid pending-order target position kind '{order.target_position_kind}'"
        )
    if order.order_role not in VALID_PENDING_ORDER_ROLES:
        raise ValueError(f"Invalid pending-order role '{order.order_role}'")
    if order.order_role == "entry":
        if (
            order.closes_position_kind is not None
            or order.closes_position_id is not None
        ):
            raise ValueError("entry pending order cannot carry a close target")
    else:
        if order.closes_position_kind not in VALID_PENDING_ORDER_POSITION_KINDS:
            raise ValueError("exit pending order requires a valid close target kind")
        if order.closes_position_id is None or order.closes_position_id <= 0:
            raise ValueError("exit pending order requires a positive close target id")
        if order.closes_position_kind != order.target_position_kind:
            raise ValueError("exit pending-order close target must match its target kind")
        if order.target_position_kind == "option":
            if order.vehicle not in ("option_full", "option_undersized"):
                raise ValueError("option exit requires an option vehicle")
            if order.side != "sell":
                raise ValueError("option exit must be a sell order")
        elif order.vehicle != "shares":
            raise ValueError("long_term exit requires the shares vehicle")
    if order.side not in VALID_PENDING_ORDER_SIDES:
        raise ValueError(f"Invalid pending-order side '{order.side}'")
    if order.requested_qty <= 0:
        raise ValueError("pending-order requested_qty must be positive")
    if order.requested_limit_price is not None and order.requested_limit_price <= 0:
        raise ValueError("pending-order requested_limit_price must be positive")
    if order.signal_id is not None and order.signal_id <= 0:
        raise ValueError("pending-order signal_id must be positive")
    if order.lifecycle_status not in VALID_PENDING_ORDER_STATUSES:
        raise ValueError(f"Invalid pending-order status '{order.lifecycle_status}'")
    if order.broker_order_id is not None and (
        not isinstance(order.broker_order_id, str)
        or not order.broker_order_id.strip()
    ):
        raise ValueError("pending-order broker_order_id must not be empty")
    client_order_id = _resolved_pending_order_client_id(order)
    is_unbound_status = order.lifecycle_status in {"prepared", "abandoned"}
    if is_unbound_status != (order.broker_order_id is None):
        raise ValueError(
            "prepared/abandoned pending orders must be unbound; "
            "all broker lifecycle rows must be bound"
        )
    if order.filled_qty < 0:
        raise ValueError("pending-order filled_qty must not be negative")
    has_fill_price = (
        order.filled_avg_price is not None and order.filled_avg_price > 0
    )
    if (order.filled_qty > 0) != has_fill_price:
        raise ValueError(
            "pending-order filled_qty and positive filled_avg_price must appear together"
        )
    last_fill_at = order.last_fill_at
    last_fill_time_source = order.last_fill_time_source
    if (
        order.filled_qty > 0
        and last_fill_at is None
        and last_fill_time_source is None
        and order.last_refreshed_at is not None
    ):
        last_fill_at = order.last_refreshed_at
        last_fill_time_source = "observed"
    has_fill_time = last_fill_at is not None
    has_fill_time_source = last_fill_time_source is not None
    if order.filled_qty > 0:
        if not has_fill_time or not has_fill_time_source:
            raise ValueError(
                "positive pending-order fill requires last_fill_at and "
                "last_fill_time_source"
            )
        if last_fill_time_source not in VALID_PENDING_ORDER_FILL_TIME_SOURCES:
            raise ValueError(
                "Invalid pending-order last_fill_time_source "
                f"'{last_fill_time_source}'"
            )
    elif has_fill_time or has_fill_time_source:
        raise ValueError("zero-fill pending order cannot carry fill-time metadata")
    if order.fees_dollars is not None:
        if order.filled_qty == 0:
            raise ValueError("zero-fill pending order cannot carry fees")
        if order.fees_dollars < 0:
            raise ValueError("pending-order fees_dollars must not be negative")
    is_terminal = order.lifecycle_status in TERMINAL_PENDING_ORDER_STATUSES
    has_terminal = (
        order.terminal_at is not None
        and order.terminal_reason is not None
        and bool(order.terminal_reason.strip())
    )
    if is_terminal != has_terminal:
        raise ValueError(
            "terminal pending-order status requires terminal_reason and terminal_at"
        )
    if not is_terminal and (
        order.terminal_at is not None or order.terminal_reason is not None
    ):
        raise ValueError("nonterminal pending order cannot carry terminal fields")
    if order.lifecycle_status in {"prepared", "abandoned"}:
        if order.broker_status is not None:
            raise ValueError(
                "prepared/abandoned pending order cannot carry broker status"
            )
        if order.filled_qty != 0 or order.filled_avg_price is not None:
            raise ValueError("prepared/abandoned pending order cannot carry a fill")
        if order.position_kind is not None or order.position_id is not None:
            raise ValueError("prepared/abandoned pending order cannot link a position")
    if order.lifecycle_status == "prepared" and order.last_refreshed_at is not None:
        raise ValueError("prepared pending order cannot have a broker refresh time")
    has_position_link = order.position_kind is not None or order.position_id is not None
    if has_position_link:
        if order.order_role != "entry":
            raise ValueError(
                "exit pending order cannot use the entry-materialization link"
            )
        if order.position_kind not in VALID_PENDING_ORDER_POSITION_KINDS:
            raise ValueError(f"Invalid pending-order position kind '{order.position_kind}'")
        if order.position_id is None or order.position_id <= 0:
            raise ValueError("pending-order position_id must be positive when linked")
        if order.position_kind != order.target_position_kind:
            raise ValueError("pending-order position kind must match its target")
        if order.filled_qty <= 0 or not has_fill_price:
            raise ValueError("pending order cannot link a position before a usable fill")
    _validate_pending_order_payload(order)
    return client_order_id, last_fill_at, last_fill_time_source


def insert_pending_order(order: PendingOrder) -> int:
    """Insert one prepared/accepted immutable intent and return its ledger id."""
    client_order_id, last_fill_at, last_fill_time_source = (
        _validate_pending_order(order)
    )
    sql = """
        INSERT INTO pending_orders (
            client_order_id, broker_order_id, order_role, ticker, broker_symbol,
            asset_class, vehicle, target_position_kind, closes_position_kind,
            closes_position_id, side, requested_qty, requested_limit_price,
            submitted_at, signal_id, intent_payload_version, intent_payload_json,
            lifecycle_status, broker_status, filled_qty, filled_avg_price,
            last_fill_at, last_fill_time_source, fees_dollars, last_refreshed_at,
            terminal_reason, terminal_at, position_kind, position_id
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?
        )
    """
    params = (
        client_order_id,
        order.broker_order_id,
        order.order_role,
        order.ticker,
        order.broker_symbol,
        order.asset_class,
        order.vehicle,
        order.target_position_kind,
        order.closes_position_kind,
        order.closes_position_id,
        order.side,
        order.requested_qty,
        order.requested_limit_price,
        order.submitted_at.isoformat(),
        order.signal_id,
        order.intent_payload_version,
        order.intent_payload_json,
        order.lifecycle_status,
        order.broker_status,
        order.filled_qty,
        order.filled_avg_price,
        last_fill_at.isoformat() if last_fill_at is not None else None,
        last_fill_time_source,
        order.fees_dollars,
        (
            order.last_refreshed_at.isoformat()
            if order.last_refreshed_at is not None else None
        ),
        order.terminal_reason,
        order.terminal_at.isoformat() if order.terminal_at is not None else None,
        order.position_kind,
        order.position_id,
    )
    conn = get_connection()
    try:
        cur = conn.execute(sql, params)
        conn.commit()
        new_id = cur.lastrowid
        if new_id is None:
            raise RuntimeError("INSERT did not return a row id")
        return new_id
    finally:
        conn.close()


def _row_to_pending_order(row: sqlite3.Row) -> PendingOrder:
    return PendingOrder(
        client_order_id=str(row["client_order_id"]),
        broker_order_id=(
            str(row["broker_order_id"])
            if row["broker_order_id"] is not None else None
        ),
        order_role=str(row["order_role"]),
        ticker=str(row["ticker"]),
        broker_symbol=str(row["broker_symbol"]),
        asset_class=str(row["asset_class"]),
        vehicle=str(row["vehicle"]),
        target_position_kind=str(row["target_position_kind"]),
        closes_position_kind=row["closes_position_kind"],
        closes_position_id=row["closes_position_id"],
        side=str(row["side"]),
        requested_qty=float(row["requested_qty"]),
        requested_limit_price=row["requested_limit_price"],
        submitted_at=datetime.fromisoformat(row["submitted_at"]),
        intent_payload_json=str(row["intent_payload_json"]),
        signal_id=row["signal_id"],
        intent_payload_version=int(row["intent_payload_version"]),
        lifecycle_status=str(row["lifecycle_status"]),
        broker_status=row["broker_status"],
        filled_qty=float(row["filled_qty"]),
        filled_avg_price=row["filled_avg_price"],
        last_fill_at=(
            datetime.fromisoformat(row["last_fill_at"])
            if row["last_fill_at"] else None
        ),
        last_fill_time_source=row["last_fill_time_source"],
        fees_dollars=row["fees_dollars"],
        last_refreshed_at=(
            datetime.fromisoformat(row["last_refreshed_at"])
            if row["last_refreshed_at"] else None
        ),
        terminal_reason=row["terminal_reason"],
        terminal_at=(
            datetime.fromisoformat(row["terminal_at"]) if row["terminal_at"] else None
        ),
        position_kind=row["position_kind"],
        position_id=row["position_id"],
        id=int(row["id"]),
    )


def get_pending_order(broker_order_id: str | None) -> PendingOrder | None:
    """Return the unique ledger row for a broker order id, if present."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM pending_orders WHERE broker_order_id = ?",
            (broker_order_id,),
        ).fetchone()
    finally:
        conn.close()
    return _row_to_pending_order(row) if row is not None else None


def get_pending_order_by_client_order_id(
    client_order_id: str,
) -> PendingOrder | None:
    """Return one durable intent by its unique client order identity."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM pending_orders WHERE client_order_id = ?",
            (client_order_id,),
        ).fetchone()
    finally:
        conn.close()
    return _row_to_pending_order(row) if row is not None else None


def get_pending_orders(limit: int = 100) -> list[PendingOrder]:
    """Recent pending-order ledger rows, newest submission first."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM pending_orders "
            "ORDER BY submitted_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_pending_order(row) for row in rows]


def get_nonterminal_pending_orders() -> list[PendingOrder]:
    """Orders whose broker lifecycle still needs a future refresh."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM pending_orders WHERE terminal_at IS NULL "
            "ORDER BY submitted_at ASC, id ASC"
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_pending_order(row) for row in rows]


def get_pending_exit_orders_for_position(
    position_kind: str, position_id: int,
) -> list[PendingOrder]:
    """Return every exit attempt for one immutable typed close target."""
    if position_kind not in VALID_PENDING_ORDER_POSITION_KINDS:
        raise ValueError(f"Invalid pending-order position kind '{position_kind}'")
    if position_id <= 0:
        raise ValueError("pending-order position_id must be positive")
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM pending_orders "
            "WHERE order_role = 'exit' AND closes_position_kind = ? "
            "AND closes_position_id = ? ORDER BY submitted_at ASC, id ASC",
            (position_kind, position_id),
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_pending_order(row) for row in rows]


def get_pending_entry_orders_for_position(
    position_kind: str, position_id: int,
) -> list[PendingOrder]:
    """Return every entry row linked to one materialized typed position."""
    if position_kind not in VALID_PENDING_ORDER_POSITION_KINDS:
        raise ValueError(f"Invalid pending-order position kind '{position_kind}'")
    if position_id <= 0:
        raise ValueError("pending-order position_id must be positive")
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM pending_orders "
            "WHERE order_role = 'entry' AND position_kind = ? "
            "AND position_id = ? ORDER BY submitted_at ASC, id ASC",
            (position_kind, position_id),
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_pending_order(row) for row in rows]


def update_pending_order(pending_order_id: int, **fields: Any) -> None:
    """Bind once, or update cumulative broker truth and the position link."""
    unknown = set(fields) - _PENDING_ORDER_UPDATABLE_FIELDS
    if unknown:
        raise ValueError(
            f"Unknown pending_order field(s): {sorted(unknown)} "
            f"(allowed: {sorted(_PENDING_ORDER_UPDATABLE_FIELDS)})"
        )
    if not fields:
        return
    normalized: dict[str, Any] = {}
    for key, value in fields.items():
        normalized[key] = value.isoformat() if isinstance(value, datetime) else value
    if (
        "broker_order_id" in normalized
        and normalized["broker_order_id"] is not None
        and (
            not isinstance(normalized["broker_order_id"], str)
            or not normalized["broker_order_id"].strip()
        )
    ):
        raise ValueError("pending-order broker_order_id must not be empty")
    if (
        "lifecycle_status" in normalized
        and normalized["lifecycle_status"] not in VALID_PENDING_ORDER_STATUSES
    ):
        raise ValueError(
            f"Invalid pending-order status '{normalized['lifecycle_status']}'"
        )
    if "filled_qty" in normalized and normalized["filled_qty"] < 0:
        raise ValueError("pending-order filled_qty must not be negative")
    if (
        "filled_avg_price" in normalized
        and normalized["filled_avg_price"] is not None
        and normalized["filled_avg_price"] <= 0
    ):
        raise ValueError("pending-order filled_avg_price must be positive")
    if (
        "last_fill_time_source" in normalized
        and normalized["last_fill_time_source"] is not None
        and normalized["last_fill_time_source"]
        not in VALID_PENDING_ORDER_FILL_TIME_SOURCES
    ):
        raise ValueError(
            "Invalid pending-order last_fill_time_source "
            f"'{normalized['last_fill_time_source']}'"
        )
    if (
        "fees_dollars" in normalized
        and normalized["fees_dollars"] is not None
        and normalized["fees_dollars"] < 0
    ):
        raise ValueError("pending-order fees_dollars must not be negative")
    if (
        "position_kind" in normalized
        and normalized["position_kind"] is not None
        and normalized["position_kind"] not in VALID_PENDING_ORDER_POSITION_KINDS
    ):
        raise ValueError(
            f"Invalid pending-order position kind '{normalized['position_kind']}'"
        )
    if (
        "position_id" in normalized
        and normalized["position_id"] is not None
        and normalized["position_id"] <= 0
    ):
        raise ValueError("pending-order position_id must be positive")
    conn = get_connection()
    try:
        current = conn.execute(
            "SELECT filled_qty FROM pending_orders WHERE id = ?",
            (pending_order_id,),
        ).fetchone()
        if (
            current is not None
            and "filled_qty" in normalized
            and normalized["filled_qty"] > float(current["filled_qty"])
            and "last_fill_at" not in normalized
            and "last_fill_time_source" not in normalized
        ):
            observed_at = normalized.get("last_refreshed_at")
            if observed_at is None:
                observed_at = datetime.now(UTC).isoformat()
            normalized["last_fill_at"] = observed_at
            normalized["last_fill_time_source"] = "observed"
        if ("last_fill_at" in normalized) != (
            "last_fill_time_source" in normalized
        ):
            raise ValueError(
                "last_fill_at and last_fill_time_source must be updated together"
            )
        set_clause = ", ".join(f"{key} = ?" for key in normalized)
        sql = f"UPDATE pending_orders SET {set_clause} WHERE id = ?"  # noqa: S608
        conn.execute(sql, [*normalized.values(), pending_order_id])
        conn.commit()
    finally:
        conn.close()


def get_pending_orders_with_fills() -> list[PendingOrder]:
    """Bound rows carrying a positive cumulative fill — materialization candidates.

    Includes terminal rows: a canceled/rejected/expired order that filled part
    of its quantity still owns a real partial position to materialize.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM pending_orders "
            "WHERE order_role = 'entry' AND filled_qty > 0 "
            "ORDER BY submitted_at ASC, id ASC"
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_pending_order(row) for row in rows]


def get_pending_exit_orders_with_fills() -> list[PendingOrder]:
    """Every exit row carrying cumulative execution truth, terminal or not."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM pending_orders "
            "WHERE order_role = 'exit' AND filled_qty > 0 "
            "ORDER BY submitted_at ASC, id ASC"
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_pending_order(row) for row in rows]


def _guard_unlinked_pending_order(
    conn: sqlite3.Connection,
    pending_order_id: int,
    position_kind: str,
    *,
    filled_qty: float,
    filled_avg_price: float,
) -> None:
    """Confirm, inside the write transaction, that this row may still link.

    Closes the read-check-write window: the position link is only inserted when
    the ledger row still exists, still targets ``position_kind``, and is not
    already linked. A concurrent or replayed materialization therefore cannot
    insert a second position.
    """
    row = conn.execute(
        "SELECT order_role, target_position_kind, position_id, filled_qty, "
        "filled_avg_price "
        "FROM pending_orders WHERE id = ?",
        (pending_order_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"pending order {pending_order_id} not found")
    if row["order_role"] != "entry":
        raise ValueError("exit pending order cannot materialize an entry position")
    if row["position_id"] is not None:
        raise ValueError("pending order already links a materialized position")
    if row["target_position_kind"] != position_kind:
        raise ValueError(
            "materialized position kind does not match the pending-order target"
        )
    if (
        float(row["filled_qty"]) != filled_qty
        or row["filled_avg_price"] is None
        or float(row["filled_avg_price"]) != filled_avg_price
    ):
        raise ValueError("pending-order fill changed before the position could link")


def _guard_linked_pending_order_fill(
    conn: sqlite3.Connection,
    pending_order_id: int,
    position_kind: str,
    position_id: int,
    *,
    filled_qty: float,
    filled_avg_price: float,
) -> None:
    """Verify a cumulative position update still matches durable ledger truth."""
    row = conn.execute(
        "SELECT order_role, position_kind, position_id, filled_qty, "
        "filled_avg_price "
        "FROM pending_orders WHERE id = ?",
        (pending_order_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"pending order {pending_order_id} not found")
    if row["order_role"] != "entry":
        raise ValueError("exit pending order cannot update an entry position")
    if row["position_kind"] != position_kind or row["position_id"] != position_id:
        raise ValueError("pending-order position link changed before the fill update")
    if (
        float(row["filled_qty"]) != filled_qty
        or row["filled_avg_price"] is None
        or float(row["filled_avg_price"]) != filled_avg_price
    ):
        raise ValueError("pending-order fill changed before the position could update")


def _link_pending_order_position(
    conn: sqlite3.Connection,
    pending_order_id: int,
    position_kind: str,
    position_id: int,
) -> None:
    """Bind the eventual position link on an OPEN transaction; caller commits."""
    cur = conn.execute(
        "UPDATE pending_orders SET position_kind = ?, position_id = ? WHERE id = ?",
        (position_kind, position_id, pending_order_id),
    )
    if cur.rowcount != 1:
        raise RuntimeError("pending-order position link did not update exactly one row")


def materialize_new_option_position(
    pending_order_id: int, pos: OptionPosition,
) -> int:
    """Atomically insert an option position and link its pending-order row.

    The insert and the ledger link commit together or roll back together, so a
    failed link can never leave an orphan position and a replay can never insert
    a second one. Returns the new position row id.
    """
    if pos.premium_entry is None:
        raise ValueError("option materialization requires a broker average fill price")
    conn = get_connection()
    conn.isolation_level = None
    try:
        conn.execute("BEGIN IMMEDIATE")
        _guard_unlinked_pending_order(
            conn,
            pending_order_id,
            "option",
            filled_qty=pos.contracts,
            filled_avg_price=pos.premium_entry,
        )
        position_id = _insert_option_position_row(conn, pos)
        _link_pending_order_position(conn, pending_order_id, "option", position_id)
        conn.commit()
        return position_id
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def adopt_legacy_option_position(
    pending_order_id: int,
    expected: OptionPosition,
) -> int | None:
    """Adopt one uniquely provable eager option row into the fill lifecycle.

    Schema-v24/v25 compatibility rows use synthetic ``legacy-*`` client IDs.
    Before the final lifecycle cutover, the option submit path also inserted an
    eager position carrying Alpaca's broker order ID.  That provider identity
    is strong enough to adopt only when exactly one matching position exists
    and its immutable contract identity agrees with the pending intent.

    The adoption, correction to actual cumulative fill truth, and ledger link
    are one transaction.  Missing, duplicate, conflicting, or already-linked
    candidates return ``None`` so callers can log/skip for manual review rather
    than guessing.  Long-term rows deliberately have no equivalent helper:
    their table carries no broker order identity.
    """
    if expected.order_id is None or not expected.order_id.strip():
        raise ValueError("legacy option adoption requires a broker order ID")
    if expected.premium_entry is None or expected.premium_entry <= 0:
        raise ValueError("legacy option adoption requires a broker fill price")
    if expected.contracts <= 0:
        raise ValueError("legacy option adoption requires a positive fill")

    conn = get_connection()
    conn.isolation_level = None
    try:
        conn.execute("BEGIN IMMEDIATE")
        _guard_unlinked_pending_order(
            conn,
            pending_order_id,
            "option",
            filled_qty=expected.contracts,
            filled_avg_price=expected.premium_entry,
        )
        candidates = conn.execute(
            "SELECT * FROM option_positions WHERE order_id = ?",
            (expected.order_id,),
        ).fetchall()
        if len(candidates) != 1:
            conn.commit()
            return None
        row = candidates[0]
        identity_matches = (
            row["symbol"] == expected.symbol
            and row["underlying"] == expected.underlying
            and row["option_type"] == expected.option_type
            and float(row["strike"]) == expected.strike
            and row["expiry"] == expected.expiry
            and int(row["multiplier"]) == expected.multiplier
            and row["signal_id"] == expected.signal_id
            and row["vehicle"] == expected.vehicle
        )
        if not identity_matches:
            conn.commit()
            return None
        position_id = int(row["id"])
        existing_link = conn.execute(
            "SELECT id FROM pending_orders "
            "WHERE position_kind = 'option' AND position_id = ? AND id != ?",
            (position_id, pending_order_id),
        ).fetchone()
        if existing_link is not None:
            conn.commit()
            return None
        conn.execute(
            "UPDATE option_positions SET contracts = ?, premium_entry = ? "
            "WHERE id = ?",
            (expected.contracts, expected.premium_entry, position_id),
        )
        _link_pending_order_position(
            conn, pending_order_id, "option", position_id
        )
        conn.commit()
        return position_id
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def materialize_new_long_term_position(
    pending_order_id: int, pos: LongTermPosition,
) -> int:
    """Atomically insert a long-term position and link its pending-order row.

    Same all-or-nothing guarantee as :func:`materialize_new_option_position`.
    Returns the new position row id.
    """
    conn = get_connection()
    conn.isolation_level = None
    try:
        conn.execute("BEGIN IMMEDIATE")
        _guard_unlinked_pending_order(
            conn,
            pending_order_id,
            "long_term",
            filled_qty=pos.qty,
            filled_avg_price=pos.entry_price,
        )
        position_id = _insert_long_term_position_row(conn, pos)
        _link_pending_order_position(
            conn, pending_order_id, "long_term", position_id
        )
        conn.commit()
        return position_id
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()
