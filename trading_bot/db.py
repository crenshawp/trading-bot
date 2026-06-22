"""SQLite database layer — schema, CRUD, and helpers.

Only stdlib sqlite3 is used. Foreign keys ON, WAL journal mode, ISO 8601
datetime strings stored as TEXT (we manage adapters ourselves instead of
relying on sqlite3's deprecated defaults).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from trading_bot import config
from trading_bot.models import (
    VALID_ASSET_CLASSES,
    VALID_DIRECTIONS,
    VALID_MARKET_REGIMES,
    VALID_OUTCOMES,
    VALID_PAIR_STATUSES,
    VALID_PREDICTION_DIRECTIONS,
    VALID_PREDICTION_OUTCOMES,
    VALID_TRACK_MODES,
    VALID_VIX_BANDS,
    VALID_WATCHLIST_STATUSES,
    DailyPerf,
    Prediction,
    Signal,
    Trade,
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
# Version 3 (Phase 2.2) — adds trades.vix_level + trades.vix_band + vix_snapshots.
# Version 2 (Phase 2.1) — adds trades.market_regime + regime_snapshots table.
SCHEMA_VERSION = 12

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
    earnings_risk INTEGER NOT NULL DEFAULT 0,
    news_risk INTEGER NOT NULL DEFAULT 0,
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
    "signal_pair_status", "signal_pair_transitions",
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
        int(signal.earnings_risk),
        int(signal.news_risk),
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
        earnings_risk=bool(row["earnings_risk"]),
        news_risk=bool(row["news_risk"]),
        raw_indicators_json=row["raw_indicators_json"],
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
            sentiment_score, sentiment_label, heavy_news, headline_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        int(trade.heavy_news),
        trade.headline_count,
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
        heavy_news=bool(row["heavy_news"]) if "heavy_news" in keys else False,
        headline_count=row["headline_count"] if "headline_count" in keys else None,
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
