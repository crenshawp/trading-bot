"""SQLite database layer — schema, CRUD, and helpers.

Only stdlib sqlite3 is used. Foreign keys ON, WAL journal mode, ISO 8601
datetime strings stored as TEXT (we manage adapters ourselves instead of
relying on sqlite3's deprecated defaults).
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any

from trading_bot import config
from trading_bot.models import (
    VALID_ASSET_CLASSES,
    VALID_DIRECTIONS,
    VALID_OUTCOMES,
    DailyPerf,
    Signal,
    Trade,
)

SCHEMA_VERSION = 1

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
"""

# Whitelist for update_trade — never interpolate user input into SQL column names.
_TRADE_UPDATABLE_FIELDS: frozenset[str] = frozenset(
    {"opened_at", "closed_at", "exit_price", "outcome", "pnl_pct", "pnl_dollars", "notes"}
)

_TABLE_NAMES: tuple[str, ...] = ("signals", "trades", "daily_performance")


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


def init_db() -> None:
    """Create the schema if absent. Idempotent."""
    conn = get_connection()
    try:
        conn.executescript(_SCHEMA_SQL)
        cur = conn.execute("SELECT version FROM schema_version LIMIT 1")
        if cur.fetchone() is None:
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
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
    sql = """
        INSERT INTO trades (
            signal_id, opened_at, closed_at, exit_price,
            outcome, pnl_pct, pnl_dollars, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
