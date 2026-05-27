"""Generic key/value settings persisted in SQLite.

Reusable for any toggleable feature. Phase 2.2b uses it for the
prediction engine's master switch, active window, and ticker list,
but future phases can register additional keys without touching this
module.

All values are stored as TEXT. Use the typed helpers (``get_bool`` /
``set_bool``) for booleans so casing variations like ``"True"`` /
``"true"`` / ``"1"`` parse consistently.
"""

from __future__ import annotations

from datetime import UTC, datetime

from trading_bot import db

# Accepted spellings for truthy/falsy values, lowercased before lookup.
_TRUTHY: frozenset[str] = frozenset({"true", "1", "yes", "on"})
_FALSY: frozenset[str] = frozenset({"false", "0", "no", "off"})


def get(key: str, default: str | None = None) -> str | None:
    """Return the value for ``key``, or ``default`` if absent."""
    conn = db.get_connection()
    try:
        row = conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return default
    return str(row["value"])


def set(key: str, value: str) -> None:  # noqa: A001 - module-level name, callers use settings.set
    """Insert or update the value for ``key``. Updates ``updated_at``."""
    now_iso = datetime.now(UTC).isoformat()
    sql = """
        INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value = excluded.value,
            updated_at = excluded.updated_at
    """
    conn = db.get_connection()
    try:
        conn.execute(sql, (key, value, now_iso))
        conn.commit()
    finally:
        conn.close()


def get_bool(key: str, default: bool = False) -> bool:
    """Return ``key`` as a bool. Case-insensitive — ``"True"`` / ``"true"``
    / ``"1"`` all parse as True. Unknown strings fall back to ``default``."""
    raw = get(key)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in _TRUTHY:
        return True
    if normalized in _FALSY:
        return False
    return default


def set_bool(key: str, value: bool) -> None:
    """Persist a bool as ``"true"`` or ``"false"`` so ``get_bool`` round-trips."""
    set(key, "true" if value else "false")


def delete(key: str) -> None:
    """Remove a settings row. Idempotent — missing keys are a no-op."""
    conn = db.get_connection()
    try:
        conn.execute("DELETE FROM settings WHERE key = ?", (key,))
        conn.commit()
    finally:
        conn.close()


def get_updated_at(key: str) -> datetime | None:
    """Return the ``updated_at`` timestamp for ``key``, or None if missing.

    Used by the CLI ``predictions status`` view to show when the master
    switch was last flipped.
    """
    conn = db.get_connection()
    try:
        row = conn.execute(
            "SELECT updated_at FROM settings WHERE key = ?", (key,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return datetime.fromisoformat(str(row["updated_at"]))
