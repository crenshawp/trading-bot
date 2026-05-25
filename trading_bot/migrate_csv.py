"""One-shot migration: read signal_log.csv and insert rows into the signals table.

The historical CSV schema evolved over time, so column matching is defensive:
canonical names plus a few known aliases. Anything we don't recognise gets
packed into ``raw_indicators_json`` so no information is lost.

Idempotent — the UNIQUE INDEX on ``(timestamp, ticker, signal_type)`` ensures
re-running this is safe. Malformed rows are skipped, not raised.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from trading_bot import config, db
from trading_bot.models import Signal

# Canonical model field → ordered list of CSV column names that map to it.
_COLUMN_ALIASES: dict[str, list[str]] = {
    "timestamp":   ["timestamp", "Timestamp", "time", "ts"],
    "ticker":      ["ticker", "Ticker", "symbol"],
    "asset_class": ["asset_class", "asset_type", "AssetClass"],
    "signal_type": ["signal_type", "setup", "signal", "strategy"],
    "direction":   ["direction", "Direction", "side"],
    "entry_price": ["entry_price", "price", "Price", "entry"],
    "stop_loss":   ["stop_loss", "StopLoss", "sl"],
    "take_profit": ["take_profit", "TakeProfit", "tp"],
}

_KNOWN_COLUMNS: frozenset[str] = frozenset(
    col for aliases in _COLUMN_ALIASES.values() for col in aliases
)

_TIMESTAMP_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%m/%d/%Y %H:%M:%S",
)


def _first_present(row: dict[str, Any], canonical: str) -> Any:
    """Return the first non-empty value for ``canonical`` or its known aliases."""
    for alias in _COLUMN_ALIASES.get(canonical, [canonical]):
        if alias in row:
            value = row[alias]
            if pd.notna(value) and str(value).strip() != "":
                return value
    return None


def _parse_timestamp(raw: Any) -> datetime | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    # Python 3.11+ fromisoformat accepts the space separator.
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _normalize_direction(raw: Any) -> str | None:
    """Take the first whitespace-separated token and lowercase it.

    The CSV stores values like ``"LONG 📈"``; we want ``"long"``.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    return text.split()[0].lower().strip() or None


def _normalize_signal_type(raw: Any) -> str | None:
    """``"Oversold Reversal"`` → ``"oversold_reversal"``."""
    if raw is None:
        return None
    text = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
    return text or None


def _infer_asset_class(ticker: str, raw: Any) -> str:
    if raw is not None:
        value = str(raw).strip().lower()
        if value in {"stock", "crypto"}:
            return value
    return "crypto" if ticker.endswith("-USD") else "stock"


def _coerce_float(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _row_to_signal(row: dict[str, Any]) -> Signal | None:
    timestamp   = _parse_timestamp(_first_present(row, "timestamp"))
    ticker_raw  = _first_present(row, "ticker")
    signal_type = _normalize_signal_type(_first_present(row, "signal_type"))
    direction   = _normalize_direction(_first_present(row, "direction"))
    entry_price = _coerce_float(_first_present(row, "entry_price"))

    if (
        timestamp is None
        or not ticker_raw
        or not signal_type
        or not direction
        or entry_price is None
    ):
        return None

    ticker = str(ticker_raw).strip()
    asset_class = _infer_asset_class(ticker, _first_present(row, "asset_class"))

    # Stash any unrecognised columns so we never silently lose CSV data.
    extras: dict[str, Any] = {}
    for key, value in row.items():
        if key in _KNOWN_COLUMNS:
            continue
        if pd.notna(value):
            extras[key] = (
                value if isinstance(value, str | int | float | bool) else str(value)
            )
    raw_indicators_json = json.dumps(extras, default=str) if extras else None

    return Signal(
        timestamp=timestamp,
        ticker=ticker,
        asset_class=asset_class,
        signal_type=signal_type,
        direction=direction,
        entry_price=entry_price,
        stop_loss=_coerce_float(_first_present(row, "stop_loss")),
        take_profit=_coerce_float(_first_present(row, "take_profit")),
        raw_indicators_json=raw_indicators_json,
    )


def migrate_csv(csv_path: Path | None = None) -> tuple[int, int]:
    """Migrate CSV to DB. Returns ``(imported, skipped)``.

    If ``csv_path`` is ``None``, uses ``config.CSV_LEGACY_PATH``. Returns
    ``(0, 0)`` if the CSV is missing or empty.
    """
    path = csv_path if csv_path is not None else config.CSV_LEGACY_PATH
    if not path.exists():
        return (0, 0)

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return (0, 0)
    if df.empty:
        return (0, 0)

    imported = 0
    skipped = 0
    for raw_row in df.to_dict(orient="records"):
        signal = _row_to_signal(raw_row)
        if signal is None:
            skipped += 1
            continue
        try:
            db.insert_signal(signal)
        except ValueError:
            skipped += 1
            continue
        imported += 1
    return (imported, skipped)
