"""Tests for trading_bot.migrate_csv — CSV → SQLite migration."""

from pathlib import Path

from trading_bot import db
from trading_bot.migrate_csv import migrate_csv


def _write_csv(tmp_path: Path, header: list[str], rows: list[list[str]]) -> Path:
    csv_path = tmp_path / "signal_log.csv"
    lines = [",".join(header), *(",".join(row) for row in rows)]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path


def test_migrate_missing_csv_returns_zero(tmp_db: Path, tmp_path: Path) -> None:
    imported, skipped = migrate_csv(tmp_path / "does-not-exist.csv")
    assert (imported, skipped) == (0, 0)


def test_migrate_empty_csv_does_not_error(tmp_db: Path, tmp_path: Path) -> None:
    csv = tmp_path / "empty.csv"
    csv.write_text("timestamp,ticker\n", encoding="utf-8")
    imported, skipped = migrate_csv(csv)
    assert (imported, skipped) == (0, 0)


def test_migrate_valid_rows(tmp_db: Path, tmp_path: Path) -> None:
    csv = _write_csv(
        tmp_path,
        [
            "timestamp", "ticker", "asset_type", "trade_type",
            "direction", "setup", "price", "take_profit", "stop_loss", "confidence",
        ],
        [
            [
                "2026-04-03 11:44:15", "ETH-USD", "crypto", "CRYPTO TRADE",
                "LONG", "Oversold Reversal", "2052.84", "2075.08", "2039.50", "High",
            ],
            [
                "2026-04-03 11:51:31", "BTC-USD", "crypto", "CRYPTO TRADE",
                "LONG", "Oversold Reversal", "66950.17", "67591.82", "66565.18", "High",
            ],
        ],
    )
    imported, skipped = migrate_csv(csv)
    assert imported == 2
    assert skipped == 0
    rows = db.get_signals()
    assert len(rows) == 2
    eth = next(r for r in rows if r.ticker == "ETH-USD")
    assert eth.signal_type == "oversold_reversal"
    assert eth.direction == "long"
    assert eth.asset_class == "crypto"
    assert eth.entry_price == 2052.84
    # Unknown columns get stashed as JSON
    assert eth.raw_indicators_json is not None
    assert "confidence" in eth.raw_indicators_json
    assert "trade_type" in eth.raw_indicators_json


def test_migrate_strips_direction_emoji(tmp_db: Path, tmp_path: Path) -> None:
    csv = _write_csv(
        tmp_path,
        ["timestamp", "ticker", "direction", "setup", "price"],
        [["2026-04-03 11:44:15", "ETH-USD", "LONG 📈", "Oversold Reversal", "2000"]],
    )
    imported, _ = migrate_csv(csv)
    assert imported == 1
    [sig] = db.get_signals()
    assert sig.direction == "long"


def test_migrate_is_idempotent(tmp_db: Path, tmp_path: Path) -> None:
    csv = _write_csv(
        tmp_path,
        ["timestamp", "ticker", "asset_type", "direction", "setup", "price"],
        [["2026-04-03 11:44:15", "ETH-USD", "crypto", "LONG", "Oversold Reversal", "2052.84"]],
    )
    migrate_csv(csv)
    migrate_csv(csv)
    assert len(db.get_signals()) == 1


def test_migrate_skips_malformed_rows(tmp_db: Path, tmp_path: Path) -> None:
    csv = _write_csv(
        tmp_path,
        ["timestamp", "ticker", "asset_type", "direction", "setup", "price"],
        [
            ["2026-04-03 11:44:15", "ETH-USD", "crypto", "LONG", "Oversold Reversal", "2052.84"],
            # missing entry_price
            ["2026-04-04 12:00:00", "BTC-USD", "crypto", "LONG", "Oversold Reversal", ""],
            # bad timestamp
            ["not-a-date", "ETH-USD", "crypto", "LONG", "Oversold Reversal", "1.0"],
            # invalid direction — passes _row_to_signal but rejected by db.insert_signal
            ["2026-04-05 12:00:00", "ETH-USD", "crypto", "sideways", "Oversold Reversal", "1.0"],
        ],
    )
    imported, skipped = migrate_csv(csv)
    assert imported == 1
    assert skipped == 3


def test_migrate_infers_crypto_from_ticker(tmp_db: Path, tmp_path: Path) -> None:
    csv = _write_csv(
        tmp_path,
        ["timestamp", "ticker", "direction", "setup", "price"],
        [["2026-04-03 11:44:15", "BTC-USD", "LONG", "Oversold Reversal", "66000"]],
    )
    migrate_csv(csv)
    [sig] = db.get_signals()
    assert sig.asset_class == "crypto"


def test_migrate_infers_stock_when_ticker_not_usd(tmp_db: Path, tmp_path: Path) -> None:
    csv = _write_csv(
        tmp_path,
        ["timestamp", "ticker", "direction", "setup", "price"],
        [["2026-04-03 11:44:15", "GOOGL", "CALL", "EMA21 Pullback", "180"]],
    )
    migrate_csv(csv)
    [sig] = db.get_signals()
    assert sig.asset_class == "stock"
    assert sig.signal_type == "ema21_pullback"
