"""Tests for trading_bot.scanner — wiring only.

We verify the new plumbing (secrets read, db init, log_signal dict→Signal
translation, dry-run gating of network calls) without exercising the
signal-detection math or live yfinance / requests calls. The math is
unchanged from the legacy tradingbot.py and validated by 30 days of
forward testing — testing it now would be brittle and out of scope
(spec §6: "Do not test the signal detection logic itself in this phase").
"""

import sqlite3
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import keyring
import pytest

from trading_bot import db, scanner

_SERVICE = "trading_bot"


def _set_all_secrets() -> None:
    keyring.set_password(_SERVICE, "DISCORD_WEBHOOK_URL", "discord_test")
    keyring.set_password(_SERVICE, "PUSHOVER_USER_KEY", "pu_test")
    keyring.set_password(_SERVICE, "PUSHOVER_APP_TOKEN", "pa_test")
    keyring.set_password(_SERVICE, "NEWSAPI_KEY", "news_test")


# ───────────────────────── log_signal ─────────────────────────


def test_log_signal_inserts_into_db(tmp_db: Path) -> None:
    sid = scanner.log_signal({
        "ticker":      "BTC-USD",
        "asset_type":  "crypto",
        "trade_type":  "⚡ CRYPTO TRADE",
        "direction":   "LONG 📈",
        "setup":       "Oversold Reversal",
        "price":       50000.0,
        "take_profit": 51000.0,
        "stop_loss":   49500.0,
        "confidence":  "High",
    })
    assert sid is not None
    [row] = db.get_signals()
    assert row.ticker == "BTC-USD"
    assert row.asset_class == "crypto"
    assert row.signal_type == "oversold_reversal"
    assert row.direction == "long"
    assert row.entry_price == 50000.0
    assert row.take_profit == 51000.0
    assert row.stop_loss == 49500.0


def test_log_signal_returns_id(tmp_db: Path) -> None:
    sid = scanner.log_signal({
        "ticker":      "GOOGL",
        "asset_type":  "stock",
        "trade_type":  "📆 SWING TRADE",
        "direction":   "CALL 📈",
        "setup":       "EMA21 Pullback",
        "price":       180.0,
        "take_profit": 185.0,
        "stop_loss":   178.0,
        "confidence":  "High",
    })
    assert isinstance(sid, int)
    assert sid >= 1


def test_log_signal_handles_missing_indicators(tmp_db: Path) -> None:
    """Optional indicator fields default to None when absent from the dict."""
    sid = scanner.log_signal({
        "ticker":     "META",
        "asset_type": "stock",
        "direction":  "CALL",          # no emoji decoration
        "setup":      "EMA21 Pullback",
        "price":      500.0,
        # stop_loss / take_profit intentionally missing
    })
    assert sid is not None
    [row] = db.get_signals()
    assert row.stop_loss is None
    assert row.take_profit is None
    assert row.rsi is None
    assert row.atr is None
    assert row.macd is None
    assert row.earnings_risk is False


def test_log_signal_normalizes_signal_type(tmp_db: Path) -> None:
    scanner.log_signal({
        "ticker":      "META",
        "asset_type":  "stock",
        "trade_type":  "📆",
        "direction":   "CALL 📈",
        "setup":       "Higher High Breakout",
        "price":       500.0,
        "take_profit": 510.0,
        "stop_loss":   495.0,
        "confidence":  "High",
    })
    [row] = db.get_signals()
    assert row.signal_type == "higher_high_breakout"


def test_log_signal_opens_a_trade(tmp_db: Path) -> None:
    """Phase 1.3: every tradable signal opens a corresponding Trade row."""
    sid = scanner.log_signal({
        "ticker":      "GOOGL",
        "asset_type":  "stock",
        "direction":   "CALL 📈",
        "setup":       "EMA21 Pullback",
        "price":       180.0,
        "take_profit": 185.0,
        "stop_loss":   178.0,
    })
    assert sid is not None
    trade = db.get_trade_by_signal_id(sid)
    assert trade is not None
    assert trade.outcome == "open"
    assert trade.closed_at is None


def test_log_signal_trade_references_signal_id(tmp_db: Path) -> None:
    sid = scanner.log_signal({
        "ticker":      "BTC-USD",
        "asset_type":  "crypto",
        "direction":   "LONG 📈",
        "setup":       "Oversold Reversal",
        "price":       50000.0,
        "take_profit": 51000.0,
        "stop_loss":   49500.0,
    })
    assert sid is not None
    trade = db.get_trade_by_signal_id(sid)
    assert trade is not None
    assert trade.signal_id == sid


def test_log_signal_does_not_double_open_trade_on_dedupe(tmp_db: Path) -> None:
    """Calling log_signal twice with the same signal must not create two trades."""
    payload = {
        "ticker":      "META",
        "asset_type":  "stock",
        "direction":   "CALL 📈",
        "setup":       "Higher High Breakout",
        "price":       500.0,
        "take_profit": 510.0,
        "stop_loss":   495.0,
    }
    sid1 = scanner.log_signal(payload)
    sid2 = scanner.log_signal(payload)
    # Timestamps differ (each call uses datetime.now), so dedupe may or may
    # not fire — but if both produce trades we'd be double-opening, which is
    # the actual hazard. Check that count of trades == count of distinct sids.
    distinct_sids = {sid1, sid2} - {None}
    conn = db.get_connection()
    try:
        trade_count = conn.execute("SELECT COUNT(*) c FROM trades").fetchone()["c"]
    finally:
        conn.close()
    assert trade_count == len(distinct_sids)


def test_log_signal_skips_warning_entries(tmp_db: Path) -> None:
    """Legacy "⚠️ WARNING" earnings/news risk alerts don't fit the schema."""
    sid = scanner.log_signal({
        "ticker":      "TSLA",
        "asset_type":  "stock",
        "trade_type":  "📆 SWING TRADE",
        "direction":   "⚠️ WARNING",
        "setup":       "Earnings Risk",
        "price":       200.0,
        "take_profit": 0.0,
        "stop_loss":   0.0,
        "confidence":  "N/A",
    })
    assert sid is None
    assert db.get_signals() == []


# ───────────────────────── _load_secrets ─────────────────────────


def test_credentials_loaded_from_secrets_layer(
    mock_keyring: dict[str, str], local_env: None
) -> None:
    _set_all_secrets()
    scanner._load_secrets()
    assert scanner.DISCORD_WEBHOOK_URL == "discord_test"
    assert scanner.PUSHOVER_USER_KEY == "pu_test"
    assert scanner.PUSHOVER_APP_TOKEN == "pa_test"
    assert scanner.NEWSAPI_KEY == "news_test"


def test_load_secrets_raises_with_actionable_message_when_missing(
    mock_keyring: dict[str, str], local_env: None
) -> None:
    # Empty mock_keyring → get_required should KeyError with CLI hint
    with pytest.raises(KeyError, match="python -m trading_bot secrets set"):
        scanner._load_secrets()


# ───────────────────────── main() — dry-run ─────────────────────────


def test_scanner_main_calls_init_db(
    tmp_db: Path,
    mock_keyring: dict[str, str],
    local_env: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--dry-run path: main() must call db.init_db() and exit cleanly."""
    _set_all_secrets()

    init_calls: list[bool] = []
    original_init = db.init_db

    def spy_init_db() -> None:
        init_calls.append(True)
        original_init()

    monkeypatch.setattr("trading_bot.db.init_db", spy_init_db)
    # Stub the scan functions — we're testing main()'s orchestration, not them.
    monkeypatch.setattr("trading_bot.scanner.scan_stocks", lambda: None)
    monkeypatch.setattr("trading_bot.scanner.scan_crypto", lambda: None)
    monkeypatch.setattr("trading_bot.scanner.is_market_open", lambda: True)

    with patch.object(sys, "argv", ["tradingbot", "--dry-run"]):
        scanner.main()

    assert init_calls, "main() must call db.init_db()"
    assert "DRY RUN" in capsys.readouterr().out


def test_dry_run_send_notification_skips_network(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With DRY_RUN flipped on, send_notification must short-circuit before
    any requests.post() call and print the alert instead."""
    posts: list[str] = []

    def fake_post(url: str, **_kwargs: Any) -> None:
        posts.append(url)

    monkeypatch.setattr("trading_bot.scanner.requests.post", fake_post)
    monkeypatch.setattr(scanner, "DRY_RUN", True)

    scanner.send_notification({
        "ticker":      "BTC-USD",
        "asset_type":  "crypto",
        "trade_type":  "⚡ CRYPTO TRADE",
        "direction":   "LONG 📈",
        "setup":       "Oversold Reversal",
        "detail":      "test",
        "price":       50000.0,
        "take_profit": 51000.0,
        "stop_loss":   49500.0,
        "confidence":  "High",
        "hold_days":   "2-8 hours",
    })

    assert posts == [], "no Discord/Pushover POSTs should fire in dry-run"
    out = capsys.readouterr().out
    assert "DRY-RUN signal" in out
    assert "BTC-USD" in out
    # Should still have written to the DB
    assert len(db.get_signals()) == 1


# ─────────────────── _active_stock_watchlist (Phase 3.1) ───────────────────


def test_active_watchlist_empty_falls_back_to_seed(
    tmp_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An empty active_watchlist must fall back to the hardcoded seed list
    with a logged warning — never silently scan an empty watchlist."""
    assert db.get_active_watchlist() == []  # fresh DB, nothing seeded
    result = scanner._active_stock_watchlist()
    assert result == list(scanner.STOCK_WATCHLIST)
    assert "empty" in capsys.readouterr().err


def test_active_watchlist_seeded_is_used(tmp_db: Path) -> None:
    db.seed_active_watchlist(["AAA", "BBB"])
    assert scanner._active_stock_watchlist() == ["AAA", "BBB"]


def test_active_watchlist_unreadable_falls_back_to_seed(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def boom() -> list[str]:
        raise sqlite3.OperationalError("no such table: active_watchlist")

    monkeypatch.setattr(db, "get_active_watchlist", boom)
    result = scanner._active_stock_watchlist()
    assert result == list(scanner.STOCK_WATCHLIST)
    assert "unreadable" in capsys.readouterr().err
