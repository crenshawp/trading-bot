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


# ─────────────── shadow tracking + alert suppression (Phase 3.1-LIVE) ───────────────


def _offline_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force regime/VIX tagging offline so log_signal never hits the network —
    it catches the fetch errors and tags the trade 'unknown'."""
    from trading_bot import regime, vix

    def _no_regime(*_a: object, **_k: object) -> object:
        raise regime.RegimeFetchError("offline")

    def _no_vix(*_a: object, **_k: object) -> object:
        raise vix.VixFetchError("offline")

    monkeypatch.setattr("trading_bot.regime.get_current_regime", _no_regime)
    monkeypatch.setattr("trading_bot.vix.get_current_vix", _no_vix)


def _stock_signal(ticker: str = "MSFT") -> dict[str, Any]:
    return {
        "ticker":      ticker,
        "asset_type":  "stock",
        "trade_type":  "📆 SWING TRADE",
        "direction":   "CALL 📈",
        "setup":       "EMA21 Pullback",
        "detail":      "test",
        "price":       100.0,
        "take_profit": 104.0,
        "stop_loss":   97.0,
        "confidence":  "High",
        "hold_days":   "3-5 days",
    }


def _trade_track_modes() -> list[str]:
    conn = db.get_connection()
    try:
        rows = conn.execute("SELECT track_mode FROM trades").fetchall()
    finally:
        conn.close()
    return [r["track_mode"] for r in rows]


def test_shadow_signal_opens_shadow_trade_and_suppresses_alert(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _offline_context(monkeypatch)
    posts: list[str] = []
    monkeypatch.setattr(
        "trading_bot.scanner.requests.post",
        lambda url, **_k: posts.append(url),
    )
    monkeypatch.setattr(scanner, "DRY_RUN", False)

    scanner.send_notification(_stock_signal(), track_mode="shadow", alert=False)

    assert posts == [], "shadow signals must NOT alert"
    assert _trade_track_modes() == ["shadow"]
    assert "suppressed" in capsys.readouterr().out


def test_active_signal_alerts_and_opens_active_trade(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _offline_context(monkeypatch)
    posts: list[str] = []
    monkeypatch.setattr(
        "trading_bot.scanner.requests.post",
        lambda url, **_k: posts.append(url),
    )
    monkeypatch.setattr(scanner, "DRY_RUN", False)

    scanner.send_notification(_stock_signal(), track_mode="active", alert=True)

    assert posts, "active signals must still alert (Discord + Pushover)"
    assert _trade_track_modes() == ["active"]


def _stub_shadow_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    universe: list[str],
    active: list[str],
    fire_for: set[str] | None = None,
) -> None:
    """Wire scan_shadow's dependencies for an offline run. ``fire_for`` names
    produce a signal; the rest produce none. ``None`` means every ticker fires."""
    monkeypatch.setattr(scanner, "is_market_open", lambda: True)
    monkeypatch.setattr(scanner, "_active_stock_watchlist", lambda: active)
    monkeypatch.setattr(scanner, "SHADOW_UNIVERSE", universe)
    monkeypatch.setattr(scanner, "_SHADOW_THROTTLE_SECONDS", 0)
    monkeypatch.setattr(scanner, "get_stock_data", lambda t: {"df": t})
    monkeypatch.setattr(scanner, "add_stock_indicators", lambda df: df)

    def fake_detect(ticker: str, _df: object) -> list[dict[str, Any]]:
        if fire_for is None or ticker in fire_for:
            return [_stock_signal(ticker)]
        return []

    monkeypatch.setattr(scanner, "detect_stock_signals", fake_detect)


def test_scan_shadow_targets_only_non_active_tickers(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_shadow_pipeline(
        monkeypatch, universe=["AAA", "BBB", "CCC"], active=["AAA"],
    )
    calls: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        scanner, "send_notification",
        lambda s, **kw: calls.append((s["ticker"], kw)),
    )

    scanner.scan_shadow()

    # AAA is on the active watchlist -> not shadowed.
    assert [t for t, _ in calls] == ["BBB", "CCC"]
    for _, kw in calls:
        assert kw["track_mode"] == "shadow"
        assert kw["alert"] is False


def test_scan_shadow_logs_per_ticker_error_and_continues(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _stub_shadow_pipeline(
        monkeypatch, universe=["BAD", "GOOD"], active=[], fire_for={"GOOD"},
    )

    def fake_get(ticker: str) -> object:
        if ticker == "BAD":
            raise RuntimeError("rate limited")
        return {"df": ticker}

    monkeypatch.setattr(scanner, "get_stock_data", fake_get)
    fired: list[str] = []
    monkeypatch.setattr(
        scanner, "send_notification", lambda s, **_kw: fired.append(s["ticker"]),
    )

    scanner.scan_shadow()

    err = capsys.readouterr().err
    assert "BAD" in err and "rate limited" in err   # logged, not silent
    assert fired == ["GOOD"]                          # sweep continued past BAD


def test_run_shadow_scan_never_raises(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def boom() -> None:
        raise RuntimeError("systemic shadow failure")

    monkeypatch.setattr(scanner, "scan_shadow", boom)
    # Must not propagate — the active pipeline is sacred.
    scanner._run_shadow_scan()
    assert "shadow scan error" in capsys.readouterr().err


# ─────────────── benched scanning + alert suppression (Phase 3.3) ───────────────


def _stub_stock_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    entries: list[tuple[str, str]],
) -> None:
    """Wire scan_stocks' dependencies for an offline run with the given
    ``(ticker, status)`` watchlist entries; every ticker fires one signal."""
    monkeypatch.setattr(scanner, "is_market_open", lambda: True)
    monkeypatch.setattr(scanner, "_active_stock_watchlist_entries", lambda: entries)
    monkeypatch.setattr(scanner, "get_stock_data", lambda t: {"df": t})
    monkeypatch.setattr(scanner, "add_stock_indicators", lambda df: df)
    monkeypatch.setattr(
        scanner, "detect_stock_signals", lambda t, _df: [_stock_signal(t)],
    )


def test_scan_stocks_alerts_active_suppresses_benched(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_stock_pipeline(
        monkeypatch, entries=[("ACT", "active"), ("BEN", "benched")],
    )
    calls: dict[str, dict[str, Any]] = {}
    monkeypatch.setattr(
        scanner, "send_notification",
        lambda s, **kw: calls.__setitem__(s["ticker"], kw),
    )

    scanner.scan_stocks()

    # Active fires with defaults (alerts); benched is shadow-tagged + suppressed.
    assert calls["ACT"] == {}
    assert calls["BEN"] == {"track_mode": "shadow", "alert": False}


def test_benched_ticker_scanned_logged_alert_suppressed(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _offline_context(monkeypatch)
    posts: list[str] = []
    monkeypatch.setattr(
        "trading_bot.scanner.requests.post", lambda url, **_k: posts.append(url),
    )
    monkeypatch.setattr(scanner, "DRY_RUN", False)
    _stub_stock_pipeline(monkeypatch, entries=[("BEN", "benched")])

    scanner.scan_stocks()

    assert posts == [], "benched alerts must be suppressed"
    assert _trade_track_modes() == ["shadow"], "benched outcome still logged"
    assert "suppressed" in capsys.readouterr().out


def test_benched_ticker_not_in_shadow_universe_is_still_scanned(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # BEN is benched and NOT in the shadow universe — it must still be scanned
    # via scan_stocks (it lives in the watchlist), so it never goes dark.
    monkeypatch.setattr(scanner, "SHADOW_UNIVERSE", ["SOMETHING_ELSE"])
    _stub_stock_pipeline(monkeypatch, entries=[("BEN", "benched")])
    fired: list[str] = []
    monkeypatch.setattr(
        scanner, "send_notification", lambda s, **_kw: fired.append(s["ticker"]),
    )

    scanner.scan_stocks()

    assert fired == ["BEN"]
