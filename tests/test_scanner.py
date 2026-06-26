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
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import keyring
import pytest

from trading_bot import db, scanner
from trading_bot.models import Signal, Trade

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


def _stock_signal(
    ticker: str = "MSFT", setup: str = "EMA21 Pullback",
) -> dict[str, Any]:
    return {
        "ticker":      ticker,
        "asset_type":  "stock",
        "trade_type":  "📆 SWING TRADE",
        "direction":   "CALL 📈",
        "setup":       setup,
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


def _offline_alert_externals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the active-alert path (earnings blackout + sentiment) offline."""
    monkeypatch.setattr(
        "trading_bot.earnings.is_in_blackout", lambda *a, **k: (False, "no blackout"),
    )
    monkeypatch.setattr(scanner, "_score_signal_sentiment", lambda s: None)


def _alerted(kw: dict[str, Any]) -> bool:
    """True if a captured send_notification call would alert (not suppressed)."""
    return kw.get("alert", True) is True and kw.get("track_mode", "active") == "active"


def test_scan_stocks_alerts_active_suppresses_benched(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_stock_pipeline(
        monkeypatch, entries=[("ACT", "active"), ("BEN", "benched")],
    )
    _offline_alert_externals(monkeypatch)
    calls: dict[str, dict[str, Any]] = {}
    monkeypatch.setattr(
        scanner, "send_notification",
        lambda s, **kw: calls.__setitem__(s["ticker"], kw),
    )

    scanner.scan_stocks()

    # Active alerts (via _emit_active_signal); benched is shadow + suppressed.
    assert _alerted(calls["ACT"])
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


# ─────────────── per-pair alert gate (Phase 4) ───────────────


def _stub_two_setup_pipeline(
    monkeypatch: pytest.MonkeyPatch, *, entries: list[tuple[str, str]],
) -> None:
    """Each ticker fires two distinct setups: EMA21 Pullback + Trend Continuation."""
    monkeypatch.setattr(scanner, "is_market_open", lambda: True)
    monkeypatch.setattr(scanner, "_active_stock_watchlist_entries", lambda: entries)
    monkeypatch.setattr(scanner, "get_stock_data", lambda t: {"df": t})
    monkeypatch.setattr(scanner, "add_stock_indicators", lambda df: df)
    monkeypatch.setattr(
        scanner, "detect_stock_signals",
        lambda t, _df: [
            _stock_signal(t, "EMA21 Pullback"),
            _stock_signal(t, "Trend Continuation"),
        ],
    )


def test_muted_pair_is_shadow_while_enabled_pair_on_same_ticker_alerts(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db.add_to_active_watchlist("MSFT", "seed")  # active ticker
    db.set_signal_pair_status("MSFT", "ema21_pullback", "muted")
    # trend_continuation has no row -> default-enabled.
    _stub_two_setup_pipeline(monkeypatch, entries=[("MSFT", "active")])
    _offline_alert_externals(monkeypatch)
    calls: dict[str, dict[str, Any]] = {}
    monkeypatch.setattr(
        scanner, "send_notification",
        lambda s, **kw: calls.__setitem__(s["setup"], kw),
    )

    scanner.scan_stocks()

    # Muted pair: shadow + suppressed. Enabled pair on the same ticker: alerts.
    assert calls["EMA21 Pullback"] == {"track_mode": "shadow", "alert": False}
    assert _alerted(calls["Trend Continuation"])


def test_benched_ticker_alerts_on_nothing_regardless_of_pair_status(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db.add_to_active_watchlist("MSFT", "seed")
    db.set_watchlist_status("MSFT", "benched")
    # Pair explicitly ENABLED — ticker status must still dominate.
    db.set_signal_pair_status("MSFT", "ema21_pullback", "enabled")
    _stub_two_setup_pipeline(monkeypatch, entries=[("MSFT", "benched")])
    calls: dict[str, dict[str, Any]] = {}
    monkeypatch.setattr(
        scanner, "send_notification",
        lambda s, **kw: calls.__setitem__(s["setup"], kw),
    )

    scanner.scan_stocks()

    # Every setup on a benched ticker is shadow + suppressed.
    assert calls["EMA21 Pullback"] == {"track_mode": "shadow", "alert": False}
    assert calls["Trend Continuation"] == {"track_mode": "shadow", "alert": False}


# ─────────────── earnings blackout + sentiment enrichment (Phase 5) ───────────────


def _trade_count() -> int:
    conn = db.get_connection()
    try:
        return int(conn.execute("SELECT COUNT(*) c FROM trades").fetchone()["c"])
    finally:
        conn.close()


def test_earnings_blackout_suppresses_alert_and_opens_no_trade(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db.add_to_active_watchlist("MSFT", "seed")
    _stub_stock_pipeline(monkeypatch, entries=[("MSFT", "active")])
    monkeypatch.setattr(
        "trading_bot.earnings.is_in_blackout",
        lambda *a, **k: (True, "earnings within window"),
    )
    posts: list[str] = []
    monkeypatch.setattr(
        "trading_bot.scanner.requests.post", lambda url, **_k: posts.append(url),
    )
    monkeypatch.setattr(scanner, "DRY_RUN", False)

    scanner.scan_stocks()

    assert posts == [], "earnings-blackout must suppress the alert"
    assert _trade_count() == 0, "earnings-blackout must not open a trade"
    assert "earnings-blackout" in capsys.readouterr().err


def test_news_sentiment_skipped_for_benched_ticker(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db.add_to_active_watchlist("MSFT", "seed")
    db.set_watchlist_status("MSFT", "benched")
    _stub_stock_pipeline(monkeypatch, entries=[("MSFT", "benched")])
    scored: list[str] = []
    monkeypatch.setattr(
        scanner, "_score_signal_sentiment",
        lambda s: scored.append(s["ticker"]),
    )
    monkeypatch.setattr(scanner, "send_notification", lambda s, **kw: None)

    scanner.scan_stocks()

    # Benched -> never reaches _emit_active_signal, so no news/LLM work.
    assert scored == []


def test_news_sentiment_skipped_for_muted_pair(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db.add_to_active_watchlist("MSFT", "seed")
    db.set_signal_pair_status("MSFT", "ema21_pullback", "muted")
    _stub_two_setup_pipeline(monkeypatch, entries=[("MSFT", "active")])
    monkeypatch.setattr(
        "trading_bot.earnings.is_in_blackout", lambda *a, **k: (False, "ok"),
    )
    scored: list[str] = []
    monkeypatch.setattr(
        scanner, "_score_signal_sentiment",
        lambda s: scored.append(s["setup"]),
    )
    monkeypatch.setattr(scanner, "send_notification", lambda s, **kw: None)

    scanner.scan_stocks()

    # Only the enabled pair triggers sentiment work; the muted one is skipped.
    assert scored == ["Trend Continuation"]


def test_sentiment_persisted_on_trade_row(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading_bot.sentiment import SentimentResult

    _offline_context(monkeypatch)
    monkeypatch.setattr(
        "trading_bot.scanner.requests.post", lambda url, **_k: None,
    )
    monkeypatch.setattr(scanner, "DRY_RUN", False)

    sent = SentimentResult(0.6, "bullish", "Beat.", 9, True, True)
    scanner.send_notification(_stock_signal("GOOGL"), sentiment=sent)

    conn = db.get_connection()
    try:
        row = conn.execute(
            "SELECT sentiment_score, sentiment_label, heavy_news, headline_count "
            "FROM trades"
        ).fetchone()
    finally:
        conn.close()
    assert row["sentiment_score"] == pytest.approx(0.6)
    assert row["sentiment_label"] == "bullish"
    assert row["heavy_news"] == 1
    assert row["headline_count"] == 9


def test_alert_includes_sentiment_context(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot.sentiment import SentimentResult

    _offline_context(monkeypatch)
    monkeypatch.setattr(scanner, "DRY_RUN", True)

    sent = SentimentResult(0.6, "bullish", "Strong beat.", 9, True, True)
    scanner.send_notification(_stock_signal("GOOGL"), sentiment=sent)

    out = capsys.readouterr().out
    assert "Sentiment:" in out
    assert "bullish" in out
    assert "+0.60" in out
    assert "Heavy news" in out


# ─────────────── indicator-family attach point (Phase 6) ───────────────


def test_indicators_skipped_for_benched_ticker(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db.add_to_active_watchlist("MSFT", "seed")
    db.set_watchlist_status("MSFT", "benched")
    _stub_stock_pipeline(monkeypatch, entries=[("MSFT", "benched")])
    computed: list[str] = []
    monkeypatch.setattr(
        scanner, "_compute_signal_indicators",
        lambda s, df: computed.append(s["ticker"]),
    )
    monkeypatch.setattr(scanner, "send_notification", lambda s, **kw: None)

    scanner.scan_stocks()

    # Benched -> never reaches _emit_active_signal, so no indicator work.
    assert computed == []


def test_indicators_skipped_for_muted_pair(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db.add_to_active_watchlist("MSFT", "seed")
    db.set_signal_pair_status("MSFT", "ema21_pullback", "muted")
    _stub_two_setup_pipeline(monkeypatch, entries=[("MSFT", "active")])
    monkeypatch.setattr(
        "trading_bot.earnings.is_in_blackout", lambda *a, **k: (False, "ok"),
    )
    monkeypatch.setattr(scanner, "_score_signal_sentiment", lambda s: None)
    computed: list[str] = []
    monkeypatch.setattr(
        scanner, "_compute_signal_indicators",
        lambda s, df: computed.append(s["setup"]),
    )
    monkeypatch.setattr(scanner, "send_notification", lambda s, **kw: None)

    scanner.scan_stocks()

    # Only the enabled pair triggers indicator work; the muted one is skipped.
    assert computed == ["Trend Continuation"]


def test_indicators_computed_for_active_enabled_signal(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db.add_to_active_watchlist("MSFT", "seed")
    _stub_stock_pipeline(monkeypatch, entries=[("MSFT", "active")])
    _offline_alert_externals(monkeypatch)
    computed: list[str] = []
    monkeypatch.setattr(
        scanner, "_compute_signal_indicators",
        lambda s, df: computed.append(s["ticker"]),
    )
    monkeypatch.setattr(scanner, "send_notification", lambda s, **kw: None)

    scanner.scan_stocks()

    assert computed == ["MSFT"]


def test_indicator_context_persisted_on_trade_row(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading_bot.indicators import IndicatorContext

    _offline_context(monkeypatch)
    monkeypatch.setattr("trading_bot.scanner.requests.post", lambda url, **_k: None)
    monkeypatch.setattr(scanner, "DRY_RUN", False)

    ctx = IndicatorContext(
        atr=2.5, realized_vol=0.018, vol_regime="normal", rsi=54.3, adx=27.1,
        obv=123456.0, correlation=0.42, concentration="moderate", ok=True,
    )
    scanner.send_notification(_stock_signal("GOOGL"), indicators=ctx)

    conn = db.get_connection()
    try:
        row = conn.execute(
            "SELECT ind_atr, ind_realized_vol, ind_vol_regime, ind_rsi, ind_adx, "
            "ind_obv, ind_correlation, ind_concentration FROM trades"
        ).fetchone()
    finally:
        conn.close()
    assert row["ind_atr"] == pytest.approx(2.5)
    assert row["ind_realized_vol"] == pytest.approx(0.018)
    assert row["ind_vol_regime"] == "normal"
    assert row["ind_rsi"] == pytest.approx(54.3)
    assert row["ind_adx"] == pytest.approx(27.1)
    assert row["ind_obv"] == pytest.approx(123456.0)
    assert row["ind_correlation"] == pytest.approx(0.42)
    assert row["ind_concentration"] == "moderate"


def test_scan_persists_real_indicator_values_end_to_end(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    db.add_to_active_watchlist("GOOGL", "seed")
    closes = [100.0 + i + (2.0 if i % 2 else 0.0) for i in range(40)]
    frame = pd.DataFrame(
        {
            "Open": closes,
            "High": [c + 2.0 for c in closes],
            "Low": [c - 2.0 for c in closes],
            "Close": closes,
            "Volume": [1_000_000] * 40,
        },
        index=pd.date_range("2026-01-01", periods=40, freq="D"),
    )
    _offline_context(monkeypatch)
    monkeypatch.setattr(scanner, "is_market_open", lambda: True)
    monkeypatch.setattr(
        scanner, "_active_stock_watchlist_entries", lambda: [("GOOGL", "active")],
    )
    monkeypatch.setattr(scanner, "get_stock_data", lambda t: frame)
    monkeypatch.setattr(scanner, "add_stock_indicators", lambda d: d)
    monkeypatch.setattr(
        scanner, "detect_stock_signals", lambda t, _d: [_stock_signal("GOOGL")],
    )
    monkeypatch.setattr(
        "trading_bot.earnings.is_in_blackout", lambda *a, **k: (False, "ok"),
    )
    monkeypatch.setattr(scanner, "_score_signal_sentiment", lambda s: None)
    monkeypatch.setattr(scanner, "DRY_RUN", True)  # no network; log_signal still runs

    scanner.scan_stocks()

    conn = db.get_connection()
    try:
        row = conn.execute(
            "SELECT track_mode, ind_vol_regime, ind_rsi, ind_adx, ind_obv, "
            "ind_concentration FROM trades"
        ).fetchone()
    finally:
        conn.close()
    assert row["track_mode"] == "active"
    # Real families computed from the frame at the signal's reference candle.
    assert row["ind_vol_regime"] in {"low", "normal", "high"}
    assert row["ind_rsi"] is not None
    assert row["ind_adx"] is not None
    assert row["ind_obv"] is not None
    # Only one active name -> correlation can't be computed -> unknown.
    assert row["ind_concentration"] == "unknown"


# ─────────────── risk recommendation attach point (Phase 7) ───────────────


def test_risk_skipped_for_benched_ticker(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db.add_to_active_watchlist("MSFT", "seed")
    db.set_watchlist_status("MSFT", "benched")
    _stub_stock_pipeline(monkeypatch, entries=[("MSFT", "benched")])
    computed: list[str] = []
    monkeypatch.setattr(
        scanner, "_compute_signal_risk",
        lambda s, ctx: computed.append(s["ticker"]),
    )
    monkeypatch.setattr(scanner, "send_notification", lambda s, **kw: None)

    scanner.scan_stocks()

    # Benched -> never reaches _emit_active_signal, so no risk work.
    assert computed == []


def test_risk_skipped_for_muted_pair(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db.add_to_active_watchlist("MSFT", "seed")
    db.set_signal_pair_status("MSFT", "ema21_pullback", "muted")
    _stub_two_setup_pipeline(monkeypatch, entries=[("MSFT", "active")])
    monkeypatch.setattr(
        "trading_bot.earnings.is_in_blackout", lambda *a, **k: (False, "ok"),
    )
    monkeypatch.setattr(scanner, "_score_signal_sentiment", lambda s: None)
    monkeypatch.setattr(scanner, "_compute_signal_indicators", lambda s, df: None)
    computed: list[str] = []
    monkeypatch.setattr(
        scanner, "_compute_signal_risk",
        lambda s, ctx: computed.append(s["setup"]),
    )
    monkeypatch.setattr(scanner, "send_notification", lambda s, **kw: None)

    scanner.scan_stocks()

    # Only the enabled pair triggers risk work; the muted one is skipped.
    assert computed == ["Trend Continuation"]


def test_risk_computed_for_active_enabled_signal(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db.add_to_active_watchlist("MSFT", "seed")
    _stub_stock_pipeline(monkeypatch, entries=[("MSFT", "active")])
    _offline_alert_externals(monkeypatch)
    monkeypatch.setattr(scanner, "_compute_signal_indicators", lambda s, df: None)
    computed: list[str] = []
    monkeypatch.setattr(
        scanner, "_compute_signal_risk",
        lambda s, ctx: computed.append(s["ticker"]),
    )
    monkeypatch.setattr(scanner, "send_notification", lambda s, **kw: None)

    scanner.scan_stocks()

    assert computed == ["MSFT"]


def test_risk_assessment_persisted_on_trade_row(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trading_bot.risk import RiskAssessment

    _offline_context(monkeypatch)
    monkeypatch.setattr("trading_bot.scanner.requests.post", lambda url, **_k: None)
    monkeypatch.setattr(scanner, "DRY_RUN", False)

    assessment = RiskAssessment(
        recommended_size=20.0, stop_distance=3.0, dollar_risk=60.0, risk_pct=0.6,
        position_pct=20.0, capped=True, total_risk_pct=4.5,
        portfolio_verdict="ok", position_verdict="would-exceed-position",
        cluster_risk_pct=3.0, cluster_verdict="ok", ok=False, reason="ok",
    )
    scanner.send_notification(_stock_signal("GOOGL"), risk=assessment)

    conn = db.get_connection()
    try:
        row = conn.execute(
            "SELECT risk_recommended_size, risk_pct, risk_position_pct, risk_capped, "
            "risk_total_pct, risk_portfolio_verdict, risk_position_verdict, "
            "risk_cluster_pct, risk_cluster_verdict FROM trades"
        ).fetchone()
    finally:
        conn.close()
    assert row["risk_recommended_size"] == pytest.approx(20.0)
    assert row["risk_pct"] == pytest.approx(0.6)
    assert row["risk_capped"] == 1
    assert row["risk_total_pct"] == pytest.approx(4.5)
    assert row["risk_portfolio_verdict"] == "ok"
    assert row["risk_position_verdict"] == "would-exceed-position"
    assert row["risk_cluster_verdict"] == "ok"


def test_risk_alert_includes_sizing_and_advisory(
    tmp_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from trading_bot.risk import RiskAssessment

    _offline_context(monkeypatch)
    monkeypatch.setattr(scanner, "DRY_RUN", True)

    assessment = RiskAssessment(
        recommended_size=20.0, risk_pct=0.6, capped=True, ok=False,
        portfolio_verdict="would-exceed-portfolio", position_verdict="ok",
        cluster_verdict="ok", reason="ok",
    )
    scanner.send_notification(_stock_signal("GOOGL"), risk=assessment)

    out = capsys.readouterr().out
    assert "Risk:" in out
    assert "20.00 units" in out
    assert "capped" in out
    assert "would-exceed-portfolio" in out


def _make_seed_signal() -> Signal:
    """A minimal persisted Signal to hang a pre-existing open trade on."""
    return Signal(
        timestamp=datetime(2026, 5, 1, 9, 31), ticker="META", asset_class="stock",
        signal_type="ema21_pullback", direction="call", entry_price=500.0,
    )


def test_scan_persists_real_risk_and_counts_open_book_end_to_end(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    # Pre-seed an OPEN active trade carrying 2.0% recorded risk (no total yet).
    seed_sid = db.insert_signal(_make_seed_signal())
    db.insert_trade(Trade(
        signal_id=seed_sid, opened_at=datetime(2026, 5, 1), outcome="open",
        track_mode="active", risk_pct=2.0, ind_concentration="concentrated",
    ))

    db.add_to_active_watchlist("GOOGL", "seed")
    closes = [100.0 + i + (2.0 if i % 2 else 0.0) for i in range(40)]
    frame = pd.DataFrame(
        {
            "Open": closes,
            "High": [c + 2.0 for c in closes],
            "Low": [c - 2.0 for c in closes],
            "Close": closes,
            "Volume": [1_000_000] * 40,
        },
        index=pd.date_range("2026-01-01", periods=40, freq="D"),
    )
    _offline_context(monkeypatch)
    monkeypatch.setattr(scanner, "is_market_open", lambda: True)
    monkeypatch.setattr(
        scanner, "_active_stock_watchlist_entries", lambda: [("GOOGL", "active")],
    )
    monkeypatch.setattr(scanner, "get_stock_data", lambda t: frame)
    monkeypatch.setattr(scanner, "add_stock_indicators", lambda d: d)
    monkeypatch.setattr(
        scanner, "detect_stock_signals", lambda t, _d: [_stock_signal("GOOGL")],
    )
    monkeypatch.setattr(
        "trading_bot.earnings.is_in_blackout", lambda *a, **k: (False, "ok"),
    )
    monkeypatch.setattr(scanner, "_score_signal_sentiment", lambda s: None)
    monkeypatch.setattr(scanner, "DRY_RUN", True)  # no network; log_signal still runs

    scanner.scan_stocks()

    # Only the candidate carries a portfolio total; it must include the pre-seeded
    # 2.0% open risk plus its own recommended risk.
    conn = db.get_connection()
    try:
        row = conn.execute(
            "SELECT risk_recommended_size, risk_pct, risk_total_pct FROM trades "
            "WHERE risk_total_pct IS NOT NULL"
        ).fetchone()
    finally:
        conn.close()
    assert row["risk_recommended_size"] is not None       # sized off the real ATR
    assert row["risk_pct"] is not None
    assert row["risk_total_pct"] == pytest.approx(2.0 + row["risk_pct"])
