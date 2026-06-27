"""Tests for trading_bot.readiness — the unified readiness gate (Phase 10).

Synthetic data only, Pushover mocked. The central correctness requirements:
thresholds equal the existing constants (behavior preserved), readiness flips
exactly at the threshold, and the crossing notification fires exactly once.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from trading_bot import config, db, readiness
from trading_bot.models import Signal, Trade

_TS = datetime(2026, 1, 1, 9, 0)


def _seed_resolved(
    n: int, *, outcome: str = "win", pnl: float | None = 2.0,
    track_mode: str = "active", start: int = 0,
) -> int:
    """Insert n resolved trades (each its own signal). Returns next free idx."""
    idx = start
    for _ in range(n):
        ts = _TS + timedelta(minutes=idx)
        sid = db.insert_signal(Signal(
            timestamp=ts, ticker=f"T{idx}", asset_class="stock",
            signal_type="ema21_pullback", direction="call", entry_price=100.0,
        ))
        db.insert_trade(Trade(
            signal_id=sid, opened_at=ts, closed_at=ts + timedelta(days=1),
            outcome=outcome, pnl_pct=pnl, track_mode=track_mode,
        ))
        idx += 1
    return idx


# ───────────────────────── registry / centralization ────────────────────────────


def test_registry_thresholds_equal_centralized_constants() -> None:
    caps = {c.name: c for c in readiness.REGISTRY}
    assert caps["watchlist_rotation"].threshold == config.SM_MIN_CLOSED_SIGNALS
    assert caps["pair_gating"].threshold == config.SP_MIN_CLOSED_SIGNALS
    assert caps["shadow_promotion"].threshold == config.MIN_SHADOW_SIGNALS
    assert caps["self_optimization"].threshold == config.SO_MIN_SAMPLE


def test_min_shadow_signals_back_compat_alias() -> None:
    from trading_bot import shadow_discovery
    assert shadow_discovery.MIN_SHADOW_SIGNALS == config.MIN_SHADOW_SIGNALS


def test_capability_lookup_unknown_raises() -> None:
    assert readiness.capability("watchlist_rotation").kind == "deterministic"
    with pytest.raises(KeyError, match="unknown capability"):
        readiness.capability("does_not_exist")


def test_scope_count_rejects_unknown_scope() -> None:
    with pytest.raises(ValueError, match="unknown readiness scope"):
        readiness._scope_count("bogus")


# ───────────────────────── scope counts ─────────────────────────────────────────


def test_resolved_count_by_scope(tmp_db: Path) -> None:
    idx = _seed_resolved(3, outcome="win", track_mode="active")
    idx = _seed_resolved(2, outcome="loss", pnl=-1.0, track_mode="active", start=idx)
    idx = _seed_resolved(1, outcome="expired", pnl=None, track_mode="active", start=idx)
    idx = _seed_resolved(4, outcome="win", track_mode="shadow", start=idx)
    # 'all' scope (watchlist_rotation): 5 active + 4 shadow = 9 (expired excluded)
    assert readiness.resolved_count("watchlist_rotation") == 9
    # 'active' scope (self_optimization): 5
    assert readiness.resolved_count("self_optimization") == 5
    # 'shadow' scope (shadow_promotion): 4
    assert readiness.resolved_count("shadow_promotion") == 4


# ───────────────────────── readiness crossing ───────────────────────────────────


def test_is_ready_flips_exactly_at_threshold(tmp_db: Path) -> None:
    threshold = config.SM_MIN_CLOSED_SIGNALS        # watchlist_rotation, scope 'all'
    _seed_resolved(threshold - 1)
    assert readiness.is_ready("watchlist_rotation") is False    # n-1: not ready
    _seed_resolved(1, start=threshold)
    assert readiness.is_ready("watchlist_rotation") is True     # n: ready


def test_is_ready_scopes_are_independent(tmp_db: Path) -> None:
    # 10 active resolved makes the 'all'-scope cap ready but not the shadow one.
    _seed_resolved(config.SM_MIN_CLOSED_SIGNALS, track_mode="active")
    assert readiness.is_ready("watchlist_rotation") is True
    assert readiness.is_ready("shadow_promotion") is False      # no shadow trades


# ───────────────────────── evaluation + one-time notification ────────────────────

NOW = datetime(2026, 6, 30, 12, 0)


class _Recorder:
    """A fake notifier that records (title, message) calls and can be told to
    succeed or fail the send."""

    def __init__(self, *, succeed: bool = True) -> None:
        self.calls: list[tuple[str, str]] = []
        self.succeed = succeed

    def __call__(self, title: str, message: str) -> bool:
        self.calls.append((title, message))
        return self.succeed

    def for_cap(self, name: str) -> list[tuple[str, str]]:
        return [c for c in self.calls if name in c[1]]


def _result(results: list[readiness.ReadinessResult], name: str) -> readiness.ReadinessResult:
    return next(r for r in results if r.capability == name)


def test_below_threshold_stays_warming_no_announce(tmp_db: Path) -> None:
    _seed_resolved(config.SM_MIN_CLOSED_SIGNALS - 1)        # 9 < 10
    rec = _Recorder()
    results = readiness.evaluate_readiness(notifier=rec, now=NOW)
    wl = _result(results, "watchlist_rotation")
    assert wl.status == "warming"
    assert wl.newly_announced is False
    assert rec.calls == []


def test_crossing_flips_ready_and_announces_exactly_once(tmp_db: Path) -> None:
    _seed_resolved(config.SM_MIN_CLOSED_SIGNALS)            # 10 -> ready
    rec = _Recorder()
    results = readiness.evaluate_readiness(notifier=rec, now=NOW)
    wl = _result(results, "watchlist_rotation")
    assert wl.status == "ready"
    assert wl.newly_announced is True
    assert len(rec.for_cap("watchlist_rotation")) == 1
    assert "switched on automatically" in rec.for_cap("watchlist_rotation")[0][1]

    # A second pass must NOT re-announce (announced persists).
    results2 = readiness.evaluate_readiness(notifier=rec, now=NOW)
    wl2 = _result(results2, "watchlist_rotation")
    assert wl2.newly_announced is False
    assert wl2.announced is True
    assert len(rec.for_cap("watchlist_rotation")) == 1     # still exactly one


def test_n_at_crossing_recorded_and_frozen(tmp_db: Path) -> None:
    _seed_resolved(config.SM_MIN_CLOSED_SIGNALS)           # cross at 10
    readiness.evaluate_readiness(notifier=_Recorder(), now=NOW)
    _seed_resolved(5, start=999)                           # more data after crossing
    readiness.evaluate_readiness(notifier=_Recorder(), now=NOW)
    row = db.get_readiness_state("watchlist_rotation")
    assert row["n_at_crossing"] == config.SM_MIN_CLOSED_SIGNALS   # frozen at first cross
    assert row["crossed_at"] == NOW.isoformat()


def test_send_failure_leaves_announced_false_then_retries(tmp_db: Path) -> None:
    _seed_resolved(config.SM_MIN_CLOSED_SIGNALS)
    failing = _Recorder(succeed=False)
    readiness.evaluate_readiness(notifier=failing, now=NOW)
    row = db.get_readiness_state("watchlist_rotation")
    assert row["status"] == "ready"        # crossing still latched
    assert row["announced"] is False       # but send failed -> not announced
    assert len(failing.for_cap("watchlist_rotation")) == 1

    ok = _Recorder(succeed=True)
    readiness.evaluate_readiness(notifier=ok, now=NOW)     # retry succeeds
    row2 = db.get_readiness_state("watchlist_rotation")
    assert row2["announced"] is True
    assert len(ok.for_cap("watchlist_rotation")) == 1


def test_announce_message_wording_by_kind() -> None:
    rec = _Recorder()
    det = readiness.Capability("det_x", "deterministic", 10, "all", "t")
    ml = readiness.Capability("ml_x", "ml", 500, "all", "t")
    assert readiness._announce(det, 10, rec) is True
    assert readiness._announce(ml, 500, rec) is True
    assert "switched on automatically" in rec.calls[0][1]
    assert "Awaiting the build" in rec.calls[1][1] and "to be built" in rec.calls[1][1]


def test_announce_swallows_a_raising_notifier(
    tmp_db: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _seed_resolved(config.SM_MIN_CLOSED_SIGNALS)

    def raising(_title: str, _message: str) -> bool:
        raise RuntimeError("boom")

    results = readiness.evaluate_readiness(notifier=raising, now=NOW)
    wl = _result(results, "watchlist_rotation")
    assert wl.status == "ready"
    assert wl.announced is False           # raise treated as a failed send
    assert "notifier error" in capsys.readouterr().err


# ───────────────────────── default Pushover notifier ────────────────────────────


def test_pushover_notify_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("trading_bot.readiness.secrets.get_secret", lambda _n: "key")

    class _Resp:
        status_code = 200

    monkeypatch.setattr("trading_bot.readiness.requests.post", lambda *a, **k: _Resp())
    assert readiness._pushover_notify("t", "m") is True


def test_pushover_notify_missing_creds(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("trading_bot.readiness.secrets.get_secret", lambda _n: None)
    assert readiness._pushover_notify("t", "m") is False
    assert "creds unset" in capsys.readouterr().err


def test_pushover_notify_non_2xx(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("trading_bot.readiness.secrets.get_secret", lambda _n: "key")

    class _Resp:
        status_code = 429

    monkeypatch.setattr("trading_bot.readiness.requests.post", lambda *a, **k: _Resp())
    assert readiness._pushover_notify("t", "m") is False
    assert "HTTP 429" in capsys.readouterr().err


def test_pushover_notify_exception(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("trading_bot.readiness.secrets.get_secret", lambda _n: "key")

    def boom(*_a: object, **_k: object) -> object:
        raise RuntimeError("net down")

    monkeypatch.setattr("trading_bot.readiness.requests.post", boom)
    assert readiness._pushover_notify("t", "m") is False
    assert "error" in capsys.readouterr().err
