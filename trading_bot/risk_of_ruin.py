"""Risk of ruin — circuit breakers + emergency shutdown (Phase 15).

The mandatory safety layer required before any live-switch decision. Two
severity tiers, kept structurally distinct:

* **TIER 1 (pause)** — 7 consecutive resolved losses OR a 20% peak-to-trough
  EQUITY drawdown → stop opening NEW positions by REVOKING the
  ``new_position_entry`` capability, Pushover-notify, and await operator
  re-authorization. Existing positions are NOT touched — their own watchers
  (Phase 13 options / Phase 14 long-term) keep managing them.
* **TIER 2 (emergency shutdown)** — catastrophic failure (repeated broker
  errors, an unhandled core-loop exception, an unreconcilable position) →
  close EVERY open position via the EXISTING closers, confirm closure via
  broker reconciliation, then halt. Never claims a closure the broker did not
  confirm; never fully shuts down while a position remains open.

AUTHORIZATION LEDGER — the capability pattern mirrored from Phase 10: a named
capability with persisted state and ONE gate function consulted by actors.
Phase 10's ``readiness_state`` table only admits warming/ready, so this ledger
persists in the existing settings store (permissive-by-default: no row ==
authorized, matching the Phase 4/10 ledger conventions). Re-authorization is
operator-only and token-gated (the confirmation-token discipline is defined
here; Phase 10 had no token CLI).

Still PAPER ONLY — this phase builds the gate a future live-switch depends on;
it does not flip that switch.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import requests

from trading_bot import config, db, secrets, settings
from trading_bot.broker.base import Broker
from trading_bot.broker.reconcile import reconcile
from trading_bot.models import OptionPosition

# A notifier sends (title, message) and returns True on a confirmed send —
# dependency-injected so tests capture it; the default posts to Pushover.
Notifier = Callable[[str, str], bool]

_HTTP_TIMEOUT_SECONDS = 10

# Persisted state keys (settings store).
_STATE_KEY = "ror.state"                    # normal | paused | holding | halted
_AUTH_KEY_PREFIX = "auth."                  # auth.<capability> = authorized|revoked
_AUTH_REASON_SUFFIX = ".reason"
_BROKER_ERROR_STREAK_KEY = "ror.broker_error_streak"
_RECONCILE_STREAK_KEY = "ror.reconcile_divergence_streak"
_LAST_EVENT_KEY = "ror.last_event"

STATE_NORMAL = "normal"
STATE_PAUSED = "paused"          # Tier 1 tripped — new entries revoked
STATE_HOLDING = "holding"        # Tier 2 in progress — closes pending (market shut)
STATE_HALTED = "halted"          # Tier 2 complete or unconfirmable — full stop


# ── notification (fail-soft, mirrors readiness._pushover_notify) ─────────────


def _pushover_notify(title: str, message: str) -> bool:
    """Post to Pushover; True only on a 2xx send. Fail-soft, never raises."""
    try:
        token = secrets.get_secret("PUSHOVER_APP_TOKEN")
        user = secrets.get_secret("PUSHOVER_USER_KEY")
        if not token or not user:
            print("  risk: Pushover creds unset - notification skipped",
                  file=sys.stderr)
            return False
        resp = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={"token": token, "user": user, "title": title, "message": message},
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
        status = getattr(resp, "status_code", None)
        if status is not None and 200 <= status < 300:
            return True
        print(f"  risk: Pushover HTTP {status}", file=sys.stderr)
        return False
    except Exception as exc:  # noqa: BLE001 - notification must never raise
        print(f"  risk: Pushover error ({exc})", file=sys.stderr)
        return False


# ── authorization ledger (the Phase 10 capability pattern) ───────────────────


def is_authorized(capability: str) -> bool:
    """The ONE gate actors consult. Permissive-by-default: no row == authorized."""
    return settings.get(f"{_AUTH_KEY_PREFIX}{capability}", "authorized") == "authorized"


def revoke(capability: str, reason: str) -> None:
    """Revoke a capability, recording why. Logged; idempotent."""
    settings.set(f"{_AUTH_KEY_PREFIX}{capability}", "revoked")
    settings.set(f"{_AUTH_KEY_PREFIX}{capability}{_AUTH_REASON_SUFFIX}", reason)
    print(f"  risk: capability {capability} REVOKED ({reason})", file=sys.stderr)


def authorize(capability: str) -> None:
    """Restore a capability (operator re-authorization). Logged; idempotent."""
    settings.set(f"{_AUTH_KEY_PREFIX}{capability}", "authorized")
    settings.delete(f"{_AUTH_KEY_PREFIX}{capability}{_AUTH_REASON_SUFFIX}")
    print(f"  risk: capability {capability} authorized", file=sys.stderr)


def revoke_reason(capability: str) -> str | None:
    """The recorded reason for a revocation, or None when authorized."""
    if is_authorized(capability):
        return None
    return settings.get(f"{_AUTH_KEY_PREFIX}{capability}{_AUTH_REASON_SUFFIX}")


def is_entry_authorized() -> bool:
    """May the bot open NEW positions? (The gate the allocation plan checks.)"""
    return is_authorized(config.ENTRY_CAPABILITY)


# ── persisted tier state ──────────────────────────────────────────────────────


def get_state() -> str:
    """Current risk state: normal | paused | holding | halted."""
    return settings.get(_STATE_KEY, STATE_NORMAL) or STATE_NORMAL


def _set_state(state: str) -> None:
    settings.set(_STATE_KEY, state)


# ── equity snapshot recording + drawdown ─────────────────────────────────────


def record_equity_snapshot(equity: float | None, *, now: datetime | None = None) -> bool:
    """Record one equity observation (from ``Broker.get_account().equity``).

    Called once per scan cycle. A missing equity (account unreadable) records
    nothing and returns False — an outage must not fabricate a drawdown.
    """
    if equity is None or equity <= 0.0:
        print("  risk: no usable equity - snapshot skipped", file=sys.stderr)
        return False
    moment = now if now is not None else datetime.now(UTC)
    db.insert_equity_snapshot(equity, moment)
    return True


def current_drawdown_pct() -> float | None:
    """Peak-to-trough drawdown % from the equity snapshots, or None if no data.

    Reads EQUITY (which already reflects unrealized losses on open positions),
    not closed-trade PnL alone. Phase 23 scope audit: ALREADY correctly scoped
    — account equity moves only on broker-tracked positions; signal-tracking
    trades never touch it (``pnl_dollars`` is always None), so no change was
    needed here.
    """
    peak = db.get_equity_peak()
    latest = db.get_latest_equity()
    if peak is None or latest is None or peak <= 0.0:
        return None
    return (peak - latest) / peak * 100.0


# ── Tier 1: consecutive losses + drawdown → pause new entries ────────────────


def consecutive_losses() -> int:
    """The trailing run of resolved REAL-EXECUTION losses, newest first.

    Phase 23 scope audit: the source query excludes shadow-tracked trades AND
    the data-only crypto swing signals — paper losses from a subsystem that
    can never touch capital must not pause real swing/long-term entry."""
    streak = 0
    for outcome in db.get_recent_resolved_outcomes(limit=max(
        config.MAX_CONSECUTIVE_LOSSES * 2, 20,
    )):
        if outcome == "loss":
            streak += 1
        else:
            break
    return streak


@dataclass(frozen=True)
class Tier1Result:
    """The standing after one Tier-1 evaluation pass."""

    tripped: bool
    trigger: str | None          # 'consecutive_losses' | 'drawdown' | None
    losses: int
    drawdown_pct: float | None
    state: str


def evaluate_tier1(
    *, notifier: Notifier | None = None, now: datetime | None = None,
) -> Tier1Result:
    """Evaluate the Tier-1 breakers; trip → revoke new entries + Pushover.

    Trips when the loss streak reaches ``MAX_CONSECUTIVE_LOSSES`` (exactly at,
    not before) OR the equity drawdown reaches ``MAX_DRAWDOWN_PCT``. Tripping
    REVOKES ``new_position_entry`` via the authorization ledger and pauses the
    state — it does NOT touch existing open positions (their own watchers keep
    running). Already paused/halted states are left as-is (no re-trip spam).
    """
    send = notifier if notifier is not None else _pushover_notify
    losses = consecutive_losses()
    drawdown = current_drawdown_pct()
    state = get_state()

    if state in (STATE_PAUSED, STATE_HOLDING, STATE_HALTED):
        return Tier1Result(False, None, losses, drawdown, state)

    trigger: str | None = None
    if losses >= config.MAX_CONSECUTIVE_LOSSES:
        trigger = "consecutive_losses"
    elif drawdown is not None and drawdown >= config.MAX_DRAWDOWN_PCT:
        trigger = "drawdown"

    if trigger is None:
        return Tier1Result(False, None, losses, drawdown, state)

    detail = (
        f"{losses} consecutive losses (limit {config.MAX_CONSECUTIVE_LOSSES})"
        if trigger == "consecutive_losses"
        else f"drawdown {drawdown:.1f}% (limit {config.MAX_DRAWDOWN_PCT:.0f}%)"
    )
    revoke(config.ENTRY_CAPABILITY, f"tier1: {detail}")
    _set_state(STATE_PAUSED)
    settings.set(_LAST_EVENT_KEY, f"tier1 pause: {detail}")
    print(f"  risk: TIER 1 PAUSE - {detail}", file=sys.stderr)
    try:
        send(
            "TIER 1 CIRCUIT BREAKER - NEW ENTRIES PAUSED",
            f"{detail}. New position entry revoked; existing positions remain "
            "managed by their watchers. Re-authorize via: python -m trading_bot "
            "risk reauthorize <token>",
        )
    except Exception as exc:  # noqa: BLE001 - a notifier must never break the breaker
        print(f"  risk: tier1 notifier error ({exc})", file=sys.stderr)
    return Tier1Result(True, trigger, losses, drawdown, STATE_PAUSED)


# ── Tier 2 detection: catastrophic failure triggers ───────────────────────────


def _get_streak(key: str) -> int:
    raw = settings.get(key, "0") or "0"
    try:
        return int(raw)
    except ValueError:
        return 0


def record_broker_result(ok: bool) -> int:
    """Record one broker order outcome across the equity/options order paths.

    A structured error/rejection increments the consecutive-error streak; a
    success resets it. Returns the streak. FAIL-SOFT: a persistence error is
    logged and the last-known streak returned — the counter must never take
    down an order path.
    """
    try:
        streak = 0 if ok else _get_streak(_BROKER_ERROR_STREAK_KEY) + 1
        settings.set(_BROKER_ERROR_STREAK_KEY, str(streak))
        if not ok:
            print(f"  risk: broker error streak {streak}", file=sys.stderr)
        return streak
    except Exception as exc:  # noqa: BLE001 - the counter must never break an order path
        print(f"  risk: broker-error counter failed ({exc})", file=sys.stderr)
        return _get_streak(_BROKER_ERROR_STREAK_KEY)


def broker_error_streak() -> int:
    """The current consecutive broker-error count."""
    return _get_streak(_BROKER_ERROR_STREAK_KEY)


def record_reconcile_result(ok: bool) -> int:
    """Record one conclusive reconciliation result.

    A clean success resets the streak; a real divergence (``ok=False``)
    increments it once. Provider-unavailable or malformed checks must not call
    this function at all. Returns the streak and remains fail-soft.
    """
    try:
        streak = 0 if ok else _get_streak(_RECONCILE_STREAK_KEY) + 1
        settings.set(_RECONCILE_STREAK_KEY, str(streak))
        if not ok:
            print(f"  risk: reconcile divergence streak {streak}", file=sys.stderr)
        return streak
    except Exception as exc:  # noqa: BLE001 - the counter must never break a check
        print(f"  risk: reconcile counter failed ({exc})", file=sys.stderr)
        return _get_streak(_RECONCILE_STREAK_KEY)


def reconcile_divergence_streak() -> int:
    """The current consecutive reconcile-divergence count."""
    return _get_streak(_RECONCILE_STREAK_KEY)


def check_catastrophic() -> str | None:
    """Return a catastrophic-trigger reason, or None when all detectors are calm.

    Two persisted detectors: the consecutive broker-error streak and the
    consecutive unreconcilable-position streak. (The third trigger — an
    unhandled core-loop exception — arrives via :func:`run_guarded` directly.)
    """
    errors = broker_error_streak()
    if errors >= config.MAX_CONSECUTIVE_BROKER_ERRORS:
        return (
            f"{errors} consecutive broker errors "
            f"(limit {config.MAX_CONSECUTIVE_BROKER_ERRORS})"
        )
    divergences = reconcile_divergence_streak()
    if divergences >= config.MAX_CONSECUTIVE_RECONCILE_DIVERGENCES:
        return (
            f"unreconcilable positions across {divergences} consecutive checks "
            f"(limit {config.MAX_CONSECUTIVE_RECONCILE_DIVERGENCES})"
        )
    return None


def _default_catastrophic_handler(reason: str) -> None:
    """Fail-safe default when no orchestrator is injected: hard-revoke entry and
    record the trigger so the next loop pass (or operator) runs the Tier-2
    shutdown. The full orchestrator (Section 4) is wired in by callers."""
    revoke(config.ENTRY_CAPABILITY, f"tier2: {reason}")
    _set_state(STATE_HOLDING)
    settings.set(_LAST_EVENT_KEY, f"tier2 trigger: {reason}")
    print(f"  risk: CATASTROPHIC TRIGGER recorded - {reason}", file=sys.stderr)


# ── Tier 2: emergency shutdown orchestrator ──────────────────────────────────


@dataclass(frozen=True)
class ShutdownResult:
    """The outcome of one emergency-shutdown pass.

    ``status``:
    * ``holding``     — closes submitted but not all accepted (e.g. market
      closed) or the broker still shows positions: the bot stays ALIVE in a
      minimal holding state and retries next cycle. It never fully shuts down
      while a position remains open.
    * ``unconfirmed`` — the broker itself was unreachable at confirmation:
      halted in a safe non-crashing state, closure NOT claimed, manual
      intervention summoned.
    * ``halted``      — every position broker-confirmed closed; full halt.
    """

    status: str
    trigger: str
    closed: list[str]
    pending: list[str]
    note: str = ""


def _emergency_option_outcome(pos: OptionPosition, exit_price: float | None) -> str:
    """Outcome label for an emergency option close, from realized premium PnL."""
    if exit_price is None or pos.premium_entry is None:
        return "expired"
    if exit_price > pos.premium_entry:
        return "win"
    if exit_price < pos.premium_entry:
        return "loss"
    return "breakeven"


def emergency_shutdown(
    broker: Broker,
    *,
    trigger: str,
    notifier: Notifier | None = None,
    now: datetime | None = None,
    option_price_fetch: Callable[[str], float | None] | None = None,
    long_term_price_fetch: Callable[[str], float | None] | None = None,
) -> ShutdownResult:
    """TIER 2: close EVERY open position, confirm closure, then halt.

    The ONE shutdown path — the manual kill switch and every auto-detected
    catastrophic trigger run THIS function; there is no separate/weaker route.

    1. Hard-revoke all new-position entry immediately.
    2. Close every open position via the EXISTING closers — the Phase 13 options
       closer, the Phase 14 long-term closer, and the Phase 11 equity order path
       for any remaining broker-side position. Closing logic is never
       reimplemented here.
    3. Market closed (a close is rejected): do NOT skip — return ``holding`` so
       the still-alive loop retries automatically at the next cycle/open.
    4. Confirm closure via broker reconciliation — a position is never claimed
       closed without broker confirmation.
    5. Broker unreachable at confirmation: log the unresolvable state, Pushover
       for manual intervention, halt safely WITHOUT claiming closure.
    6. All confirmed closed: full halt + emergency Pushover, event recorded.
    """
    # Lazy imports: options_execution / long_term import THIS module for the
    # broker-error detector, so the orchestrator resolves them at call time.
    from trading_bot import long_term as lt
    from trading_bot import options_execution as oe

    send = notifier if notifier is not None else _pushover_notify
    moment = now if now is not None else datetime.now(UTC)

    # 1 — block all new entry, immediately and unconditionally.
    revoke(config.ENTRY_CAPABILITY, f"tier2: {trigger}")
    _set_state(STATE_HOLDING)
    settings.set(_LAST_EVENT_KEY, f"tier2 shutdown: {trigger}")
    print(f"  risk: EMERGENCY SHUTDOWN initiated ({trigger})", file=sys.stderr)

    closed: list[str] = []
    pending: list[str] = []

    # 2a — options book, via the Phase 13 closer.
    for pos in db.get_open_option_positions():
        try:
            price = option_price_fetch(pos.symbol) if option_price_fetch else None
            exit_price = price if price is not None else pos.premium_entry
            order, _pnl = oe.close_option_position(
                broker, pos, exit_price=exit_price, now=moment,
                outcome=_emergency_option_outcome(pos, exit_price),
            )
            if order.ok:
                closed.append(f"option {pos.symbol} x{pos.contracts:g} @ {exit_price}")
            else:
                pending.append(f"option {pos.symbol} ({order.reason})")
        except Exception as exc:  # noqa: BLE001 - one position must never block the rest
            print(f"  risk: option close error for {pos.symbol}: {exc}",
                  file=sys.stderr)
            pending.append(f"option {pos.symbol} (error: {exc})")

    # 2b — long-term book, via the Phase 14 closer.
    for lt_pos in db.get_open_long_term_positions():
        try:
            price = (
                long_term_price_fetch(lt_pos.ticker)
                if long_term_price_fetch else None
            )
            exit_price = price if price is not None else lt_pos.entry_price
            order = lt.close_long_term_position(
                broker, lt_pos, exit_price=exit_price, now=moment,
                reason="emergency_shutdown",
            )
            if order.ok:
                closed.append(f"long-term {lt_pos.ticker} x{lt_pos.qty:g} @ {exit_price}")
            else:
                pending.append(f"long-term {lt_pos.ticker} ({order.reason})")
        except Exception as exc:  # noqa: BLE001 - one position must never block the rest
            print(f"  risk: long-term close error for {lt_pos.ticker}: {exc}",
                  file=sys.stderr)
            pending.append(f"long-term {lt_pos.ticker} (error: {exc})")

    # 2c — any remaining broker-side position, via the Phase 11 equity path.
    positions = broker.get_positions()
    if positions.ok:
        for bpos in positions.positions:
            try:
                price = (
                    bpos.market_value / bpos.qty
                    if bpos.market_value is not None and bpos.qty
                    else bpos.avg_entry_price
                )
                side = "sell" if bpos.side != "short" else "buy"
                order = broker.submit_order(
                    bpos.symbol, abs(bpos.qty), side,
                    order_type="limit", limit_price=price, time_in_force="day",
                )
                record_broker_result(order.ok)
                if order.ok:
                    closed.append(f"equity {bpos.symbol} x{bpos.qty:g}")
                else:
                    pending.append(f"equity {bpos.symbol} ({order.reason})")
            except Exception as exc:  # noqa: BLE001 - one position must never block the rest
                print(f"  risk: equity close error for {bpos.symbol}: {exc}",
                      file=sys.stderr)
                pending.append(f"equity {bpos.symbol} (error: {exc})")
    else:
        pending.append(f"broker positions unreadable ({positions.reason})")

    # 3 — anything not accepted (e.g. market closed): stay alive and retry.
    if pending:
        note = (
            f"{len(pending)} close(s) pending - holding state, retrying next "
            "cycle (market closed or broker degraded)"
        )
        print(f"  risk: {note}", file=sys.stderr)
        _safe_send(send, "EMERGENCY SHUTDOWN HOLDING - CLOSES PENDING",
                   f"Trigger: {trigger}. Pending: {'; '.join(pending)}")
        return ShutdownResult("holding", trigger, closed, pending, note)

    # 4 — confirm closure via reconciliation; the broker is authoritative.
    report = reconcile(broker)
    if not report.ok:
        # 5 — broker unreachable: never fabricate a closure.
        note = "broker unreachable - closure NOT confirmed"
        print(f"  risk: {note}", file=sys.stderr)
        _safe_send(
            send,
            "UNABLE TO CONFIRM CLOSURE - MANUAL INTERVENTION REQUIRED",
            f"Trigger: {trigger}. Close orders submitted: "
            f"{'; '.join(closed) or '(none)'} - but reconciliation failed "
            f"({report.note}). Verify positions manually.",
        )
        _set_state(STATE_HALTED)
        settings.set(_LAST_EVENT_KEY, f"tier2 UNCONFIRMED halt: {trigger}")
        return ShutdownResult("unconfirmed", trigger, closed, pending, note)

    if report.broker_symbols:
        note = (
            f"broker still holds {report.broker_symbols} - holding state, "
            "retrying next cycle"
        )
        print(f"  risk: {note}", file=sys.stderr)
        return ShutdownResult("holding", trigger, closed, pending, note)

    # 6 — every position broker-confirmed closed: full halt.
    _set_state(STATE_HALTED)
    settings.set(_LAST_EVENT_KEY, f"tier2 halt complete: {trigger}")
    _safe_send(
        send,
        "BOT HAS ENCOUNTERED A SEVERE ERROR SHUT DOWN INITIATED",
        f"Trigger: {trigger}. Positions closed: {'; '.join(closed) or '(none)'}. "
        "Closure broker-confirmed via reconciliation. Restart requires: "
        "python -m trading_bot risk reauthorize <token>",
    )
    print("  risk: SHUTDOWN COMPLETE - all positions confirmed closed",
          file=sys.stderr)
    return ShutdownResult("halted", trigger, closed, pending, "confirmed closed")


def _safe_send(send: Notifier, title: str, message: str) -> None:
    """Notify without ever letting a notifier failure break the shutdown."""
    try:
        send(title, message)
    except Exception as exc:  # noqa: BLE001 - the shutdown must not depend on Pushover
        print(f"  risk: notifier error ({exc})", file=sys.stderr)


# ── operator controls: re-authorization + manual kill switch ─────────────────


def reauthorize(token: str) -> tuple[bool, str]:
    """Clear a Tier-1 pause OR a Tier-2 halt and restore capabilities.

    OPERATOR-ONLY and token-gated (the confirmation-token discipline extending
    the Phase 10 capability pattern): the exact ``ROR_REAUTHORIZE_TOKEN`` must be
    supplied or nothing changes. Restores the entry capability, resets the
    catastrophic detectors, and returns the state to normal. The bot never
    leaves a paused/halted state any other way.
    """
    if token != config.ROR_REAUTHORIZE_TOKEN:
        return False, (
            "invalid confirmation token - nothing changed "
            f"(expected {config.ROR_REAUTHORIZE_TOKEN!r})"
        )
    previous = get_state()
    authorize(config.ENTRY_CAPABILITY)
    settings.set(_BROKER_ERROR_STREAK_KEY, "0")
    settings.set(_RECONCILE_STREAK_KEY, "0")
    _set_state(STATE_NORMAL)
    settings.set(_LAST_EVENT_KEY, f"operator re-authorization (was {previous})")
    print(f"  risk: operator re-authorization (was {previous})", file=sys.stderr)
    return True, f"re-authorized: state {previous} -> normal, entry restored"


def kill_switch(
    token: str,
    broker: Broker,
    *,
    notifier: Notifier | None = None,
    now: datetime | None = None,
    option_price_fetch: Callable[[str], float | None] | None = None,
    long_term_price_fetch: Callable[[str], float | None] | None = None,
) -> ShutdownResult | None:
    """The manual kill switch — the IDENTICAL Tier-2 shutdown path, on demand.

    Token-gated: a wrong token returns None and NOTHING happens. A correct token
    runs :func:`emergency_shutdown` itself — the very same orchestrator an
    auto-detected catastrophic trigger runs; there is no separate or weaker
    human-invoked code path.
    """
    if token != config.ROR_KILLSWITCH_TOKEN:
        print("  risk: kill switch REFUSED (invalid confirmation token)",
              file=sys.stderr)
        return None
    return emergency_shutdown(
        broker, trigger="manual kill switch", notifier=notifier, now=now,
        option_price_fetch=option_price_fetch,
        long_term_price_fetch=long_term_price_fetch,
    )


def run_guarded(
    cycle_fn: Callable[[], None],
    *,
    on_catastrophic: Callable[[str], object] | None = None,
) -> bool:
    """Run one core-loop cycle under the top-level catastrophic guard.

    ANY unhandled exception is caught (full context logged), routed to the
    catastrophic handler (the Tier-2 orchestrator in production), and False is
    returned — the process NEVER crashes silently out of the loop. A failing
    handler is itself caught, so the guard cannot raise.
    """
    try:
        cycle_fn()
        return True
    except Exception as exc:  # noqa: BLE001 - THE top-level guard: catch everything
        import traceback

        reason = f"unhandled exception in core loop: {exc!r}"
        print(f"  risk: {reason}\n{traceback.format_exc()}", file=sys.stderr)
        handler = (
            on_catastrophic if on_catastrophic is not None
            else _default_catastrophic_handler
        )
        try:
            handler(reason)
        except Exception as handler_exc:  # noqa: BLE001 - the guard must never raise
            print(
                f"  risk: catastrophic handler itself failed ({handler_exc})",
                file=sys.stderr,
            )
        return False
