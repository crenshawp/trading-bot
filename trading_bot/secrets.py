"""Unified secret access — one interface, two backends.

- Railway runtime (``RAILWAY_ENVIRONMENT`` is set): read from ``os.environ``;
  ``set``/``delete`` raise because Railway secrets live in the dashboard.
- Local runtime: read from the OS keychain via ``keyring`` first, then fall
  back to a ``.env`` file at the project root.

Every entry point validates ``name`` against :data:`KNOWN_SECRETS` so typos
fail loudly instead of silently returning ``None``.
"""

import os
from pathlib import Path

import keyring

from trading_bot.config import ROOT_DIR, RUNTIME

KNOWN_SECRETS: frozenset[str] = frozenset({
    "DISCORD_WEBHOOK_URL",
    "PUSHOVER_USER_KEY",
    "PUSHOVER_APP_TOKEN",
    "NEWSAPI_KEY",
    # Phase 5: advisory LLM sentiment scoring. Optional — when unset, sentiment
    # fails soft to neutral and signals proceed unaffected.
    "ANTHROPIC_API_KEY",
    # Phase 11: Alpaca PAPER-trading credentials. Optional — when unset, the
    # broker layer fails soft (every read returns ok=False) and never trades.
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
})

_SERVICE_NAME = "trading_bot"
_ENV_FILE: Path = ROOT_DIR / ".env"


def _validate(name: str) -> None:
    if name not in KNOWN_SECRETS:
        raise ValueError(
            f"Unknown secret '{name}'. Add it to KNOWN_SECRETS in "
            f"trading_bot/secrets.py before using it."
        )


def _read_env_file() -> dict[str, str]:
    """Parse a minimal .env file. No python-dotenv dependency."""
    result: dict[str, str] = {}
    if not _ENV_FILE.exists():
        return result
    for raw in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def get_secret(name: str) -> str | None:
    """Return the secret's value, or ``None`` if unset."""
    _validate(name)
    if RUNTIME == "railway":
        return os.environ.get(name)
    # local: keyring first, then .env fallback
    value: str | None = keyring.get_password(_SERVICE_NAME, name)
    if value is not None:
        return value
    return _read_env_file().get(name)


def set_secret(name: str, value: str) -> None:
    """Store the secret. Raises on Railway — set those via the dashboard."""
    _validate(name)
    if RUNTIME == "railway":
        raise RuntimeError("Set secrets via Railway dashboard, not at runtime")
    keyring.set_password(_SERVICE_NAME, name, value)


def delete_secret(name: str) -> None:
    """Delete the secret. Raises on Railway — delete those via the dashboard."""
    _validate(name)
    if RUNTIME == "railway":
        raise RuntimeError("Set secrets via Railway dashboard, not at runtime")
    keyring.delete_password(_SERVICE_NAME, name)


def list_secrets() -> list[str]:
    """Return which ``KNOWN_SECRETS`` are currently set (sorted)."""
    if RUNTIME == "railway":
        return sorted(name for name in KNOWN_SECRETS if os.environ.get(name))
    env_keys = set(_read_env_file().keys())
    found: list[str] = []
    for name in KNOWN_SECRETS:
        if keyring.get_password(_SERVICE_NAME, name) is not None or name in env_keys:
            found.append(name)
    return sorted(found)


def get_required(name: str) -> str:
    """Get a secret or raise with an actionable message."""
    value = get_secret(name)
    if value is None:
        if RUNTIME == "local":
            raise KeyError(
                f"Secret '{name}' not set. Run: "
                f"python -m trading_bot secrets set {name}"
            )
        raise KeyError(
            f"Secret '{name}' not set. Add it in the Railway dashboard "
            f"under Variables."
        )
    return value
