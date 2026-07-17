"""Tests for trading_bot.secrets — local keyring and Railway env-var backends."""

import os
from unittest.mock import patch

import pytest
from keyring.errors import NoKeyringError

import trading_bot.secrets as secrets_module

# --- Validation ---


def test_unknown_secret_raises_value_error(mock_keyring: dict[str, str], local_env: None) -> None:
    with pytest.raises(ValueError, match="Unknown secret"):
        secrets_module.get_secret("NOT_A_REAL_KEY")


# --- Local backend ---


def test_local_set_then_get(mock_keyring: dict[str, str], local_env: None) -> None:
    secrets_module.set_secret("PUSHOVER_USER_KEY", "test_value")
    assert secrets_module.get_secret("PUSHOVER_USER_KEY") == "test_value"


def test_local_delete(mock_keyring: dict[str, str], local_env: None) -> None:
    secrets_module.set_secret("PUSHOVER_USER_KEY", "test_value")
    secrets_module.delete_secret("PUSHOVER_USER_KEY")
    assert secrets_module.get_secret("PUSHOVER_USER_KEY") is None


def test_local_list_secrets(mock_keyring: dict[str, str], local_env: None) -> None:
    secrets_module.set_secret("PUSHOVER_USER_KEY", "a")
    secrets_module.set_secret("NEWSAPI_KEY", "b")
    result = secrets_module.list_secrets()
    assert "PUSHOVER_USER_KEY" in result
    assert "NEWSAPI_KEY" in result


def test_get_required_raises_with_helpful_message(
    mock_keyring: dict[str, str], local_env: None
) -> None:
    with pytest.raises(KeyError, match="python -m trading_bot secrets set"):
        secrets_module.get_required("PUSHOVER_USER_KEY")


# --- Railway backend ---


def test_railway_get_reads_environ(railway_env: None) -> None:
    with (
        patch.dict(os.environ, {"PUSHOVER_USER_KEY": "railway_value"}),
        patch("trading_bot.secrets.RUNTIME", "railway"),
    ):
        assert secrets_module.get_secret("PUSHOVER_USER_KEY") == "railway_value"


def test_railway_set_raises(railway_env: None) -> None:
    with (
        patch("trading_bot.secrets.RUNTIME", "railway"),
        pytest.raises(RuntimeError, match="Railway dashboard"),
    ):
        secrets_module.set_secret("PUSHOVER_USER_KEY", "x")


def test_railway_list_returns_set_keys(railway_env: None) -> None:
    with (
        patch.dict(os.environ, {"PUSHOVER_USER_KEY": "x", "NEWSAPI_KEY": "y"}),
        patch("trading_bot.secrets.RUNTIME", "railway"),
    ):
        result = secrets_module.list_secrets()
        assert "PUSHOVER_USER_KEY" in result
        assert "NEWSAPI_KEY" in result


def test_get_required_railway_message(railway_env: None) -> None:
    with (
        patch("trading_bot.secrets.RUNTIME", "railway"),
        patch.dict(os.environ, {}, clear=True),
        pytest.raises(KeyError, match="Railway dashboard"),
    ):
        secrets_module.get_required("NEWSAPI_KEY")


# --- Keyring backend failure: must fail soft, never propagate ---


def test_get_secret_fails_soft_when_keyring_backend_missing(local_env: None) -> None:
    """A missing/broken keyring backend degrades to None, it does not raise."""
    with (
        patch("keyring.get_password", side_effect=NoKeyringError("no backend")),
        patch.object(secrets_module, "_read_env_file", return_value={}),
    ):
        assert secrets_module.get_secret("NEWSAPI_KEY") is None


def test_get_secret_falls_back_to_env_when_keyring_raises(local_env: None) -> None:
    """When keyring raises, the .env fallback is still consulted."""
    with (
        patch("keyring.get_password", side_effect=NoKeyringError("no backend")),
        patch.object(secrets_module, "_read_env_file", return_value={"NEWSAPI_KEY": "from_env"}),
    ):
        assert secrets_module.get_secret("NEWSAPI_KEY") == "from_env"


def test_list_secrets_fails_soft_when_keyring_raises(local_env: None) -> None:
    """list_secrets degrades to the .env view rather than crashing."""
    with (
        patch("keyring.get_password", side_effect=NoKeyringError("no backend")),
        patch.object(secrets_module, "_read_env_file", return_value={"PUSHOVER_USER_KEY": "x"}),
    ):
        assert secrets_module.list_secrets() == ["PUSHOVER_USER_KEY"]
