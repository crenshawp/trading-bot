"""Tests for trading_bot.config — runtime detection and path constants."""

import importlib
import os
from unittest.mock import patch

import trading_bot.config as config


def test_runtime_is_railway_when_env_set() -> None:
    with patch.dict(os.environ, {"RAILWAY_ENVIRONMENT": "production"}):
        importlib.reload(config)
        assert config.RUNTIME == "railway"


def test_runtime_is_local_when_env_absent() -> None:
    env = {k: v for k, v in os.environ.items() if k != "RAILWAY_ENVIRONMENT"}
    with patch.dict(os.environ, env, clear=True):
        importlib.reload(config)
        assert config.RUNTIME == "local"


def test_db_path_differs_by_runtime() -> None:
    # Railway branch — compare via as_posix() so the assertion holds on Windows
    # too (Path("/data/...") becomes a WindowsPath there, str() uses backslashes).
    with patch.dict(os.environ, {"RAILWAY_ENVIRONMENT": "production"}):
        importlib.reload(config)
        assert config.DB_PATH.as_posix() == "/data/trading_bot.db"

    # Local branch — filename is the same, parent is the project root.
    env = {k: v for k, v in os.environ.items() if k != "RAILWAY_ENVIRONMENT"}
    with patch.dict(os.environ, env, clear=True):
        importlib.reload(config)
        assert config.DB_PATH.name == "trading_bot.db"
