"""Shared pytest fixtures for the trading-bot test suite."""

import os
from collections.abc import Iterator
from unittest.mock import patch

import pytest


@pytest.fixture
def railway_env() -> Iterator[None]:
    """Simulate Railway environment."""
    with patch.dict(os.environ, {"RAILWAY_ENVIRONMENT": "production"}):
        yield


@pytest.fixture
def local_env() -> Iterator[None]:
    """Simulate local environment (no RAILWAY_ENVIRONMENT)."""
    env = {k: v for k, v in os.environ.items() if k != "RAILWAY_ENVIRONMENT"}
    with patch.dict(os.environ, env, clear=True):
        yield


@pytest.fixture
def mock_keyring() -> Iterator[dict[str, str]]:
    """In-memory keyring backend for tests."""
    store: dict[str, str] = {}

    def fake_get(service: str, name: str) -> str | None:
        return store.get(f"{service}:{name}")

    def fake_set(service: str, name: str, value: str) -> None:
        store[f"{service}:{name}"] = value

    def fake_delete(service: str, name: str) -> None:
        store.pop(f"{service}:{name}", None)

    with (
        patch("keyring.get_password", side_effect=fake_get),
        patch("keyring.set_password", side_effect=fake_set),
        patch("keyring.delete_password", side_effect=fake_delete),
    ):
        yield store
