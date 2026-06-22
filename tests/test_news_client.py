"""Tests for trading_bot.news_client — fail-soft NewsAPI wrapper.

All network is mocked: requests.get and the secrets layer are monkeypatched.
"""

from collections.abc import Iterator
from typing import Any

import pytest

from trading_bot import news_client


class _Resp:
    def __init__(self, status: int, payload: Any) -> None:
        self.status_code = status
        self._payload = payload

    def json(self) -> Any:
        return self._payload


@pytest.fixture(autouse=True)
def _clear_cache() -> Iterator[None]:
    news_client.clear_cache()
    yield
    news_client.clear_cache()


def _key(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    monkeypatch.setattr("trading_bot.secrets.get_secret", lambda _name: value)


def test_no_api_key_fails_soft(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _key(monkeypatch, None)
    result = news_client.fetch_headlines("GOOGL")
    assert result.ok is False
    assert result.count == 0
    assert "NEWSAPI_KEY not set" in capsys.readouterr().err


def test_success_parses_titles(monkeypatch: pytest.MonkeyPatch) -> None:
    _key(monkeypatch, "key")
    payload = {"articles": [{"title": "A"}, {"title": "B"}, {"title": ""}, {"x": 1}]}
    monkeypatch.setattr(
        "trading_bot.news_client.requests.get",
        lambda url, timeout=None: _Resp(200, payload),
    )
    result = news_client.fetch_headlines("GOOGL")
    assert result.ok is True
    assert result.headlines == ["A", "B"]  # blank / title-less skipped


def test_http_error_fails_soft(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _key(monkeypatch, "key")
    monkeypatch.setattr(
        "trading_bot.news_client.requests.get",
        lambda url, timeout=None: _Resp(429, {}),
    )
    result = news_client.fetch_headlines("GOOGL")
    assert result.ok is False
    assert "HTTP 429" in capsys.readouterr().err


def test_exception_fails_soft(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _key(monkeypatch, "key")

    def boom(url: str, timeout: int | None = None) -> _Resp:
        raise RuntimeError("net down")

    monkeypatch.setattr("trading_bot.news_client.requests.get", boom)
    result = news_client.fetch_headlines("GOOGL")
    assert result.ok is False
    assert "net down" in capsys.readouterr().err


def test_empty_articles_fails_soft(monkeypatch: pytest.MonkeyPatch) -> None:
    _key(monkeypatch, "key")
    monkeypatch.setattr(
        "trading_bot.news_client.requests.get",
        lambda url, timeout=None: _Resp(200, {"articles": []}),
    )
    result = news_client.fetch_headlines("GOOGL")
    assert result.ok is False
    assert result.count == 0


def test_cache_dedupes_within_cycle(monkeypatch: pytest.MonkeyPatch) -> None:
    _key(monkeypatch, "key")
    calls: list[str] = []

    def fake_get(url: str, timeout: int | None = None) -> _Resp:
        calls.append(url)
        return _Resp(200, {"articles": [{"title": "A"}]})

    monkeypatch.setattr("trading_bot.news_client.requests.get", fake_get)
    news_client.fetch_headlines("GOOGL")
    news_client.fetch_headlines("GOOGL")
    assert len(calls) == 1  # second call served from cache

    news_client.clear_cache()
    news_client.fetch_headlines("GOOGL")
    assert len(calls) == 2  # cache cleared -> refetch
