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
        lambda url, timeout=None, **_: _Resp(200, payload),
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
        lambda url, timeout=None, **_: _Resp(429, {}),
    )
    result = news_client.fetch_headlines("GOOGL")
    assert result.ok is False
    assert "HTTP 429" in capsys.readouterr().err


def test_exception_fails_soft(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _key(monkeypatch, "key")

    def boom(url: str, timeout: int | None = None, **_: object) -> _Resp:
        raise RuntimeError("net down")

    monkeypatch.setattr("trading_bot.news_client.requests.get", boom)
    result = news_client.fetch_headlines("GOOGL")
    assert result.ok is False
    assert "net down" in capsys.readouterr().err


def test_empty_articles_fails_soft(monkeypatch: pytest.MonkeyPatch) -> None:
    _key(monkeypatch, "key")
    monkeypatch.setattr(
        "trading_bot.news_client.requests.get",
        lambda url, timeout=None, **_: _Resp(200, {"articles": []}),
    )
    result = news_client.fetch_headlines("GOOGL")
    assert result.ok is False
    assert result.count == 0


def test_cache_dedupes_within_cycle(monkeypatch: pytest.MonkeyPatch) -> None:
    _key(monkeypatch, "key")
    calls: list[str] = []

    def fake_get(url: str, timeout: int | None = None, **_: object) -> _Resp:
        calls.append(url)
        return _Resp(200, {"articles": [{"title": "A"}]})

    monkeypatch.setattr("trading_bot.news_client.requests.get", fake_get)
    news_client.fetch_headlines("GOOGL")
    news_client.fetch_headlines("GOOGL")
    assert len(calls) == 1  # second call served from cache

    news_client.clear_cache()
    news_client.fetch_headlines("GOOGL")
    assert len(calls) == 2  # cache cleared -> refetch


# ---------------------------------------------------------------------------
# Credential handling — the key must never reach the URL or a log line
# ---------------------------------------------------------------------------


def test_api_key_is_sent_as_a_header_not_in_the_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the key was interpolated into the query string as `apiKey=`.

    `requests` renders the full request URL into its connection/proxy/retry
    exception messages, and the fail-soft handler prints that message to
    stderr — so one DNS blip wrote the live NEWSAPI_KEY into the Railway
    deploy log, once per ticker in the scan loop.
    """
    _key(monkeypatch, "SUPERSECRET123")
    seen: dict[str, Any] = {}

    def fake_get(url: str, timeout: int | None = None, **kwargs: Any) -> _Resp:
        seen["url"] = url
        seen["headers"] = kwargs.get("headers")
        return _Resp(200, {"articles": [{"title": "A"}]})

    monkeypatch.setattr("trading_bot.news_client.requests.get", fake_get)
    assert news_client.fetch_headlines("GOOGL").ok is True

    assert "SUPERSECRET123" not in seen["url"]
    assert "apiKey" not in seen["url"]
    assert seen["headers"] == {"X-Api-Key": "SUPERSECRET123"}


def test_exception_text_carrying_the_key_is_redacted(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Defence in depth: even if the key reaches the exception some other way
    (a redirect, a proxy error, a library change), it is scrubbed on the way
    out — a credential must never be one library detail from the deploy log."""
    _key(monkeypatch, "SUPERSECRET123")

    def boom(url: str, timeout: int | None = None, **_: object) -> _Resp:
        raise RuntimeError(
            "Max retries exceeded with url: /v2/everything?apiKey=SUPERSECRET123"
        )

    monkeypatch.setattr("trading_bot.news_client.requests.get", boom)
    assert news_client.fetch_headlines("GOOGL").ok is False

    err = capsys.readouterr().err
    assert "SUPERSECRET123" not in err
    assert "***REDACTED***" in err  # the failure is still reported, just scrubbed


def test_redact_is_a_no_op_without_a_secret() -> None:
    assert news_client.redact("plain text", None) == "plain text"
    assert news_client.redact("plain text", "") == "plain text"
