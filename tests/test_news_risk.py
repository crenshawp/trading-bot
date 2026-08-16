"""Direct tests for scanner.check_news_risk.

The function had no direct coverage at all — every test that touches it
monkeypatches it wholesale, and scanner.py is omitted from the coverage report,
so a permanently-dead keyword and an unchecked HTTP status both went unnoticed.
"""

from __future__ import annotations

from typing import Any

import pytest

from trading_bot import scanner


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self._payload


def _articles(*titles: str | None) -> dict[str, Any]:
    return {"articles": [{"title": t} for t in titles]}


@pytest.fixture()
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scanner, "NEWSAPI_KEY", "test-key")


def _patch_get(
    monkeypatch: pytest.MonkeyPatch, response: _FakeResponse
) -> None:
    monkeypatch.setattr(
        scanner.requests, "get", lambda *_a, **_kw: response
    )


# ---------------------------------------------------------------------------
# "SEC" was an uppercase literal tested against a lower-cased headline
# ---------------------------------------------------------------------------


def test_sec_investigation_is_high_risk(
    monkeypatch: pytest.MonkeyPatch, _key: None
) -> None:
    """A HIGH verdict is what suppresses real signals for the ticker.

    `detect_stock_signals` returns only a WARNING entry when the verdict
    contains "HIGH", discarding the tradeable signals. With "SEC" unmatchable
    the ticker kept firing EMA21-Pullback signals into the execution cycle.
    """
    _patch_get(monkeypatch, _FakeResponse(_articles("SEC opens formal probe into Acme")))

    assert "HIGH" in scanner.check_news_risk("ACME")


def test_lowercase_sec_mention_is_high_risk(
    monkeypatch: pytest.MonkeyPatch, _key: None
) -> None:
    _patch_get(monkeypatch, _FakeResponse(_articles("Acme faces sec scrutiny")))

    assert "HIGH" in scanner.check_news_risk("ACME")


@pytest.mark.parametrize(
    "headline",
    [
        "Tech sector drops on rate fears",
        "Acme beats by a second straight quarter",
        "Insecure by design: Acme ships a patch",
    ],
)
def test_sec_does_not_fire_on_substrings(
    monkeypatch: pytest.MonkeyPatch, _key: None, headline: str
) -> None:
    """A naive lower-cased "sec" substring would suppress signals constantly."""
    _patch_get(monkeypatch, _FakeResponse(_articles(headline)))

    assert "HIGH" not in scanner.check_news_risk("ACME")


def test_other_high_risk_words_still_match_as_substrings(
    monkeypatch: pytest.MonkeyPatch, _key: None
) -> None:
    """Unchanged behaviour: "hack" must still match "hacked"."""
    _patch_get(monkeypatch, _FakeResponse(_articles("Acme hacked overnight")))

    assert "HIGH" in scanner.check_news_risk("ACME")


# ---------------------------------------------------------------------------
# HTTP status was never checked
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [401, 426, 429, 500])
def test_bad_http_status_is_reported_to_stderr(
    monkeypatch: pytest.MonkeyPatch,
    _key: None,
    capsys: pytest.CaptureFixture[str],
    status: int,
) -> None:
    """A dead key must not look like "no news today".

    requests does not raise on 4xx/5xx, so the error body fell through to
    `.get("articles", [])` -> [] -> "UNKNOWN" with nothing printed: the news
    gate off for the whole watchlist, permanently and invisibly.
    """
    _patch_get(
        monkeypatch,
        _FakeResponse({"status": "error", "code": "apiKeyInvalid"}, status_code=status),
    )

    verdict = scanner.check_news_risk("ACME")

    assert verdict == "UNKNOWN"
    assert "NewsAPI HTTP" in capsys.readouterr().err


def test_empty_article_list_stays_quiet(
    monkeypatch: pytest.MonkeyPatch, _key: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """A genuine "no articles" 200 is not an error and must not warn."""
    _patch_get(monkeypatch, _FakeResponse({"articles": []}))

    assert scanner.check_news_risk("ACME") == "UNKNOWN"
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# NewsAPI sends "title": null for removed articles
# ---------------------------------------------------------------------------


def test_null_title_does_not_abort_the_check(
    monkeypatch: pytest.MonkeyPatch, _key: None
) -> None:
    """`article.get("title", "")` returns None for an explicit null.

    `None.lower()` raised into the outer except, downgrading the ticker's whole
    news check to UNKNOWN — so one removed article hid a real fraud headline
    sitting next to it.
    """
    _patch_get(monkeypatch, _FakeResponse(_articles(None, "Acme fraud probe widens")))

    assert "HIGH" in scanner.check_news_risk("ACME")
