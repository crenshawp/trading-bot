"""Tests for trading_bot.sentiment — advisory LLM scoring. LLM is mocked."""

import pytest

from trading_bot import sentiment
from trading_bot.news_client import NewsResult


def _news(count: int, *, ok: bool = True) -> NewsResult:
    return NewsResult("GOOGL", headlines=["headline"] * count, ok=ok)


def test_score_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sentiment, "_call_llm", lambda _p: '{"score": 0.8, "rationale": "Great."}'
    )
    result = sentiment.score("GOOGL", _news(3))
    assert result.ok is True
    assert result.score == pytest.approx(0.8)
    assert result.label == "bullish"
    assert result.rationale == "Great."
    assert result.headline_count == 3


def test_llm_error_fails_soft_neutral(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(_p: str) -> str:
        raise RuntimeError("timeout")

    monkeypatch.setattr(sentiment, "_call_llm", boom)
    result = sentiment.score("GOOGL", _news(3))
    assert result.ok is False
    assert result.score == 0.0
    assert result.label == "neutral"
    assert "sentiment LLM error" in capsys.readouterr().err


def test_no_news_is_neutral_without_calling_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[int] = []
    monkeypatch.setattr(
        sentiment, "_call_llm", lambda _p: called.append(1) or "{}"
    )
    result = sentiment.score("GOOGL", NewsResult("GOOGL", [], ok=False))
    assert result.ok is False
    assert result.score == 0.0
    assert called == []  # LLM never called when there's no news


def test_heavy_news_flag_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sentiment, "_call_llm", lambda _p: '{"score": 0.1, "rationale": "x"}'
    )
    assert sentiment.score("GOOGL", _news(3)).heavy_news is False
    assert sentiment.score("GOOGL", _news(8)).heavy_news is True  # threshold 8


def test_label_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    cases = [(0.5, "bullish"), (-0.5, "bearish"), (0.0, "neutral"),
             (0.24, "neutral"), (0.25, "bullish"), (-0.25, "bearish")]
    for value, label in cases:
        monkeypatch.setattr(
            sentiment, "_call_llm",
            lambda _p, _v=value: f'{{"score": {_v}, "rationale": "x"}}',
        )
        assert sentiment.score("GOOGL", _news(3)).label == label


def test_parse_clamps_and_strips_markdown() -> None:
    score, rationale = sentiment._parse_llm(
        '```json\n{"score": 2.5, "rationale": "Hot"}\n```'
    )
    assert score == 1.0  # clamped to [-1, 1]
    assert rationale == "Hot"


def test_parse_missing_score_raises() -> None:
    with pytest.raises(ValueError, match="score"):
        sentiment._parse_llm('{"rationale": "x"}')


def test_parse_no_json_raises() -> None:
    with pytest.raises(ValueError, match="no JSON"):
        sentiment._parse_llm("not json at all")


def test_call_llm_without_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("trading_bot.secrets.get_secret", lambda _name: None)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        sentiment._call_llm("prompt")
