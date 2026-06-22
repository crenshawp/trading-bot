"""Advisory LLM sentiment scoring — Phase 5.

Given fetched headlines for a ticker, ask an LLM for a sentiment score in
``[-1, 1]`` plus a one-line rationale. STRICTLY ADVISORY — sentiment is stored
and shown but NEVER suppresses a signal.

There was no pre-existing LLM client in the codebase, so this introduces a
minimal one that follows the project's conventions: the API key comes from the
secrets layer (``ANTHROPIC_API_KEY``), the ``anthropic`` SDK is imported lazily,
and the model is a Haiku (cheap/fast, since scoring runs once per fired signal).
EVERY failure path — no key, SDK missing, API error/timeout, unparseable reply,
or simply no news — returns a neutral 0.0 score, is logged, and lets the signal
proceed.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from typing import Any

from trading_bot import config, secrets
from trading_bot.news_client import NewsResult

# Score thresholds for the human-readable label.
_BULLISH_AT = 0.25
_BEARISH_AT = -0.25
_LLM_MAX_TOKENS = 200


@dataclass(frozen=True)
class SentimentResult:
    """Advisory sentiment for a fired signal.

    ``score`` is in ``[-1, 1]`` (0.0 neutral). ``ok`` is False whenever the
    result fell back to neutral (no news or LLM failure) — callers persist it
    either way but never gate on it.
    """

    score: float
    label: str
    rationale: str
    headline_count: int
    heavy_news: bool
    ok: bool


def _label_for(score: float) -> str:
    if score >= _BULLISH_AT:
        return "bullish"
    if score <= _BEARISH_AT:
        return "bearish"
    return "neutral"


def _heavy(headline_count: int) -> bool:
    return headline_count >= config.SENTIMENT_HEAVY_NEWS_THRESHOLD


def _neutral(headline_count: int, rationale: str) -> SentimentResult:
    return SentimentResult(
        score=0.0,
        label="neutral",
        rationale=rationale,
        headline_count=headline_count,
        heavy_news=_heavy(headline_count),
        ok=False,
    )


def score(ticker: str, news: NewsResult) -> SentimentResult:
    """Score sentiment from fetched headlines. Never raises; advisory only."""
    count = news.count
    if not news.ok or count == 0:
        # No news to score — neutral, but still flag heavy-news off and proceed.
        return _neutral(count, "no news data")

    try:
        raw = _call_llm(_build_prompt(ticker, news.headlines))
        score_val, rationale = _parse_llm(raw)
    except Exception as exc:  # noqa: BLE001 - sentiment must never block a signal
        print(
            f"  sentiment LLM error for {ticker}, scoring neutral: {exc}",
            file=sys.stderr,
        )
        return _neutral(count, f"llm error: {exc}")

    return SentimentResult(
        score=score_val,
        label=_label_for(score_val),
        rationale=rationale,
        headline_count=count,
        heavy_news=_heavy(count),
        ok=True,
    )


def _build_prompt(ticker: str, headlines: list[str]) -> str:
    joined = "\n".join(f"- {h}" for h in headlines)
    return (
        "You are a financial news sentiment classifier. Based ONLY on these "
        f"recent headlines about {ticker}, judge the near-term sentiment for "
        "the stock.\n\n"
        f"{joined}\n\n"
        'Respond with ONLY a JSON object: {"score": <number from -1.0 to 1.0>, '
        '"rationale": "<one short sentence>"}. '
        "score: -1 very bearish, 0 neutral, +1 very bullish."
    )


def _parse_llm(raw: str) -> tuple[float, str]:
    """Extract ``(score, rationale)`` from the LLM reply. Raises on bad shape."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match is None:
        raise ValueError(f"no JSON object in LLM reply: {raw!r}")
    data = json.loads(match.group(0))
    if not isinstance(data, dict) or "score" not in data:
        raise ValueError(f"LLM reply missing 'score': {raw!r}")
    score_val = max(-1.0, min(1.0, float(data["score"])))
    rationale = str(data.get("rationale", "")).strip() or "no rationale"
    return score_val, rationale


def _call_llm(prompt: str) -> str:
    """Call the Anthropic API and return the text reply. Raises on any problem.

    Isolated so tests can monkeypatch it without an SDK or network. The
    ``anthropic`` package is imported lazily so it is not a hard dependency —
    a missing SDK simply raises here and is caught by :func:`score`.
    """
    api_key = secrets.get_secret("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    import anthropic  # lazy: not a hard dependency

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=config.SENTIMENT_LLM_MODEL,
        max_tokens=_LLM_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    blocks: Any = response.content
    return str(blocks[0].text)
