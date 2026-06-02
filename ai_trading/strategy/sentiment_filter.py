"""Sentiment-aware strategy filter.

Wraps the moving average signal with a sentiment overlay that can
block trades when market sentiment is strongly against the signal direction.

This is a conservative filter — it only BLOCKS trades, never initiates them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from ai_trading.data.news_sentiment import NewsArticle, fetch_news
from ai_trading.ml.sentiment import SentimentAnalyzer

logger = logging.getLogger(__name__)


# --- Sentiment cache (#12) ---
class _SentimentCache:
    """In-memory cache for sentiment results to avoid redundant API calls."""

    def __init__(self, ttl_sec: int = 600) -> None:
        self.ttl_sec = ttl_sec
        self._store: dict[str, tuple[float, float]] = {}  # symbol -> (timestamp, score)

    def get(self, symbol: str) -> float | None:
        if self.ttl_sec <= 0:
            return None
        entry = self._store.get(symbol)
        if entry is None:
            return None
        ts, score = entry
        if time.time() - ts > self.ttl_sec:
            del self._store[symbol]
            return None
        return score

    def put(self, symbol: str, score: float) -> None:
        if self.ttl_sec <= 0:
            return
        self._store[symbol] = (time.time(), score)


_sentiment_cache = _SentimentCache(ttl_sec=600)


def configure_sentiment_cache(ttl_sec: int) -> None:
    """Update the sentiment cache TTL. Called from bot startup."""
    global _sentiment_cache
    _sentiment_cache = _SentimentCache(ttl_sec=ttl_sec)


@dataclass(slots=True)
class SentimentFilterResult:
    """Result of sentiment filter evaluation."""

    original_signal: str  # The MA signal before filtering
    filtered_signal: str  # The signal after sentiment filter
    sentiment_score: float  # Current aggregate sentiment
    blocked: bool  # Whether the filter blocked the trade
    reason: str  # Human-readable explanation


def apply_sentiment_filter(
    signal: str,
    symbol: str,
    buy_threshold: float = -0.3,
    sell_threshold: float = 0.3,
    provider: str = "rss",
    api_key: str = "",
    keywords: list[str] | None = None,
) -> SentimentFilterResult:
    """Apply sentiment filter to a trading signal.

    Rules:
    - Block BUY if aggregate sentiment is below buy_threshold (very negative news)
    - Block SELL if aggregate sentiment is above sell_threshold (very positive news)
    - HOLD signals pass through unchanged

    Args:
        signal: Original signal from MA strategy ("BUY", "SELL", or "HOLD").
        symbol: Stock ticker symbol.
        buy_threshold: Block BUY if sentiment below this (default -0.3).
        sell_threshold: Block SELL if sentiment above this (default 0.3).
        provider: News provider ("rss" or "alphavantage").
        api_key: API key if needed.
        keywords: Extra search keywords.

    Returns:
        SentimentFilterResult with filtered signal and metadata.
    """
    # If signal is HOLD, no need to check sentiment
    if signal == "HOLD":
        return SentimentFilterResult(
            original_signal="HOLD",
            filtered_signal="HOLD",
            sentiment_score=0.0,
            blocked=False,
            reason="No action to filter",
        )

    # Check cache first
    cached_score = _sentiment_cache.get(symbol)
    if cached_score is not None:
        score = cached_score
        return _evaluate_sentiment(signal, score, buy_threshold, sell_threshold, cached=True)

    # Fetch news and compute sentiment
    try:
        articles = fetch_news(
            symbol=symbol,
            provider=provider,
            api_key=api_key,
            max_articles=15,
            keywords=keywords,
        )
    except Exception as exc:
        logger.warning("Sentiment filter: news fetch failed (%s), passing signal through.", exc)
        return SentimentFilterResult(
            original_signal=signal,
            filtered_signal=signal,
            sentiment_score=0.0,
            blocked=False,
            reason=f"News fetch failed: {exc}",
        )

    if not articles:
        return SentimentFilterResult(
            original_signal=signal,
            filtered_signal=signal,
            sentiment_score=0.0,
            blocked=False,
            reason="No articles found, passing signal through",
        )

    analyzer = SentimentAnalyzer()
    aggregate = analyzer.aggregate_sentiment(articles)
    score = aggregate.score

    # Cache the result
    _sentiment_cache.put(symbol, score)

    return _evaluate_sentiment(signal, score, buy_threshold, sell_threshold, cached=False)


def _evaluate_sentiment(
    signal: str,
    score: float,
    buy_threshold: float,
    sell_threshold: float,
    cached: bool = False,
) -> SentimentFilterResult:
    """Apply sentiment thresholds and return filter result."""
    source_note = " (cached)" if cached else ""

    if signal == "BUY" and score < buy_threshold:
        return SentimentFilterResult(
            original_signal="BUY",
            filtered_signal="HOLD",
            sentiment_score=score,
            blocked=True,
            reason=(
                f"BUY blocked: sentiment {score:.3f} < threshold {buy_threshold}{source_note}"
            ),
        )

    if signal == "SELL" and score > sell_threshold:
        return SentimentFilterResult(
            original_signal="SELL",
            filtered_signal="HOLD",
            sentiment_score=score,
            blocked=True,
            reason=(
                f"SELL blocked: sentiment {score:.3f} > threshold {sell_threshold}{source_note}"
            ),
        )

    return SentimentFilterResult(
        original_signal=signal,
        filtered_signal=signal,
        sentiment_score=score,
        blocked=False,
        reason=f"Sentiment {score:.3f} within thresholds, signal passes{source_note}",
    )
