"""News sentiment data fetcher using free sources.

Supported providers:
- "rss" (default): Fetches from Google News RSS — no API key needed, unlimited.
- "alphavantage": Uses Alpha Vantage News Sentiment endpoint — free tier (25 req/day).

All providers normalize output to a list of NewsArticle dataclass instances.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import unescape
from typing import Sequence
from urllib.error import URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# Timeout for HTTP requests in seconds
_HTTP_TIMEOUT = 15


@dataclass(slots=True)
class NewsArticle:
    """Normalized news article from any provider."""

    headline: str
    source: str
    timestamp: datetime
    url: str = ""
    relevance_score: float = 1.0
    # Categories: general, earnings, politics, ceo, analyst, macro
    category: str = "general"


def _clean_html(text: str) -> str:
    """Strip HTML tags and unescape entities."""
    clean = re.sub(r"<[^>]+>", "", text)
    return unescape(clean).strip()


# ---------------------------------------------------------------------------
# Google News RSS (free, no API key)
# ---------------------------------------------------------------------------


def fetch_google_news_rss(
    symbol: str,
    max_articles: int = 20,
    keywords: Sequence[str] | None = None,
) -> list[NewsArticle]:
    """Fetch news from Google News RSS feed.

    This is completely free with no API key required.
    Covers: general news, CEO statements, political news, earnings, etc.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "SPY").
        max_articles: Maximum number of articles to return.
        keywords: Additional search keywords to include (e.g., ["CEO", "earnings"]).

    Returns:
        List of NewsArticle instances sorted by timestamp (newest first).
    """
    # Build search query: symbol + optional keywords
    query_parts = [symbol, "stock"]
    if keywords:
        query_parts.extend(keywords)
    query = " ".join(query_parts)

    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"

    articles: list[NewsArticle] = []
    try:
        req = Request(url, headers={"User-Agent": "AI-Trading-Bot/0.1"})
        with urlopen(req, timeout=_HTTP_TIMEOUT) as response:
            data = response.read()

        root = ET.fromstring(data)
        channel = root.find("channel")
        if channel is None:
            return articles

        for item in channel.findall("item")[:max_articles]:
            title_el = item.find("title")
            pub_date_el = item.find("pubDate")
            link_el = item.find("link")
            source_el = item.find("source")

            headline = _clean_html(title_el.text) if title_el is not None and title_el.text else ""
            if not headline:
                continue

            # Parse publication date
            timestamp = datetime.now(timezone.utc)
            if pub_date_el is not None and pub_date_el.text:
                try:
                    timestamp = _parse_rss_date(pub_date_el.text)
                except (ValueError, TypeError):
                    pass

            source = ""
            if source_el is not None and source_el.text:
                source = source_el.text
            elif source_el is not None:
                source = source_el.get("url", "Google News")

            link = link_el.text if link_el is not None and link_el.text else ""

            category = _categorize_headline(headline)

            articles.append(
                NewsArticle(
                    headline=headline,
                    source=source or "Google News",
                    timestamp=timestamp,
                    url=link,
                    relevance_score=1.0,
                    category=category,
                )
            )

    except (URLError, ET.ParseError, OSError) as exc:
        logger.warning("Google News RSS fetch failed: %s", exc)

    return sorted(articles, key=lambda a: a.timestamp, reverse=True)


def _parse_rss_date(date_str: str) -> datetime:
    """Parse RSS pubDate format (RFC 822)."""
    # Example: "Mon, 26 May 2025 14:30:00 GMT"
    from email.utils import parsedate_to_datetime

    return parsedate_to_datetime(date_str).astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Alpha Vantage News Sentiment (free tier: 25 requests/day)
# ---------------------------------------------------------------------------


def fetch_alphavantage_news(
    symbol: str,
    api_key: str,
    max_articles: int = 20,
) -> list[NewsArticle]:
    """Fetch news from Alpha Vantage News Sentiment API (free tier).

    Free tier allows 25 requests/day with up to 50 articles per request.
    Provides relevance scores and sentiment pre-computed.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL").
        api_key: Alpha Vantage API key (free at alphavantage.co).
        max_articles: Maximum number of articles to return.

    Returns:
        List of NewsArticle instances sorted by timestamp (newest first).
    """
    import json

    url = (
        f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT"
        f"&tickers={quote_plus(symbol)}&limit={max_articles}&apikey={api_key}"
    )

    articles: list[NewsArticle] = []
    try:
        req = Request(url, headers={"User-Agent": "AI-Trading-Bot/0.1"})
        with urlopen(req, timeout=_HTTP_TIMEOUT) as response:
            data = json.loads(response.read())

        feed = data.get("feed", [])
        for item in feed[:max_articles]:
            headline = item.get("title", "")
            if not headline:
                continue

            # Parse timestamp: "20250526T143000"
            time_str = item.get("time_published", "")
            timestamp = datetime.now(timezone.utc)
            if time_str:
                try:
                    timestamp = datetime.strptime(time_str[:15], "%Y%m%dT%H%M%S").replace(
                        tzinfo=timezone.utc
                    )
                except (ValueError, TypeError):
                    pass

            # Extract relevance score for this specific ticker
            relevance = 1.0
            ticker_sentiments = item.get("ticker_sentiment", [])
            for ts in ticker_sentiments:
                if ts.get("ticker", "").upper() == symbol.upper():
                    relevance = float(ts.get("relevance_score", 1.0))
                    break

            source = item.get("source", "Alpha Vantage")
            link = item.get("url", "")
            category = _categorize_headline(headline)

            articles.append(
                NewsArticle(
                    headline=headline,
                    source=source,
                    timestamp=timestamp,
                    url=link,
                    relevance_score=relevance,
                    category=category,
                )
            )

    except (URLError, OSError, json.JSONDecodeError, KeyError) as exc:
        logger.warning("Alpha Vantage news fetch failed: %s", exc)

    return sorted(articles, key=lambda a: a.timestamp, reverse=True)


# ---------------------------------------------------------------------------
# Unified fetch interface
# ---------------------------------------------------------------------------


def fetch_news(
    symbol: str,
    provider: str = "rss",
    api_key: str = "",
    max_articles: int = 20,
    keywords: Sequence[str] | None = None,
) -> list[NewsArticle]:
    """Unified news fetcher supporting multiple free providers.

    Args:
        symbol: Stock ticker symbol.
        provider: "rss" (Google News, free, no key), "alphavantage" (free tier),
                  or "reddit" (free, no key — social media sentiment).
        api_key: API key (required for alphavantage only).
        max_articles: Maximum articles to return.
        keywords: Extra search keywords (used by RSS provider).

    Returns:
        List of NewsArticle instances, newest first.
    """
    if provider == "alphavantage":
        if not api_key:
            logger.warning("Alpha Vantage requires an API key. Falling back to RSS.")
            return fetch_google_news_rss(symbol, max_articles, keywords)
        return fetch_alphavantage_news(symbol, api_key, max_articles)

    if provider == "reddit":
        from ai_trading.data.social_sentiment import fetch_reddit_posts

        return fetch_reddit_posts(symbol, max_posts=max_articles)

    # Default: RSS (Google News)
    return fetch_google_news_rss(symbol, max_articles, keywords)


# ---------------------------------------------------------------------------
# Helper: categorize headlines
# ---------------------------------------------------------------------------

_CATEGORY_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("earnings", re.compile(r"\b(earnings|revenue|profit|quarterly|EPS|guidance)\b", re.IGNORECASE)),
    ("ceo", re.compile(r"\b(CEO|chief executive|executive officer|founder)\b", re.IGNORECASE)),
    ("politics", re.compile(
        r"\b(president|congress|senate|regulation|tariff|sanctions?|fed\b|federal reserve|"
        r"politician|policy|legislation|white house|treasury)\b",
        re.IGNORECASE,
    )),
    ("analyst", re.compile(r"\b(upgrade|downgrade|price target|analyst|rating)\b", re.IGNORECASE)),
    ("macro", re.compile(
        r"\b(inflation|GDP|unemployment|interest rate|CPI|jobs report|recession)\b",
        re.IGNORECASE,
    )),
    ("social", re.compile(
        r"\b(YOLO|diamond hands|to the moon|squeeze|apes?|tendies|DD|due diligence|WSB)\b",
        re.IGNORECASE,
    )),
]


def _categorize_headline(headline: str) -> str:
    """Categorize a headline based on keyword patterns."""
    for category, pattern in _CATEGORY_PATTERNS:
        if pattern.search(headline):
            return category
    return "general"
