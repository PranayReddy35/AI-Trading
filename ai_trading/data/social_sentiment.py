"""Social media sentiment data fetcher using free sources.

Supported providers:
- "reddit" (default): Fetches from Reddit public JSON API — no API key needed, rate-limited.
  Searches popular finance subreddits (r/wallstreetbets, r/stocks, r/investing, r/options).

Note on other platforms:
- Twitter/X: Free API tier does NOT support reading tweets (only posting). Not viable.
- Discord: Requires bot membership in specific servers. No public API for sentiment.

All providers normalize output to NewsArticle dataclass instances (same as news_sentiment.py).
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from datetime import datetime, timezone
from typing import Sequence
from urllib.error import URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from ai_trading.data.news_sentiment import NewsArticle, _categorize_headline

logger = logging.getLogger(__name__)

# Timeout for HTTP requests in seconds
_HTTP_TIMEOUT = 15

# Rate limit: Reddit asks for max 1 request per 2 seconds for non-OAuth
_REDDIT_RATE_LIMIT_SEC = 2.0

# Popular finance subreddits for stock sentiment
_FINANCE_SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "options",
    "stockmarket",
]

_REDDIT_USER_AGENT = "AI-Trading-Bot/0.1 (educational project)"


# ---------------------------------------------------------------------------
# Reddit public JSON API (free, no API key)
# ---------------------------------------------------------------------------


def fetch_reddit_posts(
    symbol: str,
    max_posts: int = 20,
    subreddits: Sequence[str] | None = None,
    time_filter: str = "week",
) -> list[NewsArticle]:
    """Fetch posts mentioning a stock ticker from Reddit finance subreddits.

    Uses Reddit's public JSON API (append .json to any URL). This is completely
    free with no API key required. Rate limit: ~30 requests/minute for
    unauthenticated access.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "SPY").
        max_posts: Maximum number of posts to return total.
        subreddits: List of subreddit names to search. Defaults to popular finance subs.
        time_filter: Time filter for search: "hour", "day", "week", "month", "year", "all".

    Returns:
        List of NewsArticle instances sorted by timestamp (newest first).
    """
    if subreddits is None:
        subreddits = _FINANCE_SUBREDDITS

    articles: list[NewsArticle] = []
    posts_per_sub = max(3, max_posts // len(subreddits))

    for subreddit in subreddits:
        if len(articles) >= max_posts:
            break

        sub_articles = _fetch_subreddit_search(
            symbol=symbol,
            subreddit=subreddit,
            limit=posts_per_sub,
            time_filter=time_filter,
        )
        articles.extend(sub_articles)

        # Respect Reddit rate limit
        time.sleep(_REDDIT_RATE_LIMIT_SEC)

    # Deduplicate by URL
    seen_urls: set[str] = set()
    unique_articles: list[NewsArticle] = []
    for article in articles:
        if article.url not in seen_urls:
            seen_urls.add(article.url)
            unique_articles.append(article)

    # Sort by timestamp (newest first) and limit
    unique_articles.sort(key=lambda a: a.timestamp, reverse=True)
    return unique_articles[:max_posts]


def _fetch_subreddit_search(
    symbol: str,
    subreddit: str,
    limit: int = 10,
    time_filter: str = "week",
) -> list[NewsArticle]:
    """Search a specific subreddit for posts about a ticker.

    Args:
        symbol: Stock ticker to search for.
        subreddit: Subreddit name (without r/ prefix).
        limit: Max posts to fetch from this subreddit.
        time_filter: Reddit time filter.

    Returns:
        List of NewsArticle instances from this subreddit.
    """
    # Search for ticker symbol (e.g., "$AAPL" or "AAPL")
    query = f"{symbol} OR ${symbol}"
    url = (
        f"https://www.reddit.com/r/{subreddit}/search.json"
        f"?q={quote_plus(query)}&restrict_sr=on&sort=new"
        f"&t={time_filter}&limit={limit}"
    )

    articles: list[NewsArticle] = []
    try:
        req = Request(url, headers={"User-Agent": _REDDIT_USER_AGENT})
        with urlopen(req, timeout=_HTTP_TIMEOUT) as response:
            data = json.loads(response.read())

        posts = data.get("data", {}).get("children", [])
        for post in posts:
            post_data = post.get("data", {})
            title = post_data.get("title", "").strip()
            if not title:
                continue

            # Verify ticker is actually mentioned (not just partial match)
            if not _ticker_mentioned(symbol, title, post_data.get("selftext", "")):
                continue

            # Parse timestamp (Reddit uses Unix epoch)
            created_utc = post_data.get("created_utc", 0)
            if not created_utc:
                logger.debug("Skipping Reddit post with missing timestamp: %s", title[:50])
                continue
            timestamp = datetime.fromtimestamp(created_utc, tz=timezone.utc)

            # Use upvote ratio and score as relevance signal
            score = post_data.get("score", 1)
            upvote_ratio = post_data.get("upvote_ratio", 0.5)
            num_comments = post_data.get("num_comments", 0)

            # Relevance: combination of engagement metrics (normalized 0-1)
            relevance = _compute_reddit_relevance(score, upvote_ratio, num_comments)

            permalink = post_data.get("permalink", "")
            post_url = f"https://www.reddit.com{permalink}" if permalink else ""

            category = _categorize_headline(title)
            # Override category if it's clearly social/discussion
            if category == "general":
                category = "social"

            articles.append(
                NewsArticle(
                    headline=title,
                    source=f"r/{subreddit}",
                    timestamp=timestamp,
                    url=post_url,
                    relevance_score=relevance,
                    category=category,
                )
            )

    except (URLError, OSError, json.JSONDecodeError, KeyError) as exc:
        logger.warning("Reddit fetch from r/%s failed: %s", subreddit, exc)

    return articles


def _ticker_mentioned(symbol: str, title: str, body: str) -> bool:
    """Check if a ticker is genuinely mentioned (not just a partial word match).

    Looks for $SYMBOL, standalone SYMBOL (word boundary), or symbol in common
    stock discussion patterns.
    """
    text = f"{title} {body}"
    # Match $AAPL or standalone AAPL with word boundaries
    pattern = rf"(?:\$|\b){re.escape(symbol)}(?:\b|[^a-zA-Z])"
    return bool(re.search(pattern, text, re.IGNORECASE))


def _compute_reddit_relevance(score: int, upvote_ratio: float, num_comments: int) -> float:
    """Compute a relevance score (0.0-1.0) from Reddit engagement metrics.

    Higher score, upvote ratio, and comment count indicate more relevant/impactful posts.
    """
    # Normalize score: log scale, cap at ~1000 upvotes
    score_norm = min(1.0, math.log1p(max(score, 0)) / math.log1p(1000))

    # Upvote ratio already 0-1
    ratio_norm = max(0.0, upvote_ratio)

    # Comment count: log scale, cap at ~500 comments
    comment_norm = min(1.0, math.log1p(max(num_comments, 0)) / math.log1p(500))

    # Weighted combination
    relevance = 0.4 * score_norm + 0.3 * ratio_norm + 0.3 * comment_norm
    return round(max(0.1, min(1.0, relevance)), 3)


# ---------------------------------------------------------------------------
# Unified social media fetch interface
# ---------------------------------------------------------------------------


def fetch_social_sentiment(
    symbol: str,
    providers: Sequence[str] | None = None,
    max_posts: int = 20,
    subreddits: Sequence[str] | None = None,
    time_filter: str = "week",
) -> list[NewsArticle]:
    """Unified social media sentiment fetcher.

    Currently supports Reddit (free, no API key). Twitter/X and Discord are
    not available as free-of-cost read APIs.

    Args:
        symbol: Stock ticker symbol.
        providers: List of providers to use. Currently only ["reddit"] is supported.
        max_posts: Maximum posts to return.
        subreddits: Custom subreddit list (Reddit only). Defaults to finance subs.
        time_filter: Time filter for Reddit search.

    Returns:
        List of NewsArticle instances, newest first.
    """
    if providers is None:
        providers = ["reddit"]

    all_articles: list[NewsArticle] = []

    for provider in providers:
        if provider == "reddit":
            articles = fetch_reddit_posts(
                symbol=symbol,
                max_posts=max_posts,
                subreddits=subreddits,
                time_filter=time_filter,
            )
            all_articles.extend(articles)
        else:
            logger.warning(
                "Social media provider '%s' is not supported (free tier). "
                "Currently only 'reddit' is available at no cost.",
                provider,
            )

    # Sort all by timestamp and deduplicate
    all_articles.sort(key=lambda a: a.timestamp, reverse=True)
    return all_articles[:max_posts]
