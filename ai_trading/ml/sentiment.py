"""Sentiment analysis module using VADER (free, no API key, no GPU).

Provides:
- Individual headline scoring (positive/negative/neutral + compound)
- Aggregate daily sentiment from multiple articles
- Weighted scoring by article category (CEO, politics, earnings get higher weight)
- Rolling sentiment features for ML integration

VADER (Valence Aware Dictionary and sEntiment Reasoner) is specifically tuned
for social media and news text, making it suitable for financial headline analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ai_trading.data.news_sentiment import NewsArticle

logger = logging.getLogger(__name__)

# Category weights: how much each news category should influence the final score.
# Earnings and CEO statements are most directly market-moving.
CATEGORY_WEIGHTS: dict[str, float] = {
    "earnings": 1.5,
    "ceo": 1.3,
    "analyst": 1.2,
    "politics": 1.1,
    "macro": 1.1,
    "social": 0.8,  # Social media posts are noisier, lower weight
    "general": 1.0,
}


@dataclass(slots=True)
class SentimentScore:
    """Sentiment score for a single article."""

    headline: str
    compound: float  # -1.0 to +1.0 (overall sentiment)
    positive: float  # 0.0 to 1.0
    negative: float  # 0.0 to 1.0
    neutral: float  # 0.0 to 1.0
    category: str
    weight: float  # Combined weight (category * relevance)


@dataclass(slots=True)
class AggregateSentiment:
    """Aggregated sentiment from multiple articles."""

    score: float  # Weighted average compound score (-1.0 to +1.0)
    num_articles: int
    num_positive: int
    num_negative: int
    num_neutral: int
    strongest_positive: str  # Most positive headline
    strongest_negative: str  # Most negative headline
    category_breakdown: dict[str, float]  # Average score per category


class SentimentAnalyzer:
    """Financial news sentiment analyzer using VADER.

    VADER is well-suited for short text (headlines) and requires no
    API keys, GPU, or large model downloads.
    """

    def __init__(self) -> None:
        self._analyzer = SentimentIntensityAnalyzer()

    def score_headline(self, article: NewsArticle) -> SentimentScore:
        """Score a single news headline.

        Args:
            article: NewsArticle with headline and metadata.

        Returns:
            SentimentScore with compound, positive, negative, neutral scores.
        """
        scores = self._analyzer.polarity_scores(article.headline)
        category_weight = CATEGORY_WEIGHTS.get(article.category, 1.0)
        combined_weight = category_weight * article.relevance_score

        return SentimentScore(
            headline=article.headline,
            compound=scores["compound"],
            positive=scores["pos"],
            negative=scores["neg"],
            neutral=scores["neu"],
            category=article.category,
            weight=combined_weight,
        )

    def score_articles(self, articles: Sequence[NewsArticle]) -> list[SentimentScore]:
        """Score multiple articles.

        Args:
            articles: List of NewsArticle instances.

        Returns:
            List of SentimentScore instances.
        """
        return [self.score_headline(article) for article in articles]

    def aggregate_sentiment(
        self,
        articles: Sequence[NewsArticle],
        time_decay_hours: float = 48.0,
    ) -> AggregateSentiment:
        """Compute weighted aggregate sentiment from multiple articles.

        The aggregate score is a weighted average of individual compound scores,
        where weights are determined by article category, relevance, and recency.
        Recent articles receive higher weight via exponential time decay.

        Args:
            articles: List of NewsArticle instances.
            time_decay_hours: Half-life for time decay in hours. Articles older
                than this get exponentially less weight. Set to 0 to disable.

        Returns:
            AggregateSentiment with overall score and breakdown.
        """
        if not articles:
            return AggregateSentiment(
                score=0.0,
                num_articles=0,
                num_positive=0,
                num_negative=0,
                num_neutral=0,
                strongest_positive="",
                strongest_negative="",
                category_breakdown={},
            )

        scores = self.score_articles(articles)

        # Apply time-decay weighting: recent articles matter more
        now = datetime.now(timezone.utc)
        effective_weights = []
        for i, s in enumerate(scores):
            base_weight = s.weight
            if time_decay_hours > 0 and i < len(articles):
                article_age_hours = (now - articles[i].timestamp).total_seconds() / 3600.0
                # Exponential decay: weight halves every time_decay_hours
                decay_factor = 0.5 ** (article_age_hours / time_decay_hours)
                base_weight *= decay_factor
            effective_weights.append(base_weight)

        # Weighted average
        total_weight = sum(effective_weights)
        if total_weight == 0:
            total_weight = 1.0

        weighted_score = sum(
            s.compound * w for s, w in zip(scores, effective_weights)
        ) / total_weight

        # Counts
        num_positive = sum(1 for s in scores if s.compound >= 0.05)
        num_negative = sum(1 for s in scores if s.compound <= -0.05)
        num_neutral = len(scores) - num_positive - num_negative

        # Strongest headlines
        strongest_pos = max(scores, key=lambda s: s.compound)
        strongest_neg = min(scores, key=lambda s: s.compound)

        # Category breakdown
        category_scores: dict[str, list[float]] = {}
        for s in scores:
            category_scores.setdefault(s.category, []).append(s.compound)
        category_breakdown = {
            cat: sum(vals) / len(vals) for cat, vals in category_scores.items()
        }

        return AggregateSentiment(
            score=round(weighted_score, 4),
            num_articles=len(scores),
            num_positive=num_positive,
            num_negative=num_negative,
            num_neutral=num_neutral,
            strongest_positive=strongest_pos.headline,
            strongest_negative=strongest_neg.headline,
            category_breakdown=category_breakdown,
        )


# ---------------------------------------------------------------------------
# Feature engineering for ML integration
# ---------------------------------------------------------------------------


def compute_sentiment_features(
    articles: Sequence[NewsArticle],
    reference_time: datetime | None = None,
) -> dict[str, float]:
    """Compute sentiment features suitable for ML model integration.

    Returns a dict with:
        - sentiment_score_1d: Average sentiment of articles from last 24h
        - sentiment_score_3d: Average sentiment of articles from last 3 days
        - news_volume_1d: Number of articles in last 24h
        - news_volume_3d: Number of articles in last 3 days
        - sentiment_momentum: Change in sentiment (3d vs older)
        - positive_ratio: Fraction of positive articles
        - negative_ratio: Fraction of negative articles
        - earnings_sentiment: Average sentiment of earnings-related articles
        - political_sentiment: Average sentiment of political articles
        - ceo_sentiment: Average sentiment of CEO-related articles

    Args:
        articles: All available articles (should span at least 7 days ideally).
        reference_time: Reference point for time calculations. Defaults to now.

    Returns:
        Dict of feature name → value.
    """
    if reference_time is None:
        reference_time = datetime.now(timezone.utc)

    analyzer = SentimentAnalyzer()

    # Time windows
    cutoff_1d = reference_time - timedelta(days=1)
    cutoff_3d = reference_time - timedelta(days=3)
    cutoff_7d = reference_time - timedelta(days=7)

    articles_1d = [a for a in articles if a.timestamp >= cutoff_1d]
    articles_3d = [a for a in articles if a.timestamp >= cutoff_3d]
    articles_older = [a for a in articles if cutoff_7d <= a.timestamp < cutoff_3d]

    # Aggregate scores
    agg_1d = analyzer.aggregate_sentiment(articles_1d)
    agg_3d = analyzer.aggregate_sentiment(articles_3d)
    agg_older = analyzer.aggregate_sentiment(articles_older)

    # Sentiment momentum: recent vs older
    momentum = agg_3d.score - agg_older.score if agg_older.num_articles > 0 else 0.0

    # Ratios
    total = max(len(articles_3d), 1)
    positive_ratio = agg_3d.num_positive / total
    negative_ratio = agg_3d.num_negative / total

    # Category-specific scores
    earnings_articles = [a for a in articles_3d if a.category == "earnings"]
    political_articles = [a for a in articles_3d if a.category == "politics"]
    ceo_articles = [a for a in articles_3d if a.category == "ceo"]
    social_articles = [a for a in articles_3d if a.category == "social"]

    earnings_sent = analyzer.aggregate_sentiment(earnings_articles).score
    political_sent = analyzer.aggregate_sentiment(political_articles).score
    ceo_sent = analyzer.aggregate_sentiment(ceo_articles).score
    social_sent = analyzer.aggregate_sentiment(social_articles).score

    return {
        "sentiment_score_1d": agg_1d.score,
        "sentiment_score_3d": agg_3d.score,
        "news_volume_1d": float(len(articles_1d)),
        "news_volume_3d": float(len(articles_3d)),
        "sentiment_momentum": round(momentum, 4),
        "positive_ratio": round(positive_ratio, 4),
        "negative_ratio": round(negative_ratio, 4),
        "earnings_sentiment": earnings_sent,
        "political_sentiment": political_sent,
        "ceo_sentiment": ceo_sent,
        "social_sentiment": social_sent,
    }
