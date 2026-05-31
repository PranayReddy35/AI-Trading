from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Settings:
    api_key: str
    api_secret: str
    symbol: str = "SPY"
    fast_ma: int = 5
    slow_ma: int = 20
    lookback_days: int = 90
    max_shares: int = 1
    min_cash_threshold: float = 100.0
    max_daily_trades: int = 1
    max_consecutive_errors: int = 3
    paper_only: bool = True
    log_path: Path = Path("logs/bot.log")
    journal_path: Path = Path("logs/journal.jsonl")

    # --- Real-trading enhancements ---
    # Order type: "market" or "limit"
    order_type: str = "market"
    # For limit orders: offset from current price in % (e.g., 0.1 = 0.1% above/below)
    limit_price_offset_pct: float = 0.1
    # Stop-loss: percentage below entry to place stop (0 = disabled)
    stop_loss_pct: float = 0.0
    # Daily loss limit as % of portfolio equity (0 = disabled)
    daily_loss_limit_pct: float = 0.0
    # Max portfolio exposure as % of equity (e.g., 95.0 = never use more than 95%)
    max_portfolio_exposure_pct: float = 95.0
    # Require explicit confirmation before live orders (interactive mode)
    require_confirmation: bool = True
    # Max retries for transient API errors
    max_api_retries: int = 3
    # Seconds to wait for order fill before timeout (0 = no wait)
    order_fill_timeout_sec: int = 60
    # Webhook URL for notifications (empty = disabled)
    webhook_url: str = ""
    # Notify on: trade, error, risk_reject, daily_summary
    notify_events: list[str] = field(default_factory=lambda: ["trade", "error"])
    # Cooldown seconds between trades (prevents rapid re-entry)
    trade_cooldown_sec: int = 0
    # Pre-flight check: require minimum equity before trading
    min_equity: float = 0.0

    # --- Sentiment / News settings ---
    # News provider: "rss" (Google News, free, no key) or "alphavantage" (free tier)
    news_provider: str = "rss"
    # API key for Alpha Vantage news (free at alphavantage.co, 25 req/day)
    news_api_key: str = ""
    # Enable sentiment filter on MA strategy (blocks trades on extreme sentiment)
    use_sentiment_filter: bool = False
    # Sentiment threshold: block BUY if sentiment below this (-1 to 1)
    sentiment_buy_threshold: float = -0.3
    # Sentiment threshold: block SELL if sentiment above this (-1 to 1)
    sentiment_sell_threshold: float = 0.3
    # Include sentiment features in ML model
    use_sentiment_in_ml: bool = True
    # Extra keywords to search for news (comma-separated)
    news_keywords: str = ""
    # Social media sentiment: include Reddit posts in sentiment analysis (free, no key)
    use_social_sentiment: bool = False
    # Comma-separated subreddits to search (default: wallstreetbets,stocks,investing,options,stockmarket)
    social_subreddits: str = ""
    # Time filter for Reddit search: "hour", "day", "week", "month", "year", "all"
    social_time_filter: str = "week"

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.getenv("APCA_API_KEY_ID", "")
        api_secret = os.getenv("APCA_API_SECRET_KEY", "")
        notify_events_raw = os.getenv("BOT_NOTIFY_EVENTS", "trade,error")
        return cls(
            api_key=api_key,
            api_secret=api_secret,
            symbol=os.getenv("BOT_SYMBOL", "SPY").upper(),
            fast_ma=int(os.getenv("BOT_FAST_MA", "5")),
            slow_ma=int(os.getenv("BOT_SLOW_MA", "20")),
            lookback_days=int(os.getenv("BOT_LOOKBACK_DAYS", "90")),
            max_shares=max(1, int(os.getenv("BOT_MAX_SHARES", "1"))),
            min_cash_threshold=float(os.getenv("BOT_MIN_CASH_THRESHOLD", "100")),
            max_daily_trades=max(1, int(os.getenv("BOT_MAX_DAILY_TRADES", "1"))),
            max_consecutive_errors=max(1, int(os.getenv("BOT_MAX_CONSECUTIVE_ERRORS", "3"))),
            paper_only=os.getenv("BOT_PAPER_ONLY", "true").lower() == "true",
            log_path=Path(os.getenv("BOT_LOG_PATH", "logs/bot.log")),
            journal_path=Path(os.getenv("BOT_JOURNAL_PATH", "logs/journal.jsonl")),
            order_type=os.getenv("BOT_ORDER_TYPE", "market").lower(),
            limit_price_offset_pct=float(os.getenv("BOT_LIMIT_PRICE_OFFSET_PCT", "0.1")),
            stop_loss_pct=float(os.getenv("BOT_STOP_LOSS_PCT", "0")),
            daily_loss_limit_pct=float(os.getenv("BOT_DAILY_LOSS_LIMIT_PCT", "0")),
            max_portfolio_exposure_pct=float(os.getenv("BOT_MAX_PORTFOLIO_EXPOSURE_PCT", "95")),
            require_confirmation=os.getenv("BOT_REQUIRE_CONFIRMATION", "true").lower() == "true",
            max_api_retries=max(1, int(os.getenv("BOT_MAX_API_RETRIES", "3"))),
            order_fill_timeout_sec=max(0, int(os.getenv("BOT_ORDER_FILL_TIMEOUT_SEC", "60"))),
            webhook_url=os.getenv("BOT_WEBHOOK_URL", ""),
            notify_events=[e.strip() for e in notify_events_raw.split(",") if e.strip()],
            trade_cooldown_sec=max(0, int(os.getenv("BOT_TRADE_COOLDOWN_SEC", "0"))),
            min_equity=float(os.getenv("BOT_MIN_EQUITY", "0")),
            news_provider=os.getenv("BOT_NEWS_PROVIDER", "rss").lower(),
            news_api_key=os.getenv("BOT_NEWS_API_KEY", ""),
            use_sentiment_filter=os.getenv("BOT_USE_SENTIMENT_FILTER", "false").lower() == "true",
            sentiment_buy_threshold=float(os.getenv("BOT_SENTIMENT_BUY_THRESHOLD", "-0.3")),
            sentiment_sell_threshold=float(os.getenv("BOT_SENTIMENT_SELL_THRESHOLD", "0.3")),
            use_sentiment_in_ml=os.getenv("BOT_USE_SENTIMENT_IN_ML", "true").lower() == "true",
            news_keywords=os.getenv("BOT_NEWS_KEYWORDS", ""),
            use_social_sentiment=os.getenv("BOT_USE_SOCIAL_SENTIMENT", "false").lower() == "true",
            social_subreddits=os.getenv("BOT_SOCIAL_SUBREDDITS", ""),
            social_time_filter=os.getenv("BOT_SOCIAL_TIME_FILTER", "week").lower(),
        )

    def validate(self) -> None:
        if not self.api_key or not self.api_secret:
            raise ValueError("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY.")
        if self.fast_ma <= 0 or self.slow_ma <= 0 or self.fast_ma >= self.slow_ma:
            raise ValueError("Require 0 < fast_ma < slow_ma.")
        if self.order_type not in ("market", "limit"):
            raise ValueError("order_type must be 'market' or 'limit'.")
        if self.daily_loss_limit_pct < 0 or self.daily_loss_limit_pct > 100:
            raise ValueError("daily_loss_limit_pct must be 0-100.")
        if self.max_portfolio_exposure_pct <= 0 or self.max_portfolio_exposure_pct > 100:
            raise ValueError("max_portfolio_exposure_pct must be >0 and <=100.")
        if self.stop_loss_pct < 0 or self.stop_loss_pct > 50:
            raise ValueError("stop_loss_pct must be 0-50.")

    @property
    def is_live(self) -> bool:
        return not self.paper_only
