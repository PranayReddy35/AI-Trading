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

    # --- Multi-symbol settings ---
    # Comma-separated list of symbols to trade (overrides symbol if set)
    symbols: str = ""
    # Max total positions open at once across all symbols
    max_open_positions: int = 5
    # Per-symbol capital allocation as % of equity (0 = equal weight)
    per_symbol_allocation_pct: float = 0.0

    # --- Intraday settings ---
    # Bar timeframe: "1Day", "1Hour", "30Min", "15Min", "5Min"
    bar_timeframe: str = "1Day"

    # --- Kelly criterion position sizing ---
    # Use Kelly criterion to size positions (requires win_rate and avg_win/loss)
    use_kelly_sizing: bool = False
    # Fraction of Kelly to use (0.5 = half-Kelly, safer)
    kelly_fraction: float = 0.5
    # Max shares Kelly is allowed to recommend (caps Kelly output)
    kelly_max_shares: int = 10

    # --- Trailing stop-loss ---
    # Trailing stop as % below peak price (0 = disabled)
    trailing_stop_pct: float = 0.0

    # --- Portfolio drawdown halt ---
    # Halt ALL trading if portfolio drops this % from its peak (0 = disabled)
    portfolio_drawdown_halt_pct: float = 0.0

    # --- Correlation filter ---
    # Block new positions if correlation with existing positions exceeds this (0 = disabled)
    correlation_filter_threshold: float = 0.0

    # --- Auto ML retraining ---
    # Retrain ML model every N days (0 = disabled)
    ml_retrain_days: int = 0
    # Path to save/load trained ML model
    ml_model_path: str = "models/ensemble.joblib"

    # --- Daily summary ---
    # Send daily summary notification at this UTC hour (HH:MM, "" = disabled)
    daily_summary_time: str = ""

    # --- EOD close ---
    # Automatically close all positions N minutes before market close (0 = disabled)
    close_before_eod: int = 0

    # --- Gap-open protection ---
    # If price gaps more than this % from prior close on open, skip/exit the position (0 = disabled)
    gap_open_protection_pct: float = 0.0

    # --- Buy-the-dip strategy ---
    # Enable dip-buying: buy when RSI oversold + price pulled back from recent high
    dip_buy_enabled: bool = False
    # RSI(14) must be at or below this to trigger (e.g. 35 = oversold)
    dip_rsi_threshold: float = 35.0
    # Price must have dropped at least this % from its N-day high
    dip_drop_pct: float = 5.0
    # Look this many bars back for the recent high
    dip_lookback_days: int = 20
    # Long-term MA period used as trend filter
    dip_long_ma_period: int = 50
    # Only buy dips when price is above the long-term MA (avoids falling knives)
    dip_require_uptrend: bool = True

    # --- Partial profit taking ---
    # When unrealized gain reaches this %, sell partial_profit_sell_pct of the position (0 = disabled)
    partial_profit_trigger_pct: float = 0.0
    # How much of the position to sell when trigger fires (default 50 = sell half, hold rest forever)
    partial_profit_sell_pct: float = 50.0

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
            symbols=os.getenv("BOT_SYMBOLS", ""),
            max_open_positions=max(1, int(os.getenv("BOT_MAX_OPEN_POSITIONS", "5"))),
            per_symbol_allocation_pct=float(os.getenv("BOT_PER_SYMBOL_ALLOCATION_PCT", "0")),
            bar_timeframe=os.getenv("BOT_BAR_TIMEFRAME", "1Day"),
            use_kelly_sizing=os.getenv("BOT_USE_KELLY_SIZING", "false").lower() == "true",
            kelly_fraction=float(os.getenv("BOT_KELLY_FRACTION", "0.5")),
            kelly_max_shares=max(1, int(os.getenv("BOT_KELLY_MAX_SHARES", "10"))),
            trailing_stop_pct=float(os.getenv("BOT_TRAILING_STOP_PCT", "0")),
            portfolio_drawdown_halt_pct=float(os.getenv("BOT_PORTFOLIO_DRAWDOWN_HALT_PCT", "0")),
            correlation_filter_threshold=float(os.getenv("BOT_CORRELATION_FILTER_THRESHOLD", "0")),
            ml_retrain_days=max(0, int(os.getenv("BOT_ML_RETRAIN_DAYS", "0"))),
            ml_model_path=os.getenv("BOT_ML_MODEL_PATH", "models/ensemble.joblib"),
            daily_summary_time=os.getenv("BOT_DAILY_SUMMARY_TIME", ""),
            close_before_eod=max(0, int(os.getenv("BOT_CLOSE_BEFORE_EOD", "0"))),
            gap_open_protection_pct=float(os.getenv("BOT_GAP_OPEN_PROTECTION_PCT", "0")),
            dip_buy_enabled=os.getenv("BOT_DIP_BUY_ENABLED", "false").lower() == "true",
            dip_rsi_threshold=float(os.getenv("BOT_DIP_RSI_THRESHOLD", "35")),
            dip_drop_pct=float(os.getenv("BOT_DIP_DROP_PCT", "5")),
            dip_lookback_days=max(5, int(os.getenv("BOT_DIP_LOOKBACK_DAYS", "20"))),
            dip_long_ma_period=max(10, int(os.getenv("BOT_DIP_LONG_MA_PERIOD", "50"))),
            dip_require_uptrend=os.getenv("BOT_DIP_REQUIRE_UPTREND", "true").lower() == "true",
            partial_profit_trigger_pct=float(os.getenv("BOT_PARTIAL_PROFIT_TRIGGER_PCT", "0")),
            partial_profit_sell_pct=max(1.0, min(99.0, float(os.getenv("BOT_PARTIAL_PROFIT_SELL_PCT", "50")))),
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
        if self.kelly_fraction <= 0 or self.kelly_fraction > 1:
            raise ValueError("kelly_fraction must be in (0, 1].")
        if self.trailing_stop_pct < 0 or self.trailing_stop_pct > 50:
            raise ValueError("trailing_stop_pct must be 0-50.")

    def get_symbols(self) -> list[str]:
        """Return list of symbols to trade (multi-symbol or single)."""
        if self.symbols:
            return [s.strip().upper() for s in self.symbols.split(",") if s.strip()]
        return [self.symbol.upper()]

    @property
    def is_live(self) -> bool:
        return not self.paper_only
