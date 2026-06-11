from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from ai_trading.env import load_dotenv


@dataclass(slots=True)
class Settings:
    api_key: str
    api_secret: str
    broker: str = "alpaca"
    robinhood_agentic_enabled: bool = False
    robinhood_agentic_account_number: str = ""
    robinhood_agentic_buying_power: float = 0.0
    robinhood_agentic_equity: float = 0.0
    robinhood_order_intents_path: str = "logs/robinhood_order_intents.jsonl"
    robinhood_use_dollar_orders: bool = False
    robinhood_dollar_amount_per_trade: float = 0.0
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
    # Dry-run stock orders: log intended stock trades without submitting them.
    stock_dry_run: bool = False
    # Global safety stop: when true, no new stock orders are submitted.
    kill_switch: bool = False
    # Send a Discord/journal preview before submitting stock orders.
    notify_trade_preview: bool = True
    # Require latest-price metadata before submitting stock orders.
    require_fresh_price_for_orders: bool = True
    # Max allowed latest-price age for order decisions.
    max_latest_price_age_sec: int = 300
    # Block SELL orders too when latest price is stale; default keeps exits possible.
    stale_price_blocks_sell: bool = False
    # Block BUY orders when latest stock price comes from fallback/caution feeds.
    block_caution_feeds_for_buys: bool = False
    # Max BUY submissions in one bot run/cycle (0 = disabled).
    max_buys_per_cycle: int = 0
    # Force an exit when an open symbol loss reaches this % (0 = disabled).
    max_symbol_loss_pct: float = 0.0
    # Block new BUYs after a gap-up larger than this % from prior bar close (0 = disabled).
    block_buy_gap_up_pct: float = 0.0
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
    # Historical win rate for Kelly (0-1)
    kelly_win_rate: float = 0.52
    # Average winning trade return (positive decimal, e.g. 0.015 = 1.5%)
    kelly_avg_win: float = 0.015
    # Average losing trade return (positive decimal, e.g. 0.01 = 1%)
    kelly_avg_loss: float = 0.01

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
    # Max bars to hold remainder after partial profit (0 = hold indefinitely)
    partial_profit_max_hold_bars: int = 0
    # Trailing stop % for remainder after partial profit (0 = disabled)
    partial_profit_trailing_stop_pct: float = 0.0

    # --- Error streak decay ---
    # Hours after which consecutive error streak resets (0 = never resets until manual clear)
    error_streak_decay_hours: float = 0.0

    # --- Strategy mode ---
    # Which strategy to use for signal generation: "ma" (basic MA) or "ensemble" (multi-strategy)
    strategy_mode: str = "ma"

    # --- Risk state persistence ---
    # Path to persist risk manager state (empty = no persistence)
    risk_state_file: str = "logs/risk_state.json"

    # --- Data caching ---
    # Cache TTL in seconds for market data (0 = no caching)
    data_cache_ttl_sec: int = 300
    # Pull latest IEX quote/trade and patch the final bar before signal/sizing.
    use_latest_price: bool = True
    # Stock data feed: auto, sip, delayed_sip, iex, boats, overnight, otc.
    market_data_feed: str = "auto"

    # --- Sentiment caching ---
    # Cache TTL in seconds for sentiment data (0 = no caching)
    sentiment_cache_ttl_sec: int = 600

    # --- ML model staleness ---
    # Alert if model file is older than this many days (0 = disabled)
    ml_model_max_age_days: int = 0

    # --- ATR-based risk sizing (new) ---
    use_atr_stops: bool = False
    atr_stop_mult: float = 2.0
    atr_period: int = 14
    risk_per_trade_pct: float = 0.5  # % of equity to risk on entry→stop move

    # --- Adaptive thresholds + ensemble (new) ---
    use_adaptive_thresholds: bool = False
    base_buy_threshold: float = 0.15
    base_sell_threshold: float = -0.15
    use_ensemble_signal: bool = False  # if True, use ensemble instead of MA crossover

    # --- Multi-timeframe confirmation (new) ---
    use_mtf_confirmation: bool = False
    mtf_timeframes: str = "1Day,1Hour"

    # --- Bar cache (new) ---
    cache_enabled: bool = False
    cache_dir: str = ".cache/bars"
    cache_ttl_sec: int = 60

    # --- Correlation-aware size scaling (new, soft alternative to filter) ---
    correlation_scale_soft: float = 0.6
    correlation_scale_hard: float = 0.9
    use_correlation_scaling: bool = False

    # --- Macro filters (new) ---
    use_spy_trend_filter: bool = False
    spy_trend_window: int = 200
    use_vix_size_scaling: bool = False
    vix_full_below: float = 20.0
    vix_half_above: float = 25.0
    vix_zero_above: float = 35.0
    earnings_blackout_days: int = 0
    use_volume_confirmation: bool = False
    volume_min_ratio: float = 0.8
    use_spread_filter: bool = False
    max_spread_bps: float = 10.0

    # --- Portfolio sizing & exits (new) ---
    use_vol_targeting: bool = False
    target_vol_pct: float = 1.0
    max_position_pct: float = 20.0
    max_portfolio_heat_pct: float = 6.0
    use_trailing_atr_stop: bool = False
    trailing_atr_mult: float = 2.5
    time_stop_max_bars: int = 0
    time_stop_min_r: float = 0.5
    partial_take_r: float = 1.0      # R-multiple to take partial profit
    breakeven_r: float = 1.0          # R-multiple to move stop to breakeven

    # --- Meta-labeling (new) ---
    use_meta_label: bool = False
    meta_model_path: str = "models/meta_label.joblib"
    meta_min_prob: float = 0.55
    meta_size_scale: bool = False    # scale qty by predicted probability

    # --- Options trading (new) ---
    options_enabled: bool = False
    options_strategies: str = "long_call,csp,bull_call"
    options_qty: int = 1
    options_min_pop: float = 0.60
    options_max_risk_pct: float = 1.0       # % of options BP per trade
    options_min_dte: int = 21
    options_max_dte: int = 45
    options_target_delta: float = 0.30
    options_spread_width: float = 5.0
    options_data_source: str = "auto"        # 'auto' | 'alpaca' | 'yfinance'
    options_top_n: int = 5
    options_slippage_pct: float = 0.0
    options_allow_naked: bool = False
    options_dry_run: bool = True             # safe default: log, don't submit
    research_auto_queue: bool = False
    research_queue_limit: int = 5
    research_mode: str = "memo"
    research_goals: str = "long-term capital appreciation"
    research_risk_tolerance: str = "moderate"
    research_time_horizon: str = "5+ years"
    robinhood_refresh_on_open: bool = True
    robinhood_rotation_enabled: bool = False
    robinhood_rotation_buy_limit: int = 2
    robinhood_rotation_min_buy_score: float = 65.0
    robinhood_rotation_trim_score_max: float = 45.0
    robinhood_rotation_exit_score_max: float = 35.0
    robinhood_execution_mode: str = "intent_only"
    robinhood_auto_approve: bool = False
    robinhood_auto_dispatch: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        api_key = os.getenv("APCA_API_KEY_ID", "")
        api_secret = os.getenv("APCA_API_SECRET_KEY", "")
        notify_events_raw = os.getenv("BOT_NOTIFY_EVENTS", "trade,error")
        return cls(
            api_key=api_key,
            api_secret=api_secret,
            broker=os.getenv("BOT_BROKER", "alpaca").strip().lower() or "alpaca",
            robinhood_agentic_enabled=os.getenv("ROBINHOOD_AGENTIC_ENABLED", "false").lower() == "true",
            robinhood_agentic_account_number=os.getenv("ROBINHOOD_AGENTIC_ACCOUNT_NUMBER", "").strip(),
            robinhood_agentic_buying_power=float(os.getenv("ROBINHOOD_AGENTIC_BUYING_POWER", "0")),
            robinhood_agentic_equity=float(os.getenv("ROBINHOOD_AGENTIC_EQUITY", "0")),
            robinhood_order_intents_path=os.getenv(
                "ROBINHOOD_ORDER_INTENTS_PATH",
                "logs/robinhood_order_intents.jsonl",
            ),
            robinhood_use_dollar_orders=os.getenv("ROBINHOOD_USE_DOLLAR_ORDERS", "false").lower() == "true",
            robinhood_dollar_amount_per_trade=max(0.0, float(os.getenv("ROBINHOOD_DOLLAR_AMOUNT_PER_TRADE", "0"))),
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
            stock_dry_run=os.getenv("BOT_STOCK_DRY_RUN", "false").lower() == "true",
            kill_switch=os.getenv("BOT_KILL_SWITCH", "false").lower() == "true",
            notify_trade_preview=os.getenv("BOT_NOTIFY_TRADE_PREVIEW", "true").lower() == "true",
            require_fresh_price_for_orders=os.getenv("BOT_REQUIRE_FRESH_PRICE_FOR_ORDERS", "true").lower() == "true",
            max_latest_price_age_sec=max(1, int(os.getenv("BOT_MAX_LATEST_PRICE_AGE_SEC", "300"))),
            stale_price_blocks_sell=os.getenv("BOT_STALE_PRICE_BLOCKS_SELL", "false").lower() == "true",
            block_caution_feeds_for_buys=os.getenv("BOT_BLOCK_CAUTION_FEEDS_FOR_BUYS", "false").lower() == "true",
            max_buys_per_cycle=max(0, int(os.getenv("BOT_MAX_BUYS_PER_CYCLE", "0"))),
            max_symbol_loss_pct=max(0.0, float(os.getenv("BOT_MAX_SYMBOL_LOSS_PCT", "0"))),
            block_buy_gap_up_pct=max(0.0, float(os.getenv("BOT_BLOCK_BUY_GAP_UP_PCT", "0"))),
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
            kelly_win_rate=float(os.getenv("BOT_KELLY_WIN_RATE", "0.52")),
            kelly_avg_win=float(os.getenv("BOT_KELLY_AVG_WIN", "0.015")),
            kelly_avg_loss=float(os.getenv("BOT_KELLY_AVG_LOSS", "0.01")),
            trailing_stop_pct=float(os.getenv("BOT_TRAILING_STOP_PCT", "0")),
            portfolio_drawdown_halt_pct=float(os.getenv("BOT_PORTFOLIO_DRAWDOWN_HALT_PCT", "0")),
            correlation_filter_threshold=float(os.getenv("BOT_CORRELATION_FILTER_THRESHOLD", "0")),
            ml_retrain_days=max(0, int(os.getenv("BOT_ML_RETRAIN_DAYS", "0"))),
            ml_model_path=os.getenv("BOT_ML_MODEL_PATH", "models/ensemble.joblib"),
            data_cache_ttl_sec=max(0, int(os.getenv("BOT_DATA_CACHE_TTL_SEC", "300"))),
            use_latest_price=os.getenv("BOT_USE_LATEST_PRICE", "true").lower() == "true",
            market_data_feed=os.getenv("BOT_MARKET_DATA_FEED", "auto").lower(),
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
            partial_profit_max_hold_bars=max(0, int(os.getenv("BOT_PARTIAL_PROFIT_MAX_HOLD_BARS", "0"))),
            partial_profit_trailing_stop_pct=max(0.0, float(os.getenv("BOT_PARTIAL_PROFIT_TRAILING_STOP_PCT", "0"))),
            use_atr_stops=os.getenv("BOT_USE_ATR_STOPS", "false").lower() == "true",
            atr_stop_mult=float(os.getenv("BOT_ATR_STOP_MULT", "2.0")),
            atr_period=max(2, int(os.getenv("BOT_ATR_PERIOD", "14"))),
            risk_per_trade_pct=float(os.getenv("BOT_RISK_PER_TRADE_PCT", "0.5")),
            use_adaptive_thresholds=os.getenv("BOT_USE_ADAPTIVE_THRESHOLDS", "false").lower() == "true",
            base_buy_threshold=float(os.getenv("BOT_BASE_BUY_THRESHOLD", "0.15")),
            base_sell_threshold=float(os.getenv("BOT_BASE_SELL_THRESHOLD", "-0.15")),
            use_ensemble_signal=os.getenv("BOT_USE_ENSEMBLE_SIGNAL", "false").lower() == "true",
            use_mtf_confirmation=os.getenv("BOT_USE_MTF_CONFIRMATION", "false").lower() == "true",
            mtf_timeframes=os.getenv("BOT_MTF_TIMEFRAMES", "1Day,1Hour"),
            cache_enabled=os.getenv("BOT_CACHE_ENABLED", "false").lower() == "true",
            cache_dir=os.getenv("BOT_CACHE_DIR", ".cache/bars"),
            cache_ttl_sec=max(0, int(os.getenv("BOT_CACHE_TTL_SEC", "60"))),
            use_correlation_scaling=os.getenv("BOT_USE_CORRELATION_SCALING", "false").lower() == "true",
            correlation_scale_soft=float(os.getenv("BOT_CORRELATION_SCALE_SOFT", "0.6")),
            correlation_scale_hard=float(os.getenv("BOT_CORRELATION_SCALE_HARD", "0.9")),
            use_spy_trend_filter=os.getenv("BOT_USE_SPY_TREND_FILTER", "false").lower() == "true",
            spy_trend_window=max(50, int(os.getenv("BOT_SPY_TREND_WINDOW", "200"))),
            use_vix_size_scaling=os.getenv("BOT_USE_VIX_SIZE_SCALING", "false").lower() == "true",
            vix_full_below=float(os.getenv("BOT_VIX_FULL_BELOW", "20")),
            vix_half_above=float(os.getenv("BOT_VIX_HALF_ABOVE", "25")),
            vix_zero_above=float(os.getenv("BOT_VIX_ZERO_ABOVE", "35")),
            earnings_blackout_days=max(0, int(os.getenv("BOT_EARNINGS_BLACKOUT_DAYS", "0"))),
            use_volume_confirmation=os.getenv("BOT_USE_VOLUME_CONFIRMATION", "false").lower() == "true",
            volume_min_ratio=float(os.getenv("BOT_VOLUME_MIN_RATIO", "0.8")),
            use_spread_filter=os.getenv("BOT_USE_SPREAD_FILTER", "false").lower() == "true",
            max_spread_bps=float(os.getenv("BOT_MAX_SPREAD_BPS", "10")),
            use_vol_targeting=os.getenv("BOT_USE_VOL_TARGETING", "false").lower() == "true",
            target_vol_pct=float(os.getenv("BOT_TARGET_VOL_PCT", "1.0")),
            max_position_pct=float(os.getenv("BOT_MAX_POSITION_PCT", "20")),
            max_portfolio_heat_pct=float(os.getenv("BOT_MAX_PORTFOLIO_HEAT_PCT", "6")),
            use_trailing_atr_stop=os.getenv("BOT_USE_TRAILING_ATR_STOP", "false").lower() == "true",
            trailing_atr_mult=float(os.getenv("BOT_TRAILING_ATR_MULT", "2.5")),
            time_stop_max_bars=max(0, int(os.getenv("BOT_TIME_STOP_MAX_BARS", "0"))),
            time_stop_min_r=float(os.getenv("BOT_TIME_STOP_MIN_R", "0.5")),
            partial_take_r=float(os.getenv("BOT_PARTIAL_TAKE_R", "1.0")),
            breakeven_r=float(os.getenv("BOT_BREAKEVEN_R", "1.0")),
            use_meta_label=os.getenv("BOT_USE_META_LABEL", "false").lower() == "true",
            meta_model_path=os.getenv("BOT_META_MODEL_PATH", "models/meta_label.joblib"),
            meta_min_prob=float(os.getenv("BOT_META_MIN_PROB", "0.55")),
            meta_size_scale=os.getenv("BOT_META_SIZE_SCALE", "false").lower() == "true",
            options_enabled=os.getenv("BOT_OPTIONS_ENABLED", "false").lower() == "true",
            options_strategies=os.getenv("BOT_OPTIONS_STRATEGIES", "long_call,csp,bull_call"),
            options_qty=max(1, int(os.getenv("BOT_OPTIONS_QTY", "1"))),
            options_min_pop=float(os.getenv("BOT_OPTIONS_MIN_POP", "0.60")),
            options_max_risk_pct=float(os.getenv("BOT_OPTIONS_MAX_RISK_PCT", "1.0")),
            options_min_dte=max(0, int(os.getenv("BOT_OPTIONS_MIN_DTE", "21"))),
            options_max_dte=max(1, int(os.getenv("BOT_OPTIONS_MAX_DTE", "45"))),
            options_target_delta=float(os.getenv("BOT_OPTIONS_TARGET_DELTA", "0.30")),
            options_spread_width=float(os.getenv("BOT_OPTIONS_SPREAD_WIDTH", "5.0")),
            options_data_source=os.getenv("BOT_OPTIONS_DATA_SOURCE", "auto").lower(),
            options_top_n=max(1, int(os.getenv("BOT_OPTIONS_TOP_N", "5"))),
            options_slippage_pct=float(os.getenv("BOT_OPTIONS_SLIPPAGE_PCT", "0")),
            options_allow_naked=os.getenv("BOT_OPTIONS_ALLOW_NAKED", "false").lower() == "true",
            options_dry_run=os.getenv("BOT_OPTIONS_DRY_RUN", "true").lower() == "true",
            research_auto_queue=os.getenv("BOT_RESEARCH_AUTO_QUEUE", "false").lower() == "true",
            research_queue_limit=max(1, int(os.getenv("BOT_RESEARCH_QUEUE_LIMIT", "5"))),
            research_mode=os.getenv("BOT_RESEARCH_MODE", "memo").lower(),
            research_goals=os.getenv("BOT_RESEARCH_GOALS", "long-term capital appreciation"),
            research_risk_tolerance=os.getenv("BOT_RESEARCH_RISK_TOLERANCE", "moderate"),
            research_time_horizon=os.getenv("BOT_RESEARCH_TIME_HORIZON", "5+ years"),
            robinhood_refresh_on_open=os.getenv("ROBINHOOD_REFRESH_ON_OPEN", "true").lower() == "true",
            robinhood_rotation_enabled=os.getenv("ROBINHOOD_ROTATION_ENABLED", "false").lower() == "true",
            robinhood_rotation_buy_limit=max(0, int(os.getenv("ROBINHOOD_ROTATION_BUY_LIMIT", "2"))),
            robinhood_rotation_min_buy_score=float(os.getenv("ROBINHOOD_ROTATION_MIN_BUY_SCORE", "65")),
            robinhood_rotation_trim_score_max=float(os.getenv("ROBINHOOD_ROTATION_TRIM_SCORE_MAX", "45")),
            robinhood_rotation_exit_score_max=float(os.getenv("ROBINHOOD_ROTATION_EXIT_SCORE_MAX", "35")),
            robinhood_execution_mode=os.getenv("ROBINHOOD_EXECUTION_MODE", "intent_only").lower(),
            robinhood_auto_approve=os.getenv("ROBINHOOD_AUTO_APPROVE", "false").lower() == "true",
            robinhood_auto_dispatch=os.getenv("ROBINHOOD_AUTO_DISPATCH", "false").lower() == "true",
        )

    def validate(self) -> None:
        if not self.api_key or not self.api_secret:
            raise ValueError("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY for market data.")
        if self.broker not in ("alpaca", "robinhood"):
            raise ValueError("BOT_BROKER must be 'alpaca' or 'robinhood'.")
        if self.broker == "robinhood":
            if not self.robinhood_agentic_enabled:
                raise ValueError("ROBINHOOD_AGENTIC_ENABLED must be true for BOT_BROKER=robinhood.")
            if not self.robinhood_agentic_account_number:
                raise ValueError("ROBINHOOD_AGENTIC_ACCOUNT_NUMBER is required for BOT_BROKER=robinhood.")
            if self.robinhood_agentic_buying_power <= 0 or self.robinhood_agentic_equity <= 0:
                raise ValueError(
                    "Set ROBINHOOD_AGENTIC_BUYING_POWER and ROBINHOOD_AGENTIC_EQUITY "
                    "from a fresh Robinhood portfolio check before BOT_BROKER=robinhood."
                )
            if self.robinhood_execution_mode not in {"intent_only", "approval_queue", "auto_dispatch"}:
                raise ValueError("ROBINHOOD_EXECUTION_MODE must be intent_only, approval_queue, or auto_dispatch.")
            if not self.stock_dry_run and self.robinhood_execution_mode == "intent_only":
                raise ValueError(
                    "BOT_STOCK_DRY_RUN must be true for BOT_BROKER=robinhood when "
                    "ROBINHOOD_EXECUTION_MODE=intent_only."
                )
            if self.robinhood_auto_dispatch and self.robinhood_execution_mode != "auto_dispatch":
                raise ValueError("ROBINHOOD_AUTO_DISPATCH requires ROBINHOOD_EXECUTION_MODE=auto_dispatch.")
            if self.robinhood_use_dollar_orders:
                if self.robinhood_dollar_amount_per_trade <= 0:
                    raise ValueError("ROBINHOOD_DOLLAR_AMOUNT_PER_TRADE must be > 0 when dollar orders are enabled.")
                if self.order_type != "market":
                    raise ValueError("Robinhood dollar/fractional orders require BOT_ORDER_TYPE=market.")
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
        if self.strategy_mode not in ("ma", "ensemble"):
            raise ValueError("strategy_mode must be 'ma' or 'ensemble'.")
        if self.kelly_win_rate <= 0 or self.kelly_win_rate >= 1:
            raise ValueError("kelly_win_rate must be in (0, 1).")
        if self.kelly_avg_win <= 0:
            raise ValueError("kelly_avg_win must be > 0.")
        if self.kelly_avg_loss <= 0:
            raise ValueError("kelly_avg_loss must be > 0.")

    # --- Configuration presets (#15) ---
    @classmethod
    def conservative(cls, api_key: str = "", api_secret: str = "") -> "Settings":
        """Preset for conservative, low-risk trading."""
        return cls(
            api_key=api_key or os.getenv("APCA_API_KEY_ID", ""),
            api_secret=api_secret or os.getenv("APCA_API_SECRET_KEY", ""),
            max_shares=1,
            max_daily_trades=1,
            daily_loss_limit_pct=2.0,
            max_portfolio_exposure_pct=50.0,
            stop_loss_pct=3.0,
            trailing_stop_pct=5.0,
            trade_cooldown_sec=600,
            portfolio_drawdown_halt_pct=5.0,
            use_kelly_sizing=True,
            kelly_fraction=0.25,
            kelly_max_shares=3,
            strategy_mode="ensemble",
            use_sentiment_filter=True,
            correlation_filter_threshold=0.70,
            error_streak_decay_hours=2.0,
        )

    @classmethod
    def aggressive(cls, api_key: str = "", api_secret: str = "") -> "Settings":
        """Preset for aggressive, higher-risk trading."""
        return cls(
            api_key=api_key or os.getenv("APCA_API_KEY_ID", ""),
            api_secret=api_secret or os.getenv("APCA_API_SECRET_KEY", ""),
            max_shares=10,
            max_daily_trades=5,
            daily_loss_limit_pct=5.0,
            max_portfolio_exposure_pct=90.0,
            stop_loss_pct=5.0,
            trailing_stop_pct=8.0,
            trade_cooldown_sec=60,
            use_kelly_sizing=True,
            kelly_fraction=0.75,
            kelly_max_shares=20,
            strategy_mode="ensemble",
            dip_buy_enabled=True,
            partial_profit_trigger_pct=10.0,
            partial_profit_sell_pct=50.0,
            partial_profit_max_hold_bars=20,
            partial_profit_trailing_stop_pct=5.0,
            correlation_filter_threshold=0.85,
            error_streak_decay_hours=1.0,
        )

    @classmethod
    def mean_reversion(cls, api_key: str = "", api_secret: str = "") -> "Settings":
        """Preset optimized for mean-reversion / range-bound markets."""
        return cls(
            api_key=api_key or os.getenv("APCA_API_KEY_ID", ""),
            api_secret=api_secret or os.getenv("APCA_API_SECRET_KEY", ""),
            fast_ma=3,
            slow_ma=10,
            max_shares=5,
            max_daily_trades=3,
            daily_loss_limit_pct=3.0,
            stop_loss_pct=2.0,
            trailing_stop_pct=3.0,
            trade_cooldown_sec=300,
            use_kelly_sizing=True,
            kelly_fraction=0.5,
            strategy_mode="ensemble",
            dip_buy_enabled=True,
            dip_rsi_threshold=30.0,
            dip_drop_pct=3.0,
            use_sentiment_filter=True,
            error_streak_decay_hours=2.0,
        )

    def get_symbols(self) -> list[str]:
        """Return list of symbols to trade (multi-symbol or single)."""
        if self.symbols:
            return [s.strip().upper() for s in self.symbols.split(",") if s.strip()]
        return [self.symbol.upper()]

    @property
    def is_live(self) -> bool:
        return not self.paper_only
