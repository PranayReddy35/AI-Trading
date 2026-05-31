# AI-Trading

> ⚠️ **This software can trade with real money. Use at your own risk.**
>
> Default mode is **paper trading** (no real money). To enable live trading, you must explicitly set `BOT_PAPER_ONLY=false`. Always start with paper trading and validate your strategy thoroughly before going live.

## What is included

Production-ready module structure:

- `ai_trading/config.py` – runtime settings from environment variables (paper + live)
- `ai_trading/data/` – market data access (Alpaca)
- `ai_trading/strategy/` – signal generation (MA, ensemble, regime-adaptive)
- `ai_trading/strategy/ensemble.py` – **multi-strategy ensemble** with regime detection
- `ai_trading/risk/` – comprehensive risk checks and safety guards
- `ai_trading/broker/` – Alpaca broker wrapper (market + limit orders, stop-loss, retry logic, order tracking)
- `ai_trading/storage/` – logging + JSONL journaling
- `ai_trading/notifications/` – webhook alerts (Slack, Discord, or generic)
- `ai_trading/bot.py` – trading bot (single run, paper or live)
- `ai_trading/runner.py` – daily scheduling runner with graceful shutdown
- `ai_trading/backtest.py` – basic backtest (rule-based MA logic)
- `ai_trading/backtest_realistic.py` – **realistic backtest** with slippage, commissions, Kelly sizing, Monte Carlo risk analysis
- `ai_trading/ml/predict_direction.py` – basic ML (logistic regression) next-day direction script
- `ai_trading/ml/ensemble_model.py` – **advanced ensemble ML** (GradientBoosting + RandomForest) with 35+ features and walk-forward validation
- `ai_trading/data/news_sentiment.py` – news fetcher (Google News RSS free, Alpha Vantage free tier)
- `ai_trading/data/social_sentiment.py` – social media fetcher (Reddit public API, free, no key)
- `ai_trading/ml/sentiment.py` – VADER-based sentiment scoring and feature engineering
- `ai_trading/strategy/sentiment_filter.py` – sentiment overlay that can block trades on extreme news

## Safety and risk controls

`RiskManager` includes:

- **Paper-trading-only safety guard** (blocks live orders unless explicitly enabled)
- **Market-closed guard**
- **Cash threshold guard**
- **Duplicate/open order prevention**
- **Max shares / position sizing cap**
- **Max daily trades**
- **Consecutive error stop** (halts after N consecutive failures)
- **No-shorting behavior** (sell only existing long positions)
- **Daily loss limit** (% of equity, halts trading if breached)
- **Max portfolio exposure** (% of equity, prevents over-allocation)
- **Minimum equity guard** (won't trade if equity drops too low)
- **Trade cooldown** (prevents rapid re-entry after a trade)
- **Order confirmation** (interactive prompt for live orders)
- **Pre-flight checks** (account status validation before every run)

## Setup

```bash
cd AI-Trading
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Environment variables

### Required

```bash
export APCA_API_KEY_ID="YOUR_API_KEY"
export APCA_API_SECRET_KEY="YOUR_API_SECRET"
```

### Mode selection

```bash
# Paper trading (default, safe)
export BOT_PAPER_ONLY="true"

# Live trading (REAL MONEY - use with caution)
export BOT_PAPER_ONLY="false"
```

### Trading parameters

```bash
export BOT_SYMBOL="SPY"
export BOT_FAST_MA="5"
export BOT_SLOW_MA="20"
export BOT_LOOKBACK_DAYS="90"
export BOT_MAX_SHARES="1"
export BOT_MIN_CASH_THRESHOLD="100"
export BOT_MAX_DAILY_TRADES="1"
export BOT_MAX_CONSECUTIVE_ERRORS="3"
```

### Order execution

```bash
# Order type: "market" or "limit"
export BOT_ORDER_TYPE="market"

# For limit orders: price offset from current price in %
export BOT_LIMIT_PRICE_OFFSET_PCT="0.1"

# Stop-loss: % below entry price (0 = disabled)
export BOT_STOP_LOSS_PCT="2.0"

# Seconds to wait for order fill (0 = don't wait)
export BOT_ORDER_FILL_TIMEOUT_SEC="60"

# Max API retries for transient failures
export BOT_MAX_API_RETRIES="3"
```

### Risk management

```bash
# Daily loss limit as % of equity (0 = disabled)
export BOT_DAILY_LOSS_LIMIT_PCT="3.0"

# Max portfolio exposure as % of equity
export BOT_MAX_PORTFOLIO_EXPOSURE_PCT="95"

# Minimum equity to allow trading (0 = disabled)
export BOT_MIN_EQUITY="1000"

# Cooldown between trades in seconds (0 = disabled)
export BOT_TRADE_COOLDOWN_SEC="300"

# Require interactive confirmation for live orders
export BOT_REQUIRE_CONFIRMATION="true"
```

### Notifications

```bash
# Webhook URL for Slack/Discord/generic notifications (empty = disabled)
export BOT_WEBHOOK_URL="https://hooks.slack.com/services/..."

# Comma-separated event types to notify on
export BOT_NOTIFY_EVENTS="trade,error,risk_reject"
```

### News sentiment

```bash
# News provider: "rss" (Google News, free, no key) or "alphavantage" (free, 25 req/day)
export BOT_NEWS_PROVIDER="rss"

# Alpha Vantage API key (free at alphavantage.co — only needed if using alphavantage provider)
export BOT_NEWS_API_KEY=""

# Enable sentiment filter on trading signals (blocks trades on extreme sentiment)
export BOT_USE_SENTIMENT_FILTER="true"

# Sentiment thresholds (-1.0 to 1.0): block BUY if below, block SELL if above
export BOT_SENTIMENT_BUY_THRESHOLD="-0.3"
export BOT_SENTIMENT_SELL_THRESHOLD="0.3"

# Include sentiment features in ML model predictions
export BOT_USE_SENTIMENT_IN_ML="true"

# Extra news search keywords (comma-separated, e.g., "CEO,earnings,Fed")
export BOT_NEWS_KEYWORDS=""
```

### Social media sentiment (Reddit)

```bash
# Enable Reddit social media sentiment (free, no API key needed)
export BOT_USE_SOCIAL_SENTIMENT="true"

# Custom subreddits to search (comma-separated, default: wallstreetbets,stocks,investing,options,stockmarket)
export BOT_SOCIAL_SUBREDDITS=""

# Time filter for Reddit search: "hour", "day", "week", "month", "year", "all"
export BOT_SOCIAL_TIME_FILTER="week"
```

> **Note on other social platforms:**
> - **Twitter/X**: The free API tier does NOT support reading tweets (only posting). Not viable at zero cost.
> - **Discord**: Requires bot membership in specific servers. No public API for sentiment data.

## Run the bot

### Paper trading (safe, default)

```bash
python -m ai_trading.bot
```

### Live trading

```bash
# Interactive (will prompt for confirmation)
BOT_PAPER_ONLY=false python -m ai_trading.bot

# Automated (no confirmation prompt - for scheduled runs)
BOT_PAPER_ONLY=false python -m ai_trading.bot --no-confirm
```

### With symbol override

```bash
python -m ai_trading.bot --symbol QQQ
```

## Scheduling (daily execution)

Use OS scheduler (cron/systemd/task scheduler) in production. The built-in runner provides:
- Graceful shutdown (SIGINT/SIGTERM)
- Health check mode
- Configurable run time

```bash
# Run once immediately
python -m ai_trading.runner

# Run daily at 20:10 UTC with graceful shutdown support
python -m ai_trading.runner --loop --run-time 20:10 --no-confirm

# Health check (validate config without trading)
python -m ai_trading.runner --health-check
```

### Example cron entry (recommended for production)

```cron
# Run at 3:55 PM ET (19:55 UTC) every weekday
55 19 * * 1-5 cd /path/to/AI-Trading && .venv/bin/python -m ai_trading.bot --no-confirm >> logs/cron.log 2>&1
```

## Backtest script (rule-based MA)

```bash
python -m ai_trading.backtest --symbol SPY --start 2018-01-01 --end 2025-01-01 --fast-ma 5 --slow-ma 20 --max-shares 1
```

You can also use local CSV:

```bash
python -m ai_trading.backtest --csv /path/to/bars.csv
```

Expected CSV columns: `date,open,high,low,close,volume`.

## Realistic backtest (advanced)

The realistic backtest engine addresses all shortcomings of basic backtesting:

- **Transaction costs**: Commissions, slippage, spread, and market impact modeling
- **Kelly criterion position sizing**: Optimal bet sizing based on historical win rate
- **Monte Carlo risk-of-ruin analysis**: 10,000 simulation paths to estimate ruin probability
- **Multi-strategy support**: Test MA strategy or full ensemble strategy
- **Advanced metrics**: Sharpe ratio, Sortino, max drawdown, Calmar, profit factor, expectancy
- **Benchmark comparison**: Always compared against buy-and-hold

```bash
# Realistic backtest with ensemble strategy (recommended)
python -m ai_trading.backtest_realistic --symbol SPY --start 2015-01-01 --end 2025-01-01 --strategy ensemble --initial-cash 100000

# With custom transaction costs
python -m ai_trading.backtest_realistic --symbol SPY --strategy ensemble --commission-per-share 0.005 --slippage-bps 5

# Basic MA strategy for comparison
python -m ai_trading.backtest_realistic --symbol SPY --strategy ma --fast-ma 10 --slow-ma 30

# Disable Kelly sizing (use fixed position size)
python -m ai_trading.backtest_realistic --symbol SPY --strategy ensemble --no-kelly
```

### Live trading readiness checklist (automated)

The realistic backtest automatically evaluates 8 criteria:
- ✓ Sharpe ratio > 1.5
- ✓ Profit factor > 1.5
- ✓ Beats buy-and-hold benchmark
- ✓ Max drawdown < 20%
- ✓ Win rate > 50%
- ✓ Ruin probability < 5%
- ✓ Kelly fraction > 0
- ✓ Positive expectancy per trade

## Advanced ML model (ensemble)

The advanced ML path replaces logistic regression with a production-grade ensemble:

- **Model**: VotingClassifier (GradientBoosting 60% + RandomForest 40%)
- **Features**: 35+ technical indicators (RSI, MACD, Bollinger Bands, ATR, OBV, stochastic, momentum, patterns)
- **Validation**: Walk-forward (rolling window) instead of single train/test split
- **Calibration**: Isotonic probability calibration for reliable confidence estimates

```bash
# Walk-forward validation with 5 folds
python -m ai_trading.ml.ensemble_model --symbol SPY --start 2015-01-01 --end 2025-01-01 --folds 5

# Save trained model
python -m ai_trading.ml.ensemble_model --symbol SPY --save-model ensemble_model.joblib
```

## Multi-strategy ensemble

The ensemble strategy combines 4 uncorrelated signal generators:

| Strategy | Best in regime | Description |
|----------|---------------|-------------|
| Trend following | Bull/Bear trends | Adaptive EMA crossover with slope confirmation |
| Mean reversion | Sideways markets | Bollinger Bands + RSI oversold/overbought |
| Momentum | Breakouts | Rate of change + volume surge detection |
| ML model | All regimes | Ensemble probability as directional signal |

Strategies are weighted by detected **market regime**:
- **Bull trend**: Trend (45%) + Momentum (35%) + Mean reversion (10%) + ML (10%)
- **Bear trend**: Trend (40%) + Momentum (20%) + Mean reversion (20%) + ML (20%)
- **Sideways**: Mean reversion (45%) + ML (30%) + Momentum (15%) + Trend (10%)
- **High volatility**: Mean reversion (30%) + ML (30%) + Trend (20%) + Momentum (20%)

A trade is only executed when ≥2 strategies agree on direction (consensus filter).

## ML script (basic — educational)

The basic ML path is intentionally separate and beginner-friendly.

- Model: logistic regression
- Features: 1d return, 5d return, 10d volatility, distance from 20d MA, volume ratio vs 20d average
- Target: next-day direction (`close[t+1] > close[t]`)

```bash
python -m ai_trading.ml.predict_direction --symbol SPY --start 2018-01-01 --end 2025-01-01
```

### With sentiment analysis (live news)

```bash
python -m ai_trading.ml.predict_direction --symbol AAPL --with-sentiment
```

This fetches live news headlines, scores them with VADER sentiment analysis, and combines the result with the technical prediction. Covers: earnings calls, CEO statements, political news, analyst upgrades/downgrades, macro events.

### With social media sentiment (Reddit)

```bash
python -m ai_trading.ml.predict_direction --symbol AAPL --with-sentiment --news-provider reddit
```

This fetches posts from Reddit finance subreddits (r/wallstreetbets, r/stocks, r/investing, etc.), analyzes social sentiment using VADER, and incorporates it into the ML prediction. Completely free, no API key needed.

Use Alpha Vantage (free, 25 req/day) for better relevance scoring:

```bash
python -m ai_trading.ml.predict_direction --symbol AAPL --with-sentiment --news-provider alphavantage --news-api-key YOUR_KEY
```

Optional model export:

```bash
python -m ai_trading.ml.predict_direction --symbol SPY --save-model model.joblib
```

## Logs and artifacts

- Bot log: `logs/bot.log`
- Trade journal (JSONL): `logs/journal.jsonl`

Journal events include: `account_state`, `signal`, `decision`, `order`, `fill_status`, `stop_loss`, `risk_reject`, `error`, `preflight_failed`, `user_cancel`

## Production checklist

Before going live with real money:

1. ✅ Paper trade for at least 2-4 weeks
2. ✅ Verify backtest results match paper trading behavior
3. ✅ Set conservative `BOT_MAX_SHARES` (start with 1)
4. ✅ Set `BOT_DAILY_LOSS_LIMIT_PCT` (e.g., 2-3%)
5. ✅ Set `BOT_STOP_LOSS_PCT` (e.g., 2-5%)
6. ✅ Configure webhook notifications
7. ✅ Set `BOT_MIN_EQUITY` as a circuit breaker
8. ✅ Review journal logs daily
9. ✅ Test with the smallest possible position size first
10. ✅ Have a plan for what to do if the bot malfunctions

## Important warnings

- **This is not financial advice.** Use at your own risk.
- Automated trading systems can produce duplicate or erroneous orders.
- Paper trading fills do not perfectly match live market conditions.
- Past performance (backtest) does not guarantee future results.
- Always monitor your bot and have a kill switch ready.
- Start with tiny positions and scale up gradually only after consistent results.
