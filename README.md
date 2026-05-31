# AI-Trading

> ⚠️ **This software can trade with real money. Use at your own risk.**
>
> Default mode is **paper trading** (no real money). To enable live trading, you must explicitly set `BOT_PAPER_ONLY=false`. Always start with paper trading and validate your strategy thoroughly before going live.

## What is included

Production-ready module structure:

- `ai_trading/config.py` – runtime settings from environment variables (paper + live)
- `ai_trading/data/` – market data access (Alpaca)
- `ai_trading/strategy/` – moving-average signal generation
- `ai_trading/risk/` – comprehensive risk checks and safety guards
- `ai_trading/broker/` – Alpaca broker wrapper (market + limit orders, stop-loss, retry logic, order tracking)
- `ai_trading/storage/` – logging + JSONL journaling
- `ai_trading/notifications/` – webhook alerts (Slack, Discord, or generic)
- `ai_trading/bot.py` – trading bot (single run, paper or live)
- `ai_trading/runner.py` – daily scheduling runner with graceful shutdown
- `ai_trading/backtest.py` – backtest matching rule-based MA logic with daily bars
- `ai_trading/ml/predict_direction.py` – separate ML (logistic regression) next-day direction script

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

## ML script (separate from rule-based bot)

The ML path is intentionally separate and beginner-friendly.

- Model: logistic regression
- Features: 1d return, 5d return, 10d volatility, distance from 20d MA, volume ratio vs 20d average
- Target: next-day direction (`close[t+1] > close[t]`)

```bash
python -m ai_trading.ml.predict_direction --symbol SPY --start 2018-01-01 --end 2025-01-01
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
