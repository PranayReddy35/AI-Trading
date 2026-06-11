# AI-Trading

> ⚠️ **This software can trade with real money. Use at your own risk.**
>
> Default mode is **paper trading** (no real money). To enable live trading, you must explicitly set `BOT_PAPER_ONLY=false`. Always start with paper trading and validate your strategy thoroughly before going live.

## What is included

Production-ready module structure:

- `ai_trading/config.py` – runtime settings from environment variables (paper + live)
- `ai_trading/data/` – market data access (Alpaca, yfinance) + symbol universe loader
- `ai_trading/data/universe.py` – **S&P 500 / Nasdaq 100 / Dow 30** ticker loader (Wikipedia, cached weekly)
- `ai_trading/strategy/` – signal generation (MA, ensemble, regime-adaptive, patterns)
- `ai_trading/strategy/ensemble.py` – **multi-strategy ensemble** with regime detection
- `ai_trading/strategy/market_filters.py` – **macro gates**: SPY 200DMA, VIX scaling, earnings blackout, volume confirm, spread filter
- `ai_trading/risk/` – comprehensive risk checks and safety guards
- `ai_trading/risk/portfolio_sizing.py` – **vol-targeted sizing**, portfolio heat cap, fractional Kelly
- `ai_trading/risk/exits.py` – **trailing ATR stop**, partial take-profit, time stop, break-even stop
- `ai_trading/broker/` – Alpaca broker wrapper (market + limit orders, stop-loss, retry logic, order tracking)
- `ai_trading/storage/` – logging + JSONL journaling
- `ai_trading/notifications/` – webhook alerts (Slack, Discord, or generic)
- `ai_trading/bot.py` – trading bot (single run, paper or live)
- `ai_trading/runner.py` – daily scheduling runner with graceful shutdown
- `ai_trading/scanner.py` – **live market scanner** (EOD + intraday) with RS, BB squeeze, meta-label, ATR levels, correlation dedup
- `ai_trading/dashboard.py` – Streamlit dashboard (positions, P&L, live scanner, sell scanner, **Position Advisor with sizing**, regime heatmap, Options Lab)
- `ai_trading/backtest.py` – basic backtest (rule-based MA logic)
- `ai_trading/backtest_realistic.py` – realistic backtest with slippage, commissions, Kelly sizing, Monte Carlo
- `ai_trading/backtest/ensemble_cost_aware.py` – **cost-aware ensemble backtest** (commission + slippage + spread)
- `ai_trading/backtest/regime_optimizer.py` – **walk-forward optimizer** for per-symbol regime weights
- `ai_trading/ml/predict_direction.py` – basic ML (logistic regression) next-day direction script
- `ai_trading/ml/ensemble_model.py` – advanced ensemble ML (GradientBoosting + RandomForest) with 35+ features
- `ai_trading/ml/meta_label.py` – **meta-labeling**: predicts P(hit +1R before −1R) via triple-barrier labels
- `ai_trading/data/news_sentiment.py` – news fetcher (Google News RSS free, Alpha Vantage free tier)
- `ai_trading/data/social_sentiment.py` – social media fetcher (Reddit public API, free)
- `ai_trading/ml/sentiment.py` – VADER-based sentiment scoring and feature engineering
- `ai_trading/strategy/sentiment_filter.py` – sentiment overlay that can block trades on extreme news

## Safety and risk controls

`RiskManager` and the trade pipeline include:

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
- **Macro filters** — SPY 200DMA gate, VIX size scaling, earnings blackout, volume confirmation, spread filter
- **Vol-targeted sizing** — caps each position to a fixed % of equity in daily volatility
- **Portfolio heat cap** — total open dollar-risk across all positions stays under % of equity
- **Trailing ATR stop / partial TP / time stop / break-even shift**
- **Meta-label gate** — skips BUYs the ML model thinks won't reach +1R before −1R
- **Correlation dedup** (scanner) — surfaces only diversified picks

## Setup

For the project architecture and subsystem map, see [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md).

For a step-by-step local operating guide, including what to do when Codex/AI credits are exhausted, see [`LOCAL_RUNBOOK.md`](LOCAL_RUNBOOK.md).

For free Streamlit hosting guidance and limitations, see [`FREE_DEPLOYMENT.md`](FREE_DEPLOYMENT.md).

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
>
> - **Twitter/X**: The free API tier does NOT support reading tweets (only posting). Not viable at zero cost.
> - **Discord**: Requires bot membership in specific servers. No public API for sentiment data.

### Macro / quality filters

```bash
# SPY 200-day moving average gate — block new longs in risk-off regimes
export BOT_USE_SPY_TREND_FILTER="true"
export BOT_SPY_TREND_WINDOW="200"

# VIX-based size scaling — full size below 20, half above 25, zero above 35
export BOT_USE_VIX_SIZE_SCALING="true"
export BOT_VIX_FULL_BELOW="20"
export BOT_VIX_HALF_ABOVE="25"
export BOT_VIX_ZERO_ABOVE="35"

# Earnings blackout — skip new positions N days before earnings (0 = disabled)
export BOT_EARNINGS_BLACKOUT_DAYS="2"

# Volume confirmation — reject signals when bar volume < ratio × 20d avg
export BOT_USE_VOLUME_CONFIRMATION="true"
export BOT_VOLUME_MIN_RATIO="0.8"

# Spread filter — reject quotes wider than N basis points
export BOT_USE_SPREAD_FILTER="true"
export BOT_MAX_SPREAD_BPS="10"
```

### Position sizing (vol targeting + portfolio heat)

```bash
# Vol-targeted sizing — target % of equity per trade in daily volatility
export BOT_USE_VOL_TARGETING="true"
export BOT_TARGET_VOL_PCT="1.0"
export BOT_MAX_POSITION_PCT="20.0"

# Portfolio heat cap — total open dollar-risk as % of equity
export BOT_MAX_PORTFOLIO_HEAT_PCT="6.0"
```

### Trailing exits

```bash
# Trailing ATR stop — ratchets stop up to (peak − k × ATR) for longs
export BOT_USE_TRAILING_ATR_STOP="true"
export BOT_TRAILING_ATR_MULT="2.5"

# Time stop — exit if trade hasn't made min_progress R within max_bars (0 = disabled)
export BOT_TIME_STOP_MAX_BARS="0"
export BOT_TIME_STOP_MIN_R="0.5"

# Partial take-profit and break-even shift triggers (in R multiples)
export BOT_PARTIAL_TAKE_R="1.0"
export BOT_BREAKEVEN_R="1.0"
```

### Meta-labeling ML

```bash
# Enable meta-label: reject BUYs with P(hit +1R before -1R) below threshold
export BOT_USE_META_LABEL="true"
export BOT_META_MODEL_PATH="models/meta_label.joblib"
export BOT_META_MIN_PROB="0.55"
# If true, scale position size by meta probability instead of just gating
export BOT_META_SIZE_SCALE="false"
```

Train the meta-label model before enabling it:

```bash
python -c "from ai_trading.ml.meta_label import train_and_save; \
  train_and_save('SPY', out_path='models/meta_label.joblib', period='3y')"
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

## Cost-aware ensemble backtest

Models real transaction costs (per-share commission, slippage in bps, half-spread) and reports
trades, win-rate, profit factor, expectancy, Sharpe, max-drawdown, and total costs.

```bash
python -m ai_trading.backtest.ensemble_cost_aware SPY,QQQ,AAPL,MSFT,NVDA \
    --period 2y --commission 0.005 --slip-bps 5 --spread-bps 2 \
    --out logs/cost_aware_2y.json
```

## Walk-forward regime-weight optimizer

Rolling train/test windows pick the best per-symbol weights for each strategy component
(trend, momentum, mean-reversion, patterns, ML) within each detected regime. Output is
consumed by `ai_trading/strategy/ensemble.py` at signal time.

```bash
python -m ai_trading.backtest.regime_optimizer SPY,QQQ,AAPL,MSFT,NVDA \
    --period 2y --train 180 --test 60 --out logs/regime_weights.json
```

## Live market scanner

Ranks symbols by composite buy-opportunity score using free data (yfinance EOD,
Alpaca IEX 5-min intraday). Includes liquidity gate, relative strength vs SPY,
Bollinger-band squeeze, meta-label P(win), ATR-based entry/stop/target levels,
macro quality gates, and correlation-based dedup.

```bash
# Scan all major indexes (S&P 500 + Nasdaq 100 + Dow 30 = ~516 unique tickers)
python -m ai_trading.scanner --universe all --top 10 --only-pass

# Specific indexes
python -m ai_trading.scanner --universe sp500,dow30 --top 20

# Mix index universe with custom tickers
python -m ai_trading.scanner --universe nasdaq100 --symbols COIN,RIVN --top 15

# Force live (Alpaca intraday) mode during market hours
python -m ai_trading.scanner --mode live --universe all --top 10

# Refresh the cached index lists from Wikipedia (cache TTL: 7 days)
python -m ai_trading.scanner --universe all --refresh-universe --top 10
```

Available flags:

| Flag                  | Default           | Purpose                              |
| --------------------- | ----------------- | ------------------------------------ |
| `--symbols`           | env `BOT_SYMBOLS` | Comma-separated tickers              |
| `--universe`          | (none)            | `sp500`, `nasdaq100`, `dow30`, `all` |
| `--refresh-universe`  | false             | Force re-fetch index lists           |
| `--mode`              | `auto`            | `auto` / `live` / `eod`              |
| `--top`               | 10                | Top N results                        |
| `--min-price`         | 5.0               | Liquidity gate: min last close       |
| `--min-dollar-vol`    | 5,000,000         | Liquidity gate: 20d avg $-volume     |
| `--no-filters`        | false             | Skip macro quality gates             |
| `--earnings-blackout` | 2                 | Days before earnings to flag (0=off) |
| `--no-meta`           | false             | Skip meta-label inference            |
| `--no-dedup`          | false             | Skip correlation dedup               |
| `--max-corr`          | 0.85              | Dedup threshold                      |
| `--only-pass`         | false             | Hide picks that fail quality gates   |

Scanner output columns: `Score` · `Sig` · `Price` · `1D%` · `5D%` (or `Intra%`) · `RSI` ·
`VolX` · `MA%` (or `VWAP%`) · `RS%` (vs SPY) · `Sqz` (BB squeeze 0–1) · `Meta` (P(win) 0–1) ·
`Entry`/`Stop`/`Tgt` (ATR-based, 2:1 R:R) · `R%` (risk per trade) · `Gate` (`PASS` or rejection
reasons) · `Reason` (ranked drivers).

## Symbol universe loader

Standalone CLI for fetching / refreshing the cached index ticker lists.

```bash
# Print every ticker in all major indexes (de-duplicated)
python -m ai_trading.data.universe all

# Count members of a single index
python -m ai_trading.data.universe --count-only sp500

# Force refresh from Wikipedia
python -m ai_trading.data.universe --refresh all
```

Cached under `logs/universe/{sp500,nasdaq100,dow30}.json` (TTL 7 days). Falls back
to a built-in static list if Wikipedia is unreachable. Tickers are normalized for
yfinance (`BRK.B` → `BRK-B`).

## Options trading

The bot includes a full options stack under `ai_trading/options/`:

| File             | Purpose                                                                   |
| ---------------- | ------------------------------------------------------------------------- |
| `chains.py`      | Fetch + normalize option chains (Alpaca primary, yfinance fallback)       |
| `greeks.py`      | Black-Scholes pricing, greeks (Δ/Γ/Θ/V/ρ), IV solver, POP helper          |
| `strategies.py`  | Build candidates for 10 strategies with max profit / max loss / POP / R:R |
| `scanner.py`     | Top-level scanner; can pre-filter underlyings via the equity scanner      |
| `broker.py`      | Single-leg + multi-leg (MLEG) option orders via alpaca-py                 |
| `integration.py` | Hook called from `bot.run_once` when `BOT_OPTIONS_ENABLED=true`           |
| `runner.py`      | CLI: `scan`, `trade`, `positions`, `close`                                |

Refresh local Robinhood dashboard snapshots:

```bash
python -m ai_trading.broker.robinhood_snapshot
```

This updates `logs/robinhood_portfolios.json` and `logs/robinhood_quotes.json` for the dashboard using local Robinhood credentials when available.

Generate structured equity research prompts locally:

```bash
make research-prompt TICKER=NVDA MODE=memo
```

Supported `MODE` values:
- `memo`
- `earnings`
- `valuation`
- `debate`

The Streamlit dashboard also includes a **Research** page that can build these prompts from Robinhood-held symbols, manual tickers, and your own goals/risk/time-horizon inputs.

### Robinhood market-open automation

You can now enable a guarded Robinhood market-open workflow:

```bash
export BOT_BROKER="robinhood"
export ROBINHOOD_REFRESH_ON_OPEN="true"
export ROBINHOOD_ROTATION_ENABLED="true"
export ROBINHOOD_ROTATION_BUY_LIMIT="2"
export ROBINHOOD_ROTATION_MIN_BUY_SCORE="65"
export ROBINHOOD_ROTATION_TRIM_SCORE_MAX="45"
export ROBINHOOD_ROTATION_EXIT_SCORE_MAX="35"
```

Execution modes:

```bash
# safest: generate intents only
export ROBINHOOD_EXECUTION_MODE="intent_only"
export BOT_STOCK_DRY_RUN="true"

# queue executable approvals automatically, but do not dispatch
export ROBINHOOD_EXECUTION_MODE="approval_queue"
export BOT_STOCK_DRY_RUN="false"
export ROBINHOOD_AUTO_APPROVE="true"

# queue and dispatch approvals to an external Robinhood executor webhook
export ROBINHOOD_EXECUTION_MODE="auto_dispatch"
export BOT_STOCK_DRY_RUN="false"
export ROBINHOOD_AUTO_APPROVE="true"
export ROBINHOOD_AUTO_DISPATCH="true"
export ROBINHOOD_EXECUTOR_WEBHOOK_URL="https://your-executor-endpoint"
```

What this does:
- refreshes Robinhood snapshots at the start of a market-open bot cycle
- scores current holdings for hold/trim/sell rotation decisions
- scores candidate buys from your configured symbol universe
- journals a `robinhood_rotation_plan`
- optionally creates and dispatches Robinhood approval records for tightly rule-locked orders

Keep snapshots refreshing automatically:

```bash
python -m ai_trading.broker.robinhood_snapshot --loop --interval-sec 60
```

### Supported strategies

| Strategy                 | Bias                       | Legs       | Risk                            |
| ------------------------ | -------------------------- | ---------- | ------------------------------- |
| `long_call`              | bullish                    | 1          | defined (premium)               |
| `long_put`               | bearish                    | 1          | defined (premium)               |
| `csp` (cash-secured put) | neutral / bullish          | 1          | defined (strike × 100 − credit) |
| `covered_call`           | neutral / slightly bullish | 1 (+stock) | defined (cost basis)            |
| `bull_call` / `bear_put` | directional                | 2          | defined (debit)                 |
| `bull_put` / `bear_call` | directional credit         | 2          | defined (width − credit)        |
| `iron_condor`            | neutral / range-bound      | 4          | defined (wing width − credit)   |
| `short_strangle`         | neutral / high IV          | 2          | **undefined** (naked)           |

### CLI

```bash
# Scan only — top candidates for a few underlyings, all strategies:
python -m ai_trading.options.scanner \
    --underlying SPY --underlying AAPL --underlying NVDA \
    --strategies long_call,csp,bull_call,iron_condor \
    --min-dte 21 --max-dte 45 --delta 0.30 --top 15

# Scan with equity-scanner pre-filter (picks bullish names from a universe):
python -m ai_trading.options.scanner --universe sp500 \
    --strategies bull_call,csp --top 20

# Trade: scan + place top N paper orders with gates (POP, max risk):
python -m ai_trading.options.runner trade \
    --universe dow30 --strategies bull_call,csp \
    --top 3 --qty 1 --min-pop 0.65 --max-risk-pct 1.0 \
    --confirm --paper

# List + close:
python -m ai_trading.options.runner positions --paper
python -m ai_trading.options.runner close SPY250620C00500000 --paper
```

### Bot integration (env vars)

Setting `BOT_OPTIONS_ENABLED=true` makes the regular bot loop also run an
options cycle each invocation against the configured `BOT_SYMBOLS`:

```ini
BOT_OPTIONS_ENABLED=true
BOT_OPTIONS_STRATEGIES=long_call,csp,bull_call
BOT_OPTIONS_QTY=1
BOT_OPTIONS_MIN_POP=0.60             # only place trades with ≥60% POP
BOT_OPTIONS_MAX_RISK_PCT=1.0         # max risk per trade as % of options BP
BOT_OPTIONS_MIN_DTE=21
BOT_OPTIONS_MAX_DTE=45
BOT_OPTIONS_TARGET_DELTA=0.30        # |Δ| target for long/short legs
BOT_OPTIONS_SPREAD_WIDTH=5.0
BOT_OPTIONS_DATA_SOURCE=auto         # auto | alpaca | yfinance
BOT_OPTIONS_TOP_N=5
BOT_OPTIONS_SLIPPAGE_PCT=0
BOT_OPTIONS_ALLOW_NAKED=false        # block undefined-risk strategies
BOT_OPTIONS_DRY_RUN=true             # SAFE DEFAULT: log candidates, don't submit
```

To start placing real paper orders, flip `BOT_OPTIONS_DRY_RUN=false`. Live
orders also require `BOT_PAPER_ONLY=false` (the broker handles paper/live mode).

### Dashboard

The Streamlit dashboard has a new **🎯 Options Lab** page that lets you
configure underlyings + strategies, run a scan interactively, inspect any
candidate's full leg breakdown, and submit it as a paper order with one click.
It also shows open option positions.

### Position Advisor (Buy / Hold / Trim / Sell, with sizing)

The dashboard's **🧮 Position Advisor** page tells you what to do with a
position you already own. It combines your **P&L** with **live market signals**
(RSI, trend, momentum, volume, MA/VWAP gap) and returns a recommended action
plus a **concrete share count and dollar amount** to transact.

Two input modes:

- **✍️ Single Position** — type symbol, shares, average cost, optional market value.
- **📥 Upload Portfolio CSV (Robinhood / etc.)** — upload a positions export or
  **paste CSV / tab-separated text** straight from your broker, Google Sheets, or
  Excel. The parser handles ragged rows, currency values with embedded commas
  (`$1,067.00`, `$1,234,567.00`), parenthesised negatives, and column-name
  variants (`Symbol`/`Ticker`, `Shares`/`Quantity`, `Average cost`/`Cost basis`,
  `Market value`/`Equity`/`Current value`). Skipped rows (cash, pending activity,
  totals) are reported with the reason so nothing silently disappears.

Two **holding horizons**:

- **⚡ Short-term (swing/day-trade)** — aggressive trimming on RSI≥70 / gap≥6% /
  momentum spikes. TAKE PROFIT closes the full position.
- **📈 Long-term investor (let winners run)** — ignores minor overbought signals
  when the multi-week trend is strong (≥55% consistency). Only true blowoffs
  (RSI≥80, gap≥12%, or 5-day momentum≥20%) trigger TAKE PROFIT, and even then it
  leaves a 25% runner. A single weak short-term signal won't shake you out of a
  multi-month winner.

Sizing rules (varies by horizon and how stretched / strong the setup is):

| Action                       | Sizing                                                                                                               |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| TAKE PROFIT                  | Sell 100% (short-term) / 75% with runner (long-term)                                                                 |
| TRIM                         | Sell 25% / 33% / 50% based on RSI + gap + momentum stretch                                                           |
| SELL (weak setup, in profit) | Sell all shares                                                                                                      |
| BUY MORE                     | Add 25% / 33% / 50% of position value (scales with bot score; up to 60% in long-term mode on high-conviction setups) |
| HOLD                         | No action — hold current size                                                                                        |

### Sell Scanner (full-market mode + long-term toggle)

The **💸 Sell Scanner** page now supports:

- **🌐 Scan ENTIRE market (~12K tickers)** — not just the curated S&P / Nasdaq /
  Dow universe. Use pre-filters to keep the scan tractable.
- **📈 Long-term investor** horizon — same logic as Position Advisor: dampens
  trim/sell signals on stocks with strong multi-week trends, only flags true
  blowoffs.
- **Search box, action multiselect, and pagination** (25/page) so you can find a
  specific ticker or filter to just TAKE PROFIT / TRIM rows.

### Journal events

Options activity appears in `logs/journal.jsonl` as:

- `option_open` — order submitted (includes full `candidate` dict with legs)
- `option_close` — manual close issued
- `option_order_error` — submit failed
- `option_dry_run` — would have submitted but `BOT_OPTIONS_DRY_RUN=true`
- `option_cycle_error` — fatal error in the options cycle

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

| Strategy        | Best in regime   | Description                                    |
| --------------- | ---------------- | ---------------------------------------------- |
| Trend following | Bull/Bear trends | Adaptive EMA crossover with slope confirmation |
| Mean reversion  | Sideways markets | Bollinger Bands + RSI oversold/overbought      |
| Momentum        | Breakouts        | Rate of change + volume surge detection        |
| ML model        | All regimes      | Ensemble probability as directional signal     |

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
- Dashboard log: `logs/dashboard.log`
- Trade journal (JSONL): `logs/journal.jsonl`
- Symbol-universe cache: `logs/universe/{sp500,nasdaq100,dow30}.json`
- Cost-aware backtest report: `logs/cost_aware_*.json`
- Regime-weight optimizer output: `logs/regime_weights.json`
- Trained meta-label model: `models/meta_label.joblib`

Journal events include: `account_state`, `signal`, `decision`, `order`, `fill_status`,
`stop_loss`, `risk_reject`, `error`, `preflight_failed`, `user_cancel`,
`spy_trend_reject`, `vix_reject`, `earnings_blackout_reject`, `volume_reject`,
`meta_label_reject`, `heat_reject`, `correlation_reject`, `trailing_stop_update`,
`partial_take`, `time_stop`, `breakeven_shift`

## Tests

```bash
python -m pytest tests/ -q
```

83 tests covering: indicators, patterns, risk manager, position sizing & exits,
market filters, meta-label, scanner enhancements, universe loader.

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

## Deploy (Streamlit Community Cloud — free)

The dashboard can be deployed for free on [Streamlit Community Cloud](https://share.streamlit.io/) so anyone with the link can access it.

Important: Streamlit Community Cloud is a good home for the dashboard UI, but it is not a reliable always-on scheduler/worker for the trading bot. Use Streamlit Cloud for the dashboard, and run the bot separately on your own machine, a VPS, cron, GitHub Actions, Railway, Render, or another worker platform.

### Steps

1. Push this repository to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
3. Click **"New app"** and select:
   - **Repository**: `PranayReddy35/AI-Trading`
   - **Branch**: `main`
   - **Main file path**: `ai_trading/dashboard.py`
4. Under **Advanced settings → Secrets**, add your secrets (TOML format):

```toml
APCA_API_KEY_ID = "YOUR_API_KEY"
APCA_API_SECRET_KEY = "YOUR_API_SECRET"
BOT_BROKER = "robinhood"
BOT_PAPER_ONLY = "false"
BOT_STOCK_DRY_RUN = "true"
BOT_SHOW_ALPACA_PAPER = "false"
BOT_NOTIFY_EVENTS = "trade,error,risk_reject,daily_summary,drawdown,scanner_summary"

BOT_WEBHOOK_URL = ""
BOT_BUY_WEBHOOK_URL = "YOUR_BUY_CHANNEL_WEBHOOK_URL"
BOT_SELL_WEBHOOK_URL = "YOUR_SELL_CHANNEL_WEBHOOK_URL"
BOT_OTHER_WEBHOOK_URL = "YOUR_OTHER_ALERTS_WEBHOOK_URL"

ROBINHOOD_AGENTIC_ENABLED = "true"
ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "YOUR_AGENTIC_ACCOUNT_NUMBER"
ROBINHOOD_AGENTIC_BUYING_POWER = "100"
ROBINHOOD_AGENTIC_EQUITY = "100"
ROBINHOOD_USERNAME = "YOUR_ROBINHOOD_LOGIN"
ROBINHOOD_PASSWORD = "YOUR_ROBINHOOD_PASSWORD"
ROBINHOOD_MFA_CODE = ""
ROBINHOOD_DEVICE_TOKEN = ""
ROBINHOOD_USE_DOLLAR_ORDERS = "true"
ROBINHOOD_DOLLAR_AMOUNT_PER_TRADE = "25"
ROBINHOOD_ORDER_INTENTS_PATH = "logs/robinhood_order_intents.jsonl"
ROBINHOOD_APPROVALS_PATH = "logs/robinhood_approvals.jsonl"
ROBINHOOD_QUOTE_SYMBOLS = "SPY,QQQ"
ROBINHOOD_QUOTES_PATH = "logs/robinhood_quotes.json"
ROBINHOOD_PORTFOLIOS_PATH = "logs/robinhood_portfolios.json"
ROBINHOOD_SNAPSHOT_TTL_SEC = "300"
ROBINHOOD_REFRESH_ON_OPEN = "true"
ROBINHOOD_ROTATION_ENABLED = "true"
ROBINHOOD_ROTATION_BUY_LIMIT = "2"
ROBINHOOD_ROTATION_MIN_BUY_SCORE = "65"
ROBINHOOD_ROTATION_TRIM_SCORE_MAX = "45"
ROBINHOOD_ROTATION_EXIT_SCORE_MAX = "35"
ROBINHOOD_EXECUTION_MODE = "approval_queue"
ROBINHOOD_AUTO_APPROVE = "true"
ROBINHOOD_AUTO_DISPATCH = "false"
```

See `.streamlit/secrets.example.toml` for a longer copy/paste template.

5. Click **Deploy**. You'll get a public URL like `https://your-app.streamlit.app`.

### Share

Anyone with the link can view the dashboard — no login required. The app refreshes market data in real-time during market hours.

## Important warnings

- **This is not financial advice.** Use at your own risk.
- Automated trading systems can produce duplicate or erroneous orders.
- Paper trading fills do not perfectly match live market conditions.
- Past performance (backtest) does not guarantee future results.
- Always monitor your bot and have a kill switch ready.
- Start with tiny positions and scale up gradually only after consistent results.
