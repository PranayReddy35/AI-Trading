# Project Overview

This document explains what the AI-Trading project is, how the main pieces fit together, and where to look when operating or modifying it.

For step-by-step local commands, use [`LOCAL_RUNBOOK.md`](LOCAL_RUNBOOK.md). For setup, environment variables, and Streamlit deployment, use [`README.md`](README.md). For free hosting guidance and limits, use [`FREE_DEPLOYMENT.md`](FREE_DEPLOYMENT.md).

## Purpose

AI-Trading is a Python trading research, scanning, dashboard, and execution-intent system.

It can:

- Scan stocks for buy candidates.
- Scan holdings for sell, trim, hold, or take-profit candidates.
- Show a Streamlit dashboard for portfolio monitoring and action review.
- Generate Robinhood Agentic order intents for manual review.
- Support Alpaca paper/live broker workflows.
- Send Discord or webhook notifications.
- Backtest strategies and run options scans.

It cannot guarantee profit. It should be treated as decision-support and automation tooling, not as a promise of gains.

## Current Broker Model

The project now has two broker concepts:

| Area | Role |
| --- | --- |
| Robinhood Agentic | Primary local dashboard and order-intent workflow. The bot records Robinhood order intents for review. |
| Alpaca | Market data, historical bars, legacy paper positions, and older broker execution paths. |

Important Robinhood behavior:

- `BOT_BROKER=robinhood` enables the Robinhood Agentic facade.
- `BOT_STOCK_DRY_RUN=true` keeps local stock orders in review-intent mode.
- Robinhood order intents are written to `logs/robinhood_order_intents.jsonl`.
- Dashboard approvals are written to `logs/robinhood_approvals.jsonl`.
- Real Robinhood order placement still requires external Agentic review/confirmation.
- The dashboard reads Robinhood portfolio/quote snapshots from local JSON files.

## High-Level Architecture

```text
Environment/.env
    |
    v
ai_trading/config.py
    |
    +--> ai_trading/bot.py ---------> broker facade -> Alpaca or Robinhood intent log
    |          |
    |          +--> strategy/risk/notifications/journal
    |
    +--> ai_trading/dashboard.py ---> Streamlit UI
    |          |
    |          +--> scanner results, portfolio snapshots, action queue, Discord alerts
    |
    +--> ai_trading/scanner.py -----> buy/sell candidate scoring
```

## Main Entry Points

| File | Purpose |
| --- | --- |
| `ai_trading/dashboard.py` | Streamlit Robinhood Agent Console and scanner UI. |
| `ai_trading/bot.py` | Runs one trading decision cycle. |
| `ai_trading/runner.py` | Schedules/runs bot cycles and health checks. |
| `ai_trading/scanner.py` | CLI and library for stock scanning. |
| `ai_trading/broker/robinhood_intents.py` | Inspect latest Robinhood intent records. |
| `ai_trading/broker/robinhood_health.py` | Validate Robinhood Agentic config. |
| `ai_trading/options/runner.py` | Options scanner/trader CLI. |

## Dashboard Pages

| Page | What it does |
| --- | --- |
| Portfolio | Shows Robinhood investing/agentic snapshots, equity P/L, holdings, and action recommendations. |
| Action Queue | Combines buy/sell/trim/watch rows into one prioritized operator queue. |
| Buy Scanner | Finds buy candidates across curated, index, custom, or full-market universes. |
| Sell Scanner | Scores holdings or symbols for exits, trims, and take-profit candidates. |
| Position Advisor | Manually analyze one symbol or uploaded holdings. |
| Patterns | Pattern/heatmap analysis. |
| Options | Options strategy scanner/lab. |

## Action Queue Logic

The Action Queue is the main operational surface for Robinhood review.

It combines:

- Buy scanner results.
- Robinhood holding recommendations.
- Sell/trim/take-profit candidates.
- Hold/watch rows.
- Data freshness gates.
- Buying power and open-position gates.
- Discord notification routing.

Rows are labeled:

| State | Meaning |
| --- | --- |
| `READY` | Candidate is ready for manual review. |
| `REVIEW` | Candidate may be useful, but needs extra confirmation. |
| `BLOCK` | Candidate should not be acted on until the issue is fixed. |
| `WATCH` | Informational, not a trade action. |

Examples of blocked conditions:

- Robinhood quote snapshot is stale.
- Scanner quality gate failed.
- Daily trade cap is exhausted.
- Open-position cap is exhausted.
- Suggested buy exceeds available buying power.

## Data Sources

| Data | Source |
| --- | --- |
| Historical daily bars | yfinance and/or Alpaca. |
| Intraday live bars | Alpaca IEX/SIP depending on config and account access. |
| Robinhood displayed portfolio prices | Local Robinhood quote snapshot when available. |
| Portfolio holdings | Local Robinhood portfolio snapshot when available. |
| Index universes | `ai_trading/data/universe.py`, cached under `logs/universe/`. |

Robinhood snapshot files:

```text
logs/robinhood_portfolios.json
logs/robinhood_quotes.json
```

The dashboard warns when snapshots are missing or stale.

## Configuration

Configuration is loaded from environment variables, usually via `.env`.

Core files:

| File | Purpose |
| --- | --- |
| `.env` | Local secrets/config. Ignored by Git. |
| `.env.example` | Safe template for local config. |
| `.streamlit/secrets.example.toml` | Safe template for Streamlit Cloud secrets. |
| `ai_trading/config.py` | Parses environment variables into runtime settings. |

Important Robinhood settings:

```dotenv
BOT_BROKER=robinhood
BOT_PAPER_ONLY=false
BOT_STOCK_DRY_RUN=true
ROBINHOOD_AGENTIC_ENABLED=true
ROBINHOOD_AGENTIC_ACCOUNT_NUMBER=
ROBINHOOD_USE_DOLLAR_ORDERS=true
ROBINHOOD_DOLLAR_AMOUNT_PER_TRADE=25
ROBINHOOD_ORDER_INTENTS_PATH=logs/robinhood_order_intents.jsonl
ROBINHOOD_APPROVALS_PATH=logs/robinhood_approvals.jsonl
ROBINHOOD_QUOTES_PATH=logs/robinhood_quotes.json
ROBINHOOD_PORTFOLIOS_PATH=logs/robinhood_portfolios.json
```

## Notifications

Notifications are handled by `ai_trading/notifications/alerter.py`.

Routes:

| Variable | Used for |
| --- | --- |
| `BOT_BUY_WEBHOOK_URL` | Buy candidates and buy trade previews. |
| `BOT_SELL_WEBHOOK_URL` | Sell, trim, and take-profit candidates. |
| `BOT_OTHER_WEBHOOK_URL` | Risk rejects, scanner summaries, errors, daily summaries. |
| `BOT_WEBHOOK_URL` | Fallback route. |

Events are controlled by:

```dotenv
BOT_NOTIFY_EVENTS=trade,error,risk_reject,daily_summary,drawdown,scanner_summary
```

## Strategy Components

| Package/File | Purpose |
| --- | --- |
| `ai_trading/strategy/moving_average.py` | Basic MA signals. |
| `ai_trading/strategy/ensemble.py` | Multi-strategy signal ensemble. |
| `ai_trading/strategy/market_filters.py` | Macro and market-condition gates. |
| `ai_trading/strategy/patterns.py` | Pattern detection. |
| `ai_trading/strategy/multi_timeframe.py` | Multi-timeframe confirmation. |
| `ai_trading/strategy/sentiment_filter.py` | News/social sentiment gating. |

## Risk Components

| Package/File | Purpose |
| --- | --- |
| `ai_trading/risk/manager.py` | Core risk checks and trade gates. |
| `ai_trading/risk/sizing.py` | Position sizing helpers. |
| `ai_trading/risk/portfolio_sizing.py` | Vol-targeted and portfolio-aware sizing. |
| `ai_trading/risk/exits.py` | Trailing stops, partial profit, time stops. |
| `ai_trading/risk/correlation.py` | Correlation checks. |
| `ai_trading/risk/correlation_scaling.py` | Correlation-aware size scaling. |

Key risk settings include:

```dotenv
BOT_MAX_DAILY_TRADES=25
BOT_MAX_OPEN_POSITIONS=10
BOT_DAILY_LOSS_LIMIT_PCT=3.0
BOT_STOP_LOSS_PCT=2.0
BOT_PARTIAL_PROFIT_TRIGGER_PCT=1.5
BOT_PARTIAL_PROFIT_SELL_PCT=75.0
BOT_PARTIAL_PROFIT_TRAILING_STOP_PCT=1.0
BOT_GAP_OPEN_PROTECTION_PCT=4.0
```

## Logs and Journals

| File | Purpose |
| --- | --- |
| `logs/bot.log` | Bot runtime logs. |
| `logs/journal.jsonl` | Structured event journal. |
| `logs/robinhood_order_intents.jsonl` | Robinhood Agentic order intents. |
| `logs/robinhood_approvals.jsonl` | Dashboard-approved Robinhood execution queue. |
| `logs/robinhood_portfolios.json` | Local Robinhood portfolio snapshot. |
| `logs/robinhood_quotes.json` | Local Robinhood quote snapshot. |

`logs/` is ignored by Git.

## Local Commands

Install:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Run dashboard:

```bash
python -m streamlit run ai_trading/dashboard.py --server.port 8501 --server.address 127.0.0.1
```

Run bot once:

```bash
python -m ai_trading.bot --no-confirm
```

Health checks:

```bash
python -m ai_trading.runner --health-check
python -m ai_trading.broker.robinhood_health
```

Inspect Robinhood intent:

```bash
python -m ai_trading.broker.robinhood_intents latest
```

## Make Targets

`Makefile` includes shortcuts:

| Target | Command |
| --- | --- |
| `make setup` | Create venv and install dev dependencies. |
| `make install` | Reinstall package in editable mode. |
| `make test` | Run tests. |
| `make health` | Run general health check. |
| `make robinhood-health` | Run Robinhood config health check. |
| `make dashboard` | Start Streamlit dashboard. |
| `make bot` | Run one bot cycle. |
| `make scanner` | Run sample scanner command. |

## Streamlit Cloud Deployment

Use:

```text
Repository: PranayReddy35/AI-Trading
Branch: main
Main file path: ai_trading/dashboard.py
Secrets template: .streamlit/secrets.example.toml
```

Do not upload real `.env` or `.streamlit/secrets.toml` files to GitHub.

## Security Notes

Ignored by Git:

- `.env`
- `.streamlit/secrets.toml`
- `logs/`
- `.cache/`
- `models/`
- local SSH key filenames added during setup

Never commit:

- API keys.
- Discord webhook URLs.
- Robinhood account numbers.
- SSH private keys.
- Trading logs with sensitive account data.

## Known Limitations

- Robinhood portfolio/quote snapshots are local JSON files; the Streamlit app does not directly refresh them by itself.
- Technical indicators still rely on historical bars from Alpaca/yfinance.
- The dashboard can surface candidates, but it does not guarantee profitable trades.
- Real Robinhood order placement requires Agentic review/confirmation outside the local dry-run intent flow.
- Streamlit Cloud cannot access local snapshot files unless they are generated or uploaded in that deployment environment.

## Recommended Reading Order

1. [`README.md`](README.md): setup, config, features, deployment.
2. [`LOCAL_RUNBOOK.md`](LOCAL_RUNBOOK.md): exact local commands for operating without Codex.
3. [`FREE_DEPLOYMENT.md`](FREE_DEPLOYMENT.md): free Streamlit hosting guidance and limitations.
4. `PROJECT_OVERVIEW.md`: architecture and subsystem map.
5. `.env.example`: local configuration template.
6. `.streamlit/secrets.example.toml`: Streamlit Cloud secrets template.
