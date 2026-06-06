# Free Deployment Guide

This guide explains how to deploy the AI-Trading dashboard for free and what the free deployment can and cannot do.

## Recommended Free Setup

Use this split:

| Component | Free host | Purpose |
| --- | --- | --- |
| Streamlit dashboard | Streamlit Community Cloud | Interactive UI for scans, portfolio review, and action queue. |
| Source control | GitHub | Stores code and deploys Streamlit app. |
| Secrets | Streamlit Cloud secrets | Holds Alpaca keys, bot settings, Discord webhooks, Robinhood config placeholders. |
| Local machine | Your laptop | Optional place to run bot cycles and refresh local Robinhood snapshots. |

This is the most practical free setup because Streamlit Community Cloud is designed for web apps, not always-on trading daemons.

## What Runs on Streamlit Cloud

Good fit:

- Dashboard UI.
- Buy Scanner.
- Sell Scanner.
- Position Advisor.
- Action Queue.
- Discord notification buttons.
- Alpaca/yfinance market-data powered views.
- Reading Streamlit secrets.

Poor fit:

- Always-on autonomous trading loops.
- Long-running schedulers.
- Local Robinhood MCP/Agentic connector access.
- Durable local files that must survive app restarts.
- Guaranteed real-time execution.

## Important Robinhood Limitation

The local app uses Robinhood snapshot files:

```text
logs/robinhood_portfolios.json
logs/robinhood_quotes.json
```

On Streamlit Cloud, those files will not automatically exist unless you add a cloud-safe refresh mechanism or upload/generate snapshots there.

That means:

- The dashboard can still deploy and run.
- Scanner pages can still work through Alpaca/yfinance.
- Robinhood portfolio panels may show missing or stale snapshots.
- Real Robinhood order placement still requires Agentic review outside the Streamlit app.

Keep this setting on cloud deployments:

```toml
BOT_STOCK_DRY_RUN = "true"
```

## Streamlit Cloud Steps

1. Push code to GitHub.
2. Go to Streamlit Community Cloud.
3. Create a new app.
4. Choose:

```text
Repository: PranayReddy35/AI-Trading
Branch: main
Main file path: ai_trading/dashboard.py
```

5. Open app settings.
6. Paste secrets from `.streamlit/secrets.example.toml`.
7. Replace placeholders with real values.
8. Deploy.

## Streamlit Secrets Template

Paste this into Streamlit Cloud secrets:

```toml
APCA_API_KEY_ID = "YOUR_ALPACA_KEY"
APCA_API_SECRET_KEY = "YOUR_ALPACA_SECRET"

BOT_BROKER = "robinhood"
BOT_PAPER_ONLY = "false"
BOT_STOCK_DRY_RUN = "true"
BOT_SHOW_ALPACA_PAPER = "false"
BOT_NOTIFY_EVENTS = "trade,error,risk_reject,daily_summary,drawdown,scanner_summary"

BOT_WEBHOOK_URL = ""
BOT_BUY_WEBHOOK_URL = "YOUR_BUY_DISCORD_WEBHOOK"
BOT_SELL_WEBHOOK_URL = "YOUR_SELL_DISCORD_WEBHOOK"
BOT_OTHER_WEBHOOK_URL = "YOUR_OTHER_DISCORD_WEBHOOK"

ROBINHOOD_AGENTIC_ENABLED = "true"
ROBINHOOD_AGENTIC_ACCOUNT_NUMBER = "YOUR_AGENTIC_ACCOUNT_NUMBER"
ROBINHOOD_USE_DOLLAR_ORDERS = "true"
ROBINHOOD_DOLLAR_AMOUNT_PER_TRADE = "25"
ROBINHOOD_AGENTIC_BUYING_POWER = "0"
ROBINHOOD_AGENTIC_EQUITY = "0"
ROBINHOOD_ORDER_INTENTS_PATH = "logs/robinhood_order_intents.jsonl"
ROBINHOOD_QUOTES_PATH = "logs/robinhood_quotes.json"
ROBINHOOD_PORTFOLIOS_PATH = "logs/robinhood_portfolios.json"
ROBINHOOD_SNAPSHOT_TTL_SEC = "300"

BOT_MAX_BUYS_PER_CYCLE = "0"
BOT_MAX_DAILY_TRADES = "25"
BOT_MAX_OPEN_POSITIONS = "10"
BOT_MAX_SHARES = "1"
BOT_MIN_CASH_THRESHOLD = "25"

BOT_DAILY_LOSS_LIMIT_PCT = "3.0"
BOT_STOP_LOSS_PCT = "2.0"
BOT_GAP_OPEN_PROTECTION_PCT = "4.0"
BOT_PARTIAL_PROFIT_TRIGGER_PCT = "1.5"
BOT_PARTIAL_PROFIT_SELL_PCT = "75.0"
BOT_PARTIAL_PROFIT_MAX_HOLD_BARS = "0"
BOT_PARTIAL_PROFIT_TRAILING_STOP_PCT = "1.0"

BOT_DASHBOARD_SCAN_RESULT_TTL_SEC = "300"
BOT_CACHE_TTL_SEC = "60"
BOT_MARKET_DATA_FEED = "auto"
```

## Robustness Checklist

Before relying on the deployed dashboard:

1. Confirm the app boots without import errors.
2. Confirm `BOT_STOCK_DRY_RUN=true`.
3. Confirm Alpaca keys work for scanner data.
4. Confirm Discord route buttons work.
5. Confirm stale Robinhood snapshot warnings are visible and understood.
6. Keep real order review outside Streamlit.
7. Check logs after every deploy.

## If You Need an Always-On Bot

Free Streamlit is not the right always-on bot host.

Better options:

- Run the bot locally from your laptop using `LOCAL_RUNBOOK.md`.
- Use a cheap VPS when you are ready for always-on operation.
- Use GitHub Actions only for non-live scheduled scans or dry-run reports.

Avoid running live autonomous trading from a free web app process. Streamlit can restart, sleep, lose local state, or rerun unexpectedly.

## Recommended Operating Model

Use Streamlit Cloud as the cockpit:

1. Open dashboard.
2. Run scanners.
3. Review Action Queue.
4. Send Discord summaries if useful.
5. Review Robinhood Agentic orders manually.

Use local runs for bot cycles:

```bash
python -m ai_trading.bot --no-confirm
python -m ai_trading.broker.robinhood_intents latest
```

That gives you a free hosted UI without pretending a free dashboard host is a reliable trading execution server.
