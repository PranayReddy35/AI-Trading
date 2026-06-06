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

## How to Overcome the Poor Fits

The solution is not to force Streamlit Cloud to behave like a trading server. Use Streamlit as the UI, then move state, scheduling, and execution-adjacent work into services that are better suited for those jobs.

| Problem | Free workaround | More robust option | Remaining tradeoff |
| --- | --- | --- | --- |
| Always-on autonomous trading loops | Run bot cycles locally with `python -m ai_trading.bot --no-confirm`, or trigger dry-run scans from GitHub Actions. | Cheap VPS, home mini-PC, or cloud VM running a supervised process. | Free web apps can sleep/restart; always-on reliability usually costs money. |
| Long-running schedulers | Use GitHub Actions `schedule` for non-live dry-run scans/reports. | VPS cron/systemd timer or managed job scheduler. | Scheduled jobs can be delayed; do not use free scheduled jobs for time-critical live execution. |
| Local Robinhood MCP/Agentic connector access | Keep Robinhood execution review local/manual; Streamlit only displays intents and queues. | Build a secure backend service that has authorized broker/API access and strong audit controls. | Do not expose Robinhood credentials or direct order execution through a public dashboard. |
| Durable local files | Store snapshots, intents, and journal data in Supabase/Postgres or another external database. | Managed Postgres plus backups. | Free databases can pause or hit limits; keep schema/backups reproducible. |
| Guaranteed real-time execution | Use dashboard for review only; rely on broker-side data/order confirmations. | Broker API backend near market-data/execution provider with monitoring. | You can reduce latency, but you cannot guarantee real-time behavior on public internet/free infra. |

### Recommended Robust Free-ish Architecture

```text
Streamlit Cloud UI
    |
    +--> Reads config from Streamlit Secrets
    +--> Reads/writes durable state from external DB
    +--> Shows scanners, action queue, portfolio review
    |
External DB: Supabase/Postgres
    |
    +--> Stores snapshots, order intents, journal events, user settings
    |
Scheduler: local cron, GitHub Actions, or VPS cron
    |
    +--> Runs scans/bot dry-run cycles
    +--> Writes results to DB
    +--> Sends Discord alerts
    |
Manual Robinhood Agentic Review
    |
    +--> User reviews/places real Robinhood orders
```

### Practical Upgrade Path

1. **Phase 1: Free UI**
   Deploy Streamlit Cloud with `BOT_STOCK_DRY_RUN=true`. Use it for scanning, queue review, and Discord alerts.

2. **Phase 2: Durable State**
   Add an external database for snapshots, journal events, scanner results, and order intents. This removes dependence on Streamlit's local filesystem.

3. **Phase 3: Scheduled Dry Runs**
   Add scheduled dry-run scans. Use GitHub Actions for reports, or local cron if you want the job to run from your own machine.

4. **Phase 4: Robust Worker**
   Move the bot loop to a supervised worker on a VPS or always-on local machine. Keep the Streamlit app as the cockpit.

5. **Phase 5: Execution Controls**
   Keep real orders behind manual confirmation, strict risk gates, audit logs, and Discord alerts. Do not let a public dashboard place live orders without a separate approval layer.

### What We Should Build Next

To make this app robust for cloud use, the next engineering steps are:

1. Add a storage abstraction:

```text
local JSONL/files  <-->  Supabase/Postgres
```

2. Store these records outside the Streamlit filesystem:

```text
scanner_results
sell_scanner_results
robinhood_portfolio_snapshots
robinhood_quote_snapshots
order_intents
approval_queue
journal_events
dashboard_settings
```

3. Add a background-worker command:

```bash
python -m ai_trading.worker --scan --write-db
python -m ai_trading.worker --bot-cycle --dry-run --write-db
```

4. Add dashboard DB mode:

```toml
BOT_STORAGE_BACKEND = "supabase"
SUPABASE_URL = "..."
SUPABASE_SERVICE_ROLE_KEY = "..."
```

5. Add health checks:

```text
last_scan_at
last_snapshot_at
last_worker_heartbeat_at
last_discord_alert_at
```

This lets the dashboard show whether the worker is alive and whether data is fresh.

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
ROBINHOOD_APPROVALS_PATH = "logs/robinhood_approvals.jsonl"
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

## Using It From an iPhone Away From Home

When you are away from your laptop, use the system as a mobile review-and-approval workflow.

### Safe Mobile Workflow

```text
Worker runs scans / bot dry-run cycle
    |
    +--> Writes scanner results and order intents
    +--> Sends Discord alert
    |
iPhone
    |
    +--> Open Streamlit dashboard
    +--> Review Action Queue
    +--> Review reason, score, P/L, data freshness, and gate state
    +--> Approve manually through Robinhood Agentic / Robinhood app
```

Current behavior:

- The dashboard can be used from Safari on iPhone.
- Discord alerts can notify you when buy/sell/trim candidates appear.
- The bot can create Robinhood order intents.
- `BOT_STOCK_DRY_RUN=true` prevents the cloud dashboard from silently placing real stock orders.
- Real Robinhood execution still requires manual review/confirmation.

### iPhone Checklist

Before leaving home:

1. Make sure the Streamlit app is deployed and opens on your iPhone.
2. Add the Streamlit URL to your iPhone home screen.
3. Confirm Discord mobile notifications work.
4. Confirm buy/sell/other webhooks route to the right Discord channels.
5. Confirm `BOT_STOCK_DRY_RUN=true`.
6. Confirm the dashboard's Action Queue clearly shows stale-data warnings.

When an alert comes in:

1. Open Discord and read the alert.
2. Open Streamlit dashboard on iPhone.
3. Go to `Action Queue`.
4. Check `State`, `Gate Reason`, `Priority`, `P/L`, and `Reason`.
5. If the action still makes sense, place/review the real order through Robinhood Agentic or the Robinhood app.
6. Do not act on `BLOCK` rows.

### If You Want More Automation

The next safer automation step is not "let Streamlit place orders." It is:

```text
Intent created -> Discord alert -> mobile approval -> backend executes -> audit log
```

That requires:

- Durable database for intents and approvals.
- Authenticated mobile dashboard access.
- One-time approval tokens or signed approval links.
- Strict risk checks before execution.
- Full audit trail.
- Kill switch.

Until that approval backend exists, the safest mobile setup is dashboard + Discord + manual Robinhood confirmation.

### Path B: Robinhood Approval Queue

The dashboard now supports the first half of Path B:

```text
Action Queue row -> Approve Selected Candidate -> logs/robinhood_approvals.jsonl
```

An approval record contains:

- Approval ID.
- Redacted account number.
- Buy/sell side.
- Symbol.
- Dollar amount or quantity.
- Reason and gate context.
- Pending execution status.
- Robinhood-compatible executor payload without storing the full account number.

The missing final piece is the secure executor that can consume this queue and call Robinhood Agentic execution. The executor must run in a trusted environment with access to the Robinhood Agentic connector or a supported Robinhood execution backend.

### Path B Phase 2: Executor CLI

The repo includes an executor queue processor:

```bash
python -m ai_trading.broker.robinhood_executor list
python -m ai_trading.broker.robinhood_executor next
python -m ai_trading.broker.robinhood_executor payload
```

The `payload` command prints the exact payloads needed for:

```text
review_equity_order
place_equity_order
```

The place payload includes a stable `ref_id` so retries can be idempotent.

After an external Robinhood Agentic executor places the order, mark it:

```bash
python -m ai_trading.broker.robinhood_executor mark-executed APPROVAL_ID --order-id ROBINHOOD_ORDER_ID
```

If execution fails:

```bash
python -m ai_trading.broker.robinhood_executor mark-failed APPROVAL_ID --note "reason"
```

For a future trusted backend, dispatch approvals by webhook:

```bash
ROBINHOOD_EXECUTOR_WEBHOOK_URL="https://your-trusted-executor.example/robinhood" \
python -m ai_trading.broker.robinhood_executor dispatch-webhook APPROVAL_ID
```

That webhook receiver must own the actual Robinhood Agentic execution call and must implement authentication, idempotency, logging, and a kill switch.
