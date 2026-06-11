# Local Runbook: Bot and Dashboard

Use this guide when you want to run the trading bot and Streamlit dashboard yourself from Terminal, including when Codex/AI credits are exhausted.

This file assumes you are on the same computer as the repo and you are starting from the project root:

```bash
cd "/Users/pranayreddyalwa/Desktop/Trading Bots/Claude OPUS co pilot/AI-Trading/AI-Trading"
```

## 1. Safety First

This project can create live trading intents and, depending on settings, submit real orders through supported brokers.

Before running anything:

1. Keep real trading disabled until you understand the config.
2. Use small trade sizes.
3. Keep `BOT_STOCK_DRY_RUN=true` for Robinhood Agentic review mode.
4. Never commit `.env`, `.streamlit/secrets.toml`, `logs/`, or API keys.
5. No scanner or bot can guarantee profit.

## 2. Open Terminal

Open macOS Terminal, then go to the repo:

```bash
cd "/Users/pranayreddyalwa/Desktop/Trading Bots/Claude OPUS co pilot/AI-Trading/AI-Trading"
```

Check that you are in the right place:

```bash
pwd
ls
```

You should see files like:

```text
README.md
Makefile
ai_trading
requirements.txt
pyproject.toml
```

## 3. Create or Activate the Virtual Environment

If `.venv` does not exist yet:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

If `.venv` already exists:

```bash
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

You can also use Make:

```bash
make setup
```

For later runs, only activate the venv:

```bash
source .venv/bin/activate
```

## 4. Configure `.env`

Create `.env` from the example if needed:

```bash
cp .env.example .env
```

Open `.env` in your editor and set your real values:

```bash
open -e .env
```

Minimum useful local config:

```dotenv
APCA_API_KEY_ID=your_alpaca_key
APCA_API_SECRET_KEY=your_alpaca_secret

BOT_BROKER=robinhood
BOT_PAPER_ONLY=false
BOT_STOCK_DRY_RUN=true
BOT_SHOW_ALPACA_PAPER=false

BOT_NOTIFY_EVENTS=trade,error,risk_reject,daily_summary,drawdown,scanner_summary
BOT_BUY_WEBHOOK_URL=
BOT_SELL_WEBHOOK_URL=
BOT_OTHER_WEBHOOK_URL=

ROBINHOOD_AGENTIC_ENABLED=true
ROBINHOOD_AGENTIC_ACCOUNT_NUMBER=
ROBINHOOD_USE_DOLLAR_ORDERS=true
ROBINHOOD_DOLLAR_AMOUNT_PER_TRADE=25
ROBINHOOD_ORDER_INTENTS_PATH=logs/robinhood_order_intents.jsonl
ROBINHOOD_APPROVALS_PATH=logs/robinhood_approvals.jsonl
ROBINHOOD_QUOTES_PATH=logs/robinhood_quotes.json
ROBINHOOD_PORTFOLIOS_PATH=logs/robinhood_portfolios.json
ROBINHOOD_SNAPSHOT_TTL_SEC=300
```

Important:

- `BOT_STOCK_DRY_RUN=true` means the local bot records Robinhood order intents for review instead of directly placing stock orders.
- The dashboard can show Robinhood portfolio/quote data only when `logs/robinhood_portfolios.json` and `logs/robinhood_quotes.json` exist.
- If those snapshot files are missing, the dashboard still runs, but Robinhood portfolio/action panels will show missing or stale data.

## 5. Health Checks

Run these before using the bot:

```bash
python -m ai_trading.runner --health-check
python -m ai_trading.broker.robinhood_health
```

Or with Make:

```bash
make health
make robinhood-health
```

Expected result:

- No Python import errors.
- Robinhood health should say Agentic proposal mode is configured when `BOT_BROKER=robinhood`.
- If it complains about missing account number, set `ROBINHOOD_AGENTIC_ACCOUNT_NUMBER` in `.env`.

## 6. Run the Dashboard Locally

Start Streamlit:

```bash
python -m streamlit run ai_trading/dashboard.py --server.port 8501 --server.address 127.0.0.1
```

Or:

```bash
make dashboard
```

Open:

```text
http://127.0.0.1:8501
```

Useful dashboard pages:

- `Portfolio`: Robinhood account snapshots, P/L, and position recommendations.
- `Action Queue`: prioritized buy/sell/trim/watch plan with readiness gates.
- `Buy Scanner`: scan for buy candidates.
- `Sell Scanner`: scan holdings/watchlist for trim or exit candidates.
- `Position Advisor`: manually evaluate one symbol or uploaded holdings.
- `Research`: build memo, earnings, valuation, and bull-vs-bear prompts from holdings or manual tickers.

Stop the dashboard with `Control-C` in the Terminal running Streamlit.

## 6A. Deploy the Dashboard to Streamlit Community Cloud

Use Streamlit Community Cloud for the dashboard UI only.

Important:

- The dashboard deploys well on Streamlit Cloud.
- The trading bot is not a reliable fit for Streamlit Cloud as an always-on worker.
- Keep the bot running locally, on a VPS, cron, GitHub Actions, Railway, Render, or another worker process.

Before deploying:

1. Push this repo to GitHub.
2. In Streamlit Community Cloud, create a new app from this repo.
3. Set the main file path to:

```text
ai_trading/dashboard.py
```

4. In the Streamlit app settings, open `Secrets`.
5. Copy the contents of:

```text
.streamlit/secrets.example.toml
```

6. Paste them into the Streamlit Secrets editor.
7. Replace placeholder values with your real keys and settings.

Recommended deployment notes:

- Do not paste your local `.env` file directly into git.
- Do not commit `.streamlit/secrets.toml`.
- Keep `BOT_STOCK_DRY_RUN=true` until you have verified the deployed dashboard behavior.
- For Streamlit-hosted dashboards, local file paths like `logs/robinhood_quotes.json` only work if the app itself generates or ships those files in its runtime environment.
- Robinhood refreshes that depend on your local machine or local snapshots will not magically sync into Streamlit Cloud unless the deployed app can generate them there too.

## 7. Run the Bot Once

Run one cycle:

```bash
python -m ai_trading.bot --no-confirm
```

Or:

```bash
make bot
```

With Robinhood dry-run mode, the bot writes order intents to:

```text
logs/robinhood_order_intents.jsonl
```

If you are using the newer guarded Robinhood execution flow instead of pure dry-run,
the bot may also create approval records in:

```text
logs/robinhood_approvals.jsonl
```

Inspect the latest Robinhood intent:

```bash
python -m ai_trading.broker.robinhood_intents latest
```

Or:

```bash
make robinhood-intent
```

## 8. Run the Scheduler

Run the daily scheduler once during the configured time:

```bash
python -m ai_trading.runner
```

Run it in loop mode:

```bash
python -m ai_trading.runner --loop --run-time 20:10 --no-confirm
```

Stop it with `Control-C`.

## 9. Run the Scanner from Terminal

Quick scanner:

```bash
python -m ai_trading.scanner --symbols SPY,QQQ,AAPL --top 10
```

Or:

```bash
make scanner
```

The dashboard scanner is usually easier because it stores results in the UI and feeds the Action Queue.

## 10. Robinhood Snapshot Files

The local dashboard reads Robinhood data from local JSON snapshots:

```text
logs/robinhood_portfolios.json
logs/robinhood_quotes.json
```

When AI/Codex credits are exhausted, the local app cannot ask the chat Robinhood connector to refresh those files for you. You have three options:

1. Use the latest existing snapshots and watch for stale warnings in the dashboard.
2. Manually create/update those JSON files if you have another Robinhood data export path.
3. Keep using Alpaca/yfinance scanner data for technical indicators, while Robinhood snapshots are used only when available for portfolio prices.

The dashboard will clearly label missing or stale Robinhood snapshots.

You can now refresh those files locally with:

```bash
python -m ai_trading.broker.robinhood_snapshot
```

Or:

```bash
make robinhood-refresh
```

You can also refresh Robinhood snapshots directly inside the dashboard:

1. Open the sidebar.
2. Expand `Robinhood Portfolios`.
3. Choose a `Refresh scope`:
   - `Portfolio only`
   - `Portfolio + watchlist`
   - `Portfolio + scanner universe`
4. Click `Refresh Robinhood Snapshot`.

After a successful refresh, the dashboard now:

- clears stale snapshot-backed cached tables
- automatically re-scores Robinhood holdings on the next page render
- updates the Portfolio action rows without needing a second manual `Score Portfolios` click

The sidebar and Portfolio page both show Robinhood snapshot freshness and age.

To keep Robinhood snapshots updating automatically in the background:

```bash
python -m ai_trading.broker.robinhood_snapshot --loop --interval-sec 60
```

Or:

```bash
make robinhood-refresh-loop
```

Build a research prompt locally:

```bash
make research-prompt TICKER=NVDA MODE=memo
```

Or use the dashboard's **Research** page to generate memo, earnings, valuation, and bull-vs-bear prompts using Robinhood holdings or manual symbols.

Required local env vars for this command:

```dotenv
ROBINHOOD_USERNAME=your_login_email
ROBINHOOD_PASSWORD=your_login_password
ROBINHOOD_AGENTIC_ACCOUNT_NUMBER=your_account_number
```

Optional:

```dotenv
ROBINHOOD_MFA_CODE=
ROBINHOOD_DEVICE_TOKEN=
ROBINHOOD_QUOTE_SYMBOLS=SPY,QQQ
ROBINHOOD_ACCOUNT_NUMBERS=
```

`ROBINHOOD_QUOTE_SYMBOLS` lets you keep extra names refreshed even when they are not current holdings.
`ROBINHOOD_ACCOUNT_NUMBERS` lets you include additional Robinhood account numbers in the combined snapshot when needed.

### Robinhood market-open automation

Optional Robinhood automation settings:

```dotenv
ROBINHOOD_REFRESH_ON_OPEN=true
ROBINHOOD_ROTATION_ENABLED=true
ROBINHOOD_ROTATION_BUY_LIMIT=2
ROBINHOOD_ROTATION_MIN_BUY_SCORE=65
ROBINHOOD_ROTATION_TRIM_SCORE_MAX=45
ROBINHOOD_ROTATION_EXIT_SCORE_MAX=35
```

Robinhood execution modes:

```dotenv
# safest
ROBINHOOD_EXECUTION_MODE=intent_only
BOT_STOCK_DRY_RUN=true

# auto-create approvals, but do not dispatch them
ROBINHOOD_EXECUTION_MODE=approval_queue
BOT_STOCK_DRY_RUN=false
ROBINHOOD_AUTO_APPROVE=true
ROBINHOOD_AUTO_DISPATCH=false

# auto-create and dispatch approvals to your executor webhook
ROBINHOOD_EXECUTION_MODE=auto_dispatch
BOT_STOCK_DRY_RUN=false
ROBINHOOD_AUTO_APPROVE=true
ROBINHOOD_AUTO_DISPATCH=true
ROBINHOOD_EXECUTOR_WEBHOOK_URL=https://your-executor-endpoint
```

What this Robinhood automation does:

- refreshes Robinhood snapshots at bot start
- scores current holdings for hold/trim/sell rotation
- scores replacement buy candidates from your configured symbol universe
- writes a `robinhood_rotation_plan` into the journal
- optionally creates and dispatches Robinhood approval records through the guarded executor flow

## 11. Discord Alerts

Set these in `.env`:

```dotenv
BOT_BUY_WEBHOOK_URL=https://discord.com/api/webhooks/...
BOT_SELL_WEBHOOK_URL=https://discord.com/api/webhooks/...
BOT_OTHER_WEBHOOK_URL=https://discord.com/api/webhooks/...
BOT_NOTIFY_EVENTS=trade,error,risk_reject,daily_summary,drawdown,scanner_summary
```

Dashboard routing:

- Buy candidates go to the buy webhook.
- Sell/trim candidates go to the sell webhook.
- Scanner summaries, risk rejects, errors, and daily summaries go to the other webhook.

## 12. Logs and Files to Check

Common files:

```text
logs/bot.log
logs/journal.jsonl
logs/robinhood_order_intents.jsonl
logs/robinhood_approvals.jsonl
logs/robinhood_portfolios.json
logs/robinhood_quotes.json
```

View recent bot log lines:

```bash
tail -n 80 logs/bot.log
```

View recent journal events:

```bash
tail -n 20 logs/journal.jsonl
```

View recent Robinhood intents:

```bash
tail -n 20 logs/robinhood_order_intents.jsonl
```

View recent dashboard approvals:

```bash
tail -n 20 logs/robinhood_approvals.jsonl
```

List pending Robinhood approvals:

```bash
python -m ai_trading.broker.robinhood_executor list
```

Print the next approval's Robinhood Agentic review/place payload:

```bash
python -m ai_trading.broker.robinhood_executor payload
```

After an external Robinhood Agentic executor places the order:

```bash
python -m ai_trading.broker.robinhood_executor mark-executed APPROVAL_ID --order-id ROBINHOOD_ORDER_ID
```

If execution fails:

```bash
python -m ai_trading.broker.robinhood_executor mark-failed APPROVAL_ID --note "reason"
```

## 13. Common Problems

### `ModuleNotFoundError`

Install dependencies again:

```bash
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

### `streamlit: command not found`

Use Python module execution:

```bash
python -m streamlit run ai_trading/dashboard.py --server.port 8501
```

### Port `8501` is already in use

Find and stop the old Streamlit process:

```bash
lsof -nP -iTCP:8501 -sTCP:LISTEN
pkill -f streamlit
```

Then restart:

```bash
python -m streamlit run ai_trading/dashboard.py --server.port 8501 --server.address 127.0.0.1
```

### Dashboard shows stale Robinhood prices

That means `logs/robinhood_quotes.json` is old, missing, or the current scored table was built before the latest quote refresh.

Use one of these:

```bash
make robinhood-refresh
```

or use the dashboard sidebar:

1. Expand `Robinhood Portfolios`
2. Click `Refresh Robinhood Snapshot`

The dashboard will now auto re-score Robinhood holdings after a successful refresh. If a scanner table still looks old, re-run that specific scanner page.

### Dashboard says `No market data returned for this holding`

This usually means one of these:

- the holding is crypto (`BTC`, `ETH`, `DOGE`, etc.), and the stock technical scanner does not produce a score for crypto
- Robinhood quote data is available, but no technical bars were returned for that symbol
- the symbol is missing from both Robinhood quote refresh coverage and the fallback market-data path

The dashboard now shows quote-backed fallback rows when it can, but unsupported holdings may still appear as quote-only instead of fully scored.

### Bot is not creating Robinhood intents

Check:

```bash
python -m ai_trading.broker.robinhood_health
```

Confirm `.env` has:

```dotenv
BOT_BROKER=robinhood
BOT_PAPER_ONLY=false
BOT_STOCK_DRY_RUN=true
ROBINHOOD_AGENTIC_ENABLED=true
ROBINHOOD_AGENTIC_ACCOUNT_NUMBER=your_account_number
```

### GitHub push fails

Authenticate GitHub first. SSH path:

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub
```

Add the public key in GitHub:

```text
GitHub -> Settings -> SSH and GPG keys -> New SSH key
```

Then:

```bash
git remote set-url origin git@github.com:PranayReddy35/AI-Trading.git
git push origin main
```

## 14. Normal Local Workflow

Use this checklist when running without Codex:

```bash
cd "/Users/pranayreddyalwa/Desktop/Trading Bots/Claude OPUS co pilot/AI-Trading/AI-Trading"
source .venv/bin/activate
python -m ai_trading.runner --health-check
python -m ai_trading.broker.robinhood_health
python -m streamlit run ai_trading/dashboard.py --server.port 8501 --server.address 127.0.0.1
```

In a second Terminal:

```bash
cd "/Users/pranayreddyalwa/Desktop/Trading Bots/Claude OPUS co pilot/AI-Trading/AI-Trading"
source .venv/bin/activate
python -m ai_trading.bot --no-confirm
python -m ai_trading.broker.robinhood_intents latest
```

Review the dashboard before taking any real trading action.
