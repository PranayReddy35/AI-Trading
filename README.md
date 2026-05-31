# AI-Trading (Educational Starter)

> ⚠️ **Educational only — not financial advice.**
>
> This project defaults to **paper trading only**, tiny sizing, long-only behavior (no shorting), and daily-bar workflows. It is intentionally conservative and is **not** for high-frequency trading.

## What is included

Clean module structure:

- `ai_trading/config.py` – runtime settings from environment variables
- `ai_trading/data/` – market data access
- `ai_trading/strategy/` – moving-average signal generation
- `ai_trading/risk/` – risk checks and safety guards
- `ai_trading/broker/` – Alpaca paper broker wrapper
- `ai_trading/storage/` – logging + JSONL journaling
- `ai_trading/bot.py` – rule-based paper bot (single run)
- `ai_trading/runner.py` – daily scheduling stub/runner guidance
- `ai_trading/backtest.py` – backtest matching rule-based MA logic with daily bars
- `ai_trading/ml/predict_direction.py` – separate ML (logistic regression) next-day direction script

## Safety and risk controls

`RiskManager` includes:

- paper-trading-only safety guard
- market-closed guard
- cash threshold guard
- duplicate/open order prevention
- max shares / position sizing cap
- max daily trades
- consecutive error stop
- no-shorting behavior (sell only existing long position)

## Setup

```bash
cd /tmp/workspace/PranayReddy35/AI-Trading
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Environment variables (paper bot)

```bash
export APCA_API_KEY_ID="YOUR_PAPER_KEY"
export APCA_API_SECRET_KEY="YOUR_PAPER_SECRET"

# Optional runtime config
export BOT_SYMBOL="SPY"
export BOT_FAST_MA="5"
export BOT_SLOW_MA="20"
export BOT_LOOKBACK_DAYS="90"
export BOT_MAX_SHARES="1"
export BOT_MIN_CASH_THRESHOLD="100"
export BOT_MAX_DAILY_TRADES="1"
export BOT_MAX_CONSECUTIVE_ERRORS="3"
export BOT_PAPER_ONLY="true"
```

## Run the paper bot (single cycle)

```bash
python -m ai_trading.bot
```

Artifacts:
- logs: `logs/bot.log`
- journal (signals/orders/account/errors): `logs/journal.jsonl`

## Scheduling guidance (daily execution)

Use OS scheduler in practice (cron/systemd/task scheduler). A simple runner stub is included:

```bash
# Run once immediately
python -m ai_trading.runner

# Run every day at a UTC time (stub loop)
python -m ai_trading.runner --loop --run-time 20:10
```

## Backtest script (rule-based MA)

```bash
python -m ai_trading.backtest --symbol SPY --start 2018-01-01 --end 2025-01-01 --fast-ma 5 --slow-ma 20 --max-shares 1
```

You can also use local CSV:

```bash
python -m ai_trading.backtest --csv /absolute/path/to/bars.csv
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
python -m ai_trading.ml.predict_direction --symbol SPY --save-model /absolute/path/model.joblib
```

## Notes

- Keep this project in paper mode unless you fully understand and accept market, execution, and software risks.
- Daily bars and tiny sizing are intentional to keep behavior conservative and educational.
