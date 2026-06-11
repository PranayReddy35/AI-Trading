PYTHON := .venv/bin/python
PIP := $(PYTHON) -m pip
STREAMLIT := $(PYTHON) -m streamlit

.PHONY: setup install test health robinhood-health robinhood-refresh robinhood-refresh-loop robinhood-intent robinhood-intent-with-quote robinhood-approvals robinhood-executor-payload dashboard bot scanner research-prompt

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e '.[dev]'

install:
	$(PIP) install -e '.[dev]'

test:
	$(PYTHON) -m pytest -q

health:
	$(PYTHON) -m ai_trading.runner --health-check

robinhood-health:
	$(PYTHON) -m ai_trading.broker.robinhood_health

robinhood-refresh:
	$(PYTHON) -m ai_trading.broker.robinhood_snapshot

robinhood-refresh-loop:
	$(PYTHON) -m ai_trading.broker.robinhood_snapshot --loop

robinhood-intent:
	$(PYTHON) -m ai_trading.broker.robinhood_intents latest

robinhood-intent-with-quote:
	$(PYTHON) -m ai_trading.broker.robinhood_intents latest --quote-json "$(QUOTE_JSON)"

robinhood-approvals:
	$(PYTHON) -m ai_trading.broker.robinhood_executor list

robinhood-executor-payload:
	$(PYTHON) -m ai_trading.broker.robinhood_executor payload

dashboard:
	$(STREAMLIT) run ai_trading/dashboard.py --server.headless true --server.port 8501 --browser.gatherUsageStats false

bot:
	$(PYTHON) -m ai_trading.bot --no-confirm

scanner:
	$(PYTHON) -m ai_trading.scanner --symbols SPY,QQQ,AAPL --top 10

research-prompt:
	$(PYTHON) -m ai_trading.research --ticker "$(TICKER)" --mode "$(MODE)"
