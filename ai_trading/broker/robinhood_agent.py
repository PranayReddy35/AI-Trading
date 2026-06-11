from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_trading.broker.alpaca_broker import AlpacaBroker
from ai_trading.broker.robinhood_health import mask_account_number
from ai_trading.broker.robinhood_snapshot import load_fresh_robinhood_quotes


logger = logging.getLogger("ai_trading")


class RobinhoodAgenticExecutionRequired(Exception):
    """Raised when local code tries to submit a Robinhood Agentic order directly."""


class RobinhoodAgenticBroker:
    """Robinhood Agentic broker facade for strategy/risk dry runs.

    Robinhood Agentic Trading executes through the Robinhood MCP tool layer. This
    local facade lets the bot target the Agentic account, generate auditable order
    intents, and reuse Alpaca market-data/clock support without silently routing
    real Robinhood orders through an unsupported local API.
    """

    def __init__(
        self,
        *,
        account_number: str,
        buying_power: float,
        equity: float,
        alpaca_api_key: str,
        alpaca_api_secret: str,
        paper: bool = False,
        intents_path: str | Path = "logs/robinhood_order_intents.jsonl",
    ) -> None:
        self.account_number = account_number
        self.buying_power = float(buying_power)
        self.equity = float(equity)
        self.paper = paper
        self.intents_path = Path(intents_path)
        self._market = AlpacaBroker(alpaca_api_key, alpaca_api_secret, paper=True)

    def is_market_open(self) -> bool:
        return self._market.is_market_open()

    def minutes_to_close(self) -> float:
        return self._market.minutes_to_close()

    def get_all_tradable_symbols(self) -> list[str]:
        return self._market.get_all_tradable_symbols()

    def get_latest_price(self, symbol: str) -> float:
        """Return latest market-data price while Robinhood execution stays intent-only."""
        prices = self.get_latest_prices([symbol])
        return float(prices.get(symbol.upper()) or self._market.get_latest_price(symbol))

    def get_latest_prices(self, symbols: list[str]) -> dict[str, float]:
        """Return latest market-data prices while Robinhood execution stays intent-only."""
        clean = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
        robinhood_quotes = load_fresh_robinhood_quotes()
        out = {
            symbol: robinhood_quotes[symbol]
            for symbol in clean
            if symbol in robinhood_quotes
        }
        missing = [symbol for symbol in clean if symbol not in out]
        if missing:
            fallback = self._market.get_latest_prices(missing)
            out.update({str(symbol).upper(): float(price) for symbol, price in fallback.items() if float(price) > 0})
        return out

    def account_state(self) -> dict:
        return {
            "status": "ACTIVE",
            "cash": self.buying_power,
            "buying_power": self.buying_power,
            "equity": self.equity,
            "pattern_day_trader": False,
            "last_equity": self.equity,
            "daytrade_count": 0,
            "portfolio_value": self.equity,
            "broker": "robinhood",
            "account_number_masked": mask_account_number(self.account_number),
        }

    def position_qty(self, symbol: str) -> int:
        return 0

    def position_details(self, symbol: str) -> dict | None:
        return None

    def all_positions(self) -> list[dict]:
        return []

    def has_open_order(self, symbol: str) -> bool:
        return False

    def cancel_all_orders(self) -> int:
        raise RobinhoodAgenticExecutionRequired(
            "Use the Robinhood MCP connector to cancel Agentic orders."
        )

    def cancel_orders_for_symbol(self, symbol: str) -> int:
        raise RobinhoodAgenticExecutionRequired(
            "Use the Robinhood MCP connector to cancel Agentic orders."
        )

    def close_position(self, symbol: str) -> None:
        self.record_order_intent(symbol=symbol, side="sell", qty=0, reason="close_position")
        raise RobinhoodAgenticExecutionRequired(
            "Use the Robinhood MCP connector to review and place Robinhood close orders."
        )

    def record_order_intent(
        self,
        *,
        symbol: str,
        side: str,
        qty: int = 0,
        reason: str = "",
        order_type: str = "market",
        limit_price: float | None = None,
        price: float | None = None,
        dollar_amount: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        intent = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "broker": "robinhood",
            "account_number_masked": mask_account_number(self.account_number),
            "symbol": symbol.upper(),
            "side": side.lower(),
            "quantity": str(qty) if dollar_amount is None else None,
            "dollar_amount": f"{dollar_amount:.2f}" if dollar_amount is not None else None,
            "type": order_type,
            "time_in_force": "gfd",
            "market_hours": "regular_hours",
            "limit_price": f"{limit_price:.2f}" if limit_price is not None else None,
            "reference_price": price,
            "reason": reason,
            "mcp_tool": "review_equity_order",
        }
        if extra:
            intent.update(extra)
        intent = {key: value for key, value in intent.items() if value is not None}

        self.intents_path.parent.mkdir(parents=True, exist_ok=True)
        with self.intents_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(intent, sort_keys=True) + "\n")
        logger.info("Robinhood Agentic order intent recorded: %s", intent)
        return intent

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        order_type: str = "market",
        limit_price: float | None = None,
        max_retries: int = 3,
    ) -> Any:
        intent = self.record_order_intent(
            symbol=symbol,
            side=side,
            qty=qty,
            reason="submit_order attempted",
            order_type=order_type,
            limit_price=limit_price,
        )
        raise RobinhoodAgenticExecutionRequired(
            "Robinhood Agentic orders must be reviewed/submitted through the MCP "
            f"connector. Intent recorded for {intent['side']} {intent['quantity']} {intent['symbol']}."
        )

    def submit_stop_loss(self, symbol: str, qty: int, stop_price: float) -> Any:
        self.record_order_intent(
            symbol=symbol,
            side="sell",
            qty=qty,
            reason="stop loss",
            order_type="stop_market",
            extra={"stop_price": f"{stop_price:.2f}"},
        )
        raise RobinhoodAgenticExecutionRequired(
            "Use the Robinhood MCP connector to review and place stop orders."
        )

    def wait_for_fill(self, order_id: str, timeout_sec: int = 60) -> dict:
        time.sleep(0)
        return {"status": "not_submitted", "order_id": order_id}


def create_broker(settings) -> AlpacaBroker | RobinhoodAgenticBroker:
    if settings.broker == "alpaca":
        return AlpacaBroker(settings.api_key, settings.api_secret, paper=settings.paper_only)
    if settings.broker == "robinhood":
        return RobinhoodAgenticBroker(
            account_number=settings.robinhood_agentic_account_number,
            buying_power=settings.robinhood_agentic_buying_power,
            equity=settings.robinhood_agentic_equity,
            alpaca_api_key=settings.api_key,
            alpaca_api_secret=settings.api_secret,
            paper=settings.paper_only,
            intents_path=settings.robinhood_order_intents_path,
        )
    raise ValueError(f"Unsupported broker: {settings.broker}")
