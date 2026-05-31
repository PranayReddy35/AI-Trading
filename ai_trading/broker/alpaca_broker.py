from __future__ import annotations

import logging
import time
from typing import Any

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    TrailingStopOrderRequest,
)

logger = logging.getLogger("ai_trading")


class OrderError(Exception):
    """Raised when an order fails after retries."""


class AlpacaBroker:
    def __init__(self, api_key: str, api_secret: str, paper: bool = True) -> None:
        self.paper = paper
        self.client = TradingClient(api_key, api_secret, paper=paper)

    def is_market_open(self) -> bool:
        return bool(self.client.get_clock().is_open)

    def account_state(self) -> dict:
        account = self.client.get_account()
        return {
            "status": str(account.status),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "equity": float(account.equity),
            "pattern_day_trader": bool(account.pattern_day_trader),
            "last_equity": float(account.last_equity),
            "daytrade_count": int(account.daytrade_count),
            "portfolio_value": float(account.portfolio_value),
        }

    def position_qty(self, symbol: str) -> int:
        try:
            position = self.client.get_open_position(symbol)
            return int(float(position.qty))
        except Exception:
            return 0

    def position_details(self, symbol: str) -> dict | None:
        """Get full position details including cost basis and unrealized P&L."""
        try:
            position = self.client.get_open_position(symbol)
            return {
                "symbol": str(position.symbol),
                "qty": int(float(position.qty)),
                "avg_entry_price": float(position.avg_entry_price),
                "current_price": float(position.current_price),
                "market_value": float(position.market_value),
                "unrealized_pl": float(position.unrealized_pl),
                "unrealized_plpc": float(position.unrealized_plpc),
            }
        except Exception:
            return None

    def all_positions(self) -> list[dict]:
        """Get all open positions."""
        positions = self.client.get_all_positions()
        return [
            {
                "symbol": str(p.symbol),
                "qty": int(float(p.qty)),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
            }
            for p in positions
        ]

    def has_open_order(self, symbol: str) -> bool:
        orders = self.client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        )
        return len(orders) > 0

    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count of cancelled orders."""
        responses = self.client.cancel_orders()
        count = len(responses) if responses else 0
        logger.info("Cancelled %d open orders", count)
        return count

    def cancel_orders_for_symbol(self, symbol: str) -> int:
        """Cancel all open orders for a specific symbol."""
        orders = self.client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        )
        count = 0
        for order in orders:
            try:
                self.client.cancel_order_by_id(str(order.id))
                count += 1
            except Exception as exc:
                logger.warning("Failed to cancel order %s: %s", order.id, exc)
        return count

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        order_type: str = "market",
        limit_price: float | None = None,
        max_retries: int = 3,
    ) -> Any:
        """Submit an order with retry logic for transient failures."""
        order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL

        for attempt in range(1, max_retries + 1):
            try:
                if order_type == "limit" and limit_price is not None:
                    order_request = LimitOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=order_side,
                        time_in_force=TimeInForce.DAY,
                        limit_price=round(limit_price, 2),
                    )
                else:
                    order_request = MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=order_side,
                        time_in_force=TimeInForce.DAY,
                    )
                order = self.client.submit_order(order_data=order_request)
                logger.info(
                    "Order submitted: id=%s symbol=%s side=%s qty=%s type=%s",
                    order.id, symbol, side, qty, order_type,
                )
                return order
            except Exception as exc:
                if attempt >= max_retries:
                    raise OrderError(
                        f"Order failed after {max_retries} attempts: {exc}"
                    ) from exc
                wait = 2 ** attempt
                logger.warning(
                    "Order attempt %d/%d failed (%s), retrying in %ds...",
                    attempt, max_retries, exc, wait,
                )
                time.sleep(wait)

    def submit_stop_loss(
        self,
        symbol: str,
        qty: int,
        stop_price: float,
    ) -> Any:
        """Submit a stop-loss order to protect a position."""
        order_request = TrailingStopOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            trail_price=round(stop_price, 2),
        )
        order = self.client.submit_order(order_data=order_request)
        logger.info("Stop-loss order submitted: id=%s stop_price=%.2f", order.id, stop_price)
        return order

    def wait_for_fill(self, order_id: str, timeout_sec: int = 60) -> dict:
        """Poll order status until filled or timeout. Returns order state."""
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            order = self.client.get_order_by_id(order_id)
            status = str(order.status)
            if status in ("filled", "partially_filled"):
                filled_qty = int(float(order.filled_qty)) if order.filled_qty else 0
                filled_avg_price = (
                    float(order.filled_avg_price) if order.filled_avg_price else 0.0
                )
                return {
                    "status": status,
                    "filled_qty": filled_qty,
                    "filled_avg_price": filled_avg_price,
                    "order_id": str(order.id),
                }
            if status in ("cancelled", "expired", "rejected", "suspended"):
                return {
                    "status": status,
                    "filled_qty": 0,
                    "filled_avg_price": 0.0,
                    "order_id": str(order.id),
                }
            time.sleep(2)

        return {
            "status": "timeout",
            "filled_qty": 0,
            "filled_avg_price": 0.0,
            "order_id": order_id,
        }

    def get_latest_price(self, symbol: str) -> float:
        """Get latest trade price for a symbol."""
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestTradeRequest

        data_client = StockHistoricalDataClient(
            self.client._api_key, self.client._secret_key
        )
        request = StockLatestTradeRequest(symbol_or_symbols=symbol)
        trade = data_client.get_stock_latest_trade(request)
        if isinstance(trade, dict):
            return float(trade[symbol].price)
        return float(trade.price)
