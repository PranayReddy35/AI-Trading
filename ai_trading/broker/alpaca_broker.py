from __future__ import annotations

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest


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
        }

    def position_qty(self, symbol: str) -> int:
        try:
            position = self.client.get_open_position(symbol)
            return int(float(position.qty))
        except Exception:
            return 0

    def has_open_order(self, symbol: str) -> bool:
        orders = self.client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        )
        return len(orders) > 0

    def submit_order(self, symbol: str, side: str, qty: int):
        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        return self.client.submit_order(order_data=order_request)
