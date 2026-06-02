"""Options broker — wraps alpaca-py for single + multi-leg option orders.

Uses the existing alpaca-py TradingClient (paper or live). For multi-leg
strategies we use OrderClass.MLEG with a list of OptionLegRequest legs.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    OrderClass,
    OrderSide,
    OrderType,
    PositionIntent,
    TimeInForce,
)
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    OptionLegRequest,
)

from ai_trading.options.strategies import StrategyCandidate

logger = logging.getLogger("ai_trading.options.broker")


class OptionsBroker:
    """Thin wrapper around alpaca-py TradingClient for option orders."""

    def __init__(self, api_key: str | None = None, api_secret: str | None = None, paper: bool = True):
        api_key = api_key or os.environ.get("APCA_API_KEY_ID", "")
        api_secret = api_secret or os.environ.get("APCA_API_SECRET_KEY", "")
        if not api_key or not api_secret:
            raise RuntimeError("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY")
        self.client = TradingClient(api_key, api_secret, paper=paper)
        self.paper = paper

    # ── account ─────────────────────────────────────────────────────────────
    def options_buying_power(self) -> float:
        acct = self.client.get_account()
        return float(getattr(acct, "options_buying_power", acct.buying_power) or acct.buying_power)

    def options_trading_level(self) -> int:
        acct = self.client.get_account()
        return int(getattr(acct, "options_trading_level", 0) or 0)

    def all_option_positions(self) -> list[dict]:
        out: list[dict] = []
        try:
            positions = self.client.get_all_positions()
        except Exception as exc:
            logger.warning("get_all_positions failed: %s", exc)
            return out
        for p in positions:
            asset_class = str(getattr(p, "asset_class", "") or "")
            if "option" not in asset_class.lower():
                continue
            out.append({
                "symbol": str(p.symbol),
                "qty": int(float(p.qty)),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(getattr(p, "current_price", 0) or 0),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
            })
        return out

    # ── single-leg ──────────────────────────────────────────────────────────
    def submit_single_leg(
        self,
        occ_symbol: str,
        qty: int,
        side: str,                       # 'buy' | 'sell'
        intent: str,                      # 'open' | 'close'
        order_type: str = "limit",       # 'market' | 'limit'
        limit_price: float | None = None,
        tif: str = "day",
        max_retries: int = 3,
    ) -> Any:
        side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        intent_enum = {
            ("buy", "open"):  PositionIntent.BUY_TO_OPEN,
            ("buy", "close"): PositionIntent.BUY_TO_CLOSE,
            ("sell", "open"): PositionIntent.SELL_TO_OPEN,
            ("sell", "close"):PositionIntent.SELL_TO_CLOSE,
        }[(side.lower(), intent.lower())]
        tif_enum = TimeInForce.DAY if tif == "day" else TimeInForce.GTC

        for attempt in range(1, max_retries + 1):
            try:
                if order_type == "market":
                    req = MarketOrderRequest(
                        symbol=occ_symbol, qty=qty, side=side_enum,
                        time_in_force=tif_enum, position_intent=intent_enum,
                    )
                else:
                    if limit_price is None:
                        raise ValueError("limit_price required for limit orders")
                    req = LimitOrderRequest(
                        symbol=occ_symbol, qty=qty, side=side_enum,
                        time_in_force=tif_enum, position_intent=intent_enum,
                        limit_price=round(float(limit_price), 2),
                    )
                order = self.client.submit_order(order_data=req)
                logger.info("Option order submitted id=%s sym=%s side=%s qty=%s type=%s",
                            order.id, occ_symbol, side, qty, order_type)
                return order
            except Exception as exc:
                if attempt >= max_retries:
                    raise
                wait = min(2 ** attempt, 8)
                logger.warning("Attempt %d/%d failed (%s), retry in %ds", attempt, max_retries, exc, wait)
                time.sleep(wait)

    # ── multi-leg ───────────────────────────────────────────────────────────
    def submit_multi_leg(
        self,
        legs: list[dict],                # each: {occ_symbol, side ('buy'|'sell'), intent ('open'|'close'), ratio}
        qty: int,
        order_type: str = "limit",
        limit_price: float | None = None,
        tif: str = "day",
    ) -> Any:
        leg_requests: list[OptionLegRequest] = []
        for leg in legs:
            side = leg["side"].lower()
            intent = leg.get("intent", "open").lower()
            intent_enum = {
                ("buy", "open"):  PositionIntent.BUY_TO_OPEN,
                ("buy", "close"): PositionIntent.BUY_TO_CLOSE,
                ("sell", "open"): PositionIntent.SELL_TO_OPEN,
                ("sell", "close"):PositionIntent.SELL_TO_CLOSE,
            }[(side, intent)]
            leg_requests.append(OptionLegRequest(
                symbol=leg["occ_symbol"],
                ratio_qty=int(leg.get("ratio", 1)),
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                position_intent=intent_enum,
            ))

        tif_enum = TimeInForce.DAY if tif == "day" else TimeInForce.GTC
        if order_type == "market":
            req = MarketOrderRequest(
                qty=qty,
                order_class=OrderClass.MLEG,
                legs=leg_requests,
                time_in_force=tif_enum,
            )
        else:
            if limit_price is None:
                raise ValueError("limit_price required for limit multi-leg orders")
            req = LimitOrderRequest(
                qty=qty,
                order_class=OrderClass.MLEG,
                legs=leg_requests,
                time_in_force=tif_enum,
                limit_price=round(float(limit_price), 2),
            )
        order = self.client.submit_order(order_data=req)
        logger.info("MLEG option order submitted id=%s legs=%d qty=%s", order.id, len(leg_requests), qty)
        return order

    # ── high-level: place a StrategyCandidate ───────────────────────────────
    def place_strategy(
        self,
        cand: StrategyCandidate,
        qty: int = 1,
        order_type: str = "limit",
        price_slippage_pct: float = 0.0,
        intent: str = "open",
    ) -> Any:
        """Place an order matching a StrategyCandidate.

        For single-leg: submits a simple order. For multi-leg: submits MLEG.
        Limit price = mid of net debit/credit, adjusted by price_slippage_pct
        (positive = pay more / receive less).
        """
        # Net price per spread = sum(leg.mid * ratio * sign(side))
        # Sign convention for Alpaca MLEG limit: positive = net debit, negative = net credit
        net = 0.0
        for leg in cand.legs:
            sign = 1 if leg.side == "buy" else -1
            net += sign * leg.ratio * leg.mid
        if price_slippage_pct:
            net = net * (1.0 + price_slippage_pct / 100.0) if net > 0 else net * (1.0 - price_slippage_pct / 100.0)

        if len(cand.legs) == 1:
            leg = cand.legs[0]
            return self.submit_single_leg(
                occ_symbol=leg.occ_symbol,
                qty=qty,
                side=leg.side,
                intent=intent,
                order_type=order_type,
                limit_price=abs(net) if order_type == "limit" else None,
            )
        legs_dict = [
            {"occ_symbol": leg.occ_symbol, "side": leg.side, "intent": intent, "ratio": leg.ratio}
            for leg in cand.legs
        ]
        return self.submit_multi_leg(
            legs=legs_dict,
            qty=qty,
            order_type=order_type,
            limit_price=net if order_type == "limit" else None,
        )

    def close_position(self, occ_symbol: str) -> Any:
        """Market-close an option position by OCC symbol."""
        try:
            return self.client.close_position(occ_symbol)
        except Exception as exc:
            logger.warning("Failed to close option position %s: %s", occ_symbol, exc)
            return None
