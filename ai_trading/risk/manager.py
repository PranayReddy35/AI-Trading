from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(slots=True)
class RiskResult:
    allowed: bool
    reason: str
    approved_qty: int = 0


class RiskManager:
    def __init__(
        self,
        *,
        paper_only: bool,
        min_cash_threshold: float,
        max_shares: int,
        max_daily_trades: int,
        max_consecutive_errors: int,
    ) -> None:
        self.paper_only = paper_only
        self.min_cash_threshold = min_cash_threshold
        self.max_shares = max_shares
        self.max_daily_trades = max_daily_trades
        self.max_consecutive_errors = max_consecutive_errors

        self._day: date | None = None
        self._trades_today = 0
        self._consecutive_errors = 0

    def _reset_for_day(self, today: date) -> None:
        if self._day != today:
            self._day = today
            self._trades_today = 0

    def register_trade(self, today: date) -> None:
        self._reset_for_day(today)
        self._trades_today += 1

    def register_error(self) -> None:
        self._consecutive_errors += 1

    def clear_error_streak(self) -> None:
        self._consecutive_errors = 0

    def evaluate(
        self,
        *,
        today: date,
        paper_mode: bool,
        market_open: bool,
        cash: float,
        has_open_order: bool,
        side: str,
        requested_qty: int,
        current_position_qty: int,
    ) -> RiskResult:
        self._reset_for_day(today)

        if self.paper_only and not paper_mode:
            return RiskResult(False, "paper-trading-only safety guard blocked non-paper mode")

        if self._consecutive_errors >= self.max_consecutive_errors:
            return RiskResult(False, "consecutive error stop triggered")

        if not market_open:
            return RiskResult(False, "market-closed guard")

        if has_open_order:
            return RiskResult(False, "duplicate/open order prevention")

        if self._trades_today >= self.max_daily_trades:
            return RiskResult(False, "max daily trades reached")

        normalized_side = side.upper()
        if normalized_side not in {"BUY", "SELL"}:
            return RiskResult(False, "unsupported side")

        qty = max(0, min(requested_qty, self.max_shares))
        if qty <= 0:
            return RiskResult(False, "max shares/position sizing rejected quantity")

        if normalized_side == "BUY":
            if cash < self.min_cash_threshold:
                return RiskResult(False, "cash threshold guard")
        else:
            if current_position_qty <= 0:
                return RiskResult(False, "no position to sell (no shorting)")
            qty = min(qty, current_position_qty)
            if qty <= 0:
                return RiskResult(False, "sell quantity reduced to zero")

        return RiskResult(True, "approved", approved_qty=qty)
