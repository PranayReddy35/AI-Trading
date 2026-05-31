from __future__ import annotations

import time
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
        daily_loss_limit_pct: float = 0.0,
        max_portfolio_exposure_pct: float = 95.0,
        min_equity: float = 0.0,
        trade_cooldown_sec: int = 0,
    ) -> None:
        self.paper_only = paper_only
        self.min_cash_threshold = min_cash_threshold
        self.max_shares = max_shares
        self.max_daily_trades = max_daily_trades
        self.max_consecutive_errors = max_consecutive_errors
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_portfolio_exposure_pct = max_portfolio_exposure_pct
        self.min_equity = min_equity
        self.trade_cooldown_sec = trade_cooldown_sec

        self._day: date | None = None
        self._trades_today = 0
        self._consecutive_errors = 0
        self._last_trade_ts: float = 0.0

    def _reset_for_day(self, today: date) -> None:
        if self._day != today:
            self._day = today
            self._trades_today = 0

    def register_trade(self, today: date) -> None:
        self._reset_for_day(today)
        self._trades_today += 1
        self._last_trade_ts = time.time()

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
        equity: float = 0.0,
        last_equity: float = 0.0,
        portfolio_value: float = 0.0,
    ) -> RiskResult:
        self._reset_for_day(today)

        # Paper-only guard: if configured as paper_only, block live trading
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

        # Trade cooldown check
        if self.trade_cooldown_sec > 0:
            elapsed = time.time() - self._last_trade_ts
            if self._last_trade_ts > 0 and elapsed < self.trade_cooldown_sec:
                remaining = int(self.trade_cooldown_sec - elapsed)
                return RiskResult(False, f"trade cooldown active ({remaining}s remaining)")

        # Minimum equity guard
        if self.min_equity > 0 and equity > 0 and equity < self.min_equity:
            return RiskResult(
                False,
                f"equity ${equity:.2f} below minimum ${self.min_equity:.2f}",
            )

        # Daily loss limit check (compares current equity vs start-of-day equity)
        if self.daily_loss_limit_pct > 0 and last_equity > 0 and equity > 0:
            daily_change_pct = ((equity - last_equity) / last_equity) * 100.0
            if daily_change_pct <= -self.daily_loss_limit_pct:
                return RiskResult(
                    False,
                    f"daily loss limit reached ({daily_change_pct:.2f}% vs limit -{self.daily_loss_limit_pct:.1f}%)",
                )

        normalized_side = side.upper()
        if normalized_side not in {"BUY", "SELL"}:
            return RiskResult(False, "unsupported side")

        qty = max(0, min(requested_qty, self.max_shares))
        if qty <= 0:
            return RiskResult(False, "max shares/position sizing rejected quantity")

        if normalized_side == "BUY":
            if cash < self.min_cash_threshold:
                return RiskResult(False, "cash threshold guard")

            # Max portfolio exposure check
            if self.max_portfolio_exposure_pct < 100 and equity > 0:
                max_invested = equity * (self.max_portfolio_exposure_pct / 100.0)
                currently_invested = equity - cash
                remaining_capacity = max_invested - currently_invested
                if remaining_capacity <= 0:
                    return RiskResult(
                        False,
                        f"max portfolio exposure {self.max_portfolio_exposure_pct}% reached",
                    )
        else:
            if current_position_qty <= 0:
                return RiskResult(False, "no position to sell (no shorting)")
            qty = min(qty, current_position_qty)
            if qty <= 0:
                return RiskResult(False, "sell quantity reduced to zero")

        return RiskResult(True, "approved", approved_qty=qty)
