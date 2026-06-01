from __future__ import annotations

import time
from dataclasses import dataclass, field
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
        portfolio_drawdown_halt_pct: float = 0.0,
        use_kelly_sizing: bool = False,
        kelly_fraction: float = 0.5,
        kelly_max_shares: int = 10,
        trailing_stop_pct: float = 0.0,
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
        self.portfolio_drawdown_halt_pct = portfolio_drawdown_halt_pct
        self.use_kelly_sizing = use_kelly_sizing
        self.kelly_fraction = kelly_fraction
        self.kelly_max_shares = kelly_max_shares
        self.trailing_stop_pct = trailing_stop_pct

        self._day: date | None = None
        self._trades_today = 0
        self._consecutive_errors = 0
        self._last_trade_ts: float = 0.0
        # Portfolio peak equity for drawdown halt
        self._peak_equity: float = 0.0
        # Trailing stop tracking: {symbol: peak_price_since_entry}
        self._trailing_peaks: dict[str, float] = {}
        # Partial profit tracking: symbols where half was already sold (rest held forever)
        self._partial_profit_taken: set[str] = set()

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

    def update_trailing_peak(self, symbol: str, current_price: float) -> None:
        """Update the peak price for trailing stop tracking."""
        prev = self._trailing_peaks.get(symbol, 0.0)
        self._trailing_peaks[symbol] = max(prev, current_price)

    def clear_trailing_peak(self, symbol: str) -> None:
        """Clear trailing peak after a position is closed."""
        self._trailing_peaks.pop(symbol, None)

    # --- Partial profit helpers ---
    def has_partial_profit_taken(self, symbol: str) -> bool:
        """True if the 50% partial profit sell has already fired for this symbol."""
        return symbol in self._partial_profit_taken

    def mark_partial_profit_taken(self, symbol: str) -> None:
        """Record that partial profit was taken; remaining shares are held forever."""
        self._partial_profit_taken.add(symbol)

    def clear_partial_profit(self, symbol: str) -> None:
        """Reset when the full position is eventually closed."""
        self._partial_profit_taken.discard(symbol)

    def should_trail_stop(self, symbol: str, current_price: float) -> bool:
        """Return True if trailing stop is breached for this symbol."""
        if self.trailing_stop_pct <= 0:
            return False
        peak = self._trailing_peaks.get(symbol, 0.0)
        if peak <= 0:
            return False
        drop_pct = (peak - current_price) / peak * 100.0
        return drop_pct >= self.trailing_stop_pct

    def kelly_qty(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        equity: float,
        price: float,
    ) -> int:
        """Compute position size using fractional Kelly criterion.

        Args:
            win_rate: Historical win rate (0-1).
            avg_win: Average winning trade return (positive).
            avg_loss: Average losing trade return (positive value, e.g., 0.02 = 2%).
            equity: Current portfolio equity.
            price: Current share price.

        Returns:
            Number of shares to buy.
        """
        if avg_loss <= 0 or price <= 0 or equity <= 0:
            return 1
        b = avg_win / avg_loss  # win/loss ratio
        q = 1.0 - win_rate
        kelly_f = (win_rate * b - q) / b  # Kelly formula
        kelly_f = max(0.0, kelly_f) * self.kelly_fraction
        dollar_amount = equity * kelly_f
        shares = int(dollar_amount / price)
        return max(1, min(shares, self.kelly_max_shares))

    def is_gap_open_too_large(
        self,
        symbol: str,
        current_price: float,
        prior_close: float,
        gap_open_protection_pct: float,
    ) -> tuple[bool, str]:
        """Return (True, reason) if a gap-open is larger than the protection threshold.

        A gap is calculated as abs((current - prior_close) / prior_close) * 100.
        If the gap is adverse (against position) or simply too large, we should skip/exit.
        """
        if gap_open_protection_pct <= 0 or prior_close <= 0 or current_price <= 0:
            return False, ""
        gap_pct = abs(current_price - prior_close) / prior_close * 100.0
        if gap_pct >= gap_open_protection_pct:
            return True, f"gap-open {gap_pct:.1f}% >= threshold {gap_open_protection_pct:.1f}%"
        return False, ""

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
        if self.min_equity > 0 and 0 < equity < self.min_equity:
            return RiskResult(
                False,
                f"equity ${equity:.2f} below minimum ${self.min_equity:.2f}",
            )

        # Portfolio peak drawdown halt
        if self.portfolio_drawdown_halt_pct > 0 and equity > 0:
            if equity > self._peak_equity:
                self._peak_equity = equity
            if self._peak_equity > 0:
                drawdown_pct = (self._peak_equity - equity) / self._peak_equity * 100.0
                if drawdown_pct >= self.portfolio_drawdown_halt_pct:
                    return RiskResult(
                        False,
                        f"portfolio drawdown halt: {drawdown_pct:.2f}% from peak "
                        f"(limit {self.portfolio_drawdown_halt_pct:.1f}%)",
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
        if self.min_equity > 0 and 0 < equity < self.min_equity:
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
