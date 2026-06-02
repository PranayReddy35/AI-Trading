from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

logger = logging.getLogger("ai_trading")


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
        kelly_win_rate: float = 0.52,
        kelly_avg_win: float = 0.015,
        kelly_avg_loss: float = 0.01,
        trailing_stop_pct: float = 0.0,
        error_streak_decay_hours: float = 0.0,
        partial_profit_max_hold_bars: int = 0,
        partial_profit_trailing_stop_pct: float = 0.0,
        state_file: str = "",
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
        self.kelly_win_rate = kelly_win_rate
        self.kelly_avg_win = kelly_avg_win
        self.kelly_avg_loss = kelly_avg_loss
        self.trailing_stop_pct = trailing_stop_pct
        self.error_streak_decay_hours = error_streak_decay_hours
        self.partial_profit_max_hold_bars = partial_profit_max_hold_bars
        self.partial_profit_trailing_stop_pct = partial_profit_trailing_stop_pct
        self._state_file = state_file

        self._day: date | None = None
        self._trades_today = 0
        self._consecutive_errors = 0
        self._last_trade_ts: float = 0.0
        self._last_error_ts: float = 0.0
        # Portfolio peak equity for drawdown halt
        self._peak_equity: float = 0.0
        # Start-of-day equity for accurate daily loss calculation
        self._start_of_day_equity: float = 0.0
        # Trailing stop tracking: {symbol: peak_price_since_entry}
        self._trailing_peaks: dict[str, float] = {}
        # Partial profit tracking: symbols where half was already sold
        self._partial_profit_taken: set[str] = set()
        # Track bars since partial profit for exit conditions
        self._partial_profit_bar_count: dict[str, int] = {}
        # Track peak since partial profit for trailing stop on remainder
        self._partial_profit_peaks: dict[str, float] = {}

        # Restore persisted state if available
        self._load_state()

    # --- State persistence (#5) ---
    def _state_path(self) -> Path | None:
        if not self._state_file:
            return None
        return Path(self._state_file)

    def _load_state(self) -> None:
        """Load persisted risk state from disk."""
        path = self._state_path()
        if path is None or not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            saved_day = data.get("day")
            if saved_day:
                saved_date = date.fromisoformat(saved_day)
                today = date.today()
                if saved_date == today:
                    self._day = saved_date
                    self._trades_today = data.get("trades_today", 0)
                    self._consecutive_errors = data.get("consecutive_errors", 0)
                    self._last_error_ts = data.get("last_error_ts", 0.0)
            self._peak_equity = data.get("peak_equity", 0.0)
            self._start_of_day_equity = data.get("start_of_day_equity", 0.0)
            self._trailing_peaks = data.get("trailing_peaks", {})
            self._partial_profit_taken = set(data.get("partial_profit_taken", []))
            self._partial_profit_bar_count = data.get("partial_profit_bar_count", {})
            self._partial_profit_peaks = data.get("partial_profit_peaks", {})
            logger.info("Restored risk manager state from %s", path)
        except Exception as exc:
            logger.warning("Failed to load risk state from %s: %s", path, exc)

    def save_state(self) -> None:
        """Persist current risk state to disk for crash recovery."""
        path = self._state_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "day": self._day.isoformat() if self._day else None,
                "trades_today": self._trades_today,
                "consecutive_errors": self._consecutive_errors,
                "last_error_ts": self._last_error_ts,
                "peak_equity": self._peak_equity,
                "start_of_day_equity": self._start_of_day_equity,
                "trailing_peaks": self._trailing_peaks,
                "partial_profit_taken": list(self._partial_profit_taken),
                "partial_profit_bar_count": self._partial_profit_bar_count,
                "partial_profit_peaks": self._partial_profit_peaks,
            }
            path.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            logger.warning("Failed to save risk state to %s: %s", path, exc)

    def _reset_for_day(self, today: date) -> None:
        if self._day != today:
            self._day = today
            self._trades_today = 0

    def set_start_of_day_equity(self, equity: float) -> None:
        """Record start-of-day equity for accurate daily loss tracking."""
        if self._start_of_day_equity <= 0:
            self._start_of_day_equity = equity

    def register_trade(self, today: date) -> None:
        self._reset_for_day(today)
        self._trades_today += 1
        self._last_trade_ts = time.time()
        self.save_state()

    def register_error(self) -> None:
        self._consecutive_errors += 1
        self._last_error_ts = time.time()
        self.save_state()

    def clear_error_streak(self) -> None:
        self._consecutive_errors = 0
        self.save_state()

    def update_trailing_peak(self, symbol: str, current_price: float) -> None:
        """Update the peak price for trailing stop tracking."""
        prev = self._trailing_peaks.get(symbol, 0.0)
        self._trailing_peaks[symbol] = max(prev, current_price)

    def clear_trailing_peak(self, symbol: str) -> None:
        """Clear trailing peak after a position is closed."""
        self._trailing_peaks.pop(symbol, None)
        self.save_state()

    # --- Partial profit helpers ---
    def has_partial_profit_taken(self, symbol: str) -> bool:
        """True if the 50% partial profit sell has already fired for this symbol."""
        return symbol in self._partial_profit_taken

    def mark_partial_profit_taken(self, symbol: str, entry_price: float = 0.0) -> None:
        """Record that partial profit was taken; track remainder for exit rules."""
        self._partial_profit_taken.add(symbol)
        self._partial_profit_bar_count[symbol] = 0
        if entry_price > 0:
            self._partial_profit_peaks[symbol] = entry_price
        self.save_state()

    def clear_partial_profit(self, symbol: str) -> None:
        """Reset when the full position is eventually closed."""
        self._partial_profit_taken.discard(symbol)
        self._partial_profit_bar_count.pop(symbol, None)
        self._partial_profit_peaks.pop(symbol, None)
        self.save_state()

    def increment_partial_profit_bars(self, symbol: str) -> None:
        """Increment the bar counter for partial-profit remainder positions."""
        if symbol in self._partial_profit_bar_count:
            self._partial_profit_bar_count[symbol] += 1

    def update_partial_profit_peak(self, symbol: str, current_price: float) -> None:
        """Update peak price for partial profit trailing stop."""
        prev = self._partial_profit_peaks.get(symbol, 0.0)
        self._partial_profit_peaks[symbol] = max(prev, current_price)

    def should_exit_partial_remainder(self, symbol: str, current_price: float) -> tuple[bool, str]:
        """Check if the remaining shares after partial profit should be sold.

        Exit conditions:
        1. Time decay: held more than max_hold_bars since partial profit
        2. Trailing stop: price dropped too far from peak since partial profit

        Returns:
            (should_exit, reason)
        """
        if symbol not in self._partial_profit_taken:
            return False, ""

        # Time-based exit
        if self.partial_profit_max_hold_bars > 0:
            bars_held = self._partial_profit_bar_count.get(symbol, 0)
            if bars_held >= self.partial_profit_max_hold_bars:
                return True, f"partial remainder time exit: {bars_held} bars >= {self.partial_profit_max_hold_bars}"

        # Trailing stop exit for remainder
        if self.partial_profit_trailing_stop_pct > 0:
            peak = self._partial_profit_peaks.get(symbol, 0.0)
            if peak > 0 and current_price > 0:
                drop_pct = (peak - current_price) / peak * 100.0
                if drop_pct >= self.partial_profit_trailing_stop_pct:
                    return True, (
                        f"partial remainder trailing stop: dropped {drop_pct:.1f}% from peak "
                        f"(limit {self.partial_profit_trailing_stop_pct:.1f}%)"
                    )

        return False, ""

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

        # Time-based error streak decay (#18): reset errors after N hours of no errors
        if self.error_streak_decay_hours > 0 and self._consecutive_errors > 0:
            elapsed_hours = (time.time() - self._last_error_ts) / 3600.0
            if elapsed_hours >= self.error_streak_decay_hours:
                logger.info(
                    "Error streak decayed: %d errors cleared after %.1f hours",
                    self._consecutive_errors, elapsed_hours,
                )
                self._consecutive_errors = 0

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
