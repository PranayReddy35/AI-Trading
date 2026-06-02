"""Exit-logic primitives: trailing ATR stop, time stop, R-multiple targets.

These compute desired exit *prices* / *decisions* given the current bar and
the per-position state. The bot is responsible for actually submitting orders.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ai_trading.strategy.indicators import atr


@dataclass(slots=True)
class TrailState:
    """Per-position trailing state. Persist on RiskManager / Journal."""
    entry: float
    peak: float           # highest close since entry (for longs)
    initial_stop: float   # 1R stop set at entry
    bars_held: int = 0
    partial_taken: bool = False


def trail_atr_stop(
    bars: pd.DataFrame,
    state: TrailState,
    *,
    side: str = "long",
    atr_period: int = 14,
    atr_mult: float = 2.0,
) -> float:
    """Return the current trailing stop price.

    For longs: stop = max(initial_stop, peak - atr_mult * ATR)
    For shorts: stop = min(initial_stop, trough + atr_mult * ATR)
    """
    a = atr(bars, period=atr_period)
    if a.empty:
        return state.initial_stop
    a_val = float(a.iloc[-1])
    if side == "long":
        trailing = state.peak - atr_mult * a_val
        return max(state.initial_stop, trailing)
    trailing = state.peak + atr_mult * a_val
    return min(state.initial_stop, trailing)


def should_time_stop(state: TrailState, *, max_bars: int, min_progress_r: float = 0.5,
                     current_price: float = 0.0) -> tuple[bool, str]:
    """Exit if position has been held >= max_bars without reaching min_progress_r.

    Useful to free capital from going-nowhere trades.
    """
    if max_bars <= 0 or state.bars_held < max_bars:
        return False, f"bars_held={state.bars_held} < max_bars={max_bars}"
    r = (current_price - state.entry) / max(1e-9, state.entry - state.initial_stop)
    if r >= min_progress_r:
        return False, f"R={r:.2f} >= {min_progress_r}, keep holding"
    return True, f"time stop: held {state.bars_held} bars, R={r:.2f} < {min_progress_r}"


def r_multiple(state: TrailState, price: float) -> float:
    """Current unrealized P/L in R-multiples (long-only convention)."""
    risk_per_share = state.entry - state.initial_stop
    if risk_per_share <= 0:
        return 0.0
    return (price - state.entry) / risk_per_share


def should_partial_take(state: TrailState, price: float, *,
                        trigger_r: float = 1.0) -> bool:
    """Take partial profit when up >= trigger_r R-multiples and not yet taken."""
    if state.partial_taken:
        return False
    return r_multiple(state, price) >= trigger_r


def breakeven_stop(state: TrailState, price: float, *,
                   trigger_r: float = 1.0) -> float | None:
    """Once up trigger_r, move the stop to entry (free trade)."""
    if r_multiple(state, price) >= trigger_r and state.initial_stop < state.entry:
        return state.entry
    return None
