"""Risk-based position sizing and ATR stop-loss helpers.

`atr_stop_price(entry, atr_value, side, mult)` — stop price `mult` ATRs from entry.
`risk_based_qty(entry, stop, equity, risk_pct, max_shares)` — shares to risk
    `risk_pct` of equity on the move from entry → stop.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ai_trading.strategy.indicators import atr


@dataclass(slots=True)
class StopAndSize:
    qty: int
    stop_price: float
    risk_per_share: float
    atr_value: float


def atr_stop_price(entry: float, atr_value: float, side: str = "BUY", mult: float = 2.0) -> float:
    """Return a stop-loss price `mult` ATRs from entry."""
    if side.upper() == "BUY":
        return max(0.0, entry - mult * atr_value)
    return entry + mult * atr_value


def risk_based_qty(
    entry: float,
    stop: float,
    equity: float,
    risk_pct: float,
    max_shares: int = 1_000_000,
) -> int:
    """Shares such that (entry - stop) * qty == risk_pct/100 * equity.

    Capped by `max_shares` and by available equity (entry * qty ≤ equity).
    """
    risk_dollars = max(0.0, equity * risk_pct / 100.0)
    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0 or risk_dollars <= 0 or entry <= 0:
        return 0
    qty = int(risk_dollars // risk_per_share)
    qty = min(qty, max_shares, int(equity // entry))
    return max(0, qty)


def compute_atr_stop_and_size(
    bars: pd.DataFrame,
    entry: float,
    equity: float,
    *,
    risk_pct: float = 0.5,
    atr_period: int = 14,
    atr_mult: float = 2.0,
    max_shares: int = 1_000_000,
    side: str = "BUY",
) -> StopAndSize:
    """One-shot helper: compute ATR, stop price, and risk-based share count."""
    a = float(atr(bars, atr_period).iloc[-1])
    stop = atr_stop_price(entry, a, side=side, mult=atr_mult)
    qty = risk_based_qty(entry, stop, equity, risk_pct, max_shares)
    return StopAndSize(qty=qty, stop_price=stop, risk_per_share=abs(entry - stop), atr_value=a)


def adaptive_thresholds(
    bars: pd.DataFrame,
    base_buy: float = 0.15,
    base_sell: float = -0.15,
    atr_period: int = 14,
) -> tuple[float, float]:
    """Scale ensemble thresholds by symbol volatility.

    More volatile symbols need a stronger signal to act on (reduces noise);
    quieter symbols can act on smaller signals.

    The scale factor is `vol_ratio = atr_pct / 0.02` clipped to [0.5, 2.0],
    so a symbol with 2% ATR keeps base thresholds, 4% ATR doubles them, etc.
    """
    a = float(atr(bars, atr_period).iloc[-1])
    c = float(bars["close"].iloc[-1])
    if c <= 0:
        return base_buy, base_sell
    vol_ratio = max(0.5, min(2.0, (a / c) / 0.02))
    return base_buy * vol_ratio, base_sell * vol_ratio
