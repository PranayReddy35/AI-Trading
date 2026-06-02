"""Portfolio-level sizing helpers: volatility targeting and portfolio heat cap.

Volatility targeting: each position contributes ~target_vol_pct of portfolio
vol. Uses 20-day realised daily vol (std of pct returns) as the estimate.

Portfolio heat: total $-risk across open positions (sum of qty * stop-distance)
must stay below max_heat_pct * equity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


def realised_daily_vol(bars: pd.DataFrame, lookback: int = 20) -> float:
    """Std of daily pct returns over `lookback` bars. 0 if not enough data."""
    if "close" not in bars or len(bars) < lookback + 1:
        return 0.0
    rets = bars["close"].pct_change().dropna().iloc[-lookback:]
    if rets.empty:
        return 0.0
    return float(rets.std())


def vol_targeted_qty(
    *,
    bars: pd.DataFrame,
    entry: float,
    equity: float,
    target_vol_pct: float = 1.0,
    max_position_pct: float = 20.0,
    max_shares: int = 100,
    lookback: int = 20,
) -> int:
    """Size so the position's 1-day expected vol ≈ target_vol_pct of equity.

    qty * entry * sigma ≈ target_vol_pct/100 * equity
    """
    if entry <= 0 or equity <= 0:
        return 0
    sigma = realised_daily_vol(bars, lookback)
    if sigma <= 0:
        return 0
    target_dollar_vol = (target_vol_pct / 100.0) * equity
    notional = target_dollar_vol / sigma
    qty = int(notional / entry)
    # Cap at max single-position notional
    max_notional = (max_position_pct / 100.0) * equity
    qty = min(qty, int(max_notional / entry))
    return max(0, min(qty, max_shares))


@dataclass(slots=True)
class HeatCheck:
    allowed: bool
    current_heat_pct: float
    projected_heat_pct: float
    reason: str


def portfolio_heat_check(
    *,
    open_risks: dict[str, float],   # {symbol: dollar_risk_per_position}
    new_symbol: str,
    new_dollar_risk: float,
    equity: float,
    max_heat_pct: float = 6.0,
) -> HeatCheck:
    """Return whether opening a new position keeps total $-risk under cap."""
    if equity <= 0:
        return HeatCheck(False, 0.0, 0.0, "non-positive equity")
    # Replace any existing risk for the same symbol (re-entry)
    others = sum(v for sym, v in open_risks.items() if sym != new_symbol)
    current = others
    projected = others + max(0.0, new_dollar_risk)
    cur_pct = current / equity * 100.0
    proj_pct = projected / equity * 100.0
    if proj_pct > max_heat_pct:
        return HeatCheck(
            False, cur_pct, proj_pct,
            f"projected heat {proj_pct:.2f}% > cap {max_heat_pct:.1f}% "
            f"(current {cur_pct:.2f}%, new ${new_dollar_risk:.0f})",
        )
    return HeatCheck(True, cur_pct, proj_pct,
                     f"heat OK: {cur_pct:.2f}%→{proj_pct:.2f}% (cap {max_heat_pct:.1f}%)")


def fractional_kelly_qty(
    *,
    win_rate: float,
    avg_win_r: float,    # avg win in R-multiples (e.g. 1.5)
    avg_loss_r: float,   # avg loss in R-multiples (typically 1.0 since stop = 1R)
    equity: float,
    entry: float,
    risk_per_share: float,
    fraction: float = 0.25,
    max_shares: int = 100,
) -> int:
    """Kelly sizing expressed in R-multiples.

    f* = (p*b - q) / b, where b = avg_win_r / avg_loss_r.
    Use only `fraction` of f* (default ¼ Kelly).
    """
    if risk_per_share <= 0 or entry <= 0 or equity <= 0:
        return 0
    if avg_loss_r <= 0:
        return 0
    b = avg_win_r / avg_loss_r
    q = 1.0 - win_rate
    f_star = (win_rate * b - q) / b
    f_star = max(0.0, f_star) * fraction
    # Risk-based: dollars_at_risk = f_star * equity, qty = that / risk_per_share
    qty = int((f_star * equity) / risk_per_share)
    # Sanity cap by notional
    max_notional = equity * 0.30
    qty = min(qty, int(max_notional / entry))
    return max(0, min(qty, max_shares))
