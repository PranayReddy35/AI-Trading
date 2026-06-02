"""Multi-timeframe (MTF) confirmation helper.

`mtf_signal()` fetches bars at multiple timeframes via the supplied
data accessor, computes the ensemble signal on each, and returns a combined
StrategySignal that fires only when timeframes agree.

The data accessor is any callable `get_bars(symbol, lookback_days, timeframe)`
returning a DataFrame (e.g. `AlpacaMarketData.get_bars`).
"""

from __future__ import annotations

import logging
from typing import Callable

import pandas as pd

from ai_trading.strategy.types import StrategySignal

logger = logging.getLogger(__name__)


# Higher-timeframe weights: longer timeframes get more authority.
_DEFAULT_TF_WEIGHTS: dict[str, float] = {
    "1Day": 1.0,
    "1Hour": 0.6,
    "30Min": 0.4,
    "15Min": 0.3,
    "5Min": 0.2,
}


def mtf_signal(
    symbol: str,
    get_bars: Callable[[str, int, str], pd.DataFrame],
    timeframes: list[str] | tuple[str, ...] = ("1Day", "1Hour"),
    lookback_days: int = 120,
) -> StrategySignal:
    """Combine ensemble signals across multiple timeframes.

    Returns a StrategySignal whose `signal` is the weighted sign-average of
    the per-timeframe ensemble strengths, and whose `confidence` reflects
    the fraction of timeframes that agreed.
    """
    # Local import to avoid cycle with strategy.ensemble at import time.
    from ai_trading.strategy.ensemble import compute_ensemble_signal

    contributions: list[tuple[str, float, float, str]] = []  # (tf, signed, weight, signal_label)
    for tf in timeframes:
        try:
            bars = get_bars(symbol, lookback_days, tf)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MTF: failed to fetch %s %s: %s", symbol, tf, exc)
            continue
        if bars is None or len(bars) < 30:
            continue
        es = compute_ensemble_signal(bars)
        w = _DEFAULT_TF_WEIGHTS.get(tf, 0.3)
        contributions.append((tf, es.strength, w, es.signal))

    if not contributions:
        return StrategySignal("mtf", 0.0, 0.0, "no timeframes available")

    total_w = sum(w for _, _, w, _ in contributions)
    combined = sum(s * w for _, s, w, _ in contributions) / total_w if total_w > 0 else 0.0

    agree_buy = sum(1 for _, _, _, lab in contributions if lab == "BUY")
    agree_sell = sum(1 for _, _, _, lab in contributions if lab == "SELL")
    agree = max(agree_buy, agree_sell)
    confidence = agree / len(contributions)

    reason = ", ".join(f"{tf}:{lab}({s:+.2f})" for tf, s, _, lab in contributions)
    return StrategySignal("mtf", float(combined), float(confidence), reason)
