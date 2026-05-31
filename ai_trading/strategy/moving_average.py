from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class SignalResult:
    signal: str
    close: float
    fast_ma: float
    slow_ma: float


def moving_average_signal(bars: pd.DataFrame, fast: int, slow: int) -> SignalResult:
    closes = bars["close"].astype(float)
    fast_ma = closes.rolling(fast).mean()
    slow_ma = closes.rolling(slow).mean()

    close = float(closes.iloc[-1])
    latest_fast = float(fast_ma.iloc[-1]) if pd.notna(fast_ma.iloc[-1]) else float("nan")
    latest_slow = float(slow_ma.iloc[-1]) if pd.notna(slow_ma.iloc[-1]) else float("nan")

    if pd.isna(latest_fast) or pd.isna(latest_slow):
        return SignalResult("HOLD", close, latest_fast, latest_slow)
    if latest_fast > latest_slow:
        return SignalResult("BUY", close, latest_fast, latest_slow)
    if latest_fast < latest_slow:
        return SignalResult("SELL", close, latest_fast, latest_slow)
    return SignalResult("HOLD", close, latest_fast, latest_slow)
