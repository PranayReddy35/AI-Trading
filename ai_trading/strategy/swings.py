"""Swing/pivot detection using scipy.signal.find_peaks.

Returns indices and prices of confirmed swing highs and lows with prominence
filtering — far less noisy than 3-bar pivots.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


@dataclass(slots=True)
class Swings:
    high_idx: np.ndarray  # positions in the input series
    high_prices: np.ndarray
    low_idx: np.ndarray
    low_prices: np.ndarray


def detect_swings(close: pd.Series, prominence_pct: float = 0.02, distance: int = 5) -> Swings:
    """Detect swing highs and lows.

    Args:
        close: price series.
        prominence_pct: required peak prominence as a fraction of mean price
            (0.02 = 2 %). Higher → fewer, more significant swings.
        distance: minimum bars between swings.
    """
    arr = close.astype(float).values
    if len(arr) < distance * 2 + 1:
        empty = np.array([], dtype=int)
        return Swings(empty, np.array([]), empty, np.array([]))

    mean_price = float(np.mean(arr))
    prom = max(mean_price * prominence_pct, 1e-6)

    high_idx, _ = find_peaks(arr, prominence=prom, distance=distance)
    low_idx, _ = find_peaks(-arr, prominence=prom, distance=distance)

    return Swings(
        high_idx=high_idx,
        high_prices=arr[high_idx],
        low_idx=low_idx,
        low_prices=arr[low_idx],
    )


def last_swing_high_low(close: pd.Series, prominence_pct: float = 0.02) -> tuple[int, float, int, float] | None:
    """Return (hi_idx, hi_price, lo_idx, lo_price) of the most-recent swing pair, or None."""
    s = detect_swings(close, prominence_pct=prominence_pct)
    if len(s.high_idx) == 0 or len(s.low_idx) == 0:
        return None
    hi_idx = int(s.high_idx[-1])
    lo_idx = int(s.low_idx[-1])
    return hi_idx, float(close.iloc[hi_idx]), lo_idx, float(close.iloc[lo_idx])
