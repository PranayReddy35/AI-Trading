"""Correlation-aware size scaling.

Instead of a hard block (see `is_too_correlated`), this returns a multiplier
in [0, 1] to scale position size down when a new position would add too much
correlated exposure.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ai_trading.risk.correlation import compute_pairwise_correlations

logger = logging.getLogger(__name__)


def correlation_scale(
    new_symbol: str,
    existing_symbols: list[str],
    bars_by_symbol: dict[str, pd.DataFrame],
    soft_threshold: float = 0.6,
    hard_threshold: float = 0.9,
    lookback: int = 60,
) -> float:
    """Return a size multiplier in [0, 1] based on max correlation with existing positions.

    - If max |corr| ≤ soft_threshold → 1.0 (no scaling).
    - If max |corr| ≥ hard_threshold → 0.0 (effective block).
    - Linear ramp between thresholds.
    """
    if not existing_symbols:
        return 1.0
    relevant = {s: bars_by_symbol[s] for s in [new_symbol] + existing_symbols if s in bars_by_symbol}
    if len(relevant) < 2:
        return 1.0
    corr = compute_pairwise_correlations(relevant, lookback=lookback)
    if corr.empty or new_symbol not in corr.columns:
        return 1.0
    max_c = 0.0
    for existing in existing_symbols:
        if existing in corr.columns:
            c = corr.loc[new_symbol, existing]
            if not np.isnan(c):
                max_c = max(max_c, abs(float(c)))
    if max_c <= soft_threshold:
        return 1.0
    if max_c >= hard_threshold:
        return 0.0
    return float(1.0 - (max_c - soft_threshold) / (hard_threshold - soft_threshold))
