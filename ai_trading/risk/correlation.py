"""Correlation filter: block new positions that are highly correlated with existing ones.

High correlation between positions means they will likely move together,
reducing diversification and increasing portfolio risk.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("ai_trading")


def compute_pairwise_correlations(
    bars_by_symbol: dict[str, pd.DataFrame],
    lookback: int = 60,
) -> pd.DataFrame:
    """Compute pairwise return correlations between symbols.

    Args:
        bars_by_symbol: Dict of {symbol: OHLCV DataFrame}.
        lookback: Number of recent bars to use for correlation.

    Returns:
        Correlation matrix as a DataFrame.
    """
    returns: dict[str, pd.Series] = {}
    for sym, bars in bars_by_symbol.items():
        close = bars["close"].astype(float).tail(lookback)
        returns[sym] = close.pct_change().dropna()

    if not returns:
        return pd.DataFrame()

    return_df = pd.DataFrame(returns).dropna()
    return return_df.corr()


def is_too_correlated(
    new_symbol: str,
    existing_symbols: list[str],
    bars_by_symbol: dict[str, pd.DataFrame],
    threshold: float = 0.85,
    lookback: int = 60,
) -> tuple[bool, str]:
    """Check if new_symbol is too correlated with any existing position.

    Args:
        new_symbol: Symbol being considered for a new position.
        existing_symbols: Symbols of currently held positions.
        bars_by_symbol: Price data for all symbols.
        threshold: Correlation threshold above which to block (e.g., 0.85).
        lookback: Bars to use for correlation calculation.

    Returns:
        (blocked, reason) — blocked=True means the trade should be skipped.
    """
    if threshold <= 0 or not existing_symbols:
        return False, ""

    relevant = {s: bars_by_symbol[s] for s in [new_symbol] + existing_symbols if s in bars_by_symbol}
    if len(relevant) < 2:
        return False, ""

    corr_matrix = compute_pairwise_correlations(relevant, lookback=lookback)
    if corr_matrix.empty or new_symbol not in corr_matrix.columns:
        return False, ""

    for existing in existing_symbols:
        if existing not in corr_matrix.columns:
            continue
        corr = corr_matrix.loc[new_symbol, existing]
        if not np.isnan(corr) and corr >= threshold:
            # Only block positive correlations — negative correlations are
            # hedges and actually improve portfolio diversification.
            reason = (
                f"{new_symbol} correlation with {existing} is {corr:.2f} "
                f"(threshold {threshold:.2f})"
            )
            logger.info("Correlation filter blocked %s: %s", new_symbol, reason)
            return True, reason

    return False, ""
