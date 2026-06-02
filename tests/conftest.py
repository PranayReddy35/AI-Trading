"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_bars(closes: np.ndarray, base_vol: int = 1_000_000) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame from a close-price array."""
    n = len(closes)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.002, n)
    opens = closes * (1 - noise)
    highs = np.maximum(opens, closes) * (1 + np.abs(noise) + 0.001)
    lows = np.minimum(opens, closes) * (1 - np.abs(noise) - 0.001)
    vol = (rng.integers(80, 120, n) * base_vol // 100).astype(int)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vol},
        index=idx,
    )


@pytest.fixture
def uptrend_bars() -> pd.DataFrame:
    closes = np.linspace(100, 150, 120) + np.random.default_rng(1).normal(0, 0.5, 120)
    return _make_bars(closes)


@pytest.fixture
def downtrend_bars() -> pd.DataFrame:
    closes = np.linspace(150, 100, 120) + np.random.default_rng(2).normal(0, 0.5, 120)
    return _make_bars(closes)


@pytest.fixture
def sideways_bars() -> pd.DataFrame:
    closes = 100 + np.sin(np.linspace(0, 8 * np.pi, 120)) * 3 + np.random.default_rng(3).normal(0, 0.3, 120)
    return _make_bars(closes)


@pytest.fixture
def double_top_bars() -> pd.DataFrame:
    """Synthetic double-top: peak, dip, peak, then break below the intermediate trough."""
    rng = np.random.default_rng(7)
    closes = np.concatenate([
        np.linspace(100, 130, 30),   # rise to peak 1
        np.linspace(130, 115, 15),   # dip
        np.linspace(115, 130, 15),   # rise to peak 2
        np.linspace(130, 108, 25),   # break below dip
    ])
    closes += rng.normal(0, 0.3, len(closes))
    return _make_bars(closes)


@pytest.fixture
def double_bottom_bars() -> pd.DataFrame:
    rng = np.random.default_rng(8)
    closes = np.concatenate([
        np.linspace(150, 110, 30),
        np.linspace(110, 125, 15),
        np.linspace(125, 110, 15),
        np.linspace(110, 135, 25),
    ])
    closes += rng.normal(0, 0.3, len(closes))
    return _make_bars(closes)
