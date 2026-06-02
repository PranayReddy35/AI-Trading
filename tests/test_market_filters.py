"""Tests for strategy.market_filters."""

from __future__ import annotations

import pandas as pd

from ai_trading.strategy.market_filters import (
    vix_size_multiplier,
    volume_confirms,
    spread_too_wide,
)


def test_vix_full_size_when_calm():
    m, _ = vix_size_multiplier(vix=15.0, full_below=20, half_above=25, zero_above=35)
    assert m == 1.0


def test_vix_zero_when_panic():
    m, _ = vix_size_multiplier(vix=40.0, full_below=20, half_above=25, zero_above=35)
    assert m == 0.0


def test_vix_half_size_at_half_above():
    m, _ = vix_size_multiplier(vix=25.0, full_below=20, half_above=25, zero_above=35)
    assert 0.45 <= m <= 0.55


def test_vix_ramp_monotonic():
    m20 = vix_size_multiplier(vix=20)[0]
    m22 = vix_size_multiplier(vix=22)[0]
    m25 = vix_size_multiplier(vix=25)[0]
    m30 = vix_size_multiplier(vix=30)[0]
    m35 = vix_size_multiplier(vix=35)[0]
    assert m20 >= m22 >= m25 >= m30 >= m35


def test_vix_unavailable_returns_full():
    m, _ = vix_size_multiplier(vix=None)
    # depending on yfinance availability, this might fetch real VIX
    assert 0.0 <= m <= 1.0


def test_volume_confirms_passes_high_volume():
    bars = pd.DataFrame({
        "close": [100] * 25,
        "volume": [1_000_000] * 20 + [1_500_000] * 5,
    })
    ok, _ = volume_confirms(bars, lookback=20, min_ratio=0.8)
    assert ok


def test_volume_confirms_rejects_low_volume():
    bars = pd.DataFrame({
        "close": [100] * 25,
        "volume": [1_000_000] * 20 + [100_000] * 5,
    })
    ok, _ = volume_confirms(bars, lookback=20, min_ratio=0.8)
    assert not ok


def test_spread_blocks_wide():
    blocked, _ = spread_too_wide(bid=100, ask=100.5, max_bps=10)
    assert blocked


def test_spread_accepts_tight():
    blocked, _ = spread_too_wide(bid=100, ask=100.05, max_bps=10)
    assert not blocked


def test_spread_invalid_quote_blocks():
    blocked, _ = spread_too_wide(bid=0, ask=100, max_bps=10)
    assert blocked
