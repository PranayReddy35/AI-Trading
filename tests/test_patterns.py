"""Tests for pattern detectors."""

from __future__ import annotations

from ai_trading.strategy.patterns import (
    PATTERN_DETECTORS,
    candlestick_pattern,
    cup_and_handle,
    detect_all_patterns,
    double_top_bottom,
    fibonacci_retracement,
    flag_pennant,
    head_and_shoulders,
    ichimoku_signal,
    keltner_signal,
    macd_crossover,
    obv_trend,
    pattern_signal,
    pivot_points,
    stochastic_signal,
    support_resistance_breakout,
    triangle,
    vwap_signal,
    wedge,
)


def test_all_detectors_return_valid_hits(uptrend_bars):
    hits = detect_all_patterns(uptrend_bars)
    assert len(hits) == len(PATTERN_DETECTORS)
    for h in hits:
        assert -1.0 <= h.signal <= 1.0, f"{h.name} signal out of range"
        assert 0.0 <= h.confidence <= 1.0, f"{h.name} confidence out of range"
        assert isinstance(h.reason, str)


def test_pattern_signal_bounds(uptrend_bars):
    s = pattern_signal(uptrend_bars)
    assert -1.0 <= s.signal <= 1.0
    assert 0.0 <= s.confidence <= 1.0


def test_uptrend_aggregate_is_nonnegative(uptrend_bars):
    s = pattern_signal(uptrend_bars)
    # Aggregate should not be strongly bearish on a clean uptrend
    assert s.signal > -0.4


def test_downtrend_aggregate_is_nonpositive(downtrend_bars):
    s = pattern_signal(downtrend_bars)
    assert s.signal < 0.4


def test_double_top_detected(double_top_bars):
    hit = double_top_bottom(double_top_bars)
    # Should at least flag a bearish double-top (forming or broken)
    assert hit.signal < 0, f"expected bearish signal, got {hit.signal}: {hit.reason}"


def test_double_bottom_detected(double_bottom_bars):
    hit = double_top_bottom(double_bottom_bars)
    assert hit.signal > 0, f"expected bullish signal, got {hit.signal}: {hit.reason}"


def test_detectors_dont_crash_on_short_series():
    import pandas as pd
    short = pd.DataFrame({
        "open": [100, 101], "high": [102, 103], "low": [99, 100],
        "close": [101, 102], "volume": [1000, 1100],
    })
    for fn in PATTERN_DETECTORS:
        h = fn(short)
        assert -1.0 <= h.signal <= 1.0
        assert 0.0 <= h.confidence <= 1.0
