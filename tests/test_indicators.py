"""Tests for indicators module."""

from __future__ import annotations

import numpy as np

from ai_trading.strategy.indicators import (
    atr,
    atr_pct,
    bollinger,
    ichimoku,
    keltner,
    obv,
    rsi,
    stochastic,
    true_range,
    vwap,
)


def test_atr_positive_and_bounded(uptrend_bars):
    a = atr(uptrend_bars)
    assert (a.dropna() > 0).all()
    # ATR should be small relative to price
    assert atr_pct(uptrend_bars) < 0.5


def test_rsi_in_range(uptrend_bars):
    r = rsi(uptrend_bars["close"])
    assert r.between(0, 100).all()


def test_stochastic_in_range(uptrend_bars):
    k, d = stochastic(uptrend_bars)
    assert k.between(-1, 101).all()
    assert d.between(-1, 101).all()


def test_bollinger_ordering(uptrend_bars):
    b = bollinger(uptrend_bars["close"])
    valid = b.upper.dropna().index.intersection(b.lower.dropna().index)
    assert (b.upper.loc[valid] >= b.lower.loc[valid]).all()


def test_keltner_ordering(uptrend_bars):
    b = keltner(uptrend_bars)
    valid = b.upper.dropna().index.intersection(b.lower.dropna().index)
    assert (b.upper.loc[valid] >= b.lower.loc[valid]).all()


def test_vwap_finite(uptrend_bars):
    v = vwap(uptrend_bars)
    assert np.isfinite(v.iloc[-1])


def test_obv_runs(uptrend_bars):
    o = obv(uptrend_bars)
    assert len(o) == len(uptrend_bars)


def test_ichimoku_components(uptrend_bars):
    ic = ichimoku(uptrend_bars)
    for s in (ic.tenkan, ic.kijun, ic.senkou_a, ic.senkou_b):
        assert len(s) == len(uptrend_bars)


def test_true_range_nonneg(uptrend_bars):
    tr = true_range(uptrend_bars).dropna()
    assert (tr >= 0).all()
