"""Tests for risk-based sizing helpers."""

from __future__ import annotations

import pytest

from ai_trading.risk.sizing import (
    adaptive_thresholds,
    atr_stop_price,
    compute_atr_stop_and_size,
    risk_based_qty,
)


def test_atr_stop_price_buy():
    assert atr_stop_price(100, 2, "BUY", mult=2) == pytest.approx(96)


def test_atr_stop_price_sell():
    assert atr_stop_price(100, 2, "SELL", mult=2) == pytest.approx(104)


def test_atr_stop_not_negative():
    assert atr_stop_price(1, 10, "BUY", mult=2) == 0


def test_risk_based_qty_basic():
    # Risk 1% of $10k = $100; entry 100 stop 95 → $5/share risk → 20 shares
    assert risk_based_qty(entry=100, stop=95, equity=10_000, risk_pct=1.0) == 20


def test_risk_based_qty_cap_by_equity():
    # 1% of $1000 = $10; $1/share risk → 10 shares; but entry $200 caps at floor(1000/200)=5
    assert risk_based_qty(entry=200, stop=199, equity=1_000, risk_pct=1.0) == 5


def test_risk_based_qty_zero_risk_returns_zero():
    assert risk_based_qty(entry=100, stop=100, equity=10_000, risk_pct=1.0) == 0


def test_compute_atr_stop_and_size(uptrend_bars):
    entry = float(uptrend_bars["close"].iloc[-1])
    sz = compute_atr_stop_and_size(uptrend_bars, entry=entry, equity=100_000, risk_pct=0.5)
    assert sz.qty > 0
    assert sz.stop_price < entry
    assert sz.atr_value > 0


def test_adaptive_thresholds_scale_with_volatility(uptrend_bars, sideways_bars):
    bt_up, st_up = adaptive_thresholds(uptrend_bars)
    bt_sw, st_sw = adaptive_thresholds(sideways_bars)
    # Both should produce valid thresholds
    for bt, st in ((bt_up, st_up), (bt_sw, st_sw)):
        assert bt > 0 and st < 0
        # Within [0.5x, 2x] of base
        assert 0.5 * 0.15 <= bt <= 2.0 * 0.15
        assert -2.0 * 0.15 <= st <= -0.5 * 0.15
