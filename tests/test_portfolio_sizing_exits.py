"""Tests for risk.portfolio_sizing and risk.exits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_trading.risk.portfolio_sizing import (
    realised_daily_vol,
    vol_targeted_qty,
    portfolio_heat_check,
    fractional_kelly_qty,
)
from ai_trading.risk.exits import (
    TrailState,
    trail_atr_stop,
    should_time_stop,
    r_multiple,
    should_partial_take,
    breakeven_stop,
)


def _bars(prices: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="D")
    return pd.DataFrame({
        "open": prices, "high": [p * 1.01 for p in prices],
        "low":  [p * 0.99 for p in prices], "close": prices,
        "volume": [1_000_000] * len(prices),
    }, index=idx)


# ── vol targeting ───────────────────────────────────────────────────────────


def test_realised_vol_zero_for_constant_price():
    bars = _bars([100.0] * 25)
    assert realised_daily_vol(bars) == 0.0


def test_realised_vol_positive_for_random_walk():
    rng = np.random.default_rng(7)
    rets = rng.normal(0, 0.01, 60)
    prices = 100 * np.exp(np.cumsum(rets))
    sigma = realised_daily_vol(_bars(prices.tolist()))
    assert 0.005 < sigma < 0.02


def test_vol_targeted_qty_scales_inversely_with_vol():
    low_vol = _bars([100 + 0.1 * i for i in range(30)])           # tiny drift
    qty_low = vol_targeted_qty(bars=low_vol, entry=100.0, equity=100_000,
                               target_vol_pct=1.0, max_shares=10_000,
                               max_position_pct=10_000)
    rng = np.random.default_rng(3)
    high_vol_prices = (100 + np.cumsum(rng.normal(0, 5, 30))).clip(min=10)
    qty_high = vol_targeted_qty(bars=_bars(high_vol_prices.tolist()),
                                entry=100.0, equity=100_000,
                                target_vol_pct=1.0, max_shares=10_000,
                                max_position_pct=10_000)
    assert qty_low > qty_high


def test_vol_targeted_qty_zero_with_no_vol():
    qty = vol_targeted_qty(bars=_bars([100.0] * 30), entry=100.0,
                           equity=100_000, target_vol_pct=1.0)
    assert qty == 0


# ── portfolio heat ──────────────────────────────────────────────────────────


def test_heat_allows_when_under_cap():
    hc = portfolio_heat_check(open_risks={"A": 200.0}, new_symbol="B",
                              new_dollar_risk=100.0, equity=10_000, max_heat_pct=6.0)
    assert hc.allowed
    assert hc.projected_heat_pct == pytest.approx(3.0)


def test_heat_blocks_when_over_cap():
    hc = portfolio_heat_check(open_risks={"A": 500.0, "B": 100.0},
                              new_symbol="C", new_dollar_risk=50.0,
                              equity=10_000, max_heat_pct=6.0)
    assert not hc.allowed


def test_heat_replaces_existing_symbol_risk():
    # Re-entering A should not double-count
    hc = portfolio_heat_check(open_risks={"A": 500.0}, new_symbol="A",
                              new_dollar_risk=50.0, equity=10_000, max_heat_pct=6.0)
    assert hc.allowed
    assert hc.projected_heat_pct == pytest.approx(0.5)


# ── fractional Kelly ────────────────────────────────────────────────────────


def test_kelly_zero_when_no_edge():
    qty = fractional_kelly_qty(win_rate=0.5, avg_win_r=1.0, avg_loss_r=1.0,
                               equity=10_000, entry=100, risk_per_share=2.0)
    assert qty == 0


def test_kelly_positive_with_edge():
    qty = fractional_kelly_qty(win_rate=0.6, avg_win_r=1.5, avg_loss_r=1.0,
                               equity=10_000, entry=50, risk_per_share=2.0,
                               fraction=0.5, max_shares=1000)
    assert qty > 0


# ── exits ───────────────────────────────────────────────────────────────────


def test_trail_atr_stop_ratchets_up_with_peak():
    bars = _bars(list(range(100, 130)))
    state = TrailState(entry=100, peak=120, initial_stop=95)
    stop = trail_atr_stop(bars, state, atr_mult=2.0)
    # peak - 2*ATR should be well above 95
    assert stop > 95


def test_trail_atr_stop_respects_initial_stop_floor():
    bars = _bars(list(range(100, 130)))
    state = TrailState(entry=100, peak=101, initial_stop=99)
    stop = trail_atr_stop(bars, state, atr_mult=10.0)  # huge mult
    assert stop == 99


def test_time_stop_triggers_when_no_progress():
    state = TrailState(entry=100, peak=100, initial_stop=95, bars_held=10)
    exit_, _ = should_time_stop(state, max_bars=10, min_progress_r=0.5,
                                current_price=100.5)  # only 0.1R
    assert exit_


def test_time_stop_skips_when_winning():
    state = TrailState(entry=100, peak=103, initial_stop=95, bars_held=10)
    exit_, _ = should_time_stop(state, max_bars=10, min_progress_r=0.5,
                                current_price=103.0)  # 0.6R
    assert not exit_


def test_r_multiple_basic():
    state = TrailState(entry=100, peak=100, initial_stop=90)
    assert r_multiple(state, 110) == pytest.approx(1.0)
    assert r_multiple(state, 95) == pytest.approx(-0.5)


def test_partial_take_only_once():
    state = TrailState(entry=100, peak=100, initial_stop=90)
    assert should_partial_take(state, 110, trigger_r=1.0)
    state.partial_taken = True
    assert not should_partial_take(state, 120, trigger_r=1.0)


def test_breakeven_returns_entry_after_trigger():
    state = TrailState(entry=100, peak=100, initial_stop=90)
    assert breakeven_stop(state, 105, trigger_r=0.5) == 100
    assert breakeven_stop(state, 102, trigger_r=0.5) is None
