"""Tests for scanner enhancement helpers: liquidity, RS, BB squeeze, ATR levels,
meta-prob fallback, correlation dedup."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_trading import scanner as sc
from ai_trading.scanner import ScanResult


def _bars(n: int = 200, base: float = 100.0, vol: int = 1_000_000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drift = np.cumsum(rng.normal(0.001, 0.01, n))
    close = base * np.exp(drift)
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": [vol] * n},
        index=idx,
    )


# ── liquidity ─────────────────────────────────────────────────────────────────

def test_liquidity_passes_blue_chip():
    b = _bars(base=200.0, vol=5_000_000)
    ok, _, adv = sc._liquidity_ok(b, min_price=5.0, min_dollar_vol=5_000_000.0)
    assert ok and adv > 5_000_000


def test_liquidity_rejects_penny_stock():
    b = _bars(base=2.0, vol=10_000_000)
    ok, reason, _ = sc._liquidity_ok(b, min_price=5.0, min_dollar_vol=5_000_000.0)
    assert not ok and "price" in reason


def test_liquidity_rejects_low_dollar_vol():
    b = _bars(base=50.0, vol=1000)
    ok, reason, adv = sc._liquidity_ok(b, min_price=5.0, min_dollar_vol=5_000_000.0)
    assert not ok and "vol" in reason and adv < 5_000_000


# ── relative strength ────────────────────────────────────────────────────────

def test_rel_strength_positive_outperformance():
    sym = pd.Series(np.linspace(100, 110, 30))   # +10%
    spy = pd.Series(np.linspace(100, 105, 30))   # +5%
    rs = sc._rel_strength_pct(sym, spy, lookback=20)
    assert rs > 0


def test_rel_strength_underperformance_is_negative():
    sym = pd.Series(np.linspace(100, 102, 30))
    spy = pd.Series(np.linspace(100, 108, 30))
    rs = sc._rel_strength_pct(sym, spy, lookback=20)
    assert rs < 0


def test_rel_strength_short_history_zero():
    assert sc._rel_strength_pct(pd.Series([1, 2]), pd.Series([1, 2])) == 0.0


# ── BB squeeze ────────────────────────────────────────────────────────────────

def test_squeeze_score_in_unit_range():
    s = sc._bb_squeeze_score(_bars(n=200)["close"])
    assert 0.0 <= s <= 1.0


def test_squeeze_high_for_flat_series():
    # Flat then noisy → flat tail should be highly compressed
    n = 200
    close = pd.Series(np.concatenate([
        100 + np.random.default_rng(1).normal(0, 5, n - 30),
        np.full(30, 100.0),
    ]))
    s = sc._bb_squeeze_score(close, period=20, lookback=120)
    assert s >= 0.7


# ── ATR levels ────────────────────────────────────────────────────────────────

def test_atr_levels_produce_valid_rr():
    b = _bars(n=100)
    lv = sc._atr_levels(b, atr_mult_stop=2.0, risk_reward=2.0)
    assert {"entry", "stop", "target"} <= set(lv)
    assert lv["stop"] < lv["entry"] < lv["target"]
    # Reward % should be ≈ 2× risk %
    assert lv["reward_pct"] == pytest.approx(2 * lv["risk_pct"], rel=1e-2)


def test_atr_levels_empty_on_short_history():
    assert sc._atr_levels(_bars(n=5)) == {}


# ── meta-probability fallback ────────────────────────────────────────────────

def test_meta_probability_none_when_no_model(monkeypatch, tmp_path):
    # Reset cache, point to non-existent file
    sc._META_MODEL_CACHE.update({"model": None, "path": None, "tried": False})
    monkeypatch.setenv("BOT_META_MODEL_PATH", str(tmp_path / "no_such.joblib"))
    assert sc._meta_probability(_bars()) is None
    # Re-arm cache for other tests
    sc._META_MODEL_CACHE.update({"model": None, "path": None, "tried": False})


# ── correlation dedup ────────────────────────────────────────────────────────

def _mk_result(sym: str, score: float) -> ScanResult:
    return ScanResult(
        symbol=sym, score=score, signal="BUY", close=100.0, change_pct=0.0,
        momentum_5d=0.0, rsi=50.0, volume_surge=1.0, ma_gap_pct=0.0,
        trend_consistency=50.0, reason="", mode="eod",
    )


def test_diversify_drops_highly_correlated():
    rng = np.random.default_rng(11)
    base = pd.Series(np.cumsum(rng.normal(0, 0.01, 200)))
    idx = pd.date_range("2024-01-01", periods=200, freq="B")
    # A and B are nearly identical → highly correlated
    a = pd.DataFrame({"close": 100 + base.values}, index=idx)
    b = pd.DataFrame({"close": 100 + base.values + rng.normal(0, 0.001, 200)}, index=idx)
    # C is independent
    c = pd.DataFrame({"close": 100 + np.cumsum(rng.normal(0, 0.01, 200))}, index=idx)
    results = [_mk_result("A", 80), _mk_result("B", 70), _mk_result("C", 60)]
    bars = {"A": a, "B": b, "C": c}
    kept = sc._diversify(results, bars, max_correlation=0.85, lookback=60, keep_top=3)
    syms = [r.symbol for r in kept]
    assert "A" in syms and "C" in syms and "B" not in syms


def test_diversify_no_op_when_disabled():
    results = [_mk_result("A", 80), _mk_result("B", 70)]
    out = sc._diversify(results, {}, max_correlation=0.0)
    assert [r.symbol for r in out] == ["A", "B"]


# ── quality gates wiring (filters unavailable still passes) ──────────────────

def test_quality_gates_returns_pass_when_volume_ok(monkeypatch):
    # Force volume_confirms to return True, skip network calls for other gates
    from ai_trading.strategy import market_filters as mf
    monkeypatch.setattr(mf, "spy_trend_ok", lambda window=200: (True, "ok"))
    monkeypatch.setattr(mf, "vix_size_multiplier", lambda **kw: (1.0, "ok"))
    monkeypatch.setattr(mf, "in_earnings_blackout", lambda sym, blackout_days=2: mf.EarningsCheck(False, None, None, "ok"))
    qpass, flags = sc._quality_gates("SPY", _bars(), earnings_blackout_days=2)
    assert qpass and flags == "ok"


def test_quality_gates_flag_spy_off(monkeypatch):
    from ai_trading.strategy import market_filters as mf
    monkeypatch.setattr(mf, "spy_trend_ok", lambda window=200: (False, "off"))
    monkeypatch.setattr(mf, "vix_size_multiplier", lambda **kw: (1.0, "ok"))
    monkeypatch.setattr(mf, "in_earnings_blackout", lambda sym, blackout_days=2: mf.EarningsCheck(False, None, None, "ok"))
    qpass, flags = sc._quality_gates("SPY", _bars(), earnings_blackout_days=2)
    assert not qpass and "spy_trend" in flags
