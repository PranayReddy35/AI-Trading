"""Tests for options module: greeks, strategies, chains parsing, scoring."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from ai_trading.options.greeks import (
    bs_price,
    bs_greeks,
    implied_vol,
    pop_short_option,
)
from ai_trading.options.chains import OptionContract, parse_occ
from ai_trading.options.strategies import (
    build_long_call,
    build_long_put,
    build_cash_secured_put,
    build_covered_call,
    build_vertical_spread,
    build_iron_condor,
    build_strangle,
)


# ─────────────────────────────────────────────────────────────────────────────
# Greeks
# ─────────────────────────────────────────────────────────────────────────────

def test_bs_price_atm_call_roughly_correct():
    # ATM call, S=K=100, T=30d, sigma=20%, r=4.5%
    p = bs_price(100, 100, 30 / 365, 0.045, 0.20, "call")
    # Approx $2.50 (matches BS tables)
    assert 2.0 < p < 3.5


def test_bs_price_intrinsic_at_expiry():
    assert bs_price(110, 100, 0, 0.045, 0.20, "call") == 10.0
    assert bs_price(90, 100, 0, 0.045, 0.20, "put") == 10.0
    assert bs_price(100, 100, 0, 0.045, 0.20, "call") == 0.0


def test_bs_greeks_delta_ranges():
    g_call = bs_greeks(100, 100, 30/365, 0.045, 0.20, "call")
    g_put  = bs_greeks(100, 100, 30/365, 0.045, 0.20, "put")
    assert 0.0 < g_call.delta < 1.0
    assert -1.0 < g_put.delta < 0.0
    # Theta negative for both long calls/puts
    assert g_call.theta < 0
    assert g_put.theta < 0
    # Vega positive
    assert g_call.vega > 0


def test_implied_vol_roundtrip():
    true_vol = 0.30
    price = bs_price(100, 105, 45/365, 0.045, true_vol, "call")
    iv = implied_vol(price, 100, 105, 45/365, 0.045, "call")
    assert abs(iv - true_vol) < 1e-3


def test_pop_short_put_high_when_far_otm():
    # 60-strike put on $100 stock, low IV → expires worthless almost surely
    pop = pop_short_option(100, 60, 30/365, 0.045, 0.20, "put", credit=0.10)
    assert pop > 0.99


def test_pop_short_call_high_when_far_otm():
    pop = pop_short_option(100, 140, 30/365, 0.045, 0.20, "call", credit=0.10)
    assert pop > 0.99


# ─────────────────────────────────────────────────────────────────────────────
# OCC parsing
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_occ_call():
    p = parse_occ("AAPL250620C00200000")
    assert p == {
        "underlying": "AAPL",
        "expiry": date(2025, 6, 20),
        "type": "call",
        "strike": 200.0,
    }


def test_parse_occ_put():
    p = parse_occ("SPY261218P00350000")
    assert p == {
        "underlying": "SPY",
        "expiry": date(2026, 12, 18),
        "type": "put",
        "strike": 350.0,
    }


def test_parse_occ_invalid_returns_none():
    assert parse_occ("INVALID") is None
    assert parse_occ("") is None


# ─────────────────────────────────────────────────────────────────────────────
# Strategy builders — use a synthetic chain
# ─────────────────────────────────────────────────────────────────────────────

def _make_chain(spot: float = 100.0, dte: int = 30) -> list[OptionContract]:
    """Build a synthetic chain: strikes 80..120 step 5, calls + puts, with BS-priced greeks."""
    expiry = date.today() + timedelta(days=dte)
    strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120]
    chain: list[OptionContract] = []
    iv = 0.30
    T = dte / 365.0
    for K in strikes:
        for ot in ("call", "put"):
            g = bs_greeks(spot, K, T, 0.045, iv, ot)
            mid = max(g.price, 0.05)
            occ = f"TEST{expiry.strftime('%y%m%d')}{'C' if ot == 'call' else 'P'}{int(K*1000):08d}"
            chain.append(OptionContract(
                occ_symbol=occ,
                underlying="TEST",
                type=ot,
                strike=float(K),
                expiry=expiry,
                dte=dte,
                bid=mid * 0.98,
                ask=mid * 1.02,
                mid=mid,
                last=mid,
                volume=500,
                open_interest=1000,
                iv=iv,
                delta=g.delta,
                gamma=g.gamma,
                theta=g.theta,
                vega=g.vega,
                underlying_price=spot,
                source="test",
            ))
    return chain


def test_build_long_call_returns_candidate():
    chain = _make_chain()
    cands = build_long_call(chain, "TEST", 100.0, target_delta=0.40, min_dte=20, max_dte=40)
    assert len(cands) >= 1
    c = cands[0]
    assert c.strategy == "long_call"
    assert c.legs[0].side == "buy"
    assert c.debit_credit > 0       # debit
    assert c.max_loss == c.debit_credit
    assert c.max_profit == float("inf")
    assert 0.0 < c.pop < 1.0


def test_build_long_put_returns_candidate():
    chain = _make_chain()
    cands = build_long_put(chain, "TEST", 100.0, target_delta=-0.40, min_dte=20, max_dte=40)
    assert len(cands) >= 1
    assert cands[0].strategy == "long_put"
    assert cands[0].legs[0].type == "put"


def test_build_csp_returns_credit():
    chain = _make_chain()
    cands = build_cash_secured_put(chain, "TEST", 100.0, target_delta=-0.30, min_dte=20, max_dte=40)
    assert len(cands) >= 1
    c = cands[0]
    assert c.strategy == "cash_secured_put"
    assert c.debit_credit < 0       # credit
    assert c.bp_requirement > 0
    assert c.legs[0].side == "sell"
    # POP should be reasonably high for OTM put
    assert c.pop > 0.50


def test_build_covered_call_uses_cost_basis():
    chain = _make_chain()
    cands = build_covered_call(chain, "TEST", 100.0, cost_basis=95.0,
                                target_delta=0.25, min_dte=20, max_dte=40)
    assert len(cands) >= 1
    c = cands[0]
    assert c.strategy == "covered_call"
    assert c.legs[0].side == "sell"
    # bp_requirement = 0 since stock is the collateral
    assert c.bp_requirement == 0.0


def test_build_bull_call_spread():
    chain = _make_chain()
    cands = build_vertical_spread(chain, "TEST", 100.0, direction="bull_call",
                                    width=5.0, long_delta=0.50, min_dte=20, max_dte=40)
    assert len(cands) >= 1
    c = cands[0]
    assert c.strategy == "bull_call"
    assert len(c.legs) == 2
    assert c.debit_credit > 0       # debit
    assert c.max_loss > 0 and c.max_profit > 0
    # R:R bounded
    assert 0 < c.risk_reward < 99


def test_build_bear_put_spread():
    chain = _make_chain()
    cands = build_vertical_spread(chain, "TEST", 100.0, direction="bear_put",
                                    width=5.0, long_delta=0.50, min_dte=20, max_dte=40)
    assert len(cands) >= 1
    assert cands[0].strategy == "bear_put"


def test_build_bull_put_credit_spread():
    chain = _make_chain()
    cands = build_vertical_spread(chain, "TEST", 100.0, direction="bull_put",
                                    width=5.0, long_delta=0.30, min_dte=20, max_dte=40)
    assert len(cands) >= 1
    c = cands[0]
    assert c.strategy == "bull_put"
    assert c.debit_credit < 0       # credit


def test_build_iron_condor():
    chain = _make_chain()
    cands = build_iron_condor(chain, "TEST", 100.0, short_delta=0.20,
                                wing_width=5.0, min_dte=20, max_dte=40)
    assert len(cands) >= 1
    c = cands[0]
    assert c.strategy == "iron_condor"
    assert len(c.legs) == 4
    assert c.debit_credit < 0       # net credit
    assert len(c.breakevens) == 2
    assert c.breakevens[0] < c.underlying_price < c.breakevens[1]


def test_build_strangle_naked_undefined_loss():
    chain = _make_chain()
    cands = build_strangle(chain, "TEST", 100.0, short_delta=0.20, min_dte=20, max_dte=40)
    assert len(cands) >= 1
    c = cands[0]
    assert c.strategy == "short_strangle"
    assert c.max_loss == float("inf")
    assert len(c.legs) == 2


def test_no_candidates_when_dte_out_of_range():
    chain = _make_chain(dte=10)
    cands = build_long_call(chain, "TEST", 100.0, min_dte=30, max_dte=60)
    assert cands == []
