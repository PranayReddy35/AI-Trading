"""Black-Scholes pricing, greeks, IV solver, and probability helpers.

European-style approximation. Good enough for screening US equity options
(which are American-style) — for short-dated near-ATM contracts the difference
is small. Dividends are ignored (q = 0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass(slots=True)
class Greeks:
    price: float
    delta: float
    gamma: float
    theta: float   # per-day
    vega: float    # per 1 vol point (0.01)
    rho: float     # per 1 rate point (0.01)


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan"), float("nan")
    vT = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / vT
    d2 = d1 - vT
    return d1, d2


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """Black-Scholes price. T in years. sigma annualized. option_type 'call' | 'put'."""
    if T <= 0:
        # intrinsic at expiry
        if option_type == "call":
            return max(0.0, S - K)
        return max(0.0, K - S)
    if sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bs_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> Greeks:
    """Compute price + greeks. theta is per-day, vega per 1 vol point."""
    price = bs_price(S, K, T, r, sigma, option_type)
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return Greeks(price=price, delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    pdf_d1 = _norm_pdf(d1)
    sqrt_T = math.sqrt(T)
    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * pdf_d1 * sqrt_T / 100.0  # per 1 vol point

    if option_type == "call":
        delta = _norm_cdf(d1)
        theta = (
            -S * pdf_d1 * sigma / (2.0 * sqrt_T)
            - r * K * math.exp(-r * T) * _norm_cdf(d2)
        ) / 365.0
        rho = K * T * math.exp(-r * T) * _norm_cdf(d2) / 100.0
    else:
        delta = _norm_cdf(d1) - 1.0
        theta = (
            -S * pdf_d1 * sigma / (2.0 * sqrt_T)
            + r * K * math.exp(-r * T) * _norm_cdf(-d2)
        ) / 365.0
        rho = -K * T * math.exp(-r * T) * _norm_cdf(-d2) / 100.0

    return Greeks(price=price, delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-4,
    max_iter: int = 100,
) -> float:
    """Solve for implied volatility via bisection. Returns NaN if no solution."""
    if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return float("nan")

    # No-arb floor / ceiling
    intrinsic = max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    if market_price < intrinsic - tol:
        return float("nan")

    lo, hi = 1e-4, 5.0  # 0.01% to 500% vol
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p = bs_price(S, K, T, r, mid, option_type)
        diff = p - market_price
        if abs(diff) < tol:
            return mid
        if diff > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            return mid
    return 0.5 * (lo + hi)


def pop_short_option(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "put",
    credit: float = 0.0,
) -> float:
    """Probability of profit for a short option (under BS / lognormal).

    For a short put expiring OTM the option expires worthless; with credit
    received, breakeven is at K - credit (for puts) or K + credit (for calls).
    Returns P(S_T >= breakeven) for short put, P(S_T <= breakeven) for short call.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    breakeven = (K - credit) if option_type == "put" else (K + credit)
    if breakeven <= 0:
        return 1.0
    vT = sigma * math.sqrt(T)
    d = (math.log(S / breakeven) + (r - 0.5 * sigma * sigma) * T) / vT
    if option_type == "put":
        return _norm_cdf(d)        # P(S_T >= breakeven)
    return _norm_cdf(-d)            # P(S_T <= breakeven)
