"""Reusable technical indicators (pure-Python / numpy / pandas — no paid sources).

All functions accept an OHLCV `pd.DataFrame` with lowercase columns and return
either a `pd.Series` or a small dataclass of series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------


def true_range(bars: pd.DataFrame) -> pd.Series:
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    close = bars["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder smoothing)."""
    tr = true_range(bars)
    # Wilder's smoothing == EMA with alpha = 1/period
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def atr_pct(bars: pd.DataFrame, period: int = 14) -> float:
    """Latest ATR as a fraction of latest close (0.02 = 2%)."""
    a = atr(bars, period).iloc[-1]
    c = float(bars["close"].iloc[-1])
    if c <= 0 or not np.isfinite(a):
        return 0.0
    return float(a) / c


# ---------------------------------------------------------------------------
# Trend / momentum
# ---------------------------------------------------------------------------


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.astype(float).diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1 / period, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def stochastic(bars: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
    """Stochastic oscillator (%K, %D)."""
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    close = bars["close"].astype(float)
    ll = low.rolling(k_period).min()
    hh = high.rolling(k_period).max()
    k = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k.fillna(50), d.fillna(50)


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


def vwap(bars: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP from session start.

    For intraday bars, callers should pass only today's bars. For daily bars,
    this returns a running VWAP over the full series.
    """
    typical = (bars["high"].astype(float) + bars["low"].astype(float) + bars["close"].astype(float)) / 3
    vol = bars["volume"].astype(float)
    cum_pv = (typical * vol).cumsum()
    cum_v = vol.cumsum().replace(0, np.nan)
    return (cum_pv / cum_v).ffill()


def obv(bars: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    close = bars["close"].astype(float)
    vol = bars["volume"].astype(float)
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * vol).cumsum()


# ---------------------------------------------------------------------------
# Bands
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Bands:
    middle: pd.Series
    upper: pd.Series
    lower: pd.Series


def bollinger(close: pd.Series, period: int = 20, mult: float = 2.0) -> Bands:
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std()
    return Bands(mid, mid + mult * sd, mid - mult * sd)


def keltner(bars: pd.DataFrame, period: int = 20, mult: float = 2.0) -> Bands:
    """Keltner channels (EMA ± mult * ATR). Less noisy than Bollinger."""
    close = bars["close"].astype(float)
    mid = close.ewm(span=period, adjust=False).mean()
    a = atr(bars, period)
    return Bands(mid, mid + mult * a, mid - mult * a)


# ---------------------------------------------------------------------------
# Ichimoku Cloud
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Ichimoku:
    tenkan: pd.Series      # conversion line (9)
    kijun: pd.Series       # base line (26)
    senkou_a: pd.Series    # leading span A
    senkou_b: pd.Series    # leading span B (52)
    chikou: pd.Series      # lagging span


def ichimoku(bars: pd.DataFrame) -> Ichimoku:
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    close = bars["close"].astype(float)

    def _mid(n: int) -> pd.Series:
        return (high.rolling(n).max() + low.rolling(n).min()) / 2

    tenkan = _mid(9)
    kijun = _mid(26)
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = _mid(52).shift(26)
    chikou = close.shift(-26)
    return Ichimoku(tenkan, kijun, senkou_a, senkou_b, chikou)
