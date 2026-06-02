from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class SignalResult:
    signal: str
    close: float
    fast_ma: float
    slow_ma: float


@dataclass(slots=True)
class DipSignalResult:
    signal: str        # "BUY" or "HOLD"
    close: float
    rsi: float
    drop_from_high_pct: float   # how far price fell from recent high
    above_long_ma: bool
    reason: str


def _compute_rsi(closes: pd.Series, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    delta = closes.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi_series = 100 - (100 / (1 + rs))
    val = rsi_series.iloc[-1]
    return float(val) if pd.notna(val) else 50.0


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """Compute the Average Directional Index (ADX) for trend strength.

    Returns a value 0-100. ADX > 20 indicates a trending market,
    ADX < 20 indicates a ranging/sideways market.
    """
    if len(close) < period * 2 + 1:
        return 0.0

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    # Smoothed averages (Wilder's smoothing)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1.0 / period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1.0 / period, min_periods=period).mean() / atr)

    # ADX
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan")) * 100
    adx = dx.ewm(alpha=1.0 / period, min_periods=period).mean()

    val = adx.iloc[-1]
    return float(val) if pd.notna(val) else 0.0


def moving_average_signal(
    bars: pd.DataFrame,
    fast: int,
    slow: int,
    adx_threshold: float = 20.0,
) -> SignalResult:
    """Generate BUY/SELL/HOLD signal based on MA crossover with ADX confirmation.

    When ADX is below adx_threshold, the market is considered sideways
    and crossover signals are suppressed to reduce whipsaws.

    Args:
        bars: OHLCV DataFrame.
        fast: Fast moving average period.
        slow: Slow moving average period.
        adx_threshold: Minimum ADX for signal generation (0 = disabled).
    """
    closes = bars["close"].astype(float)
    fast_ma = closes.rolling(fast).mean()
    slow_ma = closes.rolling(slow).mean()

    close = float(closes.iloc[-1])
    latest_fast = float(fast_ma.iloc[-1]) if pd.notna(fast_ma.iloc[-1]) else float("nan")
    latest_slow = float(slow_ma.iloc[-1]) if pd.notna(slow_ma.iloc[-1]) else float("nan")

    if pd.isna(latest_fast) or pd.isna(latest_slow):
        return SignalResult("HOLD", close, latest_fast, latest_slow)

    # ADX confirmation: suppress signals in ranging markets
    if adx_threshold > 0 and len(bars) >= 30:
        high = bars["high"].astype(float)
        low = bars["low"].astype(float)
        adx = _compute_adx(high, low, closes)
        if adx < adx_threshold:
            return SignalResult("HOLD", close, latest_fast, latest_slow)

    if latest_fast > latest_slow:
        return SignalResult("BUY", close, latest_fast, latest_slow)
    if latest_fast < latest_slow:
        return SignalResult("SELL", close, latest_fast, latest_slow)
    return SignalResult("HOLD", close, latest_fast, latest_slow)


def dip_buy_signal(
    bars: pd.DataFrame,
    rsi_threshold: float = 35.0,
    drop_pct: float = 5.0,
    lookback_days: int = 20,
    long_ma_period: int = 50,
    require_above_long_ma: bool = True,
) -> DipSignalResult:
    """Buy-the-dip signal: fires when a stock has pulled back meaningfully
    from a recent high while remaining in a longer-term uptrend.

    Conditions (ALL must be true):
    1. RSI(14) <= rsi_threshold  → oversold / washed out
    2. Price dropped >= drop_pct % from the N-day high → actual dip
    3. (optional) Price still above long_ma_period MA → don't buy falling knives

    Returns DipSignalResult with signal="BUY" when all conditions pass.
    """
    if len(bars) < max(lookback_days, long_ma_period, 15):
        return DipSignalResult("HOLD", 0.0, 50.0, 0.0, True, "insufficient data")

    closes = bars["close"].astype(float)
    close = float(closes.iloc[-1])

    # RSI
    rsi = _compute_rsi(closes)

    # Drop from recent high
    recent_high = float(closes.iloc[-lookback_days:].max())
    drop_from_high_pct = (recent_high - close) / recent_high * 100.0 if recent_high > 0 else 0.0

    # Long-term trend MA
    long_ma_series = closes.rolling(long_ma_period).mean()
    long_ma_val = float(long_ma_series.iloc[-1]) if pd.notna(long_ma_series.iloc[-1]) else close
    above_long_ma = close >= long_ma_val

    # Evaluate conditions
    rsi_ok = rsi <= rsi_threshold
    dip_ok = drop_from_high_pct >= drop_pct
    trend_ok = above_long_ma or not require_above_long_ma

    if rsi_ok and dip_ok and trend_ok:
        reason = (
            f"dip buy: RSI={rsi:.1f}≤{rsi_threshold}, "
            f"down {drop_from_high_pct:.1f}% from {lookback_days}d high, "
            f"{'above' if above_long_ma else 'below'} {long_ma_period}MA"
        )
        return DipSignalResult("BUY", close, rsi, drop_from_high_pct, above_long_ma, reason)

    reasons = []
    if not rsi_ok:   reasons.append(f"RSI={rsi:.1f}>{rsi_threshold}")
    if not dip_ok:   reasons.append(f"drop={drop_from_high_pct:.1f}%<{drop_pct}%")
    if not trend_ok: reasons.append(f"below {long_ma_period}MA")
    return DipSignalResult("HOLD", close, rsi, drop_from_high_pct, above_long_ma, "; ".join(reasons))

