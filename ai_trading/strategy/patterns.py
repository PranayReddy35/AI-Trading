"""Chart pattern detectors.

Implements classic technical analysis patterns and combines them into
a single `pattern_signal()` returning a `StrategySignal` consumable by
the ensemble.

Included pattern detectors:
- Fibonacci retracement (bounce from 38.2 / 50 / 61.8 levels)
- Support / resistance breakout (rolling N-bar high/low)
- Double top / double bottom (two-peak reversal)
- Candlestick patterns (bullish/bearish engulfing, hammer, shooting star)
- MACD signal-line crossover with histogram confirmation
- Pivot points (classic floor-trader pivot, daily basis)

Each detector returns a signed score in [-1, +1] with a confidence in [0, 1].
The aggregator combines them via confidence-weighted averaging.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# StrategySignal lives in strategy/types.py to avoid circular imports with ensemble.
from ai_trading.strategy.types import StrategySignal
from ai_trading.strategy.swings import detect_swings, last_swing_high_low
from ai_trading.strategy.indicators import (
    atr,
    bollinger,
    ichimoku,
    keltner,
    obv,
    rsi,
    stochastic,
    vwap,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PatternHit:
    """Single pattern detection result."""

    name: str
    signal: float  # -1..+1
    confidence: float  # 0..1
    reason: str
    levels: dict = field(default_factory=dict)


def _swing_high_low(close: pd.Series, lookback: int = 60) -> tuple[float, float, int, int]:
    """Return (swing_high, swing_low, hi_idx, lo_idx) using scipy-detected swings.

    Falls back to argmax/argmin within the window if no prominent swings are found.
    """
    window = close.iloc[-lookback:].reset_index(drop=True)
    # Drop NaN so argmax/argmin and scipy peak-finding work correctly.
    clean = window.dropna()
    if clean.empty:
        return float("nan"), float("nan"), 0, 0
    swings = detect_swings(clean, prominence_pct=0.02)
    if len(swings.high_idx) > 0 and len(swings.low_idx) > 0:
        hi_idx = int(swings.high_idx[-1])
        lo_idx = int(swings.low_idx[-1])
        return float(clean.iloc[hi_idx]), float(clean.iloc[lo_idx]), hi_idx, lo_idx
    hi_idx = int(clean.values.argmax())
    lo_idx = int(clean.values.argmin())
    return float(clean.iloc[hi_idx]), float(clean.iloc[lo_idx]), hi_idx, lo_idx


# ---------------------------------------------------------------------------
# Fibonacci retracement
# ---------------------------------------------------------------------------


FIB_RATIOS = (0.236, 0.382, 0.5, 0.618, 0.786)


def fibonacci_retracement(bars: pd.DataFrame, lookback: int = 60, tol: float = 0.01) -> PatternHit:
    """Detect bounce / rejection at key Fibonacci retracement levels.

    - In an uptrend (low precedes high): retracement levels are measured *down*
      from the swing high. A bounce at 38.2/50/61.8 is bullish.
    - In a downtrend (high precedes low): retracement levels are measured *up*
      from the swing low. Rejection at 38.2/50/61.8 is bearish.

    `tol` is the proximity to a level (as % of price) required to count as a touch.
    """
    close = bars["close"].astype(float)
    if len(close) < lookback:
        return PatternHit("fibonacci", 0.0, 0.0, "insufficient data")

    swing_hi, swing_lo, hi_idx, lo_idx = _swing_high_low(close, lookback)
    rng = swing_hi - swing_lo
    if not np.isfinite(rng) or rng <= 0:
        return PatternHit("fibonacci", 0.0, 0.0, "flat range")

    price = float(close.iloc[-1])
    uptrend = lo_idx < hi_idx  # low formed first → measure retracement from high

    levels: dict[str, float] = {}
    for r in FIB_RATIOS:
        levels[f"{r:.3f}"] = swing_hi - rng * r if uptrend else swing_lo + rng * r

    # Find the closest level
    closest_name, closest_price = min(
        levels.items(), key=lambda kv: abs(price - kv[1])
    )
    dist_pct = abs(price - closest_price) / price
    if dist_pct > tol:
        return PatternHit(
            "fibonacci",
            0.0,
            0.2,
            f"price {price:.2f} not near any fib (closest {closest_name}@{closest_price:.2f})",
            levels=levels,
        )

    # Bounce confirmation: 2-bar reversal in the right direction
    prev = float(close.iloc[-3]) if len(close) >= 3 else price
    last = float(close.iloc[-1])
    direction_up = last > prev

    ratio = float(closest_name)
    strength = 0.5 + (0.618 - abs(ratio - 0.618)) * 0.5  # peak at 0.618
    strength = max(0.3, min(1.0, strength))

    if uptrend and direction_up:
        return PatternHit(
            "fibonacci",
            min(1.0, strength),
            0.7,
            f"bullish bounce at fib {closest_name} ({closest_price:.2f})",
            levels=levels,
        )
    if (not uptrend) and (not direction_up):
        return PatternHit(
            "fibonacci",
            -min(1.0, strength),
            0.7,
            f"bearish rejection at fib {closest_name} ({closest_price:.2f})",
            levels=levels,
        )

    return PatternHit(
        "fibonacci",
        0.0,
        0.3,
        f"at fib {closest_name} but no confirmation",
        levels=levels,
    )


# ---------------------------------------------------------------------------
# Support / Resistance breakout
# ---------------------------------------------------------------------------


def support_resistance_breakout(bars: pd.DataFrame, lookback: int = 20) -> PatternHit:
    """Detect breakout above resistance or breakdown below support.

    Resistance = rolling `lookback`-bar high (excluding current bar).
    Support    = rolling `lookback`-bar low  (excluding current bar).
    Requires volume > 1.2× 20-bar average to confirm.
    """
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    close = bars["close"].astype(float)
    volume = bars["volume"].astype(float) if "volume" in bars else None

    if len(close) < lookback + 2:
        return PatternHit("sr_breakout", 0.0, 0.0, "insufficient data")

    resistance = float(high.iloc[-(lookback + 1):-1].max())
    support = float(low.iloc[-(lookback + 1):-1].min())
    price = float(close.iloc[-1])

    vol_ok = True
    vol_surge = 1.0
    if volume is not None and len(volume) >= 20:
        avg = float(volume.iloc[-20:].mean())
        vol_surge = float(volume.iloc[-1]) / avg if avg > 0 else 1.0
        vol_ok = vol_surge >= 1.2

    if price > resistance:
        strength = min(1.0, (price - resistance) / resistance * 20)
        conf = 0.7 if vol_ok else 0.4
        return PatternHit(
            "sr_breakout",
            strength,
            conf,
            f"breakout above {resistance:.2f} (vol {vol_surge:.1f}x)",
            levels={"resistance": resistance, "support": support},
        )
    if price < support:
        strength = -min(1.0, (support - price) / support * 20)
        conf = 0.7 if vol_ok else 0.4
        return PatternHit(
            "sr_breakout",
            strength,
            conf,
            f"breakdown below {support:.2f} (vol {vol_surge:.1f}x)",
            levels={"resistance": resistance, "support": support},
        )

    # Inside range — small positional bias toward the nearer level (mean-revert from extremes)
    rng = resistance - support
    if rng > 0:
        pos = (price - support) / rng  # 0..1
        if pos > 0.9:
            return PatternHit("sr_breakout", -0.3, 0.4, "near resistance", levels={"resistance": resistance, "support": support})
        if pos < 0.1:
            return PatternHit("sr_breakout", 0.3, 0.4, "near support", levels={"resistance": resistance, "support": support})
    return PatternHit("sr_breakout", 0.0, 0.2, "inside range", levels={"resistance": resistance, "support": support})


# ---------------------------------------------------------------------------
# Double top / double bottom
# ---------------------------------------------------------------------------


def double_top_bottom(bars: pd.DataFrame, lookback: int = 80, tol: float = 0.02) -> PatternHit:
    """Detect double-top (bearish) or double-bottom (bullish) using scipy swings.

    Looks for two peaks/troughs within `lookback` bars whose prices match within
    `tol` (fraction of price). Neckline break (intermediate extreme) confirms.
    """
    close = bars["close"].astype(float)
    if len(close) < lookback:
        return PatternHit("double_top_bottom", 0.0, 0.0, "insufficient data")

    window = close.iloc[-lookback:].reset_index(drop=True)
    swings = detect_swings(window, prominence_pct=0.015, distance=4)
    price = float(close.iloc[-1])

    # Double top: last two swing highs within tol, current price < intermediate trough
    if len(swings.high_idx) >= 2:
        i1, i2 = int(swings.high_idx[-2]), int(swings.high_idx[-1])
        p1, p2 = float(window.iloc[i1]), float(window.iloc[i2])
        if abs(p1 - p2) / max(p1, p2) < tol and i2 - i1 >= 5:
            mid_trough = float(window.iloc[i1:i2].min())
            if price < mid_trough:
                return PatternHit(
                    "double_top_bottom",
                    -0.8, 0.75,
                    f"double top at {p1:.2f}/{p2:.2f}, broke neckline {mid_trough:.2f}",
                    levels={"peak1": p1, "peak2": p2, "neckline": mid_trough},
                )
            return PatternHit(
                "double_top_bottom",
                -0.4, 0.55,
                f"double top forming at {p1:.2f}/{p2:.2f}",
                levels={"peak1": p1, "peak2": p2, "neckline": mid_trough},
            )

    if len(swings.low_idx) >= 2:
        i1, i2 = int(swings.low_idx[-2]), int(swings.low_idx[-1])
        t1, t2 = float(window.iloc[i1]), float(window.iloc[i2])
        if abs(t1 - t2) / max(t1, t2) < tol and i2 - i1 >= 5:
            mid_peak = float(window.iloc[i1:i2].max())
            if price > mid_peak:
                return PatternHit(
                    "double_top_bottom",
                    0.8, 0.75,
                    f"double bottom at {t1:.2f}/{t2:.2f}, broke neckline {mid_peak:.2f}",
                    levels={"trough1": t1, "trough2": t2, "neckline": mid_peak},
                )
            return PatternHit(
                "double_top_bottom",
                0.4, 0.55,
                f"double bottom forming at {t1:.2f}/{t2:.2f}",
                levels={"trough1": t1, "trough2": t2, "neckline": mid_peak},
            )

    return PatternHit("double_top_bottom", 0.0, 0.2, "no pattern")


# ---------------------------------------------------------------------------
# Candlestick patterns
# ---------------------------------------------------------------------------


def candlestick_pattern(bars: pd.DataFrame) -> PatternHit:
    """Detect single/two-bar reversal candles on the latest bars.

    Patterns:
    - Bullish / bearish engulfing
    - Hammer (bullish), shooting star (bearish)
    """
    if len(bars) < 2:
        return PatternHit("candlestick", 0.0, 0.0, "insufficient data")

    o = bars["open"].astype(float).values
    h = bars["high"].astype(float).values
    l = bars["low"].astype(float).values
    c = bars["close"].astype(float).values

    o1, c1 = float(o[-2]), float(c[-2])
    o2, h2, l2, c2 = float(o[-1]), float(h[-1]), float(l[-1]), float(c[-1])

    prev_bear = c1 < o1
    prev_bull = c1 > o1
    cur_bull = c2 > o2
    cur_bear = c2 < o2

    # Engulfing
    if prev_bear and cur_bull and o2 <= c1 and c2 >= o1 and (c2 - o2) > (o1 - c1):
        return PatternHit("candlestick", 0.7, 0.65, "bullish engulfing")
    if prev_bull and cur_bear and o2 >= c1 and c2 <= o1 and (o2 - c2) > (c1 - o1):
        return PatternHit("candlestick", -0.7, 0.65, "bearish engulfing")

    # Hammer / shooting star (single-bar)
    body = abs(c2 - o2)
    rng = h2 - l2
    if rng <= 0:
        return PatternHit("candlestick", 0.0, 0.1, "doji-like, no range")
    upper_wick = h2 - max(o2, c2)
    lower_wick = min(o2, c2) - l2

    # Hammer: small body near top, long lower wick (>= 2x body), small upper wick
    if body / rng < 0.35 and lower_wick >= 2 * body and upper_wick <= body:
        return PatternHit("candlestick", 0.6, 0.55, "hammer")
    # Shooting star: small body near bottom, long upper wick
    if body / rng < 0.35 and upper_wick >= 2 * body and lower_wick <= body:
        return PatternHit("candlestick", -0.6, 0.55, "shooting star")

    return PatternHit("candlestick", 0.0, 0.2, "no candle pattern")


# ---------------------------------------------------------------------------
# MACD crossover
# ---------------------------------------------------------------------------


def macd_crossover(bars: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> PatternHit:
    """Detect MACD line crossing signal line, with histogram confirmation."""
    close = bars["close"].astype(float)
    if len(close) < slow + signal + 2:
        return PatternHit("macd", 0.0, 0.0, "insufficient data")

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig

    h_now, h_prev = float(hist.iloc[-1]), float(hist.iloc[-2])
    price = float(close.iloc[-1])
    norm = max(abs(h_now), abs(h_prev), 1e-9) / price  # normalize by price scale

    # Zero-cross of histogram = MACD/signal crossover
    if h_prev <= 0 < h_now:
        strength = min(1.0, norm * 500)
        return PatternHit("macd", max(0.4, strength), 0.7, f"bullish MACD cross (hist {h_now:+.4f})")
    if h_prev >= 0 > h_now:
        strength = -min(1.0, norm * 500)
        return PatternHit("macd", min(-0.4, strength), 0.7, f"bearish MACD cross (hist {h_now:+.4f})")

    # Persistent positive/negative histogram = continuation bias
    if h_now > 0 and hist.iloc[-3] > 0:
        return PatternHit("macd", 0.2, 0.4, "MACD above signal (bullish)")
    if h_now < 0 and hist.iloc[-3] < 0:
        return PatternHit("macd", -0.2, 0.4, "MACD below signal (bearish)")

    return PatternHit("macd", 0.0, 0.2, "MACD flat")


# ---------------------------------------------------------------------------
# Pivot points (classic floor-trader)
# ---------------------------------------------------------------------------


def pivot_points(bars: pd.DataFrame) -> PatternHit:
    """Classic pivot points based on the prior bar (daily HLC).

    P = (H + L + C) / 3
    R1 = 2P − L,  S1 = 2P − H
    R2 = P + (H − L),  S2 = P − (H − L)

    Signal: bullish near S1/S2, bearish near R1/R2 (proximity ≤ 0.5% of price).
    """
    if len(bars) < 2:
        return PatternHit("pivot", 0.0, 0.0, "insufficient data")

    prev = bars.iloc[-2]
    h, l, c = float(prev["high"]), float(prev["low"]), float(prev["close"])
    price = float(bars["close"].iloc[-1])

    p = (h + l + c) / 3
    r1 = 2 * p - l
    s1 = 2 * p - h
    r2 = p + (h - l)
    s2 = p - (h - l)

    levels = {"P": p, "R1": r1, "R2": r2, "S1": s1, "S2": s2}
    tol = price * 0.005  # 0.5%

    if abs(price - s2) <= tol:
        return PatternHit("pivot", 0.7, 0.6, f"at S2 support ({s2:.2f})", levels=levels)
    if abs(price - s1) <= tol:
        return PatternHit("pivot", 0.5, 0.55, f"at S1 support ({s1:.2f})", levels=levels)
    if abs(price - r2) <= tol:
        return PatternHit("pivot", -0.7, 0.6, f"at R2 resistance ({r2:.2f})", levels=levels)
    if abs(price - r1) <= tol:
        return PatternHit("pivot", -0.5, 0.55, f"at R1 resistance ({r1:.2f})", levels=levels)

    # Above/below pivot bias (weak)
    if price > p:
        return PatternHit("pivot", 0.15, 0.3, f"above pivot {p:.2f}", levels=levels)
    return PatternHit("pivot", -0.15, 0.3, f"below pivot {p:.2f}", levels=levels)


# ---------------------------------------------------------------------------
# Head & Shoulders / Inverse H&S
# ---------------------------------------------------------------------------


def head_and_shoulders(bars: pd.DataFrame, lookback: int = 120, tol: float = 0.04) -> PatternHit:
    """Detect classic H&S (bearish) or inverse H&S (bullish).

    Requires three swing highs (L, H, R) with H > L, H > R, and L ≈ R within `tol`.
    Confirmation = price closes below (above) the neckline formed by the two troughs.
    """
    close = bars["close"].astype(float)
    if len(close) < lookback:
        return PatternHit("head_shoulders", 0.0, 0.0, "insufficient data")

    window = close.iloc[-lookback:].reset_index(drop=True)
    sw = detect_swings(window, prominence_pct=0.02, distance=4)
    price = float(close.iloc[-1])

    # Bearish H&S: need three peaks L,H,R and two troughs between them
    if len(sw.high_idx) >= 3 and len(sw.low_idx) >= 2:
        l_i, h_i, r_i = int(sw.high_idx[-3]), int(sw.high_idx[-2]), int(sw.high_idx[-1])
        L, H, R = float(window.iloc[l_i]), float(window.iloc[h_i]), float(window.iloc[r_i])
        if H > L and H > R and abs(L - R) / max(L, R) < tol and (r_i - l_i) >= 10:
            troughs = [i for i in sw.low_idx if l_i < int(i) < r_i]
            if len(troughs) >= 2:
                t1, t2 = int(troughs[0]), int(troughs[-1])
                neckline = (float(window.iloc[t1]) + float(window.iloc[t2])) / 2
                if price < neckline:
                    return PatternHit(
                        "head_shoulders", -0.85, 0.8,
                        f"H&S broke neckline {neckline:.2f} (L={L:.2f} H={H:.2f} R={R:.2f})",
                        levels={"left": L, "head": H, "right": R, "neckline": neckline},
                    )
                return PatternHit(
                    "head_shoulders", -0.45, 0.55,
                    f"H&S forming, neckline {neckline:.2f}",
                    levels={"left": L, "head": H, "right": R, "neckline": neckline},
                )

    # Inverse H&S: three troughs with middle lowest
    if len(sw.low_idx) >= 3 and len(sw.high_idx) >= 2:
        l_i, h_i, r_i = int(sw.low_idx[-3]), int(sw.low_idx[-2]), int(sw.low_idx[-1])
        L, H, R = float(window.iloc[l_i]), float(window.iloc[h_i]), float(window.iloc[r_i])
        if H < L and H < R and abs(L - R) / max(L, R) < tol and (r_i - l_i) >= 10:
            peaks = [i for i in sw.high_idx if l_i < int(i) < r_i]
            if len(peaks) >= 2:
                t1, t2 = int(peaks[0]), int(peaks[-1])
                neckline = (float(window.iloc[t1]) + float(window.iloc[t2])) / 2
                if price > neckline:
                    return PatternHit(
                        "head_shoulders", 0.85, 0.8,
                        f"inverse H&S broke neckline {neckline:.2f}",
                        levels={"left": L, "head": H, "right": R, "neckline": neckline},
                    )
                return PatternHit(
                    "head_shoulders", 0.45, 0.55,
                    f"inverse H&S forming, neckline {neckline:.2f}",
                    levels={"left": L, "head": H, "right": R, "neckline": neckline},
                )

    return PatternHit("head_shoulders", 0.0, 0.2, "no H&S")


# ---------------------------------------------------------------------------
# Triangles (ascending / descending / symmetrical)
# ---------------------------------------------------------------------------


def _linfit(y: np.ndarray) -> tuple[float, float]:
    x = np.arange(len(y), dtype=float)
    if len(x) < 2:
        return 0.0, float(y[-1]) if len(y) else 0.0
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def triangle(bars: pd.DataFrame, lookback: int = 60) -> PatternHit:
    """Detect ascending / descending / symmetrical triangles.

    Fits linear regressions to the swing highs and swing lows over `lookback`.
    - Ascending  : flat highs + rising lows  → bullish breakout bias.
    - Descending : falling highs + flat lows → bearish breakdown bias.
    - Symmetrical: falling highs + rising lows → continuation in prior direction.
    Signal fires once price breaks above (below) the upper (lower) trendline.
    """
    close = bars["close"].astype(float)
    if len(close) < lookback:
        return PatternHit("triangle", 0.0, 0.0, "insufficient data")

    window = close.iloc[-lookback:].reset_index(drop=True)
    sw = detect_swings(window, prominence_pct=0.015, distance=3)
    if len(sw.high_idx) < 3 or len(sw.low_idx) < 3:
        return PatternHit("triangle", 0.0, 0.2, "too few swings")

    hi_slope, hi_int = _linfit(sw.high_prices[-4:])
    lo_slope, lo_int = _linfit(sw.low_prices[-4:])
    mean_price = float(window.mean())
    # Normalize slopes to fraction-of-price-per-swing
    hi_n = hi_slope / mean_price if mean_price > 0 else 0
    lo_n = lo_slope / mean_price if mean_price > 0 else 0

    # Project current trendline levels at the last bar position
    last_hi_pos = float(len(sw.high_prices) - 1)
    last_lo_pos = float(len(sw.low_prices) - 1)
    upper = hi_slope * last_hi_pos + hi_int
    lower = lo_slope * last_lo_pos + lo_int
    price = float(close.iloc[-1])

    flat = 0.002  # 0.2 % per swing considered "flat"
    if abs(hi_n) < flat and lo_n > flat:
        kind = "ascending"
        bias = +1.0
    elif hi_n < -flat and abs(lo_n) < flat:
        kind = "descending"
        bias = -1.0
    elif hi_n < -flat and lo_n > flat:
        kind = "symmetrical"
        # Use recent slope of close as continuation direction
        recent_slope = float(np.polyfit(np.arange(20), window.iloc[-20:].values, 1)[0])
        bias = 1.0 if recent_slope > 0 else -1.0
    else:
        return PatternHit("triangle", 0.0, 0.2, "no triangle")

    if price > upper:
        return PatternHit(
            "triangle", min(1.0, 0.5 + 0.5 * (price - upper) / max(upper, 1e-9) * 50),
            0.7, f"{kind} triangle breakout above {upper:.2f}",
            levels={"upper": upper, "lower": lower},
        )
    if price < lower:
        return PatternHit(
            "triangle", max(-1.0, -0.5 - 0.5 * (lower - price) / max(lower, 1e-9) * 50),
            0.7, f"{kind} triangle breakdown below {lower:.2f}",
            levels={"upper": upper, "lower": lower},
        )
    # Inside triangle: weak bias toward expected breakout direction
    return PatternHit(
        "triangle", 0.2 * bias, 0.35,
        f"{kind} triangle forming (upper={upper:.2f} lower={lower:.2f})",
        levels={"upper": upper, "lower": lower},
    )


# ---------------------------------------------------------------------------
# Flags & Pennants (continuation after strong move)
# ---------------------------------------------------------------------------


def flag_pennant(bars: pd.DataFrame, pole_lookback: int = 10, flag_lookback: int = 10) -> PatternHit:
    """Detect bull/bear flags: sharp move (pole) then tight consolidation (flag)."""
    close = bars["close"].astype(float)
    if len(close) < pole_lookback + flag_lookback + 2:
        return PatternHit("flag", 0.0, 0.0, "insufficient data")

    pole_start = close.iloc[-(pole_lookback + flag_lookback)]
    pole_end = close.iloc[-flag_lookback]
    pole_move_pct = (pole_end - pole_start) / pole_start if pole_start > 0 else 0

    flag_segment = close.iloc[-flag_lookback:]
    flag_range = (flag_segment.max() - flag_segment.min()) / flag_segment.mean()
    flag_slope = float(np.polyfit(np.arange(len(flag_segment)), flag_segment.values, 1)[0])
    flag_slope_pct = flag_slope / flag_segment.mean() if flag_segment.mean() > 0 else 0

    # Bull flag: strong up move then small downward/sideways drift, breakout above flag high
    if pole_move_pct > 0.05 and flag_range < 0.04 and flag_slope_pct <= 0:
        flag_high = float(flag_segment.max())
        price = float(close.iloc[-1])
        if price > flag_high:
            return PatternHit("flag", 0.7, 0.7, f"bull flag breakout (pole +{pole_move_pct*100:.1f}%)",
                              levels={"flag_high": flag_high})
        return PatternHit("flag", 0.3, 0.5, f"bull flag forming (pole +{pole_move_pct*100:.1f}%)",
                          levels={"flag_high": flag_high})

    if pole_move_pct < -0.05 and flag_range < 0.04 and flag_slope_pct >= 0:
        flag_low = float(flag_segment.min())
        price = float(close.iloc[-1])
        if price < flag_low:
            return PatternHit("flag", -0.7, 0.7, f"bear flag breakdown (pole {pole_move_pct*100:.1f}%)",
                              levels={"flag_low": flag_low})
        return PatternHit("flag", -0.3, 0.5, f"bear flag forming (pole {pole_move_pct*100:.1f}%)",
                          levels={"flag_low": flag_low})

    return PatternHit("flag", 0.0, 0.2, "no flag")


# ---------------------------------------------------------------------------
# Wedges (rising / falling) — reversal bias
# ---------------------------------------------------------------------------


def wedge(bars: pd.DataFrame, lookback: int = 60) -> PatternHit:
    """Detect rising (bearish) or falling (bullish) wedges.

    Both trendlines slope in the same direction and converge:
      rising wedge  : both up, lower slope > upper slope (converging) → bearish reversal
      falling wedge : both down, upper slope > lower slope → bullish reversal
    """
    close = bars["close"].astype(float)
    if len(close) < lookback:
        return PatternHit("wedge", 0.0, 0.0, "insufficient data")

    window = close.iloc[-lookback:].reset_index(drop=True)
    sw = detect_swings(window, prominence_pct=0.012, distance=3)
    if len(sw.high_idx) < 3 or len(sw.low_idx) < 3:
        return PatternHit("wedge", 0.0, 0.2, "too few swings")

    hi_slope, _ = _linfit(sw.high_prices[-4:])
    lo_slope, _ = _linfit(sw.low_prices[-4:])
    mean_price = float(window.mean())
    hi_n = hi_slope / mean_price if mean_price > 0 else 0
    lo_n = lo_slope / mean_price if mean_price > 0 else 0

    # Rising wedge: both slopes positive, lower steeper → reversal down
    if hi_n > 0.001 and lo_n > 0.001 and lo_n > hi_n:
        return PatternHit("wedge", -0.6, 0.6, f"rising wedge (bearish reversal)")
    # Falling wedge: both slopes negative, upper steeper down → reversal up
    if hi_n < -0.001 and lo_n < -0.001 and hi_n < lo_n:
        return PatternHit("wedge", 0.6, 0.6, f"falling wedge (bullish reversal)")
    return PatternHit("wedge", 0.0, 0.2, "no wedge")


# ---------------------------------------------------------------------------
# Cup and Handle
# ---------------------------------------------------------------------------


def cup_and_handle(bars: pd.DataFrame, lookback: int = 120) -> PatternHit:
    """Detect a cup (U-shape) followed by a handle (small pullback), then breakout."""
    close = bars["close"].astype(float)
    if len(close) < lookback:
        return PatternHit("cup_handle", 0.0, 0.0, "insufficient data")

    w = close.iloc[-lookback:].reset_index(drop=True)
    # Split into cup (first 75%) and handle (last 25%)
    cup_end = int(len(w) * 0.75)
    cup = w.iloc[:cup_end]
    handle = w.iloc[cup_end:]
    if len(cup) < 20 or len(handle) < 5:
        return PatternHit("cup_handle", 0.0, 0.2, "segments too short")

    cup_left = float(cup.iloc[0])
    cup_right = float(cup.iloc[-1])
    cup_bottom = float(cup.min())
    cup_top = max(cup_left, cup_right)

    # Cup symmetry: left ≈ right within 5 %, bottom well below top
    rim_diff = abs(cup_left - cup_right) / max(cup_left, cup_right)
    depth = (cup_top - cup_bottom) / cup_top if cup_top > 0 else 0
    if rim_diff > 0.05 or depth < 0.08:
        return PatternHit("cup_handle", 0.0, 0.2, f"no cup (rim_diff={rim_diff:.2f} depth={depth:.2f})")

    # Handle: shallow pullback (< 50 % of cup depth), small range, recent recovery
    handle_low = float(handle.min())
    handle_pullback = (cup_top - handle_low) / cup_top
    if handle_pullback > depth * 0.6:
        return PatternHit("cup_handle", 0.0, 0.3, "handle too deep")

    price = float(close.iloc[-1])
    if price > cup_top:
        return PatternHit(
            "cup_handle", 0.8, 0.75,
            f"cup & handle breakout above {cup_top:.2f}",
            levels={"rim": cup_top, "cup_bottom": cup_bottom, "handle_low": handle_low},
        )
    return PatternHit(
        "cup_handle", 0.35, 0.5,
        f"cup & handle forming (rim {cup_top:.2f})",
        levels={"rim": cup_top, "cup_bottom": cup_bottom, "handle_low": handle_low},
    )


# ---------------------------------------------------------------------------
# Indicator-derived signals: Ichimoku, Stochastic, OBV trend, VWAP, Keltner
# ---------------------------------------------------------------------------


def ichimoku_signal(bars: pd.DataFrame) -> PatternHit:
    """Ichimoku Cloud bias.

    Bullish: price > cloud AND tenkan > kijun AND cloud is green (senkou_a > senkou_b).
    Bearish: mirror.
    Inside cloud → low-confidence neutral.
    """
    if len(bars) < 60:
        return PatternHit("ichimoku", 0.0, 0.0, "insufficient data")
    ic = ichimoku(bars)
    price = float(bars["close"].iloc[-1])
    sa = float(ic.senkou_a.iloc[-1]) if pd.notna(ic.senkou_a.iloc[-1]) else price
    sb = float(ic.senkou_b.iloc[-1]) if pd.notna(ic.senkou_b.iloc[-1]) else price
    tk = float(ic.tenkan.iloc[-1]) if pd.notna(ic.tenkan.iloc[-1]) else price
    kj = float(ic.kijun.iloc[-1]) if pd.notna(ic.kijun.iloc[-1]) else price

    cloud_top, cloud_bot = max(sa, sb), min(sa, sb)
    green_cloud = sa > sb
    levels = {"tenkan": tk, "kijun": kj, "senkou_a": sa, "senkou_b": sb}

    if price > cloud_top and tk > kj and green_cloud:
        return PatternHit("ichimoku", 0.7, 0.7, "bullish: above green cloud, TK>KJ", levels=levels)
    if price < cloud_bot and tk < kj and not green_cloud:
        return PatternHit("ichimoku", -0.7, 0.7, "bearish: below red cloud, TK<KJ", levels=levels)
    if price > cloud_top:
        return PatternHit("ichimoku", 0.3, 0.5, "above cloud", levels=levels)
    if price < cloud_bot:
        return PatternHit("ichimoku", -0.3, 0.5, "below cloud", levels=levels)
    return PatternHit("ichimoku", 0.0, 0.3, "inside cloud (no trend)", levels=levels)


def stochastic_signal(bars: pd.DataFrame) -> PatternHit:
    """%K/%D crossover with overbought/oversold context."""
    if len(bars) < 20:
        return PatternHit("stochastic", 0.0, 0.0, "insufficient data")
    k, d = stochastic(bars)
    k1, k0 = float(k.iloc[-2]), float(k.iloc[-1])
    d1, d0 = float(d.iloc[-2]), float(d.iloc[-1])

    # Bullish cross in oversold zone
    if k1 < d1 and k0 > d0 and k0 < 30:
        return PatternHit("stochastic", 0.7, 0.65, f"bullish cross in oversold (%K={k0:.0f})")
    if k1 > d1 and k0 < d0 and k0 > 70:
        return PatternHit("stochastic", -0.7, 0.65, f"bearish cross in overbought (%K={k0:.0f})")
    if k0 < 20:
        return PatternHit("stochastic", 0.3, 0.4, f"oversold %K={k0:.0f}")
    if k0 > 80:
        return PatternHit("stochastic", -0.3, 0.4, f"overbought %K={k0:.0f}")
    return PatternHit("stochastic", 0.0, 0.2, f"neutral %K={k0:.0f}")


def obv_trend(bars: pd.DataFrame, period: int = 20) -> PatternHit:
    """On-Balance Volume trend confirms or contradicts price trend."""
    if len(bars) < period * 2:
        return PatternHit("obv", 0.0, 0.0, "insufficient data")
    o = obv(bars)
    o_now, o_prev = float(o.iloc[-1]), float(o.iloc[-period])
    p_now, p_prev = float(bars["close"].iloc[-1]), float(bars["close"].iloc[-period])
    obv_chg = (o_now - o_prev) / max(abs(o_prev), 1.0)
    price_chg = (p_now - p_prev) / p_prev if p_prev > 0 else 0

    # Divergence: price up but OBV down (bearish) or price down OBV up (bullish)
    if price_chg > 0.02 and obv_chg < -0.05:
        return PatternHit("obv", -0.5, 0.55, f"bearish divergence (price +{price_chg*100:.1f}% OBV {obv_chg*100:+.1f}%)")
    if price_chg < -0.02 and obv_chg > 0.05:
        return PatternHit("obv", 0.5, 0.55, f"bullish divergence (price {price_chg*100:.1f}% OBV {obv_chg*100:+.1f}%)")
    # Confirmation
    if price_chg > 0.02 and obv_chg > 0.05:
        return PatternHit("obv", 0.3, 0.4, "OBV confirms uptrend")
    if price_chg < -0.02 and obv_chg < -0.05:
        return PatternHit("obv", -0.3, 0.4, "OBV confirms downtrend")
    return PatternHit("obv", 0.0, 0.2, "OBV flat")


def vwap_signal(bars: pd.DataFrame) -> PatternHit:
    """Price vs running VWAP. For intraday, callers can pre-slice to today's bars."""
    if len(bars) < 20:
        return PatternHit("vwap", 0.0, 0.0, "insufficient data")
    v = vwap(bars)
    price = float(bars["close"].iloc[-1])
    vw = float(v.iloc[-1]) if pd.notna(v.iloc[-1]) else price
    if vw <= 0:
        return PatternHit("vwap", 0.0, 0.1, "vwap unavailable")
    diff = (price - vw) / vw
    if abs(diff) < 0.001:
        return PatternHit("vwap", 0.0, 0.3, f"at VWAP {vw:.2f}")
    sig = float(np.clip(diff * 30, -1.0, 1.0))
    side = "above" if diff > 0 else "below"
    return PatternHit("vwap", sig, 0.45, f"{side} VWAP by {abs(diff)*100:.2f}%", levels={"vwap": vw})


def keltner_signal(bars: pd.DataFrame) -> PatternHit:
    """Keltner channel breakout (ATR-based, less noisy than Bollinger)."""
    if len(bars) < 30:
        return PatternHit("keltner", 0.0, 0.0, "insufficient data")
    b = keltner(bars)
    price = float(bars["close"].iloc[-1])
    up = float(b.upper.iloc[-1]) if pd.notna(b.upper.iloc[-1]) else price
    lo = float(b.lower.iloc[-1]) if pd.notna(b.lower.iloc[-1]) else price
    if price > up:
        return PatternHit("keltner", 0.6, 0.6, f"breakout above upper Keltner {up:.2f}",
                          levels={"upper": up, "lower": lo})
    if price < lo:
        return PatternHit("keltner", -0.6, 0.6, f"breakdown below lower Keltner {lo:.2f}",
                          levels={"upper": up, "lower": lo})
    return PatternHit("keltner", 0.0, 0.2, "inside Keltner channel",
                      levels={"upper": up, "lower": lo})


# ---------------------------------------------------------------------------
# Aggregator → StrategySignal (consumed by ensemble)
# ---------------------------------------------------------------------------


PATTERN_DETECTORS = (
    fibonacci_retracement,
    support_resistance_breakout,
    double_top_bottom,
    candlestick_pattern,
    macd_crossover,
    pivot_points,
    head_and_shoulders,
    triangle,
    flag_pennant,
    wedge,
    cup_and_handle,
    ichimoku_signal,
    stochastic_signal,
    obv_trend,
    vwap_signal,
    keltner_signal,
)


def detect_all_patterns(bars: pd.DataFrame) -> list[PatternHit]:
    """Run all pattern detectors and return the raw hits."""
    out: list[PatternHit] = []
    for fn in PATTERN_DETECTORS:
        try:
            out.append(fn(bars))
        except Exception as exc:  # noqa: BLE001 — never let a single detector kill the run
            out.append(PatternHit(fn.__name__, 0.0, 0.0, f"error: {exc}"))
    return out


def pattern_signal(bars: pd.DataFrame) -> StrategySignal:
    """Combine all pattern detectors into a single StrategySignal.

    Confidence-weighted average of signals; overall confidence scales with
    the number of agreeing patterns.
    """
    hits = detect_all_patterns(bars)
    if not hits:
        return StrategySignal("patterns", 0.0, 0.0, "no detectors")

    valid_hits = [
        h for h in hits
        if np.isfinite(h.signal) and np.isfinite(h.confidence)
    ]
    if not valid_hits:
        return StrategySignal("patterns", 0.0, 0.0, "no valid pattern hits")

    weighted_sum = sum(h.signal * h.confidence for h in valid_hits)
    weight_total = sum(h.confidence for h in valid_hits)
    combined = weighted_sum / weight_total if weight_total > 0 else 0.0

    agree = sum(
        1
        for h in valid_hits
        if (h.signal > 0.1 and combined > 0) or (h.signal < -0.1 and combined < 0)
    )
    confidence = min(1.0, 0.25 * agree + 0.1)  # 1 agree -> 0.35, 3 -> 0.85

    # Build a compact reason string from the strongest valid hits.
    strong = sorted(valid_hits, key=lambda h: abs(h.signal) * h.confidence, reverse=True)[:3]
    reason = "; ".join(f"{h.name}:{h.reason}" for h in strong if abs(h.signal) > 0.05) or "no strong pattern"

    return StrategySignal("patterns", float(np.clip(combined, -1.0, 1.0)), confidence, reason)
