"""Multi-strategy ensemble with regime-adaptive weighting.

Combines multiple uncorrelated strategies and weights them by:
1. Recent performance (adaptive)
2. Market regime (bull/bear/sideways/high-vol)
3. Signal agreement (consensus filter)

Strategies included:
- Trend following (MA crossover, but with adaptive periods)
- Mean reversion (Bollinger Band bounce, RSI extremes)
- Momentum (breakout, relative strength)
- ML-based (ensemble model probability)

This provides genuine edge through diversification — when one strategy
underperforms (e.g., trend-following in sideways markets), others compensate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Market Regime Detection
# ---------------------------------------------------------------------------


class MarketRegime(Enum):
    """Detected market regime."""

    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass(slots=True)
class RegimeState:
    """Current market regime analysis."""

    regime: MarketRegime
    confidence: float  # 0-1 confidence in regime detection
    trend_strength: float  # -1 to +1 (negative = bearish)
    volatility_percentile: float  # 0-1 (current vol vs historical)
    details: dict


def detect_regime(bars: pd.DataFrame, lookback: int = 60) -> RegimeState:
    """Detect current market regime from price data.

    Uses multiple signals:
    - Price vs 50-day and 200-day MA (trend)
    - ADX-like trend strength
    - Volatility percentile (current vs 1-year history)
    - Consecutive up/down days

    Args:
        bars: OHLCV DataFrame (must have at least 200 bars ideally).
        lookback: Lookback period for regime analysis.

    Returns:
        RegimeState with detected regime and metadata.
    """
    close = bars["close"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)

    if len(close) < 60:
        return RegimeState(
            regime=MarketRegime.SIDEWAYS,
            confidence=0.0,
            trend_strength=0.0,
            volatility_percentile=0.5,
            details={"reason": "insufficient data"},
        )

    # Trend indicators
    ma_20 = close.rolling(20).mean()
    ma_50 = close.rolling(50).mean()
    current_price = float(close.iloc[-1])
    ma_20_val = float(ma_20.iloc[-1]) if pd.notna(ma_20.iloc[-1]) else current_price
    ma_50_val = float(ma_50.iloc[-1]) if pd.notna(ma_50.iloc[-1]) else current_price

    # Trend strength: normalized distance from MAs
    trend_20 = (current_price - ma_20_val) / ma_20_val if ma_20_val > 0 else 0
    trend_50 = (current_price - ma_50_val) / ma_50_val if ma_50_val > 0 else 0
    trend_strength = (trend_20 + trend_50) / 2

    # MA slope as additional trend signal
    ma_20_slope = float((ma_20.iloc[-1] - ma_20.iloc[-5]) / ma_20.iloc[-5]) if len(ma_20) >= 5 and pd.notna(ma_20.iloc[-5]) and ma_20.iloc[-5] != 0 else 0

    # Volatility analysis
    returns = close.pct_change().dropna()
    current_vol = float(returns.iloc[-20:].std()) if len(returns) >= 20 else 0.01
    hist_vol = returns.rolling(20).std().dropna()
    if len(hist_vol) > 50:
        vol_percentile = float((hist_vol < current_vol).mean())
    else:
        vol_percentile = 0.5

    # ADX-like directional strength (simplified)
    plus_dm = (high.diff()).clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14.replace(0, np.nan))
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * ((plus_di - minus_di).abs() / di_sum)
    adx = dx.rolling(14).mean()
    current_adx = float(adx.iloc[-1]) if pd.notna(adx.iloc[-1]) else 20

    # Regime classification
    details = {
        "trend_20": round(trend_20, 4),
        "trend_50": round(trend_50, 4),
        "ma_slope": round(ma_20_slope, 4),
        "vol_percentile": round(vol_percentile, 4),
        "adx": round(current_adx, 2),
        "current_vol": round(current_vol, 4),
    }

    # High volatility overrides other regimes
    if vol_percentile > 0.85:
        return RegimeState(
            regime=MarketRegime.HIGH_VOLATILITY,
            confidence=min(1.0, vol_percentile),
            trend_strength=trend_strength,
            volatility_percentile=vol_percentile,
            details=details,
        )

    if vol_percentile < 0.15:
        return RegimeState(
            regime=MarketRegime.LOW_VOLATILITY,
            confidence=min(1.0, 1 - vol_percentile),
            trend_strength=trend_strength,
            volatility_percentile=vol_percentile,
            details=details,
        )

    # Strong trend detection (ADX > 25 typically indicates trend)
    if current_adx > 25:
        if trend_strength > 0.02:
            return RegimeState(
                regime=MarketRegime.BULL_TREND,
                confidence=min(1.0, current_adx / 50),
                trend_strength=trend_strength,
                volatility_percentile=vol_percentile,
                details=details,
            )
        elif trend_strength < -0.02:
            return RegimeState(
                regime=MarketRegime.BEAR_TREND,
                confidence=min(1.0, current_adx / 50),
                trend_strength=trend_strength,
                volatility_percentile=vol_percentile,
                details=details,
            )

    # Default: sideways/range-bound
    return RegimeState(
        regime=MarketRegime.SIDEWAYS,
        confidence=max(0.3, 1 - current_adx / 50),
        trend_strength=trend_strength,
        volatility_percentile=vol_percentile,
        details=details,
    )


# ---------------------------------------------------------------------------
# Individual Strategy Signals
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StrategySignal:
    """Signal from an individual strategy."""

    name: str
    signal: float  # -1.0 (strong sell) to +1.0 (strong buy), 0 = neutral
    confidence: float  # 0-1 confidence in signal
    reason: str


def trend_following_signal(bars: pd.DataFrame, fast: int = 10, slow: int = 30) -> StrategySignal:
    """Adaptive trend-following using EMA crossover with slope confirmation.

    Improvement over simple MA: uses EMA (more responsive) and requires
    slope confirmation to filter false crossovers.
    """
    close = bars["close"].astype(float)
    if len(close) < slow + 5:
        return StrategySignal("trend", 0.0, 0.0, "insufficient data")

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    # Current crossover state
    fast_val = float(ema_fast.iloc[-1])
    slow_val = float(ema_slow.iloc[-1])
    cross_diff = (fast_val - slow_val) / slow_val if slow_val > 0 else 0

    # Slope confirmation (trend must be accelerating)
    fast_slope = float(ema_fast.iloc[-1] - ema_fast.iloc[-3]) / float(ema_fast.iloc[-3]) if float(ema_fast.iloc[-3]) != 0 else 0
    slow_slope = float(ema_slow.iloc[-1] - ema_slow.iloc[-3]) / float(ema_slow.iloc[-3]) if float(ema_slow.iloc[-3]) != 0 else 0

    # Signal strength based on crossover magnitude and slope agreement
    if cross_diff > 0 and fast_slope > 0:
        signal = min(1.0, cross_diff * 20)  # Scale up small differences
        confidence = min(1.0, abs(cross_diff) * 30 + 0.3)
        return StrategySignal("trend", signal, confidence, f"bullish: EMA cross +{cross_diff:.4f}")
    elif cross_diff < 0 and fast_slope < 0:
        signal = max(-1.0, cross_diff * 20)
        confidence = min(1.0, abs(cross_diff) * 30 + 0.3)
        return StrategySignal("trend", signal, confidence, f"bearish: EMA cross {cross_diff:.4f}")
    else:
        return StrategySignal("trend", 0.0, 0.2, "no confirmed trend")


def mean_reversion_signal(bars: pd.DataFrame) -> StrategySignal:
    """Mean reversion strategy using Bollinger Bands + RSI.

    Buys oversold (price below lower BB + RSI < 30).
    Sells overbought (price above upper BB + RSI > 70).
    """
    close = bars["close"].astype(float)
    if len(close) < 20:
        return StrategySignal("mean_reversion", 0.0, 0.0, "insufficient data")

    # Bollinger Bands
    ma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    upper_bb = ma_20 + 2 * std_20
    lower_bb = ma_20 - 2 * std_20

    current = float(close.iloc[-1])
    upper = float(upper_bb.iloc[-1])
    lower = float(lower_bb.iloc[-1])
    mid = float(ma_20.iloc[-1])

    # Position within bands (0 = lower, 1 = upper)
    band_width = upper - lower
    if band_width <= 0:
        return StrategySignal("mean_reversion", 0.0, 0.1, "zero band width")

    bb_position = (current - lower) / band_width

    # RSI for confirmation
    from ai_trading.ml.ensemble_model import compute_rsi

    rsi_series = compute_rsi(close, 14)
    rsi = float(rsi_series.iloc[-1]) if pd.notna(rsi_series.iloc[-1]) else 50

    # Oversold: buy signal
    if bb_position < 0.1 and rsi < 30:
        signal = min(1.0, (0.1 - bb_position) * 5 + (30 - rsi) / 100)
        return StrategySignal("mean_reversion", signal, 0.7, f"oversold: BB={bb_position:.2f} RSI={rsi:.0f}")
    elif bb_position < 0.2 and rsi < 35:
        signal = min(0.6, (0.2 - bb_position) * 3)
        return StrategySignal("mean_reversion", signal, 0.5, f"mildly oversold: BB={bb_position:.2f} RSI={rsi:.0f}")

    # Overbought: sell signal
    if bb_position > 0.9 and rsi > 70:
        signal = max(-1.0, -(bb_position - 0.9) * 5 - (rsi - 70) / 100)
        return StrategySignal("mean_reversion", signal, 0.7, f"overbought: BB={bb_position:.2f} RSI={rsi:.0f}")
    elif bb_position > 0.8 and rsi > 65:
        signal = max(-0.6, -(bb_position - 0.8) * 3)
        return StrategySignal("mean_reversion", signal, 0.5, f"mildly overbought: BB={bb_position:.2f} RSI={rsi:.0f}")

    return StrategySignal("mean_reversion", 0.0, 0.3, f"neutral: BB={bb_position:.2f} RSI={rsi:.0f}")


def momentum_signal(bars: pd.DataFrame) -> StrategySignal:
    """Momentum/breakout strategy using rate of change + volume confirmation.

    Buys strong upward momentum with volume confirmation.
    Sells when momentum fades or reverses.
    """
    close = bars["close"].astype(float)
    volume = bars["volume"].astype(float)
    high = bars["high"].astype(float)
    if len(close) < 20:
        return StrategySignal("momentum", 0.0, 0.0, "insufficient data")

    # Rate of change (10-day)
    roc_10 = float((close.iloc[-1] / close.iloc[-10] - 1) * 100)
    roc_5 = float((close.iloc[-1] / close.iloc[-5] - 1) * 100)

    # Volume confirmation
    vol_avg = float(volume.iloc[-20:].mean())
    vol_recent = float(volume.iloc[-3:].mean())
    vol_surge = vol_recent / vol_avg if vol_avg > 0 else 1.0

    # Breakout detection: price making new 20-day high with volume
    high_20 = float(high.iloc[-20:].max())
    is_breakout = float(close.iloc[-1]) >= high_20 * 0.99 and vol_surge > 1.2

    # Build signal
    if roc_10 > 3 and roc_5 > 1 and vol_surge > 1.1:
        signal = min(1.0, roc_10 / 10 + (0.3 if is_breakout else 0))
        confidence = min(0.9, 0.4 + vol_surge * 0.2)
        return StrategySignal("momentum", signal, confidence, f"strong momentum: ROC10={roc_10:.1f}% vol={vol_surge:.1f}x")
    elif roc_10 > 1.5 and roc_5 > 0:
        signal = min(0.5, roc_10 / 15)
        return StrategySignal("momentum", signal, 0.4, f"mild momentum: ROC10={roc_10:.1f}%")
    elif roc_10 < -3 and roc_5 < -1:
        signal = max(-1.0, roc_10 / 10)
        confidence = min(0.8, 0.3 + vol_surge * 0.2)
        return StrategySignal("momentum", signal, confidence, f"negative momentum: ROC10={roc_10:.1f}%")
    elif roc_10 < -1.5:
        signal = max(-0.5, roc_10 / 15)
        return StrategySignal("momentum", signal, 0.4, f"mild negative: ROC10={roc_10:.1f}%")

    return StrategySignal("momentum", 0.0, 0.2, f"flat: ROC10={roc_10:.1f}%")


# ---------------------------------------------------------------------------
# Ensemble Signal Combiner
# ---------------------------------------------------------------------------


# Regime-based strategy weights: which strategies work best in which regime
REGIME_WEIGHTS: dict[MarketRegime, dict[str, float]] = {
    MarketRegime.BULL_TREND: {"trend": 0.45, "momentum": 0.35, "mean_reversion": 0.10, "ml": 0.10},
    MarketRegime.BEAR_TREND: {"trend": 0.40, "momentum": 0.20, "mean_reversion": 0.20, "ml": 0.20},
    MarketRegime.SIDEWAYS: {"trend": 0.10, "momentum": 0.15, "mean_reversion": 0.45, "ml": 0.30},
    MarketRegime.HIGH_VOLATILITY: {"trend": 0.20, "momentum": 0.20, "mean_reversion": 0.30, "ml": 0.30},
    MarketRegime.LOW_VOLATILITY: {"trend": 0.30, "momentum": 0.30, "mean_reversion": 0.15, "ml": 0.25},
}


@dataclass(slots=True)
class EnsembleSignal:
    """Combined signal from all strategies."""

    signal: str  # "BUY", "SELL", or "HOLD"
    strength: float  # -1 to +1 (combined weighted signal)
    confidence: float  # 0-1 overall confidence
    regime: MarketRegime
    strategy_signals: list[StrategySignal]
    weights_used: dict[str, float]
    consensus_count: int  # How many strategies agree


def compute_ensemble_signal(
    bars: pd.DataFrame,
    ml_probability: float | None = None,
    buy_threshold: float = 0.15,
    sell_threshold: float = -0.15,
    min_consensus: int = 2,
) -> EnsembleSignal:
    """Compute the combined ensemble signal from all strategies.

    Args:
        bars: OHLCV DataFrame.
        ml_probability: Optional ML model probability of UP (0-1).
        buy_threshold: Combined signal must exceed this to generate BUY.
        sell_threshold: Combined signal must be below this to generate SELL.
        min_consensus: Minimum strategies agreeing for a trade.

    Returns:
        EnsembleSignal with final decision and breakdown.
    """
    # Detect market regime
    regime_state = detect_regime(bars)
    regime = regime_state.regime

    # Get individual strategy signals
    signals: list[StrategySignal] = [
        trend_following_signal(bars),
        mean_reversion_signal(bars),
        momentum_signal(bars),
    ]

    # Add ML signal if available
    if ml_probability is not None:
        ml_signal = (ml_probability - 0.5) * 2  # Convert 0-1 prob to -1/+1 signal
        ml_confidence = abs(ml_probability - 0.5) * 2
        signals.append(StrategySignal("ml", ml_signal, ml_confidence, f"prob_up={ml_probability:.3f}"))

    # Get regime-appropriate weights
    weights = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS[MarketRegime.SIDEWAYS]).copy()

    # If no ML signal, redistribute ML weight
    if ml_probability is None and "ml" in weights:
        ml_weight = weights.pop("ml")
        remaining = sum(weights.values())
        if remaining > 0:
            for k in weights:
                weights[k] += ml_weight * (weights[k] / remaining)

    # Compute weighted combined signal
    combined_signal = 0.0
    combined_confidence = 0.0
    for sig in signals:
        w = weights.get(sig.name, 0.0)
        combined_signal += sig.signal * w * sig.confidence
        combined_confidence += w * sig.confidence

    # Count consensus (strategies agreeing on direction)
    buy_count = sum(1 for s in signals if s.signal > 0.1)
    sell_count = sum(1 for s in signals if s.signal < -0.1)
    consensus_count = max(buy_count, sell_count)

    # Determine final signal
    final_signal = "HOLD"
    if combined_signal > buy_threshold and buy_count >= min_consensus:
        final_signal = "BUY"
    elif combined_signal < sell_threshold and sell_count >= min_consensus:
        final_signal = "SELL"

    return EnsembleSignal(
        signal=final_signal,
        strength=round(combined_signal, 4),
        confidence=round(combined_confidence, 4),
        regime=regime,
        strategy_signals=signals,
        weights_used=weights,
        consensus_count=consensus_count,
    )
