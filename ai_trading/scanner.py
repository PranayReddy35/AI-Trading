"""Live market scanner — scores and ranks symbols by buy opportunity strength.

Two scan modes
--------------
scan_live()  — uses Alpaca IEX 5-min intraday bars (real-time, market hours).
               Factors: VWAP deviation, intraday momentum, volume spike,
                        intraday RSI, price trend slope.
scan()       — uses yfinance end-of-day bars (works 24/7, good for pre-market prep).
               Factors: MA crossover, 5-day momentum, RSI, volume surge, trend,
                        relative strength vs SPY, Bollinger squeeze, meta-label P(win).

Enhancements:
  - Liquidity gate (--min-price, --min-dollar-vol) drops illiquid junk.
  - Quality gates (SPY 200DMA, VIX, earnings blackout, volume confirm) flag picks
    the bot would reject; --no-filters to disable.
  - Meta-label model loaded from BOT_META_MODEL_PATH (if present) outputs P(win).
  - ATR-based actionable entry / stop / target on top picks (2:1 R:R).
  - Correlation dedup (--max-per-cluster) avoids surfacing 5 mega-tech names.
  - Parallel yfinance fetch with retry/backoff for faster EOD scans.

Usage:
    python -m ai_trading.scanner               # live mode if market open, else EOD
    python -m ai_trading.scanner --mode live
    python -m ai_trading.scanner --mode eod
    python -m ai_trading.scanner --top 10 --symbols SPY,QQQ,NVDA
    python -m ai_trading.scanner --no-filters --no-meta --no-dedup
"""
from __future__ import annotations

import argparse
import os
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

from ai_trading.env import load_dotenv
from ai_trading.time_utils import format_local_now


load_dotenv()


# ── Scoring weights (EOD mode) ────────────────────────────────────────────────
W_MA        = 0.20
W_MOMENTUM  = 0.15
W_RSI       = 0.12
W_VOLUME    = 0.12
W_TREND     = 0.08
W_DIP       = 0.13   # dip-buy composite (RSI oversold + pullback from high + above long MA)
W_RS        = 0.10   # relative strength vs SPY (true alpha)
W_SQUEEZE   = 0.05   # Bollinger band squeeze (volatility compression — breakout setup)
W_META      = 0.05   # meta-label P(win) from ML model

# ── Scoring weights (LIVE intraday mode) ─────────────────────────────────────
LW_VWAP      = 0.22   # price vs VWAP (above = bullish)
LW_MOMENTUM  = 0.18   # intraday price return from open
LW_VOLUME    = 0.18   # volume vs avg intraday volume
LW_RSI       = 0.12   # intraday RSI (oversold bounce)
LW_TREND     = 0.08   # recent bar-by-bar slope
LW_DIP       = 0.07   # intraday dip (below VWAP + RSI oversold)
LW_RS        = 0.10   # daily-bars relative strength vs SPY
LW_META      = 0.05   # meta-label P(win) from daily-bars model

# ── Scan performance knobs ───────────────────────────────────────────────────
_YF_BATCH_SIZE = max(50, int(os.getenv("BOT_SCAN_YF_BATCH_SIZE", "200") or 200))
_YF_MAX_WORKERS = max(2, int(os.getenv("BOT_SCAN_YF_MAX_WORKERS", "12") or 12))
_SCAN_FETCH_CACHE_TTL_SEC = max(0, int(os.getenv("BOT_SCAN_FETCH_CACHE_TTL_SEC", "90") or 90))
_SCAN_FETCH_CACHE: dict[tuple, tuple[float, object]] = {}
_EOD_DATA_SOURCE = str(os.getenv("BOT_SCAN_EOD_SOURCE", "auto") or "auto").strip().lower()


@dataclass
class ScanResult:
    symbol: str
    score: float              # 0–100 composite buy score
    signal: str               # BUY / WATCH / NEUTRAL
    close: float
    change_pct: float         # % change from open (live) or prior close (EOD)
    momentum_5d: float        # intraday return from open (live) or 5-day return (EOD)
    rsi: float
    volume_surge: float       # bar vol / avg vol
    ma_gap_pct: float         # VWAP gap % (live) or MA gap % (EOD)
    trend_consistency: float  # % of recent bars trending up
    reason: str               # ranked driver summary
    upside_pct: float = 0.0   # estimated short-term upside % (score-derived)
    top_driver: str = ""      # single strongest bullish factor label
    mode: str = "live"        # "live" or "eod"
    data_source: str = ""     # primary bar source (alpaca/yfinance)
    as_of: str = field(default_factory=lambda: format_local_now("%I:%M:%S %p %Z"))
    # ── enhancements ─────────────────────────────────────────────────────────
    rel_strength_pct: float = 0.0      # outperformance vs SPY over lookback (%)
    bb_squeeze: float = 0.0            # 0..1 — higher = tighter Bollinger compression
    meta_prob: float | None = None     # P(hit +1R before -1R) from meta-label model
    avg_dollar_vol_m: float = 0.0      # 20d avg dollar volume in millions
    entry: float = 0.0                 # suggested entry (last close)
    stop: float = 0.0                  # ATR-based stop
    target: float = 0.0                # ATR-based target (2:1 R:R default)
    risk_pct: float = 0.0              # (entry-stop)/entry × 100
    reward_pct: float = 0.0            # (target-entry)/entry × 100
    quality_flags: str = ""            # "ok" or comma-separated rejection reasons
    quality_pass: bool = True          # would the bot accept this trade?


# ── Shared helpers ────────────────────────────────────────────────────────────

def _build_reason(
    score: float,
    rsi: float,
    momentum: float,
    surge: float,
    gap_pct: float,
    trend: float,
    dip: float,
    mode: str = "eod",  # "eod" or "live"
) -> tuple[str, float, str]:
    """Return (reason_str, upside_pct, top_driver) ranked by factor strength."""
    # Score each driver by its contribution magnitude
    factors: list[tuple[float, str]] = []

    # Momentum / return
    if abs(momentum) >= 0.5:
        label = ("intraday" if mode == "live" else "5d") + f" momentum {momentum:+.1f}%"
        factors.append((abs(momentum), label))

    # Oversold RSI = mean-reversion upside
    if rsi <= 30:
        factors.append((70 - rsi, f"RSI deeply oversold ({rsi:.0f}) — bounce potential"))
    elif rsi <= 45:
        factors.append((55 - rsi, f"RSI oversold ({rsi:.0f})"))

    # Volume conviction
    if surge >= 3.0:
        factors.append((surge * 1.5, f"heavy vol surge {surge:.1f}x — institutional interest"))
    elif surge >= 1.5:
        factors.append((surge, f"vol elevated {surge:.1f}x"))

    # Price vs MA/VWAP
    ref = "VWAP" if mode == "live" else "MA"
    if gap_pct >= 1.5:
        factors.append((gap_pct, f"extended above {ref} +{gap_pct:.1f}% — strong trend"))
    elif gap_pct >= 0.3:
        factors.append((gap_pct, f"above {ref} +{gap_pct:.1f}%"))
    elif gap_pct <= -1.0:
        factors.append((abs(gap_pct) * 0.8, f"below {ref} {gap_pct:.1f}% — reversion setup"))

    # Trend consistency
    if trend >= 75:
        factors.append((trend / 20, f"strong trend ({trend:.0f}% up-bars)"))
    elif trend >= 55:
        factors.append((trend / 30, f"trend {trend:.0f}% up-bars"))

    # Dip setup
    if dip >= 0.5:
        factors.append((dip * 10, f"🎯 dip setup (score {dip:.2f}) — RSI + pullback aligned"))
    elif dip >= 0.3:
        factors.append((dip * 8, f"dip setup forming ({dip:.2f})"))

    # Sort strongest first
    factors.sort(key=lambda x: x[0], reverse=True)

    if factors:
        top_driver = factors[0][1]
        reason_parts = [f[1] for f in factors[:4]]  # top 4 drivers
        reason = " · ".join(reason_parts)
    else:
        top_driver = "no strong catalyst"
        reason = "no strong catalyst"

    # Upside estimate: score-based heuristic
    # BUY (score≥65): project 2–8% upside; WATCH: 0.5–2%; NEUTRAL: ~0%
    if score >= 65:
        upside = round(2.0 + (score - 65) / 35 * 6.0, 1)   # 2–8%
    elif score >= 45:
        upside = round(0.5 + (score - 45) / 20 * 1.5, 1)   # 0.5–2%
    else:
        upside = 0.0

    # Boost upside for dip setups and heavily oversold RSI
    if dip >= 0.4:
        upside = round(upside * 1.3, 1)
    if rsi <= 30:
        upside = round(upside * 1.2, 1)

    return reason, upside, top_driver


def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff().dropna()
    if len(delta) < period:
        return 50.0
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta).clip(lower=0).rolling(period).mean()
    rs = gain.iloc[-1] / (loss.iloc[-1] or 1e-9)
    return float(100 - (100 / (1 + rs)))


def _score_symbol(bars: pd.DataFrame, fast_ma: int = 5, slow_ma: int = 20) -> dict:
    """Compute scoring factors for one symbol. Returns dict of raw factor values."""
    close = bars["close"].astype(float)
    volume = bars["volume"].astype(float) if "volume" in bars.columns else pd.Series([1.0] * len(close), index=close.index)

    if len(close) < slow_ma + 5:
        return {}

    # 1. MA gap
    fast = close.rolling(fast_ma).mean().iloc[-1]
    slow = close.rolling(slow_ma).mean().iloc[-1]
    ma_gap_pct = (fast - slow) / slow * 100 if slow else 0.0

    # 2. Momentum (5-day)
    if len(close) >= 6:
        momentum_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100
    else:
        momentum_5d = 0.0

    # 3. RSI
    rsi_val = _rsi(close)

    # 4. Volume surge
    avg_vol = volume.iloc[-21:-1].mean() if len(volume) >= 22 else volume.mean()
    today_vol = volume.iloc[-1]
    vol_surge = float(today_vol / avg_vol) if avg_vol > 0 else 1.0

    # 5. Trend consistency (% of last 10 days with positive return)
    last10 = close.iloc[-11:].pct_change().dropna()
    trend = float((last10 > 0).mean()) if len(last10) >= 5 else 0.5

    # 6. Dip-buy composite score
    dip = _dip_score(close, rsi_val, rsi_threshold=40.0, drop_pct_threshold=3.0, lookback=20, long_ma_period=50)

    # 1-day change
    change_pct = float(close.pct_change().iloc[-1] * 100)

    return {
        "close": float(close.iloc[-1]),
        "change_pct": round(change_pct, 2),
        "momentum_5d": round(momentum_5d, 2),
        "rsi": round(rsi_val, 1),
        "volume_surge": round(vol_surge, 2),
        "ma_gap_pct": round(ma_gap_pct, 2),
        "trend_consistency": round(trend * 100, 1),
        "dip_score": dip,
    }


def _normalise(series: pd.Series, invert: bool = False) -> pd.Series:
    """Min-max normalise to [0, 1]. Invert if lower = better (e.g. RSI for oversold)."""
    s = pd.to_numeric(series, errors="coerce")
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series([0.5] * len(series), index=series.index)
    norm = (s - mn) / (mx - mn)
    if invert:
        norm = 1 - norm
    return norm.fillna(0.5)


def _sf(val, default: float = 0.0) -> float:
    """Safe float: coerce pd.NA / None / NaN / strings → default."""
    try:
        if val is None or pd.isna(val):
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def _sf_opt(val) -> float | None:
    """Safe optional float: pd.NA / None / NaN → None, else float."""
    try:
        if val is None or pd.isna(val):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def _dip_score(
    close: pd.Series,
    rsi_val: float,
    rsi_threshold: float = 40.0,
    drop_pct_threshold: float = 3.0,
    lookback: int = 20,
    long_ma_period: int = 50,
) -> float:
    """Return a 0–1 score for how good a dip-buy setup is.

    0 = no dip / overbought, 1 = perfect dip setup.
    Factors: RSI oversold, pullback from recent high, still above long MA.
    """
    if len(close) < max(lookback, long_ma_period):
        return 0.0

    current = float(close.iloc[-1])
    recent_high = float(close.iloc[-lookback:].max())
    drop_pct = (recent_high - current) / recent_high * 100.0 if recent_high > 0 else 0.0

    long_ma = float(close.rolling(long_ma_period).mean().iloc[-1])
    above_long_ma = current >= long_ma if pd.notna(long_ma) else False

    # Partial scores — each component contributes
    rsi_score    = max(0.0, (rsi_threshold - rsi_val) / rsi_threshold)   # peaks when RSI→0
    drop_score   = min(1.0, drop_pct / (drop_pct_threshold * 3))          # caps at 3x threshold
    trend_score  = 1.0 if above_long_ma else 0.2                           # heavy penalty for below MA

    return round(rsi_score * 0.45 + drop_score * 0.35 + trend_score * 0.20, 4)


# ── New helpers: liquidity, RS, squeeze, ATR levels, meta, dedup ──────────────

def _avg_dollar_volume(bars: pd.DataFrame, lookback: int = 20) -> float:
    """Average daily dollar volume over the lookback window (in dollars)."""
    if "volume" not in bars or len(bars) < 2:
        return 0.0
    n = min(lookback, len(bars))
    px = bars["close"].astype(float).iloc[-n:]
    vol = bars["volume"].astype(float).iloc[-n:]
    return float((px * vol).mean())


def _liquidity_ok(
    bars: pd.DataFrame,
    *,
    min_price: float = 5.0,
    min_dollar_vol: float = 5_000_000.0,
) -> tuple[bool, str, float]:
    """Liquidity gate. Returns (passed, reason, avg_dollar_volume)."""
    if bars.empty:
        return False, "no bars", 0.0
    price = float(bars["close"].astype(float).iloc[-1])
    adv = _avg_dollar_volume(bars)
    if price < min_price:
        return False, f"price ${price:.2f} < ${min_price:.2f}", adv
    if adv < min_dollar_vol:
        return False, f"avg $vol ${adv/1e6:.1f}M < ${min_dollar_vol/1e6:.1f}M", adv
    return True, "liquid", adv


def _rel_strength_pct(
    sym_close: pd.Series,
    spy_close: pd.Series,
    lookback: int = 20,
) -> float:
    """% outperformance of sym vs SPY over the last `lookback` bars."""
    if len(sym_close) < lookback + 1 or len(spy_close) < lookback + 1:
        return 0.0
    sym_ret = float(sym_close.iloc[-1] / sym_close.iloc[-lookback - 1] - 1.0)
    spy_ret = float(spy_close.iloc[-1] / spy_close.iloc[-lookback - 1] - 1.0)
    return round((sym_ret - spy_ret) * 100, 2)


def _bb_squeeze_score(close: pd.Series, period: int = 20, lookback: int = 120) -> float:
    """0..1 score — higher = current Bollinger bandwidth is unusually tight.

    Bandwidth = (upper - lower) / middle, where bands are SMA ± 2·std.
    We compare current bandwidth's percentile vs the trailing `lookback` window;
    lower percentile = tighter squeeze = higher score (1 - percentile).
    """
    if len(close) < lookback:
        return 0.0
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    bandwidth = ((sma + 2 * std) - (sma - 2 * std)) / sma.replace(0, pd.NA)
    bw = bandwidth.dropna().iloc[-lookback:]
    if bw.empty or pd.isna(bw.iloc[-1]):
        return 0.0
    current = float(bw.iloc[-1])
    pct = float((bw <= current).mean())  # percentile of current value
    return round(max(0.0, min(1.0, 1.0 - pct)), 4)


def _atr_levels(
    bars: pd.DataFrame,
    *,
    atr_period: int = 14,
    atr_mult_stop: float = 2.0,
    risk_reward: float = 2.0,
) -> dict:
    """ATR-based entry/stop/target for a long setup. Returns dict or {}."""
    if len(bars) < atr_period + 1:
        return {}
    try:
        from ai_trading.strategy.indicators import atr as _atr
    except Exception:
        return {}
    a = _atr(bars, period=atr_period)
    if a.empty or pd.isna(a.iloc[-1]):
        return {}
    entry = float(bars["close"].astype(float).iloc[-1])
    atr_val = float(a.iloc[-1])
    if atr_val <= 0 or entry <= 0:
        return {}
    stop = max(0.01, entry - atr_mult_stop * atr_val)
    target = entry + (entry - stop) * risk_reward
    risk_pct = round((entry - stop) / entry * 100, 2)
    reward_pct = round((target - entry) / entry * 100, 2)
    return {
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "risk_pct": risk_pct,
        "reward_pct": reward_pct,
    }


_META_MODEL_CACHE: dict[str, object | None] = {"model": None, "path": None, "tried": False}


def _load_meta_model():
    """Lazy-load and cache the meta-label MetaModel. Returns None if unavailable."""
    if _META_MODEL_CACHE["tried"]:
        return _META_MODEL_CACHE["model"]
    _META_MODEL_CACHE["tried"] = True
    path = os.getenv("BOT_META_MODEL_PATH", "models/meta_label.joblib")
    if not os.path.exists(path):
        return None
    try:
        from ai_trading.ml.meta_label import MetaModel
        mm = MetaModel.load(path)
        _META_MODEL_CACHE["model"] = mm
        _META_MODEL_CACHE["path"] = path
        return mm
    except Exception as exc:
        print(f"meta-label load failed: {exc}")
        return None


def _meta_probability(bars: pd.DataFrame) -> float | None:
    """Run meta-label inference on the latest bar. Returns None if no model/data."""
    mm = _load_meta_model()
    if mm is None or len(bars) < 50:
        return None
    try:
        return round(float(mm.predict_proba_win(bars)), 4)
    except Exception:
        return None


def _quality_gates(
    symbol: str,
    bars: pd.DataFrame,
    *,
    check_spy_trend: bool = True,
    check_vix: bool = True,
    check_earnings: bool = True,
    earnings_blackout_days: int = 2,
    check_volume: bool = True,
    volume_min_ratio: float = 0.8,
    spy_trend_result: tuple[bool, str] | None = None,
    vix_result: tuple[float, str] | None = None,
    earnings_result=None,
) -> tuple[bool, str]:
    """Apply the same macro/quality filters the bot uses. Returns (pass, flags_str)."""
    flags: list[str] = []
    passed = True
    try:
        from ai_trading.strategy import market_filters as mf
    except Exception:
        return True, "filters unavailable"

    if check_spy_trend:
        ok, _ = spy_trend_result if spy_trend_result is not None else mf.spy_trend_ok()
        if not ok:
            flags.append("spy_trend")
            passed = False
    if check_vix:
        mult, _ = vix_result if vix_result is not None else mf.vix_size_multiplier()
        if mult <= 0:
            flags.append("vix_panic")
            passed = False
        elif mult < 0.5:
            flags.append(f"vix×{mult:.1f}")
    if check_earnings and earnings_blackout_days > 0:
        ec = earnings_result if earnings_result is not None else mf.in_earnings_blackout(symbol, blackout_days=earnings_blackout_days)
        if ec.blocked:
            flags.append("earnings")
            passed = False
    if check_volume and not bars.empty:
        ok, _ = mf.volume_confirms(bars, min_ratio=volume_min_ratio)
        if not ok:
            flags.append("low_vol")
            passed = False
    return passed, ",".join(flags) if flags else "ok"


def _quality_gate_context(
    candidate_symbols: list[str],
    *,
    check_spy_trend: bool = True,
    check_vix: bool = True,
    check_earnings: bool = True,
    earnings_blackout_days: int = 2,
) -> dict[str, object]:
    """Precompute scan-wide quality data so candidate rendering avoids repeat calls."""
    try:
        from ai_trading.strategy import market_filters as mf
    except Exception:
        return {"spy": None, "vix": None, "earnings": {}}

    ctx: dict[str, object] = {"spy": None, "vix": None, "earnings": {}}
    if check_spy_trend:
        try:
            ctx["spy"] = mf.spy_trend_ok()
        except Exception:
            ctx["spy"] = (True, "SPY lookup failed; allow")
    if check_vix:
        try:
            ctx["vix"] = mf.vix_size_multiplier()
        except Exception:
            ctx["vix"] = (1.0, "VIX lookup failed; full size")
    if check_earnings and earnings_blackout_days > 0:
        try:
            ctx["earnings"] = mf.earnings_blackout_map(
                tuple(candidate_symbols),
                blackout_days=earnings_blackout_days,
                max_workers=max(1, min(8, _YF_MAX_WORKERS)),
            )
        except Exception:
            ctx["earnings"] = {}
    return ctx


def _diversify(
    results: list[ScanResult],
    bars_by_symbol: dict[str, pd.DataFrame],
    *,
    max_correlation: float = 0.85,
    lookback: int = 60,
    keep_top: int | None = None,
) -> list[ScanResult]:
    """Greedy dedup: walk results in score order, skip any symbol whose correlation
    with an already-kept symbol exceeds `max_correlation`.
    """
    if not results or max_correlation <= 0:
        return results[:keep_top] if keep_top else results
    try:
        from ai_trading.risk.correlation import compute_pairwise_correlations
    except Exception:
        return results[:keep_top] if keep_top else results

    relevant = {r.symbol: bars_by_symbol[r.symbol] for r in results if r.symbol in bars_by_symbol}
    if len(relevant) < 2:
        return results[:keep_top] if keep_top else results
    corr = compute_pairwise_correlations(relevant, lookback=lookback)
    if corr.empty:
        return results[:keep_top] if keep_top else results

    kept: list[ScanResult] = []
    for r in results:
        if r.symbol not in corr.columns:
            kept.append(r)
        else:
            too_correlated = False
            for k in kept:
                if k.symbol in corr.columns:
                    c = corr.loc[r.symbol, k.symbol]
                    if pd.notna(c) and abs(float(c)) >= max_correlation:
                        too_correlated = True
                        break
            if not too_correlated:
                kept.append(r)
        if keep_top and len(kept) >= keep_top:
            break
    return kept


def _yf_download_with_retry(
    chunk: list[str],
    *,
    period: str,
    interval: str = "1d",
    max_retries: int = 3,
    backoff: float = 1.5,
) -> pd.DataFrame:
    """yfinance batch download with retry/backoff. Returns empty DataFrame on failure."""
    delay = backoff
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            df = yf.download(
                chunk,
                period=period,
                interval=interval,
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if df is not None and not df.empty:
                return df
        except Exception as exc:
            last_exc = exc
        if attempt < max_retries - 1:
            _time.sleep(delay)
            delay *= backoff
    if last_exc is not None:
        print(f"yfinance chunk retry exhausted ({chunk[0]}..{chunk[-1]}): {last_exc}")
    return pd.DataFrame()


def _parallel_yf_download(
    symbols: list[str],
    *,
    period: str,
    batch_size: int = _YF_BATCH_SIZE,
    max_workers: int = _YF_MAX_WORKERS,
) -> pd.DataFrame:
    """Download many tickers in parallel chunks. Concatenates results."""
    if not symbols:
        return pd.DataFrame()
    chunks = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_yf_download_with_retry, c, period=period) for c in chunks]
        for fut in as_completed(futures):
            df = fut.result()
            if df is not None and not df.empty:
                frames.append(df)
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def _scan_cache_get(key: tuple):
    if _SCAN_FETCH_CACHE_TTL_SEC <= 0:
        return None
    entry = _SCAN_FETCH_CACHE.get(key)
    if entry is None:
        return None
    ts, payload = entry
    if _time.time() - ts > _SCAN_FETCH_CACHE_TTL_SEC:
        _SCAN_FETCH_CACHE.pop(key, None)
        return None
    return payload


def _scan_cache_set(key: tuple, payload: object) -> None:
    if _SCAN_FETCH_CACHE_TTL_SEC <= 0:
        return
    _SCAN_FETCH_CACHE[key] = (_time.time(), payload)


def _norm_ohlcv(bars: pd.DataFrame) -> pd.DataFrame:
    if bars is None or bars.empty:
        return pd.DataFrame()
    b = bars.copy()
    b.columns = [str(c).lower() for c in b.columns]
    if "close" in b.columns:
        b = b.dropna(subset=["close"])
    return b.sort_index()


def _fetch_eod_bars(
    symbols: list[str],
    *,
    lookback_days: int,
) -> tuple[dict[str, pd.DataFrame], str]:
    """Fetch daily bars for many symbols, preferring Alpaca batched requests.

    Returns ({SYMBOL: bars}, source_label).
    """
    clean = [s.strip().upper() for s in symbols if s and s.strip()]
    if not clean:
        return {}, "none"

    cache_key = ("eod_bars", tuple(clean), int(lookback_days), _EOD_DATA_SOURCE)
    cached = _scan_cache_get(cache_key)
    if isinstance(cached, tuple) and len(cached) == 2:
        bars_map, source = cached
        if isinstance(bars_map, dict):
            return bars_map, str(source)

    source_pref = _EOD_DATA_SOURCE if _EOD_DATA_SOURCE in {"auto", "alpaca", "yfinance"} else "auto"
    if source_pref in {"auto", "alpaca"}:
        api_key = os.getenv("APCA_API_KEY_ID", "")
        api_secret = os.getenv("APCA_API_SECRET_KEY", "")
        if api_key and api_secret:
            try:
                from ai_trading.data.market_data import AlpacaMarketData

                data_feed = str(os.getenv("BOT_SCAN_EOD_FEED", "iex") or "iex")
                md = AlpacaMarketData(api_key, api_secret, cache_ttl_sec=60, data_feed=data_feed)
                bars_map: dict[str, pd.DataFrame] = {}
                for i in range(0, len(clean), _ALPACA_BATCH_SIZE):
                    chunk = clean[i:i + _ALPACA_BATCH_SIZE]
                    chunk_bars = md.get_multi_symbol_bars(chunk, lookback_days=lookback_days, timeframe="1Day")
                    for sym, bars in chunk_bars.items():
                        norm = _norm_ohlcv(bars)
                        if not norm.empty:
                            bars_map[sym.upper()] = norm
                if bars_map:
                    _scan_cache_set(cache_key, (bars_map, "alpaca"))
                    return bars_map, "alpaca"
            except Exception as exc:
                if source_pref == "alpaca":
                    print(f"Alpaca EOD fetch failed; falling back to yfinance: {exc}")

    period = f"{max(lookback_days, 120)}d"
    raw = _parallel_yf_download(clean, period=period)
    if raw.empty:
        return {}, "yfinance"
    bars_map = {sym: _extract_per_symbol(raw, sym, len(clean)) for sym in clean}
    bars_map = {sym: b for sym, b in bars_map.items() if b is not None and not b.empty}
    _scan_cache_set(cache_key, (bars_map, "yfinance"))
    return bars_map, "yfinance"


def _extract_per_symbol(raw: pd.DataFrame, sym: str, n_symbols: int) -> pd.DataFrame:
    """Slice a per-symbol OHLCV frame out of yfinance batch output."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    if n_symbols == 1:
        bars = raw.copy()
    else:
        try:
            level0 = raw.columns.get_level_values(0)
        except Exception:
            return pd.DataFrame()
        if sym not in level0:
            return pd.DataFrame()
        bars = raw[sym].copy()
    bars.columns = [c.lower() for c in bars.columns]
    return bars.dropna(subset=["close"]) if "close" in bars else pd.DataFrame()


def scan(
    symbols: list[str],
    fast_ma: int = 5,
    slow_ma: int = 20,
    lookback_days: int = 60,
    top_n: int = 10,
    *,
    min_price: float = 5.0,
    min_dollar_vol: float = 5_000_000.0,
    apply_filters: bool = True,
    earnings_blackout_days: int = 2,
    use_meta: bool = True,
    dedup: bool = True,
    max_correlation: float = 0.85,
) -> list[ScanResult]:
    """Fetch bars for all symbols, score each, and return top_n ranked results.

    Uses yfinance so no Alpaca subscription needed. Enhancements: liquidity gate,
    RS vs SPY, Bollinger squeeze, meta-label P(win), ATR levels, quality gates,
    correlation dedup.
    """
    symbols = [s.strip().upper() for s in symbols if s and s.strip()]
    if not symbols:
        return []

    # Ensure SPY is fetched once for relative-strength baseline
    fetch_symbols = list(dict.fromkeys(symbols + ["SPY"]))
    bars_map, _bars_source = _fetch_eod_bars(fetch_symbols, lookback_days=max(lookback_days, 120))
    if not bars_map:
        return []

    spy_bars = bars_map.get("SPY", pd.DataFrame())
    spy_close = spy_bars["close"].astype(float) if not spy_bars.empty else pd.Series(dtype=float)

    rows: list[dict] = []
    bars_by_symbol: dict[str, pd.DataFrame] = {}

    for sym in symbols:
        try:
            bars = bars_map.get(sym, pd.DataFrame())
            if bars.empty or len(bars) < slow_ma + 5:
                continue

            # Liquidity gate
            liq_ok, _liq_reason, adv = _liquidity_ok(
                bars, min_price=min_price, min_dollar_vol=min_dollar_vol,
            )
            if not liq_ok:
                continue

            factors = _score_symbol(bars, fast_ma, slow_ma)
            if not factors:
                continue
            factors["symbol"] = sym
            factors["avg_dollar_vol_m"] = round(adv / 1e6, 2)
            rows.append(factors)
            bars_by_symbol[sym] = bars
        except Exception:
            continue

    if not rows:
        return []

    df = pd.DataFrame(rows).set_index("symbol")

    # Normalise each factor column
    df["n_ma"]       = _normalise(df["ma_gap_pct"])            # higher gap = bullish
    df["n_momentum"] = _normalise(df["momentum_5d"])           # higher momentum = better
    df["n_rsi"]      = _normalise(df["rsi"], invert=True)      # lower RSI = more oversold = buy opp
    df["n_volume"]   = _normalise(df["volume_surge"])          # higher vol surge = conviction
    df["n_trend"]    = _normalise(df["trend_consistency"])     # more up-days = better
    df["n_dip"]      = _normalise(df["dip_score"])             # higher dip score = better setup
    # Defer heavy enrichment (RS, squeeze, meta) to the top candidates only.
    df["base_score"] = (
        df["n_ma"]       * W_MA       +
        df["n_momentum"] * W_MOMENTUM +
        df["n_rsi"]      * W_RSI      +
        df["n_volume"]   * W_VOLUME   +
        df["n_trend"]    * W_TREND    +
        df["n_dip"]      * W_DIP
    ) * 100
    df["base_score"] = pd.to_numeric(df["base_score"], errors="coerce").fillna(0.0)

    df["rs_pct"] = 0.0
    df["squeeze"] = 0.0
    df["meta_prob"] = pd.NA
    candidate_n = min(len(df), max(top_n * 5, 30))
    candidate_syms = df["base_score"].sort_values(ascending=False).head(candidate_n).index.tolist()
    for sym in candidate_syms:
        bars = bars_by_symbol.get(sym)
        if bars is None or bars.empty:
            continue
        close = bars["close"].astype(float)
        df.at[sym, "rs_pct"] = _rel_strength_pct(close, spy_close, lookback=20) if not spy_close.empty else 0.0
        df.at[sym, "squeeze"] = _bb_squeeze_score(close)
        if use_meta:
            mp = _meta_probability(bars)
            if mp is not None:
                df.at[sym, "meta_prob"] = mp

    df["n_rs"] = _normalise(df["rs_pct"])
    df["n_squeeze"] = _normalise(df["squeeze"])
    df["n_meta"] = pd.to_numeric(df["meta_prob"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    df["score"] = (
        df["base_score"] +
        (df["n_rs"] * W_RS + df["n_squeeze"] * W_SQUEEZE + df["n_meta"] * W_META) * 100
    )
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    df = df.sort_values("score", ascending=False)

    results: list[ScanResult] = []
    # Take 3× top_n initially so dedup has room to filter
    initial_n = top_n * 3 if dedup else top_n
    candidate_symbols = [str(sym) for sym in df.head(initial_n).index]
    check_earnings = len(symbols) < 1000
    quality_ctx = (
        _quality_gate_context(
            candidate_symbols,
            check_earnings=check_earnings,
            earnings_blackout_days=earnings_blackout_days,
        )
        if apply_filters else {"spy": None, "vix": None, "earnings": {}}
    )
    quality_earnings = quality_ctx.get("earnings", {})
    if not isinstance(quality_earnings, dict):
        quality_earnings = {}
    for sym, row in df.head(initial_n).iterrows():
        score = _sf(row.get("score"))
        rsi = _sf(row.get("rsi"))
        ma_gap = _sf(row.get("ma_gap_pct"))
        momentum = _sf(row.get("momentum_5d"))
        surge = _sf(row.get("volume_surge"), 1.0)
        dip = _sf(row.get("dip_score"))
        rs_pct = _sf(row.get("rs_pct"))
        squeeze = _sf(row.get("squeeze"))
        meta_prob = _sf_opt(row.get("meta_prob"))

        if score >= 65:
            signal = "BUY"
        elif score >= 45:
            signal = "WATCH"
        else:
            signal = "NEUTRAL"

        reason, upside, top_driver = _build_reason(
            score=score, rsi=rsi, momentum=momentum, surge=surge,
            gap_pct=ma_gap, trend=_sf(row.get("trend_consistency")),
            dip=dip, mode="eod",
        )
        # Annotate reason with RS/squeeze/meta if strong
        extras: list[str] = []
        if rs_pct >= 2.0:
            extras.append(f"RS vs SPY +{rs_pct:.1f}%")
        elif rs_pct <= -2.0:
            extras.append(f"RS vs SPY {rs_pct:.1f}%")
        if squeeze >= 0.85:
            extras.append(f"BB squeeze {squeeze:.2f} (breakout setup)")
        if meta_prob is not None and meta_prob >= 0.60:
            extras.append(f"meta P(win)={meta_prob:.2f}")
        if extras:
            reason = " · ".join(extras + [reason]) if reason else " · ".join(extras)

        levels = _atr_levels(bars_by_symbol[sym])
        qpass, qflags = (True, "ok")
        if apply_filters:
            qpass, qflags = _quality_gates(
                sym, bars_by_symbol[sym],
                check_earnings=check_earnings,
                earnings_blackout_days=earnings_blackout_days,
                spy_trend_result=quality_ctx.get("spy"),
                vix_result=quality_ctx.get("vix"),
                earnings_result=quality_earnings.get(str(sym)),
            )

        results.append(ScanResult(
            symbol=sym,
            score=round(score, 1),
            signal=signal,
            close=_sf(row.get("close")),
            change_pct=_sf(row.get("change_pct")),
            momentum_5d=momentum,
            rsi=rsi,
            volume_surge=surge,
            ma_gap_pct=ma_gap,
            trend_consistency=_sf(row.get("trend_consistency")),
            reason=reason,
            upside_pct=upside,
            top_driver=top_driver,
            mode="eod",
            data_source=_bars_source,
            rel_strength_pct=rs_pct,
            bb_squeeze=squeeze,
            meta_prob=meta_prob,
            avg_dollar_vol_m=_sf(row.get("avg_dollar_vol_m")),
            entry=levels.get("entry", 0.0),
            stop=levels.get("stop", 0.0),
            target=levels.get("target", 0.0),
            risk_pct=levels.get("risk_pct", 0.0),
            reward_pct=levels.get("reward_pct", 0.0),
            quality_flags=qflags,
            quality_pass=qpass,
        ))

    if dedup:
        results = _diversify(results, bars_by_symbol, max_correlation=max_correlation, keep_top=top_n)
    else:
        results = results[:top_n]

    return results


# ════════════════════════════════════════════════════════════════════════════════
# LIVE INTRADAY SCANNER  (Alpaca IEX 5-min bars)
# ════════════════════════════════════════════════════════════════════════════════

def _vwap(bars: pd.DataFrame) -> pd.Series:
    """Compute running VWAP from intraday OHLCV bars."""
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3
    vol = bars["volume"].replace(0, 1)
    cum_vol = vol.cumsum()
    cum_tp_vol = (typical * vol).cumsum()
    return cum_tp_vol / cum_vol


def _score_intraday(bars: pd.DataFrame) -> dict:
    """Score one symbol from intraday 5-min bars. Returns factor dict or {}."""
    if len(bars) < 10:
        return {}

    close = bars["close"].astype(float)
    volume = bars["volume"].astype(float).replace(0, 1)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)

    current_price = float(close.iloc[-1])
    open_price = float(close.iloc[0])

    # 1. VWAP deviation (today's running VWAP)
    vwap_series = _vwap(bars)
    vwap_now = float(vwap_series.iloc[-1])
    vwap_gap_pct = (current_price - vwap_now) / vwap_now * 100 if vwap_now else 0.0

    # 2. Intraday momentum (return from open)
    intraday_return = (current_price / open_price - 1) * 100 if open_price else 0.0

    # 3. Volume spike (last bar vs avg of all prior bars today)
    avg_bar_vol = float(volume.iloc[:-1].mean()) if len(volume) > 1 else float(volume.iloc[-1])
    last_bar_vol = float(volume.iloc[-1])
    vol_spike = last_bar_vol / avg_bar_vol if avg_bar_vol > 0 else 1.0

    # 4. Intraday RSI (14-period on close)
    rsi_val = _rsi(close, period=min(14, len(close) - 1))

    # 5. Trend slope — % of last 8 bars that are up
    last_bars = close.iloc[-9:].pct_change().dropna()
    trend = float((last_bars > 0).mean()) if len(last_bars) >= 4 else 0.5

    # 6. Intraday dip: below VWAP + RSI oversold = potential bounce setup
    intraday_dip = _dip_score(close, rsi_val, rsi_threshold=40.0, drop_pct_threshold=1.0, lookback=min(20, len(close)), long_ma_period=min(20, len(close) - 1))

    # 1-day change vs open
    change_pct = intraday_return

    return {
        "close": current_price,
        "open": open_price,
        "vwap": round(vwap_now, 2),
        "change_pct": round(change_pct, 2),
        "momentum_5d": round(intraday_return, 2),   # re-use field as intraday return
        "rsi": round(rsi_val, 1),
        "volume_surge": round(vol_spike, 2),
        "ma_gap_pct": round(vwap_gap_pct, 2),       # re-use field as VWAP gap
        "trend_consistency": round(trend * 100, 1),
        "dip_score": intraday_dip,
    }


_ALPACA_BATCH_SIZE = 500  # max symbols per Alpaca bars request (avoids 414 URI Too Large)


def scan_live(
    symbols: list[str],
    top_n: int = 10,
    lookback_minutes: int = 390,   # full trading day worth of 5-min bars
    api_key: str | None = None,
    api_secret: str | None = None,
    *,
    min_price: float = 5.0,
    min_dollar_vol: float = 5_000_000.0,
    apply_filters: bool = True,
    earnings_blackout_days: int = 2,
    use_meta: bool = True,
    dedup: bool = True,
    max_correlation: float = 0.85,
) -> list[ScanResult]:
    """Rank symbols using live Alpaca IEX 5-min intraday bars.

    Falls back gracefully to empty list if market is closed or API fails.
    """
    if not symbols:
        return []

    api_key = api_key or os.getenv("APCA_API_KEY_ID", "")
    api_secret = api_secret or os.getenv("APCA_API_SECRET_KEY", "")

    if not api_key or not api_secret:
        raise ValueError("Alpaca API credentials not found in environment.")

    try:
        from alpaca.data.enums import DataFeed
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError as exc:
        raise ImportError(f"alpaca-py not installed: {exc}") from exc

    client = StockHistoricalDataClient(api_key, api_secret)

    end = datetime.now(timezone.utc)
    # Request enough bars to cover at least today's session
    start = end - timedelta(minutes=max(lookback_minutes, 60))

    # Alpaca GET requests have a URL length limit (~8KB), so batch symbols.
    chunks = [symbols[i:i + _ALPACA_BATCH_SIZE] for i in range(0, len(symbols), _ALPACA_BATCH_SIZE)]
    frames: list[pd.DataFrame] = []
    for chunk in chunks:
        request = StockBarsRequest(
            symbol_or_symbols=chunk,
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )
        try:
            df_chunk = client.get_stock_bars(request).df
            if not df_chunk.empty:
                frames.append(df_chunk)
        except Exception as exc:
            # Skip chunks that fail rather than aborting the whole scan
            print(f"Alpaca bars chunk failed ({chunk[0]}..{chunk[-1]}): {exc}")
            continue

    if not frames:
        raise RuntimeError("Alpaca bars request failed for all symbol batches.")
    all_bars = pd.concat(frames)

    if all_bars.empty:
        return []

    rows: list[dict] = []
    for sym in symbols:
        try:
            if isinstance(all_bars.index, pd.MultiIndex):
                bars = all_bars.xs(sym).copy()
            else:
                bars = all_bars.copy()

            bars.columns = [c.lower() for c in bars.columns]
            bars = bars.dropna(subset=["close"])

            # Keep only today's bars (UTC date)
            today_utc = datetime.now(timezone.utc).date()
            bars = bars[bars.index.date == today_utc] if hasattr(bars.index, "date") else bars

            factors = _score_intraday(bars)
            if not factors:
                continue
            factors["symbol"] = sym
            rows.append(factors)
        except (KeyError, Exception):
            continue

    if not rows:
        return []

    df = pd.DataFrame(rows).set_index("symbol")

    # Normalise — for VWAP gap, slightly above 0 is ideal (not too extended)
    df["n_vwap"]     = _normalise(df["ma_gap_pct"].clip(-3, 3))   # VWAP gap stored in ma_gap_pct
    df["n_momentum"] = _normalise(df["momentum_5d"])
    df["n_volume"]   = _normalise(df["volume_surge"])
    df["n_rsi"]      = _normalise(df["rsi"], invert=True)          # oversold = higher score
    df["n_trend"]    = _normalise(df["trend_consistency"])
    df["n_dip"]      = _normalise(df["dip_score"])                 # intraday dip bounce score

    # Initial intraday-only score (used to pick a candidate set for daily-bar enrichment)
    df["score_intraday"] = (
        df["n_vwap"]     * LW_VWAP     +
        df["n_momentum"] * LW_MOMENTUM +
        df["n_volume"]   * LW_VOLUME   +
        df["n_rsi"]      * LW_RSI      +
        df["n_trend"]    * LW_TREND    +
        df["n_dip"]      * LW_DIP
    )

    # ── Enrichment: pull daily bars for the top intraday candidates to compute
    # liquidity, RS-vs-SPY, BB squeeze, meta-prob, ATR levels.
    candidate_n = max(top_n * 5, 30)
    candidates = df["score_intraday"].sort_values(ascending=False).head(candidate_n).index.tolist()
    daily_bars_map, _enrich_source = _fetch_eod_bars(list(dict.fromkeys(candidates + ["SPY"])), lookback_days=120)
    spy_daily = daily_bars_map.get("SPY", pd.DataFrame())
    spy_close = spy_daily["close"].astype(float) if not spy_daily.empty else pd.Series(dtype=float)

    daily_bars_by_symbol: dict[str, pd.DataFrame] = {}
    for sym in candidates:
        b = daily_bars_map.get(sym, pd.DataFrame())
        if not b.empty:
            daily_bars_by_symbol[sym] = b

    # Per-symbol enrichment columns
    df["rs_pct"] = 0.0
    df["squeeze"] = 0.0
    df["meta_prob"] = pd.NA
    df["avg_dollar_vol_m"] = 0.0
    df["liquid"] = True
    for sym, b in daily_bars_by_symbol.items():
        liq_ok, _r, adv = _liquidity_ok(b, min_price=min_price, min_dollar_vol=min_dollar_vol)
        df.at[sym, "liquid"] = liq_ok
        df.at[sym, "avg_dollar_vol_m"] = round(adv / 1e6, 2)
        df.at[sym, "rs_pct"] = _rel_strength_pct(b["close"].astype(float), spy_close)
        df.at[sym, "squeeze"] = _bb_squeeze_score(b["close"].astype(float))
        if use_meta:
            mp = _meta_probability(b)
            if mp is not None:
                df.at[sym, "meta_prob"] = mp

    # Drop illiquid candidates that have daily data available; keep symbols
    # without daily data (yfinance gap) untouched.
    df = df[df["liquid"]]
    if df.empty:
        return []

    df["n_rs"]      = _normalise(df["rs_pct"])
    df["n_squeeze"] = _normalise(df["squeeze"])
    df["n_meta"]    = pd.to_numeric(df["meta_prob"], errors="coerce").fillna(0.5).clip(0.0, 1.0)

    df["score"] = (
        df["n_vwap"]     * LW_VWAP     +
        df["n_momentum"] * LW_MOMENTUM +
        df["n_volume"]   * LW_VOLUME   +
        df["n_rsi"]      * LW_RSI      +
        df["n_trend"]    * LW_TREND    +
        df["n_dip"]      * LW_DIP      +
        df["n_rs"]       * LW_RS       +
        df["n_squeeze"]  * W_SQUEEZE   +
        df["n_meta"]     * LW_META
    ) * 100
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)

    df = df.sort_values("score", ascending=False)

    results: list[ScanResult] = []
    initial_n = top_n * 3 if dedup else top_n
    candidate_symbols = [str(sym) for sym in df.head(initial_n).index]
    check_earnings = len(symbols) < 1000
    quality_ctx = (
        _quality_gate_context(
            candidate_symbols,
            check_earnings=check_earnings,
            earnings_blackout_days=earnings_blackout_days,
        )
        if apply_filters else {"spy": None, "vix": None, "earnings": {}}
    )
    quality_earnings = quality_ctx.get("earnings", {})
    if not isinstance(quality_earnings, dict):
        quality_earnings = {}
    for sym, row in df.head(initial_n).iterrows():
        score = _sf(row.get("score"))
        rsi = _sf(row.get("rsi"))
        vwap_gap = _sf(row.get("ma_gap_pct"))
        momentum = _sf(row.get("momentum_5d"))
        surge = _sf(row.get("volume_surge"), 1.0)
        dip = _sf(row.get("dip_score"))
        rs_pct = _sf(row.get("rs_pct"))
        squeeze = _sf(row.get("squeeze"))
        meta_prob = _sf_opt(row.get("meta_prob"))

        if score >= 65:
            signal = "BUY"
        elif score >= 45:
            signal = "WATCH"
        else:
            signal = "NEUTRAL"

        reason, upside, top_driver = _build_reason(
            score=score, rsi=rsi, momentum=momentum, surge=surge,
            gap_pct=vwap_gap, trend=_sf(row.get("trend_consistency")),
            dip=dip, mode="live",
        )
        extras: list[str] = []
        if rs_pct >= 2.0:
            extras.append(f"RS vs SPY +{rs_pct:.1f}%")
        if squeeze >= 0.85:
            extras.append(f"BB squeeze {squeeze:.2f}")
        if meta_prob is not None and meta_prob >= 0.60:
            extras.append(f"meta P(win)={meta_prob:.2f}")
        if extras:
            reason = " · ".join(extras + ([reason] if reason else []))

        daily_b = daily_bars_by_symbol.get(sym, pd.DataFrame())
        levels = _atr_levels(daily_b) if not daily_b.empty else {}
        qpass, qflags = (True, "ok")
        if apply_filters:
            qpass, qflags = _quality_gates(
                sym, daily_b if not daily_b.empty else pd.DataFrame(),
                check_earnings=check_earnings,
                earnings_blackout_days=earnings_blackout_days,
                spy_trend_result=quality_ctx.get("spy"),
                vix_result=quality_ctx.get("vix"),
                earnings_result=quality_earnings.get(str(sym)),
            )

        results.append(ScanResult(
            symbol=sym,
            score=round(score, 1),
            signal=signal,
            close=_sf(row.get("close")),
            change_pct=_sf(row.get("change_pct")),
            momentum_5d=momentum,
            rsi=rsi,
            volume_surge=surge,
            ma_gap_pct=vwap_gap,
            trend_consistency=_sf(row.get("trend_consistency")),
            reason=reason,
            upside_pct=upside,
            top_driver=top_driver,
            mode="live",
            data_source="alpaca",
            rel_strength_pct=rs_pct,
            bb_squeeze=squeeze,
            meta_prob=meta_prob,
            avg_dollar_vol_m=_sf(row.get("avg_dollar_vol_m")),
            entry=levels.get("entry", 0.0),
            stop=levels.get("stop", 0.0),
            target=levels.get("target", 0.0),
            risk_pct=levels.get("risk_pct", 0.0),
            reward_pct=levels.get("reward_pct", 0.0),
            quality_flags=qflags,
            quality_pass=qpass,
        ))

    if dedup:
        results = _diversify(results, daily_bars_by_symbol, max_correlation=max_correlation, keep_top=top_n)
    else:
        results = results[:top_n]

    return results


def is_market_open() -> bool:
    """Return True if US equity market is currently open (9:30–16:00 ET, Mon–Fri)."""
    from datetime import time
    import zoneinfo
    now_et = datetime.now(zoneinfo.ZoneInfo("America/New_York"))
    if now_et.weekday() >= 5:   # Saturday=5, Sunday=6
        return False
    market_open = time(9, 30)
    market_close = time(16, 0)
    return market_open <= now_et.time() < market_close


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Live market scanner")
    parser.add_argument("--symbols", help="Comma-separated list (default: BOT_SYMBOLS from env)")
    parser.add_argument("--universe", default="",
                        help="Index aliases (comma-sep): sp500,nasdaq100,dow30,all. "
                             "Merged with --symbols. Cached weekly under logs/universe/.")
    parser.add_argument("--refresh-universe", action="store_true",
                        help="Force re-fetch the index lists from Wikipedia")
    parser.add_argument("--top", type=int, default=10, help="Number of top results to show")
    parser.add_argument("--fast-ma", type=int, default=5)
    parser.add_argument("--slow-ma", type=int, default=20)
    parser.add_argument(
        "--mode", choices=["auto", "live", "eod"], default="auto",
        help=(
            "auto=live if market open else eod. Live uses Alpaca IEX intraday bars; "
            "EOD uses yfinance/Alpaca daily bars. Robinhood is used later for execution quotes."
        ),
    )
    parser.add_argument("--min-price", type=float, default=5.0, help="Liquidity gate: min last close")
    parser.add_argument("--min-dollar-vol", type=float, default=5_000_000.0,
                        help="Liquidity gate: min 20d avg dollar volume")
    parser.add_argument("--no-filters", action="store_true",
                        help="Skip macro quality gates (SPY trend / VIX / earnings / volume)")
    parser.add_argument("--earnings-blackout", type=int, default=2,
                        help="Days before earnings to flag (0=disabled)")
    parser.add_argument("--no-meta", action="store_true", help="Skip meta-label P(win) inference")
    parser.add_argument("--no-dedup", action="store_true", help="Skip correlation-based dedup")
    parser.add_argument("--max-corr", type=float, default=0.85,
                        help="Drop picks more correlated than this with a higher-ranked pick")
    parser.add_argument("--only-pass", action="store_true",
                        help="Only show picks that pass all macro quality gates")
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        raw = os.getenv("BOT_SYMBOLS", os.getenv("BOT_SYMBOL", "SPY"))
        symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]

    if args.universe:
        from ai_trading.data.universe import load_universe
        aliases = [a.strip() for a in args.universe.split(",") if a.strip()]
        idx_syms = load_universe(aliases, refresh=args.refresh_universe)
        symbols = list(dict.fromkeys(symbols + idx_syms))
        print(f"Universe expanded: +{len(idx_syms)} symbols → {len(symbols)} total")

    use_live = args.mode == "live" or (args.mode == "auto" and is_market_open())
    effective_use_live = use_live
    broker = os.getenv("BOT_BROKER", "alpaca").strip().lower() or "alpaca"
    mode_label = "LIVE (Alpaca IEX 5-min)" if use_live else "EOD (historical daily bars)"
    if broker == "robinhood":
        mode_label += "; execution quotes: Robinhood"

    print(f"Scanning {len(symbols)} symbols — mode: {mode_label}")

    common_kwargs = dict(
        min_price=args.min_price,
        min_dollar_vol=args.min_dollar_vol,
        apply_filters=not args.no_filters,
        earnings_blackout_days=args.earnings_blackout,
        use_meta=not args.no_meta,
        dedup=not args.no_dedup,
        max_correlation=args.max_corr,
    )

    if use_live:
        try:
            results = scan_live(symbols, top_n=args.top, **common_kwargs)
        except Exception as exc:
            print(f"Live scan failed ({exc}), falling back to EOD...")
            effective_use_live = False
            results = scan(symbols, fast_ma=args.fast_ma, slow_ma=args.slow_ma,
                           top_n=args.top, **common_kwargs)
    else:
        results = scan(symbols, fast_ma=args.fast_ma, slow_ma=args.slow_ma,
                       top_n=args.top, **common_kwargs)

    if args.only_pass:
        results = [r for r in results if r.quality_pass]

    if not results:
        print("No results. Market may be closed, no data available, or all picks filtered out.")
        return

    if results:
        sources = sorted({r.data_source for r in results if r.data_source})
        source_label = ", ".join(sources) if sources else "unknown"
        scan_kind = "LIVE" if effective_use_live else "EOD"
        mode_label = f"{scan_kind} scan data: {source_label}"
        if broker == "robinhood":
            mode_label += "; execution quotes: Robinhood"

    print(f"\n{'='*120}")
    print(f"  TOP {args.top} BUY OPPORTUNITIES  —  {format_local_now('%Y-%m-%d %I:%M %p %Z')}  [{mode_label}]")
    print(f"{'='*120}")
    label_5d = "Intra%" if effective_use_live else "5D%"
    label_vwap = "VWAP%" if effective_use_live else "MA%"
    print(
        f"{'#':<3} {'Sym':<6} {'Score':>6} {'Sig':<6} {'Price':>8} {'1D%':>6} {label_5d:>7} "
        f"{'RSI':>5} {'VolX':>5} {label_vwap:>7} {'RS%':>6} {'Sqz':>5} {'Meta':>5} "
        f"{'Entry':>7} {'Stop':>7} {'Tgt':>7} {'R%':>5} {'Gate':<12}  Reason"
    )
    print("-" * 120)
    for i, r in enumerate(results, 1):
        meta_str = f"{r.meta_prob:.2f}" if r.meta_prob is not None else "  - "
        gate_str = "PASS" if r.quality_pass else r.quality_flags
        print(
            f"{i:<3} {r.symbol:<6} {r.score:>6.1f} {r.signal:<6} "
            f"${r.close:>7.2f} {r.change_pct:>+6.2f}% {r.momentum_5d:>+6.2f}% "
            f"{r.rsi:>5.1f} {r.volume_surge:>4.1f}x {r.ma_gap_pct:>+6.2f}% "
            f"{r.rel_strength_pct:>+6.1f} {r.bb_squeeze:>5.2f} {meta_str:>5} "
            f"${r.entry:>6.2f} ${r.stop:>6.2f} ${r.target:>6.2f} {r.risk_pct:>4.1f}% "
            f"{gate_str:<12}  {r.reason}"
        )


if __name__ == "__main__":
    main()
