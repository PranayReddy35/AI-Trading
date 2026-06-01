"""Live market scanner — scores and ranks symbols by buy opportunity strength.

Two scan modes
--------------
scan_live()  — uses Alpaca IEX 5-min intraday bars (real-time, market hours).
               Factors: VWAP deviation, intraday momentum, volume spike,
                        intraday RSI, price trend slope.
scan()       — uses yfinance end-of-day bars (works 24/7, good for pre-market prep).
               Factors: MA crossover, 5-day momentum, RSI, volume surge, trend.

Usage:
    python -m ai_trading.scanner               # live mode if market open, else EOD
    python -m ai_trading.scanner --mode live
    python -m ai_trading.scanner --mode eod
    python -m ai_trading.scanner --top 10 --symbols SPY,QQQ,NVDA
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf


# ── Scoring weights (EOD mode) ────────────────────────────────────────────────
W_MA        = 0.25
W_MOMENTUM  = 0.20
W_RSI       = 0.15
W_VOLUME    = 0.15
W_TREND     = 0.10
W_DIP       = 0.15   # dip-buy composite (RSI oversold + pullback from high + above long MA)

# ── Scoring weights (LIVE intraday mode) ─────────────────────────────────────
LW_VWAP      = 0.25   # price vs VWAP (above = bullish)
LW_MOMENTUM  = 0.20   # intraday price return from open
LW_VOLUME    = 0.20   # volume vs avg intraday volume
LW_RSI       = 0.15   # intraday RSI (oversold bounce)
LW_TREND     = 0.10   # recent bar-by-bar slope
LW_DIP       = 0.10   # intraday dip (below VWAP + RSI oversold)


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
    as_of: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%H:%M:%S UTC"))


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
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series([0.5] * len(series), index=series.index)
    norm = (series - mn) / (mx - mn)
    return 1 - norm if invert else norm


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


def scan(
    symbols: list[str],
    fast_ma: int = 5,
    slow_ma: int = 20,
    lookback_days: int = 60,
    top_n: int = 10,
) -> list[ScanResult]:
    """Fetch bars for all symbols, score each, and return top_n ranked results.

    Uses yfinance so no Alpaca subscription needed.
    """
    if not symbols:
        return []

    # Batch download via yfinance (chunks of 500 to stay under URL limits)
    period = f"{lookback_days}d"
    _BATCH_YF = 500
    raw_frames: list[pd.DataFrame] = []
    sym_chunks = [symbols[i:i + _BATCH_YF] for i in range(0, len(symbols), _BATCH_YF)]
    for chunk in sym_chunks:
        try:
            df_chunk = yf.download(
                chunk,
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if not df_chunk.empty:
                raw_frames.append(df_chunk)
        except Exception as exc:
            print(f"Scanner yfinance chunk error: {exc}")
    raw = pd.concat(raw_frames, axis=1) if raw_frames else pd.DataFrame()

    rows: list[dict] = []

    for sym in symbols:
        try:
            if len(symbols) == 1:
                bars = raw.copy()
            else:
                if sym not in raw.columns.get_level_values(0):
                    continue
                bars = raw[sym].copy()

            bars.columns = [c.lower() for c in bars.columns]
            bars = bars.dropna(subset=["close"])

            if len(bars) < slow_ma + 5:
                continue

            factors = _score_symbol(bars, fast_ma, slow_ma)
            if not factors:
                continue
            factors["symbol"] = sym
            rows.append(factors)
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

    df["score"] = (
        df["n_ma"]       * W_MA       +
        df["n_momentum"] * W_MOMENTUM +
        df["n_rsi"]      * W_RSI      +
        df["n_volume"]   * W_VOLUME   +
        df["n_trend"]    * W_TREND    +
        df["n_dip"]      * W_DIP
    ) * 100

    df = df.sort_values("score", ascending=False)

    results: list[ScanResult] = []
    for sym, row in df.head(top_n).iterrows():
        score = float(row["score"])
        rsi = float(row["rsi"])
        ma_gap = float(row["ma_gap_pct"])
        momentum = float(row["momentum_5d"])
        surge = float(row["volume_surge"])
        dip = float(row.get("dip_score", 0.0))

        if score >= 65:
            signal = "BUY"
        elif score >= 45:
            signal = "WATCH"
        else:
            signal = "NEUTRAL"

        reason, upside, top_driver = _build_reason(
            score=score, rsi=rsi, momentum=momentum, surge=surge,
            gap_pct=ma_gap, trend=float(row["trend_consistency"]),
            dip=dip, mode="eod",
        )

        results.append(ScanResult(
            symbol=sym,
            score=round(score, 1),
            signal=signal,
            close=float(row["close"]),
            change_pct=float(row["change_pct"]),
            momentum_5d=momentum,
            rsi=rsi,
            volume_surge=surge,
            ma_gap_pct=ma_gap,
            trend_consistency=float(row["trend_consistency"]),
            reason=reason,
            upside_pct=upside,
            top_driver=top_driver,
            mode="eod",
        ))

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
    _BATCH = 200
    chunks = [symbols[i:i + _BATCH] for i in range(0, len(symbols), _BATCH)]
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

    df["score"] = (
        df["n_vwap"]     * LW_VWAP     +
        df["n_momentum"] * LW_MOMENTUM +
        df["n_volume"]   * LW_VOLUME   +
        df["n_rsi"]      * LW_RSI      +
        df["n_trend"]    * LW_TREND    +
        df["n_dip"]      * LW_DIP
    ) * 100

    df = df.sort_values("score", ascending=False)

    results: list[ScanResult] = []
    for sym, row in df.head(top_n).iterrows():
        score = float(row["score"])
        rsi = float(row["rsi"])
        vwap_gap = float(row["ma_gap_pct"])
        momentum = float(row["momentum_5d"])
        surge = float(row["volume_surge"])
        vwap = float(row.get("vwap", 0))
        dip = float(row.get("dip_score", 0.0))

        if score >= 65:
            signal = "BUY"
        elif score >= 45:
            signal = "WATCH"
        else:
            signal = "NEUTRAL"

        reason, upside, top_driver = _build_reason(
            score=score, rsi=rsi, momentum=momentum, surge=surge,
            gap_pct=vwap_gap, trend=float(row["trend_consistency"]),
            dip=dip, mode="live",
        )

        results.append(ScanResult(
            symbol=sym,
            score=round(score, 1),
            signal=signal,
            close=float(row["close"]),
            change_pct=float(row["change_pct"]),
            momentum_5d=momentum,
            rsi=rsi,
            volume_surge=surge,
            ma_gap_pct=vwap_gap,
            trend_consistency=float(row["trend_consistency"]),
            reason=reason,
            upside_pct=upside,
            top_driver=top_driver,
            mode="live",
        ))

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
    parser.add_argument("--top", type=int, default=10, help="Number of top results to show")
    parser.add_argument("--fast-ma", type=int, default=5)
    parser.add_argument("--slow-ma", type=int, default=20)
    parser.add_argument(
        "--mode", choices=["auto", "live", "eod"], default="auto",
        help="auto=live if market open else eod, live=Alpaca intraday, eod=yfinance daily",
    )
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        raw = os.getenv("BOT_SYMBOLS", os.getenv("BOT_SYMBOL", "SPY"))
        symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]

    use_live = args.mode == "live" or (args.mode == "auto" and is_market_open())
    mode_label = "LIVE (Alpaca IEX 5-min)" if use_live else "EOD (yfinance daily)"

    print(f"Scanning {len(symbols)} symbols — mode: {mode_label}")

    if use_live:
        try:
            results = scan_live(symbols, top_n=args.top)
        except Exception as exc:
            print(f"Live scan failed ({exc}), falling back to EOD...")
            results = scan(symbols, fast_ma=args.fast_ma, slow_ma=args.slow_ma, top_n=args.top)
    else:
        results = scan(symbols, fast_ma=args.fast_ma, slow_ma=args.slow_ma, top_n=args.top)

    if not results:
        print("No results. Market may be closed or no data available.")
        return

    print(f"\n{'='*78}")
    print(f"  TOP {args.top} BUY OPPORTUNITIES  —  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  [{mode_label}]")
    print(f"{'='*78}")
    label_5d = "Intraday%" if use_live else "5D%"
    label_vwap = "VWAP Gap%" if use_live else "MA Gap%"
    print(f"{'#':<3} {'Symbol':<6} {'Score':>6} {'Signal':<8} {'Price':>8} {'1D%':>6} {label_5d:>10} {'RSI':>5} {'VolX':>5}  {label_vwap:>9}  Reason")
    print("-" * 78)
    for i, r in enumerate(results, 1):
        print(
            f"{i:<3} {r.symbol:<6} {r.score:>6.1f} {r.signal:<8} "
            f"${r.close:>7.2f} {r.change_pct:>+6.2f}% {r.momentum_5d:>+9.2f}% "
            f"{r.rsi:>5.1f} {r.volume_surge:>5.1f}x {r.ma_gap_pct:>+9.2f}%  {r.reason}"
        )


if __name__ == "__main__":
    main()

