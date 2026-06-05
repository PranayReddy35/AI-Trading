"""Macro-level market filters used to gate signals.

All data is free via yfinance. Results are cached for the lifetime of the
process to avoid hammering Yahoo when iterating over a symbol watchlist.

Filters:
- spy_trend_ok()        — SPY price vs 200-day moving average (risk-on/off)
- vix_size_multiplier() — scale position size down when VIX is elevated
- in_earnings_blackout()— avoid opening positions N days before earnings
- volume_confirms()     — current bar volume must be >= ratio * 20-day avg
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache

import pandas as pd


# ---------------------------------------------------------------------------
# SPY 200-DMA regime
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _spy_history(period: str = "2y") -> pd.DataFrame:
    import yfinance as yf
    df = yf.download("SPY", period=period, progress=False, auto_adjust=False)
    if df.empty:
        return df
    df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
    return df


def spy_trend_ok(window: int = 200) -> tuple[bool, str]:
    """Return (True, reason) if SPY is above its `window`-day SMA."""
    df = _spy_history()
    if df.empty or len(df) < window:
        return True, f"SPY data unavailable; allow trade (need {window} bars)"
    sma = df["close"].rolling(window).mean().iloc[-1]
    px = float(df["close"].iloc[-1])
    sma_f = float(sma)
    if px >= sma_f:
        return True, f"SPY {px:.2f} >= {window}DMA {sma_f:.2f}"
    return False, f"SPY {px:.2f} < {window}DMA {sma_f:.2f} (risk-off)"


# ---------------------------------------------------------------------------
# VIX-based size scaling
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _vix_history(period: str = "3mo") -> pd.DataFrame:
    import yfinance as yf
    df = yf.download("^VIX", period=period, progress=False, auto_adjust=False)
    if df.empty:
        return df
    df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
    return df


def current_vix() -> float | None:
    df = _vix_history()
    if df.empty:
        return None
    return float(df["close"].iloc[-1])


def vix_size_multiplier(
    vix: float | None = None,
    *,
    full_below: float = 20.0,
    half_above: float = 25.0,
    zero_above: float = 35.0,
) -> tuple[float, str]:
    """Return a size multiplier in [0,1] based on VIX level.

    VIX <= full_below  → 1.0  (normal sizing)
    VIX >= zero_above  → 0.0  (no new positions)
    half_above        → 0.5
    Linear ramp between the boundaries.
    """
    if vix is None:
        vix = current_vix()
    if vix is None:
        return 1.0, "VIX unavailable; full size"
    if vix <= full_below:
        return 1.0, f"VIX {vix:.1f} <= {full_below} (calm)"
    if vix >= zero_above:
        return 0.0, f"VIX {vix:.1f} >= {zero_above} (panic — no new positions)"
    if vix <= half_above:
        # ramp 1.0 → 0.5 between full_below and half_above
        span = max(1e-9, half_above - full_below)
        mult = 1.0 - 0.5 * (vix - full_below) / span
    else:
        # ramp 0.5 → 0.0 between half_above and zero_above
        span = max(1e-9, zero_above - half_above)
        mult = 0.5 * (1.0 - (vix - half_above) / span)
    return max(0.0, min(1.0, mult)), f"VIX {vix:.1f} → size×{mult:.2f}"


# ---------------------------------------------------------------------------
# Earnings blackout
# ---------------------------------------------------------------------------


_DEFAULT_EARNINGS_SKIP_SYMBOLS = frozenset({
    "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "VT", "VEA", "VWO",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLY", "XLP", "XLU", "XLRE",
    "GLD", "SLV", "USO", "UNG", "TLT", "IEF", "SHY", "HYG", "LQD", "EEM",
    "ARKK", "ARKG", "ARKW", "XBI", "SMH", "SOXX", "KRE", "XRT",
    "XLB", "IGV", "KWEB", "FXI", "IYR", "EWZ", "EFA", "IJH", "RSP", "SCHD",
    "VXX", "UVXY", "SVXY", "SQQQ", "TQQQ", "SPXL", "SPXS",
    "SOXL", "SOXS", "TNA", "TZA", "LABU", "LABD", "FAS", "FAZ",
    "BOIL", "KOLD", "NUGT", "DUST", "BITO", "IBIT", "FBTC", "ETHA",
    "TSLL", "TSLG", "TSDD", "GGLS", "AAPD", "AMZD", "PLTD", "DRAM",
    "IUSB", "USHY", "QID", "RWM", "SPDN",
})


def _earnings_skip_symbols() -> set[str]:
    raw = os.getenv("BOT_EARNINGS_SKIP_SYMBOLS", "")
    extra = {s.strip().upper() for s in raw.split(",") if s.strip()}
    return set(_DEFAULT_EARNINGS_SKIP_SYMBOLS) | extra


@dataclass(slots=True)
class EarningsCheck:
    blocked: bool
    next_earnings: date | None
    days_until: int | None
    reason: str


@lru_cache(maxsize=256)
def _next_earnings_date(symbol: str) -> date | None:
    """Look up the next earnings date via yfinance. None if unknown."""
    if symbol.strip().upper() in _earnings_skip_symbols():
        return None
    try:
        import yfinance as yf
        tk = yf.Ticker(symbol)
        cal = tk.calendar
        if cal is None:
            return None
        # yfinance returns either a DataFrame or dict depending on version
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if ed is None:
                return None
            if isinstance(ed, list) and ed:
                ed = ed[0]
            if hasattr(ed, "date"):
                return ed.date()
            return None
        # DataFrame case
        if "Earnings Date" in getattr(cal, "index", []):
            v = cal.loc["Earnings Date"].iloc[0]
            if hasattr(v, "date"):
                return v.date()
    except Exception:
        return None
    return None


def in_earnings_blackout(symbol: str, blackout_days: int = 2, today: date | None = None) -> EarningsCheck:
    """Block opening new positions within `blackout_days` of an earnings release."""
    if blackout_days <= 0:
        return EarningsCheck(False, None, None, "blackout disabled")
    if symbol.strip().upper() in _earnings_skip_symbols():
        return EarningsCheck(False, None, None, "earnings lookup skipped for fund/ETF symbol; allow")
    today = today or datetime.now().date()
    ed = _next_earnings_date(symbol)
    if ed is None:
        return EarningsCheck(False, None, None, "no earnings date available; allow")
    delta = (ed - today).days
    if 0 <= delta <= blackout_days:
        return EarningsCheck(
            True, ed, delta,
            f"earnings in {delta}d ({ed.isoformat()}) within {blackout_days}d blackout",
        )
    return EarningsCheck(False, ed, delta, f"earnings {ed.isoformat()} ({delta}d away)")


def earnings_blackout_map(
    symbols: list[str] | tuple[str, ...],
    blackout_days: int = 2,
    today: date | None = None,
    *,
    max_workers: int = 8,
) -> dict[str, EarningsCheck]:
    """Return earnings blackout checks for symbols with bounded parallel yfinance calls.

    yfinance exposes earnings calendars per ticker, not as a true batch endpoint. This
    wrapper limits concurrency and reuses _next_earnings_date's process cache.
    """
    clean = tuple(dict.fromkeys(str(s).strip().upper() for s in symbols if str(s).strip()))
    if not clean:
        return {}
    if blackout_days <= 0:
        return {
            sym: EarningsCheck(False, None, None, "blackout disabled")
            for sym in clean
        }
    today = today or datetime.now().date()
    workers = max(1, min(int(max_workers or 1), len(clean)))
    if workers == 1:
        return {
            sym: in_earnings_blackout(sym, blackout_days=blackout_days, today=today)
            for sym in clean
        }
    out: dict[str, EarningsCheck] = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="earnings-gate") as ex:
        futures = {
            ex.submit(in_earnings_blackout, sym, blackout_days, today): sym
            for sym in clean
        }
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                out[sym] = fut.result()
            except Exception:
                out[sym] = EarningsCheck(False, None, None, "earnings lookup failed; allow")
    return out


# ---------------------------------------------------------------------------
# Volume confirmation
# ---------------------------------------------------------------------------


def volume_confirms(bars: pd.DataFrame, *, lookback: int = 20, min_ratio: float = 0.8) -> tuple[bool, str]:
    """Reject signals on bars with abnormally low volume.

    Returns (True, reason) when current volume >= min_ratio × rolling mean.
    """
    if "volume" not in bars or len(bars) < lookback + 1:
        return True, "insufficient volume data; allow"
    avg = float(bars["volume"].iloc[-(lookback + 1):-1].mean())
    if avg <= 0:
        return True, "zero average volume; allow"
    cur = float(bars["volume"].iloc[-1])
    ratio = cur / avg
    if ratio >= min_ratio:
        return True, f"vol ratio {ratio:.2f} >= {min_ratio}"
    return False, f"vol ratio {ratio:.2f} < {min_ratio} (weak participation)"


# ---------------------------------------------------------------------------
# Spread filter (Alpaca quote)
# ---------------------------------------------------------------------------


def spread_too_wide(bid: float, ask: float, max_bps: float = 10.0) -> tuple[bool, str]:
    """Reject when (ask-bid)/mid > max_bps (basis points)."""
    if bid <= 0 or ask <= 0 or ask < bid:
        return True, f"invalid quote bid={bid} ask={ask}"
    mid = (bid + ask) / 2.0
    spread_bps = (ask - bid) / mid * 1e4
    if spread_bps > max_bps:
        return True, f"spread {spread_bps:.1f}bps > {max_bps:.1f}bps"
    return False, f"spread {spread_bps:.1f}bps OK"


def clear_macro_cache() -> None:
    """Useful for tests."""
    _spy_history.cache_clear()
    _vix_history.cache_clear()
    _next_earnings_date.cache_clear()
