"""Parquet-backed OHLCV bar cache with TTL.

Wrap any bar-fetching function with `cached_bars()` to avoid re-downloading
identical requests during a short window.

Usage:
    from ai_trading.data.cache import BarCache
    cache = BarCache(".cache/bars", ttl_seconds=60)
    bars = cache.get_or_fetch("SPY", 90, "1Day", fetch=market_data.get_bars)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)


class BarCache:
    def __init__(self, cache_dir: str | Path = ".cache/bars", ttl_seconds: int = 60) -> None:
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ttl = int(ttl_seconds)

    def _path(self, symbol: str, lookback_days: int, timeframe: str) -> Path:
        safe_tf = timeframe.replace("/", "_")
        return self.dir / f"{symbol}_{lookback_days}_{safe_tf}.parquet"

    def get(self, symbol: str, lookback_days: int, timeframe: str) -> pd.DataFrame | None:
        p = self._path(symbol, lookback_days, timeframe)
        if not p.exists():
            return None
        age = time.time() - p.stat().st_mtime
        if age > self.ttl:
            return None
        try:
            return pd.read_parquet(p)
        except Exception as exc:  # noqa: BLE001
            logger.warning("BarCache read failed for %s: %s", p, exc)
            return None

    def set(self, symbol: str, lookback_days: int, timeframe: str, bars: pd.DataFrame) -> None:
        if bars is None or bars.empty:
            return
        p = self._path(symbol, lookback_days, timeframe)
        try:
            bars.to_parquet(p)
        except Exception as exc:  # noqa: BLE001
            logger.warning("BarCache write failed for %s: %s", p, exc)

    def get_or_fetch(
        self,
        symbol: str,
        lookback_days: int,
        timeframe: str,
        fetch: Callable[[str, int, str], pd.DataFrame],
    ) -> pd.DataFrame:
        hit = self.get(symbol, lookback_days, timeframe)
        if hit is not None:
            return hit
        bars = fetch(symbol, lookback_days, timeframe)
        self.set(symbol, lookback_days, timeframe, bars)
        return bars

    def clear(self) -> int:
        n = 0
        for p in self.dir.glob("*.parquet"):
            try:
                p.unlink()
                n += 1
            except OSError:
                pass
        return n
