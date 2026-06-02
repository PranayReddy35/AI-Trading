from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


_TIMEFRAME_MAP: dict[str, TimeFrame] = {
    "1day": TimeFrame.Day,
    "1hour": TimeFrame.Hour,
    "30min": TimeFrame(30, TimeFrameUnit.Minute),
    "15min": TimeFrame(15, TimeFrameUnit.Minute),
    "5min": TimeFrame(5, TimeFrameUnit.Minute),
    "1min": TimeFrame.Minute,
}

# Trading days buffer: request extra calendar days to ensure enough trading bars
_TRADING_DAY_BUFFER = 1.5  # multiply lookback_days by this factor


class _DataCache:
    """Simple in-memory cache with TTL for market data."""

    def __init__(self, ttl_sec: int = 300) -> None:
        self.ttl_sec = ttl_sec
        self._store: dict[str, tuple[float, pd.DataFrame]] = {}

    def get(self, key: str) -> pd.DataFrame | None:
        if self.ttl_sec <= 0:
            return None
        entry = self._store.get(key)
        if entry is None:
            return None
        ts, df = entry
        if time.time() - ts > self.ttl_sec:
            del self._store[key]
            return None
        return df

    def put(self, key: str, df: pd.DataFrame) -> None:
        if self.ttl_sec <= 0:
            return
        self._store[key] = (time.time(), df)

    def clear(self) -> None:
        self._store.clear()


class AlpacaMarketData:
    def __init__(self, api_key: str, api_secret: str, cache_ttl_sec: int = 300) -> None:
        self.client = StockHistoricalDataClient(api_key, api_secret)
        self._cache = _DataCache(ttl_sec=cache_ttl_sec)

    def get_bars(
        self,
        symbol: str,
        lookback_days: int,
        timeframe: str = "1Day",
    ) -> pd.DataFrame:
        """Fetch OHLCV bars for any supported timeframe.

        Args:
            symbol: Ticker symbol.
            lookback_days: How many calendar days back to fetch.
                Internally requests extra days to account for weekends/holidays
                and ensure a minimum number of actual trading bars.
            timeframe: One of "1Day", "1Hour", "30Min", "15Min", "5Min", "1Min".
        """
        cache_key = f"{symbol}:{timeframe}:{lookback_days}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        tf = _TIMEFRAME_MAP.get(timeframe.lower(), TimeFrame.Day)
        end = datetime.now(timezone.utc)
        # Fix #11: Request extra days to guarantee enough trading bars
        adjusted_days = int(lookback_days * _TRADING_DAY_BUFFER) + 5
        start = end - timedelta(days=adjusted_days)
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )
        bars = self.client.get_stock_bars(request).df
        if bars.empty:
            raise ValueError(f"No bars for {symbol} ({timeframe})")
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(symbol)
        result = bars.sort_index()
        self._cache.put(cache_key, result)
        return result

    def get_daily_bars(self, symbol: str, lookback_days: int) -> pd.DataFrame:
        """Fetch daily OHLCV bars (backwards-compatible)."""
        return self.get_bars(symbol, lookback_days, "1Day")

    def get_multi_symbol_bars(
        self,
        symbols: list[str],
        lookback_days: int,
        timeframe: str = "1Day",
    ) -> dict[str, pd.DataFrame]:
        """Fetch bars for multiple symbols at once. Returns {symbol: DataFrame}."""
        tf = _TIMEFRAME_MAP.get(timeframe.lower(), TimeFrame.Day)
        end = datetime.now(timezone.utc)
        # Fix #11: Request extra days to guarantee enough trading bars
        adjusted_days = int(lookback_days * _TRADING_DAY_BUFFER) + 5
        start = end - timedelta(days=adjusted_days)
        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=tf,
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )
        all_bars = self.client.get_stock_bars(request).df
        result: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                if isinstance(all_bars.index, pd.MultiIndex):
                    df = all_bars.xs(sym)
                else:
                    df = all_bars
                result[sym] = df.sort_index()
            except KeyError:
                pass
        return result
