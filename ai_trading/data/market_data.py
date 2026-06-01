from __future__ import annotations

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


class AlpacaMarketData:
    def __init__(self, api_key: str, api_secret: str) -> None:
        self.client = StockHistoricalDataClient(api_key, api_secret)

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
            timeframe: One of "1Day", "1Hour", "30Min", "15Min", "5Min", "1Min".
        """
        tf = _TIMEFRAME_MAP.get(timeframe.lower(), TimeFrame.Day)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days)
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
        return bars.sort_index()

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
        start = end - timedelta(days=lookback_days)
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
