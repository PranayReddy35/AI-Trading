from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


class AlpacaMarketData:
    def __init__(self, api_key: str, api_secret: str) -> None:
        self.client = StockHistoricalDataClient(api_key, api_secret)

    def get_daily_bars(self, symbol: str, lookback_days: int) -> pd.DataFrame:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days)
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        )
        bars = self.client.get_stock_bars(request).df
        if bars.empty:
            raise ValueError(f"No daily bars for {symbol}")
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(symbol)
        return bars.sort_index()
