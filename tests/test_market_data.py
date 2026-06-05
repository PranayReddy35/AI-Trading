from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.enums import DataFeed

from ai_trading.data.market_data import AlpacaMarketData, LatestPrice


@dataclass
class _Quote:
    bid_price: float
    ask_price: float
    timestamp: object | None = None


@dataclass
class _Trade:
    price: float
    timestamp: object | None = None


class _Client:
    def __init__(
        self,
        quote=None,
        trade=None,
        quote_map: dict[str, _Quote] | None = None,
        trade_map: dict[str, _Trade] | None = None,
        quote_map_by_feed: dict[str, dict[str, _Quote]] | None = None,
        trade_map_by_feed: dict[str, dict[str, _Trade]] | None = None,
        quote_error: Exception | None = None,
        fail_feeds: set[str] | None = None,
    ):
        self.quote = quote
        self.trade = trade
        self.quote_map = quote_map or {}
        self.trade_map = trade_map or {}
        self.quote_map_by_feed = quote_map_by_feed or {}
        self.trade_map_by_feed = trade_map_by_feed or {}
        self.quote_error = quote_error
        self.fail_feeds = fail_feeds or set()
        self.quote_calls = 0
        self.trade_calls = 0

    def get_stock_latest_quote(self, request):
        self.quote_calls += 1
        if request.feed.value in self.fail_feeds:
            raise PermissionError(request.feed.value)
        if self.quote_error:
            raise self.quote_error
        symbols = request.symbol_or_symbols
        if isinstance(symbols, str):
            symbols = [symbols]
        feed_quotes = self.quote_map_by_feed.get(request.feed.value, {})
        return {sym: feed_quotes.get(sym, self.quote_map.get(sym, self.quote)) for sym in symbols}

    def get_stock_latest_trade(self, request):
        self.trade_calls += 1
        if request.feed.value in self.fail_feeds:
            raise PermissionError(request.feed.value)
        symbols = request.symbol_or_symbols
        if isinstance(symbols, str):
            symbols = [symbols]
        feed_trades = self.trade_map_by_feed.get(request.feed.value, {})
        return {sym: feed_trades.get(sym, self.trade_map.get(sym, self.trade)) for sym in symbols}


def _market_data(client: _Client) -> AlpacaMarketData:
    md = AlpacaMarketData.__new__(AlpacaMarketData)
    md.client = client
    md.data_feed = "iex"
    return md


def test_latest_price_prefers_quote_midpoint():
    md = _market_data(_Client(quote=_Quote(100.0, 100.2), trade=_Trade(99.0)))

    latest = md.get_latest_price("SPY")

    assert latest.price == 100.1
    assert latest.source == "quote_mid"
    assert latest.feed == "iex"
    assert latest.bid == 100.0
    assert latest.ask == 100.2


def test_latest_price_falls_back_to_trade_when_quote_invalid():
    md = _market_data(_Client(quote=_Quote(0.0, 0.0), trade=_Trade(101.5)))

    latest = md.get_latest_price("SPY")

    assert latest.price == 101.5
    assert latest.source == "latest_trade"
    assert latest.feed == "iex"


def test_latest_price_auto_falls_back_to_available_feed():
    md = _market_data(_Client(
        quote=_Quote(100.0, 100.2),
        trade=_Trade(99.0),
        fail_feeds={"sip", "overnight", "boats"},
    ))
    md.data_feed = "auto"

    latest = md.get_latest_price("SPY")

    assert latest.price == 100.1
    assert latest.feed == "iex"


def test_latest_prices_batches_symbols_in_one_quote_request():
    client = _Client(
        quote_map={
            "SPY": _Quote(100.0, 100.2),
            "AAPL": _Quote(200.0, 200.4),
        },
        trade=_Trade(99.0),
    )
    md = _market_data(client)

    latest = md.get_latest_prices(["SPY", "AAPL"])

    assert latest["SPY"].price == 100.1
    assert latest["AAPL"].price == 200.2
    assert client.quote_calls == 1
    assert client.trade_calls == 0


def test_auto_feed_order_moves_overnight_after_premarket_handoff():
    md = _market_data(_Client())
    md.data_feed = "auto"
    premarket = datetime(2026, 6, 5, 4, 1, tzinfo=ZoneInfo("America/New_York"))
    overnight = datetime(2026, 6, 5, 3, 59, tzinfo=ZoneInfo("America/New_York"))

    assert md._feeds(premarket)[:3] == [DataFeed.SIP, DataFeed.IEX, DataFeed.DELAYED_SIP]
    assert md._feeds(overnight)[:2] == [DataFeed.BOATS, DataFeed.OVERNIGHT]


def test_auto_latest_price_chooses_freshest_timestamp_across_feeds():
    old = datetime(2026, 6, 5, 8, 0, tzinfo=timezone.utc)
    new = datetime(2026, 6, 5, 8, 35, tzinfo=timezone.utc)
    client = _Client(
        quote_map_by_feed={
            "sip": {"SPY": _Quote(100.0, 100.2, old)},
            "iex": {"SPY": _Quote(101.0, 101.2, new)},
        },
        trade=_Trade(0.0),
    )
    md = _market_data(client)
    md.data_feed = "auto"

    latest = md.get_latest_price("SPY")

    assert latest.price == 101.1
    assert latest.feed == "iex"
    assert latest.timestamp == new.isoformat()


def test_with_latest_price_updates_last_bar_ohlc_only():
    md = _market_data(_Client())
    bars = pd.DataFrame(
        {
            "open": [98.0, 100.0],
            "high": [101.0, 102.0],
            "low": [97.0, 99.0],
            "close": [100.0, 101.0],
            "volume": [1000, 1100],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )

    patched = md.with_latest_price(bars, LatestPrice("SPY", 103.0, "test", "iex"))

    assert patched["close"].iloc[-1] == 103.0
    assert patched["high"].iloc[-1] == 103.0
    assert patched["low"].iloc[-1] == 99.0
    assert patched["close"].iloc[0] == 100.0
    assert bars["close"].iloc[-1] == 101.0
