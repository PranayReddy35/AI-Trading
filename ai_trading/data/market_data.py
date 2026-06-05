from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest, StockLatestTradeRequest
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


@dataclass(frozen=True, slots=True)
class LatestPrice:
    symbol: str
    price: float
    source: str
    feed: str
    bid: float | None = None
    ask: float | None = None
    timestamp: str | None = None
    age_seconds: float | None = None
    stale: bool = False
    session: str = ""


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
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        cache_ttl_sec: int = 300,
        data_feed: str = "auto",
    ) -> None:
        self.client = StockHistoricalDataClient(api_key, api_secret)
        self._cache = _DataCache(ttl_sec=cache_ttl_sec)
        self.data_feed = data_feed.lower()

    def _market_session(self, now: datetime | None = None) -> str:
        """Return the current US equity session in New York time."""
        now_et = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo("America/New_York"))
        mins = now_et.hour * 60 + now_et.minute
        day = now_et.weekday()

        if day < 5 and 4 * 60 <= mins < 9 * 60 + 30:
            return "premarket"
        if day < 5 and 9 * 60 + 30 <= mins < 16 * 60:
            return "regular"
        if day < 5 and 16 * 60 <= mins < 20 * 60:
            return "afterhours"
        if (day == 6 and mins >= 20 * 60) or (day in {0, 1, 2, 3} and mins >= 20 * 60) or (day in {0, 1, 2, 3, 4} and mins < 4 * 60):
            return "overnight"
        return "closed"

    def _feeds(self, now: datetime | None = None) -> list[DataFeed]:
        feed_map = {
            "iex": DataFeed.IEX,
            "sip": DataFeed.SIP,
            "delayed_sip": DataFeed.DELAYED_SIP,
            "boats": DataFeed.BOATS,
            "overnight": DataFeed.OVERNIGHT,
            "otc": DataFeed.OTC,
        }
        if self.data_feed == "auto":
            session = self._market_session(now)
            if session == "overnight":
                return [DataFeed.BOATS, DataFeed.OVERNIGHT, DataFeed.SIP, DataFeed.IEX, DataFeed.DELAYED_SIP]
            if session in {"premarket", "regular", "afterhours"}:
                return [DataFeed.SIP, DataFeed.IEX, DataFeed.DELAYED_SIP, DataFeed.BOATS, DataFeed.OVERNIGHT]
            return [DataFeed.SIP, DataFeed.IEX, DataFeed.DELAYED_SIP, DataFeed.BOATS, DataFeed.OVERNIGHT]
        return [feed_map.get(self.data_feed, DataFeed.IEX)]

    @staticmethod
    def _timestamp_epoch(ts: object | None) -> float | None:
        if ts is None:
            return None
        try:
            if isinstance(ts, datetime):
                dt = ts
            else:
                raw = str(ts).strip()
                if not raw:
                    return None
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            return None

    @staticmethod
    def _timestamp_iso(ts: object | None) -> str | None:
        if ts is None:
            return None
        if hasattr(ts, "isoformat"):
            return ts.isoformat()
        raw = str(ts).strip()
        return raw or None

    def _is_better_latest(
        self,
        candidate: LatestPrice,
        current: LatestPrice | None,
        *,
        feed_rank: dict[str, int],
    ) -> bool:
        if current is None:
            return True
        candidate_ts = self._timestamp_epoch(candidate.timestamp)
        current_ts = self._timestamp_epoch(current.timestamp)
        if candidate_ts is not None and current_ts is not None and abs(candidate_ts - current_ts) > 1e-6:
            return candidate_ts > current_ts
        if candidate_ts is not None and current_ts is None:
            return True
        if candidate_ts is None and current_ts is not None:
            return False

        source_rank = {"quote_mid": 0, "latest_trade": 1}
        candidate_source = source_rank.get(candidate.source, 9)
        current_source = source_rank.get(current.source, 9)
        if candidate_source != current_source:
            return candidate_source < current_source
        return feed_rank.get(candidate.feed, 99) < feed_rank.get(current.feed, 99)

    def _mark_staleness(self, latest: LatestPrice, *, session: str, now: datetime | None = None) -> LatestPrice:
        ts_epoch = self._timestamp_epoch(latest.timestamp)
        if ts_epoch is None:
            return replace(latest, session=session)
        now_epoch = (now or datetime.now(timezone.utc)).timestamp()
        age = max(0.0, now_epoch - ts_epoch)
        try:
            default_stale_sec = int(os.getenv("BOT_LATEST_PRICE_STALE_SEC", "300") or 300)
        except ValueError:
            default_stale_sec = 300
        stale_sec = default_stale_sec
        if latest.feed in {"delayed_sip", "overnight"}:
            stale_sec = max(stale_sec, 20 * 60)
        if session == "closed":
            stale_sec = max(stale_sec, 24 * 60 * 60)
        return replace(latest, age_seconds=age, stale=age > stale_sec, session=session)

    def get_latest_price(self, symbol: str) -> LatestPrice:
        prices = self.get_latest_prices([symbol])
        sym = symbol.upper()
        if sym not in prices:
            raise ValueError(f"No latest price for {sym}")
        return prices[sym]

    def get_latest_prices(self, symbols: list[str] | tuple[str, ...]) -> dict[str, LatestPrice]:
        """Fetch the freshest available stock price for the configured feed.

        Prefer quote midpoint when bid/ask are valid, then fall back to latest trade.
        This intentionally bypasses the historical-bars cache.
        """
        clean = list(dict.fromkeys(s.strip().upper() for s in symbols if s and s.strip()))
        if not clean:
            return {}

        out: dict[str, LatestPrice] = {}
        errors: list[str] = []
        now = datetime.now(timezone.utc)
        session = self._market_session(now)
        feeds = self._feeds(now)
        feed_rank = {feed.value: i for i, feed in enumerate(feeds)}
        for feed in feeds:
            try:
                quote_req = StockLatestQuoteRequest(symbol_or_symbols=clean, feed=feed)
                quote_res = self.client.get_stock_latest_quote(quote_req)
                for sym in clean:
                    quote = quote_res.get(sym) if isinstance(quote_res, dict) else quote_res
                    bid = float(getattr(quote, "bid_price", 0.0) or 0.0)
                    ask = float(getattr(quote, "ask_price", 0.0) or 0.0)
                    if bid > 0 and ask > 0 and ask >= bid:
                        ts = getattr(quote, "timestamp", None)
                        candidate = LatestPrice(
                            symbol=sym,
                            price=(bid + ask) / 2.0,
                            source="quote_mid",
                            feed=feed.value,
                            bid=bid,
                            ask=ask,
                            timestamp=self._timestamp_iso(ts),
                        )
                        if self._is_better_latest(candidate, out.get(sym), feed_rank=feed_rank):
                            out[sym] = candidate
            except Exception as exc:
                errors.append(f"{feed.value} quote: {exc}")

            if self.data_feed != "auto" and all(sym in out for sym in clean):
                break

            try:
                trade_req = StockLatestTradeRequest(symbol_or_symbols=clean, feed=feed)
                trade_res = self.client.get_stock_latest_trade(trade_req)
                for sym in clean:
                    trade = trade_res.get(sym) if isinstance(trade_res, dict) else trade_res
                    price = float(getattr(trade, "price", 0.0) or 0.0)
                    if price > 0:
                        ts = getattr(trade, "timestamp", None)
                        candidate = LatestPrice(
                            symbol=sym,
                            price=price,
                            source="latest_trade",
                            feed=feed.value,
                            timestamp=self._timestamp_iso(ts),
                        )
                        if self._is_better_latest(candidate, out.get(sym), feed_rank=feed_rank):
                            out[sym] = candidate
            except Exception as exc:
                errors.append(f"{feed.value} trade: {exc}")

        return {
            sym: self._mark_staleness(latest, session=session, now=now)
            for sym, latest in out.items()
        }

    def with_latest_price(self, bars: pd.DataFrame, latest: LatestPrice) -> pd.DataFrame:
        """Return bars with the final row's OHLC adjusted to the latest price."""
        if bars.empty:
            return bars
        patched = bars.copy()
        idx = patched.index[-1]
        price = float(latest.price)
        patched.loc[idx, "close"] = price
        if "high" in patched.columns:
            patched.loc[idx, "high"] = max(float(patched.loc[idx, "high"]), price)
        if "low" in patched.columns:
            patched.loc[idx, "low"] = min(float(patched.loc[idx, "low"]), price)
        if "open" in patched.columns and pd.isna(patched.loc[idx, "open"]):
            patched.loc[idx, "open"] = price
        return patched

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
            feed=self._feeds()[0] if self.data_feed != "auto" else DataFeed.IEX,
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
            feed=self._feeds()[0] if self.data_feed != "auto" else DataFeed.IEX,
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
