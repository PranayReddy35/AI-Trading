"""Option chain fetching — Alpaca primary, yfinance fallback.

Returns a normalized `OptionContract` per row with:
- occ_symbol (OCC standardized), underlying, type ('call'|'put'), strike, expiry
- bid, ask, mid, last, volume, open_interest
- iv, delta, theta, gamma, vega (computed via BS if not provided by feed)
- underlying_price (spot at time of fetch)
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from typing import Iterable

import pandas as pd

from ai_trading.options.greeks import bs_greeks, implied_vol

logger = logging.getLogger("ai_trading.options")

_OCC_RE = re.compile(
    r"^(?P<root>[A-Z]{1,6})(?P<y>\d{2})(?P<m>\d{2})(?P<d>\d{2})(?P<cp>[CP])(?P<strike>\d{8})$"
)


@dataclass(slots=True)
class OptionContract:
    occ_symbol: str
    underlying: str
    type: str          # 'call' | 'put'
    strike: float
    expiry: date       # expiration date
    dte: int           # days to expiry
    bid: float
    ask: float
    mid: float
    last: float
    volume: int
    open_interest: int
    iv: float
    delta: float
    gamma: float
    theta: float
    vega: float
    underlying_price: float
    source: str        # 'alpaca' | 'yfinance'

    def to_dict(self) -> dict:
        d = asdict(self)
        d["expiry"] = self.expiry.isoformat()
        return d


def parse_occ(symbol: str) -> dict | None:
    """Parse OCC standard symbol e.g. 'AAPL250620C00200000'.
    Returns dict or None if not OCC-format.
    """
    m = _OCC_RE.match(symbol)
    if not m:
        return None
    g = m.groupdict()
    return {
        "underlying": g["root"],
        "expiry": date(2000 + int(g["y"]), int(g["m"]), int(g["d"])),
        "type": "call" if g["cp"] == "C" else "put",
        "strike": int(g["strike"]) / 1000.0,
    }


def _mid(bid: float, ask: float) -> float:
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return ask or bid or 0.0


def _years(dte: int) -> float:
    return max(dte, 0) / 365.0


def _ensure_greeks(
    contract_dict: dict,
    spot: float,
    risk_free: float,
) -> dict:
    """Populate iv + greeks if missing using BS."""
    T = _years(contract_dict["dte"])
    K = contract_dict["strike"]
    opt_type = contract_dict["type"]
    mid = contract_dict["mid"]

    iv = contract_dict.get("iv") or 0.0
    if not iv and mid > 0 and T > 0:
        iv = implied_vol(mid, spot, K, T, risk_free, opt_type)
        if iv != iv:  # NaN
            iv = 0.0

    if iv > 0 and T > 0:
        g = bs_greeks(spot, K, T, risk_free, iv, opt_type)
        contract_dict.setdefault("delta", g.delta)
        contract_dict.setdefault("gamma", g.gamma)
        contract_dict.setdefault("theta", g.theta)
        contract_dict.setdefault("vega", g.vega)
    contract_dict["iv"] = iv
    for k in ("delta", "gamma", "theta", "vega"):
        contract_dict.setdefault(k, 0.0)
    return contract_dict


# ─────────────────────────────────────────────────────────────────────────────
# Alpaca source
# ─────────────────────────────────────────────────────────────────────────────

def _alpaca_clients():
    api_key = os.environ.get("APCA_API_KEY_ID", "")
    api_secret = os.environ.get("APCA_API_SECRET_KEY", "")
    if not api_key or not api_secret:
        raise RuntimeError("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY")
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.trading.client import TradingClient
    return (
        OptionHistoricalDataClient(api_key, api_secret),
        TradingClient(api_key, api_secret, paper=True),
    )


def _alpaca_spot(symbol: str) -> float:
    api_key = os.environ.get("APCA_API_KEY_ID", "")
    api_secret = os.environ.get("APCA_API_SECRET_KEY", "")
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestTradeRequest
    client = StockHistoricalDataClient(api_key, api_secret)
    req = StockLatestTradeRequest(symbol_or_symbols=symbol)
    res = client.get_stock_latest_trade(req)
    if isinstance(res, dict):
        return float(res[symbol].price)
    return float(res.price)


def _fetch_chain_alpaca(
    symbol: str,
    expiry: date | None = None,
    expiry_gte: date | None = None,
    expiry_lte: date | None = None,
    contract_type: str | None = None,
    strike_gte: float | None = None,
    strike_lte: float | None = None,
    risk_free: float = 0.045,
) -> list[OptionContract]:
    from alpaca.data.requests import OptionChainRequest
    from alpaca.trading.enums import ContractType

    opt_client, _ = _alpaca_clients()
    spot = _alpaca_spot(symbol)

    kwargs: dict = {"underlying_symbol": symbol}
    if expiry is not None:
        kwargs["expiration_date"] = expiry
    if expiry_gte is not None:
        kwargs["expiration_date_gte"] = expiry_gte
    if expiry_lte is not None:
        kwargs["expiration_date_lte"] = expiry_lte
    if strike_gte is not None:
        kwargs["strike_price_gte"] = strike_gte
    if strike_lte is not None:
        kwargs["strike_price_lte"] = strike_lte
    if contract_type:
        kwargs["type"] = ContractType.CALL if contract_type == "call" else ContractType.PUT

    req = OptionChainRequest(**kwargs)
    snap = opt_client.get_option_chain(req)
    # snap: dict[occ_symbol -> OptionsSnapshot]

    out: list[OptionContract] = []
    today = date.today()
    for occ, s in snap.items():
        parsed = parse_occ(occ)
        if not parsed:
            continue
        if contract_type and parsed["type"] != contract_type:
            continue
        dte = (parsed["expiry"] - today).days
        quote = getattr(s, "latest_quote", None)
        trade = getattr(s, "latest_trade", None)
        greeks = getattr(s, "greeks", None)
        iv = float(getattr(s, "implied_volatility", 0.0) or 0.0)
        bid = float(getattr(quote, "bid_price", 0.0) or 0.0) if quote else 0.0
        ask = float(getattr(quote, "ask_price", 0.0) or 0.0) if quote else 0.0
        last = float(getattr(trade, "price", 0.0) or 0.0) if trade else 0.0
        d = {
            "occ_symbol": occ,
            "underlying": parsed["underlying"],
            "type": parsed["type"],
            "strike": parsed["strike"],
            "expiry": parsed["expiry"],
            "dte": dte,
            "bid": bid,
            "ask": ask,
            "mid": _mid(bid, ask) or last,
            "last": last,
            "volume": 0,
            "open_interest": 0,
            "iv": iv,
            "delta": float(getattr(greeks, "delta", 0.0) or 0.0) if greeks else 0.0,
            "gamma": float(getattr(greeks, "gamma", 0.0) or 0.0) if greeks else 0.0,
            "theta": float(getattr(greeks, "theta", 0.0) or 0.0) if greeks else 0.0,
            "vega":  float(getattr(greeks, "vega",  0.0) or 0.0) if greeks else 0.0,
            "underlying_price": spot,
            "source": "alpaca",
        }
        d = _ensure_greeks(d, spot, risk_free)
        out.append(OptionContract(**d))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# yfinance source
# ─────────────────────────────────────────────────────────────────────────────

def _yf_spot(symbol: str) -> float:
    import yfinance as yf
    t = yf.Ticker(symbol)
    info = t.history(period="1d")
    if info is None or info.empty:
        return 0.0
    return float(info["Close"].iloc[-1])


def _fetch_chain_yfinance(
    symbol: str,
    expiry: date | None = None,
    expiry_gte: date | None = None,
    expiry_lte: date | None = None,
    contract_type: str | None = None,
    strike_gte: float | None = None,
    strike_lte: float | None = None,
    risk_free: float = 0.045,
) -> list[OptionContract]:
    import yfinance as yf

    t = yf.Ticker(symbol)
    spot = _yf_spot(symbol)
    expiries_str = t.options or []
    expiries = [datetime.strptime(e, "%Y-%m-%d").date() for e in expiries_str]

    if expiry is not None:
        expiries = [e for e in expiries if e == expiry]
    if expiry_gte is not None:
        expiries = [e for e in expiries if e >= expiry_gte]
    if expiry_lte is not None:
        expiries = [e for e in expiries if e <= expiry_lte]

    out: list[OptionContract] = []
    today = date.today()
    for e in expiries:
        try:
            chain = t.option_chain(e.strftime("%Y-%m-%d"))
        except Exception as exc:
            logger.warning("yfinance chain fetch failed for %s %s: %s", symbol, e, exc)
            continue

        sides = []
        if contract_type in (None, "call"):
            sides.append(("call", chain.calls))
        if contract_type in (None, "put"):
            sides.append(("put", chain.puts))

        for opt_type, df in sides:
            if df is None or df.empty:
                continue
            for _, row in df.iterrows():
                strike = float(row.get("strike", 0))
                if strike_gte is not None and strike < strike_gte:
                    continue
                if strike_lte is not None and strike > strike_lte:
                    continue
                bid = float(row.get("bid", 0) or 0)
                ask = float(row.get("ask", 0) or 0)
                last = float(row.get("lastPrice", 0) or 0)
                iv = float(row.get("impliedVolatility", 0) or 0)
                vol = int(row.get("volume", 0) or 0)
                oi = int(row.get("openInterest", 0) or 0)
                occ = str(row.get("contractSymbol", ""))
                dte = (e - today).days
                d = {
                    "occ_symbol": occ,
                    "underlying": symbol,
                    "type": opt_type,
                    "strike": strike,
                    "expiry": e,
                    "dte": dte,
                    "bid": bid,
                    "ask": ask,
                    "mid": _mid(bid, ask) or last,
                    "last": last,
                    "volume": vol,
                    "open_interest": oi,
                    "iv": iv,
                    "underlying_price": spot,
                    "source": "yfinance",
                }
                d = _ensure_greeks(d, spot, risk_free)
                out.append(OptionContract(**d))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_chain(
    symbol: str,
    expiry: date | None = None,
    expiry_gte: date | None = None,
    expiry_lte: date | None = None,
    contract_type: str | None = None,
    strike_gte: float | None = None,
    strike_lte: float | None = None,
    source: str = "auto",
    risk_free: float = 0.045,
) -> list[OptionContract]:
    """Fetch option chain. source = 'alpaca' | 'yfinance' | 'auto' (try alpaca, fallback)."""
    symbol = symbol.upper().strip()
    if source == "alpaca":
        return _fetch_chain_alpaca(
            symbol, expiry, expiry_gte, expiry_lte, contract_type,
            strike_gte, strike_lte, risk_free,
        )
    if source == "yfinance":
        return _fetch_chain_yfinance(
            symbol, expiry, expiry_gte, expiry_lte, contract_type,
            strike_gte, strike_lte, risk_free,
        )
    # auto: try alpaca then yfinance
    try:
        out = _fetch_chain_alpaca(
            symbol, expiry, expiry_gte, expiry_lte, contract_type,
            strike_gte, strike_lte, risk_free,
        )
        if out:
            return out
    except Exception as exc:
        logger.warning("Alpaca chain fetch failed for %s: %s — falling back to yfinance", symbol, exc)
    return _fetch_chain_yfinance(
        symbol, expiry, expiry_gte, expiry_lte, contract_type,
        strike_gte, strike_lte, risk_free,
    )


def get_quote(occ_symbol: str, source: str = "auto") -> dict:
    """Return latest bid/ask/mid for a single OCC symbol."""
    parsed = parse_occ(occ_symbol)
    if not parsed:
        raise ValueError(f"Invalid OCC symbol: {occ_symbol}")

    if source in ("auto", "alpaca"):
        try:
            opt_client, _ = _alpaca_clients()
            from alpaca.data.requests import OptionLatestQuoteRequest
            req = OptionLatestQuoteRequest(symbol_or_symbols=occ_symbol)
            res = opt_client.get_option_latest_quote(req)
            q = res[occ_symbol] if isinstance(res, dict) else res
            bid = float(getattr(q, "bid_price", 0) or 0)
            ask = float(getattr(q, "ask_price", 0) or 0)
            return {"occ_symbol": occ_symbol, "bid": bid, "ask": ask, "mid": _mid(bid, ask), "source": "alpaca"}
        except Exception as exc:
            if source == "alpaca":
                raise
            logger.warning("Alpaca quote failed (%s), falling back to yfinance", exc)
    # yfinance — refetch the row from the contract's chain
    chain = _fetch_chain_yfinance(parsed["underlying"], expiry=parsed["expiry"])
    for c in chain:
        if c.occ_symbol == occ_symbol or (c.type == parsed["type"] and abs(c.strike - parsed["strike"]) < 1e-6):
            return {"occ_symbol": occ_symbol, "bid": c.bid, "ask": c.ask, "mid": c.mid, "source": "yfinance"}
    return {"occ_symbol": occ_symbol, "bid": 0.0, "ask": 0.0, "mid": 0.0, "source": "none"}


def list_expirations(
    symbol: str,
    min_dte: int = 0,
    max_dte: int = 60,
    source: str = "auto",
) -> list[date]:
    """List available expirations for a symbol within DTE window."""
    symbol = symbol.upper().strip()
    today = date.today()
    lo = today + timedelta(days=min_dte)
    hi = today + timedelta(days=max_dte)

    if source in ("auto", "alpaca"):
        try:
            from alpaca.trading.requests import GetOptionContractsRequest
            _, trade_client = _alpaca_clients()
            req = GetOptionContractsRequest(
                underlying_symbols=[symbol],
                expiration_date_gte=lo,
                expiration_date_lte=hi,
                limit=10000,
            )
            res = trade_client.get_option_contracts(req)
            contracts = res.option_contracts or []
            uniq = sorted({c.expiration_date for c in contracts if c.expiration_date})
            if uniq:
                return uniq
        except Exception as exc:
            if source == "alpaca":
                raise
            logger.warning("Alpaca expirations failed (%s), falling back to yfinance", exc)

    import yfinance as yf
    t = yf.Ticker(symbol)
    raw = t.options or []
    out = []
    for e in raw:
        try:
            d = datetime.strptime(e, "%Y-%m-%d").date()
            if lo <= d <= hi:
                out.append(d)
        except Exception:
            continue
    return out


def chain_to_dataframe(chain: Iterable[OptionContract]) -> pd.DataFrame:
    """Convert a list of OptionContract → DataFrame for display/scoring."""
    rows = [c.to_dict() for c in chain]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
