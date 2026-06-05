"""Symbol universe loader.

Fetch commonly used US index universes from public sources, cache locally, and
fall back to hard-coded liquid lists if offline.

Usage
-----
    from ai_trading.data.universe import load_universe
    syms = load_universe(["sp500", "nasdaq100", "dow30"])  # de-duplicated
    syms = load_universe(["all"])                           # all known aliases

CLI cache refresh:
    python -m ai_trading.data.universe --refresh
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

CACHE_DIR = Path(os.getenv("BOT_UNIVERSE_CACHE", "logs/universe"))
CACHE_TTL_SECONDS = 7 * 24 * 3600   # refresh weekly

_WIKI_SOURCES: dict[str, tuple[str, int, str]] = {
    # alias → (url, table index, column name)
    "sp500":    ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", 0, "Symbol"),
    "sp100":    ("https://en.wikipedia.org/wiki/S%26P_100", -1, "Symbol"),
    "sp400":    ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", -1, "Symbol"),
    "sp600":    ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", -1, "Symbol"),
    "nasdaq100":("https://en.wikipedia.org/wiki/Nasdaq-100", 4, "Ticker"),
    "dow30":    ("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average", 2, "Symbol"),
}

_ISHARES_SOURCES: dict[str, tuple[str, str, str]] = {
    # alias -> (ETF ticker, product id, slug)
    "russell1000": ("IWB", "239707", "ishares-russell-1000-etf"),
    "russell2000": ("IWM", "239710", "ishares-russell-2000-etf"),
    "russell3000": ("IWV", "239714", "ishares-russell-3000-etf"),
}

INDEX_LABELS: dict[str, str] = {
    "sp500": "S&P 500",
    "sp100": "S&P 100",
    "sp400": "S&P MidCap 400",
    "sp600": "S&P SmallCap 600",
    "nasdaq100": "Nasdaq-100",
    "dow30": "Dow Jones 30",
    "russell1000": "Russell 1000 (IWB holdings)",
    "russell2000": "Russell 2000 (IWM holdings)",
    "russell3000": "Russell 3000 (IWV holdings)",
}

ALL_ALIASES = tuple(INDEX_LABELS)


# ── Fallback static lists (used if Wikipedia fetch fails) ─────────────────────
# Dow 30 is small and stable; SP500/Nasdaq100 fallbacks are abbreviated to the
# largest, most liquid names so scanner still works fully offline.

_FALLBACK_DOW30 = [
    "AAPL","AMGN","AMZN","AXP","BA","CAT","CRM","CSCO","CVX","DIS",
    "GS","HD","HON","IBM","JNJ","JPM","KO","MCD","MMM","MRK",
    "MSFT","NKE","NVDA","PG","SHW","TRV","UNH","V","VZ","WMT",
]

_FALLBACK_NDX_TOP = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
    "ADBE","NFLX","AMD","PEP","CSCO","TMUS","INTC","CMCSA","TXN","QCOM",
    "AMGN","HON","AMAT","INTU","BKNG","ISRG","ADP","SBUX","GILD","VRTX",
    "MDLZ","ADI","REGN","LRCX","PYPL","PANW","KLAC","SNPS","CDNS","MU",
    "MELI","ASML","ABNB","CRWD","ORLY","MAR","FTNT","CTAS","CHTR","NXPI",
]

_FALLBACK_SPX_TOP = list(dict.fromkeys(_FALLBACK_DOW30 + _FALLBACK_NDX_TOP + [
    "BRK-B","XOM","JPM","LLY","TSM","WMT","UNH","MA","V","PG","JNJ","ORCL",
    "HD","BAC","ABBV","CVX","KO","WFC","MRK","PFE","TMO","ABT","ACN","COST",
    "DHR","LIN","NKE","MCD","DIS","TXN","NEE","RTX","UPS","LOW","CAT","HON",
    "SPGI","GS","BKNG","IBM","ELV","BLK","T","SBUX","GE","PLD","DE","CB",
    "ADP","MMC","AXP","NOW","SYK","TJX","ISRG","AMT","MDT","GILD","C","MO",
    "ZTS","SCHW","ADI","BSX","PYPL","REGN","CI","PGR","SO","DUK","BMY","MMM",
    "EOG","FI","LMT","CL","CME","WM","ITW","HCA","SLB","NSC","PNC","USB",
    "MCK","FCX","APD","CSX","GD","GM","TFC","TRV","AON","ICE","SHW","PSX",
    "EMR","D","COF","MPC","FDX","BDX","ECL","MRNA","KMB","DG","ROST","NOC",
    "ORLY","KMI","WMB","HUM","AEP","MAR","PSA","AIG","JCI","F","COP","OXY",
    "PXD","HES","VLO","KHC","STZ","HSY","GIS","K","CLX","CHD","CPB",
]))

_FALLBACK_SP100_TOP = list(dict.fromkeys(_FALLBACK_DOW30 + [
    "ABBV","ABT","ACN","ADBE","AIG","AMD","AMGN","AMT","AVGO","BAC",
    "BKNG","BLK","BMY","BRK-B","C","CHTR","CMCSA","COF","COP","COST",
    "DHR","DUK","EMR","FDX","GE","GILD","GOOG","GOOGL","INTC","INTU",
    "LLY","LIN","LOW","MA","MDLZ","META","MO","NEE","NFLX","ORCL",
    "PEP","PFE","PYPL","QCOM","RTX","SBUX","SO","T","TGT","TMO",
    "TXN","USB","WBA","WFC",
]))

_FALLBACK_SP400_TOP = [
    "DECK","WSM","BLDR","RS","FIX","MANH","CSL","ITT","RPM","EME",
    "CASY","UTHR","LECO","MEDP","SAIA","USFD","WING","COKE","AIT","MIDD",
    "TREX","GNTX","OLED","THC","NYT","XPO","RGA","PEN","FDS","NBIX",
    "LII","AA","LAD","WTRG","CBSH","ATR","EWBC","ORI","JBL","NDSN",
]

_FALLBACK_SP600_TOP = [
    "MARA","ENSG","UFPI","SPSC","CALM","MMSI","EXTR","KAI","SMCI","TMDX",
    "BOOT","ACA","ALRM","GLDD","HIMS","INSP","FN","CRDO","STRL","BE",
    "ONTO","CELH","AEIS","ATI","AVAV","SKY","ASO","ARCB","HRI","MGY",
]

_FALLBACK_RUSSELL1000_TOP = list(dict.fromkeys(_FALLBACK_SPX_TOP + [
    "TSLA","SHOP","UBER","COIN","HOOD","PLTR","SNOW","DDOG","NET","CRWD",
    "DASH","TTD","RBLX","SE","SQ","AFRM","APP","KVUE","GEV","VST",
]))

_FALLBACK_RUSSELL2000_TOP = [
    "BE","CRDO","FN","STRL","NX","SFM","MARA","LUMN","HIMS","OSCR",
    "IONQ","RKLB","ACHR","RIOT","CAVA","VKTX","TMDX","SOUN","UPST","ASTS",
    "AAOI","ALAB","CIFR","IREN","CORZ","COHR","GTLB","DJT","BBAI","SMR",
    "WULF","CLOV","ENSG","UFPI","SPSC","CALM","BOOT","ACA","KAI","MMSI",
]

_FALLBACK_RUSSELL3000_TOP = list(dict.fromkeys(_FALLBACK_RUSSELL1000_TOP + _FALLBACK_RUSSELL2000_TOP))


def _cache_path(alias: str) -> Path:
    return CACHE_DIR / f"{alias}.json"


def _read_cache(alias: str) -> list[str] | None:
    p = _cache_path(alias)
    if not p.exists():
        return None
    try:
        mtime = p.stat().st_mtime
        if (time.time() - mtime) > CACHE_TTL_SECONDS:
            return None
        data = json.loads(p.read_text())
        if isinstance(data, list) and data:
            return [str(s) for s in data]
    except Exception:
        return None
    return None


def _write_cache(alias: str, symbols: list[str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        _cache_path(alias).write_text(json.dumps(symbols))
    except OSError:
        pass


def _fetch_wiki(alias: str) -> list[str]:
    """Fetch tickers from Wikipedia. Returns [] on any failure."""
    if alias not in _WIKI_SOURCES:
        return []
    url, table_idx, col = _WIKI_SOURCES[alias]
    try:
        import io

        import pandas as pd
        import urllib.request

        # Wikipedia rejects the default pandas/urllib UA with HTTP 403.
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        tables = pd.read_html(io.StringIO(html))
        if table_idx < 0 or table_idx >= len(tables):
            for t in tables:
                if col in getattr(t, "columns", []):
                    return _normalise_symbols(t[col].tolist())
                for alt in ("Ticker", "Symbol", "Ticker symbol"):
                    if alt in getattr(t, "columns", []):
                        return _normalise_symbols(t[alt].tolist())
            return []
        tbl = tables[table_idx]
        if col not in tbl.columns:
            for alt in ("Ticker", "Symbol", "Ticker symbol"):
                if alt in tbl.columns:
                    col = alt
                    break
            else:
                # Final attempt: scan every table for any of those columns
                for t in tables:
                    for alt in ("Symbol", "Ticker", "Ticker symbol"):
                        if alt in getattr(t, "columns", []):
                            return _normalise_symbols(t[alt].tolist())
                return []
        return _normalise_symbols(tbl[col].tolist())
    except Exception as exc:
        print(f"universe: wiki fetch failed for {alias}: {exc}")
        return []


def _fetch_ishares(alias: str) -> list[str]:
    """Fetch ETF holdings from iShares CSV endpoints. Returns [] on failure."""
    if alias not in _ISHARES_SOURCES:
        return []
    etf, product_id, slug = _ISHARES_SOURCES[alias]
    urls = [
        (
            f"https://www.ishares.com/us/products/{product_id}/{slug}/"
            f"1467271812596.ajax?fileType=csv&fileName={etf}_holdings&dataType=fund"
        ),
        (
            f"https://www.ishares.com/us/products/{product_id}/{slug}"
            f"?dataType=fund&fileName={etf}_holdings&fileType=csv"
        ),
    ]
    try:
        import io
        import pandas as pd
        import urllib.request

        for url in urls:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "text/csv,text/plain,*/*",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    text = resp.read().decode("utf-8-sig", errors="replace")
            except Exception:
                continue
            lines = [line for line in text.splitlines() if line.strip()]
            header_idx = None
            for i, line in enumerate(lines[:80]):
                lower = line.lower()
                if "ticker" in lower and ("name" in lower or "asset class" in lower):
                    header_idx = i
                    break
            if header_idx is None:
                continue
            csv_text = "\n".join(lines[header_idx:])
            try:
                df = pd.read_csv(io.StringIO(csv_text), on_bad_lines="skip")
            except TypeError:
                df = pd.read_csv(io.StringIO(csv_text))
            cols = {str(c).strip().lower(): c for c in df.columns}
            ticker_col = cols.get("ticker") or cols.get("holding ticker") or cols.get("symbol")
            if ticker_col is None:
                continue
            if "asset class" in cols:
                asset_col = cols["asset class"]
                df = df[df[asset_col].astype(str).str.lower().str.contains("equity|stock", na=False)]
            return _normalise_symbols(df[ticker_col].tolist())
    except Exception as exc:
        print(f"universe: iShares fetch failed for {alias}: {exc}")
    return []


def _normalise_symbols(raw: list) -> list[str]:
    """Clean tickers: uppercase, strip, replace '.' with '-' for yfinance (BRK.B → BRK-B)."""
    out: list[str] = []
    seen: set[str] = set()
    known_class_symbols = {
        "BRKB": "BRK-B",
        "BRKA": "BRK-A",
        "BFB": "BF-B",
        "BFA": "BF-A",
        "HEIA": "HEI-A",
    }
    for s in raw:
        if s is None:
            continue
        t = str(s).strip().upper().replace(".", "-")
        t = known_class_symbols.get(t, t)
        # Strip footnote markers like "BRK-B[a]"
        if "[" in t:
            t = t.split("[", 1)[0].strip()
        if (
            t
            and t not in {"-", "—", "CASH", "USD", "US DOLLAR", "US DOLLARS"}
            and t not in seen
            and len(t) <= 8
        ):
            seen.add(t)
            out.append(t)
    return out


def _fallback(alias: str) -> list[str]:
    if alias == "sp500":
        return list(_FALLBACK_SPX_TOP)
    if alias == "sp100":
        return list(_FALLBACK_SP100_TOP)
    if alias == "sp400":
        return list(_FALLBACK_SP400_TOP)
    if alias == "sp600":
        return list(_FALLBACK_SP600_TOP)
    if alias == "nasdaq100":
        return list(_FALLBACK_NDX_TOP)
    if alias == "dow30":
        return list(_FALLBACK_DOW30)
    if alias == "russell1000":
        return list(_FALLBACK_RUSSELL1000_TOP)
    if alias == "russell2000":
        return list(_FALLBACK_RUSSELL2000_TOP)
    if alias == "russell3000":
        return list(_FALLBACK_RUSSELL3000_TOP)
    return []


def get_index(alias: str, *, refresh: bool = False) -> list[str]:
    """Return tickers for a single index alias. Tries cache -> source -> fallback."""
    alias = alias.lower()
    known = set(_WIKI_SOURCES) | set(_ISHARES_SOURCES)
    if alias not in known:
        raise ValueError(f"Unknown index alias '{alias}'. Use: {list(INDEX_LABELS)}")
    if not refresh:
        cached = _read_cache(alias)
        if cached:
            return cached
    fresh = _fetch_wiki(alias) if alias in _WIKI_SOURCES else _fetch_ishares(alias)
    if fresh:
        _write_cache(alias, fresh)
        return fresh
    # Last resort: stale cache or hard-coded fallback
    stale = _read_cache(alias)
    if stale:
        return stale
    return _fallback(alias)


def load_universe(aliases: list[str], *, refresh: bool = False) -> list[str]:
    """Combine multiple index aliases (de-duplicated, order-preserved).

    Special alias 'all' expands to all known indexes.
    Bare tickers (anything not in _WIKI_SOURCES) pass through.
    """
    out: list[str] = []
    seen: set[str] = set()
    for a in aliases:
        a = a.strip().lower()
        if not a:
            continue
        if a == "all":
            members = list(ALL_ALIASES)
        elif a in INDEX_LABELS:
            members = [a]
        else:
            # Treat as a bare ticker
            t = a.upper().replace(".", "-")
            if t not in seen:
                seen.add(t)
                out.append(t)
            continue
        for idx in members:
            for sym in get_index(idx, refresh=refresh):
                if sym not in seen:
                    seen.add(sym)
                    out.append(sym)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Symbol universe loader / cache refresh")
    parser.add_argument("aliases", nargs="*", default=["all"],
                        help=f"Index aliases: {', '.join(ALL_ALIASES)}, all (default: all)")
    parser.add_argument("--refresh", action="store_true", help="Force re-fetch from Wikipedia")
    parser.add_argument("--count-only", action="store_true", help="Print count instead of list")
    args = parser.parse_args()

    syms = load_universe(args.aliases, refresh=args.refresh)
    if args.count_only:
        print(len(syms))
    else:
        print(",".join(syms))


if __name__ == "__main__":
    main()
