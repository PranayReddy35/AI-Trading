"""Streamlit dashboard — Portfolio monitor + Live market scanner.

Run with:
    streamlit run ai_trading/dashboard.py
"""
from __future__ import annotations

import json
import hashlib
import os
import pickle
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable

# Ensure project root is on sys.path so ai_trading is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from ai_trading.env import load_dotenv
from ai_trading.broker.robinhood_snapshot import refresh_snapshots
from ai_trading.research import (
    ResearchPromptSpec,
    build_prompt_for_mode,
    build_research_packet,
    build_debate_prompt,
    build_earnings_prompt,
    build_investment_memo_prompt,
    build_valuation_prompt,
)
from ai_trading.time_utils import app_timezone, format_local_now

load_dotenv(_project_root / ".env")

# ── Load Streamlit Cloud secrets into environment variables ────────────────────
# Streamlit secrets are not real environment variables. Most of the app reads
# os.environ through Settings.from_env(), so copy top-level scalar secrets here.
def _load_streamlit_secrets_into_env() -> None:
    try:
        secrets = st.secrets.to_dict()
    except (AttributeError, FileNotFoundError, KeyError):
        return
    except Exception:
        return
    for key, value in secrets.items():
        if key in os.environ or isinstance(value, (dict, list, tuple, set)):
            continue
        os.environ[str(key)] = str(value)


_load_streamlit_secrets_into_env()

from ai_trading.scanner import is_market_open, scan, scan_live
from ai_trading.data.universe import INDEX_LABELS, load_universe

st.set_page_config(
    page_title="AI Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.35rem; padding-bottom: 3rem; }
      div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.35);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 8px;
        padding: 0.75rem 0.85rem;
        min-height: 108px;
      }
      div[data-testid="stMetricLabel"] p {
        color: #94a3b8;
        font-size: 0.78rem;
      }
      div[data-testid="stMetricValue"] > div {
        font-size: 1.18rem;
        line-height: 1.15;
        white-space: normal;
      }
      div[data-testid="stMetricDelta"] > div {
        white-space: normal;
      }
      .ops-section {
        border-top: 1px solid rgba(148, 163, 184, 0.18);
        padding-top: 1rem;
        margin-top: 1rem;
      }
      .ops-kicker {
        color: #94a3b8;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
      }
      .ops-title {
        color: #e2e8f0;
        font-size: 1.22rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
      }
      .ops-note {
        color: #94a3b8;
        font-size: 0.88rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _render_page_header(title: str, subtitle: str, chips: list[str] | None = None) -> None:
    chip_html = ""
    if chips:
        chip_html = "".join(
            f"<span style='border:1px solid rgba(148,163,184,0.35);border-radius:999px;"
            f"padding:0.22rem 0.55rem;margin-right:0.35rem;color:#cbd5e1;font-size:0.78rem'>{chip}</span>"
            for chip in chips
        )
    st.markdown(
        f"""
        <div style="margin:0.15rem 0 1.0rem 0">
          <div class="ops-kicker">Robinhood Agent Console</div>
          <div style="font-size:1.55rem;font-weight:750;color:#f8fafc;line-height:1.2">{title}</div>
          <div class="ops-note" style="margin-top:0.25rem">{subtitle}</div>
          <div style="margin-top:0.7rem">{chip_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def _scan_executor() -> ThreadPoolExecutor:
    try:
        workers_raw = int(os.environ.get("BOT_DASHBOARD_SCAN_WORKERS", "2") or 2)
    except ValueError:
        workers_raw = 2
    workers = max(1, min(4, workers_raw))
    return ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dashboard-scan")


@st.cache_resource(show_spinner=False)
def _background_tasks() -> dict[str, dict[str, Any]]:
    return {}


_DASH_SCAN_CACHE_DIR = Path(os.getenv("BOT_DASHBOARD_SCAN_CACHE_DIR", ".cache/dashboard_scans"))
_DASH_SCAN_CACHE_TTL_SEC = max(0, int(os.getenv("BOT_DASHBOARD_SCAN_RESULT_TTL_SEC", "300") or 300))


def _dashboard_scan_cache_key(kind: str, *parts: Any) -> str:
    raw = json.dumps([kind, *parts], default=str, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_dashboard_scan_cache(key: str) -> dict[str, Any] | None:
    if _DASH_SCAN_CACHE_TTL_SEC <= 0:
        return None
    path = _DASH_SCAN_CACHE_DIR / f"{key}.pkl"
    if not path.exists():
        return None
    try:
        if time.time() - path.stat().st_mtime > _DASH_SCAN_CACHE_TTL_SEC:
            return None
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _write_dashboard_scan_cache(key: str, payload: dict[str, Any]) -> None:
    if _DASH_SCAN_CACHE_TTL_SEC <= 0:
        return
    try:
        _DASH_SCAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with (_DASH_SCAN_CACHE_DIR / f"{key}.pkl").open("wb") as fh:
            pickle.dump(payload, fh)
    except Exception:
        pass


def _start_background_task(
    key: str,
    fn: Callable[..., dict[str, Any]],
    *args: Any,
    meta: dict[str, Any] | None = None,
    **kwargs: Any,
) -> bool:
    tasks = _background_tasks()
    task = tasks.get(key)
    future = task.get("future") if isinstance(task, dict) else None
    if isinstance(future, Future) and not future.done():
        return False
    tasks[key] = {
        "future": _scan_executor().submit(fn, *args, **kwargs),
        "started": time.time(),
        "meta": meta or {},
    }
    return True


def _poll_background_task(key: str) -> dict[str, Any] | None:
    tasks = _background_tasks()
    task = tasks.get(key)
    if not isinstance(task, dict):
        return None
    future = task.get("future")
    started = float(task.get("started", time.time()) or time.time())
    meta = task.get("meta", {})
    if not isinstance(future, Future):
        tasks.pop(key, None)
        return None
    elapsed = max(0.0, time.time() - started)
    if not future.done():
        return {"status": "running", "elapsed": elapsed, "meta": meta}
    tasks.pop(key, None)
    try:
        return {"status": "done", "elapsed": elapsed, "meta": meta, "result": future.result()}
    except Exception as exc:
        return {"status": "error", "elapsed": elapsed, "meta": meta, "error": exc}


def _schedule_scan_poll(seconds: int = 2) -> None:
    delay_ms = max(500, int(seconds * 1000))
    components.html(
        f"""
        <script>
        setTimeout(function() {{ window.location.reload(); }}, {delay_ms});
        </script>
        """,
        height=0,
    )


def _result_source(results: list[Any], default: str) -> str:
    if not results:
        return default
    first = results[0]
    if isinstance(first, dict):
        return str(first.get("data_source") or default)
    return str(getattr(first, "data_source", default) or default)


def _run_dashboard_scan(
    symbols: tuple[str, ...],
    market_open: bool,
    fast_ma: int,
    slow_ma: int,
    top_n: int,
    scan_depth: str = "deep",
) -> dict[str, Any]:
    deep = str(scan_depth or "deep").lower() == "deep"
    cache_key = _dashboard_scan_cache_key(
        "buy",
        tuple(symbols),
        bool(market_open),
        int(fast_ma),
        int(slow_ma),
        int(top_n),
        "deep" if deep else "fast",
    )
    cached = _read_dashboard_scan_cache(cache_key)
    if cached:
        cached["cache_hit"] = True
        return cached
    if market_open:
        results = scan_live(
            list(symbols),
            top_n=top_n,
            apply_filters=deep,
            use_meta=deep,
            dedup=deep,
        )
        if results:
            payload = {
                "results": results,
                "mode": "live",
                "data_source": _result_source(results, "alpaca"),
                "fallback": False,
                "scan_depth": "deep" if deep else "fast",
            }
            _write_dashboard_scan_cache(cache_key, payload)
            return payload
    results = scan(
        list(symbols),
        fast_ma=fast_ma,
        slow_ma=slow_ma,
        top_n=top_n,
        apply_filters=deep,
        use_meta=deep,
        dedup=deep,
    )
    payload = {
        "results": results,
        "mode": "eod",
        "data_source": _result_source(results, "auto"),
        "fallback": bool(market_open),
        "scan_depth": "deep" if deep else "fast",
    }
    _write_dashboard_scan_cache(cache_key, payload)
    return payload


def _run_sell_dashboard_scan(
    symbols: tuple[str, ...],
    market_open: bool,
    fast_ma: int,
    slow_ma: int,
    scan_depth: str = "deep",
) -> dict[str, Any]:
    top_n = len(symbols)
    deep = str(scan_depth or "deep").lower() == "deep"
    cache_key = _dashboard_scan_cache_key(
        "sell",
        tuple(symbols),
        bool(market_open),
        int(fast_ma),
        int(slow_ma),
        "deep" if deep else "fast",
    )
    cached = _read_dashboard_scan_cache(cache_key)
    if cached:
        cached["cache_hit"] = True
        return cached
    if market_open:
        results = scan_live(
            list(symbols),
            top_n=top_n,
            apply_filters=deep,
            use_meta=deep,
            dedup=False,
        )
        if results:
            payload = {"results": results, "mode": "live", "fallback": False, "scan_depth": "deep" if deep else "fast"}
            _write_dashboard_scan_cache(cache_key, payload)
            return payload
    results = scan(
        list(symbols),
        fast_ma=fast_ma,
        slow_ma=slow_ma,
        top_n=top_n,
        apply_filters=deep,
        use_meta=deep,
        dedup=False,
    )
    payload = {"results": results, "mode": "eod", "fallback": bool(market_open), "scan_depth": "deep" if deep else "fast"}
    _write_dashboard_scan_cache(cache_key, payload)
    return payload

# ── Sidebar ───────────────────────────────────────────────────────────────────
LOCAL_TZ = app_timezone()

_AUTOREFRESH_LOCK_SEC = 120


def _lock_autorefresh(seconds: int = _AUTOREFRESH_LOCK_SEC) -> None:
    """Pause auto-refresh for a short window so long scans can finish and render."""
    now = __import__("time").time()
    lock_until = float(st.session_state.get("_autorefresh_lock_until", 0.0) or 0.0)
    st.session_state["_autorefresh_lock_until"] = max(lock_until, now + max(1, int(seconds)))

st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("---")
journal_path = st.sidebar.text_input(
    "Journal path", value=os.environ.get("BOT_JOURNAL_PATH", "logs/journal.jsonl")
)

# Scanner settings
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Scanner")
_env_symbols = os.environ.get("BOT_SYMBOLS", os.environ.get("BOT_SYMBOL", ""))

# Default watchlist: top liquid US names
_DEFAULT_WATCHLIST = (
    "AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,AVGO,JPM,V,"
    "MA,UNH,XOM,LLY,JNJ,WMT,PG,HD,MRK,BAC,"
    "ABBV,CVX,KO,PEP,ORCL,ADBE,CSCO,MCD,AMD,INTC,"
    "QCOM,TXN,CRM,NFLX,NOW,INTU,IBM,GS,MS,AXP,"
    "CAT,RTX,HON,BA,GE,MMM,DE,F,GM,UBER,"
    "SHOP,SQ,PYPL,COIN,HOOD,SOFI,AFRM,PLTR,RBLX,SNAP,"
    "DIS,CMCSA,SPOT,PARA,WBD,T,VZ,TMUS,AMT,"
    "SPY,QQQ,IWM,DIA,XLK,XLF,XLE,XLV,XLI,XLC,"
    "GLD,SLV,USO,TLT,HYG,EEM,VXX,SQQQ,TQQQ,SPXL,"
    "MU,AMAT,KLAC,LRCX,ASML,MRVL,ARM,SMCI,DELL,HPE,"
    "AMGN,GILD,BIIB,REGN,MRNA,PFE,BMY,ISRG,MDT,ABT,"
    "NKE,TGT,COST,LOW,TJX,SBUX,YUM,CMG,DKNG,ABNB,"
    "OXY,SLB,COP,EOG,PSX,MPC,VLO,HAL,DVN,FANG"
)

@st.cache_data(ttl=3600, show_spinner="Fetching universe from Alpaca…")
def _load_full_universe() -> list[str]:
    """Fetch all active tradable US equity tickers from Alpaca (cached 1 h)."""
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetAssetsRequest
        from alpaca.trading.enums import AssetClass, AssetStatus
        import re as _re
        client = TradingClient(
            os.environ["APCA_API_KEY_ID"],
            os.environ["APCA_API_SECRET_KEY"],
            paper=True,
        )
        assets = client.get_all_assets(
            GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
        )
        _MAIN = {"NYSE", "NASDAQ", "ARCA", "BATS", "AMEX"}
        _CLEAN = _re.compile(r"^[A-Z]{1,5}$")
        return sorted(
            a.symbol for a in assets
            if a.tradable and a.exchange.value in _MAIN and _CLEAN.match(a.symbol)
        )
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner="Fetching index universe…")
def _load_index_universe(aliases: tuple[str, ...], refresh: bool = False) -> list[str]:
    clean = tuple(a.strip().lower() for a in aliases if a and a.strip())
    if not clean:
        return []
    return load_universe(list(clean), refresh=refresh)


def _clean_symbols(raw: list[str] | tuple[str, ...]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for sym in raw:
        s = str(sym or "").strip().upper().replace(".", "-")
        if not s or s in seen:
            continue
        if not s.replace("-", "").isalnum():
            continue
        seen.add(s)
        out.append(s)
    return out

@st.cache_data(ttl=120, show_spinner="Pre-filtering with snapshots…")
def _snapshot_filter(
    symbols: list[str],
    min_price: float,
    max_price: float,
    min_volume: int,
    min_change_pct: float,
    max_change_pct: float,
    exchanges: list[str],
) -> list[str]:
    """Use Alpaca snapshots to rapidly filter symbols before deep scanning.
    Returns symbols that pass all filters. Batches to avoid 414 errors."""
    try:
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.data.requests import StockSnapshotRequest
        client = StockHistoricalDataClient(
            os.environ["APCA_API_KEY_ID"],
            os.environ["APCA_API_SECRET_KEY"],
        )
        passed: list[str] = []
        _SNAP_BATCH = 1000
        for i in range(0, len(symbols), _SNAP_BATCH):
            chunk = symbols[i:i + _SNAP_BATCH]
            try:
                snap_req = StockSnapshotRequest(symbol_or_symbols=chunk)
                snaps = client.get_stock_snapshot(snap_req)
                for sym, s in snaps.items():
                    try:
                        price = float(s.latest_trade.price) if s.latest_trade else 0.0
                        vol   = int(s.daily_bar.volume) if s.daily_bar else 0
                        chg   = float(s.daily_bar.percent_change) if (s.daily_bar and hasattr(s.daily_bar, "percent_change")) else (
                            (float(s.daily_bar.close) - float(s.daily_bar.open)) / float(s.daily_bar.open) * 100
                            if s.daily_bar and s.daily_bar.open else 0.0
                        )
                        if (min_price <= price <= max_price
                                and vol >= min_volume
                                and min_change_pct <= chg <= max_change_pct):
                            passed.append(sym)
                    except Exception:
                        continue
            except Exception:
                # If snapshot fails for a chunk, pass those symbols through unfiltered
                passed.extend(chunk)
        return passed
    except Exception:
        return symbols  # fallback: no filter

# ── Workspace navigation ──────────────────────────────────────────────────────
_qp_page = st.query_params.get("page", "portfolio")
_PAGE_LABELS = ["Portfolio", "Action Queue", "Buy Scanner", "Sell Scanner", "Position Advisor", "Patterns", "Options", "Research"]
_PAGE_KEYS = ["portfolio", "actions", "scanner", "sell", "advisor", "patterns", "options", "research"]
_page_idx = _PAGE_KEYS.index(_qp_page) if _qp_page in _PAGE_KEYS else 0
active_page = st.sidebar.radio(
    "Workspace",
    _PAGE_LABELS,
    index=_page_idx,
    key="active_page",
)
_idx = _PAGE_LABELS.index(active_page) if active_page in _PAGE_LABELS else 0
active_key = _PAGE_KEYS[_idx]
st.query_params["page"] = active_key
st.sidebar.caption(
    f"Broker: **{os.environ.get('BOT_BROKER', 'alpaca').upper()}** · "
    f"Stock dry-run: **{os.environ.get('BOT_STOCK_DRY_RUN', 'false').upper()}**"
)
st.sidebar.markdown("---")

# ── Scanner controls ──────────────────────────────────────────────────────────
_default_universe_idx = 3 if _env_symbols else 0


@st.cache_data(ttl=20, show_spinner=False)
def _latest_trade_prices(symbols: tuple[str, ...]) -> dict[str, float]:
    """Fetch latest trade prices from Alpaca for a symbol batch."""
    if not symbols:
        return {}
    try:
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestTradeRequest

        client = StockHistoricalDataClient(
            os.environ["APCA_API_KEY_ID"],
            os.environ["APCA_API_SECRET_KEY"],
        )
        req = StockLatestTradeRequest(symbol_or_symbols=list(symbols))
        trades = client.get_stock_latest_trade(req)
        out: dict[str, float] = {}
        if isinstance(trades, dict):
            for sym, trade in trades.items():
                try:
                    px = float(getattr(trade, "price", 0.0) or 0.0)
                    if px > 0:
                        out[str(sym)] = px
                except Exception:
                    continue
        return out
    except Exception:
        return {}


def _overlay_latest_prices(results: list) -> list:
    """Overwrite scanner close with latest trade prices when available."""
    if not results:
        return results
    symbols = tuple(dict.fromkeys(r.symbol for r in results if getattr(r, "symbol", "")))
    if not symbols:
        return results
    latest = _latest_trade_prices(symbols)
    if not latest:
        return results
    updated: list = []
    for r in results:
        px = latest.get(getattr(r, "symbol", ""))
        if px:
            try:
                updated.append(replace(r, close=px))
                continue
            except Exception:
                pass
        updated.append(r)
    return updated

# ── Universe selector ─────────────────────────────────────────────────────────
_universe_mode = st.sidebar.radio(
    "Universe",
    ["📋 Curated (~130)", "📈 Indexes", "🌐 Full Market (~12K)", "✍️ Custom / .env"],
    index=_default_universe_idx,
)
_use_full_universe = "Full" in _universe_mode
_use_index_universe = "Indexes" in _universe_mode
_use_custom_universe = "Custom" in _universe_mode

_index_alias_options = list(INDEX_LABELS.keys())
_selected_indexes: list[str] = []
_refresh_index_universe = False
if _use_index_universe:
    _selected_indexes = st.sidebar.multiselect(
        "Indexes",
        options=_index_alias_options,
        default=["sp500", "nasdaq100"],
        format_func=lambda a: INDEX_LABELS.get(a, a),
        help="Membership is fetched from public index/ETF holdings sources and cached locally.",
    )
    if st.sidebar.checkbox("Include all supported indexes", value=False):
        _selected_indexes = ["all"]
    _refresh_index_universe = st.sidebar.checkbox("Refresh index membership cache", value=False)

if _use_custom_universe and _env_symbols:
    default_symbols = _env_symbols
elif _use_full_universe:
    default_symbols = ""
elif _use_index_universe:
    default_symbols = ""
else:
    default_symbols = _DEFAULT_WATCHLIST

if not (_use_full_universe or _use_index_universe):
    scanner_symbols_input = st.sidebar.text_area(
        "Watchlist (comma-separated)",
        value=default_symbols,
        height=100,
    )
    _extra_raw = st.sidebar.text_input(
        "➕ Add symbols",
        value="",
        placeholder="TICKER1, TICKER2, …",
    )
    if _extra_raw:
        _base = [s.strip().upper() for s in scanner_symbols_input.split(",") if s.strip()]
        _extra = [s.strip().upper() for s in _extra_raw.split(",") if s.strip()]
        scanner_symbols_input = ",".join(list(dict.fromkeys(_base + _extra)))
else:
    scanner_symbols_input = ""

_exclude_symbols = {
    s.strip().upper()
    for s in os.environ.get("BOT_EXCLUDE_SYMBOLS", "PARA,SQ").split(",")
    if s.strip()
}
_cleanup_excluded_symbols = st.sidebar.checkbox(
    "Remove excluded/dead symbols",
    value=True,
    help="Filters BOT_EXCLUDE_SYMBOLS before scans. Useful for delisted or noisy tickers.",
)


def _apply_symbol_cleanup(symbols: list[str]) -> list[str]:
    cleaned = _clean_symbols(symbols)
    if _cleanup_excluded_symbols and _exclude_symbols:
        cleaned = [s for s in cleaned if s not in _exclude_symbols]
    return cleaned


def _sidebar_snapshot_file_status(path: str, ttl_seconds: int | None = None) -> tuple[bool, str]:
    if not path:
        return False, "missing"
    p = Path(path)
    if not p.exists():
        return False, "missing"
    try:
        ttl = max(30, int(ttl_seconds or os.environ.get("ROBINHOOD_SNAPSHOT_TTL_SEC", "300") or 300))
    except ValueError:
        ttl = 300
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        payload = None
    updated_at = str(payload.get("updated_at") or "") if isinstance(payload, dict) else ""
    try:
        age = (
            max(0.0, (datetime.now(timezone.utc) - parse_ts(updated_at)).total_seconds())
            if updated_at
            else max(0.0, time.time() - p.stat().st_mtime)
        )
    except Exception:
        return False, "unavailable"
    fresh = age <= ttl
    if age < 60:
        age_label = f"{age:.0f}s"
    elif age < 3600:
        age_label = f"{age / 60:.1f}m"
    else:
        age_label = f"{age / 3600:.1f}h"
    return fresh, f"{'fresh' if fresh else 'stale'} · {age_label}"


def _sidebar_universe_symbols(*, load_remote: bool) -> list[str]:
    if _use_full_universe:
        return _apply_symbol_cleanup(_load_full_universe()) if load_remote else []
    if _use_index_universe:
        return _apply_symbol_cleanup(
            _load_index_universe(tuple(_selected_indexes), _refresh_index_universe)
        ) if load_remote else []
    return _apply_symbol_cleanup([s for s in scanner_symbols_input.split(",") if s.strip()])


def _robinhood_refresh_scope_symbols(scope: str) -> list[str]:
    scope_key = str(scope or "").strip().lower()
    if scope_key == "portfolio + watchlist":
        return _sidebar_universe_symbols(load_remote=False)
    if scope_key == "portfolio + scanner universe":
        symbols = _sidebar_universe_symbols(load_remote=_use_full_universe or _use_index_universe)
        return symbols[:500]
    return []

# ── Pre-scan filters (shown when Full Market OR curated with >50 symbols) ────
with st.sidebar.expander("⚡ Pre-scan Filters", expanded=(_use_full_universe or _use_index_universe)):
    _f_min_price = st.number_input("Min price ($)", min_value=0.0, value=5.0, step=1.0)
    _f_max_price = st.number_input("Max price ($)", min_value=0.0, value=5000.0, step=50.0)
    _f_min_vol   = st.number_input("Min daily volume", min_value=0, value=500_000, step=100_000,
                                    format="%d")
    _f_min_chg   = st.number_input("Min daily change %", value=-20.0, step=0.5)
    _f_max_chg   = st.number_input("Max daily change %", value=20.0, step=0.5)
    _use_filters = st.checkbox(
        "Apply filters before scanning",
        value=True if _use_full_universe else False,
        disabled=_use_full_universe,
        help="Required in Full Market mode to keep scan time reasonable.",
    )

scanner_top_n    = st.sidebar.slider("Top N results", 5, 200, 25)
scanner_fast_ma  = st.sidebar.number_input("Fast MA", 3, 20, 5)
scanner_slow_ma  = st.sidebar.number_input("Slow MA", 10, 60, 20)
scanner_scan_depth = st.sidebar.radio(
    "Scan depth",
    ["Fast", "Deep"],
    index=1,
    horizontal=True,
    help="Fast skips expensive meta/quality/correlation enrichment. Deep is better for final decisions.",
)
run_scan         = st.sidebar.button("▶ Run Scanner Now")
if run_scan:
    _lock_autorefresh()

with st.sidebar.expander("Robinhood Quote Overlay", expanded=False):
    robinhood_quote_path = st.text_input(
        "Quote snapshot JSON",
        value=os.environ.get("ROBINHOOD_QUOTES_PATH", "logs/robinhood_quotes.json"),
        help=(
            "Optional local JSON snapshot from Robinhood get_equity_quotes. "
            "Matching scanner prices use this before delayed data."
        ),
    )
    st.caption("Overlays displayed prices only; indicators still use historical bars.")

with st.sidebar.expander("Robinhood Portfolios", expanded=False):
    robinhood_portfolios_path = st.text_input(
        "Portfolio snapshot JSON",
        value=os.environ.get("ROBINHOOD_PORTFOLIOS_PATH", "logs/robinhood_portfolios.json"),
        help=(
            "Local JSON snapshot containing Robinhood Investing and Agentic portfolio "
            "summaries plus equity positions."
        ),
    )
    st.caption("Dashboard reads local snapshots; Robinhood order execution still requires review.")
    _rh_refresh_scope = st.radio(
        "Refresh scope",
        ["Portfolio only", "Portfolio + watchlist", "Portfolio + scanner universe"],
        index=0,
        key="sidebar_robinhood_refresh_scope",
        help="Adds extra symbols to the Robinhood quote refresh beyond your current holdings.",
    )
    if st.button("Refresh Robinhood Snapshot", key="sidebar_refresh_robinhood_snapshot", use_container_width=True):
        try:
            with st.spinner("Refreshing Robinhood quotes and portfolios…"):
                _extra_symbols = _robinhood_refresh_scope_symbols(_rh_refresh_scope)
                result = refresh_snapshots(
                    portfolios_path=robinhood_portfolios_path,
                    quotes_path=robinhood_quote_path,
                    extra_quote_symbols=_extra_symbols,
                )
            st.cache_data.clear()
            for key in (
                "rh_portfolio_rows",
                "rh_portfolio_full",
                "rh_portfolio_ts",
                "scanner_results",
                "scanner_ts",
                "options_results",
                "options_ts",
                "patterns_results",
                "patterns_ts",
            ):
                st.session_state.pop(key, None)
            st.session_state["_rh_rescore_after_refresh"] = True
            st.success(
                f"Robinhood snapshot refreshed: {', '.join(result.account_labels) or 'accounts updated'} "
                f"· {len(result.quote_symbols)} symbols"
            )
            if _extra_symbols:
                st.caption(
                    f"Included {min(len(_extra_symbols), len(result.quote_symbols))} extra symbol"
                    f"{'' if len(_extra_symbols) == 1 else 's'} from {_rh_refresh_scope.lower()}."
                )
            st.info("Robinhood snapshot refreshed. Holdings will be re-scored automatically on the next page render.")
            st.rerun()
        except Exception as exc:
            st.error(f"Robinhood snapshot refresh failed: {exc}")
    _snapshot_ttl = max(30, int(os.environ.get("ROBINHOOD_SNAPSHOT_TTL_SEC", "300") or 300))
    _quotes_fresh, _quotes_status = _sidebar_snapshot_file_status(robinhood_quote_path, ttl_seconds=_snapshot_ttl)
    _portfolios_fresh, _portfolios_status = _sidebar_snapshot_file_status(robinhood_portfolios_path, ttl_seconds=_snapshot_ttl)
    _overall_fresh = _quotes_fresh and _portfolios_fresh
    st.caption(
        f"{'🟢' if _overall_fresh else '🟡'} Quotes: {_quotes_status} · "
        f"Portfolios: {_portfolios_status} · TTL {_snapshot_ttl}s"
    )

st.sidebar.markdown("---")
_autorefresh_on = st.sidebar.checkbox("🔁 Auto-refresh", value=True,
                                       help="Re-runs the page so live prices stay current. "
                                            "Session results (scans, options) are preserved.")
_autorefresh_sec = st.sidebar.slider("Refresh every (sec)", 10, 300, 30, step=5,
                                      disabled=not _autorefresh_on)
st.sidebar.caption(f"Page rendered: {format_local_now('%I:%M:%S %p %Z')}")
if st.sidebar.button("🔄 Refresh Now"):
    # Clear cached data only — keep session_state (scan results etc.)
    st.cache_data.clear()
    st.rerun()

# Auto-refresh skip rules: pause while scan-heavy pages are open, scan is triggered,
# or a short lock window is active while results settle in the UI.
_now_ts = __import__("time").time()
_lock_until = float(st.session_state.get("_autorefresh_lock_until", 0.0) or 0.0)
_refresh_locked = _now_ts < _lock_until
_skip_refresh = (
    active_key in {"actions", "scanner", "sell", "advisor", "patterns", "options"}
) or run_scan or _refresh_locked
if _autorefresh_on and active_key in {"actions", "scanner", "sell", "advisor", "patterns", "options"}:
    st.sidebar.caption("⏸  Auto-refresh paused on this page (results stay until you re-scan)")
elif _autorefresh_on and _refresh_locked:
    _left = int(max(1, _lock_until - _now_ts))
    st.sidebar.caption(f"⏸  Auto-refresh paused while scan results load ({_left}s)")


# ── Plain-English explainers ──────────────────────────────────────────────────

def explain_buy(r) -> str:
    """One-sentence layperson explanation for a buy-scanner result."""
    def _g(obj, key: str, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    rsi = float(_g(r, "rsi", 50) or 50)
    chg = float(_g(r, "change_pct", 0) or 0)
    mom = float(_g(r, "momentum_5d", 0) or 0)
    gap = float(_g(r, "ma_gap_pct", 0) or 0)
    vs = float(_g(r, "volume_surge", 1) or 1)
    sig = str(_g(r, "signal", "NEUTRAL") or "NEUTRAL")

    bits: list[str] = []
    if rsi < 30:
        bits.append("looks oversold (cheap bounce setup)")
    elif rsi > 70:
        bits.append("already running hot — chasing risk")
    if mom >= 5:
        bits.append(f"up {mom:.1f}% over 5 days (strong uptrend)")
    elif mom <= -5:
        bits.append(f"down {abs(mom):.1f}% over 5 days (falling knife risk)")
    if gap >= 3:
        bits.append("trading well above its average — momentum strong")
    elif gap <= -3:
        bits.append("trading below its average — trend is weak")
    if vs >= 2:
        bits.append(f"unusual volume ({vs:.1f}× normal) — big players active")
    if chg >= 3:
        bits.append(f"jumping +{chg:.1f}% today")
    elif chg <= -3:
        bits.append(f"dropping {chg:.1f}% today")

    detail = "; ".join(bits) if bits else "no strong signals either way"
    if sig == "BUY":
        return f"Good entry — {detail}."
    if sig == "WATCH":
        return f"On the edge — {detail}. Wait for confirmation."
    return f"Skip for now — {detail}."


def explain_sell(action: str, drivers: list[str], pnl_pct: float | None, rsi: float, gap: float, chg: float) -> str:
    """One-sentence layperson explanation for a profit-take result."""
    bits: list[str] = []
    if rsi >= 75:
        bits.append(f"RSI {rsi:.0f} means it's very overbought (likely to pull back)")
    elif rsi >= 65:
        bits.append(f"RSI {rsi:.0f} — getting overbought")
    if gap >= 6:
        bits.append(f"{gap:.1f}% above its average — stretched and due to cool off")
    if chg >= 4:
        bits.append(f"big spike today (+{chg:.1f}%) — strength to sell into")
    if pnl_pct is not None:
        if pnl_pct >= 15:
            bits.append(f"you're already up {pnl_pct:.1f}% — lock in the win")
        elif pnl_pct >= 5:
            bits.append(f"sitting on a {pnl_pct:.1f}% gain")
        elif pnl_pct < 0:
            bits.append(f"currently down {pnl_pct:.1f}% — wait, don't sell at a loss")
    detail = "; ".join(bits) if bits else "nothing stretched right now"
    if action == "TAKE PROFIT":
        return f"Sell into this strength — {detail}."
    if action == "TRIM":
        return f"Consider trimming part of the position — {detail}."
    if action == "RISK EXIT":
        return f"Cut downside risk here — {detail}."
    return f"Keep holding — {detail}."


def advise_position(symbol: str, shares: float, avg_cost: float, r, market_value: float | None = None,
                    horizon: str = "short") -> dict:
    """Combine P&L with bot signals → recommendation dict.
    r is a ScanResult from scanner.scan()/scan_live(). Returns:
        {action, emoji, color, rationale[list], price, market_value, total_return, pnl_pct}
    horizon: "short" (swing/day-trade) or "long" (long-term investor — let winners run, only sell on blowoffs).
    """
    if market_value is not None and market_value > 0:
        current_price = market_value / shares if shares > 0 else float(r.close)
    else:
        current_price = float(r.close)
        market_value = current_price * shares
    cost_basis = shares * avg_cost
    total_return = market_value - cost_basis
    pnl_pct = (total_return / cost_basis * 100.0) if cost_basis > 0 else 0.0

    buy_score = float(r.score)
    rsi = float(r.rsi or 50)
    gap = float(r.ma_gap_pct or 0)
    chg = float(r.change_pct or 0)
    mom = float(r.momentum_5d or 0)
    vs  = float(r.volume_surge or 1)
    trend = float(r.trend_consistency or 0)

    long_term = (horizon == "long")
    if long_term:
        # Only flag as "stretched" on true blowoffs; ignore minor extensions in strong trends
        overbought = rsi >= 80 or gap >= 12 or mom >= 20
        # Strong long-term holder: trend>=55% and not weak score — don't sell winners
        strong_holder = trend >= 55 and buy_score >= 40
    else:
        overbought = rsi >= 70 or gap >= 6 or (mom >= 8 and chg >= 3)
        strong_holder = False
    bullish = buy_score >= 65
    weak    = buy_score < 35

    action = "HOLD"; color = "#1e293b"; emoji = "⚪"; rationale: list[str] = []
    if overbought and pnl_pct >= 15:
        action = "TAKE PROFIT (SELL)"; color = "#14532d"; emoji = "💰"
        rationale.append(f"You're up {pnl_pct:.1f}% and the stock is stretched (RSI {rsi:.0f}, {gap:+.1f}% above its average).")
        if long_term:
            rationale.append("Even for a long-term hold this is a true blowoff — trimming or closing protects the gain.")
        else:
            rationale.append("Locking in this gain is the high-probability move — strength like this often cools off.")
    elif overbought and pnl_pct >= 5:
        if long_term and strong_holder:
            action = "HOLD"; color = "#1e293b"; emoji = "⚪"
            rationale.append(f"Up {pnl_pct:.1f}% with a strong long-term trend ({trend:.0f}% consistency) — let it run.")
            rationale.append("Short-term overbought, but the multi-week trend is still intact. No reason to sell a winner.")
        else:
            action = "TRIM"; color = "#713f12"; emoji = "✂️"
            rationale.append(f"You're up {pnl_pct:.1f}% and the chart looks extended (RSI {rsi:.0f}).")
            rationale.append("Sell ~25–50% to lock in some profit, let the rest ride.")
    elif weak and pnl_pct > 0:
        if long_term and trend >= 50:
            action = "HOLD"; color = "#1e293b"; emoji = "⚪"
            rationale.append(f"Short-term setup softened (score {buy_score:.0f}/100) but trend is still {trend:.0f}% — long-term thesis intact.")
            rationale.append("Don't trade out of a multi-month winner on a single weak signal.")
        else:
            action = "SELL"; color = "#450a0a"; emoji = "🔴"
            rationale.append(f"The setup has turned weak (score {buy_score:.0f}/100) and you're still in profit.")
            rationale.append("Take the gain before it slips away.")
    elif weak and pnl_pct <= 0:
        action = "HOLD (cut if stop hit)"; color = "#1e293b"; emoji = "⚪"
        rationale.append(f"Setup is weak but you're already down {pnl_pct:.1f}% — selling here locks the loss.")
        rationale.append("Define a stop (e.g. 8% below cost or recent low) and wait.")
    elif bullish and pnl_pct < -3:
        action = "HOLD"; color = "#1e293b"; emoji = "⚪"
        rationale.append(f"You're down {pnl_pct:.1f}% but the bot still sees a strong setup (score {buy_score:.0f}/100).")
        rationale.append("Don't sell at a loss into a bullish signal — give it room.")
    elif bullish and pnl_pct >= -3 and not overbought:
        action = "BUY MORE"; color = "#14532d"; emoji = "🟢"
        rationale.append(f"Score {buy_score:.0f}/100 — strong bullish setup, RSI {rsi:.0f}, trend {r.trend_consistency:.0f}%.")
        if vs >= 1.8:
            rationale.append(f"Volume is {vs:.1f}× normal — big players are positioning.")
        if pnl_pct > 0:
            rationale.append(f"You're already up {pnl_pct:.1f}% — add on strength, but keep total position size reasonable.")
        else:
            rationale.append("Average down only if your conviction matches the setup — size additions smaller than your initial buy.")
    elif buy_score >= 45:
        action = "HOLD"; color = "#713f12"; emoji = "🟡"
        rationale.append(f"Mixed signals (score {buy_score:.0f}/100) — not a clear buy or sell.")
        rationale.append("Stay put and reassess on the next scan.")
    else:
        action = "HOLD"; color = "#1e293b"; emoji = "⚪"
        rationale.append(f"Nothing standout in the data (score {buy_score:.0f}/100, P&L {pnl_pct:+.1f}%).")
        rationale.append("No reason to act today.")

    # ── Sizing suggestion ─────────────────────────────────────────────────────
    # Compute concrete share count + dollar amount the user should transact.
    sz_shares: float = 0.0
    sz_dollars: float = 0.0
    sz_pct: float = 0.0       # percent of current position (sell) OR percent-add (buy)
    sz_label: str = ""        # short label, e.g. "Sell 50%" / "Add ~$500"

    if action.startswith("TAKE PROFIT"):
        # Close most/all of it. Long-term: leave a "runner" (75% out); short-term: 100%.
        sz_pct = 75.0 if long_term else 100.0
        sz_shares = shares * sz_pct / 100.0
        sz_dollars = sz_shares * current_price
        sz_label = f"Sell {sz_pct:.0f}% (~{sz_shares:.4g} shares ≈ ${sz_dollars:,.0f})"
    elif action == "TRIM":
        # Scale trim with how stretched it is: more overbought → trim more.
        stretch = max(rsi - 70, 0) + max(gap - 6, 0) + max(mom - 8, 0)
        if stretch >= 15:
            sz_pct = 50.0
        elif stretch >= 7:
            sz_pct = 33.0
        else:
            sz_pct = 25.0
        sz_shares = shares * sz_pct / 100.0
        sz_dollars = sz_shares * current_price
        sz_label = f"Sell {sz_pct:.0f}% (~{sz_shares:.4g} shares ≈ ${sz_dollars:,.0f})"
    elif action == "SELL":
        sz_pct = 100.0
        sz_shares = shares
        sz_dollars = sz_shares * current_price
        sz_label = f"Sell all {shares:.4g} shares (≈ ${sz_dollars:,.0f})"
    elif action == "BUY MORE":
        # Scale add with score strength: 65→25%, 75→33%, 85+→50% of current position $$.
        if buy_score >= 85:
            sz_pct = 50.0
        elif buy_score >= 75:
            sz_pct = 33.0
        else:
            sz_pct = 25.0
        # In long-term mode, allow slightly bigger adds on conviction setups
        if long_term and buy_score >= 70 and trend >= 60:
            sz_pct = min(sz_pct + 15.0, 60.0)
        sz_dollars = market_value * sz_pct / 100.0
        sz_shares = sz_dollars / current_price if current_price > 0 else 0.0
        sz_label = f"Add ~{sz_pct:.0f}% of position (~{sz_shares:.4g} shares ≈ ${sz_dollars:,.0f})"
    else:
        sz_label = "No action — hold current size"

    return {
        "symbol": symbol, "action": action, "emoji": emoji, "color": color,
        "rationale": rationale, "price": current_price,
        "market_value": market_value, "total_return": total_return, "pnl_pct": pnl_pct,
        "score": buy_score, "rsi": rsi, "momentum": mom, "gap": gap, "change_pct": chg,
        "vol_surge": vs, "trend": float(r.trend_consistency or 0), "signal": r.signal,
        "driver": r.top_driver,
        "size_shares": sz_shares, "size_dollars": sz_dollars,
        "size_pct": sz_pct, "size_label": sz_label,
    }


def parse_robinhood_csv(file) -> list[dict]:
    """Parse a Robinhood positions/account CSV and return [{symbol, shares, avg_cost, market_value}].
    Tolerant of ragged rows (extra commas inside description fields), currency values with
    embedded commas like ``$1,067.00``, and column-name variants.
    Diagnostic info is attached as ``parse_robinhood_csv.last_diag`` (dict).
    """
    import io as _io
    import re as _re
    import csv as _csv

    # Read raw text so we can pre-clean commas inside currency values
    if hasattr(file, "read"):
        try:
            file.seek(0)
        except Exception:
            pass
        raw_text = file.read()
        if isinstance(raw_text, bytes):
            raw_text = raw_text.decode("utf-8", errors="replace")
    else:
        raw_text = str(file)

    # Strip commas inside $-prefixed numbers: "$1,067.00" -> "$1067.00".
    # Repeat to handle "$1,234,567.00".
    _CURRENCY_COMMA = _re.compile(r"(\$\d+),(\d{3})")
    prev = None
    while prev != raw_text:
        prev = raw_text
        raw_text = _CURRENCY_COMMA.sub(r"\1\2", raw_text)
    # Also strip commas inside bare quoted numbers like "1,234.56"
    _QUOTED_COMMA = _re.compile(r'"(-?\d+),(\d{3}(?:\.\d+)?)"')
    prev = None
    while prev != raw_text:
        prev = raw_text
        raw_text = _QUOTED_COMMA.sub(r'"\1\2"', raw_text)

    # Parse with csv module — handles quoted fields correctly, lets us pad/truncate ragged rows
    reader = _csv.reader(_io.StringIO(raw_text), skipinitialspace=True)
    rows_raw = [row for row in reader if any(c.strip() for c in row)]
    if not rows_raw:
        parse_robinhood_csv.last_diag = {"total_lines": 0, "header": [], "kept": 0, "skipped": []}
        return []

    header = [h.strip() for h in rows_raw[0]]
    n_cols = len(header)
    body = rows_raw[1:]

    # Pad short rows, truncate long ones to header length
    norm_rows = []
    for r in body:
        if len(r) < n_cols:
            r = r + [""] * (n_cols - len(r))
        elif len(r) > n_cols:
            # Join the extras into the last "name/description" field if it makes sense,
            # otherwise just truncate. We truncate from the LEFT-extra after symbol col.
            r = r[:n_cols]
        norm_rows.append([c.strip() for c in r])

    cols = {h.lower().strip(): i for i, h in enumerate(header)}

    def _find(*names):
        for n in names:
            if n in cols:
                return cols[n]
        for key, i in cols.items():
            for n in names:
                if n in key:
                    return i
        return None

    sym_i = _find("symbol", "ticker", "instrument")
    qty_i = _find("shares", "quantity", "qty", "share count")
    cost_i = _find("average cost", "avg cost", "average_cost", "cost basis per share", "average_buy_price", "cost basis", "avg price")
    mv_i = _find("market value", "equity", "current value", "position value")

    if sym_i is None or qty_i is None:
        raise ValueError(
            f"Could not find required columns. Need at least Symbol and Quantity. "
            f"Found columns: {header}"
        )

    def _to_float(v):
        if v is None: return None
        s = str(v).replace("$", "").replace(",", "").replace("%", "").strip()
        if not s or s.lower() in ("nan", "none", "-", "--"):
            return None
        if s.startswith("(") and s.endswith(")"):
            s = "-" + s[1:-1]
        try: return float(s)
        except Exception: return None

    out = []
    skipped: list[dict] = []
    for idx, row in enumerate(norm_rows, start=2):  # +2 = 1-indexed + header line
        sym = row[sym_i].strip().upper()
        if not sym or sym in ("NAN", "NONE", "TOTAL", "TOTALS", "SUBTOTAL", "GRAND TOTAL"):
            skipped.append({"line": idx, "reason": "empty/total row", "row": row[:6]})
            continue
        sym = sym.split()[0]
        if _re.fullmatch(r"-?\d+(\.\d+)?", sym):
            skipped.append({"line": idx, "reason": "numeric in symbol column", "row": row[:6]})
            continue
        if not _re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", sym):
            skipped.append({"line": idx, "reason": f"not a valid ticker ({sym!r})", "row": row[:6]})
            continue
        qty = _to_float(row[qty_i])
        if qty is None or qty <= 0:
            skipped.append({"line": idx, "reason": f"shares={row[qty_i]!r}", "row": row[:6]})
            continue
        avg = _to_float(row[cost_i]) if cost_i is not None else None
        mv = _to_float(row[mv_i]) if mv_i is not None else None
        out.append({
            "symbol": sym,
            "shares": qty,
            "avg_cost": avg if avg is not None else 0.0,
            "market_value": mv if mv is not None else 0.0,
        })

    parse_robinhood_csv.last_diag = {  # type: ignore[attr-defined]
        "total_lines": len(rows_raw),
        "header": header,
        "matched_cols": {
            "symbol": header[sym_i],
            "shares": header[qty_i],
            "avg_cost": header[cost_i] if cost_i is not None else None,
            "market_value": header[mv_i] if mv_i is not None else None,
        },
        "kept": len(out),
        "skipped": skipped,
    }
    return out


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=10)
def load_positions() -> list[dict]:
    """Fetch live open positions from Alpaca with full P&L detail."""
    api_key = os.environ.get("APCA_API_KEY_ID", "")
    api_secret = os.environ.get("APCA_API_SECRET_KEY", "")
    paper = os.environ.get("BOT_PAPER_ONLY", "true").lower() == "true"
    if not api_key or not api_secret:
        return []
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(api_key, api_secret, paper=paper)
        positions = client.get_all_positions()
        rows = []
        for p in positions:
            qty = float(p.qty)
            avg_entry = float(p.avg_entry_price)
            current_price = float(p.current_price)
            market_value = float(p.market_value)
            cost_basis = float(p.cost_basis)
            unrealized_pl = float(p.unrealized_pl)
            unrealized_plpc = float(p.unrealized_plpc) * 100.0
            # Today's return = change_today * qty
            change_today = float(p.change_today) if hasattr(p, "change_today") and p.change_today else 0.0
            todays_pl = change_today * qty
            todays_pl_pct = (change_today / (current_price - change_today) * 100.0) if (current_price - change_today) else 0.0
            rows.append({
                "Symbol": str(p.symbol),
                "Shares": int(qty),
                "Avg Cost": avg_entry,
                "Current Price": current_price,
                "Market Value": market_value,
                "Cost Basis": cost_basis,
                "Today's P&L": todays_pl,
                "Today's P&L %": todays_pl_pct,
                "Total P&L": unrealized_pl,
                "Total P&L %": unrealized_plpc,
            })
        return sorted(rows, key=lambda x: x["Total P&L"], reverse=True)
    except Exception:
        return []


@st.cache_data(ttl=15)
def load_journal(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    records = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def get_latest(records: list[dict], event_type: str) -> dict | None:
    for r in reversed(records):
        if r.get("event_type") == event_type:
            return r.get("payload", {})
    return None


def get_all(records: list[dict], *event_types: str) -> list[dict]:
    return [r for r in records if r.get("event_type") in event_types]


def get_recent_payloads(records: list[dict], event_type: str, limit: int = 25) -> list[dict]:
    out: list[dict] = []
    for record in reversed(records):
        if record.get("event_type") != event_type:
            continue
        payload = record.get("payload", {})
        if isinstance(payload, dict):
            out.append(payload)
        if len(out) >= limit:
            break
    return out


def parse_ts(ts: str) -> datetime:
    try:
        dt = datetime.fromisoformat(ts)
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)


def relative_time(ts: str) -> str:
    if not ts:
        return "—"
    try:
        diff = datetime.now(timezone.utc) - parse_ts(ts)
        s = int(diff.total_seconds())
        if s < 60:    return f"{s}s ago"
        if s < 3600:  return f"{s//60}m ago"
        if s < 86400: return f"{s//3600}h ago"
        return f"{s//86400}d ago"
    except Exception:
        return "—"


def _snapshot_file_status(path: str, ttl_seconds: int | None = None) -> tuple[bool, str]:
    if not path:
        return False, "missing"
    p = Path(path)
    if not p.exists():
        return False, "missing"
    try:
        ttl = max(30, int(ttl_seconds or os.environ.get("ROBINHOOD_SNAPSHOT_TTL_SEC", "300") or 300))
    except ValueError:
        ttl = 300
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        payload = None
    updated_at = ""
    if isinstance(payload, dict):
        updated_at = str(payload.get("updated_at") or "")
    try:
        age = (
            max(0.0, (datetime.now(timezone.utc) - parse_ts(updated_at)).total_seconds())
            if updated_at
            else max(0.0, time.time() - p.stat().st_mtime)
        )
        fresh = age <= ttl
        if age < 60:
            age_label = f"{age:.0f}s"
        elif age < 3600:
            age_label = f"{age / 60:.1f}m"
        else:
            age_label = f"{age / 3600:.1f}h"
        state = "fresh" if fresh else "stale"
        return fresh, f"{state} · {age_label}"
    except Exception:
        return False, "unavailable"


def safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def fmt_money(v: float) -> str:
    """Format dollar amount compactly: $1.23K, $1.23M, etc."""
    if abs(v) >= 1_000_000:
        return f"${v/1_000_000:+.2f}M" if v != abs(v) else f"${v/1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"${v/1_000:+.2f}K" if v != abs(v) else f"${v/1_000:.2f}K"
    return f"${v:+.2f}" if v != abs(v) else f"${v:.2f}"


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _latest_price_dict(latest, *, requested_feed: str, fallback_from: str = "") -> dict:
    return {
        "price": float(latest.price),
        "source": latest.source,
        "feed": latest.feed,
        "timestamp": latest.timestamp,
        "bid": latest.bid,
        "ask": latest.ask,
        "age_seconds": latest.age_seconds,
        "stale": latest.stale,
        "session": latest.session,
        "requested_feed": requested_feed,
        "fallback_from": fallback_from,
    }


def _quote_time_age_seconds(timestamp: str) -> float | None:
    if not timestamp:
        return None
    try:
        return max(0.0, (datetime.now(timezone.utc) - parse_ts(timestamp)).total_seconds())
    except Exception:
        return None


def _robinhood_quote_to_latest(quote: dict) -> dict | None:
    symbol = str(quote.get("symbol") or "").upper()
    if not symbol:
        return None
    ext_price = safe_float(quote.get("last_extended_hours_trade_price"), 0.0)
    last_price = safe_float(quote.get("last_trade_price"), 0.0)
    non_reg_price = safe_float(quote.get("last_non_reg_trade_price"), 0.0)
    ext_time = str(
        quote.get("updated_at")
        or quote.get("last_extended_hours_trade_time")
        or quote.get("venue_last_extended_hours_trade_time")
        or ""
    )
    last_time = str(quote.get("venue_last_trade_time") or "")
    non_reg_time = str(quote.get("venue_last_non_reg_trade_time") or "")
    candidates: list[tuple[str, float, str, str]] = []
    if ext_price > 0:
        candidates.append(("extended", ext_price, ext_time, "quote_last_extended_hours"))
    if non_reg_price > 0:
        candidates.append(("non_regular", non_reg_price, non_reg_time, "quote_last_non_regular"))
    if last_price > 0:
        candidates.append(("regular", last_price, last_time, "quote_last_trade"))
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda item: (parse_ts(item[2]).timestamp() if item[2] else float("-inf"), item[1]),
        reverse=True,
    )
    session, price, timestamp, source = ranked[0]
    if price <= 0:
        return None
    age = _quote_time_age_seconds(timestamp)
    return {
        "price": price,
        "source": source,
        "feed": "robinhood",
        "timestamp": timestamp,
        "bid": safe_float(quote.get("bid_price"), 0.0),
        "ask": safe_float(quote.get("ask_price"), 0.0),
        "age_seconds": age if age is not None else 999999.0,
        "stale": bool(age is None or age > 60),
        "session": session,
        "state": str(quote.get("state") or ""),
        "has_traded": quote.get("has_traded"),
        "requested_feed": "robinhood",
    }


def _extract_robinhood_quotes(payload: Any) -> list[dict]:
    if isinstance(payload, dict):
        if isinstance(payload.get("quote"), dict):
            return [payload["quote"]]
        data = payload.get("data")
        if isinstance(data, dict):
            return _extract_robinhood_quotes(data)
        results = payload.get("results")
        if isinstance(results, list):
            return _extract_robinhood_quotes(results)
    if isinstance(payload, list):
        quotes: list[dict] = []
        for item in payload:
            if isinstance(item, dict) and isinstance(item.get("quote"), dict):
                quotes.append(item["quote"])
            elif isinstance(item, dict):
                quotes.append(item)
        return quotes
    return []


@st.cache_data(ttl=5, show_spinner=False)
def load_robinhood_quote_snapshot(path: str) -> dict[str, dict]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        out: dict[str, dict] = {}
        for quote in _extract_robinhood_quotes(payload):
            latest = _robinhood_quote_to_latest(quote)
            if latest:
                out[str(quote.get("symbol") or "").upper()] = latest
        return out
    except Exception:
        return {}


def _robinhood_snapshot_fresh(path: str, ttl_seconds: int | None = None) -> bool:
    """Return True when the local Robinhood quote snapshot is fresh enough to trust."""
    if not path:
        return False
    ttl = ttl_seconds
    if ttl is None:
        try:
            ttl = max(30, int(os.environ.get("ROBINHOOD_SNAPSHOT_TTL_SEC", "300") or 300))
        except ValueError:
            ttl = 300
    status = _snapshot_status(path, ttl)
    return bool(status.get("fresh"))


@st.cache_data(ttl=5, show_spinner=False)
def _load_latest_price_map_cached(symbols: tuple[str, ...], preferred_feed: str) -> dict[str, dict]:
    clean = tuple(dict.fromkeys(s.strip().upper() for s in symbols if s and s.strip()))
    if not clean:
        return {}
    try:
        from ai_trading.config import Settings
        from ai_trading.data.market_data import AlpacaMarketData

        settings = Settings.from_env()
        if not settings.api_key or not settings.api_secret:
            return {}
        out: dict[str, dict] = {}

        def _fetch(feed: str, symbols_to_fetch: tuple[str, ...]):
            if not symbols_to_fetch:
                return {}
            market_data = AlpacaMarketData(
                settings.api_key,
                settings.api_secret,
                cache_ttl_sec=0,
                data_feed=feed,
            )
            return market_data.get_latest_prices(symbols_to_fetch)

        preferred_feed = (preferred_feed or "auto").strip().lower()
        latest_prices = _fetch(preferred_feed, clean)
        for symbol, latest in latest_prices.items():
            out[symbol] = _latest_price_dict(latest, requested_feed=preferred_feed)

        # If SIP is configured but unavailable, keep dashboard usable with clearly
        # labeled fallback prices instead of blank timestamps and EOD-only closes.
        missing = tuple(sym for sym in clean if sym not in out)
        if preferred_feed == "sip" and missing:
            fallback_prices = _fetch("auto", missing)
            for symbol, latest in fallback_prices.items():
                out[symbol] = _latest_price_dict(
                    latest,
                    requested_feed=preferred_feed,
                    fallback_from="sip",
                )
        for symbol in clean:
            out.setdefault(
                symbol,
                {
                    "error": "latest price unavailable",
                    "requested_feed": preferred_feed,
                    "fallback_from": "sip" if preferred_feed == "sip" else "",
                },
            )
        return out
    except Exception:
        return {}


def load_latest_price_map(symbols: tuple[str, ...]) -> dict[str, dict]:
    try:
        from ai_trading.config import Settings

        preferred_feed = Settings.from_env().market_data_feed
    except Exception:
        preferred_feed = os.environ.get("BOT_MARKET_DATA_FEED", "auto")
    latest = _load_latest_price_map_cached(symbols, str(preferred_feed or "auto").lower())
    rh_path = str(globals().get("robinhood_quote_path", "") or "")
    if rh_path and _robinhood_snapshot_fresh(rh_path):
        rh_quotes = load_robinhood_quote_snapshot(rh_path)
        for symbol in symbols:
            quote = rh_quotes.get(symbol.upper())
            if quote:
                latest[symbol.upper()] = quote
    return latest


def latest_price_for(symbol: str, fallback: float = 0.0) -> tuple[float, str, str]:
    data = load_latest_price_map((symbol.upper(),)).get(symbol.upper(), {})
    price = safe_float(data.get("price"), fallback)
    source = latest_source_label(data)
    timestamp = str(data.get("timestamp") or "")
    return price, source, timestamp


def format_price_time(timestamp: str) -> str:
    if not timestamp:
        return "—"
    try:
        return parse_ts(timestamp).astimezone(LOCAL_TZ).strftime("%Y-%m-%d %I:%M:%S %p %Z")
    except Exception:
        return timestamp


def latest_source_label(latest: dict, fallback: str = "fallback") -> str:
    feed = str(latest.get("feed") or "").upper()
    source = str(latest.get("source") or fallback)
    label = f"{feed} {source}".strip() or fallback
    caution_feeds = {"IEX", "DELAYED_SIP", "BOATS", "OVERNIGHT"}
    flags: list[str] = []
    if feed in caution_feeds:
        flags.append("CAUTION")
    if latest.get("fallback_from"):
        flags.append("SIP-FALLBACK")
    if latest.get("stale"):
        flags.append("STALE")
    if flags:
        return f"{label} {' '.join(flags)}".strip()
    return label


def data_confidence(latest: dict) -> tuple[str, int, str]:
    if not latest or latest.get("error"):
        return "LOW", 0, "latest price unavailable"
    score = 100
    reasons: list[str] = []
    feed = str(latest.get("feed") or "").lower()
    age = safe_float(latest.get("age_seconds"), 999999.0)
    bid = safe_float(latest.get("bid"), 0.0)
    ask = safe_float(latest.get("ask"), 0.0)
    if feed == "robinhood":
        reasons.append("Robinhood")
        state = str(latest.get("state") or "").lower()
        if state and state != "active":
            score -= 40
            reasons.append(f"state {state}")
        if latest.get("has_traded") is False:
            score -= 40
            reasons.append("has not traded")
    elif feed == "sip":
        reasons.append("SIP")
    elif feed == "iex":
        score -= 20
        reasons.append("IEX")
    elif feed in {"delayed_sip", "boats", "overnight"}:
        score -= 35
        reasons.append(feed.upper())
    elif feed:
        score -= 15
        reasons.append(feed.upper())
    else:
        score -= 25
        reasons.append("unknown feed")
    if latest.get("fallback_from"):
        score -= 20
        reasons.append("fallback")
    if latest.get("stale"):
        score -= 30
        reasons.append("stale")
    if age > 900:
        score -= 30
        reasons.append(">15m old")
    elif age > 300:
        score -= 15
        reasons.append(">5m old")
    elif age <= 60:
        reasons.append("fresh")
    if bid > 0 and ask > 0 and ask >= bid:
        mid = (bid + ask) / 2.0
        spread_bps = ((ask - bid) / mid * 1e4) if mid else 0.0
        if spread_bps > 50:
            score -= 25
            reasons.append(f"wide spread {spread_bps:.0f}bps")
        elif spread_bps > 20:
            score -= 10
            reasons.append(f"spread {spread_bps:.0f}bps")
    score = max(0, min(100, int(score)))
    if score >= 80:
        return "HIGH", score, ", ".join(reasons)
    if score >= 55:
        return "MEDIUM", score, ", ".join(reasons)
    return "LOW", score, ", ".join(reasons)


def robinhood_quote_confirmation(symbol: str, latest: dict | None = None) -> tuple[str, str]:
    symbol_u = str(symbol or "").upper()
    if not symbol_u:
        return "UNKNOWN", "No symbol"
    rh_latest = _robinhood_latest_price_map((symbol_u,)).get(symbol_u, {})
    if rh_latest and not rh_latest.get("stale"):
        return "CONFIRMED", (
            f"Fresh Robinhood quote confirmed at {format_price_time(str(rh_latest.get('timestamp') or ''))}"
        )
    latest = latest or {}
    if latest and not latest.get("error"):
        source = latest_source_label(latest, "market data")
        if latest.get("stale"):
            return "REVIEW", f"Robinhood quote unavailable; fallback source {source} is stale."
        return "FALLBACK", f"Robinhood quote unavailable; using current fallback source {source}."
    return "MISSING", "No fresh Robinhood quote or fallback latest price available."


def _obj_get(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _find_scan_result(results: list[Any], symbol: str) -> Any | None:
    symbol = symbol.upper()
    for item in results:
        if str(_obj_get(item, "symbol", "") or "").upper() == symbol:
            return item
    return None


def _mask_account_number(value: Any) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    if not digits:
        return "—"
    return f"****{digits[-4:]}"


def _unwrap_robinhood_data(value: Any) -> Any:
    if isinstance(value, dict) and isinstance(value.get("data"), (dict, list)):
        return value["data"]
    return value


def _coerce_robinhood_positions(value: Any) -> list[dict]:
    value = _unwrap_robinhood_data(value)
    if isinstance(value, dict):
        positions = value.get("positions") or value.get("results") or value.get("equity_positions") or []
    else:
        positions = value
    if not isinstance(positions, list):
        return []
    out: list[dict] = []
    for raw in positions:
        if not isinstance(raw, dict):
            continue
        symbol = str(raw.get("symbol") or raw.get("Symbol") or "").strip().upper()
        if not symbol:
            continue
        shares = safe_float(raw.get("quantity", raw.get("shares", raw.get("Shares"))), 0.0)
        if shares <= 0:
            continue
        avg_cost = safe_float(
            raw.get("average_buy_price", raw.get("avg_cost", raw.get("Avg Cost", raw.get("average_cost")))),
            0.0,
        )
        sellable = safe_float(raw.get("shares_available_for_sells", raw.get("sellable", shares)), shares)
        market_value = safe_float(raw.get("market_value", raw.get("Market Value")), 0.0)
        out.append({
            "symbol": symbol,
            "shares": shares,
            "avg_cost": avg_cost,
            "sellable": sellable,
            "market_value": market_value,
            "type": str(raw.get("type") or "long"),
        })
    return out


def _normalize_robinhood_portfolio_account(raw: dict, default_key: str = "") -> dict | None:
    data = _unwrap_robinhood_data(raw)
    if not isinstance(data, dict):
        return None
    portfolio = _unwrap_robinhood_data(data.get("portfolio") or data.get("summary") or {})
    if not isinstance(portfolio, dict):
        portfolio = {}
    positions = _coerce_robinhood_positions(
        data.get("positions") or data.get("equity_positions") or data.get("holdings") or []
    )
    agentic = bool(data.get("agentic_allowed") or data.get("agentic") or str(default_key).lower() == "agentic")
    nickname = str(data.get("nickname") or data.get("label") or data.get("name") or "").strip()
    label = nickname or ("Agentic" if agentic else "Investing")
    account_number = data.get("account_number") or data.get("rhs_account_number") or data.get("account")
    account_masked = str(data.get("account_masked") or data.get("account_last4") or "").strip()
    if not account_masked:
        account_masked = _mask_account_number(account_number)
    features = data.get("features") if isinstance(data.get("features"), dict) else {}
    return {
        "id": f"{label}:{account_masked}",
        "label": label,
        "account": account_masked,
        "agentic": agentic,
        "features": features,
        "portfolio": portfolio,
        "positions": positions,
        "updated_at": str(data.get("updated_at") or raw.get("updated_at") or ""),
    }


@st.cache_data(ttl=10, show_spinner=False)
def load_robinhood_portfolio_snapshot(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    data = _unwrap_robinhood_data(payload)
    accounts_raw: list[Any] = []
    if isinstance(data, dict):
        if isinstance(data.get("accounts"), list):
            accounts_raw = data["accounts"]
        elif isinstance(data.get("portfolios"), list):
            accounts_raw = data["portfolios"]
        else:
            for key in ("investing", "agentic"):
                if isinstance(data.get(key), dict):
                    item = dict(data[key])
                    item.setdefault("label", "Agentic" if key == "agentic" else "Investing")
                    item.setdefault("agentic", key == "agentic")
                    accounts_raw.append(item)
            if not accounts_raw and ("positions" in data or "portfolio" in data):
                accounts_raw = [data]
    elif isinstance(data, list):
        accounts_raw = data
    accounts: list[dict] = []
    for idx, raw in enumerate(accounts_raw):
        if not isinstance(raw, dict):
            continue
        account = _normalize_robinhood_portfolio_account(raw, default_key=str(idx))
        if account:
            accounts.append(account)
    accounts.sort(key=lambda a: (not bool(a.get("agentic") is False), str(a.get("label", ""))))
    return accounts


def _portfolio_metric(portfolio: dict, *keys: str) -> float:
    for key in keys:
        if key in portfolio:
            return safe_float(portfolio.get(key), 0.0)
    return 0.0


def _portfolio_buying_power(portfolio: dict) -> float:
    bp = portfolio.get("buying_power")
    if isinstance(bp, dict):
        return safe_float(bp.get("buying_power"), 0.0)
    return safe_float(bp, 0.0)


def _robinhood_latest_price_map(symbols: tuple[str, ...]) -> dict[str, dict]:
    rh_path = str(globals().get("robinhood_quote_path", "") or os.environ.get("ROBINHOOD_QUOTES_PATH", ""))
    quotes = load_robinhood_quote_snapshot(rh_path) if rh_path and _robinhood_snapshot_fresh(rh_path) else {}
    return {
        symbol.upper(): quotes[symbol.upper()]
        for symbol in symbols
        if symbol and symbol.upper() in quotes
    }


def _robinhood_snapshot_symbols(include_crypto: bool = False) -> list[str]:
    path = str(globals().get("robinhood_portfolios_path", "") or os.environ.get("ROBINHOOD_PORTFOLIOS_PATH", "logs/robinhood_portfolios.json"))
    accounts = load_robinhood_portfolio_snapshot(path)
    symbols: list[str] = []
    for account in accounts:
        for holding in account.get("positions", []):
            symbol = str(holding.get("symbol", "")).upper()
            if not symbol:
                continue
            if not include_crypto and str(holding.get("type") or "").lower() == "crypto":
                continue
            if symbol not in symbols:
                symbols.append(symbol)
    return symbols


def _robinhood_price_for_holding(holding: dict, latest: dict) -> tuple[float, float, float, float]:
    shares = safe_float(holding.get("shares"), 0.0)
    avg_cost = safe_float(holding.get("avg_cost"), 0.0)
    snapshot_mv = safe_float(holding.get("market_value"), 0.0)
    price = safe_float(latest.get("price"), 0.0)
    if price <= 0 and snapshot_mv > 0 and shares > 0:
        price = snapshot_mv / shares
    market_value = shares * price if price > 0 else snapshot_mv
    cost_basis = shares * avg_cost
    pnl = market_value - cost_basis if market_value and cost_basis else 0.0
    pnl_pct = (pnl / cost_basis * 100.0) if cost_basis > 0 else 0.0
    return price, market_value, pnl, pnl_pct


def _robinhood_equity_pnl(accounts: list[dict]) -> dict[str, float]:
    holdings = [
        holding
        for account in accounts
        for holding in account.get("positions", [])
        if holding.get("symbol")
    ]
    symbols = tuple(dict.fromkeys(str(h.get("symbol", "")).upper() for h in holdings))
    latest_map = _robinhood_latest_price_map(symbols)
    market_value = 0.0
    cost_basis = 0.0
    for holding in holdings:
        symbol = str(holding.get("symbol", "")).upper()
        _price, value, _pnl, _pnl_pct = _robinhood_price_for_holding(holding, latest_map.get(symbol, {}))
        market_value += value
        cost_basis += safe_float(holding.get("shares"), 0.0) * safe_float(holding.get("avg_cost"), 0.0)
    pnl = market_value - cost_basis
    pnl_pct = (pnl / cost_basis * 100.0) if cost_basis > 0 else 0.0
    return {
        "market_value": market_value,
        "cost_basis": cost_basis,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
    }


def _portfolio_net_invested(account: dict) -> float:
    portfolio = account.get("portfolio", {}) if isinstance(account.get("portfolio"), dict) else {}
    equity_cost = 0.0
    crypto_cost = 0.0
    for holding in account.get("positions", []):
        qty = safe_float(holding.get("shares"), 0.0)
        avg_cost = safe_float(holding.get("avg_cost"), 0.0)
        basis = qty * avg_cost
        if str(holding.get("type") or "").lower() == "crypto":
            crypto_cost += basis
        else:
            equity_cost += basis
    cash = _portfolio_metric(portfolio, "cash")
    return equity_cost + crypto_cost + cash


def _snapshot_status(path: str, ttl_seconds: int = 300) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {
            "exists": False,
            "fresh": False,
            "age_seconds": None,
            "label": "missing",
            "path": path,
            "updated_at": "",
        }
    age = max(0.0, time.time() - p.stat().st_mtime)
    updated_at = ""
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            updated_at = str(payload.get("updated_at") or "")
            if updated_at:
                age = max(0.0, (datetime.now(timezone.utc) - parse_ts(updated_at)).total_seconds())
    except Exception:
        pass
    fresh = age <= ttl_seconds
    if age < 60:
        age_label = f"{age:.0f}s"
    elif age < 3600:
        age_label = f"{age / 60:.1f}m"
    else:
        age_label = f"{age / 3600:.1f}h"
    return {
        "exists": True,
        "fresh": fresh,
        "age_seconds": age,
        "label": "fresh" if fresh else "stale",
        "age_label": age_label,
        "path": path,
        "updated_at": updated_at,
    }


def _render_snapshot_status_panel(portfolios_path: str, quotes_path: str) -> None:
    ttl = max(30, int(os.environ.get("ROBINHOOD_SNAPSHOT_TTL_SEC", "300") or 300))
    p_status = _snapshot_status(portfolios_path, ttl)
    q_status = _snapshot_status(quotes_path, ttl)
    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric("Portfolio Snapshot", p_status["label"].upper(), p_status.get("age_label", "—"))
    c2.metric("Quote Snapshot", q_status["label"].upper(), q_status.get("age_label", "—"))
    stale_bits = []
    if not p_status["fresh"]:
        stale_bits.append("portfolio")
    if not q_status["fresh"]:
        stale_bits.append("quotes")
    with c3:
        if stale_bits:
            st.warning(
                "Stale Robinhood snapshot: "
                + ", ".join(stale_bits)
                + ". Refresh snapshots, then use Refresh Now."
            )
        else:
            st.success("Robinhood snapshots are fresh enough for dashboard display.")
    if st.button("Refresh Dashboard Snapshot Cache", key="refresh_snapshot_cache"):
        st.cache_data.clear()
        st.rerun()


def _scanner_buy_queue_rows(records: list[dict], held_symbols: set[str]) -> list[dict]:
    results = st.session_state.get("scanner_results", [])
    if not results:
        return []
    symbols = tuple(
        dict.fromkeys(
            str(_obj_get(item, "symbol", "") or "").upper()
            for item in results
            if _obj_get(item, "symbol", "")
        )
    )
    latest_map = load_latest_price_map(symbols)
    rows: list[dict] = []
    for item in results:
        symbol = str(_obj_get(item, "symbol", "") or "").upper()
        if not symbol or symbol in held_symbols:
            continue
        score = safe_float(_obj_get(item, "score", 0.0), 0.0)
        signal = str(_obj_get(item, "signal", "") or "").upper()
        quality_raw = _obj_get(item, "quality_pass", True)
        quality_pass = str(quality_raw).strip().lower() not in {"false", "0", "no"}
        if signal != "BUY" and score < 65:
            continue
        latest = latest_map.get(symbol, {})
        conf_label, conf_score, conf_reason = data_confidence(latest)
        ticket = _manual_order_ticket(symbol, item, latest, records)
        dollar_amount = safe_float(os.environ.get("ROBINHOOD_DOLLAR_AMOUNT_PER_TRADE", "0"), 0.0)
        suggested = f"${dollar_amount:,.2f}" if dollar_amount > 0 else f"{ticket['suggested_qty']} share(s)"
        rows.append({
            "Action": "BUY",
            "Symbol": symbol,
            "Priority": round(score, 1),
            "Suggested": suggested,
            "Price": f"${ticket['latest_price']:.2f}" if ticket["latest_price"] else "—",
            "Data": f"{conf_label} ({conf_score})",
            "Trade Gate": "PASS" if quality_pass else "BLOCK",
            "Reason": explain_buy(item),
            "_sort": score if quality_pass else score - 100,
            "_payload": {
                "symbol": symbol,
                "action": "BUY",
                "score": round(score, 1),
                "suggested": suggested,
                "latest_price": ticket["latest_price"],
                "confidence": conf_label,
                "confidence_score": conf_score,
                "confidence_reason": conf_reason,
                "reason": explain_buy(item),
            },
        })
    return sorted(rows, key=lambda row: safe_float(row.get("_sort"), 0.0), reverse=True)


def _robinhood_action_queue_rows(accounts: list[dict]) -> tuple[list[dict], list[dict]]:
    rec_rows = st.session_state.get("rh_portfolio_rows", [])
    if not rec_rows:
        return [], []
    sell_rows: list[dict] = []
    hold_rows: list[dict] = []
    for row in rec_rows:
        action = str(row.get("Action", "") or "")
        action_upper = action.upper()
        queue_row = {
            "Action": action,
            "Account": row.get("Account", "Robinhood"),
            "Symbol": row.get("Symbol", ""),
            "Priority": safe_float(row.get("Score"), 0.0),
            "Suggested": row.get("Suggested", ""),
            "Price": row.get("Price", "—"),
            "P/L": f"${safe_float(row.get('P&L $'), 0.0):+,.2f} ({safe_float(row.get('P&L %'), 0.0):+.2f}%)",
            "Reason": row.get("Why", ""),
            "_sort": (
                1000 + abs(safe_float(row.get("P&L %"), 0.0))
                if any(token in action_upper for token in ("SELL", "TAKE PROFIT", "TRIM", "RISK EXIT"))
                else safe_float(row.get("Score"), 0.0)
            ),
            "_payload": {
                "symbol": row.get("Symbol", ""),
                "action": "SELL" if any(token in action_upper for token in ("SELL", "TAKE PROFIT", "TRIM", "RISK EXIT")) else "HOLD",
                "recommendation": action,
                "suggested": row.get("Suggested", ""),
                "quantity": row.get("Sellable") if any(token in action_upper for token in ("SELL", "RISK EXIT")) else None,
                "sell_quantity": row.get("Sellable"),
                "shares": row.get("Shares"),
                "sellable": row.get("Sellable"),
                "pnl_dollars": row.get("P&L $"),
                "pnl_pct": row.get("P&L %"),
                "reason": row.get("Why", ""),
            },
        }
        if any(token in action_upper for token in ("SELL", "TAKE PROFIT", "TRIM", "RISK EXIT")):
            sell_rows.append(queue_row)
        else:
            hold_rows.append(queue_row)
    return (
        sorted(sell_rows, key=lambda row: safe_float(row.get("_sort"), 0.0), reverse=True),
        sorted(hold_rows, key=lambda row: safe_float(row.get("_sort"), 0.0), reverse=True),
    )


def _queue_action_family(row: dict) -> str:
    payload = row.get("_payload", {}) if isinstance(row.get("_payload"), dict) else {}
    action = str(payload.get("action") or row.get("Action") or "").upper()
    if "BUY" in action:
        return "BUY"
    if any(token in action for token in ("SELL", "TRIM", "TAKE PROFIT", "RISK EXIT")):
        return "SELL"
    return "WATCH"


def _action_queue_operating_summary(accounts: list[dict], records: list[dict]) -> dict[str, float]:
    total_value = sum(_portfolio_metric(a.get("portfolio", {}), "total_value", "portfolio_value") for a in accounts)
    cash = sum(_portfolio_metric(a.get("portfolio", {}), "cash") for a in accounts)
    buying_power = sum(_portfolio_buying_power(a.get("portfolio", {})) for a in accounts)
    held_symbols = {
        str(holding.get("symbol", "") or "").upper()
        for account in accounts
        for holding in account.get("positions", [])
        if holding.get("symbol")
    }
    max_open_positions = max(1, int(os.environ.get("BOT_MAX_OPEN_POSITIONS", "1") or 1))
    max_daily_trades = max(1, int(os.environ.get("BOT_MAX_DAILY_TRADES", "1") or 1))
    today = datetime.now(app_timezone()).date()
    trade_events = {"order", "stock_dry_run", "partial_profit", "partial_remainder_exit"}
    trades_today = 0
    buy_events_today = 0
    sell_events_today = 0
    for record in records:
        if record.get("event_type") not in trade_events:
            continue
        try:
            if parse_ts(str(record.get("ts") or "")).astimezone(app_timezone()).date() != today:
                continue
        except Exception:
            continue
        payload = _payload(record)
        action = str(payload.get("action") or payload.get("side") or "").lower()
        trades_today += 1
        if "buy" in action:
            buy_events_today += 1
        elif "sell" in action:
            sell_events_today += 1
    return {
        "total_value": total_value,
        "cash": cash,
        "buying_power": buying_power,
        "open_positions": float(len(held_symbols)),
        "open_slots": float(max(0, max_open_positions - len(held_symbols))),
        "max_open_positions": float(max_open_positions),
        "max_daily_trades": float(max_daily_trades),
        "trades_today": float(trades_today),
        "daily_trade_slots": float(max(0, max_daily_trades - trades_today)),
        "buy_events_today": float(buy_events_today),
        "sell_events_today": float(sell_events_today),
        "dollar_per_trade": safe_float(os.environ.get("ROBINHOOD_DOLLAR_AMOUNT_PER_TRADE", "0"), 0.0),
    }


def _webhook_status_rows() -> list[dict]:
    notify_events = {
        item.strip()
        for item in os.environ.get("BOT_NOTIFY_EVENTS", "trade,error").split(",")
        if item.strip()
    }
    routes = {
        "Buy": os.environ.get("BOT_BUY_WEBHOOK_URL") or os.environ.get("DISCORD_BUY_WEBHOOK_URL"),
        "Sell": os.environ.get("BOT_SELL_WEBHOOK_URL") or os.environ.get("DISCORD_SELL_WEBHOOK_URL"),
        "Other": os.environ.get("BOT_OTHER_WEBHOOK_URL") or os.environ.get("DISCORD_OTHER_WEBHOOK_URL"),
        "Fallback": os.environ.get("BOT_WEBHOOK_URL"),
    }
    rows = []
    for route, value in routes.items():
        rows.append({
            "Route": route,
            "Configured": "YES" if str(value or "").strip() else "NO",
            "Used For": {
                "Buy": "buy trade previews/summaries",
                "Sell": "sell and trim previews/summaries",
                "Other": "risk, scanner, errors, daily summary",
                "Fallback": "any missing route",
            }[route],
        })
    rows.append({
        "Route": "Notify Events",
        "Configured": "TRADE ON" if "trade" in notify_events else "TRADE OFF",
        "Used For": ",".join(sorted(notify_events)) or "none",
    })
    return rows


def _queue_estimated_spend(row: dict, ops: dict[str, float] | None) -> str:
    if _queue_action_family(row) != "BUY":
        return ""
    dollar_per_trade = safe_float((ops or {}).get("dollar_per_trade"), 0.0)
    if dollar_per_trade > 0:
        return f"${dollar_per_trade:,.2f}"
    suggested = str(row.get("Suggested") or "")
    price = safe_float(str(row.get("Price") or "").replace("$", "").replace(",", ""), 0.0)
    if "share" in suggested.lower() and price > 0:
        qty = safe_float(suggested.split()[0], 0.0)
        if qty > 0:
            return f"${qty * price:,.2f}"
    return suggested


def _queue_review_state(row: dict, quotes_fresh: bool, ops: dict[str, float] | None = None) -> tuple[str, str]:
    family = _queue_action_family(row)
    trade_gate = str(row.get("Trade Gate", "") or "").upper()
    if trade_gate == "BLOCK":
        return "BLOCK", "Scanner quality gate failed."
    if family == "BUY" and safe_float((ops or {}).get("daily_trade_slots"), 1.0) <= 0:
        return "BLOCK", "Daily trade cap is exhausted."
    if family == "BUY" and safe_float((ops or {}).get("open_slots"), 1.0) <= 0:
        return "BLOCK", "Open-position cap is exhausted."
    if family == "BUY":
        estimated = safe_float(_queue_estimated_spend(row, ops).replace("$", "").replace(",", ""), 0.0)
        buying_power = safe_float((ops or {}).get("buying_power"), 0.0)
        if estimated > 0 and buying_power > 0 and estimated > buying_power:
            return "BLOCK", "Suggested buy exceeds available buying power."
    if family == "BUY" and not quotes_fresh:
        return "BLOCK", "Robinhood quotes are stale; refresh before considering a new buy."
    if family == "SELL" and not quotes_fresh:
        if os.environ.get("BOT_STALE_PRICE_BLOCKS_SELL", "false").lower() == "true":
            return "BLOCK", "Robinhood quotes are stale and stale-price sell blocking is enabled."
        return "REVIEW", "Robinhood quotes are stale; confirm price before exit/trim."
    if family == "WATCH":
        return "WATCH", "No trade action queued."
    return "READY", "Ready for manual review."


def _combined_action_plan_rows(
    buy_rows: list[dict],
    sell_rows: list[dict],
    hold_rows: list[dict],
    quotes_fresh: bool,
    ops: dict[str, float] | None = None,
) -> list[dict]:
    plan: list[dict] = []
    source_rows = [("Exit / Trim", sell_rows), ("New Buy", buy_rows), ("Watch", hold_rows)]
    for category, rows in source_rows:
        for row in rows:
            family = _queue_action_family(row)
            review_state, gate_reason = _queue_review_state(row, quotes_fresh, ops)
            priority = safe_float(row.get("Priority"), safe_float(row.get("_sort"), 0.0))
            if category == "Exit / Trim":
                sort_base = 3000
            elif category == "New Buy":
                sort_base = 2000
            else:
                sort_base = 1000
            if review_state == "BLOCK":
                sort_base -= 500
            elif review_state == "WATCH":
                sort_base -= 250
            display = {
                "Category": category,
                "State": review_state,
                "Action": row.get("Action", family),
                "Symbol": row.get("Symbol", ""),
                "Priority": round(priority, 1),
                "Suggested": row.get("Suggested", ""),
                "Est. Spend": _queue_estimated_spend(row, ops),
                "Price": row.get("Price", "—"),
                "P/L": row.get("P/L", ""),
                "Data": row.get("Data", ""),
                "Reason": row.get("Reason", ""),
                "Gate Reason": gate_reason,
                "_sort": sort_base + priority,
                "_payload": {
                    **(row.get("_payload", {}) if isinstance(row.get("_payload"), dict) else {}),
                    "action": family,
                    "review_state": review_state,
                    "gate_reason": gate_reason,
                    "category": category,
                    "estimated_spend": _queue_estimated_spend(row, ops),
                },
            }
            plan.append(display)
    return sorted(plan, key=lambda row: safe_float(row.get("_sort"), 0.0), reverse=True)


def _notify_queue_batch(rows: list[dict], action: str, label: str, key_prefix: str, limit: int = 5) -> None:
    actionable = [
        row for row in rows
        if _queue_action_family(row) == action and str(row.get("State", row.get("Review State", ""))).upper() != "BLOCK"
    ][:limit]
    if not actionable:
        st.caption(f"No {label.lower()} rows ready to send.")
        return
    symbols = ", ".join(str(row.get("Symbol", "")) for row in actionable if row.get("Symbol"))
    if st.button(f"Send top {len(actionable)} {label} to Discord", key=f"{key_prefix}_send_batch"):
        payload = {
            "action": action,
            "symbols": symbols,
            "count": len(actionable),
            "items": [
                {
                    "symbol": row.get("Symbol"),
                    "state": row.get("State", row.get("Review State")),
                    "priority": row.get("Priority"),
                    "suggested": row.get("Suggested"),
                    "price": row.get("Price"),
                    "pnl": row.get("P/L"),
                    "reason": row.get("Reason"),
                    "gate_reason": row.get("Gate Reason"),
                }
                for row in actionable
            ],
        }
        message = f"{label} queue: {symbols}"
        sent = _notify_dashboard_event("trade", message, payload)
        if sent:
            st.success(f"Sent {label.lower()} queue to the routed Discord channel.")
        else:
            st.warning("Notification was not sent. Check webhook URLs and BOT_NOTIFY_EVENTS.")


def _approval_queue_path() -> str:
    return os.environ.get("ROBINHOOD_APPROVALS_PATH", "logs/robinhood_approvals.jsonl")


def _approval_candidate_label(row: dict) -> str:
    return (
        f"{row.get('State', '—')} · {row.get('Action', '—')} · {row.get('Symbol', '—')} · "
        f"priority {safe_float(row.get('Priority'), 0.0):.1f}"
    )


def _approve_action_plan_row(row: dict) -> dict[str, Any]:
    from ai_trading.broker.robinhood_approvals import approval_record, write_approval

    account_number = os.environ.get("ROBINHOOD_AGENTIC_ACCOUNT_NUMBER", "").strip()
    if not account_number:
        raise ValueError("ROBINHOOD_AGENTIC_ACCOUNT_NUMBER is required before approvals can be created.")
    payload = row.get("_payload", {}) if isinstance(row.get("_payload"), dict) else {}
    payload = {
        **payload,
        "symbol": row.get("Symbol") or payload.get("symbol"),
        "action": _queue_action_family(row),
        "suggested": row.get("Suggested") or payload.get("suggested"),
        "estimated_spend": row.get("Est. Spend") or payload.get("estimated_spend"),
        "reason": row.get("Reason") or payload.get("reason"),
        "gate_reason": row.get("Gate Reason") or payload.get("gate_reason"),
        "priority": row.get("Priority") or payload.get("priority"),
    }
    record = approval_record(
        payload,
        account_number=account_number,
        dollar_amount_per_trade=safe_float(os.environ.get("ROBINHOOD_DOLLAR_AMOUNT_PER_TRADE"), 0.0),
        approved_by="streamlit_dashboard",
    )
    return write_approval(_approval_queue_path(), record)


def _render_robinhood_approval_panel(rows: list[dict], *, compact: bool = False) -> None:
    from ai_trading.broker.robinhood_approvals import pending_approvals

    if compact:
        st.caption("Approve for Robinhood execution")
    else:
        st.subheader("Approve For Robinhood Execution")
    candidates = [
        row for row in rows
        if _queue_action_family(row) in {"BUY", "SELL"}
        and str(row.get("State", "")).upper() in {"READY", "REVIEW"}
    ]
    if not candidates:
        st.caption("No ready buy/sell rows available for approval.")
    else:
        labels = [_approval_candidate_label(row) for row in candidates]
        selected_label = st.selectbox("Candidate", labels, key="rh_approval_candidate")
        selected = candidates[labels.index(selected_label)]
        payload = selected.get("_payload", {}) if isinstance(selected.get("_payload"), dict) else {}
        preview_payload = {
            "symbol": selected.get("Symbol"),
            "action": _queue_action_family(selected),
            "state": selected.get("State"),
            "suggested": selected.get("Suggested"),
            "estimated_spend": selected.get("Est. Spend"),
            "price": selected.get("Price"),
            "reason": selected.get("Reason"),
            "gate_reason": selected.get("Gate Reason"),
            "payload": payload,
        }
        with st.expander("Approval Preview", expanded=False):
            st.json(preview_payload)
        confirm = st.checkbox(
            "I approve this candidate for the Robinhood executor queue",
            key="rh_approval_confirm",
        )
        if st.button("Approve Selected Candidate", type="primary", disabled=not confirm, key="rh_approve_selected"):
            try:
                record = _approve_action_plan_row(selected)
                _record_dashboard_event(
                    journal_path,
                    "robinhood_approval",
                    {
                        "approval_id": record.get("approval_id"),
                        "order": record.get("order"),
                        "execution_status": record.get("execution_status"),
                    },
                )
                _notify_dashboard_event(
                    "trade",
                    f"Approved {record['order']['side'].upper()} {record['order']['symbol']} for Robinhood execution queue",
                    {
                        "action": record["order"]["side"].upper(),
                        "symbol": record["order"]["symbol"],
                        "approval_id": record.get("approval_id"),
                        "status": record.get("status"),
                        "order": record.get("order"),
                    },
                )
                st.success(f"Approved and queued: {record['approval_id']}")
                st.code(json.dumps(record.get("executor_order", {}), indent=2), language="json")
            except Exception as exc:
                st.error(f"Approval failed: {exc}")

    approvals = pending_approvals(_approval_queue_path())
    if approvals:
        rows_out = []
        for item in reversed(approvals[-10:]):
            order = item.get("order", {})
            rows_out.append({
                "Created": format_price_time(str(item.get("created_at") or "")),
                "Approval ID": item.get("approval_id", ""),
                "Status": item.get("status", ""),
                "Side": str(order.get("side", "")).upper(),
                "Symbol": order.get("symbol", ""),
                "Amount/Qty": order.get("dollar_amount") or order.get("quantity") or "—",
                "Execution": item.get("execution_status", ""),
            })
        if compact:
            with st.expander(f"Pending approvals ({len(approvals)})", expanded=False):
                st.dataframe(pd.DataFrame(rows_out), use_container_width=True, hide_index=True)
        else:
            st.dataframe(pd.DataFrame(rows_out), use_container_width=True, hide_index=True)
    else:
        st.caption("No pending Robinhood approvals yet.")


def _display_queue(rows: list[dict], empty: str, key_prefix: str) -> None:
    if not rows:
        st.info(empty)
        return
    table = pd.DataFrame([{k: v for k, v in row.items() if not k.startswith("_")} for row in rows])
    st.dataframe(table, use_container_width=True, hide_index=True, height=min(80 + len(rows) * 35, 650))
    family = _queue_action_family(rows[0])
    if family in {"BUY", "SELL"}:
        _notify_queue_batch(rows, family, f"{family} candidates", f"{key_prefix}_{family.lower()}")
    top = rows[0]
    payload = top.get("_payload", {})
    if st.button(f"Send top {payload.get('action', 'item')} to Discord", key=f"{key_prefix}_send_top"):
        event_type = "trade" if payload.get("action") in {"BUY", "SELL"} else "scanner_summary"
        message = f"{payload.get('action', 'ACTION')} {payload.get('symbol', '')}: {payload.get('reason', '')}"
        sent = _notify_dashboard_event(event_type, message, payload)
        if sent:
            st.success("Notification sent through configured webhook routing.")
        else:
            st.warning("Notification was not sent. Check webhook URLs and BOT_NOTIFY_EVENTS.")


def _render_action_queue(records: list[dict]) -> None:
    portfolios_path = str(globals().get("robinhood_portfolios_path", "") or "logs/robinhood_portfolios.json")
    quotes_path = str(globals().get("robinhood_quote_path", "") or "logs/robinhood_quotes.json")
    accounts = load_robinhood_portfolio_snapshot(portfolios_path)
    _render_page_header(
        "Action Queue",
        "One operational queue for Robinhood buys, sells, trims, holds, data freshness, and Discord routing.",
        ["Buy channel", "Sell channel", "Other channel", "Review before real orders"],
    )
    _render_snapshot_status_panel(portfolios_path, quotes_path)
    ttl = max(30, int(os.environ.get("ROBINHOOD_SNAPSHOT_TTL_SEC", "300") or 300))
    quote_status = _snapshot_status(quotes_path, ttl)
    quotes_fresh = bool(quote_status.get("fresh"))
    held_symbols = {
        str(holding.get("symbol", "") or "").upper()
        for account in accounts
        for holding in account.get("positions", [])
    }

    ops = _action_queue_operating_summary(accounts, records)
    buy_rows = _scanner_buy_queue_rows(records, held_symbols)
    sell_rows, hold_rows = _robinhood_action_queue_rows(accounts)
    action_plan = _combined_action_plan_rows(buy_rows, sell_rows, hold_rows, quotes_fresh, ops)
    ready_count = sum(1 for row in action_plan if str(row.get("State", "")).upper() == "READY")
    blocked_count = sum(1 for row in action_plan if str(row.get("State", "")).upper() == "BLOCK")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Ready Actions", ready_count)
    q2.metric("Blocked", blocked_count)
    q3.metric("Buy Candidates", len(buy_rows))
    q4.metric("Sell / Trim", len(sell_rows))

    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Buying Power", fmt_money(ops["buying_power"]))
    h2.metric("Cash", fmt_money(ops["cash"]))
    h3.metric("Open Slots", f"{int(ops['open_slots'])}/{int(ops['max_open_positions'])}")
    h4.metric("Trades Left Today", f"{int(ops['daily_trade_slots'])}/{int(ops['max_daily_trades'])}")
    h5.metric("Trade Size", fmt_money(ops["dollar_per_trade"]) if ops["dollar_per_trade"] > 0 else "share sizing")

    top_cmd_1, top_cmd_2, top_cmd_3 = st.columns([1, 1, 2])
    with top_cmd_1:
        if accounts and st.button("Score / Rescore Holdings", type="primary", key="queue_score_holdings_top", use_container_width=True):
            with st.spinner("Scoring Robinhood holdings..."):
                try:
                    rows, full = _recommend_robinhood_holdings(accounts, "short")
                    st.session_state["rh_portfolio_rows"] = rows
                    st.session_state["rh_portfolio_full"] = full
                    st.session_state["rh_portfolio_ts"] = format_local_now("%I:%M:%S %p %Z")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Scoring failed: {exc}")
        elif not accounts:
            st.caption("Load Robinhood snapshot to score holdings.")
    with top_cmd_2:
        if st.button("Refresh Queue Cache", key="queue_refresh_cache_top", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with top_cmd_3:
        st.caption("Command bar: score holdings, approve candidates, and send Discord alerts without scrolling.")

    with st.expander("Operator Health", expanded=False):
        status_l, status_r = st.columns([1, 1])
        with status_l:
            st.dataframe(pd.DataFrame(_webhook_status_rows()), use_container_width=True, hide_index=True)
        with status_r:
            st.dataframe(pd.DataFrame([{
                "Check": "Quote freshness",
                "Status": "PASS" if quotes_fresh else "STALE",
                "Detail": quote_status.get("age_label", "—"),
            }, {
                "Check": "Portfolio snapshot",
                "Status": _snapshot_status(portfolios_path, ttl).get("label", "missing").upper(),
                "Detail": _snapshot_status(portfolios_path, ttl).get("age_label", "—"),
            }, {
                "Check": "Daily trade cap",
                "Status": "PASS" if ops["daily_trade_slots"] > 0 else "BLOCK",
                "Detail": f"{int(ops['trades_today'])} used",
            }, {
                "Check": "Open-position cap",
                "Status": "PASS" if ops["open_slots"] > 0 else "BLOCK",
                "Detail": f"{int(ops['open_positions'])} open",
            }]), use_container_width=True, hide_index=True)
        st.caption("These checks only organize review readiness. They do not guarantee profitable trades.")

    settings_rows = [{
        "Setting": "Profit trigger",
        "Value": f"{os.environ.get('BOT_PARTIAL_PROFIT_TRIGGER_PCT', '0')}%",
    }, {
        "Setting": "Partial sell",
        "Value": f"{os.environ.get('BOT_PARTIAL_PROFIT_SELL_PCT', '0')}%",
    }, {
        "Setting": "Remainder trail",
        "Value": f"{os.environ.get('BOT_PARTIAL_PROFIT_TRAILING_STOP_PCT', '0')}%",
    }, {
        "Setting": "Daily trade cap",
        "Value": os.environ.get("BOT_MAX_DAILY_TRADES", "—"),
    }, {
        "Setting": "Open position cap",
        "Value": os.environ.get("BOT_MAX_OPEN_POSITIONS", "—"),
    }]
    with st.expander("Profit-Harvest Mode", expanded=False):
        st.dataframe(pd.DataFrame(settings_rows), use_container_width=True, hide_index=True)
        st.caption("This mode tries to harvest winners faster; it cannot guarantee profits.")

    if action_plan:
        st.subheader("Today's Action Plan")
        f1, f2, f3 = st.columns([1, 1, 1])
        with f1:
            state_filter = st.multiselect(
                "States",
                ["READY", "REVIEW", "BLOCK", "WATCH"],
                default=["READY", "REVIEW"],
                key="action_plan_state_filter",
            )
        with f2:
            category_filter = st.multiselect(
                "Categories",
                ["Exit / Trim", "New Buy", "Watch"],
                default=["Exit / Trim", "New Buy"],
                key="action_plan_category_filter",
            )
        with f3:
            min_priority = st.slider("Minimum Priority", 0, 100, 0, key="action_plan_min_priority")
        filtered_plan = [
            row for row in action_plan
            if row.get("State") in state_filter
            and row.get("Category") in category_filter
            and safe_float(row.get("Priority"), 0.0) >= min_priority
        ]
        st.markdown("**Queue Commands**")
        n1, n2, n3 = st.columns([1.15, 1, 1])
        with n1:
            _render_robinhood_approval_panel(filtered_plan, compact=True)
        with n2:
            _notify_queue_batch(filtered_plan, "BUY", "BUY candidates", "action_plan_buy")
        with n3:
            _notify_queue_batch(filtered_plan, "SELL", "SELL candidates", "action_plan_sell")
        st.caption(
            f"Showing {len(filtered_plan)} of {len(action_plan)} rows. "
            "Blocked rows are excluded from approval and batch alerts."
        )
        plan_table = pd.DataFrame([{k: v for k, v in row.items() if not k.startswith("_")} for row in filtered_plan])
        st.dataframe(plan_table, use_container_width=True, hide_index=True, height=min(120 + len(action_plan) * 35, 700))
    else:
        st.info("Run the scanners and score Robinhood holdings to populate today's action plan.")

    buy_tab, sell_tab, hold_tab = st.tabs(["Buy Queue", "Sell / Trim Queue", "Hold / Watch"])
    with buy_tab:
        _display_queue(
            buy_rows,
            "Run the Buy Scanner to populate buy candidates. Existing Robinhood holdings are excluded.",
            "buy_queue",
        )
    with sell_tab:
        _display_queue(
            sell_rows,
            "Score Robinhood holdings or run the Sell Scanner to populate sell/trim actions.",
            "sell_queue",
        )
    with hold_tab:
        _display_queue(hold_rows, "No hold/watch rows yet. Score Robinhood holdings first.", "hold_queue")


def _scan_holdings_for_advice(holdings: list[dict]) -> dict[str, Any]:
    symbols = tuple(dict.fromkeys(str(h.get("symbol", "")).upper() for h in holdings if h.get("symbol")))
    if not symbols:
        return {}
    market_open = is_market_open()
    if market_open:
        scan_out = scan_live(list(symbols), top_n=len(symbols))
        if scan_out:
            return {str(_obj_get(r, "symbol", "")).upper(): r for r in scan_out}
    scan_out = scan(
        list(symbols),
        fast_ma=int(scanner_fast_ma),
        slow_ma=int(scanner_slow_ma),
        top_n=len(symbols),
    )
    return {str(_obj_get(r, "symbol", "")).upper(): r for r in scan_out}


def _recommend_robinhood_holdings(accounts: list[dict], horizon: str) -> tuple[list[dict], list[dict]]:
    holdings: list[dict] = []
    for account in accounts:
        for holding in account.get("positions", []):
            item = dict(holding)
            item["_account_id"] = account.get("id", "")
            item["_account_label"] = account.get("label", "Robinhood")
            item["_account_masked"] = account.get("account", "—")
            holdings.append(item)
    scan_by_symbol = _scan_holdings_for_advice(holdings)
    symbols = tuple(dict.fromkeys(str(h.get("symbol", "")).upper() for h in holdings if h.get("symbol")))
    rh_latest_map = _robinhood_latest_price_map(symbols)
    fallback_latest_map = load_latest_price_map(tuple(s for s in symbols if s not in rh_latest_map))
    latest_map = {**fallback_latest_map, **rh_latest_map}
    rows: list[dict] = []
    full: list[dict] = []
    for holding in holdings:
        symbol = str(holding.get("symbol", "")).upper()
        latest = latest_map.get(symbol, {})
        latest_px = safe_float(latest.get("price"), 0.0)
        shares = safe_float(holding.get("shares"), 0.0)
        avg_cost = safe_float(holding.get("avg_cost"), 0.0)
        holding_type = str(holding.get("type") or "").lower()
        result = scan_by_symbol.get(symbol)
        if result is None:
            fallback_market_value = safe_float(holding.get("market_value"), 0.0)
            if latest_px > 0 and shares > 0:
                fallback_market_value = shares * latest_px
            fallback_cost_basis = shares * avg_cost
            fallback_pnl = fallback_market_value - fallback_cost_basis if fallback_market_value and fallback_cost_basis else 0.0
            fallback_pnl_pct = (fallback_pnl / fallback_cost_basis * 100.0) if fallback_cost_basis > 0 else 0.0
            if holding_type == "crypto":
                why = "Crypto holding uses Robinhood quote data here, but the stock scanner does not generate a technical score for crypto yet."
            elif latest_px > 0:
                why = "Live quote is available, but no technical bars were returned for scoring."
            else:
                why = "No market data returned for this holding."
            rows.append({
                "Account": holding.get("_account_label", "Robinhood"),
                "Symbol": symbol,
                "Action": "—",
                "Suggested": "No action",
                "Why": why,
                "Shares": shares,
                "Sellable": safe_float(holding.get("sellable"), 0.0),
                "Avg Cost": avg_cost,
                "Price": round(latest_px, 2) if latest_px > 0 else None,
                "Price Time": format_price_time(str(latest.get("timestamp") or "")),
                "Source": latest_source_label(latest, "robinhood snapshot") if latest else "unavailable",
                "Robinhood Check": "QUOTE ONLY" if latest_px > 0 else "NO DATA",
                "Market Value": round(fallback_market_value, 2),
                "P&L $": round(fallback_pnl, 2),
                "P&L %": round(fallback_pnl_pct, 2),
                "Score": None,
            })
            continue
        latest_px = latest.get("price")
        if latest_px:
            result.close = float(latest_px)
            holding["market_value"] = safe_float(holding.get("shares"), 0.0) * float(latest_px)
        rh_check, rh_check_note = robinhood_quote_confirmation(symbol, latest)
        rec = advise_position(
            symbol,
            safe_float(holding.get("shares"), 0.0),
            safe_float(holding.get("avg_cost"), 0.0),
            result,
            market_value=safe_float(holding.get("market_value"), 0.0) or None,
            horizon=horizon,
        )
        rec["account"] = holding.get("_account_label", "Robinhood")
        rec["account_masked"] = holding.get("_account_masked", "—")
        rec["sellable"] = safe_float(holding.get("sellable"), 0.0)
        rec["latest_time"] = format_price_time(str(latest.get("timestamp") or ""))
        rec["latest_source"] = latest_source_label(latest, "scan")
        rec["rh_check"] = rh_check
        rec["rh_check_note"] = rh_check_note
        full.append(rec)
        rows.append({
            "Account": rec["account"],
            "Symbol": symbol,
            "Action": f"{rec['emoji']} {rec['action']}",
            "Suggested": rec["size_label"],
            "Why": rec["rationale"][0] if rec["rationale"] else "",
            "Shares": round(safe_float(holding.get("shares"), 0.0), 6),
            "Sellable": round(rec["sellable"], 6),
            "Avg Cost": round(safe_float(holding.get("avg_cost"), 0.0), 2),
            "Price": round(rec["price"], 2),
            "Price Time": rec["latest_time"],
            "Source": rec["latest_source"],
            "Robinhood Check": rh_check,
            "Market Value": round(rec["market_value"], 2),
            "P&L $": round(rec["total_return"], 2),
            "P&L %": round(rec["pnl_pct"], 2),
            "Score": round(rec["score"], 0),
        })
    return rows, full


def _render_robinhood_portfolios_panel(path: str) -> None:
    st.markdown(
        """
        <div class="ops-section">
          <div class="ops-kicker">Primary Broker</div>
          <div class="ops-title">Robinhood Portfolio Command Center</div>
          <div class="ops-note">Investing and Agentic accounts, Robinhood quote-backed P&L, and position-level action reads.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    accounts = load_robinhood_portfolio_snapshot(path)
    if not accounts:
        st.info(
            f"No Robinhood portfolio snapshot found at `{path}`. "
            "Ask Codex to refresh the Robinhood portfolio snapshot, then reload the dashboard."
        )
        with st.expander("Expected snapshot shape", expanded=False):
            st.code(
                json.dumps({
                    "updated_at": "2026-06-05T20:00:00Z",
                    "accounts": [{
                        "label": "Investing",
                        "account_masked": "****9371",
                        "agentic": False,
                        "portfolio": {"total_value": "1000.00", "cash": "25.00"},
                        "positions": [{
                            "symbol": "AAPL",
                            "quantity": "1.5",
                            "average_buy_price": "150.00",
                            "shares_available_for_sells": "1.5",
                        }],
                    }],
                }, indent=2),
                language="json",
            )
        return

    total_value = sum(_portfolio_metric(a.get("portfolio", {}), "total_value", "portfolio_value") for a in accounts)
    total_equity = sum(_portfolio_metric(a.get("portfolio", {}), "equity_value") for a in accounts)
    total_cash = sum(_portfolio_metric(a.get("portfolio", {}), "cash") for a in accounts)
    total_bp = sum(_portfolio_buying_power(a.get("portfolio", {})) for a in accounts)
    total_options = sum(_portfolio_metric(a.get("portfolio", {}), "options_value") for a in accounts)
    total_crypto = sum(_portfolio_metric(a.get("portfolio", {}), "crypto_value") for a in accounts)
    equity_pnl = _robinhood_equity_pnl(accounts)
    total_net_invested = sum(_portfolio_net_invested(a) for a in accounts)
    holdings_count = sum(len(a.get("positions", [])) for a in accounts)
    top_metrics_1 = st.columns(5)
    top_metrics_1[0].metric("Accounts", len(accounts))
    top_metrics_1[1].metric("Account Value", fmt_money(total_value))
    top_metrics_1[2].metric("Net Invested", fmt_money(total_net_invested))
    top_metrics_1[3].metric("Total P/L", f"${equity_pnl['pnl']:+,.2f}", f"{equity_pnl['pnl_pct']:+.2f}%")
    top_metrics_1[4].metric("Buying Power", fmt_money(total_bp))

    top_metrics_2 = st.columns(4)
    top_metrics_2[0].metric("Equities", fmt_money(total_equity))
    top_metrics_2[1].metric("Cash", fmt_money(total_cash))
    top_metrics_2[2].metric("Options", fmt_money(total_options))
    top_metrics_2[3].metric("Crypto", fmt_money(total_crypto))
    all_symbols = tuple(dict.fromkeys(
        str(h.get("symbol", "")).upper()
        for account in accounts
        for h in account.get("positions", [])
        if h.get("symbol")
    ))
    rh_latest_all = _robinhood_latest_price_map(all_symbols)
    if rh_latest_all:
        fresh_count = sum(
            1 for latest in rh_latest_all.values()
            if safe_float(latest.get("age_seconds"), 999999.0) <= 300
        )
        st.caption(
            f"Using fresh Robinhood quotes for {len(rh_latest_all):,}/{len(all_symbols):,} holdings "
            f"({fresh_count:,} fresh within 5 minutes)."
        )
    elif all_symbols:
        st.warning(
            "Fresh Robinhood quotes are not available, so recommendation scoring is using current Alpaca/yfinance prices "
            "instead of stale snapshot data. Refresh `logs/robinhood_quotes.json` to restore Robinhood as the preferred price source."
        )

    rec_rows = st.session_state.get("rh_portfolio_rows", [])
    full_recs = st.session_state.get("rh_portfolio_full", [])
    account_summary_rows = []
    for account in accounts:
        portfolio = account.get("portfolio", {})
        positions = account.get("positions", [])
        account_pnl = _robinhood_equity_pnl([account])
        net_invested = _portfolio_net_invested(account)
        account_summary_rows.append({
            "Account": account.get("label", "Robinhood"),
            "ID": account.get("account", "—"),
            "Role": "Agentic" if account.get("agentic") else "Investing",
            "Value": fmt_money(_portfolio_metric(portfolio, "total_value", "portfolio_value")),
            "Net Invested": fmt_money(net_invested),
            "Equities": fmt_money(_portfolio_metric(portfolio, "equity_value")),
            "Options": fmt_money(_portfolio_metric(portfolio, "options_value")),
            "Crypto": fmt_money(_portfolio_metric(portfolio, "crypto_value")),
            "Total P/L": f"${account_pnl['pnl']:+,.2f}",
            "P/L %": f"{account_pnl['pnl_pct']:+.2f}%",
            "Cash": fmt_money(_portfolio_metric(portfolio, "cash")),
            "Buying Power": fmt_money(_portfolio_buying_power(portfolio)),
            "Positions": len(positions),
        })

    rec_by_account = {}
    for row in rec_rows:
        rec_by_account.setdefault(str(row.get("Account", "")), []).append(row)

    overview_tab, actions_tab, holdings_tab, features_tab, detail_tab = st.tabs(
        ["Overview", "Actions", "Holdings", "Capabilities", "Position Detail"]
    )

    with overview_tab:
        summary_cards = st.columns(max(1, min(3, len(account_summary_rows))))
        for col, row in zip(summary_cards, account_summary_rows):
            with col:
                st.markdown(
                    f"""
                    <div style="border:1px solid rgba(148,163,184,0.22);border-radius:8px;padding:0.9rem 1rem;
                                background:rgba(15,23,42,0.26);min-height:170px">
                      <div style="color:#94a3b8;font-size:0.78rem;text-transform:uppercase">{row['Role']}</div>
                      <div style="color:#f8fafc;font-size:1.15rem;font-weight:700;margin-top:0.15rem">{row['Account']}</div>
                      <div style="color:#94a3b8;font-size:0.82rem">{row['ID']}</div>
                      <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.55rem;margin-top:0.8rem">
                        <div><div style="color:#94a3b8;font-size:0.72rem">Value</div><div style="color:#e2e8f0">{row['Value']}</div></div>
                        <div><div style="color:#94a3b8;font-size:0.72rem">Net Invested</div><div style="color:#e2e8f0">{row['Net Invested']}</div></div>
                        <div><div style="color:#94a3b8;font-size:0.72rem">P/L</div><div style="color:#e2e8f0">{row['Total P/L']} ({row['P/L %']})</div></div>
                        <div><div style="color:#94a3b8;font-size:0.72rem">Cash</div><div style="color:#e2e8f0">{row['Cash']}</div></div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.dataframe(pd.DataFrame(account_summary_rows), use_container_width=True, hide_index=True)
        st.caption(f"Snapshot: `{path}`")
        controls_l, controls_r = st.columns([2, 1])
        with controls_l:
            horizon_label = st.radio(
                "Recommendation horizon",
                ["Short-term", "Long-term"],
                horizontal=True,
                key="rh_portfolio_horizon",
                help="Short-term is more aggressive about trimming stretched winners. Long-term lets healthy winners run.",
            )
        with controls_r:
            horizon = "long" if horizon_label == "Long-term" else "short"
            if st.button("Score Portfolios", type="primary", key="rh_score_portfolios", use_container_width=True):
                with st.spinner(f"Scoring {holdings_count} Robinhood positions..."):
                    try:
                        rows, full = _recommend_robinhood_holdings(accounts, horizon)
                    except Exception as exc:
                        rows, full = [], []
                        st.error(f"Robinhood portfolio scoring failed: {exc}")
                    st.session_state["rh_portfolio_rows"] = rows
                    st.session_state["rh_portfolio_full"] = full
                    st.session_state["rh_portfolio_ts"] = format_local_now("%I:%M:%S %p %Z")

        if rec_rows:
            st.caption(f"Last scoring run: {st.session_state.get('rh_portfolio_ts', '—')}")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Buy More", sum(1 for r in rec_rows if "BUY MORE" in str(r.get("Action"))))
            r2.metric("Trim", sum(1 for r in rec_rows if "TRIM" in str(r.get("Action"))))
            r3.metric("Sell / Take Profit", sum(1 for r in rec_rows if "SELL" in str(r.get("Action")) or "TAKE PROFIT" in str(r.get("Action"))))
            r4.metric("Hold", sum(1 for r in rec_rows if "HOLD" in str(r.get("Action"))))

    with actions_tab:
        if not rec_rows:
            st.info("Score the Robinhood portfolios from the Overview tab to generate action recommendations.")
        else:
            def _priority(row: dict) -> tuple[int, float]:
                action = str(row.get("Action", "")).upper()
                if "TAKE PROFIT" in action or "SELL" in action:
                    level = 0
                elif "TRIM" in action:
                    level = 1
                elif "BUY MORE" in action:
                    level = 2
                elif "HOLD" in action:
                    level = 3
                else:
                    level = 4
                return (level, -abs(safe_float(row.get("P&L %"), 0.0)))

            action_rows = sorted(rec_rows, key=_priority)
            st.dataframe(pd.DataFrame(action_rows), use_container_width=True, hide_index=True)

    with holdings_tab:
        account_tabs = st.tabs([f"{a.get('label', 'Robinhood')} {a.get('account', '')}" for a in accounts])
        for tab, account in zip(account_tabs, accounts):
            with tab:
                portfolio = account.get("portfolio", {})
                p_top = st.columns(4)
                p_top[0].metric("Value", fmt_money(_portfolio_metric(portfolio, "total_value", "portfolio_value")))
                p_top[1].metric("Net Invested", fmt_money(_portfolio_net_invested(account)))
                p_top[2].metric("Equity Value", fmt_money(_portfolio_metric(portfolio, "equity_value")))
                p_top[3].metric("Buying Power", fmt_money(_portfolio_buying_power(portfolio)))

                p_bottom = st.columns(3)
                p_bottom[0].metric("Cash", fmt_money(_portfolio_metric(portfolio, "cash")))
                p_bottom[1].metric("Options", fmt_money(_portfolio_metric(portfolio, "options_value")))
                p_bottom[2].metric("Crypto", fmt_money(_portfolio_metric(portfolio, "crypto_value")))

                account_rows = rec_by_account.get(str(account.get("label", "")), [])
                if account_rows:
                    st.dataframe(pd.DataFrame(account_rows), use_container_width=True, hide_index=True)
                    continue

                positions = account.get("positions", [])
                if not positions:
                    st.caption("No open equity positions in this snapshot.")
                    continue

                account_latest = _robinhood_latest_price_map(tuple(
                    str(h.get("symbol", "")).upper() for h in positions if h.get("symbol")
                ))
                preview_rows = []
                for h in positions:
                    symbol = str(h.get("symbol", "")).upper()
                    latest = account_latest.get(symbol, {})
                    price, market_value, pnl, pnl_pct = _robinhood_price_for_holding(h, latest)
                    preview_rows.append({
                        "Symbol": symbol,
                        "Shares": round(safe_float(h.get("shares"), 0.0), 6),
                        "Sellable": round(safe_float(h.get("sellable"), 0.0), 6),
                        "Avg Cost": round(safe_float(h.get("avg_cost"), 0.0), 2),
                        "Price": round(price, 2),
                        "Market Value": round(market_value, 2),
                        "P&L $": round(pnl, 2),
                        "P&L %": round(pnl_pct, 2),
                        "Price Time": format_price_time(str(latest.get("timestamp") or "")),
                        "Source": latest_source_label(latest, "robinhood snapshot"),
                        "Type": h.get("type", "long"),
                    })
                st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)
                st.caption("Score portfolios to add buy/hold/trim/sell actions and reasons.")

    with features_tab:
        feature_rows = []
        ipo_notes: list[str] = []
        crypto_rows: list[dict] = []
        for account in accounts:
            features = account.get("features", {}) if isinstance(account.get("features"), dict) else {}
            feature_rows.append({
                "Account": account.get("label", "Robinhood"),
                "Role": "Agentic" if account.get("agentic") else "Investing",
                "Account Type": str(features.get("account_type") or "—"),
                "Option Level": str(features.get("option_level") or "—"),
                "Fractionals": "Yes" if safe_bool(features.get("eligible_for_fractionals")) else "No",
                "DRIP Enabled": "Yes" if safe_bool(features.get("drip_enabled")) else "No",
                "Cash Mgmt": "Yes" if safe_bool(features.get("cash_management_enabled")) else "No",
                "Agentic Allowed": "Yes" if safe_bool(features.get("agentic_allowed")) else "No",
                "Crypto BP": fmt_money(safe_float(features.get("crypto_buying_power"), 0.0)),
                "Options BP": fmt_money(safe_float(features.get("options_buying_power"), 0.0)),
                "IPO Restricted": "Yes" if safe_bool(features.get("ipo_access_restricted")) else "No",
                "IPO Restriction Reason": str(features.get("ipo_access_restricted_reason") or "—"),
                "State": str(features.get("state") or "—"),
            })
            if safe_bool(features.get("ipo_access_restricted")):
                ipo_notes.append(
                    f"{account.get('label', 'Robinhood')}: {str(features.get('ipo_access_restricted_reason') or 'IPO access restricted')}"
                )
            for pos in account.get("positions", []):
                if str(pos.get("type") or "").lower() != "crypto":
                    continue
                crypto_rows.append({
                    "Account": account.get("label", "Robinhood"),
                    "Symbol": str(pos.get("symbol") or ""),
                    "Quantity": round(safe_float(pos.get("shares"), 0.0), 8),
                    "Avg Cost": round(safe_float(pos.get("avg_cost"), 0.0), 4),
                    "Market Value": round(safe_float(pos.get("market_value"), 0.0), 2),
                })
        st.dataframe(pd.DataFrame(feature_rows), use_container_width=True, hide_index=True)
        st.caption("These capability flags come from the Robinhood account snapshot and help explain what each account can actually do.")
        if ipo_notes:
            for note in ipo_notes:
                st.warning(note)
        else:
            st.success("No IPO access restrictions are currently flagged in the Robinhood account snapshot.")
        if crypto_rows:
            st.markdown("**Crypto Holdings**")
            st.dataframe(pd.DataFrame(crypto_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No crypto positions are present in the current Robinhood snapshot.")

    with detail_tab:
        if not full_recs:
            st.info("Run scoring first, then pick any holding here to inspect the rationale.")
        else:
            options = [f"{r['account']} · {r['symbol']}" for r in full_recs]
            pick = st.selectbox("Holding", options=options, key="rh_portfolio_pick")
            rec = full_recs[options.index(pick)]
            st.markdown(f"**{rec['emoji']} {rec['action']} {rec['symbol']}** · {rec['size_label']}")
            for line in rec["rationale"]:
                st.markdown(f"- {line}")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Score", f"{rec['score']:.0f}/100", rec["signal"])
            d2.metric("RSI", f"{rec['rsi']:.0f}")
            d3.metric("5-day Momentum", f"{rec['momentum']:+.2f}%")
            d4.metric("P&L", f"${rec['total_return']:+,.2f}", f"{rec['pnl_pct']:+.2f}%")
            _holding_spec = ResearchPromptSpec(
                subject=str(rec.get("symbol", "")).upper(),
                goals=st.session_state.get("research_goals", "long-term capital appreciation"),
                risk_tolerance=st.session_state.get("research_risk", "moderate"),
                time_horizon=st.session_state.get("research_horizon", "5+ years"),
                as_of_date=st.session_state.get(
                    "research_as_of_date",
                    datetime.now(app_timezone()).strftime("%B %d, %Y"),
                ),
            )
            _holding_mode = st.session_state.get("research_mode", "memo")
            with st.expander("Research packet", expanded=False):
                st.text_area(
                    "Holding prompt",
                    value=build_prompt_for_mode(_holding_mode, _holding_spec),
                    height=320,
                    key=f"holding_research_prompt_{rec['symbol']}",
                )
                if st.button("Save Holding Research Packet", key=f"holding_research_save_{rec['symbol']}"):
                    _queue_research_packet_event(
                        journal_path=journal_path,
                        spec=_holding_spec,
                        mode=_holding_mode,
                        source="robinhood_holding",
                        context={
                            "account": str(rec.get("account", "")),
                            "symbol": str(rec.get("symbol", "")).upper(),
                            "action": str(rec.get("action", "")),
                            "score": safe_float(rec.get("score", 0.0), 0.0),
                            "pnl_pct": safe_float(rec.get("pnl_pct", 0.0), 0.0),
                        },
                    )
                    st.success(f"Saved research packet for {rec['symbol']}.")


def _manual_order_ticket(symbol: str, result: Any, latest: dict, records: list[dict]) -> dict:
    try:
        from ai_trading.config import Settings

        settings = Settings.from_env()
    except Exception:
        settings = None
    price = safe_float(latest.get("price"), safe_float(_obj_get(result, "close", 0.0), 0.0))
    stop = safe_float(_obj_get(result, "stop", 0.0), 0.0)
    target = safe_float(_obj_get(result, "target", 0.0), 0.0)
    if stop <= 0 and price > 0:
        stop = price * 0.92
    if target <= 0 and price > 0:
        target = price * 1.08
    latest_account = get_latest(records, "account_state") or {}
    equity = safe_float(latest_account.get("equity"), 0.0)
    risk_pct = float(getattr(settings, "risk_per_trade_pct", 0.5) if settings else 0.5)
    max_shares = int(getattr(settings, "max_shares", 100) if settings else 100)
    risk_budget = equity * risk_pct / 100.0 if equity > 0 else 0.0
    risk_per_share = max(0.01, price - stop) if price > stop else max(0.01, price * 0.02)
    qty_by_risk = int(risk_budget // risk_per_share) if risk_budget > 0 else max_shares
    qty = max(1, min(max_shares, qty_by_risk if qty_by_risk > 0 else max_shares))
    order_type = str(getattr(settings, "order_type", "market") if settings else "market").lower()
    limit_offset = float(getattr(settings, "limit_price_offset_pct", 0.1) if settings else 0.1)
    limit_price = price * (1 + limit_offset / 100.0) if price > 0 else 0.0
    confidence, confidence_score, confidence_reason = data_confidence(latest)
    return {
        "symbol": symbol,
        "action": "BUY" if safe_float(_obj_get(result, "score", 0.0), 0.0) >= 65 else "WATCH",
        "suggested_qty": qty,
        "latest_price": price,
        "order_type": order_type,
        "limit_price": limit_price,
        "stop": stop,
        "target": target,
        "risk_budget": risk_budget,
        "risk_per_share": risk_per_share,
        "confidence": confidence,
        "confidence_score": confidence_score,
        "confidence_reason": confidence_reason,
    }


def latest_price_badge(symbol: str, fallback: float) -> str:
    price, source, timestamp = latest_price_for(symbol, fallback)
    return f"${price:,.2f} · {format_price_time(timestamp)} · {source}"


def _payload(record: dict) -> dict:
    payload = record.get("payload", {})
    return payload if isinstance(payload, dict) else {}


def _record_dashboard_event(path: str, event_type: str, payload: dict) -> None:
    try:
        from ai_trading.storage.journal import Journal

        Journal(Path(path)).write(event_type, payload)
    except Exception:
        pass


def _notify_dashboard_event(event_type: str, message: str, payload: dict) -> bool:
    try:
        from ai_trading.config import Settings
        from ai_trading.notifications.alerter import Notifier

        settings = Settings.from_env()
        return Notifier(settings.webhook_url, settings.notify_events).notify(event_type, message, payload)
    except Exception:
        return False


def _queue_research_packet_event(
    *,
    journal_path: str,
    spec: ResearchPromptSpec,
    mode: str,
    source: str,
    context: dict | None = None,
) -> dict:
    packet = build_research_packet(
        spec=spec,
        mode=mode,
        source=source,
        context=context or {},
    )
    _record_dashboard_event(journal_path, "research_packet", packet)
    return packet


def _paper_performance(records: list[dict]) -> tuple[list[dict], dict]:
    lots: dict[str, list[dict]] = {}
    trades: list[dict] = []
    for record in records:
        if record.get("event_type") not in {"order", "stock_dry_run", "partial_profit", "partial_remainder_exit"}:
            continue
        p = _payload(record)
        action = str(p.get("action") or "").upper()
        symbol = str(p.get("symbol") or "").upper()
        if action not in {"BUY", "SELL"} or not symbol:
            continue
        qty = int(safe_float(p.get("qty", p.get("sold_qty", 0)), 0.0))
        price = safe_float(p.get("price", p.get("latest_price", p.get("limit_price", 0.0))), 0.0)
        ts = parse_ts(str(record.get("ts") or ""))
        if qty <= 0 or price <= 0:
            continue
        if action == "BUY":
            lots.setdefault(symbol, []).append({"qty": qty, "price": price, "ts": ts})
            continue
        remaining = qty
        while remaining > 0 and lots.get(symbol):
            lot = lots[symbol][0]
            matched = min(remaining, int(lot["qty"]))
            pnl = (price - float(lot["price"])) * matched
            hold_hours = max(0.0, (ts - lot["ts"]).total_seconds() / 3600.0) if ts.year > 1971 else 0.0
            trades.append({
                "Symbol": symbol,
                "Qty": matched,
                "Entry": round(float(lot["price"]), 2),
                "Exit": round(price, 2),
                "P&L": round(pnl, 2),
                "P&L %": round((price / float(lot["price"]) - 1.0) * 100.0, 2),
                "Hold Hours": round(hold_hours, 1),
                "Exit Time": format_price_time(str(record.get("ts") or "")),
            })
            lot["qty"] = int(lot["qty"]) - matched
            remaining -= matched
            if lot["qty"] <= 0:
                lots[symbol].pop(0)
    wins = [t for t in trades if safe_float(t.get("P&L"), 0.0) > 0]
    pnl_values = [safe_float(t.get("P&L"), 0.0) for t in trades]
    summary = {
        "closed_trades": len(trades),
        "win_rate": (len(wins) / len(trades) * 100.0) if trades else 0.0,
        "realized_pnl": sum(pnl_values),
        "avg_pnl": (sum(pnl_values) / len(pnl_values)) if pnl_values else 0.0,
        "avg_hold_hours": (
            sum(safe_float(t.get("Hold Hours"), 0.0) for t in trades) / len(trades)
        ) if trades else 0.0,
    }
    return trades, summary


def _quality_flags_text(flags: Any) -> str:
    if flags in (None, "", [], (), {}):
        return "ok"
    if isinstance(flags, (list, tuple, set)):
        return ", ".join(str(f) for f in flags if str(f).strip()) or "ok"
    return str(flags)


def _render_operational_panels(records: list[dict], journal_path: str) -> None:
    try:
        from ai_trading.config import Settings

        settings = Settings.from_env()
    except Exception:
        settings = None

    def _setting(name: str, default: Any = "") -> Any:
        if settings is not None and hasattr(settings, name):
            return getattr(settings, name)
        env_name = f"BOT_{name.upper()}"
        return os.environ.get(env_name, default)

    latest_records = get_all(records, "latest_price")
    latest_unavailable = get_all(records, "latest_price_unavailable")
    recent_latest = [_payload(r) for r in latest_records[-200:]]
    caution_feeds = {"iex", "delayed_sip", "boats", "overnight"}
    stale_recent = sum(1 for p in recent_latest if p.get("stale"))
    caution_recent = sum(1 for p in recent_latest if str(p.get("feed") or "").lower() in caution_feeds)
    fallback_recent = sum(1 for p in recent_latest if p.get("fallback_from"))
    last_latest_record = latest_records[-1] if latest_records else {}
    last_latest_payload = _payload(last_latest_record)

    with st.expander("Data Health", expanded=False):
        h1, h2, h3, h4, h5 = st.columns(5)
        h1.metric("Latest price events", f"{len(latest_records):,}")
        h2.metric("Unavailable", f"{len(latest_unavailable):,}")
        h3.metric("Stale recent", f"{stale_recent:,}")
        h4.metric("Caution feeds", f"{caution_recent:,}")
        h5.metric("SIP fallbacks", f"{fallback_recent:,}")
        if last_latest_record:
            st.caption(
                "Last latest-price event: "
                f"{last_latest_payload.get('symbol', '—')} · "
                f"{relative_time(str(last_latest_record.get('ts', '')))} · "
                f"{latest_source_label(last_latest_payload, 'journal')}"
            )

        rows = []
        for record in list(reversed(get_all(records, "latest_price", "latest_price_unavailable")))[:15]:
            p = _payload(record)
            rows.append({
                "Event": record.get("event_type", ""),
                "Symbol": p.get("symbol", "—"),
                "Price": f"${safe_float(p.get('price'), 0.0):.2f}" if p.get("price") is not None else "—",
                "Price Time": format_price_time(str(p.get("timestamp") or "")),
                "Feed": str(p.get("feed") or "").upper() or "—",
                "Source": p.get("source", "—"),
                "Age": f"{safe_float(p.get('age_seconds'), 0.0):.0f}s" if p.get("age_seconds") is not None else "—",
                "Stale": bool(p.get("stale", False)),
                "Session": p.get("session", "—"),
                "Reason": p.get("reason", ""),
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No latest-price events have been written yet.")

    with st.expander("Execution Safety", expanded=False):
        safety_rows = [
            ("Paper only", _setting("paper_only", True)),
            ("Stock dry-run", _setting("stock_dry_run", False)),
            ("Kill switch", _setting("kill_switch", False)),
            ("Market data feed", str(_setting("market_data_feed", "auto")).upper()),
            ("Use latest price", _setting("use_latest_price", True)),
            ("Fresh price gate", _setting("require_fresh_price_for_orders", True)),
            ("Max latest age", f"{_setting('max_latest_price_age_sec', 300)}s"),
            ("Stale blocks sells", _setting("stale_price_blocks_sell", False)),
            ("Caution feed blocks buys", _setting("block_caution_feeds_for_buys", False)),
            ("Max buys per cycle", _setting("max_buys_per_cycle", 0)),
            ("Max symbol loss", f"{_setting('max_symbol_loss_pct', 0)}%"),
            ("Gap-up BUY block", f"{_setting('block_buy_gap_up_pct', 0)}%"),
            ("Order type", str(_setting("order_type", "market")).upper()),
            ("Max shares", _setting("max_shares", "—")),
            ("Max daily trades", _setting("max_daily_trades", "—")),
            ("Max open positions", _setting("max_open_positions", "—")),
            ("Discord webhook", "SET" if str(_setting("webhook_url", "") or "").strip() else "MISSING"),
            ("Discord buy channel", "SET" if os.environ.get("BOT_BUY_WEBHOOK_URL", "").strip() else "fallback"),
            ("Discord sell channel", "SET" if os.environ.get("BOT_SELL_WEBHOOK_URL", "").strip() else "fallback"),
            ("Discord other channel", "SET" if os.environ.get("BOT_OTHER_WEBHOOK_URL", "").strip() else "fallback"),
        ]
        safety_df = pd.DataFrame(
            [(name, str(value)) for name, value in safety_rows],
            columns=["Setting", "Value"],
        )
        st.dataframe(safety_df, use_container_width=True, hide_index=True)

        last_exec = None
        for record in reversed(records):
            if record.get("event_type") in {"order_preview", "order", "stock_dry_run", "risk_reject", "user_cancel"}:
                last_exec = record
                break
        if last_exec:
            p = _payload(last_exec)
            st.caption(
                "Last execution event: "
                f"{last_exec.get('event_type', '')} · {p.get('symbol', '—')} · "
                f"{p.get('action', '—')} · {relative_time(str(last_exec.get('ts', '')))}"
            )

    with st.expander("Paper Audit", expanded=False):
        today_start = datetime.now(LOCAL_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
        today_records = []
        for record in records:
            dt = parse_ts(str(record.get("ts", "")))
            if dt.year > 1971 and dt.astimezone(LOCAL_TZ) >= today_start:
                today_records.append(record)

        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric("Previews today", len(get_all(today_records, "order_preview")))
        a2.metric("Orders today", len(get_all(today_records, "order")))
        a3.metric("Dry-runs today", len(get_all(today_records, "stock_dry_run")))
        a4.metric("Risk rejects", len(get_all(today_records, "risk_reject")))
        a5.metric("Errors", len(get_all(today_records, "error", "order_error")))

        perf_rows, perf_summary = _paper_performance(records)
        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("Closed trades", perf_summary["closed_trades"])
        p2.metric("Win rate", f"{perf_summary['win_rate']:.1f}%")
        p3.metric("Realized P&L", f"${perf_summary['realized_pnl']:+,.2f}")
        p4.metric("Avg P&L", f"${perf_summary['avg_pnl']:+,.2f}")
        p5.metric("Avg hold", f"{perf_summary['avg_hold_hours']:.1f}h")
        if perf_rows:
            st.dataframe(pd.DataFrame(list(reversed(perf_rows[-20:]))), use_container_width=True, hide_index=True)

        audit_events = {
            "order_preview", "order", "stock_dry_run", "risk_reject", "fill_status",
            "partial_profit", "partial_remainder_exit", "scanner_snapshot", "sell_scanner_snapshot",
        }
        audit_rows = []
        for record in reversed(records):
            if record.get("event_type") not in audit_events:
                continue
            p = _payload(record)
            audit_rows.append({
                "Time": format_price_time(str(record.get("ts") or "")),
                "Event": record.get("event_type", ""),
                "Symbol": str(p.get("symbol", "—")),
                "Action": str(p.get("action", "—")),
                "Qty": str(p.get("qty", p.get("sold_qty", "—"))),
                "Mode": str(p.get("mode", "—")),
                "Price": str(p.get("price", p.get("latest_price", "—"))),
                "Latest Time": format_price_time(str(p.get("latest_time") or "")),
                "Latest Source": str(p.get("latest_source", p.get("data_source", "—"))),
                "Reason": str(p.get("reason", "")),
            })
            if len(audit_rows) >= 30:
                break
        if audit_rows:
            st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No execution audit events yet.")


def _render_alpaca_paper_positions() -> None:
    st.markdown(
        """
        <div class="ops-kicker">Secondary Broker View</div>
        <div class="ops-title">Alpaca / Paper Positions</div>
        <div class="ops-note">Original bot account view. Robinhood is the primary portfolio surface above.</div>
        """,
        unsafe_allow_html=True,
    )
    positions = load_positions()
    if positions:
        latest_map = load_latest_price_map(tuple(str(p.get("Symbol", "")).upper() for p in positions))
        for p in positions:
            sym = str(p.get("Symbol", "")).upper()
            latest = latest_map.get(sym, {})
            latest_px = latest.get("price")
            if latest_px:
                p["Current Price"] = float(latest_px)
                p["Market Value"] = float(p["Shares"]) * float(latest_px)
                p["Total P&L"] = float(p["Market Value"]) - float(p["Cost Basis"])
                p["Total P&L %"] = (float(p["Total P&L"]) / float(p["Cost Basis"]) * 100.0) if p["Cost Basis"] else 0.0
            p["Latest Time"] = format_price_time(str(latest.get("timestamp") or ""))
            p["Latest Source"] = latest_source_label(latest, "position")
        df_pos = pd.DataFrame(positions)
        df_display = pd.DataFrame({
            "Symbol":        df_pos["Symbol"],
            "Shares":        df_pos["Shares"],
            "Avg Cost":      df_pos["Avg Cost"].map(lambda v: f"${v:,.2f}"),
            "Price":         df_pos["Current Price"].map(lambda v: f"${v:,.2f}"),
            "Price Time":    df_pos["Latest Time"],
            "Source":        df_pos["Latest Source"],
            "Market Value":  df_pos["Market Value"].map(lambda v: f"${v:,.2f}"),
            "Today's P&L":   df_pos.apply(lambda r: "${:+,.2f}  ({:+.2f}%)".format(r["Today's P&L"], r["Today's P&L %"]), axis=1),
            "Total P&L":     df_pos.apply(lambda r: "${:+,.2f}  ({:+.2f}%)".format(r["Total P&L"], r["Total P&L %"]), axis=1),
        })
        st.dataframe(df_display, use_container_width=True, height=min(80 + len(df_display) * 35, 500))

        total_mv = df_pos["Market Value"].sum()
        total_cost = df_pos["Cost Basis"].sum()
        total_pl = df_pos["Total P&L"].sum()
        total_today = df_pos["Today's P&L"].sum()
        total_pl_pct = (total_pl / total_cost * 100) if total_cost else 0.0
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Market Value", f"${total_mv:,.2f}")
        t2.metric("Cost Basis", f"${total_cost:,.2f}")
        t3.metric("Total P&L", f"${total_pl:+,.2f}", f"{total_pl_pct:+.2f}%")
        t4.metric("Today's P&L", f"${total_today:+,.2f}")
    elif load_positions() == [] and os.environ.get("APCA_API_KEY_ID"):
        st.info("No open Alpaca/paper positions.")
    else:
        st.warning("Alpaca credentials not found — set APCA_API_KEY_ID / APCA_API_SECRET_KEY.")


def _render_bot_activity(records: list[dict], orders_all: list[dict]) -> None:
    st.markdown(
        """
        <div class="ops-kicker">Automation</div>
        <div class="ops-title">Bot Activity, Signals, and Risk</div>
        <div class="ops-note">Execution intent, latest model signals, risk rejects, and dry-run trade stats.</div>
        """,
        unsafe_allow_html=True,
    )
    signal_col, risk_col = st.columns(2)

    with signal_col:
        st.subheader("Latest Signals")
        signal_records = get_all(records, "signal")
        if signal_records:
            latest_ensemble_by_symbol: dict[str, dict] = {}
            for record in reversed(get_all(records, "ensemble_decision")):
                payload = _payload(record)
                symbol = str(payload.get("symbol", "") or "").upper()
                if symbol and symbol not in latest_ensemble_by_symbol:
                    latest_ensemble_by_symbol[symbol] = payload

            def _signal_display_score(symbol: str, payload: dict) -> float:
                raw_score = payload.get("score")
                if raw_score is not None:
                    return max(0.0, min(100.0, safe_float(raw_score, 0.0)))
                ensemble = latest_ensemble_by_symbol.get(symbol.upper(), {})
                if ensemble:
                    strength = max(-1.0, min(1.0, safe_float(ensemble.get("strength"), 0.0)))
                    confidence = max(0.0, min(1.0, safe_float(ensemble.get("confidence"), 0.0)))
                    return max(0.0, min(100.0, ((strength + 1.0) * 50.0 * 0.75) + (confidence * 25.0)))
                fast = safe_float(payload.get("fast_ma"), 0.0)
                slow = safe_float(payload.get("slow_ma"), 0.0)
                if fast > 0 and slow > 0:
                    gap_pct = ((fast - slow) / slow) * 100.0
                    return max(0.0, min(100.0, 50.0 + gap_pct * 5.0))
                signal = str(payload.get("signal", "HOLD") or "HOLD").upper()
                return {"BUY": 65.0, "HOLD": 50.0, "SELL": 35.0}.get(signal, 0.0)

            seen: set = set()
            rows = []
            latest_signal_map = load_latest_price_map(tuple(
                str(r.get("payload", {}).get("symbol", "")).upper()
                for r in signal_records[-80:]
            ))
            for r in reversed(signal_records):
                p = r.get("payload", {})
                sym = str(p.get("symbol", "?") or "?").upper()
                if sym in seen:
                    continue
                seen.add(sym)
                latest = latest_signal_map.get(str(sym).upper(), {})
                latest_price = safe_float(latest.get("price"), safe_float(p.get("close")))
                score = _signal_display_score(sym, p)
                rows.append({
                    "Symbol": sym,
                    "Score": round(score, 1),
                    "Signal": p.get("signal", "HOLD"),
                    "Close": f"${safe_float(p.get('close')):.2f}",
                    "Latest Price": f"${latest_price:.2f}",
                    "Price Time": format_price_time(str(latest.get("timestamp") or "")),
                    "Fast MA": f"${safe_float(p.get('fast_ma')):.2f}",
                    "Slow MA": f"${safe_float(p.get('slow_ma')):.2f}",
                    "Pos": int(p.get("position_qty", 0)),
                    "When": relative_time(r.get("ts", "")),
                    "_score_sort": score,
                    "_ts_sort": parse_ts(str(r.get("ts", ""))).timestamp(),
                })
            sig_emoji = {"BUY": "BUY", "SELL": "SELL", "HOLD": "HOLD"}
            df_signals = pd.DataFrame(rows)
            df_signals = df_signals.sort_values(["_score_sort", "_ts_sort"], ascending=[False, False])
            df_signals = df_signals.drop(columns=["_score_sort", "_ts_sort"], errors="ignore")
            df_signals["Signal"] = df_signals["Signal"].map(lambda v: sig_emoji.get(str(v), str(v)))
            st.dataframe(df_signals, use_container_width=True)
        else:
            st.info("No signals yet.")

    with risk_col:
        st.subheader("Risk Rejects & Errors")
        rejects = get_all(records, "risk_reject", "error", "order_error", "gap_open_protect", "correlation_reject")
        if rejects:
            type_label = {
                "error": "error", "risk_reject": "risk_reject", "order_error": "order_error",
                "gap_open_protect": "gap_open", "correlation_reject": "corr_reject",
            }
            rrows = [{
                "Type": type_label.get(r.get("event_type", ""), r.get("event_type", "")),
                "Symbol": r.get("payload", {}).get("symbol", "—"),
                "Reason": (r.get("payload", {}).get("reason") or r.get("payload", {}).get("error", ""))[:80],
                "When": relative_time(r.get("ts", "")),
            } for r in reversed(rejects[-30:])]
            st.dataframe(pd.DataFrame(rrows), use_container_width=True)
        else:
            st.success("No rejects or errors.")

    stats_col, trades_col = st.columns([1, 2])
    with stats_col:
        st.subheader("Trade Stats")
        buys_by_sym: dict[str, list] = {}
        pnl_list: list[float] = []
        for r in orders_all:
            p = r.get("payload", {})
            sym = p.get("symbol", "?")
            action = p.get("action", "")
            qty = int(p.get("qty", 0))
            price = safe_float(p.get("filled_avg_price") or p.get("limit_price") or 0)
            if action == "BUY" and price:
                buys_by_sym.setdefault(sym, []).append(price)
            elif action == "SELL" and price and buys_by_sym.get(sym):
                entry = buys_by_sym[sym].pop(0)
                pnl_list.append((price - entry) * qty)
        if pnl_list:
            wins = [p for p in pnl_list if p > 0]
            losses = [p for p in pnl_list if p <= 0]
            st.metric("Win Rate", f"{len(wins)/len(pnl_list)*100:.1f}%")
            st.metric("Avg Win", f"${sum(wins)/len(wins):+.2f}" if wins else "$0.00")
            st.metric("Avg Loss", f"${sum(losses)/len(losses):+.2f}" if losses else "$0.00")
            gl = abs(sum(losses)) or 1e-9
            st.metric("Profit Factor", f"{sum(wins)/gl:.2f}" if wins else "0.00")
            st.metric("Closed P&L", f"${sum(pnl_list):+.2f}")
        else:
            st.info("Need closed trades.")

    with trades_col:
        st.subheader("Trade Log")
        if orders_all:
            trows = [{
                "Time": r.get("ts", "")[:19],
                "Symbol": r.get("payload", {}).get("symbol", "?"),
                "Action": r.get("payload", {}).get("action", "?"),
                "Qty": r.get("payload", {}).get("qty", 0),
                "Mode": r.get("payload", {}).get("mode", "PAPER"),
                "When": relative_time(r.get("ts", "")),
            } for r in reversed(orders_all[-50:])]
            st.dataframe(pd.DataFrame(trows), use_container_width=True)
        else:
            st.info("No trades yet.")


def _render_equity_history(records: list[dict]) -> None:
    st.markdown(
        """
        <div class="ops-kicker">Performance</div>
        <div class="ops-title">Bot Equity History</div>
        <div class="ops-note">Journal-derived account state over time.</div>
        """,
        unsafe_allow_html=True,
    )
    acct_records = get_all(records, "account_state")
    if len(acct_records) < 2:
        st.info("Need at least 2 account-state data points.")
        return
    edata = [{
        "time": parse_ts(r.get("ts", "")),
        "Equity": safe_float(r.get("payload", {}).get("equity")),
        "Cash": safe_float(r.get("payload", {}).get("cash")),
    } for r in acct_records if r.get("payload", {}).get("equity")]
    df_eq = pd.DataFrame(edata).set_index("time").sort_index()
    if len(df_eq) < 2:
        st.info("Need at least 2 account-state data points.")
        return
    df_eq["Peak"] = df_eq["Equity"].cummax()
    df_eq["Drawdown %"] = (df_eq["Equity"] - df_eq["Peak"]) / df_eq["Peak"] * 100
    t1, t2 = st.tabs(["Equity & Cash", "Drawdown"])
    with t1:
        st.line_chart(df_eq[["Equity", "Cash"]], use_container_width=True)
    with t2:
        st.area_chart(df_eq[["Drawdown %"]], use_container_width=True)


# ── Load portfolio data ───────────────────────────────────────────────────────
records = load_journal(journal_path)

# ── Page title + status ───────────────────────────────────────────────────────
st.title("Robinhood Agent Console")
_paper_mode = os.environ.get("BOT_PAPER_ONLY", "true").lower() == "true"
_stock_dry_run = os.environ.get("BOT_STOCK_DRY_RUN", "false").lower() == "true"
_options_dry_run = os.environ.get("BOT_OPTIONS_DRY_RUN", "true").lower() == "true"
_latest_price_on = os.environ.get("BOT_USE_LATEST_PRICE", "true").lower() == "true"
_market_data_feed = os.environ.get("BOT_MARKET_DATA_FEED", "auto").strip().lower() or "auto"
_primary_broker = os.environ.get("BOT_BROKER", "alpaca").strip().lower() or "alpaca"
_show_alpaca_paper = os.environ.get("BOT_SHOW_ALPACA_PAPER", "false").lower() == "true"
_mode_label = "PAPER" if _paper_mode else "LIVE"
if _paper_mode:
    st.success(f"Mode: {_mode_label} · Broker: {_primary_broker.upper()} · Price feed: {_market_data_feed.upper()}")
elif _stock_dry_run:
    st.warning("Robinhood live account selected; stock dry-run is ON, so local bot actions are logged as review intents.")
else:
    st.error("LIVE TRADING MODE ACTIVE. Confirm every real Robinhood order before submission.")
if _market_data_feed != "sip":
    st.info(
        "Execution and portfolio prices prefer fresh Robinhood quote snapshots. When Robinhood quotes are stale or "
        "missing, the dashboard falls back to current Alpaca/yfinance data instead of showing old prices. "
        "Technical indicators still use historical bars from Alpaca/yfinance until a Robinhood historical-feed client is added."
    )
c1, c2 = st.columns([3, 1])
with c1:
    st.caption(f"Journal: `{journal_path}` · {len(records)} events")
with c2:
    last_ts = records[-1].get("ts", "") if records else ""
    stale = not last_ts or (datetime.now(timezone.utc) - parse_ts(last_ts)).total_seconds() > 3600
    st.metric("Bot last active", f"{'🔴' if stale else '🟢'} {relative_time(last_ts)}")

_render_operational_panels(records, journal_path)

# ── Main content area (driven by sidebar radio, preserves selection on rerun) ──


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PORTFOLIO MONITOR
# ════════════════════════════════════════════════════════════════════════════════
if active_key == "portfolio":
    _render_page_header(
        "Portfolio",
        "Robinhood Investing and Agentic holdings, quote-backed P&L, and position-level action reads.",
        ["Primary: Robinhood", "Actions require review", "Paper hidden by default"],
    )
    account = get_latest(records, "account_state")
    orders_all = get_all(records, "order")
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    orders_today = [r for r in orders_all if parse_ts(r.get("ts", "")) >= today_start]
    _rh_portfolios_path = str(globals().get("robinhood_portfolios_path", "") or "logs/robinhood_portfolios.json")
    _rh_quotes_path = str(globals().get("robinhood_quote_path", "") or "logs/robinhood_quotes.json")
    _rh_accounts_for_metrics = load_robinhood_portfolio_snapshot(_rh_portfolios_path)
    if st.session_state.pop("_rh_rescore_after_refresh", False) and _rh_accounts_for_metrics:
        _horizon_label = str(st.session_state.get("rh_portfolio_horizon", "Short-term") or "Short-term")
        _horizon = "long" if _horizon_label == "Long-term" else "short"
        try:
            _rows, _full = _recommend_robinhood_holdings(_rh_accounts_for_metrics, _horizon)
            st.session_state["rh_portfolio_rows"] = _rows
            st.session_state["rh_portfolio_full"] = _full
            st.session_state["rh_portfolio_ts"] = format_local_now("%I:%M:%S %p %Z")
            st.success("Robinhood holdings were re-scored from the refreshed snapshot.")
        except Exception as exc:
            st.warning(f"Snapshot refreshed, but automatic portfolio re-score failed: {exc}")
    _render_snapshot_status_panel(_rh_portfolios_path, _rh_quotes_path)

    # Metrics row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    if _rh_accounts_for_metrics:
        _rh_total_value = sum(_portfolio_metric(a.get("portfolio", {}), "total_value", "portfolio_value") for a in _rh_accounts_for_metrics)
        _rh_equity_value = sum(_portfolio_metric(a.get("portfolio", {}), "equity_value") for a in _rh_accounts_for_metrics)
        _rh_cash = sum(_portfolio_metric(a.get("portfolio", {}), "cash") for a in _rh_accounts_for_metrics)
        _rh_buying_power = sum(_portfolio_buying_power(a.get("portfolio", {})) for a in _rh_accounts_for_metrics)
        _rh_pnl = _robinhood_equity_pnl(_rh_accounts_for_metrics)
        _rh_holdings = sum(len(a.get("positions", [])) for a in _rh_accounts_for_metrics)
        _rh_agentic = next((a for a in _rh_accounts_for_metrics if a.get("agentic")), None)
        m1.metric("RH Account Value", fmt_money(_rh_total_value))
        m2.metric("RH Equity Value", fmt_money(_rh_equity_value))
        m3.metric("RH Total P/L", f"${_rh_pnl['pnl']:+,.2f}", f"{_rh_pnl['pnl_pct']:+.2f}%")
        m4.metric("RH Cash", fmt_money(_rh_cash))
        m5.metric("RH Buying Power", fmt_money(_rh_buying_power))
        m6.metric("RH Holdings", f"{_rh_holdings:,}")
        if _rh_agentic:
            st.caption(f"Agentic account: {_rh_agentic.get('account', '—')}")
    elif account:
        equity = safe_float(account.get("equity"))
        cash = safe_float(account.get("cash"))
        last_equity = safe_float(account.get("last_equity", equity))
        portfolio_value = safe_float(account.get("portfolio_value", equity))
        pnl_today = equity - last_equity
        pnl_pct = (pnl_today / last_equity * 100) if last_equity else 0
        acct_records = get_all(records, "account_state")
        peak = max((safe_float(r.get("payload", {}).get("equity")) for r in acct_records), default=equity) if acct_records else equity
        drawdown_pct = ((equity - peak) / peak * 100) if peak > 0 else 0
        m1.metric("Equity", fmt_money(equity), f"{pnl_pct:+.2f}%")
        m2.metric("Cash", fmt_money(cash))
        m3.metric("Portfolio Value", fmt_money(portfolio_value))
        m4.metric("P&L Today", f"${pnl_today:+,.2f}")
        m5.metric("Drawdown", f"{drawdown_pct:.2f}%")
        m6.metric("Trades Today", str(len(orders_today)))
    else:
        st.info("No account data yet — run the bot first.")

    st.markdown("---")
    _render_robinhood_portfolios_panel(_rh_portfolios_path)

    st.markdown("---")
    if _show_alpaca_paper:
        activity_tab, paper_tab, history_tab = st.tabs(["Bot Activity", "Alpaca / Paper", "History"])
        with activity_tab:
            _render_bot_activity(records, orders_all)
        with paper_tab:
            _render_alpaca_paper_positions()
        with history_tab:
            _render_equity_history(records)
    else:
        activity_tab, history_tab = st.tabs(["Bot Activity", "History"])
        with activity_tab:
            _render_bot_activity(records, orders_all)
    with history_tab:
        _render_equity_history(records)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ACTION QUEUE
# ════════════════════════════════════════════════════════════════════════════════
elif active_key == "actions":
    _render_action_queue(records)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — LIVE SCANNER
# ════════════════════════════════════════════════════════════════════════════════
elif active_key == "scanner":
    market_open = is_market_open()
    _last_scanner_source = str(st.session_state.get("scanner_data_source", "") or "")
    _last_scanner_source = _last_scanner_source.strip().lower()
    _source_label = _last_scanner_source.upper() if _last_scanner_source else "AUTO"
    mode_badge = (
        "🟢 MARKET OPEN — Live Intraday Data (Alpaca IEX 5-min)"
        if market_open
        else f"🔴 MARKET CLOSED — End-of-Day Data ({_source_label})"
    )
    _render_page_header(
        "Buy Scanner",
        "Find calm upside candidates, then verify Robinhood quote/tradability before any order review.",
        ["Ranking: technical bars", "Display prices: Robinhood overlay", mode_badge],
    )

    if market_open:
        st.caption(
            "Live mode: scores symbols on **VWAP deviation** (30%), **intraday momentum** (25%), "
            "**volume spike** (20%), **intraday RSI** (15%), **bar trend** (10%)."
        )
    else:
        st.caption(
            "EOD mode: scores symbols on **MA crossover** (30%), **5-day momentum** (25%), "
            "**RSI oversold** (20%), **volume surge** (15%), **trend consistency** (10%)."
        )
    st.caption("Technical scoring uses Alpaca/yfinance historical bars; displayed prices prefer fresh Robinhood quotes and fall back to current market data when Robinhood snapshots are stale.")
    scan_cmd_1, scan_cmd_2 = st.columns([1, 3])
    with scan_cmd_1:
        page_run_scan = st.button("Run Buy Scanner", type="primary", key="scanner_run_page", use_container_width=True)
    with scan_cmd_2:
        st.caption("Primary scanner action is available here and in the sidebar.")
    if page_run_scan:
        _lock_autorefresh()
    scanner_run_now = run_scan or page_run_scan

    # Resolve symbol list. Remote/full index membership is loaded only when scanning.
    if scanner_run_now:
        if _use_full_universe:
            with st.spinner("Loading full universe from Alpaca…"):
                symbols_to_scan = _sidebar_universe_symbols(load_remote=True)
            if not symbols_to_scan:
                st.error("Could not fetch universe from Alpaca. Check credentials.")
        elif _use_index_universe:
            with st.spinner("Loading selected index membership…"):
                symbols_to_scan = _sidebar_universe_symbols(load_remote=True)
            if not symbols_to_scan:
                st.error("No index symbols loaded. Pick at least one index or refresh cache later.")
            else:
                labels = ", ".join(INDEX_LABELS.get(a, a) for a in _selected_indexes)
                st.info(f"Loaded **{len(symbols_to_scan):,}** symbols from {labels}.")
        else:
            symbols_to_scan = _sidebar_universe_symbols(load_remote=False)
    else:
        symbols_to_scan = _sidebar_universe_symbols(load_remote=False)

    # Apply pre-scan snapshot filter to trim the universe before deep scanning
    pre_filter_count = len(symbols_to_scan)
    if scanner_run_now and symbols_to_scan and _use_filters:
        _filter_status = st.empty()
        _filter_status.info(f"⚡ Pre-filtering {pre_filter_count:,} symbols via snapshots…")
        symbols_to_scan = _snapshot_filter(
            symbols=symbols_to_scan,
            min_price=_f_min_price,
            max_price=_f_max_price,
            min_volume=int(_f_min_vol),
            min_change_pct=_f_min_chg,
            max_change_pct=_f_max_chg,
            exchanges=[],
        )
        _filter_status.success(
            f"✅ Filters reduced universe: **{pre_filter_count:,} → {len(symbols_to_scan):,} symbols** "
            f"(price ${_f_min_price:.0f}–${_f_max_price:.0f}, vol ≥ {int(_f_min_vol):,}, "
            f"chg {_f_min_chg:+.1f}% to {_f_max_chg:+.1f}%)"
        )

    if symbols_to_scan and not scanner_run_now:
        st.info(
            f"Watchlist: **{len(symbols_to_scan):,} symbols** — "
            f"{', '.join(symbols_to_scan[:12])}{'…' if len(symbols_to_scan) > 12 else ''}"
        )
    elif not symbols_to_scan and not scanner_run_now:
        if _use_full_universe:
            st.info("Full universe mode — click **Run Buy Scanner** to fetch all ~12K tickers and scan.")
        elif _use_index_universe:
            labels = ", ".join(INDEX_LABELS.get(a, a) for a in _selected_indexes) or "selected indexes"
            st.info(f"Index mode — click **Run Buy Scanner** to fetch and scan {labels}.")
        else:
            st.info("Click **Run Buy Scanner** to scan the watchlist.")

    # Only scan when user explicitly clicks the button; run the deep scan off the render thread.
    needs_scan = scanner_run_now
    if needs_scan and symbols_to_scan:
        _mode_label = "live" if market_open else "EOD"
        started = _start_background_task(
            "scanner_scan_task",
            _run_dashboard_scan,
            tuple(symbols_to_scan),
            market_open,
            int(scanner_fast_ma),
            int(scanner_slow_ma),
            int(scanner_top_n),
            str(scanner_scan_depth).lower(),
            meta={
                "mode_label": _mode_label,
                "symbol_count": len(symbols_to_scan),
                "pre_filter": pre_filter_count,
                "post_filter": len(symbols_to_scan),
                "scan_depth": str(scanner_scan_depth).lower(),
            },
        )
        if started:
            st.session_state["scanner_page"] = 0
            st.info(f"Started {_mode_label} scan for **{len(symbols_to_scan):,}** symbols in the background.")
        else:
            st.info("A scanner job is already running; keeping the dashboard responsive while it finishes.")

    scan_pending = False
    scan_task = _poll_background_task("scanner_scan_task")
    if scan_task:
        meta = scan_task.get("meta", {})
        if scan_task["status"] == "running":
            scan_pending = True
            st.info(
                f"Background scan running for **{int(meta.get('symbol_count', 0)):,}** symbols "
                f"({scan_task['elapsed']:.0f}s elapsed). Showing the last completed results until it finishes."
            )
            _schedule_scan_poll()
        elif scan_task["status"] == "done":
            payload = scan_task["result"]
            results = payload.get("results", [])
            st.session_state["scanner_results"] = results
            st.session_state["scanner_ts"] = format_local_now("%I:%M:%S %p %Z")
            st.session_state["scanner_mode"] = payload.get("mode", "eod")
            st.session_state["scanner_depth"] = payload.get("scan_depth", meta.get("scan_depth", "deep"))
            st.session_state["scanner_data_source"] = payload.get(
                "data_source",
                "alpaca" if payload.get("mode") == "live" else "auto",
            )
            st.session_state["scanner_pre_filter"] = int(meta.get("pre_filter", 0) or 0)
            st.session_state["scanner_post_filter"] = int(meta.get("post_filter", 0) or 0)
            if payload.get("fallback"):
                st.warning("Live data returned no results — completed with EOD data instead.")
            if payload.get("cache_hit"):
                st.info("Loaded from dashboard scan cache.")
            st.success(f"Scan complete: **{len(results)}** results.")
            snapshot_payload = {
                "mode": st.session_state["scanner_mode"],
                "data_source": st.session_state["scanner_data_source"],
                "scan_depth": st.session_state["scanner_depth"],
                "result_count": len(results),
                "pre_filter": st.session_state["scanner_pre_filter"],
                "post_filter": st.session_state["scanner_post_filter"],
                "top": [
                    {
                        "symbol": str(item.get("symbol", "") if isinstance(item, dict) else getattr(item, "symbol", "")),
                        "score": safe_float(item.get("score", 0.0) if isinstance(item, dict) else getattr(item, "score", 0.0)),
                        "signal": str(item.get("signal", "") if isinstance(item, dict) else getattr(item, "signal", "")),
                    }
                    for item in results[:10]
                ],
            }
            _record_dashboard_event(journal_path, "scanner_snapshot", snapshot_payload)
            _notify_dashboard_event(
                "scanner_summary",
                f"Scanner complete: {len(results)} results ({snapshot_payload['scan_depth']} {snapshot_payload['mode']})",
                snapshot_payload,
            )
        elif scan_task["status"] == "error":
            st.error(f"Scanner error: {scan_task['error']}")
            st.session_state["scanner_results"] = []

    results = st.session_state.get("scanner_results", [])
    scan_ts = st.session_state.get("scanner_ts", "—")
    scan_mode = st.session_state.get("scanner_mode", "—")
    scan_depth = st.session_state.get("scanner_depth", "deep")
    scan_source = str(st.session_state.get("scanner_data_source", "") or "")
    _s_pre  = st.session_state.get("scanner_pre_filter", 0)
    _s_post = st.session_state.get("scanner_post_filter", 0)

    # Status bar
    if scan_ts != "—":
        _filter_note = f" · filtered {_s_pre:,} → {_s_post:,}" if _s_pre > _s_post > 0 else ""
        _live_color = "#1e3a1e" if market_open else "#1e2a3a"
        _live_text  = "#86efac" if market_open else "#93c5fd"
        st.markdown(
            f"<div style='background:{_live_color};padding:8px 12px;border-radius:6px;"
            f"font-size:0.85em;color:{_live_text}'>"
            f"{'⚡ Live' if scan_mode == 'live' else '📅 EOD'} scan · "
            f"<b>{str(scan_depth).upper()}</b> · "
            f"<b>{len(results)}</b> results from <b>{_s_post or '?'}</b> symbols{_filter_note} · "
            f"as of <b>{scan_ts}</b>"
            f"{(' · source <b>' + str(scan_source).upper() + '</b>') if scan_source else ''}"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("")

    if not results and not scanner_run_now and not scan_pending:
        st.info("Click **Run Buy Scanner** above to scan the watchlist.")

    if results:
        # Legend
        lcol, mcol, rcol = st.columns(3)
        lcol.success("🟢 BUY — score ≥ 65")
        mcol.warning("🟡 WATCH — score 45–64")
        rcol.info("⚪ NEUTRAL — score < 45")
        st.markdown("---")

        # Dynamic column labels based on mode
        is_live_results = scan_mode == "live"
        col5d_label  = "Intraday %" if is_live_results else "5D %"
        col_ma_label = "VWAP Gap %" if is_live_results else "MA Gap %"
        def _rget(item, key: str, default=None):
            if isinstance(item, dict):
                return item.get(key, default)
            return getattr(item, key, default)

        table_rows = []
        for i, r in enumerate(results):
            symbol = str(_rget(r, "symbol", "") or "").upper()
            close_v = safe_float(_rget(r, "close", 0.0), 0.0)
            upside_pct = safe_float(_rget(r, "upside_pct", 0.0), 0.0)
            score_v = safe_float(_rget(r, "score", 0.0), 0.0)
            change_pct = safe_float(_rget(r, "change_pct", 0.0), 0.0)
            momentum_5d = safe_float(_rget(r, "momentum_5d", 0.0), 0.0)
            rsi_v = safe_float(_rget(r, "rsi", 0.0), 0.0)
            volume_surge = safe_float(_rget(r, "volume_surge", 0.0), 0.0)
            ma_gap_pct = safe_float(_rget(r, "ma_gap_pct", 0.0), 0.0)
            trend_consistency = safe_float(_rget(r, "trend_consistency", 0.0), 0.0)
            signal_v = str(_rget(r, "signal", "NEUTRAL") or "NEUTRAL")
            mode_v = str(_rget(r, "mode", scan_mode) or scan_mode)
            top_driver_v = str(_rget(r, "top_driver", "") or "")
            reason_v = str(_rget(r, "reason", "") or "")
            quality_raw = _rget(r, "quality_pass", True)
            quality_pass_v = str(quality_raw).strip().lower() not in {"false", "0", "no"}
            quality_flags_v = _quality_flags_text(_rget(r, "quality_flags", "ok"))
            table_rows.append({
                "Rank": i + 1,
                "Symbol": symbol,
                "Score": score_v,
                "Est. Upside": f"+{upside_pct:.1f}%" if upside_pct > 0 else "—",
                "Signal": signal_v,
                "Trade Gate": "PASS" if quality_pass_v else "BLOCK",
                "Quality Flags": quality_flags_v,
                "Price": f"${close_v:.2f}",
                "Price Time": "—",
                "Price Source": latest_source_label({}, mode_v),
                "Data Confidence": "LOW (0)",
                "1D %": f"{change_pct:+.2f}%",
                col5d_label: f"{momentum_5d:+.2f}%",
                "RSI": round(rsi_v, 1),
                "Vol Surge": f"{volume_surge:.1f}x",
                col_ma_label: f"{ma_gap_pct:+.2f}%",
                "Trend": f"{trend_consistency:.0f}%",
                "Top Driver": top_driver_v,
                "Plain English": explain_buy(r),
                "Full Reason": reason_v,
            })

        # ── Search + pagination controls ─────────────────────────────────────
        _PAGE_SIZE = 25
        search_col, spacer_col = st.columns([2, 4])
        with search_col:
            search_q = st.text_input(
                "🔎 Search symbol",
                value=st.session_state.get("scanner_search", ""),
                placeholder="e.g. NVDA",
                key="scanner_search",
            ).strip().upper()

        # Filter by search
        if search_q:
            filtered_rows = [r for r in table_rows if search_q in r["Symbol"]]
        else:
            filtered_rows = table_rows

        total_rows = len(filtered_rows)
        total_pages = max(1, (total_rows + _PAGE_SIZE - 1) // _PAGE_SIZE)

        # Reset page when scan reruns or search changes
        if scanner_run_now or st.session_state.get("_last_search") != search_q:
            st.session_state["scanner_page"] = 0
            st.session_state["_last_search"] = search_q
        current_page = st.session_state.get("scanner_page", 0)
        current_page = max(0, min(current_page, total_pages - 1))

        page_start = current_page * _PAGE_SIZE
        page_rows  = filtered_rows[page_start: page_start + _PAGE_SIZE]
        top3 = results[:min(3, len(results))]

        visible_result_symbols = tuple(
            dict.fromkeys(
                [str(row.get("Symbol", "") or "").upper() for row in page_rows]
                + [str(_rget(r, "symbol", "") or "").upper() for r in top3]
            )
        )
        latest_result_map = load_latest_price_map(visible_result_symbols)

        for row in page_rows:
            symbol = str(row.get("Symbol", "") or "").upper()
            latest = latest_result_map.get(symbol, {})
            if latest:
                price = safe_float(latest.get("price"), None)
                if price is not None:
                    row["Price"] = f"${price:.2f}"
                row["Price Time"] = format_price_time(str(latest.get("timestamp") or ""))
                row["Price Source"] = latest_source_label(latest, row.get("Price Source", ""))
                conf_label, conf_score, _conf_reason = data_confidence(latest)
                row["Data Confidence"] = f"{conf_label} ({conf_score})"
            rh_check, _rh_note = robinhood_quote_confirmation(symbol, latest)
            row["Robinhood Check"] = rh_check

        # Pagination nav
        if total_pages > 1:
            nav_l, nav_mid, nav_r = st.columns([1, 3, 1])
            with nav_l:
                if st.button("◀ Prev", disabled=current_page == 0):
                    st.session_state["scanner_page"] = current_page - 1
                    st.rerun()
            with nav_mid:
                st.markdown(
                    f"<div style='text-align:center;padding-top:6px;color:#94a3b8'>"
                    f"Page {current_page + 1} of {total_pages} "
                    f"({total_rows} result{'s' if total_rows != 1 else ''})</div>",
                    unsafe_allow_html=True,
                )
            with nav_r:
                if st.button("Next ▶", disabled=current_page >= total_pages - 1):
                    st.session_state["scanner_page"] = current_page + 1
                    st.rerun()
        else:
            st.caption(f"{total_rows} result{'s' if total_rows != 1 else ''}")

        df_scan = pd.DataFrame(page_rows)

        if df_scan.empty:
            st.info("No results match your search.")
        else:
            def style_score(v):
                if not isinstance(v, (int, float)):
                    return ""
                if v >= 65:  return "background-color:#14532d;color:#86efac;font-weight:bold"
                if v >= 45:  return "background-color:#713f12;color:#fde68a"
                return "color:#94a3b8"

            def style_signal(v):
                if v == "BUY":   return "background-color:#14532d;color:#86efac;font-weight:bold"
                if v == "WATCH": return "background-color:#713f12;color:#fde68a"
                return "color:#94a3b8"

            def style_upside(v):
                if not isinstance(v, str) or v == "—": return "color:#475569"
                try:
                    val = float(v.replace("+","").replace("%",""))
                    if val >= 5: return "color:#86efac;font-weight:bold"
                    if val >= 2: return "color:#fde68a"
                    return "color:#94a3b8"
                except Exception:
                    return ""

            # Emoji-prefix Signal; plain dataframe avoids PyArrow ThreadPoolExecutor crash.
            _SCAN_SIG = {"BUY": "🟢 BUY", "WATCH": "🟡 WATCH", "NEUTRAL": "⚪ NEUTRAL"}
            df_scan_display = df_scan.copy()
            df_scan_display["Signal"] = df_scan_display["Signal"].map(lambda v: _SCAN_SIG.get(str(v), str(v)))
            st.dataframe(df_scan_display, use_container_width=True, height=min(80 + len(page_rows) * 35, 700))

            detail_symbols = [str(r.get("Symbol", "") or "") for r in filtered_rows if r.get("Symbol")]
            if detail_symbols:
                with st.expander("Ticker Explain + Robinhood Manual Ticket", expanded=False):
                    selected_symbol = st.selectbox(
                        "Symbol",
                        options=detail_symbols[:300],
                        index=0,
                        key="scanner_detail_symbol",
                    )
                    selected_result = _find_scan_result(results, selected_symbol)
                    selected_latest = latest_result_map.get(selected_symbol.upper(), {})
                    if not selected_latest:
                        selected_latest = load_latest_price_map((selected_symbol.upper(),)).get(selected_symbol.upper(), {})
                    if selected_result is None:
                        st.info("Select a symbol from the current scan results.")
                    else:
                        conf_label, conf_score, conf_reason = data_confidence(selected_latest)
                        rh_check, rh_note = robinhood_quote_confirmation(selected_symbol, selected_latest)
                        ticket = _manual_order_ticket(selected_symbol.upper(), selected_result, selected_latest, records)
                        e1, e2, e3, e4, e5 = st.columns(5)
                        e1.metric("Score", f"{safe_float(_obj_get(selected_result, 'score'), 0.0):.1f}")
                        e2.metric("Signal", str(_obj_get(selected_result, "signal", "—")))
                        e3.metric("Data", f"{conf_label} {conf_score}")
                        e4.metric("Latest", f"${ticket['latest_price']:.2f}" if ticket["latest_price"] else "—")
                        e5.metric("Est. Upside", f"+{safe_float(_obj_get(selected_result, 'upside_pct'), 0.0):.1f}%")

                        _research_spec_selected = ResearchPromptSpec(
                            subject=selected_symbol.upper(),
                            goals=st.session_state.get("research_goals", "long-term capital appreciation"),
                            risk_tolerance=st.session_state.get("research_risk", "moderate"),
                            time_horizon=st.session_state.get("research_horizon", "5+ years"),
                            as_of_date=st.session_state.get(
                                "research_as_of_date",
                                datetime.now(app_timezone()).strftime("%B %d, %Y"),
                            ),
                        )
                        _research_mode_selected = st.session_state.get("research_mode", "memo")
                        explain_tab, research_tab, ticket_tab = st.tabs(["Explain", "Research Prompt", "Robinhood Ticket"])
                        with explain_tab:
                            st.markdown(f"**Plain English:** {explain_buy(selected_result)}")
                            st.markdown(f"**Top driver:** {_obj_get(selected_result, 'top_driver', '—')}")
                            st.markdown(f"**Full reason:** {_obj_get(selected_result, 'reason', '—')}")
                            st.markdown(f"**Robinhood quote check:** {rh_check} — {rh_note}")
                            st.markdown(
                                f"**Quality gate:** {'PASS' if bool(_obj_get(selected_result, 'quality_pass', True)) else 'BLOCK'} "
                                f"({_quality_flags_text(_obj_get(selected_result, 'quality_flags', 'ok'))})"
                            )
                            st.markdown(f"**Data confidence:** {conf_label} ({conf_score}) - {conf_reason}")
                            detail_df = pd.DataFrame([
                                ["RSI", f"{safe_float(_obj_get(selected_result, 'rsi'), 0.0):.1f}"],
                                ["1D %", f"{safe_float(_obj_get(selected_result, 'change_pct'), 0.0):+.2f}%"],
                                ["5D / intraday momentum", f"{safe_float(_obj_get(selected_result, 'momentum_5d'), 0.0):+.2f}%"],
                                ["Volume surge", f"{safe_float(_obj_get(selected_result, 'volume_surge'), 0.0):.1f}x"],
                                ["MA/VWAP gap", f"{safe_float(_obj_get(selected_result, 'ma_gap_pct'), 0.0):+.2f}%"],
                                ["Trend consistency", f"{safe_float(_obj_get(selected_result, 'trend_consistency'), 0.0):.0f}%"],
                                ["Relative strength", f"{safe_float(_obj_get(selected_result, 'rel_strength_pct'), 0.0):+.2f}%"],
                                ["Avg dollar volume", f"${safe_float(_obj_get(selected_result, 'avg_dollar_vol_m'), 0.0):.1f}M"],
                            ], columns=["Metric", "Value"])
                            st.dataframe(detail_df, use_container_width=True, hide_index=True)
                        with research_tab:
                            _scan_prompt = build_prompt_for_mode(_research_mode_selected, _research_spec_selected)
                            st.caption(
                                f"Prompt mode: **{str(_research_mode_selected).upper()}** · goals **{_research_spec_selected.goals}** · "
                                f"risk **{_research_spec_selected.risk_tolerance}** · horizon **{_research_spec_selected.time_horizon}**"
                            )
                            st.text_area(
                                "Prompt",
                                value=_scan_prompt,
                                height=360,
                                key=f"scanner_research_prompt_{selected_symbol}",
                            )
                            if st.button("Save Research Packet", key=f"scanner_research_save_{selected_symbol}"):
                                _queue_research_packet_event(
                                    journal_path=journal_path,
                                    spec=_research_spec_selected,
                                    mode=_research_mode_selected,
                                    source="scanner_detail",
                                    context={
                                        "symbol": selected_symbol.upper(),
                                        "signal": str(_obj_get(selected_result, "signal", "")),
                                        "score": safe_float(_obj_get(selected_result, "score", 0.0), 0.0),
                                        "top_driver": str(_obj_get(selected_result, "top_driver", "") or ""),
                                    },
                                )
                                st.success(f"Saved research packet for {selected_symbol.upper()}.")
                        with ticket_tab:
                            st.code(
                                "\n".join([
                                    f"{ticket['action']} {ticket['symbol']}",
                                    f"Suggested qty: {ticket['suggested_qty']}",
                                    f"Latest price: ${ticket['latest_price']:.2f}",
                                    f"Limit price: ${ticket['limit_price']:.2f}",
                                    f"Stop: ${ticket['stop']:.2f}",
                                    f"Target: ${ticket['target']:.2f}",
                                    f"Risk/share: ${ticket['risk_per_share']:.2f}",
                                    f"Risk budget: ${ticket['risk_budget']:.2f}",
                                    f"Data confidence: {ticket['confidence']} ({ticket['confidence_score']})",
                                ]),
                                language="text",
                            )
                            st.caption(
                                "Manual-use ticket only. The dashboard is not placing Robinhood orders."
                            )

        # ── Top 3 Pick Cards (always from full unfiltered results) ───────────
        st.markdown("---")
        st.subheader("⭐ Top Picks Right Now")
        card_cols = st.columns(len(top3))
        for col, r in zip(card_cols, top3):
            symbol = str(_rget(r, "symbol", "") or "").upper()
            latest = latest_result_map.get(symbol, {})
            card_price = safe_float(latest.get("price"), safe_float(_rget(r, "close", 0.0), 0.0))
            card_time = format_price_time(str(latest.get("timestamp") or ""))
            signal_v = str(_rget(r, "signal", "NEUTRAL") or "NEUTRAL")
            score_v = safe_float(_rget(r, "score", 0.0), 0.0)
            upside_v = safe_float(_rget(r, "upside_pct", 0.0), 0.0)
            change_v = safe_float(_rget(r, "change_pct", 0.0), 0.0)
            top_driver_v = str(_rget(r, "top_driver", "") or "")
            reason_v = str(_rget(r, "reason", "") or "")
            bg = "#14532d" if signal_v == "BUY" else "#713f12" if signal_v == "WATCH" else "#1e293b"
            badge = "🟢 BUY" if signal_v == "BUY" else "🟡 WATCH" if signal_v == "WATCH" else "⚪ NEUTRAL"
            upside_str = f"+{upside_v:.1f}% est. upside" if upside_v > 0 else ""
            col.markdown(
                    f"""
                    <div style="background:{bg};border-radius:12px;padding:20px;text-align:center">
                        <h2 style="margin:0 0 4px;color:white;font-size:2em">{symbol}</h2>
                        <p style="font-size:2.4em;margin:0;color:white;font-weight:bold;line-height:1">{score_v:.1f}</p>
                        <p style="margin:4px 0 2px;color:#d1fae5;font-size:1.1em">{badge}</p>
                        {f'<p style="margin:0 0 6px;color:#86efac;font-size:1.05em;font-weight:bold">{upside_str}</p>' if upside_str else ''}
                        <hr style="border-color:rgba(255,255,255,0.2);margin:8px 0">
                        <p style="margin:2px 0;color:#e2e8f0">
                            <b>${card_price:.2f}</b> &nbsp;
                            <span style="color:{'#86efac' if change_v >= 0 else '#fca5a5'}">{change_v:+.2f}% today</span>
                        </p>
                        <p style="margin:2px 0;color:#94a3b8;font-size:0.72em">{card_time}</p>
                        <p style="margin:4px 0 2px;color:#fde68a;font-size:0.85em;font-weight:bold">
                            💡 {top_driver_v}
                        </p>
                        <p style="margin:6px 0 2px;color:#e2e8f0;font-size:0.88em">🗣️ {explain_buy(r)}</p>
                        <p style="margin:2px 0;color:#94a3b8;font-size:0.75em;font-style:italic">{reason_v}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
            )

        # ── Score + Upside chart (filtered view) ───────────────────────────────────────
        st.markdown("---")
        chart_title = f"📊 Score & Est. Upside{' — search: ' + search_q if search_q else ''}"
        st.subheader(chart_title)
        chart_source = filtered_rows if search_q else table_rows
        _chart_syms    = [r["Symbol"] for r in chart_source]
        _chart_scores  = [r["Score"] for r in chart_source]
        _chart_upsides = []
        for r in chart_source:
            try:
                _chart_upsides.append(float(r["Est. Upside"].replace("+","").replace("%","")))
            except Exception:
                _chart_upsides.append(0.0)
        chart_df = pd.DataFrame({
            "Symbol": _chart_syms,
            "Score": _chart_scores,
            "Est. Upside %": _chart_upsides,
        }).set_index("Symbol")
        _ct1, _ct2 = st.tabs(["Score", "Est. Upside %"])
        with _ct1:
            st.bar_chart(chart_df[["Score"]], use_container_width=True)
        with _ct2:
            st.bar_chart(chart_df[["Est. Upside %"]], use_container_width=True)

    else:
        if not symbols_to_scan:
            st.warning("Add symbols to the watchlist in the sidebar.")
        elif scan_pending:
            pass
        elif not needs_scan:
            st.info("Click **Run Buy Scanner** above to scan the market.")
        else:
            st.warning("No results returned. Market data may be unavailable.")

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SELL SCANNER (mirror of Live Scanner, ranks symbols to SELL/TRIM)
# ════════════════════════════════════════════════════════════════════════════════
elif active_key == "sell":
    market_open = is_market_open()
    mode_badge = "🟢 MARKET OPEN — Live Intraday Data" if market_open else "🔴 MARKET CLOSED — End-of-Day Data"
    _render_page_header(
        "Sell / Trim Scanner",
        "Rank positions and watchlist names for profit-taking, trim, hold, or risk-exit decisions.",
        ["Robinhood prices preferred", "No auto-sell", mode_badge],
    )
    st.caption(
        "Flags **peaks** — overbought RSI, extension above MA/VWAP, parabolic moves, volume climaxes, "
        "and big unrealized gains. **TAKE PROFIT** ≥ 70, **TRIM** 50–70, **HOLD** < 50. "
        "Designed to exit *into strength*, not after a drawdown."
    )

    _sell_col1, _sell_col2, _sell_col3 = st.columns([1, 1, 2])
    with _sell_col1:
        _ss_include_holdings = st.checkbox("Scan my open positions", value=True, key="ss_pos")
    with _sell_col2:
        _ss_include_watch = st.checkbox("Scan watchlist too", value=False, key="ss_watch")
    with _sell_col3:
        _ss_extra = st.text_input(
            "Extra symbols (comma-separated)", value="",
            help="Tickers you're considering shorting or want a sell read on.",
            key="ss_extra",
        )

    _ss_full = st.checkbox(
        "🌐 Scan ENTIRE market (~12K tickers)", value=False, key="ss_full",
        help="Pulls every tradable symbol from Alpaca and uses the sidebar pre-scan filters "
             "to trim the universe before deep scanning. Takes 1–3 minutes."
    )
    if _ss_full and not _use_filters:
        st.warning("⚠️ Sidebar pre-scan filters are OFF. Enable **Apply filters before scanning** "
                   "in the sidebar to avoid scoring all ~12K tickers (very slow).")

    _ss_horizon = st.radio(
        "Holding horizon",
        ["⚡ Short-term (swing/day-trade)", "📈 Long-term investor (let winners run)"],
        horizontal=True, key="ss_horizon",
        help="Long-term mode dampens TAKE PROFIT signals on stocks in sustained uptrends — "
             "a strong trend with healthy RSI is a feature, not a sell signal."
    )

    _ss_run = st.button("▶ Run sell scan", type="primary", key="ss_run")
    if _ss_run:
        _lock_autorefresh()

    # Build symbol list
    def _load_open_positions() -> list[dict]:
        if os.environ.get("BOT_BROKER", "alpaca").strip().lower() == "robinhood":
            path = str(globals().get("robinhood_portfolios_path", "") or os.environ.get("ROBINHOOD_PORTFOLIOS_PATH", "logs/robinhood_portfolios.json"))
            accounts = load_robinhood_portfolio_snapshot(path)
            if not accounts:
                st.warning(
                    "No Robinhood portfolio snapshot is loaded. Refresh/create "
                    f"`{path}` or uncheck Scan my open positions."
                )
                return []
            symbols = tuple(dict.fromkeys(
                str(h.get("symbol", "")).upper()
                for account in accounts
                for h in account.get("positions", [])
                if h.get("symbol")
            ))
            latest_map = _robinhood_latest_price_map(symbols)
            rows: list[dict] = []
            for account in accounts:
                for holding in account.get("positions", []):
                    symbol = str(holding.get("symbol", "")).upper()
                    if not symbol:
                        continue
                    price, market_value, pnl, pnl_pct = _robinhood_price_for_holding(
                        holding,
                        latest_map.get(symbol, {}),
                    )
                    rows.append({
                        "symbol": symbol,
                        "qty": safe_float(holding.get("shares"), 0.0),
                        "market_value": market_value,
                        "unrealized_pl": pnl,
                        "unrealized_plpc": pnl_pct / 100.0 if pnl_pct else 0.0,
                        "avg_cost": safe_float(holding.get("avg_cost"), 0.0),
                        "current_price": price,
                        "account": account.get("label", "Robinhood"),
                    })
            if rows:
                st.caption(f"Loaded {len(rows)} Robinhood snapshot positions for sell scan.")
            return rows
        try:
            from ai_trading.broker.alpaca_broker import AlpacaBroker
            from ai_trading.config import Settings
            _s = Settings.from_env()
            _b = AlpacaBroker(api_key=_s.api_key, api_secret=_s.api_secret, paper=_s.paper_only)
            return _b.all_positions()
        except Exception as exc:
            st.error(
                "Could not load Alpaca positions. Check APCA_API_KEY_ID/APCA_API_SECRET_KEY, "
                f"BOT_PAPER_ONLY, and Alpaca account permissions. Raw error: {exc}"
            )
            return []

    if _ss_run:
        symbols_set: list[str] = []
        held_syms: list[str] = []
        positions_data: list[dict] = []
        if _ss_include_holdings:
            positions_data = _load_open_positions()
            held_syms = [p["symbol"] for p in positions_data]
            symbols_set.extend(held_syms)
        if _ss_include_watch:
            if _use_index_universe:
                with st.spinner("Loading selected index membership for sell scan…"):
                    symbols_set.extend(_sidebar_universe_symbols(load_remote=True))
            elif _use_full_universe:
                st.info("Sidebar universe is Full Market; use the sell scanner's full-market checkbox below.")
            else:
                symbols_set.extend(_sidebar_universe_symbols(load_remote=False))
        if _ss_extra:
            symbols_set.extend(s.strip().upper() for s in _ss_extra.split(",") if s.strip())

        # Full market mode — pull every tradable symbol from Alpaca
        if _ss_full:
            with st.spinner("Loading full universe from Alpaca…"):
                _full = _load_full_universe()
            if not _full:
                st.error("Could not fetch universe from Alpaca. Check credentials.")
            else:
                symbols_set.extend(_full)
                st.info(f"Loaded {len(_full):,} tradable symbols from Alpaca.")

        # dedup, keep order
        seen: set[str] = set()
        sell_syms = [s for s in symbols_set if not (s in seen or seen.add(s))]

        # Apply pre-scan snapshot filter if enabled (essential for full universe)
        if sell_syms and _use_filters and len(sell_syms) > 50:
            _pre_count = len(sell_syms)
            with st.spinner(f"⚡ Pre-filtering {_pre_count:,} symbols via snapshots…"):
                sell_syms = _snapshot_filter(
                    symbols=sell_syms,
                    min_price=_f_min_price,
                    max_price=_f_max_price,
                    min_volume=int(_f_min_vol),
                    min_change_pct=_f_min_chg,
                    max_change_pct=_f_max_chg,
                    exchanges=[],
                )
            # Always keep held positions even if filtered out
            for _h in held_syms:
                if _h not in sell_syms:
                    sell_syms.insert(0, _h)
            st.success(f"✅ Filtered {_pre_count:,} → {len(sell_syms):,} symbols (held positions always kept).")

        if not sell_syms:
            st.warning("Nothing to scan. Enable a source above or add tickers.")
            st.session_state["sell_results"] = []
        else:
            started = _start_background_task(
                "sell_scan_task",
                _run_sell_dashboard_scan,
                tuple(sell_syms),
                market_open,
                int(scanner_fast_ma),
                int(scanner_slow_ma),
                str(scanner_scan_depth).lower(),
                meta={
                    "symbol_count": len(sell_syms),
                    "held": held_syms,
                    "positions": positions_data,
                    "scan_depth": str(scanner_scan_depth).lower(),
                },
            )
            if started:
                st.session_state["sell_page"] = 0
                st.info(f"Started sell scan for **{len(sell_syms):,}** symbols in the background.")
            else:
                st.info("A sell scan is already running; showing the last completed results while it finishes.")

    sell_pending = False
    sell_task = _poll_background_task("sell_scan_task")
    if sell_task:
        meta = sell_task.get("meta", {})
        if sell_task["status"] == "running":
            sell_pending = True
            st.info(
                f"Background sell scan running for **{int(meta.get('symbol_count', 0)):,}** symbols "
                f"({sell_task['elapsed']:.0f}s elapsed). Showing the last completed sell results."
            )
            _schedule_scan_poll()
        elif sell_task["status"] == "done":
            payload = sell_task["result"]
            raw = payload.get("results", [])
            st.session_state["sell_results"] = raw
            st.session_state["sell_held"] = meta.get("held", [])
            st.session_state["sell_positions"] = meta.get("positions", [])
            st.session_state["sell_ts"] = format_local_now("%I:%M:%S %p %Z")
            st.session_state["sell_mode"] = payload.get("mode", "eod")
            st.session_state["sell_depth"] = payload.get("scan_depth", meta.get("scan_depth", "deep"))
            if payload.get("fallback"):
                st.warning("Live sell data returned no results — completed with EOD data instead.")
            if payload.get("cache_hit"):
                st.info("Loaded from dashboard scan cache.")
            st.success(f"Sell scan complete: **{len(raw)}** results.")
            sell_snapshot_payload = {
                "mode": st.session_state["sell_mode"],
                "scan_depth": st.session_state["sell_depth"],
                "result_count": len(raw),
                "held_count": len(st.session_state.get("sell_held", [])),
                "top": [
                    {
                        "symbol": str(item.get("symbol", "") if isinstance(item, dict) else getattr(item, "symbol", "")),
                        "score": safe_float(item.get("score", 0.0) if isinstance(item, dict) else getattr(item, "score", 0.0)),
                        "signal": str(item.get("signal", "") if isinstance(item, dict) else getattr(item, "signal", "")),
                    }
                    for item in raw[:10]
                ],
            }
            _record_dashboard_event(journal_path, "sell_scanner_snapshot", sell_snapshot_payload)
            _notify_dashboard_event(
                "scanner_summary",
                f"Sell scanner complete: {len(raw)} results ({sell_snapshot_payload['scan_depth']} {sell_snapshot_payload['mode']})",
                sell_snapshot_payload,
            )
        elif sell_task["status"] == "error":
            st.error(f"Sell scan failed: {sell_task['error']}")
            st.session_state["sell_results"] = []

    sell_results = st.session_state.get("sell_results", [])
    held_syms = st.session_state.get("sell_held", [])
    positions_data = st.session_state.get("sell_positions", [])
    ss_ts = st.session_state.get("sell_ts", "—")
    ss_mode = st.session_state.get("sell_mode", "—")
    pos_map = {p["symbol"]: p for p in positions_data}

    if not sell_results and not _ss_run and not sell_pending:
        st.info("Pick a source and click **▶ Run sell scan**. The scanner will rank each symbol "
                "from strongest sell to safest hold.")
    elif not sell_results and sell_pending:
        pass
    elif not sell_results:
        st.warning("Scan returned no results.")
    else:
        long_term = ("Long-term" in _ss_horizon)
        # Profit-take scoring — reward overbought/extended/parabolic + big unrealized gains
        def _profit_take_score(r, pos):
            score = 0.0
            drivers: list[str] = []
            # 1) RSI overbought (peak indicator)
            if r.rsi >= 80:
                score += 30; drivers.append(f"RSI {r.rsi:.0f} extreme overbought")
            elif r.rsi >= 70:
                score += 20 + (r.rsi - 70); drivers.append(f"RSI {r.rsi:.0f} overbought")
            elif r.rsi >= 60:
                score += (r.rsi - 60) * 1.0
            # 2) Extension above MA/VWAP (mean-reversion risk)
            gap = float(r.ma_gap_pct or 0.0)
            if gap >= 8:
                score += 20; drivers.append(f"{gap:.1f}% above MA — extended")
            elif gap >= 4:
                score += gap * 2
            elif gap >= 1:
                score += gap
            # 3) Parabolic today
            chg = float(r.change_pct or 0.0)
            if chg >= 5:
                score += 15; drivers.append(f"+{chg:.1f}% spike today")
            elif chg >= 2:
                score += chg * 2
            # 4) Hot 5-day run — fading risk
            mom = float(r.momentum_5d or 0.0)
            if mom >= 8:
                score += 10; drivers.append(f"5d +{mom:.1f}% — mean-reversion risk")
            elif mom >= 3:
                score += mom
            # 5) Volume climax with up move = distribution
            vs = float(r.volume_surge or 0.0)
            if vs >= 2.0 and chg > 0:
                score += 10; drivers.append(f"{vs:.1f}× volume climax")
            elif vs >= 1.5 and chg > 0:
                score += 5
            # 6) Unrealized gain boost (held only) — bigger gain = lock it in
            pnl_pct = None
            if pos and pos.get("unrealized_plpc") is not None:
                pnl_pct = float(pos["unrealized_plpc"]) * 100
                if pnl_pct >= 20:
                    score += 20; drivers.append(f"+{pnl_pct:.1f}% gain — lock in")
                elif pnl_pct >= 10:
                    score += 12; drivers.append(f"+{pnl_pct:.1f}% gain")
                elif pnl_pct >= 5:
                    score += 6
                elif pnl_pct < 0:
                    score -= 10  # don't profit-take into a loss

            # 7) Long-term mode: dampen score for healthy sustained uptrends
            if long_term:
                trend = float(getattr(r, "trend_consistency", 0) or 0)
                # Strong sustained trend with non-extreme RSI = let it run
                if trend >= 65 and r.rsi < 78 and gap < 12:
                    score *= 0.55
                    drivers.append(f"long-term: strong trend ({trend:.0f}%) — let it run")
                elif trend >= 50 and r.rsi < 75:
                    score *= 0.75
                    drivers.append(f"long-term: healthy trend ({trend:.0f}%) — partial discount")
                # Only flag SELL on truly extreme blowoffs in long-term mode
                if r.rsi >= 82 or gap >= 15 or mom >= 20:
                    score += 10  # restore some urgency for actual blowoffs
                    drivers.append("but blowoff conditions — still consider trimming")

            return min(100.0, max(0.0, score)), drivers, pnl_pct

        rows = []
        for r in sell_results:
            pos = pos_map.get(r.symbol)
            pt_score, pt_drivers, _pnl_pct = _profit_take_score(r, pos)
            if pos and _pnl_pct is not None and _pnl_pct <= 0:
                if pt_score >= 50:
                    sell_signal = "RISK EXIT"
                else:
                    sell_signal = "HOLD"
            else:
                if pt_score >= 70:
                    sell_signal = "TAKE PROFIT"
                elif pt_score >= 50:
                    sell_signal = "TRIM"
                else:
                    sell_signal = "HOLD"
            driver_str = " · ".join(pt_drivers[:2]) if pt_drivers else (r.top_driver or "—")
            chg_v = float(r.change_pct or 0.0)
            row = {
                "Symbol": r.symbol,
                "Held?": "✅" if r.symbol in held_syms else "",
                "Profit-Take": round(pt_score, 1),
                "Action": sell_signal,
                "Trade Gate": "PASS" if bool(getattr(r, "quality_pass", True)) else "BLOCK",
                "Quality Flags": _quality_flags_text(getattr(r, "quality_flags", "ok")),
                "Last": round(float(r.close), 2),
                "Price Time": "—",
                "Price Source": latest_source_label({}, r.mode),
                "Data Confidence": "LOW (0)",
                "Δ Today %": round(chg_v, 2),
                "RSI": round(r.rsi, 1),
                "5d Mom %": round(float(r.momentum_5d or 0.0), 2),
                "MA/VWAP Gap %": round(float(r.ma_gap_pct or 0.0), 2),
                "Vol Surge ×": round(float(r.volume_surge or 0.0), 2),
                "Why Sell": driver_str[:80],
                "Plain English": explain_sell(sell_signal, pt_drivers, _pnl_pct, float(r.rsi or 0), float(r.ma_gap_pct or 0), chg_v),
            }
            if pos:
                row["Qty"] = pos["qty"]
                row["P&L $"] = round(pos.get("unrealized_pl", 0.0), 2)
                plpc = pos.get("unrealized_plpc")
                row["P&L %"] = round(float(plpc) * 100, 2) if plpc is not None else None
            rows.append(row)

        # Sort: held first, then by profit-take score desc
        rows.sort(key=lambda r: (r["Held?"] != "✅", -r["Profit-Take"]))

        _filter_note = f" · positions {len(held_syms)}" if held_syms else ""
        st.markdown(
            f"<div style='background:#3a1e1e;padding:8px 12px;border-radius:6px;"
            f"font-size:0.85em;color:#fca5a5'>"
            f"{'⚡ Live' if ss_mode == 'live' else '📅 EOD'} sell scan · "
            f"<b>{len(rows)}</b> results{_filter_note} · as of <b>{ss_ts}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── Search + pagination ─────────────────────────────────────────────
        _SS_PAGE_SIZE = 25
        _ss_search_col, _ss_filter_col = st.columns([2, 2])
        with _ss_search_col:
            _ss_search = st.text_input(
                "🔎 Search symbol", value=st.session_state.get("sell_search", ""),
                placeholder="e.g. NVDA", key="sell_search",
            ).strip().upper()
        with _ss_filter_col:
            _ss_action_filter = st.multiselect(
                "Filter by action", options=["TAKE PROFIT", "TRIM", "RISK EXIT", "HOLD"],
                default=st.session_state.get("sell_action_filter", []),
                key="sell_action_filter",
            )

        filtered_rows = rows
        if _ss_search:
            filtered_rows = [r for r in filtered_rows if _ss_search in r["Symbol"]]
        if _ss_action_filter:
            filtered_rows = [r for r in filtered_rows if r["Action"] in _ss_action_filter]

        total_rows = len(filtered_rows)
        total_pages = max(1, (total_rows + _SS_PAGE_SIZE - 1) // _SS_PAGE_SIZE)
        if _ss_run or st.session_state.get("_sell_last_search") != _ss_search:
            st.session_state["sell_page"] = 0
            st.session_state["_sell_last_search"] = _ss_search
        current_page = max(0, min(st.session_state.get("sell_page", 0), total_pages - 1))
        page_start = current_page * _SS_PAGE_SIZE
        page_rows = filtered_rows[page_start: page_start + _SS_PAGE_SIZE]
        top_sell = [r for r in rows if r["Action"] in ("TAKE PROFIT", "TRIM", "RISK EXIT")][:3]

        visible_sell_symbols = tuple(
            dict.fromkeys(
                [str(row.get("Symbol", "") or "").upper() for row in page_rows]
                + [str(row.get("Symbol", "") or "").upper() for row in top_sell]
            )
        )
        latest_sell_map = load_latest_price_map(visible_sell_symbols)

        def _apply_latest_sell_price(row: dict) -> None:
            symbol = str(row.get("Symbol", "") or "").upper()
            latest = latest_sell_map.get(symbol, {})
            if not latest:
                return
            latest_px = safe_float(latest.get("price"), None)
            if latest_px is not None:
                row["Last"] = round(latest_px, 2)
            row["Price Time"] = format_price_time(str(latest.get("timestamp") or ""))
            row["Price Source"] = latest_source_label(latest, row.get("Price Source", ""))
            conf_label, conf_score, _conf_reason = data_confidence(latest)
            row["Data Confidence"] = f"{conf_label} ({conf_score})"

        for row in page_rows:
            _apply_latest_sell_price(row)
        for row in top_sell:
            _apply_latest_sell_price(row)

        if total_pages > 1:
            nav_l, nav_mid, nav_r = st.columns([1, 3, 1])
            with nav_l:
                if st.button("◀ Prev", disabled=current_page == 0, key="sell_prev"):
                    st.session_state["sell_page"] = current_page - 1
                    st.rerun()
            with nav_mid:
                st.markdown(
                    f"<div style='text-align:center;padding-top:6px;color:#94a3b8'>"
                    f"Page {current_page + 1} of {total_pages} "
                    f"({total_rows} result{'s' if total_rows != 1 else ''})</div>",
                    unsafe_allow_html=True,
                )
            with nav_r:
                if st.button("Next ▶", disabled=current_page >= total_pages - 1, key="sell_next"):
                    st.session_state["sell_page"] = current_page + 1
                    st.rerun()
        else:
            st.caption(f"{total_rows} result{'s' if total_rows != 1 else ''}")

        df_sell = pd.DataFrame(page_rows)

        if df_sell.empty:
            st.info("No results match your search/filter.")
            styled = None
        else:
            _SELL_ACT_EMOJI = {
                "TAKE PROFIT": "💰 TAKE PROFIT",
                "TRIM": "🟡 TRIM",
                "RISK EXIT": "🔻 RISK EXIT",
                "HOLD": "⚪ HOLD",
            }
            df_sell_display = df_sell.copy()
            df_sell_display["Action"] = df_sell_display["Action"].map(
                lambda v: _SELL_ACT_EMOJI.get(str(v), str(v))
            )
            st.dataframe(df_sell_display, use_container_width=True, height=min(80 + len(page_rows) * 35, 700))

        # ── Top sell cards ────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("� Best Profit-Taking Opportunities Right Now")
        if not top_sell:
            st.success("Nothing looks extended — no peaks to lock in right now. Let winners run.")
        else:
            card_cols = st.columns(len(top_sell))
            for col, r in zip(card_cols, top_sell):
                price_time = r.get("Price Time", "—")
                bg = "#14532d" if r["Action"] == "TAKE PROFIT" else ("#713f12" if r["Action"] == "TRIM" else "#7f1d1d")
                badge = "💰 TAKE PROFIT" if r["Action"] == "TAKE PROFIT" else ("🟡 TRIM" if r["Action"] == "TRIM" else "🔻 RISK EXIT")
                held_badge = " · 🪙 held" if r["Held?"] == "✅" else ""
                pnl_html = ""
                if r.get("P&L $") is not None:
                    pnl_val = safe_float(r.get("P&L $"))
                    pnl_pct_val = safe_float(r.get("P&L %"), 0.0)
                    pnl_color = "#86efac" if pnl_val >= 0 else "#fca5a5"
                    pnl_html = (
                        f'<p style="margin:2px 0;color:{pnl_color};font-weight:bold">'
                        f'Unrealized: ${pnl_val:+.2f} ({pnl_pct_val:+.2f}%)</p>'
                    )
                col.markdown(
                    f"""
                    <div style="background:{bg};border-radius:12px;padding:20px;text-align:center">
                      <h2 style="margin:0 0 4px;color:white;font-size:2em">{r['Symbol']}</h2>
                      <p style="font-size:2.4em;margin:0;color:white;font-weight:bold;line-height:1">{r['Profit-Take']}</p>
                      <p style="margin:4px 0 2px;color:#dcfce7;font-size:1.1em">{badge}{held_badge}</p>
                      <hr style="border-color:rgba(255,255,255,0.2);margin:8px 0">
                      <p style="margin:2px 0;color:#e2e8f0">
                        <b>${r['Last']:.2f}</b> &nbsp;
                        <span style="color:{'#fca5a5' if r['Δ Today %'] < 0 else '#86efac'}">{r['Δ Today %']:+.2f}% today</span>
                      </p>
                      <p style="margin:2px 0;color:#94a3b8;font-size:0.72em">{price_time}</p>
                      {pnl_html}
                      <p style="margin:4px 0 2px;color:#fde68a;font-size:0.85em">💡 {r['Why Sell']}</p>
                      <p style="margin:6px 0 2px;color:#e2e8f0;font-size:0.88em">🗣️ {r.get('Plain English','')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ── Optional legacy paper close for held actionable sells ─────────────
        held_sell = [r for r in rows if r["Held?"] == "✅" and r["Action"] in ("TAKE PROFIT", "TRIM", "RISK EXIT")]
        if held_sell and _show_alpaca_paper:
            st.markdown("---")
            st.subheader("Legacy Alpaca Paper Close")
            sym_opts = [f"{r['Symbol']}  ·  qty {r.get('Qty','?')}  ·  {r['Action']} (score {r['Profit-Take']})" + (f"  ·  P&L {r.get('P&L %', 0):+.1f}%" if r.get('P&L %') is not None else "") for r in held_sell]
            _pick = st.selectbox("Select position to close", options=range(len(held_sell)),
                                 format_func=lambda i: sym_opts[i], key="ss_pick")
            _confirm_close = st.checkbox("I understand this submits a real paper SELL order", key="ss_confirm")
            if st.button("Submit paper SELL (close position)", type="secondary", disabled=not _confirm_close):
                sym = held_sell[int(_pick)]["Symbol"]
                try:
                    from ai_trading.broker.alpaca_broker import AlpacaBroker
                    from ai_trading.config import Settings
                    _s = Settings.from_env()
                    _b = AlpacaBroker(api_key=_s.api_key, api_secret=_s.api_secret, paper=True)
                    _b.close_position(sym)
                    st.success(f"✓ Submitted close order for {sym}")
                except Exception as exc:
                    st.error(f"Close failed for {sym}: {exc}")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 4 — POSITION ADVISOR (manual entry → buy/sell/trim/hold recommendation)
# ════════════════════════════════════════════════════════════════════════════════
elif active_key == "advisor":
    _render_page_header(
        "Position Advisor",
        "Score a single position or uploaded portfolio for buy more, hold, trim, sell, or take-profit decisions.",
        ["Robinhood CSV friendly", "Uses current price overlay", "Reasoned sizing"],
    )
    st.caption(
        "Enter what you own and the bot combines your **P&L** with **live market signals** "
        "(RSI, trend, momentum, volume) to recommend an action in plain English."
    )

    _adv_horizon = st.radio(
        "🕒 Holding horizon",
        ["⚡ Short-term (swing/day-trade)", "📈 Long-term investor (let winners run)"],
        horizontal=True, key="adv_horizon",
        help="Long-term mode ignores minor overbought signals when the multi-week trend is strong, "
             "and only flags true blowoffs (RSI≥80, gap≥12%, or 5-day momentum≥20%) as TAKE PROFIT.",
    )
    _adv_horizon_key = "long" if "Long-term" in _adv_horizon else "short"

    _adv_tab_single, _adv_tab_csv = st.tabs(["✍️ Single Position", "📥 Upload Portfolio CSV (Robinhood / etc.)"])

    with _adv_tab_single:
        _adv_l, _adv_r = st.columns([1, 1])
        with _adv_l:
            adv_symbol = st.text_input("Symbol", value="AAPL", key="adv_sym").strip().upper()
            adv_shares = st.number_input("Number of shares", min_value=0.0, value=10.0, step=1.0, key="adv_sh")
            adv_avg_cost = st.number_input("Average cost per share ($)", min_value=0.0, value=150.00, step=0.01, format="%.2f", key="adv_avg")
        with _adv_r:
            adv_mv_mode = st.radio(
                "Use live price or enter manually?",
                ["Fetch live price", "Enter market value manually"],
                key="adv_mv_mode",
            )
            if adv_mv_mode == "Enter market value manually":
                adv_market_value = st.number_input(
                    "Current market value of position ($)", min_value=0.0,
                    value=float(adv_shares * adv_avg_cost), step=0.01, format="%.2f", key="adv_mv")
            else:
                adv_market_value = None

        _adv_go = st.button("▶ Get recommendation", type="primary", key="adv_go")

        if _adv_go:
            if not adv_symbol:
                st.error("Enter a ticker symbol.")
            elif adv_shares <= 0 or adv_avg_cost <= 0:
                st.error("Shares and average cost must be greater than zero.")
            else:
                market_open = is_market_open()
                with st.spinner(f"Analysing {adv_symbol}…"):
                    try:
                        if market_open:
                            scan_out = scan_live([adv_symbol], top_n=1)
                            if not scan_out:
                                scan_out = scan([adv_symbol], fast_ma=int(scanner_fast_ma), slow_ma=int(scanner_slow_ma), top_n=1)
                        else:
                            scan_out = scan([adv_symbol], fast_ma=int(scanner_fast_ma), slow_ma=int(scanner_slow_ma), top_n=1)
                        scan_out = _overlay_latest_prices(scan_out)
                    except Exception as exc:
                        scan_out = []
                        st.error(f"Could not fetch market data: {exc}")

                if not scan_out:
                    st.warning("No market data returned for that symbol.")
                else:
                    latest = load_latest_price_map((adv_symbol,)).get(adv_symbol, {})
                    latest_px = latest.get("price")
                    latest_time = format_price_time(str(latest.get("timestamp") or ""))
                    latest_conf, latest_conf_score, latest_conf_reason = data_confidence(latest)
                    rh_check, rh_check_note = robinhood_quote_confirmation(adv_symbol, latest)
                    if latest_px:
                        scan_out[0].close = float(latest_px)
                    rec = advise_position(adv_symbol, adv_shares, adv_avg_cost, scan_out[0], market_value=adv_market_value, horizon=_adv_horizon_key)
                    pnl_color = "#86efac" if rec["total_return"] >= 0 else "#fca5a5"
                    st.markdown(
                        f"""
                        <div style="background:{rec['color']};border-radius:14px;padding:24px;text-align:center;margin-top:12px">
                          <div style="font-size:0.95em;color:#cbd5e1">Recommendation for <b>{rec['symbol']}</b></div>
                          <div style="font-size:2.6em;color:white;font-weight:bold;margin:6px 0">{rec['emoji']} {rec['action']}</div>
                          <div style="color:#e2e8f0;font-size:1.05em">
                            Current price <b>${rec['price']:,.2f}</b> &nbsp;·&nbsp;
                            Avg cost <b>${adv_avg_cost:,.2f}</b> &nbsp;·&nbsp;
                            Shares <b>{adv_shares:g}</b>
                          </div>
                          <div style="color:#94a3b8;font-size:0.85em;margin-top:3px">
                            Latest price timestamp: <b>{latest_time}</b> &nbsp;·&nbsp;
                            Data confidence <b>{latest_conf} ({latest_conf_score})</b> &nbsp;·&nbsp;
                            Robinhood check <b>{rh_check}</b>
                          </div>
                          <div style="color:#e2e8f0;font-size:1.05em;margin-top:4px">
                            Market value <b>${rec['market_value']:,.2f}</b> &nbsp;·&nbsp;
                            Total return <b style="color:{pnl_color}">${rec['total_return']:+,.2f} ({rec['pnl_pct']:+.2f}%)</b>
                          </div>
                          <div style="margin-top:14px;padding:10px 14px;background:rgba(255,255,255,0.12);border-radius:10px;color:#fef9c3;font-size:1.15em;font-weight:bold">
                            💼 How much: {rec['size_label']}
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("### 🗣️ Why")
                    for line in rec["rationale"]:
                        st.markdown(f"- {line}")
                    st.markdown("### 📊 Signals the bot looked at")
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Bot Score", f"{rec['score']:.0f}/100", rec['signal'])
                    mc2.metric("RSI", f"{rec['rsi']:.0f}", "overbought" if rec['rsi'] >= 70 else ("oversold" if rec['rsi'] <= 30 else "neutral"))
                    mc3.metric("5-day Momentum", f"{rec['momentum']:+.2f}%")
                    mc4.metric("MA/VWAP Gap", f"{rec['gap']:+.2f}%")
                    mc1b, mc2b, mc3b, mc4b = st.columns(4)
                    mc1b.metric("Today %", f"{rec['change_pct']:+.2f}%")
                    mc2b.metric("Volume Surge", f"{rec['vol_surge']:.1f}×")
                    mc3b.metric("Trend Consistency", f"{rec['trend']:.0f}%")
                    mc4b.metric("Est. Upside", f"+{scan_out[0].upside_pct:.1f}%" if scan_out[0].upside_pct > 0 else "—")
                    st.caption(f"💡 Bot driver: **{rec['driver']}**")
                    st.caption(f"Data confidence reason: {latest_conf_reason}")
                    st.caption(f"Robinhood quote confirmation: {rh_check_note}")

    with _adv_tab_csv:
        st.markdown(
            "Export your **positions** from Robinhood (or any broker) as a CSV and drop it here. "
            "The file needs at minimum a **Symbol** column and a **Quantity** column; "
            "**Average Cost** and **Market Value** are used if present."
        )
        st.caption(
            "💡 In Robinhood: Account → Statements & History → tap a recent monthly statement, "
            "or use the web export. You can also paste positions into a CSV with columns: "
            "`Symbol, Quantity, Average Cost, Market Value`."
        )

        _csv_input_mode = st.radio(
            "How do you want to provide the data?",
            ["📂 Upload CSV file", "📋 Paste CSV / table text"],
            horizontal=True, key="adv_csv_mode",
        )

        _holdings: list[dict] = []
        if _csv_input_mode == "📂 Upload CSV file":
            _csv_file = st.file_uploader("Upload portfolio CSV", type=["csv"], key="adv_csv")
            if _csv_file is not None:
                try:
                    _holdings = parse_robinhood_csv(_csv_file)
                except Exception as exc:
                    st.error(f"Could not parse CSV: {exc}")
        else:
            st.caption(
                "Paste rows below. First line should be the header (e.g. `Symbol,Quantity,Average Cost,Market Value`). "
                "Tabs, commas, or multiple spaces all work as separators — perfect for copy/paste from Robinhood, "
                "Google Sheets, or Excel."
            )
            _csv_text = st.text_area(
                "Paste CSV / tab-separated table",
                height=200,
                placeholder="Symbol,Quantity,Average Cost,Market Value\nAAPL,10,150.00,1850.00\nNVDA,5,400.00,3200.00",
                key="adv_csv_text",
            )
            if _csv_text.strip():
                import io, re as _re
                # Normalize: tabs / multi-space → comma; leave commas alone
                lines = []
                for ln in _csv_text.splitlines():
                    if not ln.strip():
                        continue
                    if "," in ln:
                        lines.append(ln)
                    elif "\t" in ln:
                        lines.append(",".join(p.strip() for p in ln.split("\t")))
                    else:
                        lines.append(",".join(p.strip() for p in _re.split(r"\s{2,}", ln.strip())))
                _norm = "\n".join(lines)
                try:
                    _holdings = parse_robinhood_csv(io.StringIO(_norm))
                except Exception as exc:
                    st.error(f"Could not parse pasted data: {exc}")

        if _holdings:
            if True:
                _diag = getattr(parse_robinhood_csv, "last_diag", {}) or {}
                _n_skip = len(_diag.get("skipped", []))
                _matched = _diag.get("matched_cols", {})
                st.success(
                    f"Parsed **{len(_holdings)}** holdings from CSV "
                    f"({_diag.get('total_lines', 0) - 1} data lines, {_n_skip} skipped)."
                )
                _col_msg = (
                    f"Matched columns → symbol: **{_matched.get('symbol')}** · "
                    f"shares: **{_matched.get('shares')}** · "
                    f"avg cost: **{_matched.get('avg_cost') or '— (defaulted to 0)'}** · "
                    f"market value: **{_matched.get('market_value') or '— (will compute from live price)'}**"
                )
                st.caption(_col_msg)
                if _n_skip:
                    with st.expander(f"⚠️ {_n_skip} rows were skipped — click to see why"):
                        st.dataframe(pd.DataFrame(_diag["skipped"]), use_container_width=True)
                with st.expander("Preview parsed holdings"):
                    st.dataframe(pd.DataFrame(_holdings), use_container_width=True)

                if st.button("▶ Get recommendations for entire portfolio", type="primary", key="adv_csv_go"):
                    market_open = is_market_open()
                    syms = [h["symbol"] for h in _holdings]
                    with st.spinner(f"Scoring {len(syms)} holdings…"):
                        try:
                            if market_open:
                                scan_out = scan_live(syms, top_n=len(syms))
                                if not scan_out:
                                    scan_out = scan(syms, fast_ma=int(scanner_fast_ma), slow_ma=int(scanner_slow_ma), top_n=len(syms))
                            else:
                                scan_out = scan(syms, fast_ma=int(scanner_fast_ma), slow_ma=int(scanner_slow_ma), top_n=len(syms))
                            scan_out = _overlay_latest_prices(scan_out)
                        except Exception as exc:
                            scan_out = []
                            st.error(f"Scan failed: {exc}")

                    by_sym = {r.symbol: r for r in scan_out}
                    latest_holdings_map = load_latest_price_map(tuple(h["symbol"] for h in _holdings))
                    rec_rows = []
                    full_recs = []
                    for h in _holdings:
                        r = by_sym.get(h["symbol"])
                        if r is None:
                            rec_rows.append({
                                "Symbol": h["symbol"], "Action": "—", "Why": "No market data",
                                "Shares": h["shares"], "Avg Cost": h["avg_cost"],
                                "Price": None, "Market Value": h.get("market_value", 0),
                                "P&L $": None, "P&L %": None, "Score": None,
                            })
                            continue
                        latest = latest_holdings_map.get(h["symbol"], {})
                        latest_px = latest.get("price")
                        if latest_px:
                            r.close = float(latest_px)
                            h["market_value"] = float(h["shares"]) * float(latest_px)
                        mv = h.get("market_value") or None
                        rec = advise_position(h["symbol"], h["shares"], h["avg_cost"], r, market_value=mv, horizon=_adv_horizon_key)
                        full_recs.append(rec)
                        rec_rows.append({
                            "Symbol": rec["symbol"],
                            "Action": f"{rec['emoji']} {rec['action']}",
                            "Suggested": rec["size_label"],
                            "Score": round(rec["score"], 0),
                            "Shares": h["shares"],
                            "Avg Cost": round(h["avg_cost"], 2),
                            "Price": round(rec["price"], 2),
                            "Price Time": format_price_time(str(latest.get("timestamp") or "")),
                            "Market Value": round(rec["market_value"], 2),
                            "P&L $": round(rec["total_return"], 2),
                            "P&L %": round(rec["pnl_pct"], 2),
                            "Why": rec["rationale"][0] if rec["rationale"] else "",
                        })

                    st.session_state["adv_csv_rows"] = rec_rows
                    st.session_state["adv_csv_full"] = full_recs
                    st.session_state["adv_csv_ts"] = format_local_now("%I:%M:%S %p %Z")

        # Display results OUTSIDE the _holdings guard so they persist across reruns
        # even if the file-uploader widget re-renders.
        rec_rows = st.session_state.get("adv_csv_rows", [])
        full_recs = st.session_state.get("adv_csv_full", [])
        if rec_rows:
            _adv_csv_ts = st.session_state.get("adv_csv_ts", "—")
            st.caption(f"Last portfolio recommendation run: **{_adv_csv_ts}**")
            # Summary metrics
            total_mv = sum(r["Market Value"] or 0 for r in rec_rows)
            total_pnl = sum(r["P&L $"] or 0 for r in rec_rows)
            n_buy = sum(1 for r in rec_rows if "BUY" in r["Action"])
            n_sell = sum(1 for r in rec_rows if "SELL" in r["Action"] or "TAKE PROFIT" in r["Action"])
            n_trim = sum(1 for r in rec_rows if "TRIM" in r["Action"])
            n_hold = sum(1 for r in rec_rows if "HOLD" in r["Action"])
            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Portfolio Value", f"${total_mv:,.0f}")
            s2.metric("Unrealized P&L", f"${total_pnl:+,.0f}")
            s3.metric("🟢 BUY MORE", n_buy)
            s4.metric("✂️ TRIM/SELL", n_trim + n_sell)
            s5.metric("⚪ HOLD", n_hold)

            st.markdown("### Recommendations")
            df_rec = pd.DataFrame(rec_rows)

            _REC_ACT_EMOJI = {
                "BUY MORE": "🟢 BUY MORE",
                "TAKE PROFIT": "💰 TAKE PROFIT",
                "SELL": "🔴 SELL",
                "TRIM": "🟡 TRIM",
                "HOLD": "⚪ HOLD",
            }
            df_rec_display = df_rec.copy()
            df_rec_display["Action"] = df_rec_display["Action"].map(
                lambda v: _REC_ACT_EMOJI.get(str(v), str(v))
            )
            st.dataframe(df_rec_display, use_container_width=True, height=min(80 + len(rec_rows) * 35, 700))

            # Drill-down detail per holding
            st.markdown("### 🔎 Detail per holding")
            sym_pick = st.selectbox(
                "Pick a holding to see full reasoning",
                options=[r["symbol"] for r in full_recs],
                key="adv_csv_pick",
            )
            rec = next((r for r in full_recs if r["symbol"] == sym_pick), None)
            if rec:
                pnl_color = "#86efac" if rec["total_return"] >= 0 else "#fca5a5"
                st.markdown(
                    f"""
                    <div style="background:{rec['color']};border-radius:14px;padding:20px;margin-top:8px">
                      <div style="font-size:1.8em;color:white;font-weight:bold">{rec['emoji']} {rec['action']} — {rec['symbol']}</div>
                      <div style="color:#e2e8f0;margin-top:4px">
                        ${rec['price']:,.2f} · MV ${rec['market_value']:,.2f} ·
                        <b style="color:{pnl_color}">${rec['total_return']:+,.2f} ({rec['pnl_pct']:+.2f}%)</b>
                      </div>
                      <div style="margin-top:10px;padding:8px 12px;background:rgba(255,255,255,0.12);border-radius:8px;color:#fef9c3;font-weight:bold">
                        💼 How much: {rec['size_label']}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                for line in rec["rationale"]:
                    st.markdown(f"- {line}")


# ─────────────────────────────────────────────────────────────────────────────
# Patterns heatmap page
# ─────────────────────────────────────────────────────────────────────────────
elif active_key == "patterns":
    _render_page_header(
        "Patterns",
        "Pattern and ensemble signal heatmap for quick confirmation before acting on scanner results.",
        ["Confirmation layer", "Watchlist driven", "No execution"],
    )
    st.caption("Runs all pattern detectors + the regime-aware ensemble across your watchlist.")

    _rh_pattern_symbols = _robinhood_snapshot_symbols(include_crypto=False)
    _pa_source = st.radio(
        "Symbol source",
        ["Robinhood holdings", "Manual list"],
        horizontal=True,
        key="patterns_symbol_source",
    )
    if _pa_source == "Robinhood holdings":
        if _rh_pattern_symbols:
            _pa_default_symbols = ",".join(_rh_pattern_symbols)
        else:
            _pa_default_symbols = _env_symbols or _DEFAULT_WATCHLIST
            st.warning("No Robinhood holdings snapshot symbols are available yet, so the Patterns page is falling back to the manual/default list.")
    else:
        _pa_default_symbols = _env_symbols or _DEFAULT_WATCHLIST
    st.caption("Robinhood watchlist import is not implemented yet; Robinhood holdings are the available broker-linked source today.")

    _pa_symbols_raw = st.text_area(
        "Symbols (comma-separated)",
        value=_pa_default_symbols,
        height=80,
        key="patterns_symbols_raw",
    )
    _pa_all_symbols = [s.strip().upper() for s in _pa_symbols_raw.split(",") if s.strip()]
    _pa_limit_col1, _pa_limit_col2 = st.columns([1, 1])
    with _pa_limit_col1:
        _pa_symbol_limit = st.number_input("Max symbols", 10, 500, 120, 10)
    with _pa_limit_col2:
        _pa_period = st.selectbox("History window", ["6mo", "1y", "2y"], index=1)
    _pa_symbols = _pa_all_symbols[: int(_pa_symbol_limit)]
    if len(_pa_all_symbols) > len(_pa_symbols):
        st.warning(f"Pattern scan is currently using the first {len(_pa_symbols)} of {len(_pa_all_symbols)} symbols.")
    _pa_col1, _pa_col2 = st.columns([2, 1])
    with _pa_col1:
        _pa_run = st.button("▶ Run pattern scan", type="primary")
    with _pa_col2:
        if st.button("🗑️ Clear cache & re-run", help="Clears the 5-min cache so the next scan fetches fresh data"):
            st.cache_data.clear()
            st.session_state.pop("patterns_results", None)
            st.session_state.pop("patterns_ts", None)
            st.rerun()

    @st.cache_data(ttl=300, show_spinner="Fetching bars & scoring patterns…")
    def _pattern_scan(symbols: tuple[str, ...], period: str) -> pd.DataFrame:
        import yfinance as yf
        from ai_trading.strategy.patterns import detect_all_patterns, PATTERN_DETECTORS
        from ai_trading.strategy.ensemble import compute_ensemble_signal

        rows: list[dict] = []
        for sym in symbols:
            try:
                df = yf.download(sym, period=period, progress=False, auto_adjust=False)
                if df.empty:
                    continue
                df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
                hits = detect_all_patterns(df)
                es = compute_ensemble_signal(df)
                row = {
                    "symbol": sym,
                    "regime": es.regime.value,
                    "ensemble": es.signal,
                    "strength": round(es.strength, 3),
                }
                for h in hits:
                    row[h.name] = round(h.signal, 3)
                rows.append(row)
            except Exception as exc:
                rows.append({"symbol": sym, "regime": "error", "ensemble": "ERR", "strength": 0.0, "_err": str(exc)})
        return pd.DataFrame(rows)

    if _pa_run and _pa_symbols:
        _lock_autorefresh()
        st.session_state["patterns_results"] = _pattern_scan(tuple(_pa_symbols), _pa_period)
        st.session_state["patterns_ts"] = format_local_now("%I:%M:%S %p %Z")
        st.session_state["patterns_period"] = _pa_period
        st.session_state["patterns_symbols"] = len(_pa_symbols)

    df_pat = st.session_state.get("patterns_results")
    if df_pat is None:
        st.info("Click **▶ Run pattern scan** to compute pattern signals across your watchlist.")
    elif df_pat.empty:
        st.warning("No data returned.")
    else:
        df_pat = df_pat.sort_values("strength", ascending=False)
        _pa_ts = st.session_state.get("patterns_ts", "—")
        _pa_cached_period = st.session_state.get("patterns_period", _pa_period)
        _pa_symbol_count = st.session_state.get("patterns_symbols", len(df_pat))
        st.caption(f"Last scan: **{_pa_ts}** · {_pa_symbol_count} symbols · period {_pa_cached_period} · cached for 5 min")

        c1, c2, c3 = st.columns(3)
        c1.metric("Symbols scanned", len(df_pat))
        c2.metric("Bullish (BUY)", int((df_pat["ensemble"] == "BUY").sum()))
        c3.metric("Bearish (SELL)", int((df_pat["ensemble"] == "SELL").sum()))

        pattern_cols = [c for c in df_pat.columns if c not in ("symbol", "regime", "ensemble", "strength", "_err")]
        display = df_pat[["symbol", "ensemble", "regime", "strength", *pattern_cols]].copy()
        styled = display.style.background_gradient(
            cmap="RdYlGn", subset=["strength", *pattern_cols], vmin=-1, vmax=1
        )
        st.dataframe(styled, use_container_width=True, height=min(80 + len(display) * 32, 700))

        st.markdown("### Top bullish setups")
        st.dataframe(df_pat[df_pat["ensemble"] == "BUY"].head(10), use_container_width=True)
        st.markdown("### Top bearish setups")
        st.dataframe(df_pat[df_pat["ensemble"] == "SELL"].head(10), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Options Lab page
# ─────────────────────────────────────────────────────────────────────────────
elif active_key == "options":
    _render_page_header(
        "Options",
        "Options strategy research and diagnostics. Keep Robinhood order placement separate from this analysis view.",
        ["Research only", "Verify in broker", "No auto-execution"],
    )
    st.caption("Scan option chains for the best risk-adjusted strategy candidates. "
               "Powered by Alpaca options API (fallback: yfinance).")

    _rh_option_symbols = _robinhood_snapshot_symbols(include_crypto=False)
    _ol_source_mode = st.radio(
        "Underlying source",
        ["Robinhood holdings", "Manual list"],
        horizontal=True,
        key="options_symbol_source",
    )
    if _ol_source_mode == "Robinhood holdings":
        if _rh_option_symbols:
            _ol_default_symbols = ",".join(_rh_option_symbols)
        else:
            _ol_default_symbols = "SPY,QQQ,AAPL,NVDA"
            st.warning("No Robinhood holdings snapshot symbols are available yet, so Options is falling back to the manual/default list.")
    else:
        _ol_default_symbols = "SPY,QQQ,AAPL,NVDA"
    st.caption("Robinhood watchlist import is not implemented yet; Robinhood holdings are the available broker-linked source today.")

    _ol_col1, _ol_col2, _ol_col3, _ol_col4 = st.columns([3, 2, 2, 2])
    with _ol_col1:
        _ol_underlying_raw = st.text_input("Underlyings (comma-separated)", value=_ol_default_symbols, key="options_underlyings_raw")
    with _ol_col2:
        _ol_strategies = st.multiselect(
            "Strategies",
            ["long_call", "long_put", "csp", "covered_call",
             "bull_call", "bear_put", "bull_put", "bear_call",
             "iron_condor", "short_strangle"],
            default=["long_call", "csp", "bull_call"],
        )
    with _ol_col3:
        _ol_min_dte = st.number_input("Min DTE", 0, 365, 21)
        _ol_max_dte = st.number_input("Max DTE", 1, 365, 45)
    with _ol_col4:
        _ol_delta = st.slider("Target Δ (abs)", 0.05, 0.50, 0.30, 0.05)
        _ol_width = st.number_input("Spread width ($)", 1.0, 50.0, 5.0, 1.0)

    _ol_col5, _ol_col6, _ol_col7, _ol_col8 = st.columns([2, 2, 2, 3])
    with _ol_col5:
        _ol_source = st.selectbox("Data source", ["auto", "alpaca", "yfinance"])
    with _ol_col6:
        _ol_top = st.number_input("Top N results", 5, 100, 15)
    with _ol_col7:
        _ol_per_strategy_top = st.number_input("Per-strategy cap", 1, 20, 2)
    with _ol_col8:
        _ol_symbol_limit = st.number_input("Max underlyings", 5, 200, 40)
        _ol_run = st.button("▶ Run options scan", type="primary")

    @st.cache_data(ttl=120, show_spinner="Fetching chains & scoring strategies…")
    def _options_scan_cached(symbols: tuple[str, ...], strategies: tuple[str, ...],
                              min_dte: int, max_dte: int, delta: float, width: float,
                              source: str, top_n: int, per_strategy_top: int) -> list[dict]:
        from ai_trading.options.scanner import scan_options
        results = scan_options(
            underlyings=list(symbols),
            universe=None,
            use_equity_scanner=False,
            strategies=list(strategies),
            top_n=top_n,
            per_strategy_top=per_strategy_top,
            min_dte=min_dte,
            max_dte=max_dte,
            target_delta=delta,
            spread_width=width,
            source=source,
        )
        return [c.to_dict() for c in results]

    if _ol_run:
        _lock_autorefresh()
        if not _ol_strategies:
            st.warning("Pick at least one strategy.")
        else:
            all_symbols = tuple(s.strip().upper() for s in _ol_underlying_raw.split(",") if s.strip())
            symbols = all_symbols[: int(_ol_symbol_limit)]
            if len(all_symbols) > len(symbols):
                st.warning(f"Options scan is currently using the first {len(symbols)} of {len(all_symbols)} underlyings.")
            try:
                results = _options_scan_cached(
                    symbols, tuple(_ol_strategies),
                    int(_ol_min_dte), int(_ol_max_dte),
                    float(_ol_delta), float(_ol_width),
                    _ol_source, int(_ol_top), int(_ol_per_strategy_top),
                )
                st.session_state["options_results"] = results
                st.session_state["options_ts"] = format_local_now("%I:%M:%S %p %Z")
            except Exception as exc:
                st.error(f"Scan failed: {exc}")
                st.session_state.setdefault("options_results", [])

    # Always render from session_state so auto-refresh / Refresh Now keeps results visible.
    results = st.session_state.get("options_results", [])
    _ol_ts = st.session_state.get("options_ts", "—")

    if not results and not _ol_run:
        st.info("Configure parameters above and click **▶ Run options scan**.")
    elif not results:
        st.info("No candidates found. Try wider DTE / different strategies / more symbols.")
    else:
        st.caption(f"Last scan: **{_ol_ts}** · {len(results)} candidates · cached for 2 min")
        df_opt = pd.DataFrame([{
            "Symbol": r["underlying"],
            "Spot": r["underlying_price"],
            "Strategy": r["strategy"],
            "DTE": r["dte"],
            "Quote Time": format_price_time(str(r.get("quote_timestamp") or "")),
            "Quote Source": (
                f"{str(r.get('quote_source') or r.get('source') or '').upper()}"
                f"{' STALE' if r.get('quote_stale') else ''}"
            ).strip(),
            "Debit/Credit": r["debit_credit"],
            "Max Profit": r["max_profit"] if r["max_profit"] != float("inf") else float("nan"),
            "Max Loss": r["max_loss"] if r["max_loss"] != float("inf") else float("nan"),
            "POP": r["pop"],
            "R:R": min(r["risk_reward"], 99),
            "BP Req": r["bp_requirement"],
            "IV avg": r["iv_avg"],
            "Δ total": r["delta_total"],
            "Score": r["score"],
            "Note": r["notes"],
        } for r in results])

        st.metric("Candidates returned", len(df_opt))
        stale_count = int(sum(1 for r in results if r.get("quote_stale")))
        if stale_count:
            st.warning(
                f"{stale_count} option candidate{'s have' if stale_count != 1 else ' has'} stale or unknown quote timestamps. "
                "Verify live bid/ask in Robinhood and use limit orders."
            )
        styled = df_opt.style.format({
            "Spot": "${:.2f}",
            "Debit/Credit": "${:+,.0f}",
            "Max Profit": "${:,.0f}",
            "Max Loss": "${:,.0f}",
            "BP Req": "${:,.0f}",
            "POP": "{:.0%}",
            "R:R": "{:.2f}",
            "IV avg": "{:.1%}",
            "Δ total": "{:+.2f}",
            "Score": "{:.1f}",
        }).background_gradient(cmap="RdYlGn", subset=["POP", "Score"], vmin=0, vmax=100)
        st.dataframe(styled, use_container_width=True, height=min(80 + len(df_opt) * 35, 700))

        st.markdown("### Selected candidate details")
        _sel_idx = st.number_input("Inspect row #", 1, len(results), 1) - 1
        _sel = results[int(_sel_idx)]
        _sel_quote_time = format_price_time(str(_sel.get("quote_timestamp") or ""))
        _sel_quote_source = str(_sel.get("quote_source") or "").upper() or "UNKNOWN"
        if _sel.get("quote_stale"):
            st.warning(
                f"Selected option quote is stale or missing a quote timestamp "
                f"(source {_sel_quote_source}, quote time {_sel_quote_time}). Re-check bid/ask before ordering."
            )
        else:
            st.success(f"Selected option quote time: {_sel_quote_time} · source {_sel_quote_source}")
        st.json(_sel)

        with st.expander("📝 Place this as a PAPER order"):
            st.info("Submits via Alpaca paper account. Requires options trading enabled there.")
            _qty = st.number_input("Qty (contracts)", 1, 100, 1, key="opt_qty")
            _confirm = st.checkbox("I understand this places a real paper order", key="opt_confirm")
            if st.button("Submit paper order", type="secondary", disabled=not _confirm):
                try:
                    from ai_trading.options.broker import OptionsBroker
                    from ai_trading.options.strategies import StrategyCandidate, StrategyLeg
                    legs = [StrategyLeg(**leg) for leg in _sel["legs"]]
                    cand = StrategyCandidate(
                        strategy=_sel["strategy"],
                        underlying=_sel["underlying"],
                        underlying_price=_sel["underlying_price"],
                        legs=legs,
                        debit_credit=_sel["debit_credit"],
                        max_profit=_sel["max_profit"],
                        max_loss=_sel["max_loss"],
                        breakevens=_sel["breakevens"],
                        pop=_sel["pop"],
                        risk_reward=_sel["risk_reward"],
                        bp_requirement=_sel["bp_requirement"],
                        dte=_sel["dte"],
                        iv_avg=_sel["iv_avg"],
                        delta_total=_sel["delta_total"],
                        notes=_sel["notes"],
                        score=_sel["score"],
                    )
                    broker = OptionsBroker(paper=True)
                    order = broker.place_strategy(cand, qty=int(_qty), order_type="limit")
                    st.success(f"✓ Submitted order id={getattr(order, 'id', '?')}")
                except Exception as exc:
                    st.error(f"Order failed: {exc}")

    st.markdown("---")
    st.subheader("📋 Open Option Positions")
    try:
        from ai_trading.options.broker import OptionsBroker
        _ob = OptionsBroker(paper=True)
        _opos = _ob.all_option_positions()
        if _opos:
            st.dataframe(pd.DataFrame(_opos), use_container_width=True)
        else:
            st.caption("No open option positions.")
    except Exception as exc:
        st.caption(f"(could not load option positions: {exc})")

# ─────────────────────────────────────────────────────────────────────────────
# Research page
# ─────────────────────────────────────────────────────────────────────────────
elif active_key == "research":
    _render_page_header(
        "Research",
        "Generate institutional-style equity research prompts for deep dives, earnings reviews, valuation work, and balanced bull-vs-bear debates.",
        ["Prompt builder", "Robinhood-linked symbols", "Manual analysis workflow"],
    )
    st.caption(
        "This page builds structured prompts for external AI or manual analysis. "
        "It does not fetch filings or generate the memo by itself."
    )
    _recent_packets = get_recent_payloads(records, "research_packet", limit=20)
    if _recent_packets:
        _packet_labels = [
            f"{str(p.get('subject', '—')).upper()} · {str(p.get('mode', 'memo')).upper()} · {str(p.get('source', 'manual'))}"
            for p in _recent_packets
        ]
        _packet_pick = st.selectbox(
            "Recent saved packets",
            options=["Start fresh"] + _packet_labels,
            index=0,
            key="research_recent_packet_pick",
        )
        if _packet_pick != "Start fresh":
            _picked = _recent_packets[_packet_labels.index(_packet_pick)]
            st.caption(
                f"Loaded saved packet context from **{str(_picked.get('source', 'manual'))}** for "
                f"**{str(_picked.get('subject', '')).upper()}**."
            )
            st.session_state["research_mode"] = str(_picked.get("mode", "memo"))
            st.session_state["research_goals"] = str(_picked.get("goals", "long-term capital appreciation"))
            st.session_state["research_risk"] = str(_picked.get("risk_tolerance", "moderate"))
            st.session_state["research_horizon"] = str(_picked.get("time_horizon", "5+ years"))
            st.session_state["research_as_of_date"] = str(
                _picked.get("as_of_date", datetime.now(app_timezone()).strftime("%B %d, %Y"))
            )

    _rh_research_symbols = _robinhood_snapshot_symbols(include_crypto=True)
    _research_source = st.radio(
        "Coverage source",
        ["Robinhood holdings", "Manual entry"],
        horizontal=True,
        key="research_symbol_source",
    )
    _manual_default = (_env_symbols.split(",")[0].strip().upper() if _env_symbols else "") or "NVDA"
    if _research_source == "Robinhood holdings" and _rh_research_symbols:
        _research_default_subject = _rh_research_symbols[0]
    else:
        _research_default_subject = _manual_default
        if _research_source == "Robinhood holdings":
            st.warning("No Robinhood holdings snapshot symbols are available yet, so Research is falling back to manual entry.")

    _res_col1, _res_col2, _res_col3 = st.columns([2.2, 1.3, 1.5])
    with _res_col1:
        if _research_source == "Robinhood holdings" and _rh_research_symbols:
            _research_subject = st.selectbox(
                "Primary ticker / company",
                options=_rh_research_symbols,
                index=0,
                key="research_subject_select",
            )
        else:
            _research_subject = st.text_input(
                "Primary ticker / company",
                value=_research_default_subject,
                key="research_subject_manual",
            ).strip().upper()
    with _res_col2:
        _research_mode = st.selectbox(
            "Prompt type",
            ["memo", "earnings", "valuation", "debate"],
            index=0,
            key="research_mode",
        )
    with _res_col3:
        _research_as_of = st.text_input(
            "As-of date",
            value=datetime.now(app_timezone()).strftime("%B %d, %Y"),
            key="research_as_of_date",
        ).strip()

    _res_col4, _res_col5, _res_col6 = st.columns([1.6, 1.2, 1.2])
    with _res_col4:
        _research_goals = st.text_input(
            "Investment goals",
            value="long-term capital appreciation",
            key="research_goals",
        ).strip()
    with _res_col5:
        _research_risk = st.selectbox(
            "Risk tolerance",
            ["low", "moderate", "moderate-high", "high"],
            index=1,
            key="research_risk",
        )
    with _res_col6:
        _research_horizon = st.selectbox(
            "Time horizon",
            ["1 year", "3 years", "5+ years", "10+ years"],
            index=2,
            key="research_horizon",
        )

    _comparison_candidates = [s for s in _rh_research_symbols if s != _research_subject]
    _comparison_mode = st.checkbox(
        "Add comparison target",
        value=False,
        key="research_compare_on",
        help="Used by the full investment memo prompt.",
    )
    _comparison_target = ""
    if _comparison_mode:
        if _research_source == "Robinhood holdings" and _comparison_candidates:
            _comparison_target = st.selectbox(
                "Comparison ticker",
                options=[""] + _comparison_candidates,
                index=0,
                key="research_compare_select",
            )
        else:
            _comparison_target = st.text_input(
                "Comparison ticker",
                value="",
                key="research_compare_manual",
            ).strip().upper()

    _research_spec = ResearchPromptSpec(
        subject=_research_subject or _research_default_subject,
        goals=_research_goals or "long-term capital appreciation",
        risk_tolerance=_research_risk,
        time_horizon=_research_horizon,
        as_of_date=_research_as_of or datetime.now(app_timezone()).strftime("%B %d, %Y"),
        comparison_target=_comparison_target,
    )

    if _research_mode == "earnings":
        _research_prompt = build_earnings_prompt(_research_spec)
    elif _research_mode == "valuation":
        _research_prompt = build_valuation_prompt(_research_spec)
    elif _research_mode == "debate":
        _research_prompt = build_debate_prompt(_research_spec)
    else:
        _research_prompt = build_investment_memo_prompt(_research_spec)

    _research_meta_1, _research_meta_2, _research_meta_3, _research_meta_4 = st.columns(4)
    _research_meta_1.metric("Prompt Type", _research_mode.upper())
    _research_meta_2.metric("Subject", _research_spec.subject or "—")
    _research_meta_3.metric("Comparison", _research_spec.comparison_target or "—")
    _research_meta_4.metric("Length", f"{len(_research_prompt.split()):,} words")

    st.text_area(
        "Generated prompt",
        value=_research_prompt,
        height=680,
        key="research_prompt_output",
    )
    _save_col, _meta_col = st.columns([1.2, 3.0])
    with _save_col:
        if st.button("Save Packet To Journal", key="research_save_packet_page", type="primary", use_container_width=True):
            _queue_research_packet_event(
                journal_path=journal_path,
                spec=_research_spec,
                mode=_research_mode,
                source="research_page",
                context={"comparison": _research_spec.comparison_target},
            )
            st.success(f"Saved research packet for {_research_spec.subject}.")
    with _meta_col:
        st.caption("Saved packets appear here and can also be created from scanner detail or Robinhood holding detail views.")
    st.caption(
        "Use this with your preferred research workflow. The memo prompt is the full institutional template; "
        "earnings, valuation, and debate are narrower follow-up lenses."
    )

# ── Auto-refresh (non-blocking) ──
# Use a client-side timer so the Python script does not sleep and block render.
if _autorefresh_on and not _skip_refresh:
        _refresh_ms = max(1000, int(_autorefresh_sec) * 1000)
        _refresh_script = f"""
        <script>
        (function() {{
            const key = 'aiTradingAutoRefreshTimer';
            const existing = window[key];
            if (existing) {{ clearTimeout(existing); }}
            window[key] = setTimeout(function() {{ window.location.reload(); }}, {_refresh_ms});
        }})();
        </script>
        """
        st.markdown(_refresh_script, unsafe_allow_html=True)
