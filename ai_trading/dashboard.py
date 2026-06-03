"""Streamlit dashboard — Portfolio monitor + Live market scanner.

Run with:
    streamlit run ai_trading/dashboard.py
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Ensure project root is on sys.path so ai_trading is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd
import streamlit as st

# ── Load Streamlit Cloud secrets into environment variables ────────────────────
# This allows the app to work on Streamlit Cloud where secrets are configured
# via the dashboard UI and accessed through st.secrets.
_SECRET_KEYS = [
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
    "BOT_PAPER_ONLY",
    "BOT_WEBHOOK_URL",
]
for _key in _SECRET_KEYS:
    if _key not in os.environ:
        try:
            os.environ[_key] = st.secrets[_key]
        except (KeyError, FileNotFoundError):
            pass

from ai_trading.scanner import is_market_open, scan, scan_live

st.set_page_config(
    page_title="AI Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
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
    ["📋 Curated (~130)", "🌐 Full Market (~12K)"],
    index=0,
)
_use_full_universe = "Full" in _universe_mode

if _env_symbols:
    default_symbols = _env_symbols
    _use_full_universe = False
elif _use_full_universe:
    default_symbols = ""
else:
    default_symbols = _DEFAULT_WATCHLIST

if not _use_full_universe:
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

# ── Pre-scan filters (shown when Full Market OR curated with >50 symbols) ────
with st.sidebar.expander("⚡ Pre-scan Filters", expanded=_use_full_universe):
    _f_min_price = st.number_input("Min price ($)", min_value=0.0, value=5.0, step=1.0)
    _f_max_price = st.number_input("Max price ($)", min_value=0.0, value=5000.0, step=50.0)
    _f_min_vol   = st.number_input("Min daily volume", min_value=0, value=500_000, step=100_000,
                                    format="%d")
    _f_min_chg   = st.number_input("Min daily change %", value=-20.0, step=0.5)
    _f_max_chg   = st.number_input("Max daily change %", value=20.0, step=0.5)
    _use_filters = st.checkbox("Apply filters before scanning", value=_use_full_universe)

scanner_top_n    = st.sidebar.slider("Top N results", 5, 200, 25)
scanner_fast_ma  = st.sidebar.number_input("Fast MA", 3, 20, 5)
scanner_slow_ma  = st.sidebar.number_input("Slow MA", 10, 60, 20)
run_scan         = st.sidebar.button("▶ Run Scanner Now")

st.sidebar.markdown("---")
# Navigation — persisted in URL query params so meta-refresh reloads keep the right page
_qp_page = st.query_params.get("page", "portfolio")
_PAGE_LABELS = ["📊 Portfolio Monitor", "🔍 Live Scanner", "💸 Sell Scanner", "🧮 Position Advisor", "🧩 Patterns Heatmap", "🎯 Options Lab"]
_PAGE_KEYS = ["portfolio", "scanner", "sell", "advisor", "patterns", "options"]
_page_idx = _PAGE_KEYS.index(_qp_page) if _qp_page in _PAGE_KEYS else 0
active_page = st.sidebar.radio(
    "Navigate",
    _PAGE_LABELS,
    index=_page_idx,
    key="active_page",
)
# Write selection back to query params immediately so meta-refresh picks it up
_idx = _PAGE_LABELS.index(active_page) if active_page in _PAGE_LABELS else 0
st.query_params["page"] = _PAGE_KEYS[_idx]
st.sidebar.markdown("---")
_autorefresh_on = st.sidebar.checkbox("🔁 Auto-refresh", value=True,
                                       help="Re-runs the page so live prices stay current. "
                                            "Session results (scans, options) are preserved.")
_autorefresh_sec = st.sidebar.slider("Refresh every (sec)", 10, 300, 30, step=5,
                                      disabled=not _autorefresh_on)
st.sidebar.caption(f"Page rendered: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
if st.sidebar.button("🔄 Refresh Now"):
    # Clear cached data only — keep session_state (scan results etc.)
    st.cache_data.clear()
    st.rerun()

# Auto-refresh skip rules: pause while a scan is in-flight so we don't kill it.
_skip_refresh = (active_page in ("🔍 Live Scanner", "💸 Sell Scanner", "🧮 Position Advisor", "🎯 Options Lab")) or run_scan
if _autorefresh_on and active_page in ("🔍 Live Scanner", "💸 Sell Scanner", "🧮 Position Advisor", "🎯 Options Lab"):
    st.sidebar.caption("⏸  Auto-refresh paused on this page (results stay until you re-scan)")


# ── Plain-English explainers ──────────────────────────────────────────────────

def explain_buy(r) -> str:
    """One-sentence layperson explanation for a buy-scanner result."""
    rsi = float(r.rsi or 50)
    chg = float(r.change_pct or 0)
    mom = float(r.momentum_5d or 0)
    gap = float(r.ma_gap_pct or 0)
    vs  = float(r.volume_surge or 1)
    sig = r.signal

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


# ── Load portfolio data ───────────────────────────────────────────────────────
records = load_journal(journal_path)

# ── Page title + status ───────────────────────────────────────────────────────
st.title("📈 AI Trading Dashboard")
c1, c2 = st.columns([3, 1])
with c1:
    st.caption(f"Journal: `{journal_path}` · {len(records)} events")
with c2:
    last_ts = records[-1].get("ts", "") if records else ""
    stale = not last_ts or (datetime.now(timezone.utc) - parse_ts(last_ts)).total_seconds() > 3600
    st.metric("Bot last active", f"{'🔴' if stale else '🟢'} {relative_time(last_ts)}")

# ── Main content area (driven by sidebar radio, preserves selection on rerun) ──


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PORTFOLIO MONITOR
# ════════════════════════════════════════════════════════════════════════════════
if active_page == "📊 Portfolio Monitor":
    account = get_latest(records, "account_state")
    orders_all = get_all(records, "order")
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    orders_today = [r for r in orders_all if parse_ts(r.get("ts", "")) >= today_start]

    # Metrics row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    if account:
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

    # ── Live Positions Table ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📂 Open Positions")
    positions = load_positions()
    if positions:
        df_pos = pd.DataFrame(positions)
        # Format for display
        df_display = pd.DataFrame({
            "Symbol":        df_pos["Symbol"],
            "Shares":        df_pos["Shares"],
            "Avg Cost":      df_pos["Avg Cost"].map(lambda v: f"${v:,.2f}"),
            "Price":         df_pos["Current Price"].map(lambda v: f"${v:,.2f}"),
            "Market Value":  df_pos["Market Value"].map(lambda v: f"${v:,.2f}"),
            "Today's P&L":   df_pos.apply(lambda r: "${:+,.2f}  ({:+.2f}%)".format(r["Today's P&L"], r["Today's P&L %"]), axis=1),
            "Total P&L":     df_pos.apply(lambda r: "${:+,.2f}  ({:+.2f}%)".format(r["Total P&L"], r["Total P&L %"]), axis=1),
        })

        def color_pnl(v):
            try:
                num = float(str(v).split("$")[1].split(" ")[0].replace(",", ""))
                if num > 0:  return "color:#86efac;font-weight:bold"
                if num < 0:  return "color:#fca5a5;font-weight:bold"
            except Exception:
                pass
            return ""

        styled_pos = (
            df_display.style
            .map(color_pnl, subset=["Today's P&L", "Total P&L"])
        )
        st.dataframe(styled_pos, width="stretch",
                     height=min(80 + len(df_display) * 35, 500))

        # Summary totals row
        total_mv   = df_pos["Market Value"].sum()
        total_cost = df_pos["Cost Basis"].sum()
        total_pl   = df_pos["Total P&L"].sum()
        total_today = df_pos["Today's P&L"].sum()
        total_pl_pct = (total_pl / total_cost * 100) if total_cost else 0.0
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Total Market Value", f"${total_mv:,.2f}")
        t2.metric("Total Cost Basis",   f"${total_cost:,.2f}")
        t3.metric("Total P&L",  f"${total_pl:+,.2f}", f"{total_pl_pct:+.2f}%")
        t4.metric("Today's P&L", f"${total_today:+,.2f}")
    elif load_positions() == [] and os.environ.get("APCA_API_KEY_ID"):
        st.info("No open positions.")
    else:
        st.warning("Alpaca credentials not found — set APCA_API_KEY_ID / APCA_API_SECRET_KEY.")

    st.markdown("---")

    col_sig, col_risk = st.columns(2)

    with col_sig:
        st.subheader("📡 Latest Signals")
        signal_records = get_all(records, "signal")
        if signal_records:
            seen: set = set()
            rows = []
            for r in reversed(signal_records):
                p = r.get("payload", {})
                sym = p.get("symbol", "?")
                if sym in seen:
                    continue
                seen.add(sym)
                rows.append({
                    "Symbol": sym,
                    "Signal": p.get("signal", "HOLD"),
                    "Close": f"${safe_float(p.get('close')):.2f}",
                    "Fast MA": f"${safe_float(p.get('fast_ma')):.2f}",
                    "Slow MA": f"${safe_float(p.get('slow_ma')):.2f}",
                    "Pos": int(p.get("position_qty", 0)),
                    "When": relative_time(r.get("ts", "")),
                })
            def color_sig(v):
                if v == "BUY":  return "background-color:#14532d;color:#86efac"
                if v == "SELL": return "background-color:#450a0a;color:#fca5a5"
                return "color:#94a3b8"
            st.dataframe(pd.DataFrame(rows).style.map(color_sig, subset=["Signal"]), width="stretch")
        else:
            st.info("No signals yet.")

    with col_risk:
        st.subheader("🚫 Risk Rejects & Errors")
        rejects = get_all(records, "risk_reject", "error", "order_error", "gap_open_protect", "correlation_reject")
        if rejects:
            rrows = [{
                "Type": r.get("event_type", ""),
                "Symbol": r.get("payload", {}).get("symbol", "—"),
                "Reason": (r.get("payload", {}).get("reason") or r.get("payload", {}).get("error", ""))[:60],
                "When": relative_time(r.get("ts", "")),
            } for r in reversed(rejects[-30:])]
            def color_type(v):
                if v == "error": return "color:#f87171"
                if v == "risk_reject": return "color:#fbbf24"
                return ""
            st.dataframe(pd.DataFrame(rrows).style.map(color_type, subset=["Type"]), width="stretch")
        else:
            st.success("✅ No rejects or errors.")

    st.markdown("---")
    col_stats, col_trades = st.columns([1, 2])

    with col_stats:
        st.subheader("📋 Trade Stats")
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
            st.metric("Total Closed P&L", f"${sum(pnl_list):+.2f}")
        else:
            st.info("Need closed trades.")

    with col_trades:
        st.subheader("📜 Trade Log")
        if orders_all:
            trows = [{
                "Time": r.get("ts", "")[:19],
                "Symbol": r.get("payload", {}).get("symbol", "?"),
                "Action": r.get("payload", {}).get("action", "?"),
                "Qty": r.get("payload", {}).get("qty", 0),
                "Mode": r.get("payload", {}).get("mode", "PAPER"),
                "When": relative_time(r.get("ts", "")),
            } for r in reversed(orders_all[-50:])]
            def color_action(v):
                if v == "BUY":  return "background-color:#14532d;color:#86efac"
                if v == "SELL": return "background-color:#450a0a;color:#fca5a5"
                return ""
            st.dataframe(pd.DataFrame(trows).style.map(color_action, subset=["Action"]), width="stretch")
        else:
            st.info("No trades yet.")

    st.markdown("---")
    st.subheader("📉 Equity History")
    acct_records = get_all(records, "account_state")
    if len(acct_records) >= 2:
        edata = [{"time": parse_ts(r.get("ts","")),
                  "Equity": safe_float(r.get("payload",{}).get("equity")),
                  "Cash": safe_float(r.get("payload",{}).get("cash"))}
                 for r in acct_records if r.get("payload",{}).get("equity")]
        df_eq = pd.DataFrame(edata).set_index("time").sort_index()
        if len(df_eq) >= 2:
            df_eq["Peak"] = df_eq["Equity"].cummax()
            df_eq["Drawdown %"] = (df_eq["Equity"] - df_eq["Peak"]) / df_eq["Peak"] * 100
            t1, t2 = st.tabs(["Equity & Cash", "Drawdown %"])
            with t1:
                st.line_chart(df_eq[["Equity","Cash"]], width="stretch")
            with t2:
                st.area_chart(df_eq[["Drawdown %"]], width="stretch")
    else:
        st.info("Need at least 2 data points.")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — LIVE SCANNER
# ════════════════════════════════════════════════════════════════════════════════
elif active_page == "🔍 Live Scanner":
    market_open = is_market_open()
    mode_badge = "🟢 MARKET OPEN — Live Intraday Data (Alpaca IEX 5-min)" if market_open else "🔴 MARKET CLOSED — End-of-Day Data (yfinance)"
    st.subheader(f"🔍 Live Market Scanner  ·  {mode_badge}")

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

    # Resolve symbol list
    if _use_full_universe:
        if run_scan:
            _full_syms = _load_full_universe()
            if not _full_syms:
                st.error("Could not fetch universe from Alpaca. Check credentials.")
            symbols_to_scan = _full_syms
        else:
            symbols_to_scan = []
    else:
        symbols_to_scan = [s.strip().upper() for s in scanner_symbols_input.split(",") if s.strip()]

    # Apply pre-scan snapshot filter to trim the universe before deep scanning
    pre_filter_count = len(symbols_to_scan)
    if run_scan and symbols_to_scan and _use_filters:
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

    if symbols_to_scan and not run_scan:
        st.info(
            f"Watchlist: **{len(symbols_to_scan):,} symbols** — "
            f"{', '.join(symbols_to_scan[:12])}{'…' if len(symbols_to_scan) > 12 else ''}"
        )
    elif not symbols_to_scan and not run_scan:
        if _use_full_universe:
            st.info("Full universe mode — click **▶ Run Scanner Now** to fetch all ~12K tickers and scan.")
        else:
            st.info("Click **▶ Run Scanner Now** in the sidebar to scan the watchlist.")

    # Only scan when user explicitly clicks the button
    needs_scan = run_scan
    if needs_scan and symbols_to_scan:
        _mode_label = "live" if market_open else "EOD"
        with st.spinner(f"Deep-scanning **{len(symbols_to_scan):,}** symbols in {_mode_label} mode…"):
            try:
                if market_open:
                    results = scan_live(symbols_to_scan, top_n=scanner_top_n)
                    if not results:
                        st.warning("Live data returned no results — falling back to EOD data.")
                        results = scan(symbols_to_scan, fast_ma=int(scanner_fast_ma), slow_ma=int(scanner_slow_ma), top_n=scanner_top_n)
                else:
                    results = scan(symbols_to_scan, fast_ma=int(scanner_fast_ma), slow_ma=int(scanner_slow_ma), top_n=scanner_top_n)
                results = _overlay_latest_prices(results)
                st.session_state["scanner_results"] = results
                st.session_state["scanner_ts"] = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
                st.session_state["scanner_mode"] = "live" if market_open else "eod"
                st.session_state["scanner_pre_filter"] = pre_filter_count
                st.session_state["scanner_post_filter"] = len(symbols_to_scan)
            except Exception as exc:
                st.error(f"Scanner error: {exc}")
                st.session_state["scanner_results"] = []

    results = st.session_state.get("scanner_results", [])
    scan_ts = st.session_state.get("scanner_ts", "—")
    scan_mode = st.session_state.get("scanner_mode", "—")
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
            f"<b>{len(results)}</b> results from <b>{_s_post or '?'}</b> symbols{_filter_note} · "
            f"as of <b>{scan_ts}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("")

    if not results and not run_scan:
        st.info("Click **▶ Run Scanner Now** in the sidebar to scan the watchlist.")

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

        table_rows = [{
            "Rank": i + 1,
            "Symbol": r.symbol,
            "Score": r.score,
            "Est. Upside": f"+{r.upside_pct:.1f}%" if r.upside_pct > 0 else "—",
            "Signal": r.signal,
            "Price": f"${r.close:.2f}",
            "1D %": f"{r.change_pct:+.2f}%",
            col5d_label: f"{r.momentum_5d:+.2f}%",
            "RSI": round(r.rsi, 1),
            "Vol Surge": f"{r.volume_surge:.1f}x",
            col_ma_label: f"{r.ma_gap_pct:+.2f}%",
            "Trend": f"{r.trend_consistency:.0f}%",
            "Top Driver": r.top_driver,
            "Plain English": explain_buy(r),
            "Full Reason": r.reason,
        } for i, r in enumerate(results)]

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
        if run_scan or st.session_state.get("_last_search") != search_q:
            st.session_state["scanner_page"] = 0
            st.session_state["_last_search"] = search_q
        current_page = st.session_state.get("scanner_page", 0)
        current_page = max(0, min(current_page, total_pages - 1))

        page_start = current_page * _PAGE_SIZE
        page_rows  = filtered_rows[page_start: page_start + _PAGE_SIZE]

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

            styled = (
                df_scan.style
                .map(style_score,  subset=["Score"])
                .map(style_signal, subset=["Signal"])
                .map(style_upside, subset=["Est. Upside"])
            )
            st.dataframe(styled, width="stretch", height=min(80 + len(page_rows) * 35, 700))

        # ── Top 3 Pick Cards (always from full unfiltered results) ───────────
        st.markdown("---")
        st.subheader("⭐ Top Picks Right Now")
        top3 = results[:min(3, len(results))]
        card_cols = st.columns(len(top3))
        for col, r in zip(card_cols, top3):
            bg = "#14532d" if r.signal == "BUY" else "#713f12" if r.signal == "WATCH" else "#1e293b"
            badge = "🟢 BUY" if r.signal == "BUY" else "🟡 WATCH" if r.signal == "WATCH" else "⚪ NEUTRAL"
            upside_str = f"+{r.upside_pct:.1f}% est. upside" if r.upside_pct > 0 else ""
            col.markdown(
                f"""
                <div style="background:{bg};border-radius:12px;padding:20px;text-align:center">
                  <h2 style="margin:0 0 4px;color:white;font-size:2em">{r.symbol}</h2>
                  <p style="font-size:2.4em;margin:0;color:white;font-weight:bold;line-height:1">{r.score:.1f}</p>
                  <p style="margin:4px 0 2px;color:#d1fae5;font-size:1.1em">{badge}</p>
                  {f'<p style="margin:0 0 6px;color:#86efac;font-size:1.05em;font-weight:bold">{upside_str}</p>' if upside_str else ''}
                  <hr style="border-color:rgba(255,255,255,0.2);margin:8px 0">
                  <p style="margin:2px 0;color:#e2e8f0">
                    <b>${r.close:.2f}</b> &nbsp;
                    <span style="color:{'#86efac' if r.change_pct >= 0 else '#fca5a5'}">{r.change_pct:+.2f}% today</span>
                  </p>
                  <p style="margin:4px 0 2px;color:#fde68a;font-size:0.85em;font-weight:bold">
                    💡 {r.top_driver}
                  </p>
                  <p style="margin:6px 0 2px;color:#e2e8f0;font-size:0.88em">🗣️ {explain_buy(r)}</p>
                  <p style="margin:2px 0;color:#94a3b8;font-size:0.75em;font-style:italic">{r.reason}</p>
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
            st.bar_chart(chart_df[["Score"]], width="stretch")
        with _ct2:
            st.bar_chart(chart_df[["Est. Upside %"]], width="stretch")

    else:
        if not symbols_to_scan:
            st.warning("Add symbols to the watchlist in the sidebar.")
        elif not needs_scan:
            st.info("Click **▶ Run Scanner Now** in the sidebar to scan the market.")
        else:
            st.warning("No results returned. Market data may be unavailable.")

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SELL SCANNER (mirror of Live Scanner, ranks symbols to SELL/TRIM)
# ════════════════════════════════════════════════════════════════════════════════
elif active_page == "💸 Sell Scanner":
    market_open = is_market_open()
    mode_badge = "🟢 MARKET OPEN — Live Intraday Data" if market_open else "🔴 MARKET CLOSED — End-of-Day Data"
    st.subheader(f"💸 Profit-Take Scanner  ·  {mode_badge}")
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

    # Build symbol list
    def _load_open_positions() -> list[dict]:
        try:
            from ai_trading.broker.alpaca_broker import AlpacaBroker
            from ai_trading.config import Settings
            _s = Settings.from_env()
            _b = AlpacaBroker(api_key=_s.alpaca_api_key, api_secret=_s.alpaca_api_secret, paper=_s.paper_only)
            return _b.all_positions()
        except Exception as exc:
            st.error(f"Could not load positions: {exc}")
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
            symbols_set.extend(s.strip().upper() for s in scanner_symbols_input.split(",") if s.strip())
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
            with st.spinner(f"Scoring {len(sell_syms):,} symbols for sell signals…"):
                try:
                    if market_open:
                        raw = scan_live(sell_syms, top_n=len(sell_syms))
                        if not raw:
                            raw = scan(sell_syms, fast_ma=int(scanner_fast_ma), slow_ma=int(scanner_slow_ma), top_n=len(sell_syms))
                    else:
                        raw = scan(sell_syms, fast_ma=int(scanner_fast_ma), slow_ma=int(scanner_slow_ma), top_n=len(sell_syms))
                    raw = _overlay_latest_prices(raw)
                except Exception as exc:
                    st.error(f"Sell scan failed: {exc}")
                    raw = []
            st.session_state["sell_results"] = raw
            st.session_state["sell_held"] = held_syms
            st.session_state["sell_positions"] = positions_data
            st.session_state["sell_ts"] = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
            st.session_state["sell_mode"] = "live" if market_open else "eod"

    sell_results = st.session_state.get("sell_results", [])
    held_syms = st.session_state.get("sell_held", [])
    positions_data = st.session_state.get("sell_positions", [])
    ss_ts = st.session_state.get("sell_ts", "—")
    ss_mode = st.session_state.get("sell_mode", "—")
    pos_map = {p["symbol"]: p for p in positions_data}

    if not sell_results and not _ss_run:
        st.info("Pick a source and click **▶ Run sell scan**. The scanner will rank each symbol "
                "from strongest sell to safest hold.")
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
                "Last": round(r.close, 2),
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
                "Filter by action", options=["TAKE PROFIT", "TRIM", "HOLD"],
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
            def _style_sell_score(v):
                if not isinstance(v, (int, float)): return ""
                if v >= 70: return "background-color:#14532d;color:#86efac;font-weight:bold"
                if v >= 50: return "background-color:#713f12;color:#fde68a"
                return "color:#94a3b8"

            def _style_sell_action(v):
                if v == "TAKE PROFIT": return "background-color:#14532d;color:#86efac;font-weight:bold"
                if v == "TRIM": return "background-color:#713f12;color:#fde68a"
                return "color:#94a3b8"

            def _style_pnl(v):
                if not isinstance(v, (int, float)): return ""
                if v > 0: return "color:#86efac"
                if v < 0: return "color:#fca5a5"
                return ""

            styled = df_sell.style.map(_style_sell_score, subset=["Profit-Take"]).map(_style_sell_action, subset=["Action"])
            for col in ("P&L $", "P&L %"):
                if col in df_sell.columns:
                    styled = styled.map(_style_pnl, subset=[col])
            st.dataframe(styled, width="stretch", height=min(80 + len(page_rows) * 35, 700))

        # ── Top sell cards ────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("� Best Profit-Taking Opportunities Right Now")
        top_sell = [r for r in rows if r["Action"] in ("TAKE PROFIT", "TRIM")][:3]
        if not top_sell:
            st.success("Nothing looks extended — no peaks to lock in right now. Let winners run.")
        else:
            card_cols = st.columns(len(top_sell))
            for col, r in zip(card_cols, top_sell):
                bg = "#14532d" if r["Action"] == "TAKE PROFIT" else "#713f12"
                badge = "💰 TAKE PROFIT" if r["Action"] == "TAKE PROFIT" else "🟡 TRIM"
                held_badge = " · 🪙 held" if r["Held?"] == "✅" else ""
                pnl_html = ""
                if r.get("P&L $") is not None:
                    pnl_color = "#86efac" if r["P&L $"] >= 0 else "#fca5a5"
                    pnl_html = (
                        f'<p style="margin:2px 0;color:{pnl_color};font-weight:bold">'
                        f'Unrealized: ${r["P&L $"]:+.2f} ({r.get("P&L %", 0):+.2f}%)</p>'
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
                      {pnl_html}
                      <p style="margin:4px 0 2px;color:#fde68a;font-size:0.85em">💡 {r['Why Sell']}</p>
                      <p style="margin:6px 0 2px;color:#e2e8f0;font-size:0.88em">🗣️ {r.get('Plain English','')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ── One-click paper close for held TAKE PROFIT/TRIM ───────────────────
        held_sell = [r for r in rows if r["Held?"] == "✅" and r["Action"] in ("TAKE PROFIT", "TRIM")]
        if held_sell:
            st.markdown("---")
            st.subheader("⚡ Lock in profits (paper close)")
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
                    _b = AlpacaBroker(api_key=_s.alpaca_api_key, api_secret=_s.alpaca_api_secret, paper=True)
                    _b.close_position(sym)
                    st.success(f"✓ Submitted close order for {sym}")
                except Exception as exc:
                    st.error(f"Close failed for {sym}: {exc}")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 4 — POSITION ADVISOR (manual entry → buy/sell/trim/hold recommendation)
# ════════════════════════════════════════════════════════════════════════════════
elif active_page == "🧮 Position Advisor":
    st.subheader("🧮 Position Advisor — Should I Buy More, Hold, Trim, or Sell?")
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
                        st.dataframe(pd.DataFrame(_diag["skipped"]), width="stretch")
                with st.expander("Preview parsed holdings"):
                    st.dataframe(pd.DataFrame(_holdings), width="stretch")

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
                            "Market Value": round(rec["market_value"], 2),
                            "P&L $": round(rec["total_return"], 2),
                            "P&L %": round(rec["pnl_pct"], 2),
                            "Why": rec["rationale"][0] if rec["rationale"] else "",
                        })

                    st.session_state["adv_csv_rows"] = rec_rows
                    st.session_state["adv_csv_full"] = full_recs

                rec_rows = st.session_state.get("adv_csv_rows", [])
                full_recs = st.session_state.get("adv_csv_full", [])
                if rec_rows:
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

                    def _sty_action(v):
                        if "BUY" in str(v): return "background-color:#14532d;color:#86efac;font-weight:bold"
                        if "TAKE PROFIT" in str(v) or "SELL" in str(v): return "background-color:#450a0a;color:#fca5a5;font-weight:bold"
                        if "TRIM" in str(v): return "background-color:#713f12;color:#fde68a;font-weight:bold"
                        return "color:#94a3b8"

                    def _sty_pnl(v):
                        if not isinstance(v, (int, float)): return ""
                        if v > 0: return "color:#86efac"
                        if v < 0: return "color:#fca5a5"
                        return ""

                    styled = df_rec.style.map(_sty_action, subset=["Action"])
                    for col in ("P&L $", "P&L %"):
                        if col in df_rec.columns:
                            styled = styled.map(_sty_pnl, subset=[col])
                    st.dataframe(styled, width="stretch", height=min(80 + len(rec_rows) * 35, 700))

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
elif active_page == "🧩 Patterns Heatmap":
    st.title("🧩 Pattern & Ensemble Heatmap")
    st.caption("Runs all pattern detectors + the regime-aware ensemble across your watchlist.")

    _pa_symbols_raw = st.text_area(
        "Symbols (comma-separated)",
        value=_env_symbols or _DEFAULT_WATCHLIST,
        height=80,
    )
    _pa_symbols = [s.strip().upper() for s in _pa_symbols_raw.split(",") if s.strip()][:120]
    _pa_period = st.selectbox("History window", ["6mo", "1y", "2y"], index=1)
    _pa_run = st.button("▶ Run pattern scan", type="primary")

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
        df_pat = _pattern_scan(tuple(_pa_symbols), _pa_period)
        if df_pat.empty:
            st.warning("No data returned.")
        else:
            df_pat = df_pat.sort_values("strength", ascending=False)
            c1, c2, c3 = st.columns(3)
            c1.metric("Symbols scanned", len(df_pat))
            c2.metric("Bullish (BUY)", int((df_pat["ensemble"] == "BUY").sum()))
            c3.metric("Bearish (SELL)", int((df_pat["ensemble"] == "SELL").sum()))

            pattern_cols = [c for c in df_pat.columns if c not in ("symbol", "regime", "ensemble", "strength", "_err")]
            display = df_pat[["symbol", "ensemble", "regime", "strength", *pattern_cols]].copy()
            styled = display.style.background_gradient(
                cmap="RdYlGn", subset=["strength", *pattern_cols], vmin=-1, vmax=1
            )
            st.dataframe(styled, width="stretch", height=min(80 + len(display) * 32, 700))

            st.markdown("### Top bullish setups")
            st.dataframe(df_pat[df_pat["ensemble"] == "BUY"].head(10), width="stretch")
            st.markdown("### Top bearish setups")
            st.dataframe(df_pat[df_pat["ensemble"] == "SELL"].head(10), width="stretch")
    else:
        st.info("Click **▶ Run pattern scan** to compute pattern signals across your watchlist.")

# ─────────────────────────────────────────────────────────────────────────────
# Options Lab page
# ─────────────────────────────────────────────────────────────────────────────
elif active_page == "🎯 Options Lab":
    st.title("🎯 Options Lab")
    st.caption("Scan option chains for the best risk-adjusted strategy candidates. "
               "Powered by Alpaca options API (fallback: yfinance).")

    _ol_col1, _ol_col2, _ol_col3, _ol_col4 = st.columns([3, 2, 2, 2])
    with _ol_col1:
        _ol_underlying_raw = st.text_input("Underlyings (comma-separated)", value="SPY,QQQ,AAPL,NVDA")
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

    _ol_col5, _ol_col6, _ol_col7 = st.columns([2, 2, 3])
    with _ol_col5:
        _ol_source = st.selectbox("Data source", ["auto", "alpaca", "yfinance"])
    with _ol_col6:
        _ol_top = st.number_input("Top N results", 5, 100, 15)
    with _ol_col7:
        _ol_run = st.button("▶ Run options scan", type="primary")

    @st.cache_data(ttl=120, show_spinner="Fetching chains & scoring strategies…")
    def _options_scan_cached(symbols: tuple[str, ...], strategies: tuple[str, ...],
                              min_dte: int, max_dte: int, delta: float, width: float,
                              source: str, top_n: int) -> list[dict]:
        from ai_trading.options.scanner import scan_options
        results = scan_options(
            underlyings=list(symbols),
            universe=None,
            use_equity_scanner=False,
            strategies=list(strategies),
            top_n=top_n,
            per_strategy_top=2,
            min_dte=min_dte,
            max_dte=max_dte,
            target_delta=delta,
            spread_width=width,
            source=source,
        )
        return [c.to_dict() for c in results]

    if _ol_run:
        if not _ol_strategies:
            st.warning("Pick at least one strategy.")
        else:
            symbols = tuple(s.strip().upper() for s in _ol_underlying_raw.split(",") if s.strip())[:40]
            try:
                results = _options_scan_cached(
                    symbols, tuple(_ol_strategies),
                    int(_ol_min_dte), int(_ol_max_dte),
                    float(_ol_delta), float(_ol_width),
                    _ol_source, int(_ol_top),
                )
                st.session_state["options_results"] = results
                st.session_state["options_ts"] = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
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
        st.dataframe(styled, width="stretch", height=min(80 + len(df_opt) * 35, 700))

        st.markdown("### Selected candidate details")
        _sel_idx = st.number_input("Inspect row #", 1, len(results), 1) - 1
        _sel = results[int(_sel_idx)]
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
            st.dataframe(pd.DataFrame(_opos), width="stretch")
        else:
            st.caption("No open option positions.")
    except Exception as exc:
        st.caption(f"(could not load option positions: {exc})")

# ── Auto-refresh via st.rerun ──
# Unlike meta http-equiv="refresh" (full page reload that wipes session_state),
# st.rerun() re-executes the script with all session_state preserved — so the
# stored scanner / options results survive and the page only "refreshes" cached
# data + recomputes the visible widgets.
if _autorefresh_on and not _skip_refresh:
    import time as _time
    _time.sleep(int(_autorefresh_sec))
    st.rerun()
