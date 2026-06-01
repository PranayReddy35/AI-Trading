"""Streamlit dashboard — Portfolio monitor + Live market scanner.

Run with:
    streamlit run ai_trading/dashboard.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Ensure project root is on sys.path so ai_trading is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd
import streamlit as st
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
refresh_secs = st.sidebar.selectbox(
    "Auto-refresh", [0, 15, 30, 60], index=2,
    format_func=lambda x: "Off" if x == 0 else f"{x}s",
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
_page_idx = 1 if _qp_page == "scanner" else 0
active_page = st.sidebar.radio(
    "Navigate",
    ["📊 Portfolio Monitor", "🔍 Live Scanner"],
    index=_page_idx,
    key="active_page",
)
# Write selection back to query params immediately so meta-refresh picks it up
st.query_params["page"] = "scanner" if "Scanner" in active_page else "portfolio"
st.sidebar.markdown("---")
st.sidebar.caption(f"Refreshed: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
if st.sidebar.button("🔄 Refresh Page"):
    st.cache_data.clear()
    st.rerun()


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=20)
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
                  <p style="margin:2px 0;color:#94a3b8;font-size:0.78em;font-style:italic">{r.reason}</p>
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

# ── Auto-refresh via st.rerun (preserves session_state, unlike meta http-equiv refresh) ──
if refresh_secs:
    import time as _time
    _time.sleep(refresh_secs)
    st.rerun()

