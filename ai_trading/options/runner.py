"""Options runner — scan + (optionally) place paper orders.

Examples:
    # Scan only (no orders):
    python -m ai_trading.options.runner scan --underlying AAPL --strategies long_call

    # Scan + auto-place top candidate as a paper order (limit at mid):
    python -m ai_trading.options.runner trade --underlying AAPL --strategies csp \
        --top 1 --qty 1 --confirm

    # Scan universe + open top N as paper trades:
    python -m ai_trading.options.runner trade --universe dow30 --top 5 --qty 1 \
        --strategies bull_call,csp --min-pop 0.65 --max-risk-pct 1.0 --confirm
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone

from ai_trading.options.broker import OptionsBroker
from ai_trading.options.scanner import scan_options, print_results
from ai_trading.options.strategies import StrategyCandidate
from ai_trading.storage.journal import Journal


def _passes_gates(c: StrategyCandidate, min_pop: float, max_risk_dollars: float | None) -> tuple[bool, str]:
    if c.pop < min_pop:
        return False, f"pop {c.pop:.2f} < min {min_pop:.2f}"
    if max_risk_dollars is not None and c.max_loss > max_risk_dollars and c.max_loss != float("inf"):
        return False, f"max_loss ${c.max_loss:.0f} > cap ${max_risk_dollars:.0f}"
    if c.max_loss == float("inf"):
        return False, "naked / undefined risk (use --allow-naked)"
    return True, "ok"


def cmd_scan(args) -> int:
    results = scan_options(
        underlyings=args.underlying,
        universe=args.universe,
        use_equity_scanner=not args.no_equity_scan,
        strategies=[s.strip() for s in args.strategies.split(",") if s.strip()],
        top_n=args.top,
        per_strategy_top=args.per_strategy_top,
        equity_top_n=args.equity_top,
        min_dte=args.min_dte,
        max_dte=args.max_dte,
        target_delta=args.delta,
        spread_width=args.width,
        source=args.source,
    )
    print_results(results)
    return 0


def cmd_trade(args) -> int:
    logger = logging.getLogger("ai_trading.options.runner")
    journal = Journal(args.journal_path)

    if not args.paper and not args.confirm_live:
        print("Refusing to place live orders without --confirm-live (and you should still review).")
        return 2

    broker = OptionsBroker(paper=args.paper)
    level = broker.options_trading_level()
    bp = broker.options_buying_power()
    print(f"Options trading level: {level}  |  buying power: ${bp:,.2f}  |  mode: {'PAPER' if args.paper else 'LIVE'}")

    results = scan_options(
        underlyings=args.underlying,
        universe=args.universe,
        use_equity_scanner=not args.no_equity_scan,
        strategies=[s.strip() for s in args.strategies.split(",") if s.strip()],
        top_n=args.top * 3,            # over-pick so gates can filter
        per_strategy_top=args.per_strategy_top,
        equity_top_n=args.equity_top,
        min_dte=args.min_dte,
        max_dte=args.max_dte,
        target_delta=args.delta,
        spread_width=args.width,
        source=args.source,
    )

    selected: list[StrategyCandidate] = []
    max_risk_dollars = args.max_risk_pct * bp / 100.0 if args.max_risk_pct > 0 else None
    for c in results:
        ok, reason = _passes_gates(c, args.min_pop, max_risk_dollars)
        if not ok and (c.max_loss != float("inf") or not args.allow_naked):
            logger.info("skip %s %s: %s", c.underlying, c.strategy, reason)
            continue
        selected.append(c)
        if len(selected) >= args.top:
            break

    if not selected:
        print("No candidates passed gates.")
        return 0

    print()
    print_results(selected)
    print()

    if args.confirm:
        ans = input(f"Place {len(selected)} order(s) for {args.qty} contract(s) each? Type YES: ").strip()
        if ans != "YES":
            print("Cancelled.")
            return 0

    placed = 0
    for c in selected:
        try:
            order = broker.place_strategy(
                c, qty=args.qty, order_type="limit", price_slippage_pct=args.slippage_pct,
            )
            placed += 1
            journal.write({
                "event": "option_open",
                "ts": datetime.now(timezone.utc).isoformat(),
                "underlying": c.underlying,
                "strategy": c.strategy,
                "qty": args.qty,
                "order_id": str(getattr(order, "id", "")),
                "candidate": c.to_dict(),
            })
            print(f"✓ submitted: {c.underlying} {c.strategy} (id={getattr(order, 'id', '?')})")
        except Exception as exc:
            logger.error("order failed for %s %s: %s", c.underlying, c.strategy, exc)
            journal.write({
                "event": "option_order_error",
                "ts": datetime.now(timezone.utc).isoformat(),
                "underlying": c.underlying,
                "strategy": c.strategy,
                "error": str(exc),
            })

    print(f"\nPlaced {placed} order(s).")
    return 0


def cmd_positions(args) -> int:
    broker = OptionsBroker(paper=args.paper)
    positions = broker.all_option_positions()
    if not positions:
        print("No open option positions.")
        return 0
    print(f"{'Symbol':<22} {'Qty':>5} {'Entry':>8} {'Mark':>8} {'P&L':>10} {'%':>7}")
    for p in positions:
        print(f"{p['symbol']:<22} {p['qty']:>5} {p['avg_entry_price']:>8.2f} "
              f"{p['current_price']:>8.2f} {p['unrealized_pl']:>10.2f} {p['unrealized_plpc']*100:>6.2f}%")
    return 0


def cmd_close(args) -> int:
    broker = OptionsBroker(paper=args.paper)
    res = broker.close_position(args.symbol)
    print(f"close: {res}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Options runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--underlying", action="append")
    common.add_argument("--universe")
    common.add_argument("--no-equity-scan", action="store_true")
    common.add_argument("--strategies", default="long_call,csp,bull_call")
    common.add_argument("--top", type=int, default=10)
    common.add_argument("--per-strategy-top", type=int, default=2)
    common.add_argument("--equity-top", type=int, default=20)
    common.add_argument("--min-dte", type=int, default=21)
    common.add_argument("--max-dte", type=int, default=45)
    common.add_argument("--delta", type=float, default=0.30)
    common.add_argument("--width", type=float, default=5.0)
    common.add_argument("--source", choices=["auto", "alpaca", "yfinance"], default="auto")

    s = sub.add_parser("scan", parents=[common], help="Scan only — no orders")

    t = sub.add_parser("trade", parents=[common], help="Scan + place orders")
    t.add_argument("--qty", type=int, default=1)
    t.add_argument("--min-pop", type=float, default=0.55)
    t.add_argument("--max-risk-pct", type=float, default=2.0,
                   help="Max risk per trade as %% of options buying power")
    t.add_argument("--slippage-pct", type=float, default=0.0)
    t.add_argument("--allow-naked", action="store_true")
    t.add_argument("--confirm", action="store_true", help="Prompt before submitting")
    t.add_argument("--confirm-live", action="store_true")
    t.add_argument("--paper", action="store_true", default=True)
    t.add_argument("--live", dest="paper", action="store_false")
    t.add_argument("--journal-path", default=os.environ.get("BOT_JOURNAL_PATH", "logs/journal.jsonl"))

    pos = sub.add_parser("positions", help="List open option positions")
    pos.add_argument("--paper", action="store_true", default=True)
    pos.add_argument("--live", dest="paper", action="store_false")

    cl = sub.add_parser("close", help="Close an option position by OCC symbol")
    cl.add_argument("symbol")
    cl.add_argument("--paper", action="store_true", default=True)
    cl.add_argument("--live", dest="paper", action="store_false")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.cmd == "scan":
        return cmd_scan(args)
    if args.cmd == "trade":
        return cmd_trade(args)
    if args.cmd == "positions":
        return cmd_positions(args)
    if args.cmd == "close":
        return cmd_close(args)
    p.error("unknown command")
    return 1


if __name__ == "__main__":
    sys.exit(main())
