"""Options scanner.

Given a list of underlyings (or a universe alias) and a directional bias from
the equity scanner, build the top option strategy candidates per symbol.

Workflow:
1. Run the equity scanner (EOD) over the universe → get BUY/WATCH candidates
   with bullish bias, plus optional bearish picks (lowest scores).
2. For each picked underlying, fetch the option chain.
3. Generate strategy candidates per the chosen strategies & params.
4. Rank globally by composite score; print/return top N.

CLI:
    python -m ai_trading.options.scanner --underlying SPY --strategy long_call
    python -m ai_trading.options.scanner --universe sp500 --top 10 --strategies long_call,csp,vertical
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from datetime import date, timedelta
from typing import Iterable

from ai_trading.options.chains import OptionContract, get_chain
from ai_trading.options.strategies import (
    StrategyCandidate,
    build_cash_secured_put,
    build_covered_call,
    build_iron_condor,
    build_long_call,
    build_long_put,
    build_strangle,
    build_vertical_spread,
)

logger = logging.getLogger("ai_trading.options.scanner")


STRATEGY_BUILDERS = {
    "long_call":      "bullish",
    "long_put":       "bearish",
    "csp":            "bullish",
    "covered_call":   "neutral",
    "bull_call":      "bullish",
    "bear_put":       "bearish",
    "bull_put":       "bullish",
    "bear_call":      "bearish",
    "iron_condor":    "neutral",
    "short_strangle": "neutral",
}


def build_candidates_for_symbol(
    symbol: str,
    strategies: Iterable[str],
    min_dte: int = 21,
    max_dte: int = 45,
    target_delta: float = 0.30,
    spread_width: float = 5.0,
    cost_basis: float | None = None,
    source: str = "auto",
    per_strategy_top: int = 2,
) -> list[StrategyCandidate]:
    """Fetch chain once, run all requested strategy builders, return combined list."""
    today = date.today()
    chain = get_chain(
        symbol,
        expiry_gte=today + timedelta(days=min_dte),
        expiry_lte=today + timedelta(days=max_dte),
        source=source,
    )
    if not chain:
        logger.warning("No chain for %s", symbol)
        return []

    spot = chain[0].underlying_price
    out: list[StrategyCandidate] = []
    for s in strategies:
        s = s.lower()
        try:
            if s == "long_call":
                out.extend(build_long_call(chain, symbol, spot, target_delta=target_delta,
                                            min_dte=min_dte, max_dte=max_dte, top_n=per_strategy_top))
            elif s == "long_put":
                out.extend(build_long_put(chain, symbol, spot, target_delta=-target_delta,
                                           min_dte=min_dte, max_dte=max_dte, top_n=per_strategy_top))
            elif s == "csp":
                out.extend(build_cash_secured_put(chain, symbol, spot, target_delta=-target_delta,
                                                   min_dte=min_dte, max_dte=max_dte, top_n=per_strategy_top))
            elif s == "covered_call":
                if cost_basis is None or cost_basis <= 0:
                    cost_basis_use = spot   # assume bought at spot
                else:
                    cost_basis_use = cost_basis
                out.extend(build_covered_call(chain, symbol, spot, cost_basis_use,
                                                target_delta=target_delta,
                                                min_dte=min_dte, max_dte=max_dte,
                                                top_n=per_strategy_top))
            elif s in ("bull_call", "bear_put", "bull_put", "bear_call"):
                out.extend(build_vertical_spread(chain, symbol, spot, direction=s,
                                                  width=spread_width, long_delta=target_delta,
                                                  min_dte=min_dte, max_dte=max_dte,
                                                  top_n=per_strategy_top))
            elif s == "iron_condor":
                out.extend(build_iron_condor(chain, symbol, spot, short_delta=target_delta,
                                              wing_width=spread_width,
                                              min_dte=min_dte, max_dte=max_dte,
                                              top_n=per_strategy_top))
            elif s == "short_strangle":
                out.extend(build_strangle(chain, symbol, spot, short_delta=target_delta,
                                            min_dte=min_dte, max_dte=max_dte,
                                            top_n=per_strategy_top))
            else:
                logger.warning("Unknown strategy: %s", s)
        except Exception as exc:
            logger.warning("Strategy %s for %s failed: %s", s, symbol, exc)
    return out


def scan_options(
    underlyings: list[str] | None = None,
    universe: str | None = None,
    use_equity_scanner: bool = True,
    strategies: list[str] | None = None,
    top_n: int = 10,
    min_dte: int = 21,
    max_dte: int = 45,
    target_delta: float = 0.30,
    spread_width: float = 5.0,
    source: str = "auto",
    per_strategy_top: int = 2,
    equity_top_n: int = 20,
) -> list[StrategyCandidate]:
    """Top-level scan: resolve underlyings, run strategies, rank globally."""
    strategies = strategies or ["long_call", "csp", "bull_call"]

    if not underlyings:
        if use_equity_scanner and universe:
            from ai_trading.scanner import scan as eq_scan
            from ai_trading.data.universe import load_universe
            symbols = load_universe([universe])
            eq_results = eq_scan(symbols=symbols, top_n=equity_top_n, apply_filters=False, dedup=False)
            bullish = [r.symbol for r in eq_results if r.signal in ("BUY", "WATCH")]
            bearish = [r.symbol for r in eq_results if r.rel_strength_pct < -2.0]
            # Pick bullish for bullish strategies, bearish for bearish ones, all for neutral
            all_syms = bullish[:max(equity_top_n, 5)] + bearish[:5]
            underlyings = list(dict.fromkeys(all_syms))   # dedup preserving order
        elif universe:
            from ai_trading.data.universe import load_universe
            underlyings = load_universe([universe])
        else:
            raise ValueError("Either underlyings, universe, or both must be provided")

    all_cands: list[StrategyCandidate] = []
    for sym in underlyings:
        try:
            all_cands.extend(build_candidates_for_symbol(
                sym, strategies, min_dte=min_dte, max_dte=max_dte,
                target_delta=target_delta, spread_width=spread_width,
                source=source, per_strategy_top=per_strategy_top,
            ))
        except Exception as exc:
            logger.warning("scan failed for %s: %s", sym, exc)
    all_cands.sort(key=lambda c: c.score, reverse=True)
    return all_cands[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# Pretty print
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_money(v: float) -> str:
    if v == float("inf"):
        return "  ∞ "
    return f"${v:>7.0f}"


def print_results(results: list[StrategyCandidate]) -> None:
    if not results:
        print("No candidates found.")
        return
    print("=" * 110)
    print(f"{'#':>2}  {'Sym':<6} {'Strategy':<15} {'Spot':>7}  {'DTE':>4} {'Cost':>9} {'MaxP':>9} {'MaxL':>9} {'POP':>6} {'R:R':>5} {'Score':>6}  Note")
    print("-" * 110)
    for i, c in enumerate(results, 1):
        cost = c.debit_credit
        cost_str = f"D${cost:>6.0f}" if cost > 0 else f"C${-cost:>6.0f}"
        rr = "∞" if c.risk_reward >= 99 else f"{c.risk_reward:.2f}"
        print(f"{i:>2}  {c.underlying:<6} {c.strategy:<15} ${c.underlying_price:>6.2f}  "
              f"{c.dte:>3}d {cost_str:>9} {_fmt_money(c.max_profit)} {_fmt_money(c.max_loss)} "
              f"{c.pop*100:>5.1f}% {rr:>5} {c.score:>6.1f}  {c.notes}")
    print("=" * 110)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Options scanner")
    p.add_argument("--underlying", action="append", help="Underlying symbol (repeatable)")
    p.add_argument("--universe", help="Universe alias (sp500, nasdaq100, dow30, all)")
    p.add_argument("--no-equity-scan", action="store_true", help="Skip equity scanner pre-filter")
    p.add_argument("--strategies", default="long_call,csp,bull_call",
                   help="Comma-separated: long_call,long_put,csp,covered_call,bull_call,bear_put,bull_put,bear_call,iron_condor,short_strangle")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--per-strategy-top", type=int, default=2)
    p.add_argument("--equity-top", type=int, default=20)
    p.add_argument("--min-dte", type=int, default=21)
    p.add_argument("--max-dte", type=int, default=45)
    p.add_argument("--delta", type=float, default=0.30, help="Target delta (abs)")
    p.add_argument("--width", type=float, default=5.0, help="Spread width in strikes")
    p.add_argument("--source", choices=["auto", "alpaca", "yfinance"], default="auto")
    p.add_argument("--json", action="store_true", help="Output JSON instead of table")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

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

    if args.json:
        import json
        print(json.dumps([c.to_dict() for c in results], indent=2, default=str))
    else:
        print_results(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
