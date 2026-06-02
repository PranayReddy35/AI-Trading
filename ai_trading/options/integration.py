"""Integration helper: invoked from bot.run_once when BOT_OPTIONS_ENABLED=true.

Reuses already-scanned equity symbols (passed in) to pick option trades.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from ai_trading.options.broker import OptionsBroker
from ai_trading.options.scanner import scan_options
from ai_trading.options.runner import _passes_gates


def run_options_cycle(
    settings,
    candidate_symbols: list[str],
    journal,
    notifier=None,
    logger: logging.Logger | None = None,
) -> None:
    """Run a single options scan + (paper-)trade cycle.

    Activated when `settings.options_enabled` is True. Uses `candidate_symbols`
    as the underlying universe (typically the bot's `symbols` setting plus any
    equity scanner picks already gathered earlier in the loop).
    """
    logger = logger or logging.getLogger("ai_trading.options.cycle")
    if not getattr(settings, "options_enabled", False):
        return
    if not candidate_symbols:
        logger.info("options cycle: no candidate symbols, skipping")
        return

    strategies = [s.strip() for s in settings.options_strategies.split(",") if s.strip()]
    if not strategies:
        logger.info("options cycle: no strategies configured, skipping")
        return

    logger.info("options cycle: scanning %d underlyings × %d strategies",
                len(candidate_symbols), len(strategies))

    try:
        results = scan_options(
            underlyings=candidate_symbols,
            universe=None,
            use_equity_scanner=False,
            strategies=strategies,
            top_n=settings.options_top_n * 3,
            per_strategy_top=2,
            min_dte=settings.options_min_dte,
            max_dte=settings.options_max_dte,
            target_delta=settings.options_target_delta,
            spread_width=settings.options_spread_width,
            source=settings.options_data_source,
        )
    except Exception as exc:
        logger.warning("options scan failed: %s", exc)
        return

    if not results:
        logger.info("options cycle: no candidates")
        return

    broker = OptionsBroker(paper=settings.paper_only)
    bp = broker.options_buying_power()
    max_risk_dollars = (settings.options_max_risk_pct * bp / 100.0) if settings.options_max_risk_pct > 0 else None

    selected = []
    for c in results:
        ok, reason = _passes_gates(c, settings.options_min_pop, max_risk_dollars)
        if not ok and (c.max_loss != float("inf") or not settings.options_allow_naked):
            logger.info("skip option %s %s: %s", c.underlying, c.strategy, reason)
            continue
        selected.append(c)
        if len(selected) >= settings.options_top_n:
            break

    if not selected:
        logger.info("options cycle: no candidates passed gates (min_pop=%.2f)",
                    settings.options_min_pop)
        return

    if settings.options_dry_run:
        for c in selected:
            logger.info("[DRY-RUN] would place %s %s qty=%d (pop=%.2f, max_loss=$%.0f)",
                        c.underlying, c.strategy, settings.options_qty, c.pop, c.max_loss)
            journal.write("option_dry_run", {
                "underlying": c.underlying,
                "strategy": c.strategy,
                "qty": settings.options_qty,
                "candidate": c.to_dict(),
            })
        return

    for c in selected:
        try:
            order = broker.place_strategy(
                c, qty=settings.options_qty, order_type="limit",
                price_slippage_pct=settings.options_slippage_pct,
            )
            journal.write("option_open", {
                "ts": datetime.now(timezone.utc).isoformat(),
                "underlying": c.underlying,
                "strategy": c.strategy,
                "qty": settings.options_qty,
                "order_id": str(getattr(order, "id", "")),
                "candidate": c.to_dict(),
            })
            if notifier:
                notifier.notify("trade",
                                f"Options: opened {c.strategy} on {c.underlying} "
                                f"qty={settings.options_qty} pop={c.pop:.2f}")
        except Exception as exc:
            logger.error("option order failed for %s %s: %s", c.underlying, c.strategy, exc)
            journal.write("option_order_error", {
                "underlying": c.underlying,
                "strategy": c.strategy,
                "error": str(exc),
            })
