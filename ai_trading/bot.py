from __future__ import annotations

import argparse
from datetime import datetime, timezone

from ai_trading.broker.alpaca_broker import AlpacaBroker
from ai_trading.config import Settings
from ai_trading.data.market_data import AlpacaMarketData
from ai_trading.risk.manager import RiskManager
from ai_trading.storage.journal import Journal, configure_logging
from ai_trading.strategy.moving_average import moving_average_signal


WARNING = (
    "Educational project only. Not financial advice. "
    "Paper trading only by default; use tiny position sizing."
)


def run_once(symbol_override: str | None = None) -> None:
    settings = Settings.from_env()
    if symbol_override:
        settings.symbol = symbol_override.upper()
    settings.validate()

    logger = configure_logging(settings.log_path)
    journal = Journal(settings.journal_path)
    logger.info(WARNING)

    broker = AlpacaBroker(settings.api_key, settings.api_secret, paper=settings.paper_only)
    market_data = AlpacaMarketData(settings.api_key, settings.api_secret)

    risk = RiskManager(
        paper_only=True,
        min_cash_threshold=settings.min_cash_threshold,
        max_shares=settings.max_shares,
        max_daily_trades=settings.max_daily_trades,
        max_consecutive_errors=settings.max_consecutive_errors,
    )

    now = datetime.now(timezone.utc)

    try:
        account = broker.account_state()
        journal.write("account_state", account)

        bars = market_data.get_daily_bars(settings.symbol, settings.lookback_days)
        signal = moving_average_signal(bars, settings.fast_ma, settings.slow_ma)
        position_qty = broker.position_qty(settings.symbol)
        has_open_order = broker.has_open_order(settings.symbol)

        journal.write(
            "signal",
            {
                "symbol": settings.symbol,
                "signal": signal.signal,
                "close": signal.close,
                "fast_ma": signal.fast_ma,
                "slow_ma": signal.slow_ma,
                "position_qty": position_qty,
            },
        )

        action = "HOLD"
        requested_qty = 0
        if signal.signal == "BUY" and position_qty == 0:
            action = "BUY"
            requested_qty = settings.max_shares
        elif signal.signal == "SELL" and position_qty > 0:
            action = "SELL"
            requested_qty = position_qty

        if action == "HOLD":
            logger.info("No action taken: signal=%s position=%s", signal.signal, position_qty)
            journal.write("decision", {"action": "HOLD", "reason": "signal/position rules"})
            risk.clear_error_streak()
            return

        decision = risk.evaluate(
            today=now.date(),
            paper_mode=broker.paper,
            market_open=broker.is_market_open(),
            cash=float(account["cash"]),
            has_open_order=has_open_order,
            side=action,
            requested_qty=requested_qty,
            current_position_qty=position_qty,
        )

        if not decision.allowed:
            logger.info("Risk manager rejected order: %s", decision.reason)
            journal.write("risk_reject", {"action": action, "reason": decision.reason})
            return

        order = broker.submit_order(settings.symbol, action, decision.approved_qty)
        risk.register_trade(now.date())
        risk.clear_error_streak()
        logger.info("Submitted %s order %s qty=%s", action, order.id, decision.approved_qty)
        journal.write(
            "order",
            {
                "symbol": settings.symbol,
                "action": action,
                "qty": decision.approved_qty,
                "order_id": str(order.id),
            },
        )
    except Exception as exc:
        risk.register_error()
        logger.exception("Bot run failed: %s", exc)
        journal.write("error", {"error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one safe Alpaca paper-trading bot cycle.")
    parser.add_argument("--symbol", help="Optional symbol override, e.g. SPY")
    args = parser.parse_args()
    run_once(symbol_override=args.symbol)


if __name__ == "__main__":
    main()
