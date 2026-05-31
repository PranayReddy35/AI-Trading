from __future__ import annotations

import argparse
from datetime import datetime, timezone

from ai_trading.broker.alpaca_broker import AlpacaBroker, OrderError
from ai_trading.config import Settings
from ai_trading.data.market_data import AlpacaMarketData
from ai_trading.notifications.alerter import Notifier
from ai_trading.risk.manager import RiskManager
from ai_trading.storage.journal import Journal, configure_logging
from ai_trading.strategy.moving_average import moving_average_signal


LIVE_WARNING = (
    "⚠️  LIVE TRADING MODE ACTIVE. Real money is at risk. "
    "Ensure you understand all risks before proceeding."
)

PAPER_NOTICE = (
    "Paper trading mode. No real money at risk."
)


def _confirm_live_trade(settings: Settings, action: str, symbol: str, qty: int) -> bool:
    """Prompt for confirmation before placing live orders (if enabled)."""
    if settings.paper_only or not settings.require_confirmation:
        return True

    print(f"\n{'='*60}")
    print(f"  LIVE ORDER CONFIRMATION REQUIRED")
    print(f"  Action: {action} {qty} shares of {symbol}")
    print(f"  Account: LIVE (real money)")
    print(f"{'='*60}")
    response = input("Type 'YES' to confirm, anything else to cancel: ").strip()
    return response == "YES"


def _preflight_checks(settings: Settings, broker: AlpacaBroker, logger) -> bool:
    """Run pre-flight checks before trading. Returns True if all pass."""
    account = broker.account_state()

    # Check account status
    if account["status"] != "ACTIVE":
        logger.error("Account status is %s, not ACTIVE. Aborting.", account["status"])
        return False

    # Check if trading is restricted (PDT flag for live accounts)
    if not settings.paper_only and account.get("pattern_day_trader"):
        logger.warning("Account flagged as pattern day trader. Proceed with caution.")

    return True


def run_once(symbol_override: str | None = None, skip_confirmation: bool = False) -> None:
    settings = Settings.from_env()
    if symbol_override:
        settings.symbol = symbol_override.upper()
    if skip_confirmation:
        settings.require_confirmation = False
    settings.validate()

    logger = configure_logging(settings.log_path)
    journal = Journal(settings.journal_path)
    notifier = Notifier(settings.webhook_url, settings.notify_events)

    if settings.is_live:
        logger.warning(LIVE_WARNING)
        journal.write("mode", {"mode": "LIVE", "warning": LIVE_WARNING})
    else:
        logger.info(PAPER_NOTICE)

    broker = AlpacaBroker(settings.api_key, settings.api_secret, paper=settings.paper_only)
    market_data = AlpacaMarketData(settings.api_key, settings.api_secret)

    risk = RiskManager(
        paper_only=settings.paper_only,
        min_cash_threshold=settings.min_cash_threshold,
        max_shares=settings.max_shares,
        max_daily_trades=settings.max_daily_trades,
        max_consecutive_errors=settings.max_consecutive_errors,
        daily_loss_limit_pct=settings.daily_loss_limit_pct,
        max_portfolio_exposure_pct=settings.max_portfolio_exposure_pct,
        min_equity=settings.min_equity,
        trade_cooldown_sec=settings.trade_cooldown_sec,
    )

    now = datetime.now(timezone.utc)

    try:
        # Pre-flight checks
        if not _preflight_checks(settings, broker, logger):
            journal.write("preflight_failed", {"reason": "account not ready"})
            notifier.notify("error", "Pre-flight check failed: account not ready")
            return

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
            equity=float(account["equity"]),
            last_equity=float(account.get("last_equity", 0)),
            portfolio_value=float(account.get("portfolio_value", 0)),
        )

        if not decision.allowed:
            logger.info("Risk manager rejected order: %s", decision.reason)
            journal.write("risk_reject", {"action": action, "reason": decision.reason})
            notifier.notify("risk_reject", f"Order rejected: {decision.reason}")
            return

        # Confirmation for live trading
        if not _confirm_live_trade(settings, action, settings.symbol, decision.approved_qty):
            logger.info("Order cancelled by user confirmation.")
            journal.write("user_cancel", {"action": action, "reason": "confirmation denied"})
            return

        # Determine limit price if using limit orders
        limit_price = None
        if settings.order_type == "limit":
            offset_mult = 1 + (settings.limit_price_offset_pct / 100.0)
            if action == "BUY":
                # Place limit slightly above current price to improve fill chance
                limit_price = signal.close * offset_mult
            else:
                # Place limit slightly below current price (mirror of buy offset)
                limit_price = signal.close * (1 - settings.limit_price_offset_pct / 100.0)

        # Submit order with retry logic
        order = broker.submit_order(
            settings.symbol,
            action,
            decision.approved_qty,
            order_type=settings.order_type,
            limit_price=limit_price,
            max_retries=settings.max_api_retries,
        )

        risk.register_trade(now.date())
        risk.clear_error_streak()

        order_record = {
            "symbol": settings.symbol,
            "action": action,
            "qty": decision.approved_qty,
            "order_id": str(order.id),
            "order_type": settings.order_type,
            "limit_price": limit_price,
            "mode": "LIVE" if settings.is_live else "PAPER",
        }

        logger.info(
            "Submitted %s %s order %s qty=%s",
            "LIVE" if settings.is_live else "PAPER",
            action,
            order.id,
            decision.approved_qty,
        )
        journal.write("order", order_record)
        notifier.notify(
            "trade",
            f"{'LIVE' if settings.is_live else 'PAPER'} {action} {decision.approved_qty} {settings.symbol}",
            order_record,
        )

        # Wait for fill if configured
        if settings.order_fill_timeout_sec > 0:
            fill_result = broker.wait_for_fill(
                str(order.id), timeout_sec=settings.order_fill_timeout_sec
            )
            journal.write("fill_status", fill_result)
            if fill_result["status"] == "filled":
                logger.info(
                    "Order filled: qty=%s avg_price=%.2f",
                    fill_result["filled_qty"],
                    fill_result["filled_avg_price"],
                )
            elif fill_result["status"] == "timeout":
                logger.warning("Order fill timed out after %ds", settings.order_fill_timeout_sec)
                notifier.notify("error", f"Order fill timeout: {order.id}")
            else:
                logger.warning("Order status: %s", fill_result["status"])

            # Place stop-loss if configured and order was a BUY that filled
            if (
                settings.stop_loss_pct > 0
                and action == "BUY"
                and fill_result["status"] == "filled"
                and fill_result["filled_avg_price"] > 0
            ):
                stop_price = fill_result["filled_avg_price"] * (
                    1 - settings.stop_loss_pct / 100.0
                )
                try:
                    sl_order = broker.submit_stop_loss(
                        settings.symbol, fill_result["filled_qty"], stop_price
                    )
                    journal.write(
                        "stop_loss",
                        {
                            "order_id": str(sl_order.id),
                            "stop_price": stop_price,
                            "qty": fill_result["filled_qty"],
                        },
                    )
                except Exception as sl_exc:
                    logger.warning("Failed to place stop-loss: %s", sl_exc)

    except OrderError as exc:
        risk.register_error()
        logger.error("Order error: %s", exc)
        journal.write("order_error", {"error": str(exc)})
        notifier.notify("error", f"Order error: {exc}")
    except Exception as exc:
        risk.register_error()
        logger.exception("Bot run failed: %s", exc)
        journal.write("error", {"error": str(exc)})
        notifier.notify("error", f"Bot error: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one trading bot cycle (paper or live, based on BOT_PAPER_ONLY env var)."
    )
    parser.add_argument("--symbol", help="Optional symbol override, e.g. SPY")
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip order confirmation prompt (use for automated/scheduled runs)",
    )
    args = parser.parse_args()
    run_once(symbol_override=args.symbol, skip_confirmation=args.no_confirm)


if __name__ == "__main__":
    main()
