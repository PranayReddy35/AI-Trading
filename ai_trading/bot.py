from __future__ import annotations

import argparse
from datetime import datetime, timezone

from ai_trading.broker.alpaca_broker import OrderError
from ai_trading.broker.robinhood_agent import create_broker
from ai_trading.config import Settings
from ai_trading.data.market_data import AlpacaMarketData
from ai_trading.data.cache import BarCache
from ai_trading.notifications.alerter import Notifier
from ai_trading.risk.manager import RiskManager
from ai_trading.risk.correlation import is_too_correlated
from ai_trading.risk.correlation_scaling import correlation_scale
from ai_trading.risk.sizing import (
    adaptive_thresholds,
    compute_atr_stop_and_size,
)
from ai_trading.storage.journal import Journal, configure_logging
from ai_trading.storage.json_logger import log_ensemble_decision
from ai_trading.strategy.moving_average import moving_average_signal, dip_buy_signal
from ai_trading.strategy.sentiment_filter import apply_sentiment_filter, configure_sentiment_cache

import logging
from pathlib import Path


LIVE_WARNING = (
    "⚠️  LIVE TRADING MODE ACTIVE. Real money is at risk. "
    "Ensure you understand all risks before proceeding."
)

PAPER_NOTICE = (
    "Paper trading mode. No real money at risk."
)


def _record_stock_dry_run(
    *,
    settings: Settings,
    journal: Journal,
    notifier: Notifier,
    logger,
    symbol: str,
    action: str,
    qty: int,
    reason: str,
    order_type: str = "market",
    limit_price: float | None = None,
    price: float | None = None,
    broker=None,
) -> None:
    dollar_amount = None
    intent_order_type = order_type
    intent_limit_price = limit_price
    if (
        settings.broker == "robinhood"
        and settings.robinhood_use_dollar_orders
        and action.upper() == "BUY"
    ):
        dollar_amount = settings.robinhood_dollar_amount_per_trade
        intent_order_type = "market"
        intent_limit_price = None
    record = {
        "symbol": symbol,
        "action": action,
        "qty": qty,
        "dollar_amount": dollar_amount,
        "reason": reason,
        "order_type": intent_order_type,
        "limit_price": intent_limit_price,
        "price": price,
        "mode": "LIVE" if settings.is_live else "PAPER",
        "dry_run": True,
    }
    logger.info(
        "STOCK DRY RUN: would submit %s %s qty=%s (%s)",
        action,
        symbol,
        qty,
        reason,
    )
    journal.write("stock_dry_run", record)
    notifier.notify(
        "trade",
        f"DRY RUN {'LIVE' if settings.is_live else 'PAPER'} {action} {qty} {symbol}",
        record,
    )
    if settings.broker == "robinhood" and hasattr(broker, "record_order_intent"):
        broker.record_order_intent(
            symbol=symbol,
            side=action,
            qty=qty,
            reason=reason,
            order_type=intent_order_type,
            limit_price=intent_limit_price,
            price=price,
            dollar_amount=dollar_amount,
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

    # Check account status (may be a string or an enum like AccountStatus.ACTIVE)
    status = account["status"]
    status_str = status.value if hasattr(status, "value") else str(status)
    if status_str != "ACTIVE":
        logger.error("Account status is %s, not ACTIVE. Aborting.", status_str)
        return False

    # Check if trading is restricted (PDT flag for live accounts)
    if not settings.paper_only and account.get("pattern_day_trader"):
        logger.warning("Account flagged as pattern day trader. Proceed with caution.")

    return True


def _latest_price_context(latest_price) -> dict:
    if latest_price is None:
        return {
            "latest_price": None,
            "latest_time": "",
            "latest_source": "unavailable",
            "latest_feed": "",
            "latest_age_seconds": None,
            "latest_stale": True,
            "latest_confidence": "LOW",
            "latest_confidence_score": 0,
            "latest_confidence_reason": "latest price unavailable",
        }
    feed = str(getattr(latest_price, "feed", "") or "")
    source = str(getattr(latest_price, "source", "") or "")
    age = getattr(latest_price, "age_seconds", None)
    stale = bool(getattr(latest_price, "stale", False))
    score = 100
    reasons: list[str] = []
    feed_l = feed.lower()
    if feed_l == "sip":
        reasons.append("SIP")
    elif feed_l == "iex":
        score -= 20
        reasons.append("IEX")
    elif feed_l in {"delayed_sip", "boats", "overnight"}:
        score -= 35
        reasons.append(feed.upper())
    else:
        score -= 20
        reasons.append(feed or "unknown feed")
    if stale:
        score -= 30
        reasons.append("stale")
    try:
        age_f = float(age) if age is not None else None
    except Exception:
        age_f = None
    if age_f is not None:
        if age_f > 900:
            score -= 30
            reasons.append(">15m old")
        elif age_f > 300:
            score -= 15
            reasons.append(">5m old")
        elif age_f <= 60:
            reasons.append("fresh")
    score = max(0, min(100, int(score)))
    confidence = "HIGH" if score >= 80 else "MEDIUM" if score >= 55 else "LOW"
    return {
        "latest_price": float(getattr(latest_price, "price", 0.0) or 0.0),
        "latest_time": str(getattr(latest_price, "timestamp", "") or ""),
        "latest_source": f"{feed.upper()} {source}".strip(),
        "latest_feed": feed,
        "latest_age_seconds": age,
        "latest_stale": stale,
        "latest_session": str(getattr(latest_price, "session", "") or ""),
        "latest_confidence": confidence,
        "latest_confidence_score": score,
        "latest_confidence_reason": ", ".join(reasons),
    }


def _freshness_gate(settings: Settings, action: str, latest_price) -> tuple[bool, str]:
    if not settings.require_fresh_price_for_orders:
        return True, "freshness gate disabled"
    if latest_price is None:
        if action == "SELL" and not settings.stale_price_blocks_sell:
            return True, "latest price unavailable; allowing SELL"
        return False, "latest price unavailable"

    ctx = _latest_price_context(latest_price)
    age = ctx.get("latest_age_seconds")
    stale = bool(ctx.get("latest_stale"))
    if age is not None and float(age) > settings.max_latest_price_age_sec:
        stale = True
    if stale:
        if action == "SELL" and not settings.stale_price_blocks_sell:
            return True, "latest price stale; allowing SELL"
        return False, f"latest price stale ({age if age is not None else 'unknown'}s old)"

    caution_feeds = {"iex", "delayed_sip", "boats", "overnight"}
    feed = str(ctx.get("latest_feed") or "").lower()
    if action == "BUY" and settings.block_caution_feeds_for_buys and feed in caution_feeds:
        return False, f"caution market data feed for BUY: {feed}"

    return True, "latest price fresh"


def _spread_gate(settings: Settings, latest_price) -> tuple[bool, str]:
    if not settings.use_spread_filter or latest_price is None:
        return True, "spread gate disabled"
    bid = getattr(latest_price, "bid", None)
    ask = getattr(latest_price, "ask", None)
    try:
        bid_f = float(bid or 0.0)
        ask_f = float(ask or 0.0)
    except Exception:
        return True, "spread unavailable"
    if bid_f <= 0 or ask_f <= 0 or ask_f < bid_f:
        return True, "spread unavailable"
    mid = (bid_f + ask_f) / 2.0
    spread_bps = (ask_f - bid_f) / mid * 1e4 if mid else 0.0
    if spread_bps > settings.max_spread_bps:
        return False, f"spread {spread_bps:.1f}bps > {settings.max_spread_bps:.1f}bps"
    return True, f"spread {spread_bps:.1f}bps OK"


def _order_preview_payload(
    *,
    settings: Settings,
    symbol: str,
    action: str,
    qty: int,
    price: float,
    reason: str,
    order_type: str = "market",
    limit_price: float | None = None,
    latest_price=None,
) -> dict:
    return {
        "symbol": symbol,
        "action": action,
        "qty": int(qty),
        "price": float(price),
        "reason": reason,
        "order_type": order_type,
        "limit_price": limit_price,
        "mode": "LIVE" if settings.is_live else "PAPER",
        "dry_run": bool(settings.stock_dry_run),
        **_latest_price_context(latest_price),
    }


def _record_order_preview(
    *,
    settings: Settings,
    journal: Journal,
    notifier: Notifier,
    payload: dict,
) -> None:
    journal.write("order_preview", payload)
    if settings.notify_trade_preview:
        notifier.notify(
            "trade",
            f"PREVIEW {payload['mode']} {payload['action']} {payload['qty']} {payload['symbol']}",
            payload,
        )


def _prepare_order_attempt(
    *,
    settings: Settings,
    journal: Journal,
    notifier: Notifier,
    logger,
    symbol: str,
    action: str,
    qty: int,
    price: float,
    reason: str,
    order_type: str = "market",
    limit_price: float | None = None,
    latest_price=None,
) -> tuple[bool, dict]:
    if settings.kill_switch:
        reject_reason = "BOT_KILL_SWITCH=true"
        logger.warning("Kill switch blocked %s %s", action, symbol)
        payload = {
            "symbol": symbol,
            "action": action,
            "reason": reject_reason,
            **_latest_price_context(latest_price),
        }
        journal.write("risk_reject", payload)
        notifier.notify("risk_reject", f"Kill switch blocked [{symbol}]: {reject_reason}", payload)
        return False, payload

    fresh_ok, fresh_reason = _freshness_gate(settings, action, latest_price)
    if not fresh_ok:
        logger.info("Freshness gate rejected %s %s: %s", action, symbol, fresh_reason)
        payload = {
            "symbol": symbol,
            "action": action,
            "reason": fresh_reason,
            **_latest_price_context(latest_price),
        }
        journal.write("risk_reject", payload)
        notifier.notify("risk_reject", f"Freshness gate blocked [{symbol}]: {fresh_reason}", payload)
        return False, payload

    spread_ok, spread_reason = _spread_gate(settings, latest_price)
    if not spread_ok:
        logger.info("Spread gate rejected %s %s: %s", action, symbol, spread_reason)
        payload = {
            "symbol": symbol,
            "action": action,
            "reason": spread_reason,
            **_latest_price_context(latest_price),
        }
        journal.write("risk_reject", payload)
        notifier.notify("risk_reject", f"Spread gate blocked [{symbol}]: {spread_reason}", payload)
        return False, payload

    payload = _order_preview_payload(
        settings=settings,
        symbol=symbol,
        action=action,
        qty=qty,
        price=price,
        reason=f"{reason}; {fresh_reason}; {spread_reason}",
        order_type=order_type,
        limit_price=limit_price,
        latest_price=latest_price,
    )
    _record_order_preview(
        settings=settings,
        journal=journal,
        notifier=notifier,
        payload=payload,
    )

    if not _confirm_live_trade(settings, action, symbol, qty):
        logger.info("Order cancelled by user confirmation.")
        journal.write("user_cancel", {"symbol": symbol, "action": action, "reason": "confirmation denied"})
        return False, payload

    return True, payload


def _maybe_retrain_ml(settings: Settings, symbol: str, logger) -> None:
    """Retrain ML model if it's stale, per ml_retrain_days setting."""
    if settings.ml_retrain_days <= 0:
        return
    try:
        from ai_trading.ml.retrainer import should_retrain, retrain_model
        if should_retrain(settings.ml_model_path, settings.ml_retrain_days):
            retrain_model(
                symbol=symbol,
                model_path=settings.ml_model_path,
                api_key=settings.api_key,
                api_secret=settings.api_secret,
            )
    except Exception as exc:
        logger.warning("ML retraining check failed: %s", exc)


def _record_current_prices(
    symbols: list[str],
    broker: AlpacaBroker,
    journal: Journal,
    logger,
) -> None:
    """Fetch and journal latest price snapshot for symbols."""
    prices: dict[str, float] = {}
    try:
        prices = broker.get_latest_prices(symbols)
    except Exception as exc:
        logger.warning("Batch latest price fetch failed: %s", exc)
    if len(prices) < len(symbols):
        for symbol in symbols:
            if symbol in prices:
                continue
            try:
                prices[symbol] = float(broker.get_latest_price(symbol))
            except Exception as exc:
                logger.warning("Latest price fetch failed for %s: %s", symbol, exc)
    if prices:
        journal.write("price_snapshot", {"prices": prices})


def _trade_symbol(
    symbol: str,
    settings: Settings,
    broker: AlpacaBroker,
    market_data: AlpacaMarketData,
    risk: RiskManager,
    journal: Journal,
    notifier: Notifier,
    logger,
    account: dict,
    all_bars: dict,
    now: datetime,
    market_is_open: bool = True,
    cycle_state: dict | None = None,
) -> None:
    """Run one full trade cycle for a single symbol."""
    # Optional cache: skip re-downloading the same bars within TTL
    if settings.cache_enabled:
        cache = BarCache(settings.cache_dir, ttl_seconds=settings.cache_ttl_sec)
        bars = cache.get_or_fetch(
            symbol, settings.lookback_days, settings.bar_timeframe,
            fetch=market_data.get_bars,
        )
    else:
        bars = market_data.get_bars(symbol, settings.lookback_days, settings.bar_timeframe)

    latest_price = None
    if settings.use_latest_price:
        try:
            latest_price = market_data.get_latest_price(symbol)
            previous_close = float(bars["close"].iloc[-1])
            bars = market_data.with_latest_price(bars, latest_price)
            journal.write(
                "latest_price",
                {
                    "symbol": symbol,
                    "price": latest_price.price,
                    "source": latest_price.source,
                    "feed": latest_price.feed,
                    "bid": latest_price.bid,
                    "ask": latest_price.ask,
                    "timestamp": latest_price.timestamp,
                    "age_seconds": latest_price.age_seconds,
                    "stale": latest_price.stale,
                    "session": latest_price.session,
                    "previous_bar_close": previous_close,
                    "bar_timeframe": settings.bar_timeframe,
                },
            )
        except Exception as exc:
            logger.warning("Latest price unavailable for %s; using latest bar close: %s", symbol, exc)
            journal.write(
                "latest_price_unavailable",
                {"symbol": symbol, "reason": str(exc), "bar_timeframe": settings.bar_timeframe},
            )
    all_bars[symbol] = bars

    # Choose signal source: ensemble (with optional MTF confirmation) or legacy MA crossover.
    if settings.use_ensemble_signal:
        from ai_trading.strategy.ensemble import compute_ensemble_signal
        from ai_trading.strategy.moving_average import SignalResult
        if settings.use_adaptive_thresholds:
            buy_th, sell_th = adaptive_thresholds(
                bars, settings.base_buy_threshold, settings.base_sell_threshold
            )
        else:
            buy_th, sell_th = settings.base_buy_threshold, settings.base_sell_threshold
        es = compute_ensemble_signal(bars, buy_threshold=buy_th, sell_threshold=sell_th)

        # Multi-timeframe confirmation: require the higher timeframe(s) to agree in direction.
        if settings.use_mtf_confirmation and es.signal in ("BUY", "SELL"):
            from ai_trading.strategy.multi_timeframe import mtf_signal
            tfs = [t.strip() for t in settings.mtf_timeframes.split(",") if t.strip()]
            mtf = mtf_signal(symbol, market_data.get_bars, timeframes=tfs,
                             lookback_days=settings.lookback_days)
            if (es.signal == "BUY" and mtf.signal < 0) or (es.signal == "SELL" and mtf.signal > 0):
                logger.info("MTF disagrees on %s (%s vs MTF=%+.2f) — downgrading to HOLD",
                            symbol, es.signal, mtf.signal)
                journal.write("mtf_disagree", {"symbol": symbol, "ensemble": es.signal,
                                               "mtf_signal": mtf.signal, "mtf_reason": mtf.reason})
                es_signal_str = "HOLD"
            else:
                es_signal_str = es.signal
        else:
            es_signal_str = es.signal

        signal = SignalResult(es_signal_str, float(bars["close"].iloc[-1]), 0, 0)
        log_ensemble_decision(journal, symbol, es)
    else:
        signal = moving_average_signal(bars, settings.fast_ma, settings.slow_ma)
    position_qty = broker.position_qty(symbol)
    has_open_order = broker.has_open_order(symbol)

    # Buy-the-dip: override signal to BUY when RSI oversold + price pulled back
    if (
        settings.dip_buy_enabled
        and position_qty == 0
        and signal.signal != "BUY"
    ):
        dip = dip_buy_signal(
            bars,
            rsi_threshold=settings.dip_rsi_threshold,
            drop_pct=settings.dip_drop_pct,
            lookback_days=settings.dip_lookback_days,
            long_ma_period=settings.dip_long_ma_period,
            require_above_long_ma=settings.dip_require_uptrend,
        )
        if dip.signal == "BUY":
            logger.info("Dip-buy triggered for %s: %s", symbol, dip.reason)
            journal.write("dip_buy_signal", {
                "symbol": symbol,
                "rsi": round(dip.rsi, 2),
                "drop_from_high_pct": round(dip.drop_from_high_pct, 2),
                "above_long_ma": dip.above_long_ma,
                "reason": dip.reason,
            })
            # Promote to BUY by patching the signal
            from ai_trading.strategy.moving_average import SignalResult
            signal = SignalResult("BUY", dip.close, signal.fast_ma, signal.slow_ma)

    # Gap-open protection: if the first bar of the day gapped too far, skip new buys
    # and optionally exit existing position
    if settings.gap_open_protection_pct > 0 and len(bars) >= 2:
        prior_close = float(bars["close"].iloc[-2])
        current_price = float(bars["close"].iloc[-1])
        gapped, gap_reason = risk.is_gap_open_too_large(
            symbol, current_price, prior_close, settings.gap_open_protection_pct
        )
        if gapped:
            logger.warning("Gap-open protection triggered for %s: %s", symbol, gap_reason)
            journal.write("gap_open_protect", {"symbol": symbol, "reason": gap_reason,
                                                "prior_close": prior_close, "current": current_price})
            notifier.notify("risk_reject", f"Gap-open block [{symbol}]: {gap_reason}")
            if position_qty > 0:
                logger.info("Gap-open: closing existing position in %s", symbol)
                ready, preview_payload = _prepare_order_attempt(
                    settings=settings,
                    journal=journal,
                    notifier=notifier,
                    logger=logger,
                    symbol=symbol,
                    action="SELL",
                    qty=position_qty,
                    price=current_price,
                    reason=f"gap-open protection: {gap_reason}",
                    latest_price=latest_price,
                )
                if not ready:
                    return
                if settings.stock_dry_run:
                    _record_stock_dry_run(
                        settings=settings,
                        journal=journal,
                        notifier=notifier,
                        logger=logger,
                        symbol=symbol,
                        action="SELL",
                        qty=position_qty,
                        reason=f"gap-open protection: {gap_reason}",
                        price=current_price,
                        broker=broker,
                    )
                    return
                broker.close_position(symbol)
                risk.clear_trailing_peak(symbol)
                journal.write("order", {**preview_payload, "reason": "gap-open protection"})
                notifier.notify("trade", f"Gap-open SELL {symbol}: {gap_reason}", preview_payload)
            return  # skip normal signal logic for this bar

    journal.write(
        "signal",
        {
            "symbol": symbol,
            "signal": signal.signal,
            "close": signal.close,
            "fast_ma": signal.fast_ma,
            "slow_ma": signal.slow_ma,
            "position_qty": position_qty,
        },
    )

    # Partial profit taking: if up ≥ trigger %, sell sell_pct% of shares and hold the rest
    if (
        position_qty > 0
        and settings.partial_profit_trigger_pct > 0
        and not risk.has_partial_profit_taken(symbol)
    ):
        pos_detail = broker.position_details(symbol)
        if pos_detail and pos_detail["unrealized_plpc"] * 100.0 >= settings.partial_profit_trigger_pct:
            sell_qty = max(1, int(position_qty * settings.partial_profit_sell_pct / 100.0))
            logger.info(
                "Partial profit triggered for %s: up %.1f%%, selling %d/%d shares (%.0f%%)",
                symbol,
                pos_detail["unrealized_plpc"] * 100.0,
                sell_qty,
                position_qty,
                settings.partial_profit_sell_pct,
            )
            try:
                ready, preview_payload = _prepare_order_attempt(
                    settings=settings,
                    journal=journal,
                    notifier=notifier,
                    logger=logger,
                    symbol=symbol,
                    action="SELL",
                    qty=sell_qty,
                    price=signal.close,
                    reason="partial profit",
                    latest_price=latest_price,
                )
                if not ready:
                    return
                if settings.stock_dry_run:
                    _record_stock_dry_run(
                        settings=settings,
                        journal=journal,
                        notifier=notifier,
                        logger=logger,
                        symbol=symbol,
                        action="SELL",
                        qty=sell_qty,
                        reason="partial profit",
                        price=signal.close,
                        broker=broker,
                    )
                    entry_price = float(pos_detail.get("avg_entry_price", 0))
                    risk.mark_partial_profit_taken(symbol, entry_price=entry_price)
                    return
                order = broker.submit_order(
                    symbol, "SELL", sell_qty,
                    order_type="market",
                    max_retries=settings.max_api_retries,
                )
                entry_price = float(pos_detail.get("avg_entry_price", 0))
                risk.mark_partial_profit_taken(symbol, entry_price=entry_price)
                journal.write("partial_profit", {
                    **preview_payload,
                    "order_id": str(getattr(order, "id", "")),
                    "symbol": symbol,
                    "unrealized_plpc": pos_detail["unrealized_plpc"],
                    "sold_qty": sell_qty,
                    "remaining_qty": position_qty - sell_qty,
                    "trigger_pct": settings.partial_profit_trigger_pct,
                    "sell_pct": settings.partial_profit_sell_pct,
                })
                notifier.notify(
                    "trade",
                    f"💰 Partial profit SELL {sell_qty} {symbol} "
                    f"(+{pos_detail['unrealized_plpc']*100:.1f}% gain) — holding {position_qty - sell_qty} shares",
                    preview_payload,
                )
            except Exception as exc:
                logger.warning("Partial profit sell failed for %s: %s", symbol, exc)
            return  # skip normal signal logic this cycle

    # Partial remainder exit: check if remaining shares after partial profit should be sold
    if position_qty > 0 and risk.has_partial_profit_taken(symbol):
        risk.increment_partial_profit_bars(symbol)
        risk.update_partial_profit_peak(symbol, signal.close)
        should_exit, exit_reason = risk.should_exit_partial_remainder(symbol, signal.close)
        if should_exit:
            logger.info("Partial remainder exit for %s: %s", symbol, exit_reason)
            try:
                ready, preview_payload = _prepare_order_attempt(
                    settings=settings,
                    journal=journal,
                    notifier=notifier,
                    logger=logger,
                    symbol=symbol,
                    action="SELL",
                    qty=position_qty,
                    price=signal.close,
                    reason=f"partial remainder exit: {exit_reason}",
                    latest_price=latest_price,
                )
                if not ready:
                    return
                if settings.stock_dry_run:
                    _record_stock_dry_run(
                        settings=settings,
                        journal=journal,
                        notifier=notifier,
                        logger=logger,
                        symbol=symbol,
                        action="SELL",
                        qty=position_qty,
                        reason=f"partial remainder exit: {exit_reason}",
                        price=signal.close,
                        broker=broker,
                    )
                    risk.clear_partial_profit(symbol)
                    risk.clear_trailing_peak(symbol)
                    return
                order = broker.submit_order(
                    symbol, "SELL", position_qty,
                    order_type="market",
                    max_retries=settings.max_api_retries,
                )
                risk.clear_partial_profit(symbol)
                risk.clear_trailing_peak(symbol)
                journal.write("partial_remainder_exit", {
                    **preview_payload,
                    "order_id": str(getattr(order, "id", "")),
                    "symbol": symbol, "qty": position_qty, "reason": exit_reason,
                })
                notifier.notify(
                    "trade",
                    f"Partial remainder SELL {position_qty} {symbol}: {exit_reason}",
                    preview_payload,
                )
            except Exception as exc:
                logger.warning("Partial remainder exit failed for %s: %s", symbol, exc)
            return

    # Trailing stop check — force SELL if trailing stop breached
    # Skipped for symbols where partial profit was taken (they have their own exit rules above)
    if position_qty > 0 and settings.trailing_stop_pct > 0 and not risk.has_partial_profit_taken(symbol):
        risk.update_trailing_peak(symbol, signal.close)
        if risk.should_trail_stop(symbol, signal.close):
            logger.info(
                "Trailing stop triggered for %s at %.2f (peak %.2f, stop %.1f%%)",
                symbol, signal.close,
                risk._trailing_peaks.get(symbol, 0),
                settings.trailing_stop_pct,
            )
            journal.write("trailing_stop_triggered", {"symbol": symbol, "price": signal.close})
            notifier.notify("trade", f"Trailing stop triggered: SELL {symbol} at ${signal.close:.2f}")
            # Force sell
            effective_signal = "SELL"
        else:
            # Apply sentiment filter if enabled
            effective_signal = _apply_sentiment(signal.signal, symbol, settings, journal, logger)
    else:
        effective_signal = _apply_sentiment(signal.signal, symbol, settings, journal, logger)

    if position_qty > 0 and settings.max_symbol_loss_pct > 0:
        pos_detail = broker.position_details(symbol)
        loss_pct = None
        if pos_detail and pos_detail.get("unrealized_plpc") is not None:
            loss_pct = float(pos_detail["unrealized_plpc"]) * 100.0
        if loss_pct is not None and loss_pct <= -settings.max_symbol_loss_pct:
            effective_signal = "SELL"
            reason = f"symbol loss {loss_pct:.2f}% <= -{settings.max_symbol_loss_pct:.2f}%"
            logger.warning("Symbol loss stop triggered for %s: %s", symbol, reason)
            journal.write("symbol_loss_stop_triggered", {"symbol": symbol, "reason": reason})
            notifier.notify("risk_reject", f"Symbol loss stop [{symbol}]: {reason}")

    action = "HOLD"
    requested_qty = settings.max_shares
    if effective_signal == "BUY" and position_qty == 0:
        if not market_is_open:
            logger.info("Market closed — skipping BUY %s", symbol)
            journal.write("market_closed_skip", {"symbol": symbol, "signal": "BUY"})
            return
        if settings.max_buys_per_cycle > 0 and cycle_state is not None:
            buys_so_far = int(cycle_state.get("buys", 0) or 0)
            if buys_so_far >= settings.max_buys_per_cycle:
                reason = f"max buys per cycle reached ({buys_so_far}/{settings.max_buys_per_cycle})"
                logger.info("Cycle buy cap blocked BUY %s: %s", symbol, reason)
                journal.write("risk_reject", {"symbol": symbol, "action": "BUY", "reason": reason})
                return
        action = "BUY"

        if settings.block_buy_gap_up_pct > 0 and len(bars) >= 2:
            prior_close = float(bars["close"].iloc[-2])
            current_price = float(signal.close)
            if prior_close > 0 and current_price > prior_close:
                gap_up_pct = (current_price - prior_close) / prior_close * 100.0
                if gap_up_pct >= settings.block_buy_gap_up_pct:
                    reason = f"gap-up chase block {gap_up_pct:.2f}% >= {settings.block_buy_gap_up_pct:.2f}%"
                    logger.info("Gap-up chase block for BUY %s: %s", symbol, reason)
                    journal.write("gap_chase_reject", {"symbol": symbol, "reason": reason})
                    return

        # ── Macro filters (free; yfinance) ────────────────────────────────
        if settings.use_spy_trend_filter:
            from ai_trading.strategy.market_filters import spy_trend_ok
            ok, reason = spy_trend_ok(window=settings.spy_trend_window)
            if not ok:
                logger.info("SPY trend filter blocked BUY %s: %s", symbol, reason)
                journal.write("spy_trend_reject", {"symbol": symbol, "reason": reason})
                return

        vix_mult = 1.0
        if settings.use_vix_size_scaling:
            from ai_trading.strategy.market_filters import vix_size_multiplier
            vix_mult, vix_reason = vix_size_multiplier(
                full_below=settings.vix_full_below,
                half_above=settings.vix_half_above,
                zero_above=settings.vix_zero_above,
            )
            journal.write("vix_scale", {"symbol": symbol, "multiplier": round(vix_mult, 3), "reason": vix_reason})
            if vix_mult <= 0:
                logger.info("VIX scaling blocked BUY %s: %s", symbol, vix_reason)
                return

        if settings.earnings_blackout_days > 0:
            from ai_trading.strategy.market_filters import in_earnings_blackout
            ec = in_earnings_blackout(symbol, blackout_days=settings.earnings_blackout_days)
            if ec.blocked:
                logger.info("Earnings blackout blocked BUY %s: %s", symbol, ec.reason)
                journal.write("earnings_blackout_reject", {"symbol": symbol, "reason": ec.reason})
                return

        if settings.use_volume_confirmation:
            from ai_trading.strategy.market_filters import volume_confirms
            ok, reason = volume_confirms(bars, min_ratio=settings.volume_min_ratio)
            if not ok:
                logger.info("Volume filter blocked BUY %s: %s", symbol, reason)
                journal.write("volume_reject", {"symbol": symbol, "reason": reason})
                return

        # Meta-label filter: skip trade if model says it's unlikely to win
        meta_prob = 1.0
        if settings.use_meta_label:
            try:
                from ai_trading.ml.meta_label import MetaModel
                import os as _os
                if _os.path.exists(settings.meta_model_path):
                    mm = MetaModel.load(settings.meta_model_path)
                    meta_prob = mm.predict_proba_win(bars)
                    journal.write("meta_label", {"symbol": symbol, "prob_win": round(meta_prob, 4)})
                    if meta_prob < settings.meta_min_prob:
                        logger.info("Meta-label rejected %s (p=%.3f < %.3f)",
                                    symbol, meta_prob, settings.meta_min_prob)
                        journal.write("meta_label_reject", {"symbol": symbol, "prob": meta_prob})
                        return
            except Exception as exc:
                logger.warning("Meta-label inference failed for %s: %s", symbol, exc)

        # Correlation filter: block if too correlated with existing positions
        if settings.correlation_filter_threshold > 0:
            existing_positions = [p["symbol"] for p in broker.all_positions()]
            existing_with_bars = [s for s in existing_positions if s in all_bars]
            if existing_with_bars:
                blocked, reason = is_too_correlated(
                    new_symbol=symbol,
                    existing_symbols=existing_with_bars,
                    bars_by_symbol=all_bars,
                    threshold=settings.correlation_filter_threshold,
                )
                if blocked:
                    logger.info("Correlation filter blocked BUY %s: %s", symbol, reason)
                    journal.write("correlation_reject", {"symbol": symbol, "reason": reason})
                    notifier.notify("risk_reject", f"Correlation filter: {reason}")
                    return

        # Correlation-aware size scaling (soft alternative to the hard filter above)
        size_multiplier = 1.0
        if settings.use_correlation_scaling:
            existing_positions = [p["symbol"] for p in broker.all_positions()]
            existing_with_bars = [s for s in existing_positions if s in all_bars]
            if existing_with_bars:
                size_multiplier = correlation_scale(
                    new_symbol=symbol,
                    existing_symbols=existing_with_bars,
                    bars_by_symbol=all_bars,
                    soft_threshold=settings.correlation_scale_soft,
                    hard_threshold=settings.correlation_scale_hard,
                )
                if size_multiplier < 1.0:
                    journal.write("correlation_scale", {"symbol": symbol, "multiplier": round(size_multiplier, 3)})

        # ATR-based risk sizing takes precedence over Kelly when enabled
        if settings.use_atr_stops:
            sz = compute_atr_stop_and_size(
                bars,
                entry=signal.close,
                equity=float(account["equity"]),
                risk_pct=settings.risk_per_trade_pct,
                atr_period=settings.atr_period,
                atr_mult=settings.atr_stop_mult,
                max_shares=settings.max_shares,
            )
            requested_qty = max(1, int(sz.qty * size_multiplier)) if sz.qty > 0 else 0
            journal.write("atr_sizing", {
                "symbol": symbol, "qty": requested_qty,
                "stop_price": round(sz.stop_price, 2), "atr": round(sz.atr_value, 3),
                "risk_per_share": round(sz.risk_per_share, 3),
                "size_multiplier": round(size_multiplier, 3),
            })
        elif settings.use_vol_targeting:
            from ai_trading.risk.portfolio_sizing import vol_targeted_qty
            requested_qty = vol_targeted_qty(
                bars=bars, entry=signal.close, equity=float(account["equity"]),
                target_vol_pct=settings.target_vol_pct,
                max_position_pct=settings.max_position_pct,
                max_shares=settings.max_shares,
            )
            requested_qty = max(0, int(requested_qty * size_multiplier))
            journal.write("vol_target_sizing", {
                "symbol": symbol, "qty": requested_qty,
                "target_vol_pct": settings.target_vol_pct,
                "size_multiplier": round(size_multiplier, 3),
            })
        elif settings.use_kelly_sizing:
            base_qty = risk.kelly_qty(
                win_rate=0.52,   # Conservative default; replace with historical win rate
                avg_win=0.015,
                avg_loss=0.01,
                equity=float(account["equity"]),
                price=signal.close,
            )
            requested_qty = max(0, int(base_qty * size_multiplier))
        else:
            requested_qty = max(0, int(settings.max_shares * size_multiplier))

        # Apply VIX size scaling (after any sizing method)
        if settings.use_vix_size_scaling and vix_mult < 1.0:
            requested_qty = max(0, int(requested_qty * vix_mult))

        # Optional: scale by meta-label win probability
        if settings.use_meta_label and settings.meta_size_scale:
            requested_qty = max(1, int(requested_qty * meta_prob)) if requested_qty > 0 else 0

        if (
            requested_qty <= 0
            and settings.broker == "robinhood"
            and settings.robinhood_use_dollar_orders
            and settings.robinhood_dollar_amount_per_trade > 0
        ):
            requested_qty = 1
            journal.write(
                "robinhood_fractional_sizing",
                {
                    "symbol": symbol,
                    "dollar_amount": settings.robinhood_dollar_amount_per_trade,
                    "nominal_qty": requested_qty,
                    "reason": "fractional dollar-order mode",
                },
            )

        # Portfolio heat cap: total $-at-risk across open positions
        if settings.use_atr_stops and settings.max_portfolio_heat_pct > 0:
            from ai_trading.risk.portfolio_sizing import portfolio_heat_check
            new_risk = requested_qty * sz.risk_per_share if 'sz' in locals() else 0.0
            # Approximate existing heat: positions × ATR stop distance
            existing = {}
            for p in broker.all_positions():
                sp = p["symbol"]
                if sp == symbol or sp not in all_bars:
                    continue
                try:
                    p_sz = compute_atr_stop_and_size(
                        all_bars[sp], entry=float(p.get("current_price", p.get("avg_entry_price", 0)) or 0),
                        equity=float(account["equity"]),
                        risk_pct=settings.risk_per_trade_pct,
                        atr_period=settings.atr_period, atr_mult=settings.atr_stop_mult,
                        max_shares=settings.max_shares,
                    )
                    existing[sp] = float(p.get("qty", 0)) * p_sz.risk_per_share
                except Exception:
                    continue
            heat = portfolio_heat_check(
                open_risks=existing, new_symbol=symbol,
                new_dollar_risk=new_risk, equity=float(account["equity"]),
                max_heat_pct=settings.max_portfolio_heat_pct,
            )
            if not heat.allowed:
                logger.info("Portfolio heat blocked BUY %s: %s", symbol, heat.reason)
                journal.write("heat_reject", {"symbol": symbol, "reason": heat.reason,
                                              "current_pct": round(heat.current_heat_pct, 2),
                                              "projected_pct": round(heat.projected_heat_pct, 2)})
                return


    elif effective_signal == "SELL" and position_qty > 0:
        action = "SELL"
        requested_qty = position_qty

    if action == "HOLD":
        logger.info("No action for %s: signal=%s position=%s", symbol, signal.signal, position_qty)
        journal.write("decision", {"symbol": symbol, "action": "HOLD", "reason": "signal/position rules"})
        risk.clear_error_streak()
        return

    if action == "BUY":
        available_cash = max(
            0.0,
            min(
                float(account.get("cash", 0.0) or 0.0),
                float(account.get("buying_power", account.get("cash", 0.0)) or 0.0),
            ),
        )
        if settings.broker == "robinhood" and settings.robinhood_use_dollar_orders:
            estimated_notional = settings.robinhood_dollar_amount_per_trade
        else:
            estimated_notional = float(signal.close) * int(requested_qty)
        cash_buffer = max(1.0, settings.min_cash_threshold)
        if requested_qty <= 0:
            reason = "position sizing produced zero shares"
            logger.info("Affordability gate blocked BUY %s: %s", symbol, reason)
            journal.write("risk_reject", {"symbol": symbol, "action": action, "reason": reason})
            return
        if estimated_notional + cash_buffer > available_cash:
            reason = (
                f"estimated notional ${estimated_notional:.2f} + buffer ${cash_buffer:.2f} "
                f"> available cash ${available_cash:.2f}"
            )
            logger.info("Affordability gate blocked BUY %s: %s", symbol, reason)
            journal.write("risk_reject", {"symbol": symbol, "action": action, "reason": reason})
            return

    if settings.kill_switch:
        reason = "BOT_KILL_SWITCH=true"
        logger.warning("Kill switch blocked %s %s", action, symbol)
        journal.write("risk_reject", {"symbol": symbol, "action": action, "reason": reason})
        notifier.notify(
            "risk_reject",
            f"Kill switch blocked [{symbol}]: {reason}",
            {"symbol": symbol, "action": action, "reason": reason},
        )
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
        logger.info("Risk manager rejected %s %s: %s", action, symbol, decision.reason)
        journal.write("risk_reject", {"symbol": symbol, "action": action, "reason": decision.reason})
        notifier.notify("risk_reject", f"Order rejected [{symbol}]: {decision.reason}")
        return

    # Drawdown alert (informational — doesn't block, halt is handled in evaluate())
    if settings.portfolio_drawdown_halt_pct > 0 and risk._peak_equity > 0:
        equity = float(account["equity"])
        drawdown = (risk._peak_equity - equity) / risk._peak_equity * 100.0
        if drawdown >= settings.portfolio_drawdown_halt_pct * 0.75:  # warn at 75% of limit
            notifier.send_drawdown_alert(drawdown, equity, risk._peak_equity)

    limit_price = None
    if settings.order_type == "limit":
        offset_mult = 1 + (settings.limit_price_offset_pct / 100.0)
        if action == "BUY":
            limit_price = signal.close * offset_mult
        else:
            limit_price = signal.close * (1 - settings.limit_price_offset_pct / 100.0)

    ready, preview_payload = _prepare_order_attempt(
        settings=settings,
        journal=journal,
        notifier=notifier,
        logger=logger,
        symbol=symbol,
        action=action,
        qty=decision.approved_qty,
        price=signal.close,
        reason="risk-approved signal",
        order_type=settings.order_type,
        limit_price=limit_price,
        latest_price=latest_price,
    )
    if not ready:
        return

    if settings.stock_dry_run:
        _record_stock_dry_run(
            settings=settings,
            journal=journal,
            notifier=notifier,
            logger=logger,
            symbol=symbol,
            action=action,
            qty=decision.approved_qty,
            reason="risk-approved signal",
            order_type=settings.order_type,
            limit_price=limit_price,
            price=signal.close,
            broker=broker,
        )
        if action == "BUY" and cycle_state is not None:
            cycle_state["buys"] = int(cycle_state.get("buys", 0) or 0) + 1
        risk.clear_error_streak()
        return

    order = broker.submit_order(
        symbol,
        action,
        decision.approved_qty,
        order_type=settings.order_type,
        limit_price=limit_price,
        max_retries=settings.max_api_retries,
    )

    risk.register_trade(now.date())
    risk.clear_error_streak()
    if action == "BUY" and cycle_state is not None:
        cycle_state["buys"] = int(cycle_state.get("buys", 0) or 0) + 1

    # Update trailing peak on BUY
    if action == "BUY":
        risk.update_trailing_peak(symbol, signal.close)
    elif action == "SELL":
        risk.clear_trailing_peak(symbol)
        risk.clear_partial_profit(symbol)

    order_record = {
        **preview_payload,
        "symbol": symbol,
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
        action, order.id, decision.approved_qty,
    )
    journal.write("order", order_record)
    notifier.notify(
        "trade",
        f"{'LIVE' if settings.is_live else 'PAPER'} {action} {decision.approved_qty} {symbol} @ ${signal.close:.2f}",
        order_record,
    )

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

        if (
            (settings.stop_loss_pct > 0 or settings.use_atr_stops)
            and action == "BUY"
            and fill_result["status"] == "filled"
            and fill_result["filled_avg_price"] > 0
        ):
            entry_px = fill_result["filled_avg_price"]
            if settings.use_atr_stops:
                sz = compute_atr_stop_and_size(
                    bars, entry=entry_px, equity=float(account["equity"]),
                    risk_pct=settings.risk_per_trade_pct,
                    atr_period=settings.atr_period, atr_mult=settings.atr_stop_mult,
                    max_shares=settings.max_shares,
                )
                stop_price = sz.stop_price
                stop_reason = f"atr*{settings.atr_stop_mult} (ATR={sz.atr_value:.2f})"
            else:
                stop_price = entry_px * (1 - settings.stop_loss_pct / 100.0)
                stop_reason = f"{settings.stop_loss_pct}% fixed"
            try:
                sl_order = broker.submit_stop_loss(symbol, fill_result["filled_qty"], stop_price)
                journal.write(
                    "stop_loss",
                    {"order_id": str(sl_order.id), "stop_price": stop_price,
                     "qty": fill_result["filled_qty"], "reason": stop_reason},
                )
            except Exception as sl_exc:
                logger.warning("Failed to place stop-loss for %s: %s", symbol, sl_exc)


def _ensemble_signal(bars, symbol: str, settings, journal, logger):
    """Generate signal using the ensemble strategy with error isolation."""
    from ai_trading.strategy.ensemble import compute_ensemble_signal
    from ai_trading.strategy.moving_average import SignalResult

    # Get ML probability if available
    ml_probability = None
    try:
        from ai_trading.ml.predict_direction import predict_probability
        ml_probability = predict_probability(symbol, settings.ml_model_path)
    except Exception as exc:
        logger.debug("ML probability unavailable for %s: %s", symbol, exc)

    try:
        ensemble = compute_ensemble_signal(bars, ml_probability=ml_probability)
        journal.write("ensemble_signal", {
            "symbol": symbol,
            "signal": ensemble.signal,
            "strength": ensemble.strength,
            "confidence": ensemble.confidence,
            "regime": ensemble.regime.value,
            "consensus_count": ensemble.consensus_count,
            "weights": ensemble.weights_used,
        })
        close = float(bars["close"].iloc[-1])
        # Return as SignalResult for compatibility with rest of bot logic
        return SignalResult(ensemble.signal, close, 0.0, 0.0)
    except Exception as exc:
        logger.warning("Ensemble strategy failed for %s, falling back to MA: %s", symbol, exc)
        from ai_trading.strategy.moving_average import moving_average_signal
        return moving_average_signal(bars, settings.fast_ma, settings.slow_ma)


def _check_ml_model_staleness(settings, notifier, logger):
    """Alert if the ML model file is older than the configured threshold."""
    model_path = Path(settings.ml_model_path)
    if not model_path.exists():
        logger.info("ML model not found at %s — staleness check skipped", model_path)
        return
    try:
        import time as _time
        age_days = (_time.time() - model_path.stat().st_mtime) / 86400.0
        if age_days > settings.ml_model_max_age_days:
            msg = (
                f"⚠️ ML model is stale: {age_days:.1f} days old "
                f"(threshold: {settings.ml_model_max_age_days} days). "
                f"Consider retraining."
            )
            logger.warning(msg)
            notifier.notify("error", msg)
    except Exception as exc:
        logger.debug("ML staleness check failed: %s", exc)


def _apply_sentiment(signal_value: str, symbol: str, settings: Settings, journal: Journal, logger) -> str:
    """Apply sentiment filter and return effective signal."""
    if not settings.use_sentiment_filter or signal_value == "HOLD":
        return signal_value
    keywords = [k.strip() for k in settings.news_keywords.split(",") if k.strip()] or None
    sentiment_result = apply_sentiment_filter(
        signal=signal_value,
        symbol=symbol,
        buy_threshold=settings.sentiment_buy_threshold,
        sell_threshold=settings.sentiment_sell_threshold,
        provider=settings.news_provider,
        api_key=settings.news_api_key,
        keywords=keywords,
    )
    journal.write(
        "sentiment_filter",
        {
            "symbol": symbol,
            "original_signal": sentiment_result.original_signal,
            "filtered_signal": sentiment_result.filtered_signal,
            "sentiment_score": sentiment_result.sentiment_score,
            "blocked": sentiment_result.blocked,
            "reason": sentiment_result.reason,
        },
    )
    if sentiment_result.blocked:
        logger.info("Sentiment filter blocked %s for %s: %s", signal_value, symbol, sentiment_result.reason)
    return sentiment_result.filtered_signal


def run_once(symbol_override: str | None = None, skip_confirmation: bool = False) -> None:
    settings = Settings.from_env()
    if symbol_override:
        settings.symbol = symbol_override.upper()
        settings.symbols = settings.symbol
    if skip_confirmation:
        settings.require_confirmation = False
    settings.validate()

    logger = configure_logging(settings.log_path)
    journal = Journal(settings.journal_path)
    notifier = Notifier(settings.webhook_url, settings.notify_events)

    # Configure sentiment cache TTL
    if settings.sentiment_cache_ttl_sec > 0:
        configure_sentiment_cache(settings.sentiment_cache_ttl_sec)

    if settings.is_live:
        logger.warning(LIVE_WARNING)
        journal.write("mode", {"mode": "LIVE", "warning": LIVE_WARNING})
    else:
        logger.info(PAPER_NOTICE)
    if settings.stock_dry_run:
        logger.info("Stock dry-run mode is enabled. Stock orders will be logged, not submitted.")
        journal.write("mode", {"stock_dry_run": True})

    broker = create_broker(settings)
    market_data = AlpacaMarketData(
        settings.api_key, settings.api_secret,
        cache_ttl_sec=settings.data_cache_ttl_sec,
        data_feed=settings.market_data_feed,
    )

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
        portfolio_drawdown_halt_pct=settings.portfolio_drawdown_halt_pct,
        use_kelly_sizing=settings.use_kelly_sizing,
        kelly_fraction=settings.kelly_fraction,
        kelly_max_shares=settings.kelly_max_shares,
        kelly_win_rate=settings.kelly_win_rate,
        kelly_avg_win=settings.kelly_avg_win,
        kelly_avg_loss=settings.kelly_avg_loss,
        trailing_stop_pct=settings.trailing_stop_pct,
        error_streak_decay_hours=settings.error_streak_decay_hours,
        partial_profit_max_hold_bars=settings.partial_profit_max_hold_bars,
        partial_profit_trailing_stop_pct=settings.partial_profit_trailing_stop_pct,
        state_file=settings.risk_state_file,
    )

    now = datetime.now(timezone.utc)

    try:
        if not _preflight_checks(settings, broker, logger):
            journal.write("preflight_failed", {"reason": "account not ready"})
            notifier.notify("error", "Pre-flight check failed: account not ready")
            return

        cfg_symbols = settings.get_symbols()

        # Market-hours gate: when the market is closed, still fetch latest prices.
        # Order flow remains disabled until regular market hours resume.
        market_is_open = broker.is_market_open()
        if not market_is_open:
            logger.info("Market is closed — recording latest prices and skipping order cycle.")
            _record_current_prices(cfg_symbols, broker, journal, logger)
            return

        account = broker.account_state()
        journal.write("account_state", account)

        # Record start-of-day equity for accurate daily loss tracking
        risk.set_start_of_day_equity(float(account["equity"]))

        # ML model staleness alert
        if settings.ml_model_max_age_days > 0:
            _check_ml_model_staleness(settings, notifier, logger)

        # If no explicit symbol list configured, pull the full tradable universe
        # from Alpaca (NYSE + NASDAQ + ARCA + BATS + AMEX, clean tickers only).
        if len(cfg_symbols) == 1 and cfg_symbols[0] == settings.symbol.upper() and not settings.symbols:
            symbols = broker.get_all_tradable_symbols()
        else:
            symbols = cfg_symbols
        logger.info("Trading %d symbols", len(symbols))

        # EOD forced close: if within N minutes of market close, liquidate all positions
        if settings.close_before_eod > 0:
            mins_left = broker.minutes_to_close()
            if 0 < mins_left <= settings.close_before_eod:
                logger.info(
                    "EOD close triggered: %.1f min until close (threshold %d min)",
                    mins_left, settings.close_before_eod,
                )
                for pos in broker.all_positions():
                    sym = pos["symbol"]
                    try:
                        if settings.stock_dry_run:
                            _record_stock_dry_run(
                                settings=settings,
                                journal=journal,
                                notifier=notifier,
                                logger=logger,
                                symbol=sym,
                                action="SELL",
                                qty=pos["qty"],
                                reason=f"EOD close ({mins_left:.1f} min to close)",
                                broker=broker,
                            )
                            continue
                        broker.close_position(sym)
                        risk.clear_trailing_peak(sym)
                        journal.write("order", {"symbol": sym, "action": "SELL",
                                                "qty": pos["qty"], "reason": "EOD close",
                                                "mode": "PAPER" if settings.paper_only else "LIVE"})
                        logger.info("EOD closed %s (%s shares)", sym, pos["qty"])
                        notifier.notify("trade", f"EOD close: SELL {pos['qty']} {sym} ({mins_left:.0f} min to close)")
                    except Exception as exc:
                        logger.error("EOD close failed for %s: %s", sym, exc)
                return  # skip normal signal logic today

        # Optionally retrain ML model before trading
        for sym in symbols:
            _maybe_retrain_ml(settings, sym, logger)

        # Fetch bars for all symbols (used by correlation filter too)
        all_bars: dict = {}
        cycle_state: dict = {"buys": 0}

        for symbol in symbols:
            try:
                _trade_symbol(
                    symbol=symbol,
                    settings=settings,
                    broker=broker,
                    market_data=market_data,
                    risk=risk,
                    journal=journal,
                    notifier=notifier,
                    logger=logger,
                    account=account,
                    all_bars=all_bars,
                    now=now,
                    market_is_open=market_is_open,
                    cycle_state=cycle_state,
                )
            except OrderError as exc:
                risk.register_error()
                logger.error("Order error [%s]: %s", symbol, exc)
                journal.write("order_error", {"symbol": symbol, "error": str(exc)})
                notifier.notify("error", f"Order error [{symbol}]: {exc}")
            except Exception as exc:
                risk.register_error()
                logger.exception("Error trading %s: %s", symbol, exc)
                journal.write("error", {"symbol": symbol, "error": str(exc)})
                notifier.notify("error", f"Bot error [{symbol}]: {exc}")

        # Daily summary notification
        if "daily_summary" in settings.notify_events:
            equity = float(account["equity"])
            last_equity = float(account.get("last_equity", equity))
            positions = broker.all_positions()
            notifier.send_daily_summary(
                date=now.strftime("%Y-%m-%d"),
                equity=equity,
                last_equity=last_equity,
                positions=positions,
                trades_today=risk._trades_today,
                pnl_today=equity - last_equity,
            )

        # Options cycle (optional) — runs after equity trading
        if getattr(settings, "options_enabled", False):
            try:
                from ai_trading.options.integration import run_options_cycle
                # Only run options on configured symbols (not full universe).
                opt_symbols = cfg_symbols if cfg_symbols else [settings.symbol.upper()]
                run_options_cycle(
                    settings=settings,
                    candidate_symbols=opt_symbols,
                    journal=journal,
                    notifier=notifier,
                    logger=logger,
                )
            except Exception as exc:
                logger.exception("Options cycle failed: %s", exc)
                journal.write("option_cycle_error", {"error": str(exc)})

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
