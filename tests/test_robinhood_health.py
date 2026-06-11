from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ai_trading.broker.robinhood_agent import RobinhoodAgenticBroker, create_broker
from ai_trading.broker.robinhood_health import check_readiness, mask_account_number
from ai_trading.broker.robinhood_intents import (
    latest_intent,
    mark_reviewed,
    quote_sanity_check,
    review_payload,
)
from ai_trading.config import Settings


def test_mask_account_number() -> None:
    assert mask_account_number("593473374") == "*****3374"
    assert mask_account_number("") == "(missing)"


def test_check_readiness_requires_agentic_account(monkeypatch) -> None:
    monkeypatch.delenv("ROBINHOOD_AGENTIC_ACCOUNT_NUMBER", raising=False)
    monkeypatch.delenv("ROBINHOOD_AGENTIC_ENABLED", raising=False)

    result = check_readiness()

    assert not result.ok
    assert "ROBINHOOD_AGENTIC_ACCOUNT_NUMBER is not set." in result.messages


def test_check_readiness_passes_with_verified_account(monkeypatch) -> None:
    monkeypatch.setenv("ROBINHOOD_AGENTIC_ACCOUNT_NUMBER", "593473374")
    monkeypatch.setenv("ROBINHOOD_AGENTIC_ENABLED", "true")
    monkeypatch.setenv("BOT_PAPER_ONLY", "true")
    monkeypatch.setenv("BOT_STOCK_DRY_RUN", "true")
    monkeypatch.setenv("BOT_KILL_SWITCH", "true")

    result = check_readiness()

    assert result.ok
    assert "Robinhood Agentic account configured: *****3374" in result.messages


def test_robinhood_settings_validate_requires_snapshot() -> None:
    settings = Settings(
        api_key="alpaca-key",
        api_secret="alpaca-secret",
        broker="robinhood",
        robinhood_agentic_enabled=True,
        robinhood_agentic_account_number="593473374",
    )

    with pytest.raises(ValueError, match="ROBINHOOD_AGENTIC_BUYING_POWER"):
        settings.validate()


def test_create_broker_returns_robinhood_agentic_broker(monkeypatch) -> None:
    class FakeAlpacaBroker:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr("ai_trading.broker.robinhood_agent.AlpacaBroker", FakeAlpacaBroker)
    settings = Settings(
        api_key="alpaca-key",
        api_secret="alpaca-secret",
        broker="robinhood",
        robinhood_agentic_enabled=True,
        robinhood_agentic_account_number="593473374",
        robinhood_agentic_buying_power=100,
        robinhood_agentic_equity=100,
        stock_dry_run=True,
    )

    broker = create_broker(settings)

    assert isinstance(broker, RobinhoodAgenticBroker)
    assert broker.account_state()["broker"] == "robinhood"


def test_robinhood_agentic_broker_prefers_fresh_snapshot_quotes(monkeypatch) -> None:
    class FakeAlpacaBroker:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def get_latest_prices(self, symbols):
            return {symbol: 999.0 for symbol in symbols}

        def get_latest_price(self, symbol):
            return 999.0

    monkeypatch.setattr("ai_trading.broker.robinhood_agent.AlpacaBroker", FakeAlpacaBroker)
    monkeypatch.setattr(
        "ai_trading.broker.robinhood_agent.load_fresh_robinhood_quotes",
        lambda: {"SPY": 501.25},
    )

    broker = RobinhoodAgenticBroker(
        account_number="593473374",
        buying_power=100,
        equity=100,
        alpaca_api_key="alpaca-key",
        alpaca_api_secret="alpaca-secret",
        paper=True,
    )

    prices = broker.get_latest_prices(["SPY", "QQQ"])

    assert prices["SPY"] == 501.25
    assert prices["QQQ"] == 999.0


def test_review_payload_maps_intent_to_mcp_args() -> None:
    payload = review_payload(
        {
            "symbol": "spy",
            "side": "BUY",
            "quantity": "1",
            "type": "limit",
            "limit_price": "500.00",
        },
        "593473374",
    )

    assert payload == {
        "account_number": "593473374",
        "symbol": "SPY",
        "side": "buy",
        "quantity": "1",
        "type": "limit",
        "time_in_force": "gfd",
        "market_hours": "regular_hours",
        "limit_price": "500.00",
    }


def test_review_payload_maps_dollar_order_to_mcp_args() -> None:
    payload = review_payload(
        {
            "symbol": "spy",
            "side": "BUY",
            "dollar_amount": "25.00",
            "type": "market",
        },
        "593473374",
    )

    assert payload == {
        "account_number": "593473374",
        "symbol": "SPY",
        "side": "buy",
        "dollar_amount": "25.00",
        "type": "market",
        "time_in_force": "gfd",
        "market_hours": "regular_hours",
    }


def test_robinhood_dollar_orders_require_market_order() -> None:
    settings = Settings(
        api_key="alpaca-key",
        api_secret="alpaca-secret",
        broker="robinhood",
        robinhood_agentic_enabled=True,
        robinhood_agentic_account_number="593473374",
        robinhood_agentic_buying_power=100,
        robinhood_agentic_equity=100,
        robinhood_use_dollar_orders=True,
        robinhood_dollar_amount_per_trade=25,
        order_type="limit",
        stock_dry_run=True,
    )

    with pytest.raises(ValueError, match="BOT_ORDER_TYPE=market"):
        settings.validate()


def test_robinhood_live_requires_non_intent_execution_mode_when_dry_run_off() -> None:
    settings = Settings(
        api_key="alpaca-key",
        api_secret="alpaca-secret",
        broker="robinhood",
        robinhood_agentic_enabled=True,
        robinhood_agentic_account_number="593473374",
        robinhood_agentic_buying_power=100,
        robinhood_agentic_equity=100,
        stock_dry_run=False,
        robinhood_execution_mode="intent_only",
    )

    with pytest.raises(ValueError, match="ROBINHOOD_EXECUTION_MODE=intent_only"):
        settings.validate()


def test_robinhood_auto_dispatch_requires_auto_dispatch_mode() -> None:
    settings = Settings(
        api_key="alpaca-key",
        api_secret="alpaca-secret",
        broker="robinhood",
        robinhood_agentic_enabled=True,
        robinhood_agentic_account_number="593473374",
        robinhood_agentic_buying_power=100,
        robinhood_agentic_equity=100,
        stock_dry_run=False,
        robinhood_execution_mode="approval_queue",
        robinhood_auto_dispatch=True,
    )

    with pytest.raises(ValueError, match="ROBINHOOD_AUTO_DISPATCH"):
        settings.validate()


def test_partial_profit_remainder_settings_load_from_env(monkeypatch) -> None:
    monkeypatch.setenv("BOT_PARTIAL_PROFIT_TRIGGER_PCT", "1.5")
    monkeypatch.setenv("BOT_PARTIAL_PROFIT_SELL_PCT", "75")
    monkeypatch.setenv("BOT_PARTIAL_PROFIT_MAX_HOLD_BARS", "3")
    monkeypatch.setenv("BOT_PARTIAL_PROFIT_TRAILING_STOP_PCT", "1.0")

    settings = Settings.from_env()

    assert settings.partial_profit_trigger_pct == 1.5
    assert settings.partial_profit_sell_pct == 75
    assert settings.partial_profit_max_hold_bars == 3
    assert settings.partial_profit_trailing_stop_pct == 1.0


def test_latest_intent_skips_reviewed_records(tmp_path) -> None:
    path = tmp_path / "intents.jsonl"
    path.write_text(
        '{"symbol":"AAPL","side":"buy","quantity":"1","reviewed_at":"manual"}\n'
        '{"symbol":"MSFT","side":"buy","quantity":"2"}\n',
        encoding="utf-8",
    )

    record = latest_intent(path)

    assert record is not None
    assert record.line_number == 2
    assert record.payload["symbol"] == "MSFT"


def test_mark_reviewed_updates_line(tmp_path) -> None:
    path = tmp_path / "intents.jsonl"
    path.write_text('{"symbol":"SPY","side":"buy","quantity":"1"}\n', encoding="utf-8")

    payload = mark_reviewed(path, 1, "reviewed via connector")

    assert payload["reviewed_at"] == "manual"
    assert payload["review_note"] == "reviewed via connector"


def test_quote_sanity_check_passes_fresh_active_quote() -> None:
    now = datetime(2026, 6, 5, 16, 10, tzinfo=timezone.utc)
    check = quote_sanity_check(
        {"symbol": "SPY", "reference_price": 100.0},
        {
            "symbol": "SPY",
            "last_trade_price": "100.10",
            "venue_last_trade_time": "2026-06-05T16:09:50Z",
            "bid_price": "100.09",
            "ask_price": "100.11",
            "state": "active",
            "has_traded": True,
        },
        now=now,
    )

    assert check.ok
    assert check.severity == "pass"


def test_quote_sanity_check_blocks_stale_quote() -> None:
    now = datetime(2026, 6, 5, 16, 10, tzinfo=timezone.utc)
    check = quote_sanity_check(
        {"symbol": "SPY", "reference_price": 100.0},
        {
            "symbol": "SPY",
            "last_trade_price": "100.10",
            "venue_last_trade_time": "2026-06-05T16:08:00Z",
            "bid_price": "100.09",
            "ask_price": "100.11",
            "state": "active",
            "has_traded": True,
        },
        now=now,
        max_age_seconds=60,
    )

    assert not check.ok
    assert check.severity == "block"
    assert any("stale" in msg for msg in check.messages)


def test_quote_sanity_check_blocks_price_drift() -> None:
    now = datetime(2026, 6, 5, 16, 10, tzinfo=timezone.utc)
    check = quote_sanity_check(
        {"symbol": "SPY", "reference_price": 100.0},
        {
            "symbol": "SPY",
            "last_trade_price": "102.00",
            "venue_last_trade_time": "2026-06-05T16:09:59Z",
            "bid_price": "101.99",
            "ask_price": "102.01",
            "state": "active",
            "has_traded": True,
        },
        now=now,
        max_price_drift_pct=0.5,
    )

    assert not check.ok
    assert check.severity == "block"
    assert any("Price drift too high" in msg for msg in check.messages)
