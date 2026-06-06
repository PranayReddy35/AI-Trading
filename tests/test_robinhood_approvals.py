from __future__ import annotations

from ai_trading.broker.robinhood_approvals import approval_record, build_order_payload, pending_approvals, write_approval


def test_build_buy_dollar_market_order_payload() -> None:
    payload = {
        "symbol": "AAPL",
        "action": "BUY",
        "estimated_spend": "$25.00",
    }

    order = build_order_payload(payload, account_number="123456789")

    assert order == {
        "account_number": "123456789",
        "symbol": "AAPL",
        "side": "buy",
        "type": "market",
        "time_in_force": "gfd",
        "market_hours": "regular_hours",
        "dollar_amount": "25.00",
    }


def test_build_buy_uses_configured_dollar_amount_when_missing() -> None:
    payload = {"symbol": "MSFT", "action": "BUY"}

    order = build_order_payload(payload, account_number="123456789", dollar_amount_per_trade=10.0)

    assert order["dollar_amount"] == "10.00"


def test_build_sell_parses_suggested_partial_quantity_before_full_sellable() -> None:
    payload = {
        "symbol": "CCEP",
        "action": "SELL",
        "suggested": "Sell 25% (~0.264508 shares)",
        "sell_quantity": "1.058032",
    }

    order = build_order_payload(payload, account_number="123456789")

    assert order["quantity"] == "0.264508"
    assert order["side"] == "sell"


def test_approval_record_redacts_account_number() -> None:
    record = approval_record(
        {"symbol": "AAPL", "action": "BUY"},
        account_number="123456789",
        dollar_amount_per_trade=25.0,
    )

    assert record["order"]["account_number"] == "*****6789"
    assert "account_number" not in record["executor_order"]
    assert record["execution_status"] == "pending"


def test_write_and_load_pending_approval(tmp_path) -> None:
    path = tmp_path / "approvals.jsonl"
    record = approval_record(
        {"symbol": "AAPL", "action": "BUY"},
        account_number="123456789",
        dollar_amount_per_trade=25.0,
    )

    write_approval(path, record)

    loaded = pending_approvals(path)
    assert len(loaded) == 1
    assert loaded[0]["approval_id"] == record["approval_id"]
