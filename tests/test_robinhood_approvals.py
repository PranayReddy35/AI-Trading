from __future__ import annotations

from ai_trading.broker.robinhood_approvals import (
    approval_record,
    build_order_payload,
    latest_approvals,
    pending_approvals,
    place_order_payload,
    review_order_payload,
    update_approval_status,
    write_approval,
)


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
    assert record["ref_id"] == record["approval_id"]


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


def test_update_approval_status_removes_from_pending(tmp_path) -> None:
    path = tmp_path / "approvals.jsonl"
    record = approval_record(
        {"symbol": "AAPL", "action": "BUY"},
        account_number="123456789",
        dollar_amount_per_trade=25.0,
    )
    write_approval(path, record)

    update_approval_status(path, record["approval_id"], execution_status="executed", status="executed")

    assert pending_approvals(path) == []
    latest = latest_approvals(path)
    assert len(latest) == 1
    assert latest[0]["execution_status"] == "executed"


def test_review_and_place_payloads_include_account_only_at_execution_time() -> None:
    record = approval_record(
        {"symbol": "AAPL", "action": "BUY"},
        account_number="123456789",
        dollar_amount_per_trade=25.0,
    )

    review_payload = review_order_payload(record, account_number="123456789")
    place_payload = place_order_payload(record, account_number="123456789")

    assert review_payload["account_number"] == "123456789"
    assert "ref_id" not in review_payload
    assert place_payload["account_number"] == "123456789"
    assert place_payload["ref_id"] == record["approval_id"]
