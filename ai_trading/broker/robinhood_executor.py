from __future__ import annotations

import argparse
import json
import os
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from ai_trading.broker.robinhood_approvals import (
    get_approval,
    latest_approvals,
    pending_approvals,
    place_order_payload,
    review_order_payload,
    update_approval_status,
)
from ai_trading.config import Settings


def _approval_path(args: argparse.Namespace, settings: Settings) -> str:
    return args.path or os.getenv("ROBINHOOD_APPROVALS_PATH", "logs/robinhood_approvals.jsonl")


def _account_number(settings: Settings) -> str:
    account_number = settings.robinhood_agentic_account_number.strip()
    if not account_number:
        raise ValueError("ROBINHOOD_AGENTIC_ACCOUNT_NUMBER is required for executor payloads.")
    return account_number


def _find_record(path: str, approval_id: str | None) -> dict[str, Any]:
    if approval_id:
        record = get_approval(path, approval_id)
        if record is None:
            raise ValueError(f"No approval found for {approval_id}.")
        return record
    pending = pending_approvals(path)
    if not pending:
        raise ValueError("No pending Robinhood approvals found.")
    return pending[0]


def _print_records(records: list[dict[str, Any]]) -> None:
    if not records:
        print("No approvals found.")
        return
    for record in records:
        order = record.get("order", {})
        amount = order.get("dollar_amount") or order.get("quantity") or "?"
        print(
            f"{record.get('approval_id')} | {record.get('execution_status')} | "
            f"{str(order.get('side', '')).upper()} {amount} {order.get('symbol', '')} | "
            f"{record.get('created_at')}"
        )


def _post_json(url: str, payload: dict[str, Any]) -> bool:
    data = json.dumps(payload, default=str).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "ai-trading-robinhood-executor"},
        method="POST",
    )
    with urlopen(req, timeout=15) as resp:
        return 200 <= resp.status < 300


def _dispatch_webhook(path: str, record: dict[str, Any], url: str) -> int:
    payload = {
        "approval_id": record.get("approval_id"),
        "ref_id": record.get("ref_id"),
        "order": record.get("executor_order", {}),
        "source": "ai_trading.robinhood_executor",
    }
    try:
        ok = _post_json(url, payload)
    except (URLError, OSError) as exc:
        update_approval_status(
            path,
            str(record.get("approval_id")),
            execution_status="dispatch_failed",
            status="approved_dispatch_failed",
            note=str(exc),
        )
        print(f"Webhook dispatch failed: {exc}")
        return 2
    if not ok:
        update_approval_status(
            path,
            str(record.get("approval_id")),
            execution_status="dispatch_failed",
            status="approved_dispatch_failed",
            note="Webhook returned non-2xx response.",
        )
        print("Webhook dispatch failed: non-2xx response.")
        return 2
    update_approval_status(
        path,
        str(record.get("approval_id")),
        execution_status="dispatched",
        status="approved_dispatched",
        note="Sent to configured executor webhook.",
    )
    print(f"Dispatched approval {record.get('approval_id')} to executor webhook.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Process dashboard-approved Robinhood execution requests.")
    parser.add_argument("--path", default=None, help="Approval JSONL path.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_path(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--path", default=None, help="Approval JSONL path.")

    list_cmd = sub.add_parser("list", help="List latest approvals.")
    add_path(list_cmd)
    list_cmd.add_argument("--all", action="store_true", help="Include non-pending approvals.")

    payload_cmd = sub.add_parser("payload", help="Print Robinhood MCP review/place payloads.")
    add_path(payload_cmd)
    payload_cmd.add_argument("approval_id", nargs="?", default=None)
    payload_cmd.add_argument("--kind", choices=["review", "place", "both"], default="both")

    dispatch_cmd = sub.add_parser("dispatch-webhook", help="Dispatch approval to ROBINHOOD_EXECUTOR_WEBHOOK_URL.")
    add_path(dispatch_cmd)
    dispatch_cmd.add_argument("approval_id", nargs="?", default=None)
    dispatch_cmd.add_argument("--url", default=None)

    next_cmd = sub.add_parser("next", help="Print the next pending approval.")
    add_path(next_cmd)

    executed_cmd = sub.add_parser("mark-executed", help="Mark an approval as executed after external placement.")
    add_path(executed_cmd)
    executed_cmd.add_argument("approval_id")
    executed_cmd.add_argument("--order-id", default="")
    executed_cmd.add_argument("--note", default="")

    failed_cmd = sub.add_parser("mark-failed", help="Mark an approval as failed.")
    add_path(failed_cmd)
    failed_cmd.add_argument("approval_id")
    failed_cmd.add_argument("--note", default="")

    args = parser.parse_args()
    settings = Settings.from_env()
    path = _approval_path(args, settings)

    try:
        if args.command == "list":
            records = latest_approvals(path) if args.all else pending_approvals(path)
            _print_records(records)
            return 0

        if args.command == "next":
            record = _find_record(path, None)
            print(json.dumps(record, indent=2, sort_keys=True, default=str))
            return 0

        if args.command == "payload":
            record = _find_record(path, args.approval_id)
            account_number = _account_number(settings)
            if args.kind in {"review", "both"}:
                print("review_equity_order:")
                print(json.dumps(review_order_payload(record, account_number=account_number), indent=2, sort_keys=True))
            if args.kind in {"place", "both"}:
                print("place_equity_order:")
                print(json.dumps(place_order_payload(record, account_number=account_number), indent=2, sort_keys=True))
            return 0

        if args.command == "dispatch-webhook":
            record = _find_record(path, args.approval_id)
            url = args.url or os.getenv("ROBINHOOD_EXECUTOR_WEBHOOK_URL", "").strip()
            if not url:
                print("ROBINHOOD_EXECUTOR_WEBHOOK_URL is required for dispatch-webhook.")
                return 2
            return _dispatch_webhook(path, record, url)

        if args.command == "mark-executed":
            update_approval_status(
                path,
                args.approval_id,
                execution_status="executed",
                status="executed",
                order_id=args.order_id,
                note=args.note,
            )
            print(f"Marked executed: {args.approval_id}")
            return 0

        if args.command == "mark-failed":
            update_approval_status(
                path,
                args.approval_id,
                execution_status="failed",
                status="execution_failed",
                note=args.note,
            )
            print(f"Marked failed: {args.approval_id}")
            return 0
    except ValueError as exc:
        print(str(exc))
        return 2

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
