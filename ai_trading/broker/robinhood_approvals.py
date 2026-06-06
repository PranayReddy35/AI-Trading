from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_trading.broker.robinhood_health import mask_account_number


def approval_path(default: str = "logs/robinhood_approvals.jsonl") -> Path:
    return Path(default)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace("$", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _quantity_from_text(value: Any) -> str | None:
    raw = str(value or "")
    patterns = [
        r"~\s*([0-9]+(?:\.[0-9]+)?)\s+shares",
        r"all\s+([0-9]+(?:\.[0-9]+)?)\s+shares",
        r"([0-9]+(?:\.[0-9]+)?)\s+share",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            qty = _float_or_none(match.group(1))
            if qty and qty > 0:
                return f"{qty:.6f}".rstrip("0").rstrip(".")
    return None


def build_order_payload(
    row_payload: dict[str, Any],
    *,
    account_number: str,
    dollar_amount_per_trade: float = 0.0,
) -> dict[str, str]:
    """Build a Robinhood MCP order payload from an approved dashboard row.

    The payload intentionally mirrors review_equity_order/place_equity_order,
    except the caller may remove account_number before writing less-sensitive logs.
    """
    symbol = str(row_payload.get("symbol") or "").upper()
    action = str(row_payload.get("action") or "").upper()
    if not symbol:
        raise ValueError("Approval payload is missing symbol.")
    if action not in {"BUY", "SELL"}:
        raise ValueError(f"Approval action must be BUY or SELL, got {action!r}.")

    order: dict[str, str] = {
        "account_number": account_number,
        "symbol": symbol,
        "side": action.lower(),
        "type": str(row_payload.get("order_type") or "market").lower(),
        "time_in_force": str(row_payload.get("time_in_force") or "gfd").lower(),
        "market_hours": str(row_payload.get("market_hours") or "regular_hours").lower(),
    }

    limit_price = _float_or_none(row_payload.get("limit_price"))
    if order["type"] in {"limit", "stop_limit"}:
        if not limit_price:
            raise ValueError("Limit/stop-limit approvals require limit_price.")
        order["limit_price"] = f"{limit_price:.2f}"

    stop_price = _float_or_none(row_payload.get("stop_price"))
    if order["type"] in {"stop_market", "stop_limit"}:
        if not stop_price:
            raise ValueError("Stop approvals require stop_price.")
        order["stop_price"] = f"{stop_price:.2f}"

    if action == "BUY":
        dollars = _float_or_none(row_payload.get("dollar_amount") or row_payload.get("estimated_spend"))
        if not dollars and dollar_amount_per_trade > 0:
            dollars = dollar_amount_per_trade
        if dollars and dollars > 0 and order["type"] == "market":
            order["dollar_amount"] = f"{dollars:.2f}"
        else:
            qty = (
                _float_or_none(row_payload.get("quantity"))
                or _float_or_none(row_payload.get("suggested_qty"))
                or _float_or_none(row_payload.get("qty"))
            )
            if not qty or qty <= 0:
                raise ValueError("Buy approval requires dollar_amount or quantity.")
            order["quantity"] = f"{qty:.6f}".rstrip("0").rstrip(".")
    else:
        qty_text = row_payload.get("quantity") or row_payload.get("suggested") or row_payload.get("sell_quantity")
        qty = _float_or_none(qty_text)
        if not qty:
            parsed = _quantity_from_text(qty_text)
            if parsed:
                order["quantity"] = parsed
        else:
            order["quantity"] = f"{qty:.6f}".rstrip("0").rstrip(".")
        if "quantity" not in order:
            raise ValueError("Sell approval requires a concrete quantity.")

    return order


def approval_record(
    row_payload: dict[str, Any],
    *,
    account_number: str,
    dollar_amount_per_trade: float = 0.0,
    approved_by: str = "dashboard",
    status: str = "approved_pending_execution",
) -> dict[str, Any]:
    order = build_order_payload(
        row_payload,
        account_number=account_number,
        dollar_amount_per_trade=dollar_amount_per_trade,
    )
    redacted_order = dict(order)
    redacted_order["account_number"] = mask_account_number(account_number)
    approval_id = str(uuid.uuid4())
    return {
        "approval_id": approval_id,
        "ref_id": approval_id,
        "created_at": _now(),
        "updated_at": _now(),
        "status": status,
        "approved_by": approved_by,
        "order": redacted_order,
        "executor_order": {k: v for k, v in order.items() if k != "account_number"},
        "account_number_masked": mask_account_number(account_number),
        "source_payload": row_payload,
        "executor": "robinhood_agentic_connector_required",
        "execution_status": "pending",
    }


def write_approval(path: str | Path, record: dict[str, Any]) -> dict[str, Any]:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    return record


def load_approvals(path: str | Path) -> list[dict[str, Any]]:
    in_path = Path(path)
    if not in_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for raw in in_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            records.append(payload)
    return records


def latest_approvals(path: str | Path) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for record in load_approvals(path):
        approval_id = str(record.get("approval_id") or "")
        if not approval_id:
            continue
        if approval_id not in by_id:
            order.append(approval_id)
        by_id[approval_id] = record
    return [by_id[approval_id] for approval_id in order if approval_id in by_id]


def get_approval(path: str | Path, approval_id: str) -> dict[str, Any] | None:
    for record in reversed(load_approvals(path)):
        if str(record.get("approval_id") or "") == approval_id:
            return record
    return None


def pending_approvals(path: str | Path) -> list[dict[str, Any]]:
    return [
        record for record in latest_approvals(path)
        if str(record.get("execution_status") or "").lower() == "pending"
        and str(record.get("status") or "").lower().startswith("approved")
    ]


def update_approval_status(
    path: str | Path,
    approval_id: str,
    *,
    execution_status: str,
    status: str | None = None,
    note: str = "",
    order_id: str = "",
    execution_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current = get_approval(path, approval_id)
    if current is None:
        raise ValueError(f"No approval found for approval_id={approval_id}.")
    updated = dict(current)
    updated["updated_at"] = _now()
    updated["execution_status"] = execution_status
    if status:
        updated["status"] = status
    if note:
        updated["execution_note"] = note
    if order_id:
        updated["order_id"] = order_id
    if execution_result:
        updated["execution_result"] = execution_result
    return write_approval(path, updated)


def review_order_payload(record: dict[str, Any], *, account_number: str) -> dict[str, str]:
    order = {**record.get("executor_order", {})}
    order["account_number"] = account_number
    order.pop("ref_id", None)
    return {key: str(value) for key, value in order.items() if value is not None}


def place_order_payload(record: dict[str, Any], *, account_number: str) -> dict[str, str]:
    order = {**record.get("executor_order", {})}
    order["account_number"] = account_number
    order["ref_id"] = str(record.get("ref_id") or record.get("approval_id") or uuid.uuid4())
    return {key: str(value) for key, value in order.items() if value is not None}
