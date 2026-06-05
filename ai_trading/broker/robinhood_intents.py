from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_trading.config import Settings


@dataclass(frozen=True)
class IntentRecord:
    line_number: int
    payload: dict[str, Any]


@dataclass(frozen=True)
class QuoteCheck:
    ok: bool
    severity: str
    messages: list[str]
    quote_price: float | None = None
    price_drift_pct: float | None = None
    spread_bps: float | None = None


def load_intents(path: str | Path) -> list[IntentRecord]:
    intent_path = Path(path)
    if not intent_path.exists():
        return []

    records: list[IntentRecord] = []
    for idx, raw_line in enumerate(intent_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            records.append(IntentRecord(line_number=idx, payload=payload))
    return records


def latest_intent(path: str | Path) -> IntentRecord | None:
    records = load_intents(path)
    for record in reversed(records):
        if not record.payload.get("reviewed_at"):
            return record
    return records[-1] if records else None


def review_payload(intent: dict[str, Any], account_number: str) -> dict[str, str]:
    payload: dict[str, str] = {
        "account_number": account_number,
        "symbol": str(intent["symbol"]).upper(),
        "side": str(intent["side"]).lower(),
        "type": str(intent.get("type", "market")).lower(),
        "time_in_force": str(intent.get("time_in_force", "gfd")).lower(),
        "market_hours": str(intent.get("market_hours", "regular_hours")).lower(),
    }
    if "quantity" in intent:
        payload["quantity"] = str(intent["quantity"])
    if "dollar_amount" in intent:
        payload["dollar_amount"] = str(intent["dollar_amount"])
    if "limit_price" in intent:
        payload["limit_price"] = str(intent["limit_price"])
    if "stop_price" in intent:
        payload["stop_price"] = str(intent["stop_price"])
    return payload


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    raw = str(value)
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _quote_age_seconds(quote: dict[str, Any], now: datetime | None = None) -> float | None:
    now = now or datetime.now(timezone.utc)
    trade_time = _parse_dt(
        quote.get("venue_last_trade_time")
        or quote.get("last_trade_time")
        or quote.get("timestamp")
    )
    if trade_time is None:
        return None
    return max(0.0, (now.astimezone(timezone.utc) - trade_time).total_seconds())


def quote_sanity_check(
    intent: dict[str, Any],
    quote: dict[str, Any],
    *,
    max_age_seconds: int = 60,
    max_price_drift_pct: float = 0.5,
    max_spread_bps: float = 20.0,
    now: datetime | None = None,
) -> QuoteCheck:
    messages: list[str] = []
    ok = True
    severity = "pass"

    symbol = str(intent.get("symbol", "")).upper()
    quote_symbol = str(quote.get("symbol", "")).upper()
    if symbol and quote_symbol and symbol != quote_symbol:
        return QuoteCheck(
            ok=False,
            severity="block",
            messages=[f"Quote symbol {quote_symbol} does not match intent symbol {symbol}."],
        )

    state = str(quote.get("state", "") or "").lower()
    if state and state != "active":
        ok = False
        severity = "block"
        messages.append(f"Robinhood quote state is {state}, not active.")

    if quote.get("has_traded") is False:
        ok = False
        severity = "block"
        messages.append("Robinhood quote has_traded=false.")

    age = _quote_age_seconds(quote, now=now)
    if age is None:
        ok = False
        severity = "block"
        messages.append("Robinhood quote timestamp is unavailable.")
    elif age > max_age_seconds:
        ok = False
        severity = "block"
        messages.append(f"Robinhood quote is stale: {age:.0f}s old > {max_age_seconds}s.")
    else:
        messages.append(f"Robinhood quote age OK: {age:.0f}s.")

    price = _float_or_none(quote.get("last_trade_price") or quote.get("last_non_reg_trade_price"))
    reference = _float_or_none(intent.get("reference_price"))
    drift = None
    if price is None:
        ok = False
        severity = "block"
        messages.append("Robinhood last trade price is unavailable.")
    elif reference and reference > 0:
        drift = abs(price - reference) / reference * 100.0
        if drift > max_price_drift_pct:
            ok = False
            severity = "block"
            messages.append(
                f"Price drift too high: Robinhood {price:.2f} vs intent {reference:.2f} "
                f"({drift:.2f}% > {max_price_drift_pct:.2f}%)."
            )
        else:
            messages.append(f"Price drift OK: {drift:.2f}% <= {max_price_drift_pct:.2f}%.")

    bid = _float_or_none(quote.get("bid_price"))
    ask = _float_or_none(quote.get("ask_price"))
    spread_bps = None
    if bid and ask and bid > 0 and ask >= bid:
        mid = (bid + ask) / 2.0
        spread_bps = (ask - bid) / mid * 1e4 if mid > 0 else None
        if spread_bps is not None and spread_bps > max_spread_bps:
            ok = False
            severity = "block"
            messages.append(f"Spread too wide: {spread_bps:.1f}bps > {max_spread_bps:.1f}bps.")
        elif spread_bps is not None:
            messages.append(f"Spread OK: {spread_bps:.1f}bps.")
    else:
        if severity != "block":
            severity = "warn"
        messages.append("Bid/ask unavailable; spread check skipped.")

    return QuoteCheck(
        ok=ok,
        severity=severity,
        messages=messages,
        quote_price=price,
        price_drift_pct=drift,
        spread_bps=spread_bps,
    )


def mark_reviewed(path: str | Path, line_number: int, review_note: str = "") -> dict[str, Any]:
    intent_path = Path(path)
    records = load_intents(intent_path)
    updated = False
    new_lines: list[str] = []
    for record in records:
        payload = dict(record.payload)
        if record.line_number == line_number:
            payload["reviewed_at"] = "manual"
            if review_note:
                payload["review_note"] = review_note
            updated = True
        new_lines.append(json.dumps(payload, sort_keys=True))

    if not updated:
        raise ValueError(f"No intent found at line {line_number}.")

    intent_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return payload


def _print_latest(record: IntentRecord, account_number: str) -> None:
    intent = record.payload
    payload = review_payload(intent, account_number)
    print(f"Intent line: {record.line_number}")
    print(
        "Order intent: "
        f"{payload['side'].upper()} {payload.get('quantity', payload.get('dollar_amount', '?'))} "
        f"{payload['symbol']} {payload['type'].upper()}"
    )
    if intent.get("reason"):
        print(f"Reason: {intent['reason']}")
    if intent.get("reference_price") is not None:
        print(f"Reference price: {intent['reference_price']}")
    print("Robinhood MCP review_equity_order payload:")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("No order was placed.")


def _load_quote_arg(raw_quote: str | None) -> dict[str, Any] | None:
    if not raw_quote:
        return None
    maybe_path = Path(raw_quote)
    if maybe_path.exists():
        return json.loads(maybe_path.read_text(encoding="utf-8"))
    return json.loads(raw_quote)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect Robinhood Agentic order intents.")
    sub = parser.add_subparsers(dest="command", required=True)

    latest = sub.add_parser("latest", help="Print the latest unreviewed intent.")
    latest.add_argument("--path", default=None, help="Intent JSONL path.")
    latest.add_argument(
        "--quote-json",
        default=None,
        help="Robinhood quote JSON or path to JSON from get_equity_quotes results[].quote.",
    )
    latest.add_argument("--max-age-sec", type=int, default=60)
    latest.add_argument("--max-price-drift-pct", type=float, default=0.5)
    latest.add_argument("--max-spread-bps", type=float, default=20.0)

    mark = sub.add_parser("mark-reviewed", help="Mark an intent line as manually reviewed.")
    mark.add_argument("line_number", type=int)
    mark.add_argument("--path", default=None, help="Intent JSONL path.")
    mark.add_argument("--note", default="", help="Optional review note.")

    args = parser.parse_args()
    settings = Settings.from_env()
    path = args.path or settings.robinhood_order_intents_path

    if args.command == "latest":
        record = latest_intent(path)
        if record is None:
            print(f"No Robinhood intents found at {path}.")
            return 0
        if not settings.robinhood_agentic_account_number:
            print("ROBINHOOD_AGENTIC_ACCOUNT_NUMBER is required to build the MCP review payload.")
            return 2
        _print_latest(record, settings.robinhood_agentic_account_number)
        quote = _load_quote_arg(args.quote_json)
        if quote is None:
            print("Robinhood quote sanity: SKIPPED. Provide --quote-json before review/place.")
        else:
            check = quote_sanity_check(
                record.payload,
                quote,
                max_age_seconds=args.max_age_sec,
                max_price_drift_pct=args.max_price_drift_pct,
                max_spread_bps=args.max_spread_bps,
            )
            print(f"Robinhood quote sanity: {check.severity.upper()}")
            for message in check.messages:
                print(f"- {message}")
            if not check.ok:
                return 3
        return 0

    if args.command == "mark-reviewed":
        payload = mark_reviewed(path, args.line_number, args.note)
        print(
            f"Marked reviewed: line {args.line_number} "
            f"{payload.get('side', '').upper()} {payload.get('quantity', '?')} {payload.get('symbol', '')}"
        )
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
