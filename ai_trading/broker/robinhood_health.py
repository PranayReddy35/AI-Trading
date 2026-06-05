from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

from ai_trading.env import load_dotenv


TRUE_VALUES = {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class RobinhoodReadiness:
    ok: bool
    messages: list[str]


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in TRUE_VALUES


def mask_account_number(account_number: str) -> str:
    cleaned = "".join(ch for ch in account_number if ch.isdigit())
    if not cleaned:
        return "(missing)"
    return f"{'*' * max(0, len(cleaned) - 4)}{cleaned[-4:]}"


def check_readiness() -> RobinhoodReadiness:
    account_number = os.getenv("ROBINHOOD_AGENTIC_ACCOUNT_NUMBER", "").strip()
    agentic_enabled = _env_bool("ROBINHOOD_AGENTIC_ENABLED", False)
    broker = os.getenv("BOT_BROKER", "alpaca").strip().lower() or "alpaca"
    paper_only = _env_bool("BOT_PAPER_ONLY", True)
    stock_dry_run = _env_bool("BOT_STOCK_DRY_RUN", False)
    kill_switch = _env_bool("BOT_KILL_SWITCH", False)
    buying_power = os.getenv("ROBINHOOD_AGENTIC_BUYING_POWER", "").strip()
    equity = os.getenv("ROBINHOOD_AGENTIC_EQUITY", "").strip()

    ok = True
    messages: list[str] = []

    if not account_number:
        ok = False
        messages.append("ROBINHOOD_AGENTIC_ACCOUNT_NUMBER is not set.")
    else:
        messages.append(
            f"Robinhood Agentic account configured: {mask_account_number(account_number)}"
        )

    if not agentic_enabled:
        ok = False
        messages.append("ROBINHOOD_AGENTIC_ENABLED must be true after you verify the account.")
    else:
        messages.append("ROBINHOOD_AGENTIC_ENABLED=true")

    if broker == "robinhood":
        messages.append(
            "BOT_BROKER=robinhood; bot runs in Agentic proposal mode and records MCP order intents."
        )
    else:
        messages.append(f"BOT_BROKER={broker}; existing bot order routing remains Alpaca-based.")

    if broker == "robinhood":
        if not buying_power or not equity:
            ok = False
            messages.append("Set ROBINHOOD_AGENTIC_BUYING_POWER and ROBINHOOD_AGENTIC_EQUITY from a fresh portfolio check.")
        elif not stock_dry_run:
            ok = False
            messages.append("BOT_STOCK_DRY_RUN must be true for Robinhood Agentic proposal mode.")

    if not paper_only:
        messages.append("Safety warning: BOT_PAPER_ONLY=false.")
    if broker != "robinhood" and not stock_dry_run:
        messages.append("Safety warning: BOT_STOCK_DRY_RUN=false.")
    if not kill_switch:
        messages.append("Safety note: BOT_KILL_SWITCH=false.")

    return RobinhoodReadiness(ok=ok, messages=messages)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only Robinhood Agentic account readiness check."
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Path to a dotenv file. Defaults to the project .env lookup.",
    )
    args = parser.parse_args()

    load_dotenv(args.env_file)
    result = check_readiness()

    status = "PASS" if result.ok else "FAIL"
    print(f"Robinhood Agentic readiness: {status}")
    for message in result.messages:
        print(f"- {message}")
    print("- No orders were reviewed, submitted, or cancelled.")

    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
