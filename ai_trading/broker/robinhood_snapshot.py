from __future__ import annotations

import argparse
import inspect
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_trading.broker.robinhood_health import mask_account_number
from ai_trading.env import load_dotenv


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _quote_symbols_from_env() -> list[str]:
    raw = os.getenv("ROBINHOOD_QUOTE_SYMBOLS", "")
    return [_normalize_symbol(part) for part in raw.split(",") if _normalize_symbol(part)]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _quote_price(quote: dict[str, Any]) -> float:
    for key in ("last_extended_hours_trade_price", "last_non_reg_trade_price", "last_trade_price", "ask_price", "bid_price"):
        price = _as_float(quote.get(key))
        if price > 0:
            return price
    return 0.0


def _crypto_quote_price(quote: dict[str, Any]) -> float:
    for key in ("mark_price", "ask_price", "bid_price", "open_price"):
        price = _as_float(quote.get(key))
        if price > 0:
            return price
    return 0.0


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required for Robinhood snapshot refresh.")
    return value


def _import_robinhood_client():
    try:
        from robin_stocks import robinhood as rh
    except ImportError as exc:
        raise RuntimeError(
            "Missing optional dependency 'robin_stocks'. Install it with "
            "`.venv/bin/pip install robin_stocks` or add it to your environment first."
        ) from exc
    return rh


def _symbol_from_position(position: dict[str, Any], client) -> str:
    symbol = _normalize_symbol(position.get("symbol"))
    if symbol:
        return symbol
    instrument_url = str(position.get("instrument") or "").strip()
    if not instrument_url:
        return ""
    getter = getattr(client, "get_symbol_by_url", None)
    if callable(getter):
        try:
            return _normalize_symbol(getter(instrument_url))
        except Exception:
            pass
    getter = getattr(client, "get_instrument_by_url", None)
    if callable(getter):
        try:
            instrument = getter(instrument_url)
            if isinstance(instrument, dict):
                return _normalize_symbol(instrument.get("symbol"))
        except Exception:
            pass
    return ""


def _account_numbers_from_env() -> list[str]:
    account_number = _normalize_account_number(_require_env("ROBINHOOD_AGENTIC_ACCOUNT_NUMBER"))
    extra_raw = os.getenv("ROBINHOOD_ACCOUNT_NUMBERS", "").strip()
    extra = [_normalize_account_number(part) for part in extra_raw.split(",") if _normalize_account_number(part)]
    out: list[str] = []
    for value in [account_number, *extra]:
        if value and value not in out:
            out.append(value)
    return out


def _normalize_account_number(value: Any) -> str:
    cleaned = "".join(ch for ch in str(value or "") if ch.isdigit())
    return cleaned


def _account_label(profile: dict[str, Any], *, agentic_account_number: str, fallback: str) -> tuple[str, bool]:
    account_number = _normalize_account_number(profile.get("account_number") or profile.get("rhs_account_number"))
    agentic = bool(account_number and account_number == agentic_account_number)
    nickname = str(profile.get("nickname") or "").strip()
    if nickname:
        return nickname, agentic
    return ("Agentic" if agentic else fallback), agentic


def _coerce_crypto_positions(
    holdings: list[dict[str, Any]],
    crypto_quotes: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], float]:
    out: list[dict[str, Any]] = []
    total_value = 0.0
    crypto_quotes = crypto_quotes or {}
    for raw in holdings:
        if not isinstance(raw, dict):
            continue
        quantity = _as_float(raw.get("quantity") or raw.get("total_quantity"))
        if quantity <= 0:
            continue
        symbol = _normalize_symbol(raw.get("currency", {}).get("code") if isinstance(raw.get("currency"), dict) else raw.get("symbol"))
        if not symbol:
            continue
        increment = _as_float(raw.get("increment"))
        total_price = _as_float(raw.get("total_price_amount"))
        cost_bases = raw.get("cost_bases")
        avg_cost = 0.0
        if isinstance(cost_bases, list) and cost_bases:
            avg_cost = _as_float(cost_bases[0].get("direct_cost_basis"))
        market_value = total_price if total_price > 0 else 0.0
        quote = crypto_quotes.get(symbol, {})
        quote_price = _crypto_quote_price(quote)
        if quote_price > 0:
            market_value = quantity * quote_price
        elif market_value <= 0:
            market_value = quantity * increment
        total_value += market_value
        out.append(
            {
                "symbol": symbol,
                "quantity": f"{quantity:.8f}",
                "average_buy_price": f"{avg_cost:.8f}",
                "shares_available_for_sells": f"{quantity:.8f}",
                "market_value": f"{market_value:.8f}",
                "type": "crypto",
            }
        )
    return out, total_value


def _build_account_snapshot(
    *,
    label: str,
    account_number: str,
    agentic: bool,
    account_profile: dict[str, Any],
    portfolio_profile: dict[str, Any],
    positions: list[dict[str, Any]],
    quotes_by_symbol: dict[str, dict[str, Any]],
    crypto_positions: list[dict[str, Any]] | None = None,
    crypto_quotes: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    updated_at = _now_iso()
    normalized_positions: list[dict[str, Any]] = []
    equity_value = 0.0
    options_value = _as_float(account_profile.get("cash_held_for_options_collateral"))
    cash = _as_float(
        account_profile.get("portfolio_cash")
        or account_profile.get("cash")
        or account_profile.get("cash_available_for_withdrawal")
    )
    buying_power = _as_float(account_profile.get("buying_power"))
    for raw in positions:
        symbol = _normalize_symbol(raw.get("symbol"))
        if not symbol:
            continue
        quantity = _as_float(raw.get("quantity"))
        if quantity <= 0:
            continue
        avg_cost = _as_float(raw.get("average_buy_price"))
        quote = quotes_by_symbol.get(symbol, {})
        last_price = _quote_price(quote)
        market_value = _as_float(raw.get("market_value"), quantity * last_price)
        if market_value <= 0 and last_price > 0:
            market_value = quantity * last_price
        equity_value += market_value
        normalized_positions.append(
            {
                "symbol": symbol,
                "quantity": f"{quantity:.6f}",
                "average_buy_price": f"{avg_cost:.6f}",
                "shares_available_for_sells": f"{quantity:.6f}",
                "market_value": f"{market_value:.6f}",
                "type": "long",
            }
        )
    crypto_rows, crypto_value = _coerce_crypto_positions(crypto_positions or [], crypto_quotes)
    total_value = _as_float(
        portfolio_profile.get("extended_hours_equity")
        or portfolio_profile.get("equity")
        or portfolio_profile.get("market_value"),
        equity_value + crypto_value + cash,
    )
    if total_value <= 0:
        total_value = equity_value + crypto_value + cash
    return {
        "label": label,
        "account_masked": mask_account_number(account_number),
        "agentic": agentic,
        "features": {
            "account_number": account_number,
            "nickname": str(account_profile.get("nickname") or ""),
            "account_type": str(account_profile.get("type") or account_profile.get("brokerage_account_type") or ""),
            "option_level": str(account_profile.get("option_level") or ""),
            "options_buying_power": f"{_as_float(account_profile.get('option_buying_power')):.6f}",
            "crypto_buying_power": f"{_as_float(account_profile.get('crypto_buying_power')):.6f}",
            "cash_management_enabled": bool(account_profile.get("cash_management_enabled")),
            "eligible_for_fractionals": bool(account_profile.get("eligible_for_fractionals")),
            "eligible_for_drip": bool(account_profile.get("eligible_for_drip")),
            "drip_enabled": bool(account_profile.get("drip_enabled")),
            "eligible_for_cash_management": bool(account_profile.get("eligible_for_cash_management")),
            "fractional_position_closing_only": bool(account_profile.get("fractional_position_closing_only")),
            "option_trading_on_expiration_enabled": bool(account_profile.get("option_trading_on_expiration_enabled")),
            "agentic_allowed": bool(account_profile.get("agentic_allowed")),
            "ipo_access_restricted": bool(account_profile.get("ipo_access_restricted")),
            "ipo_access_restricted_reason": str(account_profile.get("ipo_access_restricted_reason") or ""),
            "state": str(account_profile.get("state") or ""),
        },
        "portfolio": {
            "total_value": f"{total_value:.6f}",
            "equity_value": f"{equity_value:.6f}",
            "options_value": f"{options_value:.6f}",
            "crypto_value": f"{crypto_value:.6f}",
            "cash": f"{cash:.6f}",
            "currency": "USD",
            "buying_power": {
                "buying_power": f"{buying_power:.6f}",
                "display_currency": "USD",
            },
        },
        "positions": normalized_positions + crypto_rows,
        "updated_at": str(
            portfolio_profile.get("updated_at")
            or account_profile.get("updated_at")
            or updated_at
        ),
    }


def _build_quotes_payload(quotes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "updated_at": _now_iso(),
        "results": [{"quote": quote} for quote in quotes if _normalize_symbol(quote.get("symbol"))],
    }


def load_fresh_robinhood_quotes(
    path: str | Path | None = None,
    *,
    ttl_seconds: int | None = None,
) -> dict[str, float]:
    p = Path(path or os.getenv("ROBINHOOD_QUOTES_PATH", "logs/robinhood_quotes.json"))
    if not p.exists():
        return {}
    ttl = ttl_seconds if ttl_seconds is not None else max(30, _env_int("ROBINHOOD_SNAPSHOT_TTL_SEC", 300))
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    updated_at = str(payload.get("updated_at") or "") if isinstance(payload, dict) else ""
    if updated_at:
        try:
            age = max(0.0, (datetime.now(timezone.utc) - datetime.fromisoformat(updated_at.replace("Z", "+00:00"))).total_seconds())
            if age > ttl:
                return {}
        except Exception:
            return {}
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        return {}
    out: dict[str, float] = {}
    for item in results:
        quote = item.get("quote") if isinstance(item, dict) else None
        if not isinstance(quote, dict):
            continue
        symbol = _normalize_symbol(quote.get("symbol"))
        if not symbol:
            continue
        price = _quote_price(quote)
        if price > 0:
            out[symbol] = price
    return out


def _login(client, *, username: str, password: str, mfa_code: str | None = None) -> None:
    login_kwargs: dict[str, Any] = {
        "username": username,
        "password": password,
        "store_session": True,
    }
    if mfa_code:
        login_kwargs["mfa_code"] = mfa_code
    device_token = os.getenv("ROBINHOOD_DEVICE_TOKEN", "").strip()
    if device_token:
        try:
            params = inspect.signature(client.login).parameters
        except (TypeError, ValueError):
            params = {}
        if "device_token" in params:
            login_kwargs["device_token"] = device_token
    client.login(**login_kwargs)


@dataclass(frozen=True)
class SnapshotRefreshResult:
    quote_symbols: list[str]
    account_labels: list[str]
    portfolio_path: Path
    quotes_path: Path


@dataclass(frozen=True)
class _AccountContext:
    label: str
    account_number: str
    agentic: bool
    account_profile: dict[str, Any]
    portfolio_profile: dict[str, Any]
    positions: list[dict[str, Any]]
    crypto_positions: list[dict[str, Any]] | None
    crypto_quotes: dict[str, dict[str, Any]] | None


def refresh_loop(
    *,
    interval_sec: int,
    client=None,
    username: str | None = None,
    password: str | None = None,
    mfa_code: str | None = None,
    portfolios_path: str | Path | None = None,
    quotes_path: str | Path | None = None,
) -> int:
    interval = max(5, int(interval_sec))
    print(f"Robinhood snapshot loop started. Refresh interval: {interval}s")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            result = refresh_snapshots(
                client=client,
                username=username,
                password=password,
                mfa_code=mfa_code,
                portfolios_path=portfolios_path,
                quotes_path=quotes_path,
            )
            labels = ", ".join(result.account_labels) or "none"
            print(
                f"[{_now_iso()}] Refresh OK: accounts={labels} symbols={len(result.quote_symbols)}"
            )
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nRobinhood snapshot loop stopped.")
        return 0


def refresh_snapshots(
    *,
    client=None,
    username: str | None = None,
    password: str | None = None,
    mfa_code: str | None = None,
    portfolios_path: str | Path | None = None,
    quotes_path: str | Path | None = None,
    extra_quote_symbols: list[str] | None = None,
) -> SnapshotRefreshResult:
    load_dotenv()
    client = client or _import_robinhood_client()
    username = username or _require_env("ROBINHOOD_USERNAME")
    password = password or _require_env("ROBINHOOD_PASSWORD")
    agentic_account_number = _require_env("ROBINHOOD_AGENTIC_ACCOUNT_NUMBER")
    target_account_numbers = _account_numbers_from_env()
    portfolios_path = Path(portfolios_path or os.getenv("ROBINHOOD_PORTFOLIOS_PATH", "logs/robinhood_portfolios.json"))
    quotes_path = Path(quotes_path or os.getenv("ROBINHOOD_QUOTES_PATH", "logs/robinhood_quotes.json"))

    _login(
        client,
        username=username,
        password=password,
        mfa_code=mfa_code or os.getenv("ROBINHOOD_MFA_CODE", "").strip() or None,
    )

    try:
        default_account_profile = client.load_account_profile(info=None)
        default_account_number = _normalize_account_number(
            default_account_profile.get("account_number") if isinstance(default_account_profile, dict) else ""
        )
        if default_account_number and default_account_number not in target_account_numbers:
            target_account_numbers.insert(0, default_account_number)
        if not target_account_numbers:
            raise ValueError("No Robinhood account numbers available for snapshot refresh.")
        quote_symbols: list[str] = []
        account_contexts: list[_AccountContext] = []
        for idx, account_number in enumerate(target_account_numbers):
            account_profile = client.load_account_profile(account_number=account_number, info=None)
            portfolio_profile = client.load_portfolio_profile(account_number=account_number, info=None)
            raw_positions = client.get_open_stock_positions(account_number=account_number, info=None) or []
            normalized_positions: list[dict[str, Any]] = []
            for raw in raw_positions:
                if not isinstance(raw, dict):
                    continue
                symbol = _symbol_from_position(raw, client)
                if not symbol:
                    continue
                item = dict(raw)
                item["symbol"] = symbol
                normalized_positions.append(item)
                if symbol not in quote_symbols:
                    quote_symbols.append(symbol)
            label, agentic = _account_label(
                account_profile if isinstance(account_profile, dict) else {},
                agentic_account_number=agentic_account_number,
                fallback="Investing" if idx == 0 else f"Account {idx + 1}",
            )
            crypto_positions = None
            crypto_quotes = None
            if idx == 0:
                try:
                    crypto_positions = client.get_crypto_positions(info=None) or []
                    crypto_symbols = [
                        _normalize_symbol(item.get("currency", {}).get("code") if isinstance(item.get("currency"), dict) else item.get("symbol"))
                        for item in crypto_positions
                        if isinstance(item, dict) and _as_float(item.get("quantity") or item.get("total_quantity")) > 0
                    ]
                    crypto_quotes = {}
                    for crypto_symbol in crypto_symbols:
                        if not crypto_symbol:
                            continue
                        try:
                            quote = client.get_crypto_quote(crypto_symbol, info=None)
                        except Exception:
                            quote = None
                        if isinstance(quote, dict):
                            crypto_quotes[crypto_symbol] = quote
                except Exception:
                    crypto_positions = None
                    crypto_quotes = None
            account_contexts.append(
                _AccountContext(
                    label=label,
                    account_number=account_number,
                    agentic=agentic,
                    account_profile=account_profile if isinstance(account_profile, dict) else {},
                    portfolio_profile=portfolio_profile if isinstance(portfolio_profile, dict) else {},
                    positions=normalized_positions,
                    crypto_positions=crypto_positions,
                    crypto_quotes=crypto_quotes,
                )
            )
        requested_extras = list(extra_quote_symbols or [])
        for extra in [*_quote_symbols_from_env(), *requested_extras]:
            if extra not in quote_symbols:
                quote_symbols.append(extra)
        quotes = client.get_quotes(quote_symbols, info=None) if quote_symbols else []
        if isinstance(quotes, dict):
            quotes = [quotes]
        quotes_by_symbol = {
            _normalize_symbol(q.get("symbol")): q
            for q in quotes
            if isinstance(q, dict) and _normalize_symbol(q.get("symbol"))
        }
        refreshed_accounts = [
                _build_account_snapshot(
                    label=ctx.label,
                    account_number=ctx.account_number,
                    agentic=ctx.agentic,
                    account_profile=ctx.account_profile,
                    portfolio_profile=ctx.portfolio_profile,
                    positions=ctx.positions,
                    quotes_by_symbol=quotes_by_symbol,
                    crypto_positions=ctx.crypto_positions,
                    crypto_quotes=ctx.crypto_quotes,
                )
            for ctx in account_contexts
        ]
        portfolio_payload = {"updated_at": _now_iso(), "accounts": refreshed_accounts}
        quote_payload = _build_quotes_payload([q for q in quotes if isinstance(q, dict)])
        portfolios_path.parent.mkdir(parents=True, exist_ok=True)
        quotes_path.parent.mkdir(parents=True, exist_ok=True)
        portfolios_path.write_text(json.dumps(portfolio_payload, indent=2) + "\n", encoding="utf-8")
        quotes_path.write_text(json.dumps(quote_payload, indent=2) + "\n", encoding="utf-8")
        return SnapshotRefreshResult(
            quote_symbols=quote_symbols,
            account_labels=[ctx.label for ctx in account_contexts],
            portfolio_path=portfolios_path,
            quotes_path=quotes_path,
        )
    finally:
        logout = getattr(client, "logout", None)
        if callable(logout):
            try:
                logout()
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh local Robinhood portfolio and quote snapshots for the dashboard.")
    parser.add_argument("--env-file", default=None, help="Optional dotenv path.")
    parser.add_argument("--username", default=None, help="Robinhood username/email. Defaults to ROBINHOOD_USERNAME.")
    parser.add_argument("--password", default=None, help="Robinhood password. Defaults to ROBINHOOD_PASSWORD.")
    parser.add_argument("--mfa-code", default=None, help="Robinhood MFA code. Defaults to ROBINHOOD_MFA_CODE.")
    parser.add_argument("--quotes-path", default=None, help="Where to write the Robinhood quotes snapshot JSON.")
    parser.add_argument("--portfolios-path", default=None, help="Where to write the Robinhood portfolios snapshot JSON.")
    parser.add_argument("--loop", action="store_true", help="Keep refreshing snapshots on an interval until stopped.")
    parser.add_argument(
        "--interval-sec",
        type=int,
        default=_env_int("ROBINHOOD_REFRESH_INTERVAL_SEC", 60),
        help="Refresh interval in seconds for --loop mode.",
    )
    args = parser.parse_args()

    load_dotenv(args.env_file)
    try:
        if args.loop:
            return refresh_loop(
                interval_sec=args.interval_sec,
                username=args.username,
                password=args.password,
                mfa_code=args.mfa_code,
                portfolios_path=args.portfolios_path,
                quotes_path=args.quotes_path,
            )
        result = refresh_snapshots(
            username=args.username,
            password=args.password,
            mfa_code=args.mfa_code,
            portfolios_path=args.portfolios_path,
            quotes_path=args.quotes_path,
        )
    except Exception as exc:
        print(f"Robinhood snapshot refresh failed: {exc}")
        return 2

    print("Robinhood snapshot refresh: PASS")
    print(f"- Portfolio snapshot: {result.portfolio_path}")
    print(f"- Quote snapshot: {result.quotes_path}")
    print(f"- Quote symbols refreshed: {len(result.quote_symbols)}")
    print(f"- Accounts refreshed: {', '.join(result.account_labels) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
