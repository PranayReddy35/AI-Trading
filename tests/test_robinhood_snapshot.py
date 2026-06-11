from __future__ import annotations

import json

from ai_trading.broker.robinhood_snapshot import (
    _build_account_snapshot,
    _build_quotes_payload,
    refresh_snapshots,
)


class FakeRobinhoodClient:
    def __init__(self) -> None:
        self.logged_in = False
        self.logged_out = False
        self.login_kwargs = {}

    def login(self, **kwargs):
        self.logged_in = True
        self.login_kwargs = kwargs
        return {"access_token": "token"}

    def logout(self):
        self.logged_out = True

    def load_account_profile(self, account_number=None, info=None):
        acct = str(account_number or "843789371")
        if acct == "593473374":
            return {
                "account_number": acct,
                "portfolio_cash": "100.00",
                "buying_power": "250.00",
                "cash_held_for_options_collateral": "15.00",
                "nickname": "Agentic Core",
            }
        return {
            "account_number": acct,
            "portfolio_cash": "250.00",
            "buying_power": "500.00",
        }

    def load_portfolio_profile(self, account_number=None, info=None):
        acct = str(account_number or "843789371")
        if acct == "593473374":
            return {"equity": "400.00"}
        return {"equity": "470.00"}

    def get_open_stock_positions(self, account_number=None, info=None):
        acct = str(account_number or "843789371")
        if acct == "593473374":
            return [
                {
                    "symbol": "AAPL",
                    "quantity": "1",
                    "average_buy_price": "180.00",
                    "shares_available_for_sells": "1",
                }
            ]
        return [
            {
                "instrument": "https://instrument/spy",
                "quantity": "2",
                "average_buy_price": "100.00",
                "shares_available_for_sells": "2",
            }
        ]

    def get_symbol_by_url(self, url: str):
        assert url == "https://instrument/spy"
        return "SPY"

    def get_crypto_positions(self, info=None):
        return [
            {
                "currency": {"code": "BTC"},
                "quantity": "0.01000000",
                "total_price_amount": "0.00000000",
                "cost_bases": [{"direct_cost_basis": "95.00"}],
            }
        ]

    def get_crypto_quote(self, symbol, info=None):
        assert symbol == "BTC"
        return {
            "symbol": "BTCUSD",
            "mark_price": "10500.00",
        }

    def get_quotes(self, symbols, info=None):
        assert symbols == ["SPY", "AAPL", "QQQ"]
        return [
            {
                "symbol": "SPY",
                "last_trade_price": "110.00",
                "venue_last_trade_time": "2026-06-11T14:30:00Z",
                "state": "active",
                "has_traded": True,
            },
            {
                "symbol": "AAPL",
                "last_trade_price": "300.00",
                "venue_last_trade_time": "2026-06-11T14:30:00Z",
                "state": "active",
                "has_traded": True,
            },
            {
                "symbol": "QQQ",
                "last_trade_price": "500.00",
                "venue_last_trade_time": "2026-06-11T14:30:00Z",
                "state": "active",
                "has_traded": True,
            },
        ]


def test_build_account_snapshot_masks_account_number() -> None:
    account = _build_account_snapshot(
        label="Investing",
        account_number="593473374",
        agentic=False,
        account_profile={"portfolio_cash": "100", "buying_power": "250"},
        portfolio_profile={},
        positions=[
            {
                "symbol": "SPY",
                "quantity": "2",
                "average_buy_price": "100",
                "shares_available_for_sells": "2",
            }
        ],
        quotes_by_symbol={"SPY": {"last_trade_price": "110"}},
    )

    assert account["account_masked"] == "*****3374"
    assert account["positions"][0]["symbol"] == "SPY"
    assert account["positions"][0]["market_value"] == "220.000000"
    assert account["portfolio"]["equity_value"] == "220.000000"
    assert account["portfolio"]["total_value"] == "320.000000"
    assert account["portfolio"]["buying_power"]["buying_power"] == "250.000000"


def test_build_quotes_payload_wraps_quote_results() -> None:
    payload = _build_quotes_payload(
        [
            {"symbol": "SPY", "last_trade_price": "110.00"},
            {"symbol": "QQQ", "last_trade_price": "500.00"},
        ]
    )

    assert len(payload["results"]) == 2
    assert payload["results"][0]["quote"]["symbol"] == "SPY"


def test_refresh_snapshots_writes_dashboard_files_with_agentic_account(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ROBINHOOD_AGENTIC_ACCOUNT_NUMBER", "593473374")
    monkeypatch.setenv("ROBINHOOD_QUOTE_SYMBOLS", "QQQ")
    client = FakeRobinhoodClient()
    portfolios_path = tmp_path / "robinhood_portfolios.json"
    quotes_path = tmp_path / "robinhood_quotes.json"

    result = refresh_snapshots(
        client=client,
        username="user@example.com",
        password="secret",
        portfolios_path=portfolios_path,
        quotes_path=quotes_path,
    )

    assert client.logged_in
    assert client.logged_out
    assert result.quote_symbols == ["SPY", "AAPL", "QQQ"]
    assert result.account_labels == ["Investing", "Agentic Core"]

    portfolio_payload = json.loads(portfolios_path.read_text(encoding="utf-8"))
    quote_payload = json.loads(quotes_path.read_text(encoding="utf-8"))
    assert len(portfolio_payload["accounts"]) == 2
    assert portfolio_payload["accounts"][0]["positions"][0]["symbol"] == "SPY"
    assert portfolio_payload["accounts"][0]["positions"][-1]["symbol"] == "BTC"
    assert portfolio_payload["accounts"][0]["portfolio"]["crypto_value"] == "105.000000"
    assert portfolio_payload["accounts"][0]["positions"][-1]["market_value"] == "105.00000000"
    assert portfolio_payload["accounts"][1]["agentic"] is True
    assert portfolio_payload["accounts"][1]["positions"][0]["symbol"] == "AAPL"
    assert quote_payload["results"][2]["quote"]["symbol"] == "QQQ"
