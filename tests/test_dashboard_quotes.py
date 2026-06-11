from ai_trading.dashboard import _robinhood_quote_to_latest


def test_robinhood_quote_prefers_extended_hours_when_newer() -> None:
    latest = _robinhood_quote_to_latest(
        {
            "symbol": "SPY",
            "last_trade_price": "599.10",
            "venue_last_trade_time": "2026-06-11T20:00:00Z",
            "last_non_reg_trade_price": "600.25",
            "venue_last_non_reg_trade_time": "2026-06-11T21:00:00Z",
            "last_extended_hours_trade_price": "601.75",
            "updated_at": "2026-06-11T21:30:00Z",
            "bid_price": "601.70",
            "ask_price": "601.80",
            "state": "active",
        }
    )

    assert latest is not None
    assert latest["price"] == 601.75
    assert latest["source"] == "quote_last_extended_hours"
    assert latest["session"] == "extended"
