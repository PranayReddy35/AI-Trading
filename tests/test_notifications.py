from __future__ import annotations

from ai_trading.notifications.alerter import Notifier


class CapturingNotifier(Notifier):
    def __init__(self) -> None:
        super().__init__(
            "https://fallback.example/webhook",
            ["trade", "risk_reject", "error", "daily_summary", "scanner_summary"],
            webhook_routes={
                "buy": "https://buy.example/webhook",
                "sell": "https://sell.example/webhook",
                "other": "https://other.example/webhook",
            },
        )
        self.sent: list[tuple[str, dict]] = []

    def _post_json(self, body: dict, webhook_url: str | None = None) -> bool:
        self.sent.append((str(webhook_url), body))
        return True


def test_notifier_routes_buy_payload_to_buy_channel() -> None:
    notifier = CapturingNotifier()

    assert notifier.notify("trade", "Preview BUY AAPL", {"symbol": "AAPL", "action": "BUY"})

    assert notifier.sent[-1][0] == "https://buy.example/webhook"


def test_notifier_routes_sell_payload_to_sell_channel() -> None:
    notifier = CapturingNotifier()

    assert notifier.notify("trade", "Trailing stop SELL AAPL", {"symbol": "AAPL", "side": "sell"})

    assert notifier.sent[-1][0] == "https://sell.example/webhook"


def test_notifier_routes_other_events_to_other_channel() -> None:
    notifier = CapturingNotifier()

    assert notifier.notify("risk_reject", "Freshness gate blocked", {"symbol": "AAPL"})

    assert notifier.sent[-1][0] == "https://other.example/webhook"


def test_daily_summary_uses_other_channel() -> None:
    notifier = CapturingNotifier()

    assert notifier.send_daily_summary(
        date="2026-06-05",
        equity=100.0,
        last_equity=99.0,
        positions=[],
        trades_today=1,
        pnl_today=1.0,
    )

    assert notifier.sent[-1][0] == "https://other.example/webhook"


def test_sell_scanner_summary_stays_in_other_channel() -> None:
    notifier = CapturingNotifier()

    assert notifier.notify("scanner_summary", "Sell scanner complete", {"result_count": 5})

    assert notifier.sent[-1][0] == "https://other.example/webhook"
