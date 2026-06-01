from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger("ai_trading")


class Notifier:
    """Send notifications via webhook (Slack, Discord, or generic JSON webhook).

    Supports any webhook that accepts a JSON POST with a "text" or "content" field.
    Set webhook_url="" to disable notifications entirely.
    """

    def __init__(self, webhook_url: str, notify_events: list[str] | None = None) -> None:
        self.webhook_url = webhook_url.strip()
        self.notify_events = set(notify_events or ["trade", "error"])

    @property
    def enabled(self) -> bool:
        return bool(self.webhook_url)

    def notify(self, event_type: str, message: str, payload: dict | None = None) -> bool:
        """Send a notification if the event type is in the subscribed set.

        Returns True if notification was sent successfully, False otherwise.
        """
        if not self.enabled:
            return False
        if event_type not in self.notify_events:
            return False

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        full_message = f"[{ts}] [{event_type.upper()}] {message}"

        # Discord uses "content"; Slack uses "text". Include both for compatibility.
        # Append payload as a code block if provided (avoids Slack-specific "attachments")
        if payload:
            payload_str = json.dumps(payload, default=str, indent=2)
            # Truncate to stay within Discord's 2000-char limit
            if len(full_message) + len(payload_str) < 1800:
                full_message = f"{full_message}\n```json\n{payload_str}\n```"

        body = {
            "content": full_message[:2000],  # Discord 2000-char limit
            "text": full_message[:2000],      # Slack fallback
        }

        try:
            data = json.dumps(body).encode("utf-8")
            req = Request(
                self.webhook_url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0",
                },
                method="POST",
            )
            with urlopen(req, timeout=10) as resp:
                if 200 <= resp.status < 300:
                    return True
                logger.warning("Webhook returned status %d", resp.status)
                return False
        except (URLError, OSError) as exc:
            logger.warning("Notification failed: %s", exc)
            return False
        except Exception as exc:
            logger.warning("Unexpected notification error: %s", exc)
            return False

    def send_daily_summary(
        self,
        *,
        date: str,
        equity: float,
        last_equity: float,
        positions: list[dict],
        trades_today: int,
        pnl_today: float,
    ) -> bool:
        """Send a daily portfolio summary to Discord."""
        if not self.enabled or "daily_summary" not in self.notify_events:
            return False

        pnl_sign = "+" if pnl_today >= 0 else ""
        lines = [
            f"**Daily Summary — {date}**",
            f"Equity: ${equity:,.2f}  (${pnl_sign}{pnl_today:,.2f} / {pnl_sign}{(pnl_today/last_equity*100) if last_equity else 0:.2f}%)",
            f"Trades today: {trades_today}",
        ]
        if positions:
            lines.append("**Open Positions:**")
            for p in positions:
                pl = p.get("unrealized_pl", 0)
                pl_sign = "+" if pl >= 0 else ""
                lines.append(
                    f"  {p['symbol']}: {p['qty']} shares  P&L {pl_sign}${pl:.2f}"
                )
        else:
            lines.append("No open positions.")

        message = "\n".join(lines)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        body = {
            "content": f"[{ts}] [DAILY_SUMMARY]\n{message}",
            "text": f"[{ts}] [DAILY_SUMMARY]\n{message}",
        }
        try:
            data = json.dumps(body).encode("utf-8")
            req = Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
                method="POST",
            )
            with urlopen(req, timeout=10) as resp:
                return 200 <= resp.status < 300
        except Exception as exc:
            logger.warning("Daily summary notification failed: %s", exc)
            return False

    def send_drawdown_alert(self, drawdown_pct: float, equity: float, peak_equity: float) -> bool:
        """Send a drawdown alert notification."""
        if not self.enabled or "drawdown" not in self.notify_events:
            return False
        message = (
            f"⚠️ Portfolio drawdown alert: {drawdown_pct:.2f}% from peak "
            f"(equity ${equity:,.2f} vs peak ${peak_equity:,.2f})"
        )
        return self.notify("drawdown", message)

