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

        # Build payload compatible with Slack/Discord/generic webhooks
        body = {
            "text": full_message,  # Slack format
            "content": full_message,  # Discord format
        }
        if payload:
            body["attachments"] = [{"text": json.dumps(payload, default=str)}]

        try:
            data = json.dumps(body).encode("utf-8")
            req = Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
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
