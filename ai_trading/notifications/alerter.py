from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from urllib.error import URLError
from urllib.request import Request, urlopen

from ai_trading.time_utils import format_local_now

logger = logging.getLogger("ai_trading")


class Notifier:
    """Send notifications via webhook (Slack, Discord, or generic JSON webhook).

    Supports any webhook that accepts a JSON POST with a "text" or "content" field.
    Set webhook_url="" to disable notifications entirely.
    """

    def __init__(
        self,
        webhook_url: str,
        notify_events: list[str] | None = None,
        webhook_routes: dict[str, str] | None = None,
    ) -> None:
        self.webhook_url = webhook_url.strip()
        self.notify_events = set(notify_events or ["trade", "error"])
        self.webhook_routes = self._load_routes(webhook_routes)

    @property
    def enabled(self) -> bool:
        return bool(self.webhook_url or any(self.webhook_routes.values()))

    @staticmethod
    def _load_routes(overrides: dict[str, str] | None = None) -> dict[str, str]:
        routes = {
            "buy": os.getenv("BOT_BUY_WEBHOOK_URL", os.getenv("DISCORD_BUY_WEBHOOK_URL", "")).strip(),
            "sell": os.getenv("BOT_SELL_WEBHOOK_URL", os.getenv("DISCORD_SELL_WEBHOOK_URL", "")).strip(),
            "other": os.getenv("BOT_OTHER_WEBHOOK_URL", os.getenv("DISCORD_OTHER_WEBHOOK_URL", "")).strip(),
        }
        if overrides:
            for key, value in overrides.items():
                if key in routes:
                    routes[key] = str(value or "").strip()
        return routes

    @staticmethod
    def _route_for(event_type: str, message: str, payload: dict | None) -> str:
        if event_type in {"error", "risk_reject", "drawdown", "daily_summary", "scanner_summary"}:
            return "other"
        text = f"{event_type} {message}".lower()
        action = str((payload or {}).get("action") or (payload or {}).get("side") or "").lower()
        if action == "buy" or " buy " in f" {text} " or text.startswith("buy "):
            return "buy"
        if action == "sell" or " sell " in f" {text} " or text.startswith("sell "):
            return "sell"
        return "other"

    def _webhook_for(self, event_type: str, message: str, payload: dict | None) -> str:
        route = self._route_for(event_type, message, payload)
        return self.webhook_routes.get(route) or self.webhook_routes.get("other") or self.webhook_url

    def _post_json(self, body: dict, webhook_url: str | None = None) -> bool:
        target = (webhook_url or self.webhook_url).strip()
        if not target:
            return False
        try:
            data = json.dumps(body).encode("utf-8")
            req = Request(
                target,
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

    @staticmethod
    def _event_color(event_type: str) -> int:
        return {
            "trade": 0x22C55E,
            "risk_reject": 0xF59E0B,
            "error": 0xEF4444,
            "drawdown": 0xF97316,
            "daily_summary": 0x3B82F6,
            "scanner_summary": 0x8B5CF6,
        }.get(event_type, 0x64748B)

    def _build_embed(self, event_type: str, message: str, payload: dict | None) -> dict | None:
        if not payload:
            return None
        fields = []
        preferred = [
            "mode", "symbol", "action", "qty", "price", "order_type", "limit_price",
            "latest_price", "latest_time", "latest_source", "latest_confidence",
            "latest_confidence_score", "reason", "order_id",
        ]
        for key in preferred:
            if key in payload and payload.get(key) is not None:
                fields.append({"name": key.replace("_", " ").title(), "value": str(payload.get(key))[:256], "inline": True})
        for key, value in payload.items():
            if key in preferred or len(fields) >= 12:
                continue
            if isinstance(value, (dict, list, tuple)):
                continue
            fields.append({"name": key.replace("_", " ").title(), "value": str(value)[:256], "inline": True})
        return {
            "title": f"{event_type.replace('_', ' ').title()}",
            "description": message[:300],
            "color": self._event_color(event_type),
            "fields": fields,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def notify(self, event_type: str, message: str, payload: dict | None = None) -> bool:
        """Send a notification if the event type is in the subscribed set.

        Returns True if notification was sent successfully, False otherwise.
        """
        if not self.enabled:
            return False
        if event_type not in self.notify_events:
            return False
        webhook_url = self._webhook_for(event_type, message, payload)
        if not webhook_url:
            return False

        ts = format_local_now()
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
        embed = self._build_embed(event_type, message, payload)
        if embed:
            body["embeds"] = [embed]

        return self._post_json(body, webhook_url)

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
        ts = format_local_now()
        body = {
            "content": f"[{ts}] [DAILY_SUMMARY]\n{message}",
            "text": f"[{ts}] [DAILY_SUMMARY]\n{message}",
        }
        try:
            return self._post_json(body, self.webhook_routes.get("other") or self.webhook_url)
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
