from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo


def app_timezone() -> ZoneInfo:
    tz_name = os.getenv("BOT_TIMEZONE") or os.getenv("BOT_DASHBOARD_TIMEZONE") or "America/Chicago"
    return ZoneInfo(tz_name)


def now_local() -> datetime:
    return datetime.now(app_timezone())


def local_iso_now() -> str:
    return now_local().isoformat()


def format_local(dt: datetime, fmt: str = "%Y-%m-%d %I:%M:%S %p %Z") -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(app_timezone()).strftime(fmt)


def format_local_now(fmt: str = "%Y-%m-%d %I:%M:%S %p %Z") -> str:
    return now_local().strftime(fmt)