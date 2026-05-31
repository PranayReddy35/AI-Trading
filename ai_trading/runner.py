from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone

from ai_trading.bot import run_once


def seconds_until(hour: int, minute: int) -> float:
    now = datetime.now(timezone.utc)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Daily scheduling stub for educational use. "
            "Use cron/systemd/task scheduler in production."
        )
    )
    parser.add_argument("--run-time", default="20:10", help="UTC HH:MM daily run time")
    parser.add_argument("--loop", action="store_true", help="Run continuously every day")
    args = parser.parse_args()

    hour, minute = [int(x) for x in args.run_time.split(":", maxsplit=1)]

    if not args.loop:
        run_once()
        return

    while True:
        wait_seconds = seconds_until(hour, minute)
        time.sleep(wait_seconds)
        run_once()


if __name__ == "__main__":
    main()
