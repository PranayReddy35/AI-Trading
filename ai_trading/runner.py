from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import datetime, timedelta, timezone

from ai_trading.bot import run_once
from ai_trading.time_utils import format_local, format_local_now


_shutdown_requested = False


def _signal_handler(signum, frame):
    """Handle graceful shutdown on SIGINT/SIGTERM."""
    global _shutdown_requested
    _shutdown_requested = True
    print(f"\nShutdown signal received (sig={signum}). Finishing current cycle...")


def seconds_until(hour: int, minute: int) -> float:
    now = datetime.now(timezone.utc)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def _format_utc_schedule(hour: int, minute: int) -> str:
    now = datetime.now(timezone.utc)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return f"{format_local(target, '%I:%M %p %Z')} ({hour:02d}:{minute:02d} UTC)"


def _health_check() -> bool:
    """Verify config and Alpaca account connectivity without placing trades."""
    try:
        from ai_trading.config import Settings
        from ai_trading.broker.robinhood_agent import create_broker

        settings = Settings.from_env()
        settings.validate()
        broker = create_broker(settings)
        account = broker.account_state()
        mode = "PAPER" if settings.paper_only else "LIVE"
        dry_run = "on" if settings.stock_dry_run else "off"
        print(
            "Health check passed: "
            f"broker={settings.broker}, mode={mode}, stock_dry_run={dry_run}, "
            f"account_status={account['status']}, "
            f"buying_power=${account['buying_power']:,.2f}, "
            f"equity=${account['equity']:,.2f}"
        )
        return True
    except Exception as exc:
        print(f"Health check failed: {exc}", file=sys.stderr)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Daily scheduling runner for the trading bot. "
            "Supports both paper and live modes. "
            "Use cron/systemd/task scheduler for production deployments."
        )
    )
    parser.add_argument("--run-time", default="20:10", help="UTC HH:MM daily run time")
    parser.add_argument("--loop", action="store_true", help="Run continuously every day")
    parser.add_argument("--symbol", help="Optional symbol override")
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip live-order confirmation (for automated scheduled runs)",
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Run health check and exit",
    )
    parser.add_argument(
        "--daily-summary",
        default="",
        help="UTC HH:MM to send daily portfolio summary (e.g. 21:00)",
    )
    args = parser.parse_args()

    if args.health_check:
        sys.exit(0 if _health_check() else 1)

    # Register graceful shutdown handlers
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    hour, minute = None, None
    try:
        parts = args.run_time.split(":", maxsplit=1)
        hour, minute = int(parts[0]), int(parts[1])
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError
    except (ValueError, IndexError):
        parser.error(f"Invalid --run-time '{args.run_time}'. Expected format: HH:MM (e.g., 20:10)")

    if not args.loop:
        run_once(symbol_override=args.symbol, skip_confirmation=args.no_confirm)
        return

    # Parse optional daily summary time
    summary_hour, summary_minute = None, None
    if args.daily_summary:
        try:
            parts = args.daily_summary.split(":", maxsplit=1)
            summary_hour, summary_minute = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            parser.error(f"Invalid --daily-summary '{args.daily_summary}'. Expected HH:MM")

    print(f"Runner started. Scheduled daily at {_format_utc_schedule(hour, minute)}.")
    if summary_hour is not None:
        print(f"Daily summary at {_format_utc_schedule(summary_hour, summary_minute)}.")
    print("Press Ctrl+C for graceful shutdown.")

    last_summary_date = None

    while not _shutdown_requested:
        now = datetime.now(timezone.utc)

        # Check if daily summary should be sent
        if (
            summary_hour is not None
            and now.hour == summary_hour
            and now.minute == summary_minute
            and last_summary_date != now.date()
        ):
            last_summary_date = now.date()
            # Add daily_summary to notify_events temporarily via env override
            import os
            existing = os.environ.get("BOT_NOTIFY_EVENTS", "trade,error")
            if "daily_summary" not in existing:
                os.environ["BOT_NOTIFY_EVENTS"] = existing + ",daily_summary"
            try:
                run_once(symbol_override=args.symbol, skip_confirmation=args.no_confirm)
            except Exception as exc:
                print(f"Daily summary run failed: {exc}", file=sys.stderr)
            continue

        wait_seconds = seconds_until(hour, minute)
        print(f"Next run in {wait_seconds:.0f}s ({wait_seconds/3600:.1f}h)")

        # Sleep in small increments to allow graceful shutdown
        sleep_end = time.time() + wait_seconds
        while time.time() < sleep_end and not _shutdown_requested:
            time.sleep(min(30, sleep_end - time.time()))

        if _shutdown_requested:
            break

        print(f"[{format_local_now()}] Starting scheduled run...")
        try:
            run_once(symbol_override=args.symbol, skip_confirmation=args.no_confirm)
        except Exception as exc:
            print(f"Run failed with error: {exc}", file=sys.stderr)

    print("Runner shut down gracefully.")


if __name__ == "__main__":
    main()
