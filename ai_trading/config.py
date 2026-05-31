from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Settings:
    api_key: str
    api_secret: str
    symbol: str = "SPY"
    fast_ma: int = 5
    slow_ma: int = 20
    lookback_days: int = 90
    max_shares: int = 1
    min_cash_threshold: float = 100.0
    max_daily_trades: int = 1
    max_consecutive_errors: int = 3
    paper_only: bool = True
    log_path: Path = Path("logs/bot.log")
    journal_path: Path = Path("logs/journal.jsonl")

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.getenv("APCA_API_KEY_ID", "")
        api_secret = os.getenv("APCA_API_SECRET_KEY", "")
        return cls(
            api_key=api_key,
            api_secret=api_secret,
            symbol=os.getenv("BOT_SYMBOL", "SPY").upper(),
            fast_ma=int(os.getenv("BOT_FAST_MA", "5")),
            slow_ma=int(os.getenv("BOT_SLOW_MA", "20")),
            lookback_days=int(os.getenv("BOT_LOOKBACK_DAYS", "90")),
            max_shares=max(1, int(os.getenv("BOT_MAX_SHARES", "1"))),
            min_cash_threshold=float(os.getenv("BOT_MIN_CASH_THRESHOLD", "100")),
            max_daily_trades=max(1, int(os.getenv("BOT_MAX_DAILY_TRADES", "1"))),
            max_consecutive_errors=max(1, int(os.getenv("BOT_MAX_CONSECUTIVE_ERRORS", "3"))),
            paper_only=os.getenv("BOT_PAPER_ONLY", "true").lower() == "true",
            log_path=Path(os.getenv("BOT_LOG_PATH", "logs/bot.log")),
            journal_path=Path(os.getenv("BOT_JOURNAL_PATH", "logs/journal.jsonl")),
        )

    def validate(self) -> None:
        if not self.api_key or not self.api_secret:
            raise ValueError("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY.")
        if self.fast_ma <= 0 or self.slow_ma <= 0 or self.fast_ma >= self.slow_ma:
            raise ValueError("Require 0 < fast_ma < slow_ma.")
