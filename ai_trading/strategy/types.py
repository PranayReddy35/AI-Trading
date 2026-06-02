"""Shared types for strategy modules (avoids circular imports)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class StrategySignal:
    """Signal from an individual strategy.

    signal: -1.0 (strong sell) .. +1.0 (strong buy), 0 = neutral
    confidence: 0..1
    """

    name: str
    signal: float
    confidence: float
    reason: str
