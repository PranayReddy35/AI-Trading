"""Structured JSON logging helper with pattern attribution.

Wraps the existing JSONL journal with a richer `log_ensemble_decision()`
that records every pattern hit and weight, enabling per-trade attribution
in the dashboard.
"""

from __future__ import annotations

from typing import Any

from ai_trading.storage.journal import Journal


def log_ensemble_decision(
    journal: Journal,
    symbol: str,
    ensemble_signal,  # ai_trading.strategy.ensemble.EnsembleSignal
    pattern_hits: list | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "symbol": symbol,
        "signal": ensemble_signal.signal,
        "strength": ensemble_signal.strength,
        "confidence": ensemble_signal.confidence,
        "regime": ensemble_signal.regime.value,
        "consensus_count": ensemble_signal.consensus_count,
        "weights": ensemble_signal.weights_used,
        "strategy_signals": [
            {"name": s.name, "signal": round(s.signal, 4), "confidence": round(s.confidence, 4), "reason": s.reason}
            for s in ensemble_signal.strategy_signals
        ],
    }
    if pattern_hits is not None:
        payload["pattern_hits"] = [
            {"name": h.name, "signal": round(h.signal, 4), "confidence": round(h.confidence, 4), "reason": h.reason}
            for h in pattern_hits
        ]
    if extra:
        payload.update(extra)
    journal.write("ensemble_decision", payload)
