"""Auto-retraining module for the ensemble ML model.

Retrains the model on recent data and saves it to disk. Called by the runner
when ml_retrain_days is configured.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

logger = logging.getLogger("ai_trading")


def should_retrain(model_path: str, retrain_every_days: int) -> bool:
    """Return True if the model is missing or older than retrain_every_days."""
    if retrain_every_days <= 0:
        return False
    p = Path(model_path)
    if not p.exists():
        return True
    import time
    age_days = (time.time() - p.stat().st_mtime) / 86400
    return age_days >= retrain_every_days


def retrain_model(
    symbol: str,
    model_path: str,
    api_key: str,
    api_secret: str,
    lookback_days: int = 365,
) -> bool:
    """Retrain and save the ensemble ML model on recent data.

    Returns True on success, False on failure.
    """
    try:
        from ai_trading.data.market_data import AlpacaMarketData
        from ai_trading.ml.ensemble_model import EnsembleModel, build_features

        logger.info("Retraining ML model for %s ...", symbol)
        market_data = AlpacaMarketData(api_key, api_secret)
        bars = market_data.get_daily_bars(symbol, lookback_days)

        features = build_features(bars)
        if len(features) < 60:
            logger.warning("Not enough data to retrain ML model (%d rows)", len(features))
            return False

        model = EnsembleModel()
        model.fit(features)

        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        logger.info("ML model retrained and saved to %s", model_path)
        return True
    except Exception as exc:
        logger.error("ML retraining failed: %s", exc)
        return False
