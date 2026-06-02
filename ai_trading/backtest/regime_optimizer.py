"""Walk-forward optimizer for ensemble regime weights.

Free-source friendly: uses yfinance historical OHLCV. Splits history into
rolling windows, simulates the ensemble on each, and finds the regime-weights
that maximize Sharpe (or PnL) on the out-of-sample slice.

Usage:
    python -m ai_trading.backtest.regime_optimizer SPY,QQQ --period 3y --train 180 --test 60

The optimizer doesn't *change* `ensemble.REGIME_WEIGHTS` automatically — it
prints a recommended weight table you can paste into the source or write to
`logs/regime_weights.json` for the bot to load.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _download(symbols: list[str], period: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out: dict[str, pd.DataFrame] = {}
    for s in symbols:
        df = yf.download(s, period=period, progress=False, auto_adjust=False)
        if df.empty:
            continue
        df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
        out[s] = df
    return out


def _simulate(bars: pd.DataFrame, weights_override: dict | None, start: int, end: int) -> tuple[float, float]:
    """Return (sharpe, total_return) of a buy-on-BUY / cash-on-SELL strategy."""
    from ai_trading.strategy.ensemble import (
        compute_ensemble_signal,
        REGIME_WEIGHTS,
        MarketRegime,
    )
    backup = {k: v.copy() for k, v in REGIME_WEIGHTS.items()}
    try:
        if weights_override is not None:
            for regime, w in weights_override.items():
                REGIME_WEIGHTS[regime] = w

        position = 0.0
        entry = 0.0
        returns: list[float] = []
        equity = 1.0
        for i in range(start, end):
            window = bars.iloc[max(0, i - 250):i + 1]
            if len(window) < 60:
                continue
            try:
                es = compute_ensemble_signal(window)
            except Exception:
                continue
            price = float(window["close"].iloc[-1])

            if es.signal == "BUY" and position == 0:
                position, entry = 1.0, price
            elif es.signal == "SELL" and position > 0:
                ret = (price - entry) / entry
                equity *= (1 + ret)
                returns.append(ret)
                position = 0.0
                entry = 0.0
        # Mark-to-market open position at the end
        if position > 0:
            ret = (float(bars["close"].iloc[end - 1]) - entry) / entry
            equity *= (1 + ret)
            returns.append(ret)

        if not returns or len(returns) < 2:
            # Not enough trades to compute Sharpe — fall back to total return
            return float(equity - 1), float(equity - 1)
        r = np.array(returns)
        if r.std() <= 0:
            return float(equity - 1), float(equity - 1)
        # Annualised Sharpe assuming each "return" is one trade; scale by
        # avg trades-per-year given the window size.
        avg_trade_freq_per_year = len(r) / max(1, (end - start) / 252.0)
        sharpe = float(r.mean() / r.std() * np.sqrt(avg_trade_freq_per_year))
        # Clip to a sane range so spurious values don't dominate selection
        sharpe = max(-5.0, min(5.0, sharpe))
        return sharpe, float(equity - 1)
    finally:
        REGIME_WEIGHTS.clear()
        REGIME_WEIGHTS.update(backup)


def _candidate_weights() -> list[dict]:
    """Generate a small grid of weight candidates for the 5 strategy components."""
    from ai_trading.strategy.ensemble import MarketRegime
    keys = ("trend", "momentum", "mean_reversion", "patterns", "ml")
    grid_steps = (0.1, 0.2, 0.3, 0.4)
    candidates: list[dict] = []
    # Per-regime independent grid would explode; instead try a few archetypes.
    archetypes = [
        {"trend": 0.40, "momentum": 0.30, "mean_reversion": 0.05, "patterns": 0.15, "ml": 0.10},
        {"trend": 0.30, "momentum": 0.25, "mean_reversion": 0.15, "patterns": 0.20, "ml": 0.10},
        {"trend": 0.20, "momentum": 0.20, "mean_reversion": 0.25, "patterns": 0.20, "ml": 0.15},
        {"trend": 0.10, "momentum": 0.10, "mean_reversion": 0.35, "patterns": 0.30, "ml": 0.15},
        {"trend": 0.25, "momentum": 0.25, "mean_reversion": 0.15, "patterns": 0.25, "ml": 0.10},
    ]
    return archetypes


def walk_forward(symbol: str, bars: pd.DataFrame, train: int, test: int) -> dict:
    from ai_trading.strategy.ensemble import MarketRegime
    n = len(bars)
    results = []
    candidates = _candidate_weights()
    i = train
    while i + test < n:
        # Pick best candidate on training window
        best_sharpe = -np.inf
        best_cand = None
        for cand in candidates:
            override = {r: cand for r in MarketRegime}
            s, _ = _simulate(bars, override, i - train, i)
            if s > best_sharpe:
                best_sharpe, best_cand = s, cand
        # Test on out-of-sample window
        override = {r: best_cand for r in MarketRegime}
        oos_sharpe, oos_ret = _simulate(bars, override, i, i + test)
        results.append({
            "fold_end": int(i + test),
            "train_sharpe": round(best_sharpe, 3),
            "oos_sharpe": round(oos_sharpe, 3),
            "oos_return": round(oos_ret, 4),
            "weights": best_cand,
        })
        logger.info("[%s] fold %d: train_sharpe=%.2f oos_sharpe=%.2f oos_ret=%.2f%%",
                    symbol, i + test, best_sharpe, oos_sharpe, oos_ret * 100)
        i += test

    # Aggregate: average the best-performing weights across folds
    if not results:
        return {"symbol": symbol, "folds": [], "recommended": None}
    keys = ("trend", "momentum", "mean_reversion", "patterns", "ml")
    avg = {k: float(np.mean([f["weights"][k] for f in results])) for k in keys}
    # Renormalize
    s = sum(avg.values())
    if s > 0:
        avg = {k: round(v / s, 3) for k, v in avg.items()}
    return {"symbol": symbol, "folds": results, "recommended": avg}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", help="Comma-separated tickers (e.g. SPY,QQQ)")
    ap.add_argument("--period", default="2y", help="yfinance period (1y, 2y, 5y, max)")
    ap.add_argument("--train", type=int, default=180, help="training window (bars)")
    ap.add_argument("--test", type=int, default=60, help="test window (bars)")
    ap.add_argument("--out", default="logs/regime_weights.json")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    data = _download(symbols, args.period)

    all_results = {}
    for sym, bars in data.items():
        logger.info("\n=== %s (%d bars) ===", sym, len(bars))
        r = walk_forward(sym, bars, args.train, args.test)
        all_results[sym] = r
        print(f"\n[{sym}] recommended weights: {r['recommended']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
