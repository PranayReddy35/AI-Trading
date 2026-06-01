"""Backtest runner for the full ensemble strategy (MA + regime detection + ML).

Usage:
    python -m ai_trading.backtest_ensemble --symbol SPY --start 2020-01-01 --end 2025-01-01
    python -m ai_trading.backtest_ensemble --symbol SPY --optimize
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf


# ── Performance metrics ───────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    symbol: str
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    total_trades: int
    profit_factor: float
    calmar_ratio: float
    benchmark_return_pct: float
    alpha_pct: float
    params: dict = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"Symbol:              {self.symbol}",
            f"Total Return:        {self.total_return_pct:+.2f}%",
            f"Annualized Return:   {self.annualized_return_pct:+.2f}%",
            f"Benchmark (B&H):     {self.benchmark_return_pct:+.2f}%",
            f"Alpha:               {self.alpha_pct:+.2f}%",
            f"Sharpe Ratio:        {self.sharpe_ratio:.3f}",
            f"Sortino Ratio:       {self.sortino_ratio:.3f}",
            f"Calmar Ratio:        {self.calmar_ratio:.3f}",
            f"Max Drawdown:        {self.max_drawdown_pct:.2f}%",
            f"Win Rate:            {self.win_rate_pct:.1f}%",
            f"Total Trades:        {self.total_trades}",
            f"Profit Factor:       {self.profit_factor:.3f}",
        ]
        if self.params:
            lines.append(f"Params:              {self.params}")
        return "\n".join(lines)


def _compute_metrics(equity_curve: pd.Series, trades: list[float], benchmark: pd.Series) -> dict:
    """Compute performance metrics from an equity curve."""
    returns = equity_curve.pct_change().dropna()
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100

    n_years = len(equity_curve) / 252
    annualized = ((1 + total_return / 100) ** (1 / max(n_years, 0.01)) - 1) * 100

    # Sharpe (annualized, risk-free ~ 0)
    sharpe = float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0

    # Sortino (downside deviation)
    neg_returns = returns[returns < 0]
    down_std = neg_returns.std() if len(neg_returns) > 0 else 1e-9
    sortino = float(returns.mean() / down_std * np.sqrt(252)) if down_std > 0 else 0.0

    # Max drawdown
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max * 100
    max_dd = float(drawdown.min())

    # Calmar
    calmar = annualized / abs(max_dd) if max_dd != 0 else 0.0

    # Win rate and profit factor
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0.0
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 1e-9
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    bmark_return = (benchmark.iloc[-1] / benchmark.iloc[0] - 1) * 100

    return {
        "total_return_pct": round(total_return, 2),
        "annualized_return_pct": round(annualized, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate_pct": round(win_rate, 1),
        "total_trades": len(trades),
        "profit_factor": round(profit_factor, 3),
        "calmar_ratio": round(calmar, 3),
        "benchmark_return_pct": round(float(bmark_return), 2),
        "alpha_pct": round(total_return - float(bmark_return), 2),
    }


# ── Signal generation ─────────────────────────────────────────────────────────

def _ma_signal(close: pd.Series, fast: int, slow: int) -> pd.Series:
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    signal = pd.Series(0, index=close.index)
    signal[fast_ma > slow_ma] = 1   # bullish
    signal[fast_ma < slow_ma] = -1  # bearish
    return signal


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta).clip(lower=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _ensemble_signal(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    fast_ma: int = 5,
    slow_ma: int = 20,
    rsi_period: int = 14,
    rsi_oversold: float = 35,
    rsi_overbought: float = 65,
    vol_lookback: int = 20,
) -> pd.Series:
    """Combine MA, RSI, and volatility regime into a single signal.

    Returns: +1 = BUY, -1 = SELL, 0 = HOLD
    """
    ma_sig = _ma_signal(close, fast_ma, slow_ma)
    rsi_val = _rsi(close, rsi_period)
    vol = close.pct_change().rolling(vol_lookback).std()
    vol_percentile = vol.rolling(252, min_periods=60).rank(pct=True)

    # Bollinger band position
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_pos = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std.replace(0, np.nan))

    # Combined score: MA trend + RSI mean-reversion + vol filter
    score = pd.Series(0.0, index=close.index)
    score += ma_sig * 0.5                         # trend component
    score += ((rsi_val < rsi_oversold).astype(float) - (rsi_val > rsi_overbought).astype(float)) * 0.3
    score += ((bb_pos < 0.2).astype(float) - (bb_pos > 0.8).astype(float)) * 0.2

    # High-vol regime reduces signal confidence
    score *= (1 - vol_percentile.clip(0, 0.5))

    signal = pd.Series(0, index=close.index)
    signal[score > 0.3] = 1
    signal[score < -0.3] = -1
    return signal


# ── Core backtest engine ──────────────────────────────────────────────────────

def run_ensemble_backtest(
    bars: pd.DataFrame,
    symbol: str = "SPY",
    fast_ma: int = 5,
    slow_ma: int = 20,
    rsi_period: int = 14,
    rsi_oversold: float = 35,
    rsi_overbought: float = 65,
    initial_capital: float = 10_000.0,
    commission_per_share: float = 0.005,
    slippage_bps: float = 5.0,
) -> BacktestResult:
    """Run ensemble strategy backtest on OHLCV bars."""
    close = bars["close"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    volume = bars["volume"].astype(float)

    signals = _ensemble_signal(close, high, low, volume, fast_ma, slow_ma, rsi_period, rsi_oversold, rsi_overbought)

    capital = initial_capital
    position = 0
    entry_price = 0.0
    equity_curve = []
    trades: list[float] = []

    for i in range(1, len(close)):
        price = float(close.iloc[i])
        sig = int(signals.iloc[i - 1])  # use prior bar signal to avoid look-ahead

        slippage = price * (slippage_bps / 10_000)

        if sig == 1 and position == 0 and capital > price:
            shares = int(capital / (price + slippage))
            cost = shares * (price + slippage) + max(shares * commission_per_share, 1.0)
            if shares > 0 and cost <= capital:
                capital -= cost
                position = shares
                entry_price = price + slippage

        elif sig == -1 and position > 0:
            proceeds = position * (price - slippage) - max(position * commission_per_share, 1.0)
            trade_pnl = proceeds - position * entry_price
            trades.append(float(trade_pnl))
            capital += position * entry_price + trade_pnl
            position = 0
            entry_price = 0.0

        equity_curve.append(capital + position * price)

    if position > 0:
        final_price = float(close.iloc[-1])
        proceeds = position * (final_price - final_price * slippage_bps / 10_000)
        trade_pnl = proceeds - position * entry_price
        trades.append(float(trade_pnl))
        equity_curve[-1] = capital + position * final_price

    equity_series = pd.Series(equity_curve, index=close.index[1:])
    benchmark = close.iloc[1:]

    metrics = _compute_metrics(equity_series, trades, benchmark)

    return BacktestResult(
        symbol=symbol,
        params={"fast_ma": fast_ma, "slow_ma": slow_ma, "rsi_period": rsi_period,
                "rsi_oversold": rsi_oversold, "rsi_overbought": rsi_overbought},
        **metrics,
    )


# ── Parameter optimization ────────────────────────────────────────────────────

def optimize_parameters(
    bars: pd.DataFrame,
    symbol: str = "SPY",
    fast_ma_range: list[int] | None = None,
    slow_ma_range: list[int] | None = None,
    rsi_oversold_range: list[float] | None = None,
    rsi_overbought_range: list[float] | None = None,
    metric: str = "sharpe_ratio",
) -> tuple[BacktestResult, pd.DataFrame]:
    """Grid search over parameter combinations.

    Args:
        bars: OHLCV DataFrame.
        symbol: Ticker symbol.
        fast_ma_range: List of fast MA periods to test.
        slow_ma_range: List of slow MA periods to test.
        rsi_oversold_range: RSI oversold thresholds.
        rsi_overbought_range: RSI overbought thresholds.
        metric: Metric to optimize ("sharpe_ratio", "total_return_pct", "calmar_ratio").

    Returns:
        (best_result, full_results_DataFrame)
    """
    fast_ma_range = fast_ma_range or [3, 5, 8, 10]
    slow_ma_range = slow_ma_range or [15, 20, 30, 50]
    rsi_oversold_range = rsi_oversold_range or [30, 35, 40]
    rsi_overbought_range = rsi_overbought_range or [60, 65, 70]

    results = []
    combos = list(itertools.product(fast_ma_range, slow_ma_range, rsi_oversold_range, rsi_overbought_range))
    # Filter invalid combos
    combos = [(f, s, ro, rob) for f, s, ro, rob in combos if f < s]

    print(f"Testing {len(combos)} parameter combinations...")

    for i, (fast, slow, oversold, overbought) in enumerate(combos):
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(combos)}...")
        try:
            r = run_ensemble_backtest(
                bars, symbol=symbol,
                fast_ma=fast, slow_ma=slow,
                rsi_oversold=oversold, rsi_overbought=overbought,
            )
            results.append({
                "fast_ma": fast, "slow_ma": slow,
                "rsi_oversold": oversold, "rsi_overbought": overbought,
                "total_return_pct": r.total_return_pct,
                "annualized_return_pct": r.annualized_return_pct,
                "sharpe_ratio": r.sharpe_ratio,
                "sortino_ratio": r.sortino_ratio,
                "max_drawdown_pct": r.max_drawdown_pct,
                "calmar_ratio": r.calmar_ratio,
                "win_rate_pct": r.win_rate_pct,
                "total_trades": r.total_trades,
                "alpha_pct": r.alpha_pct,
            })
        except Exception:
            pass

    df = pd.DataFrame(results).sort_values(metric, ascending=False)

    # Rerun best params to get full result
    best = df.iloc[0]
    best_result = run_ensemble_backtest(
        bars, symbol=symbol,
        fast_ma=int(best["fast_ma"]),
        slow_ma=int(best["slow_ma"]),
        rsi_oversold=float(best["rsi_oversold"]),
        rsi_overbought=float(best["rsi_overbought"]),
    )
    return best_result, df


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest ensemble strategy")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--fast-ma", type=int, default=5)
    parser.add_argument("--slow-ma", type=int, default=20)
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--optimize", action="store_true", help="Run grid search parameter optimization")
    parser.add_argument("--optimize-metric", default="sharpe_ratio",
                        choices=["sharpe_ratio", "total_return_pct", "calmar_ratio", "sortino_ratio"])
    parser.add_argument("--top-n", type=int, default=10, help="Show top N results in optimization")
    args = parser.parse_args()

    print("=" * 60)
    print("  ENSEMBLE STRATEGY BACKTEST")
    print("  NOT financial advice. Use for research only.")
    print("=" * 60)

    print(f"\nDownloading {args.symbol} from {args.start} to {args.end}...")
    raw = yf.download(args.symbol, start=args.start, end=args.end, progress=False)
    if raw.empty:
        print("ERROR: No data downloaded.")
        return

    # Normalize column names
    raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in raw.columns]
    raw = raw.rename(columns={"adj close": "close"}) if "adj close" in raw.columns else raw
    print(f"Loaded {len(raw)} bars.\n")

    if args.optimize:
        print(f"Running parameter optimization (metric: {args.optimize_metric})...\n")
        best, df = optimize_parameters(raw, symbol=args.symbol, metric=args.optimize_metric)
        print(f"\nTop {args.top_n} parameter sets by {args.optimize_metric}:")
        print(df.head(args.top_n).to_string(index=False))
        print(f"\n{'='*60}")
        print("BEST RESULT:")
        print(best)
    else:
        result = run_ensemble_backtest(
            raw, symbol=args.symbol,
            fast_ma=args.fast_ma, slow_ma=args.slow_ma,
            initial_capital=args.capital,
        )
        print(result)


if __name__ == "__main__":
    main()
