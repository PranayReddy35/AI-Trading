"""Realistic backtesting engine with transaction costs, slippage, and performance analytics.

Key improvements over basic backtest:
- Slippage modeling (market impact based on volatility and volume)
- Commission costs (configurable per-share or per-trade)
- Partial fill simulation
- Advanced performance metrics (Sharpe, Sortino, max drawdown, Calmar, profit factor)
- Comparison against buy-and-hold benchmark
- Risk-of-ruin analysis via Monte Carlo simulation
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from ai_trading.strategy.moving_average import moving_average_signal


# ---------------------------------------------------------------------------
# Transaction cost models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TransactionCosts:
    """Transaction cost configuration for realistic backtesting."""

    # Commission per share (e.g., $0.005/share for IBKR)
    commission_per_share: float = 0.005
    # Minimum commission per trade
    min_commission: float = 1.0
    # Slippage in basis points (1 bp = 0.01%)
    slippage_bps: float = 5.0
    # Additional market impact for larger orders (bps per 1% of ADV)
    market_impact_bps_per_pct_adv: float = 10.0
    # Spread cost in basis points (half-spread, paid on each side)
    spread_bps: float = 2.0

    def compute_slippage(
        self,
        price: float,
        qty: int,
        avg_volume: float,
        volatility: float,
        side: str,
    ) -> float:
        """Compute realistic slippage including market impact.

        Args:
            price: Current market price.
            qty: Number of shares.
            avg_volume: Average daily volume.
            volatility: Recent daily volatility (std of returns).
            side: "BUY" or "SELL".

        Returns:
            Slippage in dollar amount (always positive = cost).
        """
        # Base slippage (random market movement during execution)
        base_slippage = price * (self.slippage_bps / 10000)

        # Spread cost
        spread_cost = price * (self.spread_bps / 10000)

        # Market impact: larger orders relative to volume move price more
        pct_of_adv = (qty / avg_volume * 100) if avg_volume > 0 else 1.0
        impact_bps = self.market_impact_bps_per_pct_adv * pct_of_adv
        # Scale by volatility (higher vol = more impact)
        vol_multiplier = max(1.0, volatility / 0.01)  # Normalized to 1% daily vol
        market_impact = price * (impact_bps / 10000) * vol_multiplier

        total_slippage = (base_slippage + spread_cost + market_impact) * qty
        return total_slippage

    def compute_commission(self, qty: int) -> float:
        """Compute commission for a trade."""
        return max(self.min_commission, self.commission_per_share * qty)

    def total_cost(
        self,
        price: float,
        qty: int,
        avg_volume: float,
        volatility: float,
        side: str,
    ) -> float:
        """Total transaction cost (slippage + commission)."""
        slippage = self.compute_slippage(price, qty, avg_volume, volatility, side)
        commission = self.compute_commission(qty)
        return slippage + commission


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PerformanceMetrics:
    """Comprehensive performance analytics."""

    # Returns
    total_return_pct: float
    annualized_return_pct: float
    benchmark_return_pct: float  # Buy-and-hold
    excess_return_pct: float  # vs benchmark

    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float  # Annual return / max drawdown
    volatility_annual_pct: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float  # Gross profit / gross loss
    avg_trade_duration_days: float
    max_consecutive_losses: int

    # Transaction costs
    total_commissions: float
    total_slippage: float
    total_costs: float
    costs_as_pct_of_equity: float

    # Risk-adjusted
    expectancy: float  # Average profit per trade (after costs)
    kelly_fraction: float  # Optimal bet size via Kelly criterion


def compute_performance_metrics(
    equity_curve: pd.DataFrame,
    trades: list[dict],
    initial_cash: float,
    benchmark_prices: pd.Series,
    risk_free_rate: float = 0.05,
) -> PerformanceMetrics:
    """Compute comprehensive performance metrics.

    Args:
        equity_curve: DataFrame with 'equity' column indexed by date.
        trades: List of trade dicts with 'action', 'price', 'qty', 'cost', 'pnl'.
        initial_cash: Starting capital.
        benchmark_prices: Buy-and-hold price series (for benchmark comparison).
        risk_free_rate: Annual risk-free rate for Sharpe calculation.

    Returns:
        PerformanceMetrics dataclass.
    """
    equity = equity_curve["equity"].astype(float)
    n_days = len(equity)

    # --- Returns ---
    final_equity = float(equity.iloc[-1])
    total_return = (final_equity / initial_cash) - 1.0
    years = n_days / 252
    annualized_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1.0

    # Benchmark (buy-and-hold)
    if len(benchmark_prices) >= 2:
        benchmark_return = float(benchmark_prices.iloc[-1] / benchmark_prices.iloc[0]) - 1.0
    else:
        benchmark_return = 0.0
    excess_return = total_return - benchmark_return

    # --- Daily returns for risk metrics ---
    daily_returns = equity.pct_change().dropna()
    if len(daily_returns) < 2:
        daily_returns = pd.Series([0.0])

    # Volatility
    annual_vol = float(daily_returns.std() * np.sqrt(252))

    # Sharpe ratio
    daily_rf = risk_free_rate / 252
    excess_daily = daily_returns - daily_rf
    sharpe = float(excess_daily.mean() / excess_daily.std()) * np.sqrt(252) if excess_daily.std() > 0 else 0.0

    # Sortino ratio (only downside volatility)
    downside = daily_returns[daily_returns < daily_rf] - daily_rf
    downside_std = float(downside.std()) if len(downside) > 0 else 0.001
    sortino = float((daily_returns.mean() - daily_rf) / downside_std) * np.sqrt(252) if downside_std > 0 else 0.0

    # Max drawdown
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = float(drawdown.min())

    # Calmar ratio
    calmar = annualized_return / abs(max_dd) if max_dd != 0 else 0.0

    # --- Trade statistics ---
    completed_trades = _extract_roundtrip_trades(trades)
    total_trades = len(completed_trades)
    winning = [t for t in completed_trades if t["pnl"] > 0]
    losing = [t for t in completed_trades if t["pnl"] <= 0]
    win_rate = len(winning) / total_trades if total_trades > 0 else 0.0

    avg_win = np.mean([t["pnl_pct"] for t in winning]) if winning else 0.0
    avg_loss = np.mean([t["pnl_pct"] for t in losing]) if losing else 0.0

    gross_profit = sum(t["pnl"] for t in winning) if winning else 0.0
    gross_loss = abs(sum(t["pnl"] for t in losing)) if losing else 0.001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    avg_duration = np.mean([t["duration_days"] for t in completed_trades]) if completed_trades else 0.0

    # Max consecutive losses
    max_consec_losses = _max_consecutive_losses(completed_trades)

    # Transaction costs
    total_commissions = sum(t.get("commission", 0) for t in trades)
    total_slippage = sum(t.get("slippage", 0) for t in trades)
    total_costs = total_commissions + total_slippage
    costs_pct = (total_costs / initial_cash) * 100

    # Expectancy (average profit per trade after costs)
    expectancy = (gross_profit - gross_loss) / total_trades if total_trades > 0 else 0.0

    # Kelly criterion: f* = (bp - q) / b
    # where b = avg_win/avg_loss ratio, p = win rate, q = 1-p
    if avg_loss != 0 and win_rate > 0:
        b = abs(avg_win / avg_loss) if avg_loss != 0 else 1.0
        kelly = (b * win_rate - (1 - win_rate)) / b if b > 0 else 0.0
        kelly = max(0.0, min(1.0, kelly))  # Clamp to [0, 1]
    else:
        kelly = 0.0

    return PerformanceMetrics(
        total_return_pct=round(total_return * 100, 2),
        annualized_return_pct=round(annualized_return * 100, 2),
        benchmark_return_pct=round(benchmark_return * 100, 2),
        excess_return_pct=round(excess_return * 100, 2),
        sharpe_ratio=round(sharpe, 3),
        sortino_ratio=round(sortino, 3),
        max_drawdown_pct=round(max_dd * 100, 2),
        calmar_ratio=round(calmar, 3),
        volatility_annual_pct=round(annual_vol * 100, 2),
        total_trades=total_trades,
        winning_trades=len(winning),
        losing_trades=len(losing),
        win_rate=round(win_rate, 4),
        avg_win_pct=round(float(avg_win) * 100, 2),
        avg_loss_pct=round(float(avg_loss) * 100, 2),
        profit_factor=round(profit_factor, 3),
        avg_trade_duration_days=round(float(avg_duration), 1),
        max_consecutive_losses=max_consec_losses,
        total_commissions=round(total_commissions, 2),
        total_slippage=round(total_slippage, 2),
        total_costs=round(total_costs, 2),
        costs_as_pct_of_equity=round(costs_pct, 3),
        expectancy=round(expectancy, 2),
        kelly_fraction=round(kelly, 4),
    )


def _extract_roundtrip_trades(trades: list[dict]) -> list[dict]:
    """Extract completed round-trip trades (buy→sell pairs) with PnL."""
    roundtrips: list[dict] = []
    buy_entry: dict | None = None

    for t in trades:
        if t["action"] == "BUY" and buy_entry is None:
            buy_entry = t
        elif t["action"] == "SELL" and buy_entry is not None:
            entry_cost = buy_entry["price"] * buy_entry["qty"]
            exit_value = t["price"] * t["qty"]
            costs = buy_entry.get("cost", 0) + t.get("cost", 0)
            pnl = exit_value - entry_cost - costs
            pnl_pct = pnl / entry_cost if entry_cost > 0 else 0

            # Duration
            try:
                entry_date = pd.Timestamp(buy_entry["date"])
                exit_date = pd.Timestamp(t["date"])
                duration = (exit_date - entry_date).days
            except (ValueError, TypeError):
                duration = 0

            roundtrips.append({
                "entry_date": buy_entry["date"],
                "exit_date": t["date"],
                "entry_price": buy_entry["price"],
                "exit_price": t["price"],
                "qty": buy_entry["qty"],
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "duration_days": duration,
                "costs": costs,
            })
            buy_entry = None

    return roundtrips


def _max_consecutive_losses(roundtrips: list[dict]) -> int:
    """Count maximum consecutive losing trades."""
    max_streak = 0
    current_streak = 0
    for t in roundtrips:
        if t["pnl"] <= 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    return max_streak


# ---------------------------------------------------------------------------
# Kelly Criterion Position Sizing
# ---------------------------------------------------------------------------


def kelly_position_size(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    equity: float,
    price: float,
    kelly_fraction_cap: float = 0.25,  # Never bet more than 25% (half-Kelly common)
) -> int:
    """Compute optimal position size using Kelly Criterion.

    Kelly formula: f* = (bp - q) / b
    where:
        b = ratio of average win to average loss
        p = probability of winning
        q = probability of losing (1 - p)

    Args:
        win_rate: Historical win rate (0-1).
        avg_win: Average winning trade return (as fraction, e.g., 0.02 = 2%).
        avg_loss: Average losing trade return (as fraction, negative, e.g., -0.01 = -1%).
        equity: Current portfolio equity.
        price: Current share price.
        kelly_fraction_cap: Maximum Kelly fraction (for safety, use half-Kelly).

    Returns:
        Number of shares to buy (0 if Kelly says don't bet).
    """
    if win_rate <= 0 or avg_win <= 0 or avg_loss >= 0 or equity <= 0 or price <= 0:
        return 0

    b = avg_win / abs(avg_loss)
    p = win_rate
    q = 1 - p

    kelly = (b * p - q) / b

    # Apply safety cap (half-Kelly is common practice)
    kelly = max(0.0, min(kelly_fraction_cap, kelly))

    if kelly <= 0:
        return 0

    # Convert to number of shares
    position_value = equity * kelly
    shares = int(position_value / price)
    return max(0, shares)


# ---------------------------------------------------------------------------
# Monte Carlo Risk-of-Ruin Analysis
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RiskOfRuinResult:
    """Results from Monte Carlo risk-of-ruin simulation."""

    ruin_probability: float  # Probability of hitting ruin threshold
    median_final_equity: float
    percentile_5: float  # 5th percentile (worst case)
    percentile_95: float  # 95th percentile (best case)
    median_max_drawdown: float
    worst_drawdown: float
    avg_final_equity: float
    simulations: int
    ruin_threshold_pct: float


def monte_carlo_risk_of_ruin(
    trades: list[dict],
    initial_equity: float,
    n_simulations: int = 10000,
    n_trades_forward: int = 252,
    ruin_threshold_pct: float = 50.0,
) -> RiskOfRuinResult:
    """Run Monte Carlo simulation to estimate risk of ruin.

    Randomly resamples historical trade returns to simulate future
    equity curves and estimate probability of catastrophic loss.

    Args:
        trades: List of completed round-trip trade dicts with 'pnl_pct'.
        initial_equity: Starting equity.
        n_simulations: Number of Monte Carlo paths to simulate.
        n_trades_forward: Number of trades to simulate forward.
        ruin_threshold_pct: Consider "ruined" if equity drops below this % of initial.

    Returns:
        RiskOfRuinResult with probability estimates.
    """
    roundtrips = _extract_roundtrip_trades(trades) if trades and "pnl_pct" not in trades[0] else trades
    trade_returns = [t["pnl_pct"] for t in roundtrips if "pnl_pct" in t]

    if not trade_returns:
        return RiskOfRuinResult(
            ruin_probability=0.0,
            median_final_equity=initial_equity,
            percentile_5=initial_equity,
            percentile_95=initial_equity,
            median_max_drawdown=0.0,
            worst_drawdown=0.0,
            avg_final_equity=initial_equity,
            simulations=0,
            ruin_threshold_pct=ruin_threshold_pct,
        )

    returns_array = np.array(trade_returns)
    ruin_threshold = initial_equity * (ruin_threshold_pct / 100)
    rng = np.random.default_rng(42)

    final_equities = np.zeros(n_simulations)
    max_drawdowns = np.zeros(n_simulations)
    ruin_count = 0

    for i in range(n_simulations):
        # Randomly sample trade returns (with replacement)
        sampled_returns = rng.choice(returns_array, size=n_trades_forward, replace=True)

        equity = initial_equity
        peak = initial_equity
        max_dd = 0.0
        ruined = False

        for ret in sampled_returns:
            equity *= (1 + ret)
            peak = max(peak, equity)
            dd = (equity - peak) / peak
            max_dd = min(max_dd, dd)

            if equity < ruin_threshold:
                ruined = True
                break

        final_equities[i] = equity
        max_drawdowns[i] = max_dd
        if ruined:
            ruin_count += 1

    return RiskOfRuinResult(
        ruin_probability=round(ruin_count / n_simulations, 4),
        median_final_equity=round(float(np.median(final_equities)), 2),
        percentile_5=round(float(np.percentile(final_equities, 5)), 2),
        percentile_95=round(float(np.percentile(final_equities, 95)), 2),
        median_max_drawdown=round(float(np.median(max_drawdowns)) * 100, 2),
        worst_drawdown=round(float(np.min(max_drawdowns)) * 100, 2),
        avg_final_equity=round(float(np.mean(final_equities)), 2),
        simulations=n_simulations,
        ruin_threshold_pct=ruin_threshold_pct,
    )


# ---------------------------------------------------------------------------
# Realistic Backtest Engine
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RealisticBacktestResult:
    """Results from realistic backtesting."""

    trades: list[dict]
    equity_curve: pd.DataFrame
    metrics: PerformanceMetrics
    risk_of_ruin: RiskOfRuinResult
    costs: TransactionCosts


def run_realistic_backtest(
    bars: pd.DataFrame,
    *,
    fast_ma: int = 5,
    slow_ma: int = 20,
    initial_cash: float = 10000.0,
    max_shares: int = 10,
    costs: TransactionCosts | None = None,
    use_kelly_sizing: bool = True,
    strategy: Literal["ma", "ensemble"] = "ma",
) -> RealisticBacktestResult:
    """Run backtest with realistic transaction costs and position sizing.

    Args:
        bars: OHLCV DataFrame.
        fast_ma: Fast MA period (for MA strategy).
        slow_ma: Slow MA period (for MA strategy).
        initial_cash: Starting capital.
        max_shares: Maximum position size cap.
        costs: Transaction cost model (default realistic costs).
        use_kelly_sizing: Use Kelly criterion for position sizing.
        strategy: "ma" (moving average) or "ensemble" (multi-strategy).

    Returns:
        RealisticBacktestResult with trades, equity curve, and metrics.
    """
    if costs is None:
        costs = TransactionCosts()

    cash = float(initial_cash)
    shares = 0
    entry_price = 0.0
    trades: list[dict] = []
    curve: list[dict] = []

    # Track trade history for Kelly sizing
    completed_returns: list[float] = []

    # Pre-compute volume averages for slippage
    volumes = bars["volume"].astype(float)
    vol_avg_20 = volumes.rolling(20).mean()

    # Pre-compute volatility for slippage
    returns = bars["close"].astype(float).pct_change()
    volatility_20 = returns.rolling(20).std()

    for i in range(max(slow_ma + 5, 50), len(bars)):
        window = bars.iloc[: i + 1]
        date = window.index[-1]
        close = float(window["close"].iloc[-1])
        avg_vol = float(vol_avg_20.iloc[i]) if pd.notna(vol_avg_20.iloc[i]) else 1000000
        curr_vol = float(volatility_20.iloc[i]) if pd.notna(volatility_20.iloc[i]) else 0.01

        # Get signal based on strategy type
        if strategy == "ensemble":
            from ai_trading.strategy.ensemble import compute_ensemble_signal

            ensemble_result = compute_ensemble_signal(window)
            signal = ensemble_result.signal
        else:
            signal = moving_average_signal(window, fast_ma, slow_ma).signal

        if signal == "BUY" and shares == 0:
            # Determine position size
            if use_kelly_sizing and len(completed_returns) >= 10:
                wins = [r for r in completed_returns if r > 0]
                losses = [r for r in completed_returns if r <= 0]
                if wins and losses:
                    win_rate = len(wins) / len(completed_returns)
                    avg_win = np.mean(wins)
                    avg_loss = np.mean(losses)
                    qty = kelly_position_size(
                        win_rate=win_rate,
                        avg_win=avg_win,
                        avg_loss=avg_loss,
                        equity=cash + shares * close,
                        price=close,
                        kelly_fraction_cap=0.25,
                    )
                    qty = min(qty, max_shares)
                else:
                    qty = min(max_shares, int(cash * 0.1 / close))  # Conservative 10%
            else:
                # Start conservative until we have enough trade history
                qty = min(max_shares, int(cash * 0.1 / close))

            if qty <= 0:
                qty = 1  # Minimum 1 share

            # Apply transaction costs
            slippage = costs.compute_slippage(close, qty, avg_vol, curr_vol, "BUY")
            commission = costs.compute_commission(qty)
            total_cost = slippage + commission
            effective_price = close + (slippage / qty)  # Slippage raises effective buy price

            total_outlay = qty * effective_price + commission
            if total_outlay > cash:
                # Reduce qty to fit budget
                qty = int((cash - commission) / effective_price)
                if qty <= 0:
                    curve.append({"date": date, "cash": cash, "shares": shares, "equity": cash + shares * close})
                    continue
                slippage = costs.compute_slippage(close, qty, avg_vol, curr_vol, "BUY")
                commission = costs.compute_commission(qty)
                total_cost = slippage + commission
                effective_price = close + (slippage / qty)
                total_outlay = qty * effective_price + commission

            cash -= total_outlay
            shares += qty
            entry_price = effective_price

            trades.append({
                "date": str(date.date()) if hasattr(date, "date") else str(date),
                "action": "BUY",
                "qty": qty,
                "price": close,
                "effective_price": round(effective_price, 4),
                "slippage": round(slippage, 4),
                "commission": round(commission, 4),
                "cost": round(total_cost, 4),
            })

        elif signal == "SELL" and shares > 0:
            qty = shares

            # Apply transaction costs
            slippage = costs.compute_slippage(close, qty, avg_vol, curr_vol, "SELL")
            commission = costs.compute_commission(qty)
            total_cost = slippage + commission
            effective_price = close - (slippage / qty)  # Slippage lowers effective sell price

            proceeds = qty * effective_price - commission
            pnl = proceeds - (entry_price * qty)
            pnl_pct = (effective_price / entry_price - 1) if entry_price > 0 else 0

            cash += proceeds
            completed_returns.append(pnl_pct)
            shares = 0

            trades.append({
                "date": str(date.date()) if hasattr(date, "date") else str(date),
                "action": "SELL",
                "qty": qty,
                "price": close,
                "effective_price": round(effective_price, 4),
                "slippage": round(slippage, 4),
                "commission": round(commission, 4),
                "cost": round(total_cost, 4),
                "pnl": round(pnl, 4),
                "pnl_pct": round(pnl_pct, 6),
            })

        equity = cash + (shares * close)
        curve.append({"date": date, "cash": cash, "shares": shares, "equity": equity})

    equity_df = pd.DataFrame(curve).set_index("date")

    # Compute performance metrics
    benchmark_prices = bars["close"].iloc[max(slow_ma + 5, 50):]
    metrics = compute_performance_metrics(
        equity_curve=equity_df,
        trades=trades,
        initial_cash=initial_cash,
        benchmark_prices=benchmark_prices,
    )

    # Risk-of-ruin analysis
    risk_result = monte_carlo_risk_of_ruin(
        trades=trades,
        initial_equity=initial_cash,
        n_simulations=10000,
        n_trades_forward=min(252, max(50, len(completed_returns) * 2)),
    )

    return RealisticBacktestResult(
        trades=trades,
        equity_curve=equity_df,
        metrics=metrics,
        risk_of_ruin=risk_result,
        costs=costs,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Realistic backtest with transaction costs, Kelly sizing, and Monte Carlo risk analysis."
    )
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--csv", help="Optional CSV with columns: date,open,high,low,close,volume")
    parser.add_argument("--fast-ma", type=int, default=10)
    parser.add_argument("--slow-ma", type=int, default=30)
    parser.add_argument("--initial-cash", type=float, default=100000.0)
    parser.add_argument("--max-shares", type=int, default=100)
    parser.add_argument("--strategy", choices=["ma", "ensemble"], default="ensemble")
    parser.add_argument("--commission-per-share", type=float, default=0.005)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--no-kelly", action="store_true", help="Disable Kelly position sizing")
    args = parser.parse_args()

    from ai_trading.ml.predict_direction import load_bars

    print("=" * 70)
    print("  REALISTIC BACKTEST ENGINE")
    print("  Includes: slippage, commissions, market impact, Kelly sizing")
    print("  NOT financial advice. For research and validation only.")
    print("=" * 70)

    bars = load_bars(args.symbol, args.start, args.end, args.csv)
    print(f"\nData: {args.symbol} from {bars.index[0].date()} to {bars.index[-1].date()}")
    print(f"Total bars: {len(bars)}")

    costs = TransactionCosts(
        commission_per_share=args.commission_per_share,
        slippage_bps=args.slippage_bps,
    )

    result = run_realistic_backtest(
        bars,
        fast_ma=args.fast_ma,
        slow_ma=args.slow_ma,
        initial_cash=args.initial_cash,
        max_shares=args.max_shares,
        costs=costs,
        use_kelly_sizing=not args.no_kelly,
        strategy=args.strategy,
    )

    m = result.metrics
    print(f"\n{'='*70}")
    print(f"  PERFORMANCE SUMMARY (Strategy: {args.strategy})")
    print(f"{'='*70}")
    print(f"\n  Returns:")
    print(f"    Total return: {m.total_return_pct:+.2f}%")
    print(f"    Annualized return: {m.annualized_return_pct:+.2f}%")
    print(f"    Benchmark (buy & hold): {m.benchmark_return_pct:+.2f}%")
    print(f"    Excess return vs benchmark: {m.excess_return_pct:+.2f}%")

    print(f"\n  Risk Metrics:")
    print(f"    Sharpe ratio: {m.sharpe_ratio:.3f} {'✓' if m.sharpe_ratio > 1.5 else '✗' if m.sharpe_ratio < 1.0 else '~'}")
    print(f"    Sortino ratio: {m.sortino_ratio:.3f}")
    print(f"    Max drawdown: {m.max_drawdown_pct:.2f}%")
    print(f"    Calmar ratio: {m.calmar_ratio:.3f}")
    print(f"    Annual volatility: {m.volatility_annual_pct:.2f}%")

    print(f"\n  Trade Statistics:")
    print(f"    Total trades: {m.total_trades}")
    print(f"    Win rate: {m.win_rate:.1%}")
    print(f"    Avg win: {m.avg_win_pct:+.2f}%")
    print(f"    Avg loss: {m.avg_loss_pct:+.2f}%")
    print(f"    Profit factor: {m.profit_factor:.3f} {'✓' if m.profit_factor > 1.5 else '✗' if m.profit_factor < 1.0 else '~'}")
    print(f"    Avg trade duration: {m.avg_trade_duration_days:.1f} days")
    print(f"    Max consecutive losses: {m.max_consecutive_losses}")
    print(f"    Expectancy per trade: ${m.expectancy:.2f}")

    print(f"\n  Transaction Costs:")
    print(f"    Total commissions: ${m.total_commissions:.2f}")
    print(f"    Total slippage: ${m.total_slippage:.2f}")
    print(f"    Total costs: ${m.total_costs:.2f} ({m.costs_as_pct_of_equity:.3f}% of capital)")

    print(f"\n  Position Sizing:")
    print(f"    Kelly fraction: {m.kelly_fraction:.4f} ({m.kelly_fraction*100:.1f}% of equity per trade)")

    # Risk of Ruin
    r = result.risk_of_ruin
    print(f"\n{'='*70}")
    print(f"  RISK-OF-RUIN ANALYSIS (Monte Carlo, {r.simulations:,} simulations)")
    print(f"{'='*70}")
    print(f"    Ruin probability (equity < {r.ruin_threshold_pct}%): {r.ruin_probability:.2%}")
    print(f"    Median final equity: ${r.median_final_equity:,.2f}")
    print(f"    5th percentile (worst case): ${r.percentile_5:,.2f}")
    print(f"    95th percentile (best case): ${r.percentile_95:,.2f}")
    print(f"    Median max drawdown: {r.median_max_drawdown:.2f}%")
    print(f"    Worst drawdown: {r.worst_drawdown:.2f}%")

    # Overall readiness assessment
    print(f"\n{'='*70}")
    print(f"  LIVE TRADING READINESS ASSESSMENT")
    print(f"{'='*70}")
    checks = [
        ("Sharpe > 1.5", m.sharpe_ratio > 1.5),
        ("Profit factor > 1.5", m.profit_factor > 1.5),
        ("Beats benchmark", m.excess_return_pct > 0),
        ("Max drawdown < 20%", m.max_drawdown_pct > -20),
        ("Win rate > 50%", m.win_rate > 0.5),
        ("Ruin probability < 5%", r.ruin_probability < 0.05),
        ("Kelly fraction > 0", m.kelly_fraction > 0),
        ("Positive expectancy", m.expectancy > 0),
    ]
    passed = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(f"    {'✓' if ok else '✗'} {name}")
    print(f"\n    Score: {passed}/{len(checks)} checks passed")
    if passed >= 7:
        print("    → Strategy shows strong edge. Consider paper trading validation.")
    elif passed >= 5:
        print("    → Marginal edge. Needs more optimization before live trading.")
    else:
        print("    → NOT ready for live trading. Strategy needs fundamental improvements.")


if __name__ == "__main__":
    main()
