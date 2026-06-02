"""Cost-aware ensemble backtest runner.

Wraps the regime-aware ensemble signal with realistic costs (slippage,
spread, commission) and reports Sharpe / max drawdown / profit factor /
expectancy. Designed to be the truth-arbiter before going live.

Usage:
    python -m ai_trading.backtest.ensemble_cost_aware SPY,QQQ --period 2y \\
        --commission 0.005 --slip-bps 5 --spread-bps 2
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ai_trading.strategy.ensemble import compute_ensemble_signal


@dataclass(slots=True)
class CostConfig:
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    slippage_bps: float = 5.0
    spread_bps: float = 2.0


@dataclass(slots=True)
class BacktestReport:
    symbol: str
    period: str
    bars: int
    trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    expectancy_pct: float
    total_return_pct: float
    buy_hold_pct: float
    sharpe: float
    max_drawdown_pct: float
    total_costs: float


def _apply_costs(notional: float, qty: int, side: str, cfg: CostConfig) -> float:
    commission = max(cfg.min_commission, qty * cfg.commission_per_share)
    slip = abs(notional) * cfg.slippage_bps / 1e4
    half_spread = abs(notional) * cfg.spread_bps / 1e4
    return commission + slip + half_spread


def backtest_symbol(
    bars: pd.DataFrame,
    *,
    symbol: str,
    period: str,
    cfg: CostConfig,
    qty: int = 10,
    warmup: int = 100,
    starting_cash: float = 100_000.0,
) -> BacktestReport:
    """Walk forward over bars, generating ensemble signals each day, simulating
    market-on-open fills next bar with costs.
    """
    bars = bars.copy()
    bars.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in bars.columns]
    n = len(bars)
    if n <= warmup + 10:
        raise ValueError(f"Need > {warmup + 10} bars, got {n}")

    position = 0
    entry_px = 0.0
    cash = starting_cash
    equity_curve = [starting_cash]
    trades: list[dict] = []
    total_costs = 0.0

    for i in range(warmup, n - 1):
        window = bars.iloc[: i + 1]
        try:
            es = compute_ensemble_signal(window)
        except Exception:
            continue
        next_open = float(bars["open"].iloc[i + 1]) if "open" in bars.columns else float(bars["close"].iloc[i + 1])
        sig = es.signal

        if sig == "BUY" and position == 0:
            notional = next_open * qty
            cost = _apply_costs(notional, qty, "BUY", cfg)
            total_costs += cost
            cash -= notional + cost
            position = qty
            entry_px = next_open + cost / qty
        elif sig == "SELL" and position > 0:
            notional = next_open * position
            cost = _apply_costs(notional, position, "SELL", cfg)
            total_costs += cost
            cash += notional - cost
            pnl_pct = (next_open - entry_px) / entry_px * 100.0
            trades.append({"entry": entry_px, "exit": next_open, "pnl_pct": pnl_pct})
            position = 0
            entry_px = 0.0

        mark = cash + position * float(bars["close"].iloc[i + 1])
        equity_curve.append(mark)

    # Close any final open position at last close
    if position > 0:
        last = float(bars["close"].iloc[-1])
        notional = last * position
        cost = _apply_costs(notional, position, "SELL", cfg)
        total_costs += cost
        cash += notional - cost
        pnl_pct = (last - entry_px) / entry_px * 100.0
        trades.append({"entry": entry_px, "exit": last, "pnl_pct": pnl_pct})

    # Metrics
    final_equity = cash + position * float(bars["close"].iloc[-1])
    total_ret = (final_equity - starting_cash) / starting_cash * 100.0
    buy_hold = (float(bars["close"].iloc[-1]) - float(bars["close"].iloc[warmup])) / float(bars["close"].iloc[warmup]) * 100.0
    wins = [t["pnl_pct"] for t in trades if t["pnl_pct"] > 0]
    losses = [t["pnl_pct"] for t in trades if t["pnl_pct"] <= 0]
    win_rate = len(wins) / len(trades) if trades else 0.0
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    pf_num = sum(wins)
    pf_den = abs(sum(losses)) or 1e-9
    profit_factor = pf_num / pf_den
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    ec = np.array(equity_curve)
    rets = pd.Series(ec).pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    running_peak = pd.Series(ec).cummax().replace(0, np.nan)
    dd = (pd.Series(ec) - running_peak) / running_peak * 100.0
    max_dd = float(dd.min()) if not dd.dropna().empty else 0.0

    return BacktestReport(
        symbol=symbol, period=period, bars=n, trades=len(trades),
        win_rate=round(win_rate, 4), avg_win_pct=round(avg_win, 3),
        avg_loss_pct=round(avg_loss, 3), profit_factor=round(profit_factor, 3),
        expectancy_pct=round(expectancy, 4), total_return_pct=round(total_ret, 2),
        buy_hold_pct=round(buy_hold, 2), sharpe=round(sharpe, 3),
        max_drawdown_pct=round(max_dd, 2), total_costs=round(total_costs, 2),
    )


def _fetch_yf(symbol: str, period: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(symbol, period=period, progress=False, auto_adjust=False)
    if df.empty:
        return df
    df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("symbols", help="Comma-separated tickers (e.g. SPY,QQQ)")
    p.add_argument("--period", default="2y")
    p.add_argument("--commission", type=float, default=0.005)
    p.add_argument("--slip-bps", type=float, default=5.0)
    p.add_argument("--spread-bps", type=float, default=2.0)
    p.add_argument("--qty", type=int, default=10)
    p.add_argument("--out", default=None, help="Optional JSON output path")
    args = p.parse_args()

    cfg = CostConfig(
        commission_per_share=args.commission,
        slippage_bps=args.slip_bps,
        spread_bps=args.spread_bps,
    )
    reports: list[BacktestReport] = []
    for sym in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
        bars = _fetch_yf(sym, args.period)
        if bars.empty:
            print(f"{sym}: no data")
            continue
        try:
            rpt = backtest_symbol(bars, symbol=sym, period=args.period, cfg=cfg, qty=args.qty)
            reports.append(rpt)
            print(f"\n{sym}  ({rpt.period}, {rpt.bars} bars)")
            print(f"  trades        : {rpt.trades}")
            print(f"  win rate      : {rpt.win_rate*100:.1f}%")
            print(f"  avg win/loss  : {rpt.avg_win_pct:+.2f}% / {rpt.avg_loss_pct:+.2f}%")
            print(f"  profit factor : {rpt.profit_factor:.2f}")
            print(f"  expectancy    : {rpt.expectancy_pct:+.3f}% per trade")
            print(f"  total return  : {rpt.total_return_pct:+.2f}%  (B&H {rpt.buy_hold_pct:+.2f}%)")
            print(f"  Sharpe        : {rpt.sharpe:.2f}")
            print(f"  max drawdown  : {rpt.max_drawdown_pct:.2f}%")
            print(f"  total costs   : ${rpt.total_costs:.2f}")
        except Exception as exc:
            print(f"{sym}: backtest failed: {exc}")

    if args.out and reports:
        with open(args.out, "w") as f:
            json.dump([asdict(r) for r in reports], f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
