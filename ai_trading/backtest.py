from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd

from ai_trading.strategy.moving_average import moving_average_signal


@dataclass(slots=True)
class BacktestResult:
    trades: list[dict]
    equity_curve: pd.DataFrame


def load_bars(symbol: str, start: str, end: str, csv_path: str | None) -> pd.DataFrame:
    if csv_path:
        bars = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date")
        return bars.sort_index()

    import yfinance as yf

    bars = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if bars.empty:
        raise ValueError(f"No historical data for {symbol}")

    bars = bars.rename(columns=str.lower)
    return bars[["open", "high", "low", "close", "volume"]]


def run_backtest(
    bars: pd.DataFrame,
    *,
    fast_ma: int,
    slow_ma: int,
    initial_cash: float,
    max_shares: int,
) -> BacktestResult:
    cash = float(initial_cash)
    shares = 0
    trades: list[dict] = []
    curve: list[dict] = []

    for i in range(len(bars)):
        window = bars.iloc[: i + 1]
        date = window.index[-1]
        close = float(window["close"].iloc[-1])

        signal = moving_average_signal(window, fast_ma, slow_ma).signal

        if signal == "BUY" and shares == 0:
            qty = min(max_shares, int(cash // close))
            if qty > 0:
                cash -= qty * close
                shares += qty
                trades.append({"date": str(date.date()), "action": "BUY", "qty": qty, "price": close})
        elif signal == "SELL" and shares > 0:
            qty = shares
            cash += qty * close
            shares = 0
            trades.append({"date": str(date.date()), "action": "SELL", "qty": qty, "price": close})

        equity = cash + (shares * close)
        curve.append({"date": date, "cash": cash, "shares": shares, "equity": equity})

    return BacktestResult(trades=trades, equity_curve=pd.DataFrame(curve).set_index("date"))


def summarize(result: BacktestResult, initial_cash: float) -> dict:
    ending = float(result.equity_curve["equity"].iloc[-1])
    total_return = (ending / initial_cash) - 1.0
    return {
        "initial_cash": initial_cash,
        "ending_equity": round(ending, 2),
        "total_return_pct": round(total_return * 100, 2),
        "trade_count": len(result.trades),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest the same long-only moving-average logic used by the paper bot."
    )
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--csv", help="Optional CSV with columns: date,open,high,low,close,volume")
    parser.add_argument("--fast-ma", type=int, default=5)
    parser.add_argument("--slow-ma", type=int, default=20)
    parser.add_argument("--initial-cash", type=float, default=10000.0)
    parser.add_argument("--max-shares", type=int, default=1)
    args = parser.parse_args()

    if args.fast_ma >= args.slow_ma:
        raise ValueError("fast-ma must be less than slow-ma")

    bars = load_bars(args.symbol, args.start, args.end, args.csv)
    result = run_backtest(
        bars,
        fast_ma=args.fast_ma,
        slow_ma=args.slow_ma,
        initial_cash=args.initial_cash,
        max_shares=max(1, args.max_shares),
    )

    stats = summarize(result, args.initial_cash)
    print("Backtest summary")
    for k, v in stats.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
