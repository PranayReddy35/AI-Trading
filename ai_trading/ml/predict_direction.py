from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


WARNING = (
    "Educational model only. Not financial advice. "
    "Use for paper trading research, not live autonomous trading."
)


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


def build_features(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = bars.copy()
    df["ret_1d"] = df["close"].pct_change(1)
    df["ret_5d"] = df["close"].pct_change(5)
    df["vol_10d"] = df["ret_1d"].rolling(10).std()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["dist_ma_20"] = (df["close"] / df["ma_20"]) - 1.0
    df["vol_avg_20"] = df["volume"].rolling(20).mean()
    df["vol_ratio_20"] = (df["volume"] / df["vol_avg_20"]) - 1.0

    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna().copy()

    features = df[["ret_1d", "ret_5d", "vol_10d", "dist_ma_20", "vol_ratio_20"]]
    target = df["target"]
    return features, target


def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, dict]:
    split = int(len(X) * 0.8)
    if split <= 0 or split >= len(X):
        raise ValueError("Not enough data after feature engineering.")

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            "Need both up/down classes in train and test windows. "
            "Use a longer date range or a different dataset."
        )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, preds))
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)

    return model, {
        "accuracy": round(accuracy, 4),
        "samples_train": len(X_train),
        "samples_test": len(X_test),
        "precision_up": round(float(report.get("1", {}).get("precision", 0.0)), 4),
        "recall_up": round(float(report.get("1", {}).get("recall", 0.0)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a simple logistic regression model for next-day direction (educational)."
    )
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--csv", help="Optional CSV with columns: date,open,high,low,close,volume")
    parser.add_argument("--save-model", help="Optional path to save the trained model")
    args = parser.parse_args()

    print(WARNING)
    bars = load_bars(args.symbol, args.start, args.end, args.csv)
    X, y = build_features(bars)
    model, metrics = train_and_evaluate(X, y)

    probabilities = model.predict_proba(X.iloc[[-1]])[0]
    prob_up = float(probabilities[1])

    print("ML evaluation")
    for k, v in metrics.items():
        print(f"- {k}: {v}")
    print(f"- latest_prob_next_day_up: {prob_up:.4f}")

    if args.save_model:
        joblib.dump(model, args.save_model)
        print(f"- model_saved_to: {args.save_model}")


if __name__ == "__main__":
    main()
