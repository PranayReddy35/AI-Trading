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

# Sentiment feature columns (added when sentiment data is available)
SENTIMENT_FEATURES = [
    "sentiment_score_1d",
    "sentiment_score_3d",
    "news_volume_1d",
    "news_volume_3d",
    "sentiment_momentum",
    "positive_ratio",
    "negative_ratio",
]

BASE_FEATURES = ["ret_1d", "ret_5d", "vol_10d", "dist_ma_20", "vol_ratio_20"]


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


def build_features(bars: pd.DataFrame, sentiment_data: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.Series]:
    df = bars.copy()
    df["ret_1d"] = df["close"].pct_change(1)
    df["ret_5d"] = df["close"].pct_change(5)
    df["vol_10d"] = df["ret_1d"].rolling(10).std()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["dist_ma_20"] = (df["close"] / df["ma_20"]) - 1.0
    df["vol_avg_20"] = df["volume"].rolling(20).mean()
    df["vol_ratio_20"] = (df["volume"] / df["vol_avg_20"]) - 1.0

    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # Merge sentiment features if available
    feature_cols = list(BASE_FEATURES)
    if sentiment_data is not None and not sentiment_data.empty:
        # Align sentiment data with price data by date index
        for col in SENTIMENT_FEATURES:
            if col in sentiment_data.columns:
                df[col] = sentiment_data[col].reindex(df.index, method="ffill")
                feature_cols.append(col)
        print(f"  Sentiment features added: {[c for c in SENTIMENT_FEATURES if c in df.columns]}")

    df = df.dropna().copy()

    features = df[feature_cols]
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
    parser.add_argument(
        "--with-sentiment",
        action="store_true",
        help="Fetch live news sentiment and add as features for the latest prediction",
    )
    parser.add_argument(
        "--news-provider",
        default="rss",
        choices=["rss", "alphavantage"],
        help="News provider: 'rss' (Google News, free) or 'alphavantage' (free tier, needs key)",
    )
    parser.add_argument("--news-api-key", default="", help="API key for Alpha Vantage (optional)")
    args = parser.parse_args()

    print(WARNING)
    bars = load_bars(args.symbol, args.start, args.end, args.csv)

    # Build features (without sentiment for training on historical data)
    X, y = build_features(bars)
    model, metrics = train_and_evaluate(X, y)

    probabilities = model.predict_proba(X.iloc[[-1]])[0]
    prob_up = float(probabilities[1])

    print("ML evaluation (technical features)")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"  latest_prob_next_day_up: {prob_up:.4f}")

    # If sentiment requested, fetch live news and show sentiment analysis
    if args.with_sentiment:
        print("\nSentiment analysis (live news)")
        try:
            from ai_trading.data.news_sentiment import fetch_news
            from ai_trading.ml.sentiment import SentimentAnalyzer, compute_sentiment_features

            articles = fetch_news(
                symbol=args.symbol,
                provider=args.news_provider,
                api_key=args.news_api_key,
                max_articles=20,
            )

            if articles:
                analyzer = SentimentAnalyzer()
                aggregate = analyzer.aggregate_sentiment(articles)
                features = compute_sentiment_features(articles)

                print(f"  articles_found: {aggregate.num_articles}")
                print(f"  aggregate_sentiment: {aggregate.score:.4f}")
                print(f"  positive/negative/neutral: "
                      f"{aggregate.num_positive}/{aggregate.num_negative}/{aggregate.num_neutral}")
                print(f"  sentiment_momentum: {features['sentiment_momentum']:.4f}")

                if aggregate.category_breakdown:
                    print("  category_breakdown:")
                    for cat, score in sorted(aggregate.category_breakdown.items()):
                        print(f"    {cat}: {score:.4f}")

                # Combine technical + sentiment for a final signal
                sentiment_adjustment = aggregate.score * 0.1  # Small adjustment
                adjusted_prob = min(1.0, max(0.0, prob_up + sentiment_adjustment))
                print(f"\n  Combined prediction:")
                print(f"    technical_prob_up: {prob_up:.4f}")
                print(f"    sentiment_adjustment: {sentiment_adjustment:+.4f}")
                print(f"    adjusted_prob_up: {adjusted_prob:.4f}")
                direction = "UP" if adjusted_prob > 0.5 else "DOWN"
                confidence = abs(adjusted_prob - 0.5) * 2
                print(f"    predicted_direction: {direction} (confidence: {confidence:.1%})")

                if aggregate.strongest_positive:
                    print(f"\n  Most positive headline: {aggregate.strongest_positive[:80]}")
                if aggregate.strongest_negative:
                    print(f"  Most negative headline: {aggregate.strongest_negative[:80]}")
            else:
                print("  No articles found for sentiment analysis.")
        except ImportError as exc:
            print(f"  Sentiment modules not available: {exc}")
            print("  Install with: pip install vaderSentiment")
        except Exception as exc:
            print(f"  Sentiment analysis failed: {exc}")

    if args.save_model:
        joblib.dump(model, args.save_model)
        print(f"\nModel saved to: {args.save_model}")


if __name__ == "__main__":
    main()
