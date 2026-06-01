"""Advanced ensemble ML model for next-day direction prediction.

Improvements over basic logistic regression:
- Ensemble of GradientBoosting + RandomForest (voting classifier)
- Expanded feature set: RSI, MACD, Bollinger Bands, ATR, OBV, momentum indicators
- Walk-forward validation (rolling train/test windows)
- Probability calibration for better confidence estimates
- Feature importance analysis

This is designed to provide a genuine statistical edge through:
1. Non-linear pattern detection (tree-based models)
2. Multiple uncorrelated feature families (trend, momentum, volatility, volume)
3. Out-of-sample validation across multiple market regimes
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature engineering — expanded technical indicators
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    # Returns
    "ret_1d",
    "ret_2d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    # Volatility
    "vol_5d",
    "vol_10d",
    "vol_20d",
    "vol_ratio",  # short-term vol / long-term vol
    # Trend
    "dist_ma_10",
    "dist_ma_20",
    "dist_ma_50",
    "ma_slope_20",  # slope of 20-day MA
    "macd",
    "macd_signal",
    "macd_hist",
    # Momentum
    "rsi_14",
    "rsi_7",
    "stoch_k",
    "stoch_d",
    "momentum_10",
    "roc_10",  # Rate of change
    # Volume
    "vol_ratio_20",
    "obv_slope",  # On-balance volume slope
    "vwap_dist",  # Distance from volume-weighted price proxy
    # Bollinger Bands
    "bb_position",  # Where price is within bands (0-1)
    "bb_width",  # Band width (volatility indicator)
    # ATR
    "atr_14",
    "atr_ratio",  # ATR relative to price
    # Pattern features
    "higher_high",
    "lower_low",
    "inside_bar",
    "gap_up",
    "gap_down",
]


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD, signal line, and histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3
) -> tuple[pd.Series, pd.Series]:
    """Compute Stochastic Oscillator %K and %D."""
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    denom = highest_high - lowest_low
    k = 100 * (close - lowest_low) / denom.replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def build_advanced_features(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build comprehensive feature set from OHLCV data.

    Args:
        bars: DataFrame with columns: open, high, low, close, volume

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    df = bars.copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    open_price = df["open"].astype(float)

    # --- Returns ---
    df["ret_1d"] = close.pct_change(1)
    df["ret_2d"] = close.pct_change(2)
    df["ret_5d"] = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)
    df["ret_20d"] = close.pct_change(20)

    # --- Volatility ---
    df["vol_5d"] = df["ret_1d"].rolling(5).std()
    df["vol_10d"] = df["ret_1d"].rolling(10).std()
    df["vol_20d"] = df["ret_1d"].rolling(20).std()
    df["vol_ratio"] = df["vol_5d"] / df["vol_20d"].replace(0, np.nan)

    # --- Trend (Moving Averages) ---
    ma_10 = close.rolling(10).mean()
    ma_20 = close.rolling(20).mean()
    ma_50 = close.rolling(50).mean()
    df["dist_ma_10"] = (close / ma_10) - 1.0
    df["dist_ma_20"] = (close / ma_20) - 1.0
    df["dist_ma_50"] = (close / ma_50) - 1.0
    df["ma_slope_20"] = ma_20.pct_change(5)

    # --- MACD ---
    macd_line, signal_line, histogram = compute_macd(close)
    df["macd"] = macd_line / close  # Normalize by price
    df["macd_signal"] = signal_line / close
    df["macd_hist"] = histogram / close

    # --- Momentum (RSI, Stochastic) ---
    df["rsi_14"] = compute_rsi(close, 14) / 100.0  # Normalize 0-1
    df["rsi_7"] = compute_rsi(close, 7) / 100.0
    stoch_k, stoch_d = compute_stochastic(high, low, close)
    df["stoch_k"] = stoch_k / 100.0
    df["stoch_d"] = stoch_d / 100.0
    df["momentum_10"] = close / close.shift(10) - 1.0
    df["roc_10"] = close.pct_change(10)

    # --- Volume ---
    vol_avg_20 = volume.rolling(20).mean()
    df["vol_ratio_20"] = (volume / vol_avg_20.replace(0, np.nan)) - 1.0
    # On-balance volume slope
    obv = (np.sign(close.diff()) * volume).cumsum()
    df["obv_slope"] = obv.rolling(10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0, raw=False
    ) / volume.rolling(10).mean().replace(0, np.nan)
    # VWAP distance proxy (using rolling VWAP-like)
    typical_price = (high + low + close) / 3
    vwap_proxy = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)
    df["vwap_dist"] = (close / vwap_proxy) - 1.0

    # --- Bollinger Bands ---
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    df["bb_position"] = (close - bb_lower) / bb_range
    df["bb_width"] = bb_range / bb_mid

    # --- ATR ---
    atr_14 = compute_atr(high, low, close, 14)
    df["atr_14"] = atr_14
    df["atr_ratio"] = atr_14 / close

    # --- Pattern features ---
    df["higher_high"] = (high > high.shift(1)).astype(float)
    df["lower_low"] = (low < low.shift(1)).astype(float)
    df["inside_bar"] = ((high < high.shift(1)) & (low > low.shift(1))).astype(float)
    df["gap_up"] = (open_price > high.shift(1)).astype(float)
    df["gap_down"] = (open_price < low.shift(1)).astype(float)

    # --- Target: next-day direction ---
    df["target"] = (close.shift(-1) > close).astype(int)

    # Drop rows with NaN
    df = df.dropna().copy()

    features = df[FEATURE_COLUMNS]
    target = df["target"]
    return features, target


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class WalkForwardResult:
    """Results from one walk-forward fold."""

    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    accuracy: float
    precision_up: float
    recall_up: float
    n_test: int
    predictions: np.ndarray
    actuals: np.ndarray
    probabilities: np.ndarray


def walk_forward_validate(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
    train_pct: float = 0.7,
    min_train_size: int = 252,  # At least 1 year of trading days
) -> list[WalkForwardResult]:
    """Perform walk-forward (anchored expanding window) validation.

    This simulates real trading conditions where the model is always
    trained only on past data and tested on unseen future data.

    Args:
        X: Feature DataFrame (time-ordered).
        y: Target Series.
        n_folds: Number of test folds.
        train_pct: Minimum training data fraction.
        min_train_size: Minimum training samples.

    Returns:
        List of WalkForwardResult for each fold.
    """
    n = len(X)
    fold_size = max(1, (n - min_train_size) // n_folds)
    results: list[WalkForwardResult] = []

    for fold in range(n_folds):
        test_end = n - (n_folds - fold - 1) * fold_size
        test_start = test_end - fold_size
        train_end = test_start

        if train_end < min_train_size:
            continue
        if test_start >= test_end or test_end > n:
            continue

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue
        if len(X_test) < 10:
            continue

        # Train ensemble model
        model = _build_ensemble()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        accuracy = float(accuracy_score(y_test, preds))
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)

        results.append(
            WalkForwardResult(
                fold=fold,
                train_start=str(X_train.index[0].date()) if hasattr(X_train.index[0], "date") else str(X_train.index[0]),
                train_end=str(X_train.index[-1].date()) if hasattr(X_train.index[-1], "date") else str(X_train.index[-1]),
                test_start=str(X_test.index[0].date()) if hasattr(X_test.index[0], "date") else str(X_test.index[0]),
                test_end=str(X_test.index[-1].date()) if hasattr(X_test.index[-1], "date") else str(X_test.index[-1]),
                accuracy=round(accuracy, 4),
                precision_up=round(float(report.get("1", {}).get("precision", 0.0)), 4),
                recall_up=round(float(report.get("1", {}).get("recall", 0.0)), 4),
                n_test=len(X_test),
                predictions=preds,
                actuals=y_test.values,
                probabilities=probs,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Ensemble model construction
# ---------------------------------------------------------------------------


def _build_ensemble() -> Pipeline:
    """Build the ensemble classifier pipeline.

    Uses a VotingClassifier combining:
    - GradientBoosting: Good at capturing sequential patterns
    - RandomForest: Reduces overfitting, captures interactions

    Both use probability-based soft voting for better calibration.
    """
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=20,
        max_features="sqrt",
        random_state=42,
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=20,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    ensemble = VotingClassifier(
        estimators=[("gb", gb), ("rf", rf)],
        voting="soft",
        weights=[0.6, 0.4],  # Slightly favor GradientBoosting
    )

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ensemble", ensemble),
        ]
    )


def train_ensemble(
    X: pd.DataFrame, y: pd.Series, calibrate: bool = True
) -> Pipeline:
    """Train the full ensemble model with optional probability calibration.

    Args:
        X: Training features.
        y: Training target.
        calibrate: Whether to apply isotonic calibration.

    Returns:
        Trained Pipeline.
    """
    model = _build_ensemble()

    if calibrate and len(X) > 500:
        # Use calibrated classifier for better probability estimates
        base_ensemble = model.named_steps["ensemble"]
        calibrated = CalibratedClassifierCV(base_ensemble, cv=3, method="isotonic")
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ensemble", calibrated),
            ]
        )

    model.fit(X, y)
    return model


def get_feature_importance(model: Pipeline, feature_names: list[str]) -> dict[str, float]:
    """Extract feature importances from the ensemble model.

    Returns:
        Dict of feature name → importance score (higher = more important).
    """
    importances: dict[str, float] = {}

    try:
        ensemble = model.named_steps.get("ensemble")
        if ensemble is None:
            return importances

        # Handle CalibratedClassifierCV wrapper
        if hasattr(ensemble, "estimator"):
            ensemble = ensemble.estimator

        if hasattr(ensemble, "estimators_"):
            for name, est in ensemble.estimators_:
                if hasattr(est, "feature_importances_"):
                    for i, imp in enumerate(est.feature_importances_):
                        feat = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                        importances[feat] = importances.get(feat, 0) + float(imp)
        elif hasattr(ensemble, "feature_importances_"):
            for i, imp in enumerate(ensemble.feature_importances_):
                feat = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                importances[feat] = float(imp)
    except (AttributeError, TypeError):
        pass

    # Normalize
    total = sum(importances.values()) or 1.0
    return {k: round(v / total, 4) for k, v in sorted(importances.items(), key=lambda x: -x[1])}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ensemble ML model with walk-forward validation."
    )
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--csv", help="Optional CSV with columns: date,open,high,low,close,volume")
    parser.add_argument("--save-model", help="Path to save trained model")
    parser.add_argument("--folds", type=int, default=5, help="Walk-forward folds")
    args = parser.parse_args()

    print("=" * 60)
    print("  ENSEMBLE MODEL — Walk-Forward Validation")
    print("  NOT financial advice. Use for research only.")
    print("=" * 60)

    # Load data
    from ai_trading.ml.predict_direction import load_bars

    bars = load_bars(args.symbol, args.start, args.end, args.csv)
    print(f"\nData: {args.symbol} from {bars.index[0].date()} to {bars.index[-1].date()}")
    print(f"Total bars: {len(bars)}")

    # Build features
    X, y = build_advanced_features(bars)
    print(f"Features: {len(FEATURE_COLUMNS)} indicators")
    print(f"Samples after feature engineering: {len(X)}")
    print(f"Target distribution: UP={y.sum()}/{len(y)} ({y.mean():.1%})")

    # Walk-forward validation
    print(f"\n{'='*60}")
    print(f"  Walk-Forward Validation ({args.folds} folds)")
    print(f"{'='*60}")

    results = walk_forward_validate(X, y, n_folds=args.folds)

    if not results:
        print("ERROR: Not enough data for walk-forward validation.")
        return

    for r in results:
        print(
            f"  Fold {r.fold}: train [{r.train_start} → {r.train_end}] "
            f"test [{r.test_start} → {r.test_end}] "
            f"acc={r.accuracy:.4f} prec={r.precision_up:.4f} n={r.n_test}"
        )

    # Aggregate walk-forward metrics
    avg_acc = np.mean([r.accuracy for r in results])
    avg_prec = np.mean([r.precision_up for r in results])
    std_acc = np.std([r.accuracy for r in results])

    print(f"\n  Average accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
    print(f"  Average precision (UP): {avg_prec:.4f}")
    print(f"  Consistency (std < 0.03 is good): {'✓' if std_acc < 0.03 else '✗'}")

    edge_estimate = avg_acc - 0.5
    print(f"  Estimated edge over random: {edge_estimate:+.4f} ({edge_estimate*100:+.1f}%)")

    if avg_acc > 0.53:
        print("  ✓ Model shows potential edge (>53% accuracy)")
    elif avg_acc > 0.51:
        print("  ~ Marginal edge — needs more validation")
    else:
        print("  ✗ No reliable edge detected — do NOT use for live trading")

    # Train final model on all data
    print(f"\n{'='*60}")
    print("  Training final model on all available data...")
    final_model = train_ensemble(X, y, calibrate=True)

    # Latest prediction
    prob_up = float(final_model.predict_proba(X.iloc[[-1]])[0][1])
    print(f"  Latest probability next day UP: {prob_up:.4f}")
    print(f"  Confidence: {abs(prob_up - 0.5) * 2:.1%}")
    print(f"  Signal: {'BUY' if prob_up > 0.55 else 'SELL' if prob_up < 0.45 else 'HOLD'}")

    # Feature importance
    importance = get_feature_importance(final_model, list(X.columns))
    if importance:
        print(f"\n  Top 10 features by importance:")
        for i, (feat, imp) in enumerate(list(importance.items())[:10]):
            print(f"    {i+1}. {feat}: {imp:.4f}")

    if args.save_model:
        joblib.dump(final_model, args.save_model)
        print(f"\n  Model saved to: {args.save_model}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# EnsembleModel class — convenient wrapper for save/load/predict
# ---------------------------------------------------------------------------


# Alias for backwards compat
build_features = build_advanced_features


class EnsembleModel:
    """Wrapper around the sklearn ensemble pipeline with save/load support."""

    def __init__(self) -> None:
        self._pipeline = None

    def fit(self, features_and_target) -> None:
        """Fit from a tuple (X, y) or a DataFrame with 'target' column."""
        if isinstance(features_and_target, tuple):
            X, y = features_and_target
        else:
            y = features_and_target["target"]
            X = features_and_target[FEATURE_COLUMNS]
        self._pipeline = train_ensemble(X, y, calibrate=True)

    def predict_proba_up(self, bars: "pd.DataFrame") -> float:
        """Return probability of next-day UP move from OHLCV bars."""
        if self._pipeline is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        X, _ = build_advanced_features(bars)
        if X.empty:
            return 0.5
        return float(self._pipeline.predict_proba(X.iloc[[-1]])[0][1])

    def save(self, path: str) -> None:
        joblib.dump(self._pipeline, path)

    @classmethod
    def load(cls, path: str) -> "EnsembleModel":
        m = cls()
        m._pipeline = joblib.load(path)
        return m
