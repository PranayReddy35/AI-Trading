"""Meta-labeling: instead of predicting direction, predict whether a triggered
signal will *succeed* — i.e. hit a +1R target before its 1R stop.

This produces cleaner labels than direction prediction. The classifier is
trained on historical bars + the same ensemble/pattern signals the live bot
generates. Output probability is used to filter trades (skip if p < threshold)
or to scale size (size_mult = p).

Free-source friendly: features come from indicators we already compute. The
classifier is sklearn GradientBoosting — no GPU, fits in seconds on a year of
daily bars.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ai_trading.strategy.indicators import atr, rsi


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def _ret(series: pd.Series, n: int) -> pd.Series:
    return series.pct_change(n)


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Compact, leak-free feature set computed from past bars only."""
    close = bars["close"]
    feat = pd.DataFrame(index=bars.index)
    feat["ret_1"] = _ret(close, 1)
    feat["ret_5"] = _ret(close, 5)
    feat["ret_20"] = _ret(close, 20)
    feat["sma5_over_sma20"] = close.rolling(5).mean() / close.rolling(20).mean() - 1
    feat["sma20_over_sma50"] = close.rolling(20).mean() / close.rolling(50).mean() - 1
    feat["rsi14"] = rsi(close, period=14) / 100.0
    a = atr(bars, period=14)
    feat["atr_pct"] = a / close
    if "volume" in bars:
        v = bars["volume"]
        feat["vol_z20"] = (v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-9)
    # 20-day realised vol
    feat["vol20"] = close.pct_change().rolling(20).std()
    # distance from 20-day high/low (a poor-man's regime feature)
    hi20 = bars["high"].rolling(20).max()
    lo20 = bars["low"].rolling(20).min()
    feat["dist_hi20"] = (close - hi20) / hi20
    feat["dist_lo20"] = (close - lo20) / lo20.replace(0, np.nan)
    return feat


# ---------------------------------------------------------------------------
# Triple-barrier labels
# ---------------------------------------------------------------------------


def triple_barrier_labels(
    bars: pd.DataFrame,
    *,
    atr_mult_tp: float = 1.0,
    atr_mult_sl: float = 1.0,
    max_bars: int = 10,
    atr_period: int = 14,
    side: str = "long",
) -> pd.Series:
    """Label every bar with 1 if a hypothetical entry at close[t] hits TP
    before SL within `max_bars`, else 0.
    """
    a = atr(bars, period=atr_period)
    close = bars["close"]
    high = bars["high"]
    low = bars["low"]
    n = len(bars)
    labels = np.zeros(n, dtype=int)
    for i in range(n - 1):
        entry = float(close.iloc[i])
        a_val = float(a.iloc[i]) if not np.isnan(a.iloc[i]) else 0.0
        if a_val <= 0:
            continue
        if side == "long":
            tp = entry + atr_mult_tp * a_val
            sl = entry - atr_mult_sl * a_val
        else:
            tp = entry - atr_mult_tp * a_val
            sl = entry + atr_mult_sl * a_val
        end = min(n, i + 1 + max_bars)
        win = False
        for j in range(i + 1, end):
            hi = float(high.iloc[j])
            lo = float(low.iloc[j])
            if side == "long":
                if lo <= sl:
                    break
                if hi >= tp:
                    win = True
                    break
            else:
                if hi >= sl:
                    break
                if lo <= tp:
                    win = True
                    break
        labels[i] = 1 if win else 0
    return pd.Series(labels, index=bars.index, name="label")


# ---------------------------------------------------------------------------
# Train / predict
# ---------------------------------------------------------------------------


@dataclass
class MetaModel:
    """Wraps a fitted sklearn classifier + the feature column order."""
    model: object
    feature_cols: list[str]

    def predict_proba_win(self, bars: pd.DataFrame) -> float:
        """Probability the current setup (bars[-1]) will hit TP before SL."""
        X = build_features(bars).iloc[[-1]][self.feature_cols]
        if X.isna().any(axis=None):
            return 0.5
        return float(self.model.predict_proba(X.values)[0, 1])

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({"model": self.model, "feature_cols": self.feature_cols}, path)

    @classmethod
    def load(cls, path: str) -> "MetaModel":
        import joblib
        obj = joblib.load(path)
        return cls(model=obj["model"], feature_cols=list(obj["feature_cols"]))


def train_meta_model(
    bars: pd.DataFrame,
    *,
    atr_mult_tp: float = 1.0,
    atr_mult_sl: float = 1.0,
    max_bars: int = 10,
    n_estimators: int = 200,
) -> MetaModel:
    """Fit a GradientBoostingClassifier on the full bar history."""
    from sklearn.ensemble import GradientBoostingClassifier

    X = build_features(bars)
    y = triple_barrier_labels(
        bars, atr_mult_tp=atr_mult_tp, atr_mult_sl=atr_mult_sl, max_bars=max_bars,
    )
    # Drop rows with NaNs and the last `max_bars` rows whose labels are unknown
    df = X.copy()
    df["y"] = y
    df = df.dropna()
    df = df.iloc[:-max_bars] if len(df) > max_bars else df
    if len(df) < 50:
        raise ValueError(f"Not enough training rows ({len(df)})")
    feat_cols = list(X.columns)
    model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=3, random_state=42)
    model.fit(df[feat_cols].values, df["y"].values)
    return MetaModel(model=model, feature_cols=feat_cols)


def train_and_save(symbol: str, *, out_path: str, period: str = "3y") -> MetaModel:
    """Convenience: pull yfinance bars, train, save."""
    import yfinance as yf
    df = yf.download(symbol, period=period, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data for {symbol}")
    df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
    mm = train_meta_model(df)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    mm.save(out_path)
    return mm
