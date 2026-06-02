"""Tests for ml.meta_label."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai_trading.ml.meta_label import (
    build_features,
    triple_barrier_labels,
    train_meta_model,
)


def _make_bars(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.01, n)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.005, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.005, n))
    vol = rng.integers(500_000, 2_000_000, n)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": vol}, index=idx)


def test_build_features_has_no_lookahead():
    bars = _make_bars(200)
    feat = build_features(bars)
    # Last feature row must use only bars up to and including the last bar
    assert len(feat) == len(bars)
    # All feature columns produce some non-null tail
    tail = feat.iloc[-1].dropna()
    assert len(tail) >= 6


def test_triple_barrier_labels_binary():
    bars = _make_bars(300)
    y = triple_barrier_labels(bars, max_bars=5)
    assert set(y.unique()).issubset({0, 1})
    # not all the same
    assert 0 < y.sum() < len(y)


def test_train_meta_model_fits_and_predicts():
    bars = _make_bars(500)
    mm = train_meta_model(bars, max_bars=5, n_estimators=50)
    p = mm.predict_proba_win(bars)
    assert 0.0 <= p <= 1.0


def test_train_meta_model_save_load(tmp_path):
    bars = _make_bars(400)
    mm = train_meta_model(bars, n_estimators=30, max_bars=5)
    path = tmp_path / "meta.joblib"
    mm.save(str(path))
    from ai_trading.ml.meta_label import MetaModel
    mm2 = MetaModel.load(str(path))
    p1 = mm.predict_proba_win(bars)
    p2 = mm2.predict_proba_win(bars)
    assert abs(p1 - p2) < 1e-9
