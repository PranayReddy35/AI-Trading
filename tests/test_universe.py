"""Tests for the symbol universe loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_trading.data import universe as u


def test_normalise_symbols_replaces_dot_with_dash():
    out = u._normalise_symbols(["BRK.B", "BF.B", "AAPL"])
    assert out == ["BRK-B", "BF-B", "AAPL"]


def test_normalise_symbols_strips_footnotes_and_dedups():
    out = u._normalise_symbols(["AAPL", "aapl", "MSFT[1]", "  GOOGL  ", None])
    assert out == ["AAPL", "MSFT", "GOOGL"]


def test_load_universe_passes_through_bare_tickers(monkeypatch, tmp_path):
    monkeypatch.setattr(u, "CACHE_DIR", tmp_path)
    out = u.load_universe(["aapl", "NVDA"])
    assert out == ["AAPL", "NVDA"]


def test_load_universe_uses_fallback_when_offline(monkeypatch, tmp_path):
    monkeypatch.setattr(u, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(u, "_fetch_wiki", lambda alias: [])
    out = u.load_universe(["dow30"])
    assert len(out) == 30
    assert "AAPL" in out and "MSFT" in out


def test_load_universe_all_alias_expands(monkeypatch, tmp_path):
    monkeypatch.setattr(u, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(u, "_fetch_wiki", lambda alias: [])
    out = u.load_universe(["all"])
    # Dedup across sp500/ndx/dow30; should be > 30
    assert len(out) > 30
    assert "AAPL" in out


def test_cache_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(u, "CACHE_DIR", tmp_path)
    u._write_cache("dow30", ["AAPL", "MSFT"])
    assert u._read_cache("dow30") == ["AAPL", "MSFT"]


def test_unknown_alias_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(u, "CACHE_DIR", tmp_path)
    with pytest.raises(ValueError):
        u.get_index("ftse100")


def test_fetched_symbols_are_written_to_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(u, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(u, "_fetch_wiki", lambda alias: ["AAA", "BBB"])
    out = u.get_index("dow30", refresh=True)
    assert out == ["AAA", "BBB"]
    cached = json.loads((tmp_path / "dow30.json").read_text())
    assert cached == ["AAA", "BBB"]
