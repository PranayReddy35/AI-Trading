"""Tests for RiskManager."""

from __future__ import annotations

from datetime import date

from ai_trading.risk.manager import RiskManager


def _mk(**overrides) -> RiskManager:
    defaults = dict(
        paper_only=True,
        min_cash_threshold=100,
        max_shares=10,
        max_daily_trades=5,
        max_consecutive_errors=3,
        daily_loss_limit_pct=0.0,
        max_portfolio_exposure_pct=95.0,
        min_equity=0.0,
        trade_cooldown_sec=0,
    )
    defaults.update(overrides)
    return RiskManager(**defaults)


def test_paper_only_guard_allows_paper():
    r = _mk()
    d = r.evaluate(
        today=date.today(), paper_mode=True, market_open=True,
        cash=10_000, has_open_order=False, side="BUY",
        requested_qty=1, current_position_qty=0,
        equity=10_000, last_equity=10_000, portfolio_value=10_000,
    )
    assert d.allowed, d.reason


def test_paper_only_guard_blocks_live():
    r = _mk(paper_only=True)
    d = r.evaluate(
        today=date.today(), paper_mode=False, market_open=True,
        cash=10_000, has_open_order=False, side="BUY",
        requested_qty=1, current_position_qty=0,
        equity=10_000, last_equity=10_000, portfolio_value=10_000,
    )
    assert not d.allowed
    assert "paper" in d.reason.lower() or "live" in d.reason.lower()


def test_market_closed_block():
    r = _mk()
    d = r.evaluate(
        today=date.today(), paper_mode=True, market_open=False,
        cash=10_000, has_open_order=False, side="BUY",
        requested_qty=1, current_position_qty=0,
        equity=10_000, last_equity=10_000, portfolio_value=10_000,
    )
    assert not d.allowed


def test_low_cash_block():
    r = _mk(min_cash_threshold=1000)
    d = r.evaluate(
        today=date.today(), paper_mode=True, market_open=True,
        cash=50, has_open_order=False, side="BUY",
        requested_qty=1, current_position_qty=0,
        equity=10_000, last_equity=10_000, portfolio_value=10_000,
    )
    assert not d.allowed


def test_duplicate_order_block():
    r = _mk()
    d = r.evaluate(
        today=date.today(), paper_mode=True, market_open=True,
        cash=10_000, has_open_order=True, side="BUY",
        requested_qty=1, current_position_qty=0,
        equity=10_000, last_equity=10_000, portfolio_value=10_000,
    )
    assert not d.allowed


def test_max_daily_trades():
    r = _mk(max_daily_trades=1)
    today = date.today()
    # First trade allowed
    d1 = r.evaluate(today=today, paper_mode=True, market_open=True,
                    cash=10_000, has_open_order=False, side="BUY",
                    requested_qty=1, current_position_qty=0,
                    equity=10_000, last_equity=10_000, portfolio_value=10_000)
    assert d1.allowed
    r.register_trade(today)
    # Second blocked
    d2 = r.evaluate(today=today, paper_mode=True, market_open=True,
                    cash=10_000, has_open_order=False, side="BUY",
                    requested_qty=1, current_position_qty=0,
                    equity=10_000, last_equity=10_000, portfolio_value=10_000)
    assert not d2.allowed
