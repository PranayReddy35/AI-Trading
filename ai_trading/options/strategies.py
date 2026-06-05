"""Option strategy builders.

Each builder takes a normalized chain (list of OptionContract) plus parameters,
and returns a list of `StrategyCandidate` ranked by score.

Supported:
- Long Call (LC) — bullish directional
- Long Put (LP) — bearish directional
- Cash-Secured Put (CSP) — neutral/bullish income, willing to own stock
- Covered Call (CC) — neutral/slightly bullish income on existing equity
- Vertical Spread — bull call, bear put, bull put credit, bear call credit
- Iron Condor — neutral, range-bound
- Short Strangle — neutral, high IV
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Iterable, Literal

from ai_trading.options.chains import OptionContract
from ai_trading.options.greeks import pop_short_option

# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

Side = Literal["buy", "sell"]


@dataclass(slots=True)
class StrategyLeg:
    occ_symbol: str
    side: Side
    ratio: int = 1               # contract multiple
    type: str = "call"           # 'call' | 'put'
    strike: float = 0.0
    expiry: str = ""
    mid: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    source: str = ""
    quote_timestamp: str = ""
    trade_timestamp: str = ""
    quote_age_seconds: float | None = None
    quote_stale: bool = False


@dataclass(slots=True)
class StrategyCandidate:
    strategy: str               # 'long_call', 'csp', 'iron_condor', ...
    underlying: str
    underlying_price: float
    legs: list[StrategyLeg]
    debit_credit: float         # >0 = debit (you pay), <0 = credit (you receive). Per spread.
    max_profit: float           # Per spread (1 contract). May be float('inf') for naked long.
    max_loss: float             # Per spread, positive number. inf for naked short call.
    breakevens: list[float]
    pop: float                  # Probability of profit (0..1) estimated under BS
    risk_reward: float          # max_profit / max_loss (capped at 99)
    bp_requirement: float       # Buying-power required per spread (approx)
    dte: int
    iv_avg: float
    delta_total: float          # Net delta exposure per spread
    notes: str = ""
    score: float = 0.0          # Composite ranking score

    def to_dict(self) -> dict:
        d = asdict(self)
        d["legs"] = [asdict(leg) for leg in self.legs]
        quote_ages = [
            float(leg.quote_age_seconds)
            for leg in self.legs
            if leg.quote_age_seconds is not None
        ]
        stale_legs = [leg for leg in self.legs if leg.quote_stale or not leg.quote_timestamp]
        d["quote_age_seconds"] = max(quote_ages) if quote_ages else None
        d["quote_stale"] = bool(stale_legs)
        d["quote_timestamp"] = _oldest_quote_timestamp(self.legs)
        d["quote_source"] = ",".join(sorted({leg.source for leg in self.legs if leg.source})) or ""
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _filter(
    chain: Iterable[OptionContract],
    contract_type: str | None = None,
    min_dte: int = 0,
    max_dte: int = 365,
    min_oi: int = 0,
    min_volume: int = 0,
    max_spread_pct: float = 100.0,
) -> list[OptionContract]:
    out: list[OptionContract] = []
    for c in chain:
        if contract_type and c.type != contract_type:
            continue
        if c.dte < min_dte or c.dte > max_dte:
            continue
        if c.open_interest < min_oi:
            continue
        if c.volume < min_volume:
            continue
        if c.mid > 0 and c.ask > 0:
            spread_pct = (c.ask - c.bid) / c.mid * 100.0 if c.mid > 0 else 100.0
            if spread_pct > max_spread_pct:
                continue
        out.append(c)
    return out


def _nearest_by_delta(contracts: list[OptionContract], target_delta: float) -> OptionContract | None:
    if not contracts:
        return None
    return min(contracts, key=lambda c: abs(abs(c.delta) - abs(target_delta)))


def _nearest_by_strike(contracts: list[OptionContract], target_strike: float) -> OptionContract | None:
    if not contracts:
        return None
    return min(contracts, key=lambda c: abs(c.strike - target_strike))


def _liquidity_ok(c: OptionContract) -> bool:
    return c.bid > 0 and c.ask > 0 and (c.ask - c.bid) / max(c.mid, 0.01) <= 1.0


def _timestamp_epoch(ts: str) -> float | None:
    try:
        if not ts:
            return None
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _oldest_quote_timestamp(legs: list[StrategyLeg]) -> str:
    known = [
        (epoch, leg.quote_timestamp)
        for leg in legs
        if (epoch := _timestamp_epoch(leg.quote_timestamp)) is not None
    ]
    if not known:
        return ""
    return min(known, key=lambda item: item[0])[1]


def _leg(c: OptionContract, side: Side, ratio: int = 1) -> StrategyLeg:
    return StrategyLeg(
        occ_symbol=c.occ_symbol,
        side=side,
        ratio=ratio,
        type=c.type,
        strike=c.strike,
        expiry=c.expiry.isoformat(),
        mid=c.mid,
        bid=c.bid,
        ask=c.ask,
        source=c.source,
        quote_timestamp=c.quote_timestamp,
        trade_timestamp=c.trade_timestamp,
        quote_age_seconds=c.quote_age_seconds,
        quote_stale=c.quote_stale,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Long Call / Long Put (directional debit)
# ─────────────────────────────────────────────────────────────────────────────

def _build_long_directional(
    chain: list[OptionContract],
    underlying: str,
    spot: float,
    contract_type: str,
    target_delta: float,
    min_dte: int,
    max_dte: int,
    top_n: int = 3,
) -> list[StrategyCandidate]:
    cands: list[StrategyCandidate] = []
    candidates = _filter(chain, contract_type=contract_type, min_dte=min_dte, max_dte=max_dte)
    # Group by expiry, pick the contract closest to target delta in each.
    by_exp: dict[str, list[OptionContract]] = {}
    for c in candidates:
        by_exp.setdefault(c.expiry.isoformat(), []).append(c)

    for exp, group in by_exp.items():
        c = _nearest_by_delta(group, target_delta)
        if not c or not _liquidity_ok(c):
            continue
        debit = c.mid * 100.0           # cost per contract
        max_loss = debit
        max_profit = float("inf")        # uncapped for long single option
        # Breakeven
        be = c.strike + c.mid if contract_type == "call" else c.strike - c.mid
        # POP for long option: probability ITM by enough to cover premium
        # Approximate via delta of break-even strike — use |delta|
        pop = abs(c.delta)               # rough: P(finish ITM) ≈ |delta|
        cand = StrategyCandidate(
            strategy=f"long_{contract_type}",
            underlying=underlying,
            underlying_price=spot,
            legs=[_leg(c, "buy")],
            debit_credit=debit,
            max_profit=max_profit,
            max_loss=max_loss,
            breakevens=[be],
            pop=pop,
            risk_reward=99.0,
            bp_requirement=debit,
            dte=c.dte,
            iv_avg=c.iv,
            delta_total=c.delta,
            notes=f"Δ={c.delta:.2f}, mid=${c.mid:.2f}, BE=${be:.2f}",
        )
        cand.score = _score_directional(cand)
        cands.append(cand)
    cands.sort(key=lambda x: x.score, reverse=True)
    return cands[:top_n]


def build_long_call(
    chain: list[OptionContract], underlying: str, spot: float,
    target_delta: float = 0.40, min_dte: int = 21, max_dte: int = 60, top_n: int = 3,
) -> list[StrategyCandidate]:
    return _build_long_directional(chain, underlying, spot, "call", target_delta, min_dte, max_dte, top_n)


def build_long_put(
    chain: list[OptionContract], underlying: str, spot: float,
    target_delta: float = -0.40, min_dte: int = 21, max_dte: int = 60, top_n: int = 3,
) -> list[StrategyCandidate]:
    return _build_long_directional(chain, underlying, spot, "put", target_delta, min_dte, max_dte, top_n)


# ─────────────────────────────────────────────────────────────────────────────
# Cash-Secured Put (sell OTM put, willing to own at strike-credit)
# ─────────────────────────────────────────────────────────────────────────────

def build_cash_secured_put(
    chain: list[OptionContract], underlying: str, spot: float,
    target_delta: float = -0.30,
    min_dte: int = 21, max_dte: int = 45,
    risk_free: float = 0.045,
    top_n: int = 3,
) -> list[StrategyCandidate]:
    cands: list[StrategyCandidate] = []
    puts = _filter(chain, contract_type="put", min_dte=min_dte, max_dte=max_dte)
    by_exp: dict[str, list[OptionContract]] = {}
    for c in puts:
        by_exp.setdefault(c.expiry.isoformat(), []).append(c)
    for exp, group in by_exp.items():
        c = _nearest_by_delta(group, target_delta)
        if not c or not _liquidity_ok(c) or c.mid <= 0:
            continue
        credit = c.mid * 100.0           # received
        max_profit = credit
        max_loss = c.strike * 100.0 - credit  # if assigned and stock → $0
        be = c.strike - c.mid
        T = max(c.dte, 1) / 365.0
        pop = pop_short_option(spot, c.strike, T, risk_free, c.iv or 0.30, "put", credit=c.mid)
        bp = c.strike * 100.0 - credit  # cash secured: full strike minus credit
        cand = StrategyCandidate(
            strategy="cash_secured_put",
            underlying=underlying,
            underlying_price=spot,
            legs=[_leg(c, "sell")],
            debit_credit=-credit,
            max_profit=max_profit,
            max_loss=max_loss,
            breakevens=[be],
            pop=pop,
            risk_reward=max_profit / max(max_loss, 1.0),
            bp_requirement=bp,
            dte=c.dte,
            iv_avg=c.iv,
            delta_total=c.delta,
            notes=f"sell @${c.mid:.2f}, Δ={c.delta:.2f}, assign@${c.strike:.2f}, BE=${be:.2f}",
        )
        cand.score = _score_premium_seller(cand)
        cands.append(cand)
    cands.sort(key=lambda x: x.score, reverse=True)
    return cands[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# Covered Call (sell OTM call against existing 100 shares)
# ─────────────────────────────────────────────────────────────────────────────

def build_covered_call(
    chain: list[OptionContract], underlying: str, spot: float,
    cost_basis: float,
    target_delta: float = 0.25,
    min_dte: int = 21, max_dte: int = 45,
    risk_free: float = 0.045,
    top_n: int = 3,
) -> list[StrategyCandidate]:
    cands: list[StrategyCandidate] = []
    calls = _filter(chain, contract_type="call", min_dte=min_dte, max_dte=max_dte)
    by_exp: dict[str, list[OptionContract]] = {}
    for c in calls:
        by_exp.setdefault(c.expiry.isoformat(), []).append(c)
    for exp, group in by_exp.items():
        c = _nearest_by_delta(group, target_delta)
        if not c or not _liquidity_ok(c) or c.mid <= 0:
            continue
        credit = c.mid * 100.0
        # Max profit: (strike - cost_basis) + credit if called away
        called_pl = (c.strike - cost_basis) * 100.0 + credit
        max_profit = max(called_pl, credit)
        # Max loss: full stock loss minus credit (stock-only risk, defined by stock not by option)
        max_loss = (cost_basis * 100.0) - credit
        be = cost_basis - c.mid   # downside breakeven (covered by stock)
        T = max(c.dte, 1) / 365.0
        pop = pop_short_option(spot, c.strike, T, risk_free, c.iv or 0.30, "call", credit=c.mid)
        bp = 0.0  # using existing shares
        cand = StrategyCandidate(
            strategy="covered_call",
            underlying=underlying,
            underlying_price=spot,
            legs=[_leg(c, "sell")],
            debit_credit=-credit,
            max_profit=max_profit,
            max_loss=max_loss,
            breakevens=[be],
            pop=pop,
            risk_reward=max_profit / max(max_loss, 1.0),
            bp_requirement=bp,
            dte=c.dte,
            iv_avg=c.iv,
            delta_total=c.delta,
            notes=f"sell @${c.mid:.2f}, Δ={c.delta:.2f}, called@${c.strike:.2f} → +${called_pl:.0f}",
        )
        cand.score = _score_premium_seller(cand)
        cands.append(cand)
    cands.sort(key=lambda x: x.score, reverse=True)
    return cands[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# Vertical Spread (debit or credit)
# ─────────────────────────────────────────────────────────────────────────────

def build_vertical_spread(
    chain: list[OptionContract], underlying: str, spot: float,
    direction: Literal["bull_call", "bear_put", "bull_put", "bear_call"] = "bull_call",
    width: float = 5.0,
    long_delta: float = 0.40,
    min_dte: int = 21, max_dte: int = 60,
    risk_free: float = 0.045,
    top_n: int = 3,
) -> list[StrategyCandidate]:
    cands: list[StrategyCandidate] = []
    is_call = direction in ("bull_call", "bear_call")
    is_debit = direction in ("bull_call", "bear_put")
    contract_type = "call" if is_call else "put"
    legs = _filter(chain, contract_type=contract_type, min_dte=min_dte, max_dte=max_dte)
    by_exp: dict[str, list[OptionContract]] = {}
    for c in legs:
        by_exp.setdefault(c.expiry.isoformat(), []).append(c)

    for exp, group in by_exp.items():
        long_leg = _nearest_by_delta(group, long_delta if is_call else -long_delta)
        if not long_leg or not _liquidity_ok(long_leg):
            continue
        # Short leg: width strikes further OTM
        if direction == "bull_call":
            short_strike = long_leg.strike + width
        elif direction == "bear_put":
            short_strike = long_leg.strike - width
        elif direction == "bull_put":
            # sell higher-strike put (long_leg here is the SHORT leg from delta target)
            short_strike = long_leg.strike - width
        else:  # bear_call
            short_strike = long_leg.strike + width
        short_leg = _nearest_by_strike(group, short_strike)
        if not short_leg or short_leg.occ_symbol == long_leg.occ_symbol or not _liquidity_ok(short_leg):
            continue

        if is_debit:
            buy, sell = long_leg, short_leg
        else:
            # credit spread: sell the closer-to-money leg
            buy, sell = short_leg, long_leg
            # swap so 'buy' is the protective wing
            if (direction == "bull_put" and buy.strike > sell.strike) or \
               (direction == "bear_call" and buy.strike < sell.strike):
                buy, sell = sell, buy

        debit = (buy.mid - sell.mid) * 100.0  # positive = net debit, negative = net credit
        width_pts = abs(buy.strike - sell.strike)
        if is_debit:
            max_loss = max(debit, 1.0)
            max_profit = width_pts * 100.0 - debit
            be = (buy.strike + (buy.mid - sell.mid)) if is_call else (buy.strike - (buy.mid - sell.mid))
        else:
            credit = -debit  # positive
            max_profit = credit
            max_loss = max(width_pts * 100.0 - credit, 1.0)
            be = (sell.strike - (sell.mid - buy.mid)) if not is_call else (sell.strike + (sell.mid - buy.mid))
        T = max(buy.dte, 1) / 365.0
        iv_avg = ((buy.iv or 0) + (sell.iv or 0)) / 2.0
        # POP approximation: use short-leg POP
        if is_debit:
            pop = abs(buy.delta)  # P(reach long delta region)
        else:
            pop = pop_short_option(spot, sell.strike, T, risk_free, iv_avg or 0.30,
                                   "put" if not is_call else "call", credit=(sell.mid - buy.mid))

        bp = max_loss   # defined-risk spread
        net_delta = (buy.delta if is_debit else -sell.delta) + (-sell.delta if is_debit else buy.delta)

        cand = StrategyCandidate(
            strategy=direction,
            underlying=underlying,
            underlying_price=spot,
            legs=[_leg(buy, "buy"), _leg(sell, "sell")],
            debit_credit=debit,
            max_profit=max_profit,
            max_loss=max_loss,
            breakevens=[be],
            pop=pop,
            risk_reward=max_profit / max_loss,
            bp_requirement=bp,
            dte=buy.dte,
            iv_avg=iv_avg,
            delta_total=net_delta,
            notes=f"{direction} {buy.strike:.0f}/{sell.strike:.0f} {'debit' if is_debit else 'credit'}=${abs(debit):.0f}",
        )
        cand.score = _score_spread(cand)
        cands.append(cand)
    cands.sort(key=lambda x: x.score, reverse=True)
    return cands[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# Iron Condor (neutral, 4 legs)
# ─────────────────────────────────────────────────────────────────────────────

def build_iron_condor(
    chain: list[OptionContract], underlying: str, spot: float,
    short_delta: float = 0.20,
    wing_width: float = 5.0,
    min_dte: int = 30, max_dte: int = 60,
    risk_free: float = 0.045,
    top_n: int = 3,
) -> list[StrategyCandidate]:
    cands: list[StrategyCandidate] = []
    puts  = _filter(chain, contract_type="put",  min_dte=min_dte, max_dte=max_dte)
    calls = _filter(chain, contract_type="call", min_dte=min_dte, max_dte=max_dte)
    exp_set = sorted({c.expiry.isoformat() for c in puts} & {c.expiry.isoformat() for c in calls})

    for exp in exp_set:
        pg = [p for p in puts  if p.expiry.isoformat() == exp]
        cg = [c for c in calls if c.expiry.isoformat() == exp]
        sp = _nearest_by_delta(pg, -short_delta)
        sc = _nearest_by_delta(cg,  short_delta)
        if not sp or not sc:
            continue
        lp = _nearest_by_strike(pg, sp.strike - wing_width)
        lc = _nearest_by_strike(cg, sc.strike + wing_width)
        if not lp or not lc:
            continue
        legs_ = [sp, sc, lp, lc]
        if any(not _liquidity_ok(x) for x in legs_):
            continue
        credit = (sp.mid + sc.mid - lp.mid - lc.mid) * 100.0
        if credit <= 0:
            continue
        wp = sp.strike - lp.strike
        wc = lc.strike - sc.strike
        max_loss = max(max(wp, wc) * 100.0 - credit, 1.0)
        max_profit = credit
        be_low = sp.strike - (credit / 100.0)
        be_high = sc.strike + (credit / 100.0)
        T = max(sp.dte, 1) / 365.0
        iv_avg = sum(x.iv or 0 for x in legs_) / 4.0
        # POP ≈ P(be_low <= S_T <= be_high)
        pop_low  = pop_short_option(spot, sp.strike, T, risk_free, iv_avg or 0.30, "put",  credit=(sp.mid - lp.mid))
        pop_high = pop_short_option(spot, sc.strike, T, risk_free, iv_avg or 0.30, "call", credit=(sc.mid - lc.mid))
        pop = max(0.0, pop_low + pop_high - 1.0)  # approx joint
        cand = StrategyCandidate(
            strategy="iron_condor",
            underlying=underlying,
            underlying_price=spot,
            legs=[_leg(lp, "buy"), _leg(sp, "sell"), _leg(sc, "sell"), _leg(lc, "buy")],
            debit_credit=-credit,
            max_profit=max_profit,
            max_loss=max_loss,
            breakevens=[be_low, be_high],
            pop=pop,
            risk_reward=max_profit / max_loss,
            bp_requirement=max_loss,
            dte=sp.dte,
            iv_avg=iv_avg,
            delta_total=sp.delta + sc.delta + lp.delta + lc.delta,
            notes=f"IC {lp.strike:.0f}/{sp.strike:.0f} - {sc.strike:.0f}/{lc.strike:.0f} credit=${credit:.0f}",
        )
        cand.score = _score_spread(cand)
        cands.append(cand)
    cands.sort(key=lambda x: x.score, reverse=True)
    return cands[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# Short Strangle (naked — high BP, high IV)
# ─────────────────────────────────────────────────────────────────────────────

def build_strangle(
    chain: list[OptionContract], underlying: str, spot: float,
    short_delta: float = 0.16,
    min_dte: int = 30, max_dte: int = 60,
    risk_free: float = 0.045,
    top_n: int = 3,
) -> list[StrategyCandidate]:
    cands: list[StrategyCandidate] = []
    puts  = _filter(chain, contract_type="put",  min_dte=min_dte, max_dte=max_dte)
    calls = _filter(chain, contract_type="call", min_dte=min_dte, max_dte=max_dte)
    exp_set = sorted({c.expiry.isoformat() for c in puts} & {c.expiry.isoformat() for c in calls})
    for exp in exp_set:
        pg = [p for p in puts  if p.expiry.isoformat() == exp]
        cg = [c for c in calls if c.expiry.isoformat() == exp]
        sp = _nearest_by_delta(pg, -short_delta)
        sc = _nearest_by_delta(cg,  short_delta)
        if not sp or not sc or not _liquidity_ok(sp) or not _liquidity_ok(sc):
            continue
        credit = (sp.mid + sc.mid) * 100.0
        if credit <= 0:
            continue
        max_profit = credit
        max_loss = float("inf")           # undefined (naked)
        be_low = sp.strike - (credit / 100.0)
        be_high = sc.strike + (credit / 100.0)
        T = max(sp.dte, 1) / 365.0
        iv_avg = ((sp.iv or 0) + (sc.iv or 0)) / 2.0
        pop_low  = pop_short_option(spot, sp.strike, T, risk_free, iv_avg or 0.30, "put",  credit=sp.mid)
        pop_high = pop_short_option(spot, sc.strike, T, risk_free, iv_avg or 0.30, "call", credit=sc.mid)
        pop = max(0.0, pop_low + pop_high - 1.0)
        # Approximate Reg-T naked BP: 20% of underlying - OTM amount + premium, larger of put/call
        otm_put  = max(0.0, spot - sp.strike)
        otm_call = max(0.0, sc.strike - spot)
        bp_put  = 0.20 * spot * 100.0 - otm_put * 100.0 + sp.mid * 100.0
        bp_call = 0.20 * spot * 100.0 - otm_call * 100.0 + sc.mid * 100.0
        bp = max(bp_put, bp_call) + min(sp.mid, sc.mid) * 100.0
        cand = StrategyCandidate(
            strategy="short_strangle",
            underlying=underlying,
            underlying_price=spot,
            legs=[_leg(sp, "sell"), _leg(sc, "sell")],
            debit_credit=-credit,
            max_profit=max_profit,
            max_loss=max_loss,
            breakevens=[be_low, be_high],
            pop=pop,
            risk_reward=99.0,
            bp_requirement=bp,
            dte=sp.dte,
            iv_avg=iv_avg,
            delta_total=sp.delta + sc.delta,
            notes=f"naked strangle {sp.strike:.0f}/{sc.strike:.0f} credit=${credit:.0f} (BP~${bp:.0f})",
        )
        cand.score = _score_premium_seller(cand)
        cands.append(cand)
    cands.sort(key=lambda x: x.score, reverse=True)
    return cands[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _score_directional(c: StrategyCandidate) -> float:
    # Favor higher pop, reasonable cost (% of underlying), liquidity-friendly DTE 21-45
    cost_ratio = c.debit_credit / (c.underlying_price * 100.0) if c.underlying_price > 0 else 1.0
    dte_score = max(0.0, 1.0 - abs(c.dte - 35) / 35.0)
    return 100.0 * (0.5 * c.pop + 0.3 * dte_score + 0.2 * max(0.0, 1.0 - cost_ratio * 10.0))


def _score_premium_seller(c: StrategyCandidate) -> float:
    # Favor: high POP, decent return-on-risk (max_profit / bp), DTE 30-45
    ror = c.max_profit / max(c.bp_requirement, 1.0)
    dte_score = max(0.0, 1.0 - abs(c.dte - 35) / 35.0)
    iv_score = min(c.iv_avg / 0.40, 1.0) if c.iv_avg > 0 else 0.3   # like higher IV
    return 100.0 * (0.45 * c.pop + 0.20 * dte_score + 0.20 * min(ror * 5.0, 1.0) + 0.15 * iv_score)


def _score_spread(c: StrategyCandidate) -> float:
    # Favor: high POP * R:R, balanced DTE
    rr_clip = min(c.risk_reward, 3.0) / 3.0
    dte_score = max(0.0, 1.0 - abs(c.dte - 40) / 40.0)
    return 100.0 * (0.5 * c.pop + 0.3 * rr_clip + 0.2 * dte_score)
