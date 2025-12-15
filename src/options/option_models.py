"""
Option data models and mapping from underlying double-bottom trades
to simple call option trade specifications.

This is LAB / planning only: it does NOT fetch real option prices.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import pandas as pd


OptionType = Literal["CALL", "PUT"]


@dataclass
class UnderlyingSignal:
    """
    A clean description of the underlying trade we want to express in options.
    """
    rule_id: str          # e.g. "DB_SPY_5M_LAB_V1"
    ticker: str           # "SPY" or "QQQ"
    direction: Literal["LONG", "SHORT"]
    signal_time: pd.Timestamp  # when the entry is triggered
    entry_price: float
    stop_price: float
    target_price: float


@dataclass
class OptionContractSpec:
    """
    Describes which option contract we intend to trade.
    Does NOT assume any particular broker format.
    """
    underlying: str           # "SPY" / "QQQ"
    option_type: OptionType   # "CALL" or "PUT"
    expiry: date
    strike: float
    rule_id: str              # link back to the rule
    notes: str = ""           # free-form (e.g. "weekly call, ~10 DTE")


@dataclass
class OptionTradePlan:
    """
    Full option trade expression derived from an underlying signal.
    Position sizing / risk management can be layered on later.
    """
    signal: UnderlyingSignal
    contract: OptionContractSpec
    # In LAB we can keep this as 1 contract; sizing belongs to portfolio layer.
    contracts: int = 1
    # Optional: this can hold any LAB risk calc (e.g. implied R on the option).
    meta: dict | None = None


# ── Helpers for date handling ──────────────────────────────────

def _next_friday_on_or_after(d: date) -> date:
    """
    Return the next Friday on or after the given date.
    Monday=0, Sunday=6, Friday=4.
    """
    weekday = d.weekday()
    days_ahead = (4 - weekday) % 7
    return d + timedelta(days=days_ahead)


def choose_weekly_expiry(
    signal_time: pd.Timestamp,
    min_days_out: int = 5,
    max_days_out: int = 15,
) -> date:
    """
    Simple, broker-agnostic expiry rule for LAB:

    - Start from the signal date.
    - Pick the first Friday that is at least `min_days_out` days away.
    - If that Friday is more than `max_days_out` days away, still use it
      (we can refine this later).

    This gives you a weekly-ish expiry with a bit of time value.
    """
    d0 = signal_time.date()
    first_friday = _next_friday_on_or_after(d0)

    if (first_friday - d0).days < min_days_out:
        # jump to the next Friday
        candidate = first_friday + timedelta(days=7)
    else:
        candidate = first_friday

    return candidate


def round_strike_spy_qqq(underlying_price: float) -> float:
    """
    For SPY / QQQ, strikes are usually in 1.0 increments (sometimes 0.5).
    LAB: we keep it simple and round to the nearest whole number.
    """
    return float(round(underlying_price))


# ── Core mapping logic ────────────────────────────────────────

def map_underlying_to_call_option(
    signal: UnderlyingSignal,
    *,
    otm_pct: float = 1.0,
    min_dte: int = 5,
    max_dte: int = 15,
) -> OptionTradePlan:
    """
    Map a LONG underlying signal to a simple call option trade plan.

    Rule (LAB version):
      - Option type: CALL (because the double-bottom is bullish).
      - Expiry: next suitable weekly Friday with ~5–15 days to expiry.
      - Strike: slightly OTM by `otm_pct` (%) from the underlying entry price,
                rounded to a valid SPY/QQQ strike increment.

    This is intentionally simple; later we can make otm_pct depend on the rule,
    volatility, or desired gamma/vega exposure.
    """
    if signal.direction != "LONG":
        raise ValueError("map_underlying_to_call_option currently only supports LONG signals.")

    # Choose expiry
    expiry = choose_weekly_expiry(signal.signal_time, min_days_out=min_dte, max_days_out=max_dte)

    # Choose strike: slightly OTM
    otm_price = signal.entry_price * (1.0 + otm_pct / 100.0)
    strike = round_strike_spy_qqq(otm_price)

    contract = OptionContractSpec(
        underlying=signal.ticker,
        option_type="CALL",
        expiry=expiry,
        strike=strike,
        rule_id=signal.rule_id,
        notes=f"{signal.ticker} call, ~{min_dte}-{max_dte} DTE, ~{otm_pct:.1f}% OTM",
    )

    return OptionTradePlan(
        signal=signal,
        contract=contract,
        contracts=1,
        meta=None,
    )
