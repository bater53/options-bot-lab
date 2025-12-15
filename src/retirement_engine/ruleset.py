# src/retirement_engine/ruleset.py

"""
Declare which LAB rules are eligible for the "retirement engine".

Also provides a simple evaluate_ruleset() used by run_retirement_sim.py to
decide whether to reduce/increase risk. For now it's a portfolio-level dial.
Later we can wire this to rule_league metrics.
"""

from __future__ import annotations
from typing import Optional, Dict, Any

RETIREMENT_RULES = [
    {
        "rule_id": "LAB_SPY_TREND_PULLBACK_5M_V2",
        "description": "SPY 5m intraday trend-pullback using 15m trend + VWAP filters",
        "symbol": "SPY",
        "enabled": True,
        "max_fraction_of_daily_risk": 1.0,
        "direction": "both",  # "long", "short", "both"
    },
]

# --- Internal state for simple drawdown tracking ---
_peak_balance: Optional[float] = None


def evaluate_ruleset(
    balance: float,
    month_index: int,
    *,
    cfg: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Returns one of: "hold", "reduce_risk", "increase_risk"

    MVP logic:
      - Track peak balance.
      - If drawdown from peak exceeds threshold -> reduce_risk.
      - If we're at/near peak (within small band) -> hold.
      - (Optional) modest increase_risk when making new highs.

    This is intentionally conservative; we can evolve it later.
    """
    global _peak_balance

    # Reset state at the start of a simulation
    if month_index == 0 or _peak_balance is None:
        _peak_balance = float(balance)

    bal = float(balance)
    if bal > _peak_balance:
        _peak_balance = bal

    # Defaults (can be overridden by cfg later)
    max_drawdown_pct = 0.15   # 15% drawdown => reduce risk
    near_peak_band = 0.02     # within 2% of peak => consider "safe/normal"

    dd = 0.0 if _peak_balance <= 0 else (1.0 - (bal / _peak_balance))

    if dd >= max_drawdown_pct:
        return "reduce_risk"

    # If you're basically at highs, keep normal risk (or slightly increase)
    if dd <= near_peak_band:
        # Conservative choice: hold. If you want more aggressive, return "increase_risk".
        return "hold"

    return "hold"
