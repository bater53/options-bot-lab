# src/strategies/double_bottom_spy_calls.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from src.signals.double_bottom import find_double_bottoms, DoubleBottomSignal
from src.utils.time import to_ny, is_regular_session


# --- "Quality mode" config for high-selectivity double bottoms ---

# Lows must be within 0.25% of each other (very similar price level)
QUALITY_MAX_LOW_DIFF_PCT = 0.0025


# Bounce between the two lows must be at least 0.5% (not just noise)
QUALITY_MIN_BOUNCE_PCT = 0.005

# Reward:risk target: 1.5R (helps increase win-rate vs 2R)
QUALITY_RR = 1.5

# Entry must be at or very close to EMA50:
# allow up to 0.1% below EMA50 (still “on trend”)
QUALITY_MAX_EMA_DRIFT = 0.001


@dataclass
class PlannedOptionsTrade:
    """
    A high-level "plan" for a trade, not a broker order yet.
    """
    underlying: str
    option_symbol: str      # placeholder until you wire real chains
    side: str               # "BUY" or "SELL" (we'll use "BUY")
    qty: int

    t_entry: pd.Timestamp
    underlying_entry: float
    underlying_stop: float
    underlying_target: float

    max_risk_dollars: float


def plan_spy_double_bottom_trades(
    candles_5m: pd.DataFrame,
    *,
    max_risk_per_trade: float = 200.0,
) -> List[PlannedOptionsTrade]:
    """
    Given 5m SPY candles, detect double bottoms and convert them into
    *planned* options trades.

    This is a "quality mode" rule:
      - Only trades during 10:00–11:30 and 13:00–15:30 NY
      - Requires a strong uptrend (EMA50 > EMA200)
      - Requires a clean double bottom with a real bounce
      - Uses a 1.5R target vs 1R stop
    """

    if candles_5m.empty:
        return []

    # Normalize timestamps to NY and restrict to RTH window
    candles_5m = candles_5m.copy()
    candles_5m.index = candles_5m.index.map(to_ny)
    
    
    # Keep only RTH "good" windows:
    #   - 10:00–11:30  (after open chaos, before lunch chop)
    #   - 13:00–15:30  (post-lunch, before close games)
    session_a = candles_5m.between_time("10:00", "11:30")
    session_b = candles_5m.between_time("13:00", "15:30")
    candles_5m = pd.concat([session_a, session_b]).sort_index()
    if candles_5m.empty:
        return []
    
    
    

    # Keep only roughly 09:45–15:45 NY (avoid open/close chaos)
    candles_5m = candles_5m.between_time("09:45", "15:45")
    if candles_5m.empty:
        return []

        # Trend filters: EMA50 (medium trend) and EMA200 (longer trend)
    candles_5m["ema50"] = candles_5m["close"].ewm(span=50, adjust=False).mean()
    candles_5m["ema200"] = candles_5m["close"].ewm(span=200, adjust=False).mean()

    # High-quality double bottom detector config
    signals: List[DoubleBottomSignal] = find_double_bottoms(
        candles_5m,
        symbol="SPY",
        max_low_diff_pct=0.004,  # 0.25% band instead of 0.3%
        min_bounce_pct=0.003,     # 0.4% bounce instead of 0.3%
        buffer_stop_pct=0.001,
        rr=QUALITY_RR,
    )

    trades: List[PlannedOptionsTrade] = []

    for sig in signals:
        # Only trade during regular session
        if not is_regular_session(sig.t_entry.to_pydatetime()):
            continue 
        
        # Pull trend / EMA info at the entry bar
        if sig.t_entry not in candles_5m.index:
            # If something is slightly off with indices, skip to be safe
            continue
        
        row = candles_5m.loc[sig.t_entry]
        ema50 = row.get("ema50")
        ema200 = row.get("ema200")
        
        if pd.isna(ema50) or pd.isna(ema200):
            continue

  # Require medium-term uptrend: EMA50 above EMA200
        if ema50 <= ema200:
            continue

        # Entry must be near or slightly above EMA50 (not deep under it)
        if sig.entry_price < ema50 * (1.0 - QUALITY_MAX_EMA_DRIFT):
            continue

        # Placeholder option selection logic
        option_symbol = f"SPY_CALL_{sig.t_entry.strftime('%Y%m%d_%H%M')}"

        # Placeholder quantity (later: tie to option premium and max_risk_dollars)
        qty = 1

        trades.append(
            PlannedOptionsTrade(
                underlying="SPY",
                option_symbol=option_symbol,
                side="BUY",
                qty=qty,
                t_entry=sig.t_entry,
                underlying_entry=sig.entry_price,
                underlying_stop=sig.stop_price,
                underlying_target=sig.target_price,
                max_risk_dollars=max_risk_per_trade,
            )
        )

    return trades
