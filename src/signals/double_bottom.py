# src/signals/double_bottom.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class DoubleBottomSignal:
    symbol: str
    t_entry: pd.Timestamp     # time of confirmation bar
    l1: float                 # first low
    l2: float                 # second low
    swing_high: float         # high between the two lows
    entry_price: float        # underlying price on entry
    stop_price: float         # suggested stop on underlying
    target_price: float       # suggested target on underlying


def find_double_bottoms(
    df: pd.DataFrame,
    *,
    symbol: str,
    max_low_diff_pct: float = 0.003,   # 0.3%
    min_bounce_pct: float = 0.003,     # 0.3% bounce from L1
    buffer_stop_pct: float = 0.001,    # 0.1% below lows
    rr: float = 2.0,
) -> List[DoubleBottomSignal]:
    """
    Very simple, swing-based double-bottom detector on OHLCV data.

    df must have columns: ['open','high','low','close'] and a DateTimeIndex.
    """

    df = df.copy()
    signals: List[DoubleBottomSignal] = []

    # --- 1. Find swing lows (you can later upgrade this logic) ---
    # A simple 1-bar left/right swing:
    lows = df["low"]
    swing_low_mask = (lows.shift(1) > lows) & (lows.shift(-1) > lows)
    swing_lows = df[swing_low_mask]

    swing_low_points = list(swing_lows.itertuples())
    n = len(swing_low_points)
    if n < 2:
        return signals

    for i in range(n - 1):
        row1 = swing_low_points[i]
        row2 = swing_low_points[i + 1]

        t1, l1 = row1.Index, row1.low
        t2, l2 = row2.Index, row2.low

        # price similarity of the two lows
        low_diff_pct = abs(l2 - l1) / l1
        if low_diff_pct > max_low_diff_pct:
            continue

        # must have a bounce between t1 and t2
        between = df.loc[t1:t2]
        mid_high = between["high"].max()
        bounce_pct = (mid_high - l1) / l1
        if bounce_pct < min_bounce_pct:
            continue

        # --- confirmation: close above swing high after second low ---
        after_l2 = df.loc[t2:]
        if after_l2.empty:
            continue

        # confirmation bar = first close > mid_high
        confirm = after_l2[after_l2["close"] > mid_high]
        if confirm.empty:
            continue

        confirm_row = confirm.iloc[0]
        t_entry = confirm_row.name
        entry = float(confirm_row.close)

        stop = min(l1, l2) * (1.0 - buffer_stop_pct)
        r = entry - stop
        if r <= 0:
            continue
        target = entry + rr * r

        signals.append(
            DoubleBottomSignal(
                symbol=symbol,
                t_entry=t_entry,
                l1=float(l1),
                l2=float(l2),
                swing_high=float(mid_high),
                entry_price=entry,
                stop_price=float(stop),
                target_price=float(target),
            )
        )

    return signals
