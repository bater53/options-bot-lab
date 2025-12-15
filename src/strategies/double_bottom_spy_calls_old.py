# src/strategies/double_bottom_spy_calls.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import numpy as np


# ---------------------------
# Configs & Data Structures
# ---------------------------

@dataclass
class DoubleBottomConfig:
    max_low_diff_pct: float = 0.01       # 1% allowed difference between lows
    min_bounce_pct: float = 0.015        # neckline at least 1.5% above lows
    breakout_buffer_pct: float = 0.001   # breakout must exceed neckline by 0.1%
    min_bars_between_lows: int = 2
    max_bars_between_lows: int = 10
    max_bars_after_breakout: int = 24    # used by backtest horizon


@dataclass
class DoubleBottomPattern:
    left_idx: int
    right_idx: int
    neckline_idx: int
    breakout_idx: int


@dataclass
class PlannedOptionsTrade:
    """
    Minimal shape needed by the backtester.

    You can extend this later with:
      - option symbol
      - expiry
      - contracts, etc.
    """
    t_entry: pd.Timestamp
    underlying_entry: float
    underlying_stop: float
    underlying_target: float
    # optional metadata
    pattern: Optional[DoubleBottomPattern] = None
    note: str = ""


# ---------------------------
# Pattern Detection
# ---------------------------

def detect_double_bottoms(
    df: pd.DataFrame,
    config: Optional[DoubleBottomConfig] = None,
    debug: bool = False,
) -> List[DoubleBottomPattern]:
    """
    Basic double-bottom detector on OHLC data.

    Uses lows to find local minima, highs to form neckline, closes for breakout.
    """
    if config is None:
        config = DoubleBottomConfig()

    if "close" not in df.columns:
        raise ValueError("DataFrame must have a 'close' column")

    closes = df["close"].to_numpy(dtype=float)
    lows   = df["low"].to_numpy(dtype=float)   if "low"  in df.columns else closes
    highs  = df["high"].to_numpy(dtype=float)  if "high" in df.columns else closes

    n = len(df)
    patterns: List[DoubleBottomPattern] = []

    i = 1
    while i < n - 1:
        # local minimum candidate for first low
        if lows[i] <= lows[i - 1] and lows[i] <= lows[i + 1]:
            left = i

            if debug:
                print(f"[detect] candidate left low at {left}, price={lows[left]:.2f}")

            # search for second low within window
            search_start = left + config.min_bars_between_lows
            search_end   = min(left + config.max_bars_between_lows, n - 2)

            found_for_this_left = False

            for j in range(search_start, search_end + 1):
                # require local min as second low
                if not (lows[j] <= lows[j - 1] and lows[j] <= lows[j + 1]):
                    continue

                low_diff = abs(lows[j] - lows[left]) / lows[left]
                if low_diff > config.max_low_diff_pct:
                    if debug:
                        print(
                            f"  [skip second low @ {j}] low diff {low_diff:.4f} "
                            f"> max {config.max_low_diff_pct:.4f}"
                        )
                    continue

                # neckline = max high between lows
                seg_highs = highs[left : j + 1]
                neckline_val = float(seg_highs.max())
                neckline_local_idx = int(seg_highs.argmax())
                neckline_idx = left + neckline_local_idx

                base_low = min(lows[left], lows[j])
                bounce_pct = neckline_val / base_low - 1.0
                if bounce_pct < config.min_bounce_pct:
                    if debug:
                        print(
                            f"  [skip @ {j}] bounce {bounce_pct:.4f} "
                            f"< min {config.min_bounce_pct:.4f}"
                        )
                    continue

                # breakout: close > neckline * (1 + buffer) AFTER second low
                breakout_threshold = neckline_val * (1.0 + config.breakout_buffer_pct)
                breakout_idx = None
                breakout_search_end = min(j + config.max_bars_between_lows, n - 1)

                for k in range(j + 1, breakout_search_end + 1):
                    if closes[k] > breakout_threshold:
                        breakout_idx = k
                        break

                if breakout_idx is None:
                    if debug:
                        print(f"  [no breakout] second low {j}, neckline {neckline_val:.2f}")
                    continue

                if debug:
                    print(
                        f"[pattern] left={left}({lows[left]:.2f}), "
                        f"right={j}({lows[j]:.2f}), neckline={neckline_idx}({neckline_val:.2f}), "
                        f"breakout={breakout_idx}({closes[breakout_idx]:.2f})"
                    )

                patterns.append(
                    DoubleBottomPattern(
                        left_idx=left,
                        right_idx=j,
                        neckline_idx=neckline_idx,
                        breakout_idx=breakout_idx,
                    )
                )
                found_for_this_left = True
                # move i past the second low to avoid overlapping patterns
                i = j + 1
                break

            if not found_for_this_left:
                i += 1
        else:
            i += 1

    return patterns


# ---------------------------
# Planner: turn patterns into trades
# ---------------------------

def plan_spy_double_bottom_trades(
    candles_5m: pd.DataFrame,
    *,
    max_risk_per_trade: float = 200.0,
    config: Optional[DoubleBottomConfig] = None,
    debug: bool = False,
) -> List[PlannedOptionsTrade]:
    """
    Given 5m SPY candles (already NY tz & sorted), detect double bottoms
    and plan underlying-based trades.

    Entry:  breakout bar close.
    Stop:   0.5% below lower of the two lows.
    Target: 2R (reward:risk = 2:1).
    """
    if config is None:
        config = DoubleBottomConfig()

    if candles_5m.empty:
        return []

    patterns = detect_double_bottoms(candles_5m, config=config, debug=debug)
    if debug:
        print(f"[planner] detected {len(patterns)} patterns")

    closes = candles_5m["close"].to_numpy(dtype=float)
    lows   = candles_5m["low"].to_numpy(dtype=float) if "low" in candles_5m.columns else closes

    trades: List[PlannedOptionsTrade] = []
    index_array = candles_5m.index.to_list()

    for pat in patterns:
        breakout_idx = pat.breakout_idx
        if breakout_idx >= len(index_array):
            continue

        t_entry = index_array[breakout_idx]
        entry_price = float(closes[breakout_idx])
        bottom_low  = float(min(lows[pat.left_idx], lows[pat.right_idx]))
        stop_price  = bottom_low * 0.995  # 0.5% under the lower low
        risk_per_share = entry_price - stop_price
        if risk_per_share <= 0:
            if debug:
                print(f"[planner] skip pattern @ breakout {breakout_idx}: non-positive risk")
            continue

        # 2R target
        target_price = entry_price + 2.0 * risk_per_share

        trade = PlannedOptionsTrade(
            t_entry=t_entry,
            underlying_entry=entry_price,
            underlying_stop=stop_price,
            underlying_target=target_price,
            pattern=pat,
            note="SPY double-bottom call play",
        )
        trades.append(trade)

        if debug:
            print(
                f"[planner] trade @ {t_entry}: entry={entry_price:.2f}, "
                f"stop={stop_price:.2f}, target={target_price:.2f}"
            )

    return trades
