"""
Shared double-bottom strategy core for LAB runners.

Used by:
  - src.backtest.run_double_bottom_qqq_lab
  - (optionally) other underlying LAB runners, e.g. SPY
"""


from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf


# ── Data structures ────────────────────────────────────────────

@dataclass
class DoubleBottomPattern:
    """Represents a double-bottom pattern on the underlying."""
    idx1: int        # index of first low
    idx2: int        # index of second low
    mid_idx: int     # index of the highest high between the two lows
    low1: float
    low2: float
    mid_high: float


@dataclass
class PlannedTrade:
    """Represents a simple R-multiple trade based on a pattern."""
    entry_idx: int
    stop_idx: int
    target_idx: int
    entry: float
    stop: float
    target: float
    R: Optional[float] = None
    hit_target: bool = False
    hit_stop: bool = False


# ── Generic data loader ────────────────────────────────────────

def load_underlying_5m(
    ticker: str,
    period: str = "60d",
    interval: str = "5m",
) -> pd.DataFrame:
    """
    Load intraday candles for an underlying from yfinance.

    Normalizes columns to: Open, High, Low, Close, Volume, AdjClose.
    """
    df = yf.download(
        ticker,
        interval=interval,
        period=period,
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        raise RuntimeError(
            f"Failed to download data for {ticker}"
            f"(empty dataframe: period={period}, interval={interval})."
        )

    # If columns are MultiIndex, slice down to the underlying symbol
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = set(df.columns.get_level_values(0))
        lvl1 = set(df.columns.get_level_values(1))

        if ticker in lvl0:
            df = df.xs(ticker, axis=1, level=0)
        elif ticker in lvl1:
            df = df.xs(ticker, axis=1, level=1)
        else:
            # Fallback: first top-level symbol
            first = list(df.columns.levels[0])[0]
            df = df.xs(first, axis=1, level=0)

    # Normalize column names (case-insensitive)
    cols_lower = {str(c).lower(): c for c in df.columns}

    needed = ["open", "high", "low", "close"]
    missing = [name for name in needed if name not in cols_lower]
    if missing:
        raise RuntimeError(
            f"Unexpected columns from yfinance for {ticker}. "
            f"Needed {needed}, got {list(df.columns)}"
        )

    rename_map = {
        cols_lower["open"]: "Open",
        cols_lower["high"]: "High",
        cols_lower["low"]: "Low",
        cols_lower["close"]: "Close",
    }

    if "adj close" in cols_lower:
        rename_map[cols_lower["adj close"]] = "AdjClose"
    if "volume" in cols_lower:
        rename_map[cols_lower["volume"]] = "Volume"

    df = df.rename(columns=rename_map)
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    return df


# ── Pattern detection ──────────────────────────────────────────

def find_local_minima(lows: pd.Series, left: int = 3, right: int = 3) -> np.ndarray:
    """
    Very simple local-minimum detector:
    A bar is a local min if it is the lowest in a (left+right+1)-bar window
    and strictly below the immediate neighbors.
    """
    arr = lows.to_numpy()
    n = len(arr)
    idxs: list[int] = []

    for i in range(left, n - right):
        window = arr[i - left : i + right + 1]
        center = arr[i]
        if center == window.min() and center < arr[i - 1] and center < arr[i + 1]:
            idxs.append(i)

    return np.array(idxs, dtype=int)


def detect_double_bottoms(
    df: pd.DataFrame,
    *,
    left: int = 3,
    right: int = 3,
    max_bars_between: int = 60,
    low_tolerance_pct: float = 0.3,
    min_mid_bounce_pct: float = 0.5,
) -> List[DoubleBottomPattern]:
    """
    Detects double-bottoms on the underlying price.

    Rules:
    - both lows are local minima
    - the two lows are within `low_tolerance_pct` % of each other
    - distance between lows ≤ `max_bars_between` bars
    - the high *between* the lows is at least `min_mid_bounce_pct` % above the lows
    """
    lows = df["Low"].to_numpy()
    highs = df["High"].to_numpy()
    local_min_idx = find_local_minima(df["Low"], left=left, right=right)

    patterns: List[DoubleBottomPattern] = []

    if len(local_min_idx) < 2:
        return patterns

    for i_pos in range(len(local_min_idx)):
        i = local_min_idx[i_pos]
        for j_pos in range(i_pos + 1, len(local_min_idx)):
            j = local_min_idx[j_pos]

            if j - i > max_bars_between:
                break

            low1 = lows[i]
            low2 = lows[j]

            if abs(low2 - low1) / low1 * 100.0 > low_tolerance_pct:
                continue

            mid_slice = slice(i, j + 1)
            mid_high = highs[mid_slice].max()
            min_low = min(low1, low2)
            bounce_pct = (mid_high - min_low) / min_low * 100.0

            if bounce_pct < min_mid_bounce_pct:
                continue

            mid_candidates = np.arange(len(df))[mid_slice]
            local_highs = highs[mid_slice]
            mid_idx = int(mid_candidates[local_highs.argmax()])

            patterns.append(
                DoubleBottomPattern(
                    idx1=int(i),
                    idx2=int(j),
                    mid_idx=mid_idx,
                    low1=float(low1),
                    low2=float(low2),
                    mid_high=float(mid_high),
                )
            )
            break

    return patterns


# ── Trade planning & backtest ─────────────────────────────────

def plan_trades_from_double_bottoms(
    df: pd.DataFrame,
    patterns: List[DoubleBottomPattern],
    *,
    rr_target: float = 1.5,
    entry_buffer_pct: float = 0.05,
    stop_buffer_pct: float = 0.05,
) -> List[PlannedTrade]:
    """
    Simple underlying-only R-multiple model:

    - Entry: breakout above the mid-high plus `entry_buffer_pct`%
    - Stop: below the lower of the two lows by `stop_buffer_pct`%
    - Target: entry + rr_target * (entry - stop)
    - We step bar-by-bar after the second low and see which is hit first
      (target or stop) *after* the entry is triggered.
    """
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()

    trades: List[PlannedTrade] = []

    for p in patterns:
        entry_level = p.mid_high * (1.0 + entry_buffer_pct / 100.0)
        stop_level = min(p.low1, p.low2) * (1.0 - stop_buffer_pct / 100.0)

        risk_per_share = entry_level - stop_level
        if risk_per_share <= 0:
            continue

        target_level = entry_level + rr_target * risk_per_share

        entry_idx: Optional[int] = None
        hit_stop = False
        hit_target = False
        stop_idx: Optional[int] = None
        target_idx: Optional[int] = None

        for k in range(p.idx2 + 1, len(df)):
            bar_low = lows[k]
            bar_high = highs[k]

            if entry_idx is None:
                if bar_low <= entry_level <= bar_high:
                    entry_idx = k
                else:
                    continue

            if bar_low <= stop_level:
                hit_stop = True
                stop_idx = k
                break
            if bar_high >= target_level:
                hit_target = True
                target_idx = k
                break

        if entry_idx is None:
            continue

        R: Optional[float] = None
        if hit_target:
            R = rr_target
        elif hit_stop:
            R = -1.0

        trades.append(
            PlannedTrade(
                entry_idx=entry_idx,
                stop_idx=stop_idx or entry_idx,
                target_idx=target_idx or entry_idx,
                entry=float(entry_level),
                stop=float(stop_level),
                target=float(target_level),
                R=R,
                hit_target=hit_target,
                hit_stop=hit_stop,
            )
        )

    return trades


def summarize_trades(trades: List[PlannedTrade]) -> None:
    """
    Print LAB-style backtest summary for the trades.
    """
    n_trades = len(trades)
    wins = [t for t in trades if t.hit_target]
    losses = [t for t in trades if t.hit_stop]
    resolved = [t for t in trades if t.R is not None]
    unresolved = [t for t in trades if t.R is None]

    n_wins = len(wins)
    n_losses = len(losses)
    n_resolved = len(resolved)
    n_unresolved = len(unresolved)

    win_rate = 100.0 * n_wins / n_resolved if n_resolved > 0 else 0.0
    Rs = [t.R for t in resolved]
    avg_R = float(np.mean(Rs)) if Rs else 0.0

    print()
    print("=== Double Bottom Backtest (underlying 5m) ===")
    print(f"n_trades (total patterns turned into trades): {n_trades}")
    print(f"n_resolved (hit stop or target):            {n_resolved}")
    print(f"  n_wins:   {n_wins}")
    print(f"  n_losses: {n_losses}")
    print(f"n_unresolved (still in play at end):        {n_unresolved}")
    print(f"win_rate (resolved only):                   {win_rate:.1f}%")
    print(f"avg_R (resolved only):                      {avg_R:.2f}")
    print()
