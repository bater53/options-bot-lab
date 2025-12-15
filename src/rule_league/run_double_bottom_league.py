"""
Run a simple Rule League for double-bottom LAB rules.

Usage:
    python -m src.rule_league.run_double_bottom_league
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf

from src.rule_league.double_bottom_rules import (
    ALL_DOUBLE_BOTTOM_RULES,
    DoubleBottomRuleConfig,
)


# ── Shared data structures ─────────────────────────────────────

@dataclass
class DoubleBottomPattern:
    idx1: int
    idx2: int
    mid_idx: int
    low1: float
    low2: float
    mid_high: float


@dataclass
class PlannedTrade:
    entry_idx: int
    stop_idx: int
    target_idx: int
    entry: float
    stop: float
    target: float
    R: float | None
    hit_target: bool
    hit_stop: bool


@dataclass
class RuleResult:
    rule_id: str
    ticker: str
    n_trades: int
    n_resolved: int
    n_wins: int
    n_losses: int
    n_unresolved: int
    win_rate_resolved: float
    avg_R_resolved: float


# ── Core logic (copied from your LAB runners) ──────────────────

def load_underlying_5m(
    ticker: str,
    period: str,
    interval: str = "5m",
) -> pd.DataFrame:
    df = yf.download(
        ticker,
        interval=interval,
        period=period,
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        raise RuntimeError(
            f"Failed to download data for {ticker} (period={period}, interval={interval})"
        )

    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = set(df.columns.get_level_values(0))
        lvl1 = set(df.columns.get_level_values(1))

        if ticker in lvl0:
            df = df.xs(ticker, axis=1, level=0)
        elif ticker in lvl1:
            df = df.xs(ticker, axis=1, level=1)
        else:
            first = list(df.columns.levels[0])[0]
            df = df.xs(first, axis=1, level=0)

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


def find_local_minima(lows: pd.Series, left: int = 3, right: int = 3) -> np.ndarray:
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
    cfg: DoubleBottomRuleConfig,
) -> List[DoubleBottomPattern]:
    lows = df["Low"].to_numpy()
    highs = df["High"].to_numpy()
    local_min_idx = find_local_minima(df["Low"], left=cfg.left, right=cfg.right)

    patterns: List[DoubleBottomPattern] = []

    if len(local_min_idx) < 2:
        return patterns

    for i_pos in range(len(local_min_idx)):
        i = local_min_idx[i_pos]
        for j_pos in range(i_pos + 1, len(local_min_idx)):
            j = local_min_idx[j_pos]

            if j - i > cfg.max_bars_between:
                break

            low1 = lows[i]
            low2 = lows[j]

            if abs(low2 - low1) / low1 * 100.0 > cfg.low_tolerance_pct:
                continue

            mid_slice = slice(i, j + 1)
            mid_high = highs[mid_slice].max()
            min_low = min(low1, low2)
            bounce_pct = (mid_high - min_low) / min_low * 100.0

            if bounce_pct < cfg.min_mid_bounce_pct:
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


def plan_trades_from_double_bottoms(
    df: pd.DataFrame,
    cfg: DoubleBottomRuleConfig,
    patterns: List[DoubleBottomPattern],
) -> List[PlannedTrade]:
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()

    trades: List[PlannedTrade] = []

    for p in patterns:
        entry_level = p.mid_high * (1.0 + cfg.entry_buffer_pct / 100.0)
        stop_level = min(p.low1, p.low2) * (1.0 - cfg.stop_buffer_pct / 100.0)

        risk_per_share = entry_level - stop_level
        if risk_per_share <= 0:
            continue

        target_level = entry_level + cfg.rr_target * risk_per_share

        entry_idx: int | None = None
        hit_stop = False
        hit_target = False
        stop_idx: int | None = None
        target_idx: int | None = None

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

        R: float | None = None
        if hit_target:
            R = cfg.rr_target
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


def summarize_rule(cfg: DoubleBottomRuleConfig, trades: List[PlannedTrade]) -> RuleResult:
    n_trades = len(trades)
    wins = [t for t in trades if t.hit_target]
    losses = [t for t in trades if t.hit_stop]
    resolved = [t for t in trades if t.R is not None]
    unresolved = [t for t in trades if t.R is None]

    n_wins = len(wins)
    n_losses = len(losses)
    n_resolved = len(resolved)
    n_unresolved = len(unresolved)

    win_rate_resolved = 100.0 * n_wins / n_resolved if n_resolved > 0 else 0.0
    Rs = [t.R for t in resolved]
    avg_R_resolved = float(np.mean(Rs)) if Rs else 0.0

    return RuleResult(
        rule_id=cfg.rule_id,
        ticker=cfg.ticker,
        n_trades=n_trades,
        n_resolved=n_resolved,
        n_wins=n_wins,
        n_losses=n_losses,
        n_unresolved=n_unresolved,
        win_rate_resolved=win_rate_resolved,
        avg_R_resolved=avg_R_resolved,
    )


# ── League runner ─────────────────────────────────────────────

def run_league() -> None:
    results: List[RuleResult] = []

    for rule_id, cfg in ALL_DOUBLE_BOTTOM_RULES.items():
        print(f"\n[Rule League] Running {rule_id} on {cfg.ticker} ({cfg.period}, {cfg.interval})...")
        df = load_underlying_5m(cfg.ticker, cfg.period, cfg.interval)
        print(f"[Rule League] Loaded {len(df)} bars for {cfg.ticker}.")

        patterns = detect_double_bottoms(df, cfg)
        print(f"[Rule League] {cfg.ticker}: detected {len(patterns)} candidate double bottoms")

        trades = plan_trades_from_double_bottoms(df, cfg, patterns)
        print(f"[Rule League] {cfg.ticker}: planned {len(trades)} trades")

        result = summarize_rule(cfg, trades)
        results.append(result)

    # Rank by avg_R_resolved (descending)
    results_sorted = sorted(results, key=lambda r: r.avg_R_resolved, reverse=True)

    print("\n=== Rule League – Double-Bottom 5m (60d) ===")
    print(f"{'Rank':<4} {'Rule ID':<20} {'Ticker':<6} {'Trades':<7} "
          f"{'Res':<5} {'Win%':<7} {'AvgR':<7}")
    print("-" * 70)
    for idx, r in enumerate(results_sorted, start=1):
        print(
            f"{idx:<4} {r.rule_id:<20} {r.ticker:<6} "
            f"{r.n_trades:<7} {r.n_resolved:<5} "
            f"{r.win_rate_resolved:>6.1f}% {r.avg_R_resolved:>6.2f}"
        )
    print()


def main() -> None:
    run_league()


if __name__ == "__main__":
    main()
