from __future__ import annotations

from dataclasses import dataclass
from typing import List, Callable

import pandas as pd

from src.strategies.double_bottom_spy_calls import (
    plan_spy_double_bottom_trades,
    PlannedOptionsTrade,
)
from src.utils.time import to_ny


@dataclass
class TradeOutcome:
    trade: PlannedOptionsTrade
    hit_target: bool
    hit_stop: bool
    r_multiple: float | None  # reward in R units (2.0 = 2R, -1.0 = full loss)


@dataclass
class BacktestResult:
    trades: List[TradeOutcome]

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def n_wins(self) -> int:
        return sum(1 for t in self.trades if t.hit_target and not t.hit_stop)

    @property
    def n_losses(self) -> int:
        return sum(1 for t in self.trades if t.hit_stop and not t.hit_target)

    @property
    def win_rate(self) -> float:
        return (self.n_wins / self.n_trades) if self.n_trades else 0.0

    @property
    def avg_r(self) -> float:
        vals = [t.r_multiple for t in self.trades if t.r_multiple is not None]
        return sum(vals) / len(vals) if vals else 0.0


# ---------- Generic engine ----------

def backtest_double_bottom(
    candles_5m: pd.DataFrame,
    *,
    planner_fn: Callable[..., List[PlannedOptionsTrade]] = plan_spy_double_bottom_trades,
    max_risk_per_trade: float = 200.0,
    max_bars_ahead: int = 78,
    debug: bool = False,
) -> BacktestResult:
    """
    Generic double-bottom backtest; does not care which symbol it is.
    """
    if candles_5m.empty:
        return BacktestResult(trades=[])

    candles_5m = candles_5m.copy()
    candles_5m.index = candles_5m.index.map(to_ny)
    candles_5m = candles_5m.sort_index()

    trades = planner_fn(
        candles_5m,
        max_risk_per_trade=max_risk_per_trade,
        debug=debug,
    )
    print(f"[debug] planned {len(trades)} trades")

    outcomes: List[TradeOutcome] = []

    if not trades:
        return BacktestResult(trades=outcomes)

    for trade in trades:
        entry_time = trade.t_entry

        future_bars = candles_5m.loc[entry_time:]
        future_bars = future_bars.iloc[1 : 1 + max_bars_ahead]
        if future_bars.empty:
            continue

        entry_price = trade.underlying_entry
        stop = trade.underlying_stop
        target = trade.underlying_target

        risk_per_share = entry_price - stop
        if risk_per_share <= 0:
            continue

        hit_target = False
        hit_stop = False
        exit_price = None

        for _, bar in future_bars.iterrows():
            low = float(bar["low"])
            high = float(bar["high"])

            if low <= stop:
                hit_stop = True
                exit_price = stop
                break

            if high >= target:
                hit_target = True
                exit_price = target
                break

        if not hit_target and not hit_stop:
            continue

        r_multiple = None
        if exit_price is not None:
            r_multiple = (exit_price - entry_price) / risk_per_share

        outcomes.append(
            TradeOutcome(
                trade=trade,
                hit_target=hit_target,
                hit_stop=hit_stop,
                r_multiple=r_multiple,
            )
        )

    return BacktestResult(trades=outcomes)


def backtest_spy_double_bottom(
    candles_5m: pd.DataFrame,
    *,
    max_risk_per_trade: float = 200.0,
    max_bars_ahead: int = 78,
    debug: bool = False,
) -> BacktestResult:
    """
    Backwards-compatible SPY wrapper around the generic backtest_double_bottom.

    We keep the function name so existing code (run_double_bottom_spy)
    continues to work, but all logic is delegated to the generic engine.
    """
    return backtest_double_bottom(
        candles_5m,
        max_risk_per_trade=max_risk_per_trade,
        max_bars_ahead=max_bars_ahead,
        debug=debug,
    )
