# src/backtest/double_bottom_backtest.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


# ---------------------------
# Configs & Data Structures
# ---------------------------

@dataclass
class DoubleBottomConfig:
    max_low_diff_pct: float = 0.005      # 0.5% allowed difference between lows
    min_bounce_pct: float = 0.01         # neckline at least 1% above lows
    breakout_buffer_pct: float = 0.001   # breakout must exceed neckline by 0.1%
    min_bars_between_lows: int = 2
    max_bars_between_lows: int = 10
    max_bars_after_breakout: int = 10    # for backtest exit search

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


def backtest_spy_double_bottom(
    candles_5m: pd.DataFrame,
    *,
    max_risk_per_trade: float = 200.0,
    max_bars_ahead: int = 24,
) -> BacktestResult:
    
    from src.strategies.double_bottom_spy_calls import (
    plan_spy_double_bottom_trades,
    PlannedOptionsTrade,
    DoubleBottomConfig,   # if you export it there
)

    cfg = DoubleBottomConfig(...)
    trades = plan_spy_double_bottom_trades(
    candles_5m, max_risk_per_trade=max_risk_per_trade
    )
    print(f"[debug] planned {len(trades)} trades")
  
    """
    Very simple backtest:

    - Uses plan_spy_double_bottom_trades to generate trades.
    - For each trade, walks forward up to `max_bars_ahead` bars.
    - If price hits target first -> win.
      If price hits stop first   -> loss.
      If neither hit             -> trade skipped (no outcome).
    """

    if candles_5m.empty:
        return BacktestResult(trades=[])

    # 🔧 Normalize index to NY tz so it matches trade.t_entry
    candles_5m = candles_5m.copy()
    candles_5m.index = candles_5m.index.map(to_ny)
    candles_5m = candles_5m.sort_index()

    trades = plan_spy_double_bottom_trades(
        candles_5m, 
        max_risk_per_trade=max_risk_per_trade,
       # reward_r_multiple=2.0,
    )

    outcomes: List[TradeOutcome] = []

    if not trades:
        return BacktestResult(trades=outcomes)

    for trade in trades:
        entry_time = trade.t_entry  # tz-aware NY

        # Slice from entry_time forward
        future_bars = candles_5m.loc[entry_time:]

        # Skip the "entry" bar itself, look ahead max_bars_ahead bars
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

            # Stop first?
            if low <= stop:
                hit_stop = True
                exit_price = stop
                break

            # Target?
            if high >= target:
                hit_target = True
                exit_price = target
                break

        if not hit_target and not hit_stop:
            # No resolution within horizon
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
