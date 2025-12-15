# src/backtest/trend_pullback_backtest.py

from dataclasses import dataclass
from typing import List, Dict, Tuple
import pandas as pd


@dataclass
class TrendPullbackTrade:
    symbol: str
    direction: str  # 'long' or 'short'
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    target_price: float
    exit_time: pd.Timestamp
    exit_price: float
    hit_target: bool
    hit_stop: bool
    R: float


@dataclass
class TrendPullbackStats:
    symbol: str
    n_trades: int
    n_wins: int
    n_losses: int
    win_rate: float
    avg_R: float
    total_R: float


def backtest_trend_pullback_signals(
    df_5m: pd.DataFrame,
    signals: List[Dict],
    stop_pct: float = 0.4,
    target_R: float = 2.0,
    session_end: str = "15:55",
) -> Tuple[List[TrendPullbackTrade], TrendPullbackStats]:
    """
    df_5m: 5m OHLCV with DatetimeIndex (NY time), columns include: ['open','high','low','close',...]
    signals: list of dicts like:
        {
            "timestamp": pd.Timestamp,
            "symbol": "SPY" or "QQQ",
            "direction": "long" | "short",
            "note": "bullish_trend_pullback" | "bearish_trend_pullback",
        }
    """
    trades: List[TrendPullbackTrade] = []

    for sig in signals:
        ts = sig["timestamp"]
        symbol = sig["symbol"]
        direction = sig["direction"]

        if ts not in df_5m.index:
            # If timestamp mismatch, skip this one
            continue

        entry_time = ts
        entry_price = float(df_5m.loc[ts, "close"])

        if direction == "long":
            stop_price = entry_price * (1.0 - stop_pct / 100.0)
            target_price = entry_price + (entry_price - stop_price) * target_R
        else:  # short
            stop_price = entry_price * (1.0 + stop_pct / 100.0)
            target_price = entry_price - (stop_price - entry_price) * target_R

        # All bars for the same day
        day_mask = (df_5m.index.date == ts.date())
        day_df = df_5m.loc[day_mask]

        # Only bars strictly after entry
        fwd_df = day_df[day_df.index > entry_time]

        # Optional cutoff by session_end
        if session_end is not None:
            cutoff_ts = pd.Timestamp(
                ts.strftime(f"%Y-%m-%d {session_end}")
            ).tz_localize(ts.tz)
            fwd_df = fwd_df[fwd_df.index <= cutoff_ts]

        exit_time = None
        exit_price = None
        hit_target = False
        hit_stop = False

        if fwd_df.empty:
            # No bars after entry → flat at entry
            exit_time = entry_time
            exit_price = entry_price
        else:
            for t, row in fwd_df.iterrows():
                high = float(row["high"])
                low = float(row["low"])

                if direction == "long":
                    # Check stop first, then target (choose a consistent convention)
                    if low <= stop_price:
                        exit_time = t
                        exit_price = stop_price
                        hit_stop = True
                        break
                    if high >= target_price:
                        exit_time = t
                        exit_price = target_price
                        hit_target = True
                        break
                else:  # short
                    if high >= stop_price:
                        exit_time = t
                        exit_price = stop_price
                        hit_stop = True
                        break
                    if low <= target_price:
                        exit_time = t
                        exit_price = target_price
                        hit_target = True
                        break

            if exit_time is None:
                # Neither hit → exit at last bar of day
                exit_time = fwd_df.index[-1]
                exit_price = float(fwd_df.iloc[-1]["close"])

        if direction == "long":
            risk_per_share = entry_price - stop_price
            pnl = exit_price - entry_price
        else:
            risk_per_share = stop_price - entry_price
            pnl = entry_price - exit_price

        R = 0.0
        if risk_per_share > 0:
            R = pnl / risk_per_share

        trades.append(
            TrendPullbackTrade(
                symbol=symbol,
                direction=direction,
                entry_time=entry_time,
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                exit_time=exit_time,
                exit_price=exit_price,
                hit_target=hit_target,
                hit_stop=hit_stop,
                R=R,
            )
        )

    n_trades = len(trades)
    n_wins = sum(1 for tr in trades if tr.R > 0)
    n_losses = sum(1 for tr in trades if tr.R < 0)
    win_rate = (n_wins / n_trades * 100.0) if n_trades > 0 else 0.0
    avg_R = (sum(tr.R for tr in trades) / n_trades) if n_trades > 0 else 0.0
    total_R = sum(tr.R for tr in trades)

    stats = TrendPullbackStats(
        symbol=trades[0].symbol if trades else "N/A",
        n_trades=n_trades,
        n_wins=n_wins,
        n_losses=n_losses,
        win_rate=win_rate,
        avg_R=avg_R,
        total_R=total_R,
    )

    return trades, stats
