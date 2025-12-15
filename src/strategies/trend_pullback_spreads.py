# src/strategies/trend_pullback_spreads.py

from dataclasses import dataclass
from typing import Literal, List
import pandas as pd
import numpy as np


@dataclass
class TrendPullbackSignal:
    symbol: str
    direction: Literal["long", "short"]  # long = calls, short = puts
    timestamp: pd.Timestamp
    entry_price_underlying: float
    swing_low: float
    swing_high: float
    notes: str = ""


STRATEGY_CONFIG = {
    "symbols": ["SPY"],
    "trend_timeframe": "15min",
    "intraday_timeframe": "5min",
    "session_start": "09:45",
    "session_end": "15:30",
    "ema_fast_len": 20,
    "ema_slow_len": 50,
    "use_vwap": True,
    "max_pullback_pct_from_high": 2.0,    # max pullback from recent high (bullish)
    "max_distance_from_vwap_pct": 0.5,    # how close to VWAP counts as "in zone"
}


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def classify_trend(
    df_15m: pd.DataFrame,
    ts: pd.Timestamp,
    cfg: dict,
) -> Literal["bullish", "bearish", "none"]:
    """Very simple trend classifier on 15m bars."""
    # get last 15m bar at or before ts
    row = df_15m.loc[:ts].iloc[-1]
    ema_fast = row["ema_fast"]
    ema_slow = row["ema_slow"]
    close = row["close"]

    if close > ema_fast > ema_slow:
        return "bullish"
    if close < ema_fast < ema_slow:
        return "bearish"
    return "none"


def compute_indicators(df_5m: pd.DataFrame, df_15m: pd.DataFrame, cfg: dict) -> None:
    # trend EMAs on 15m
    df_15m["ema_fast"] = ema(df_15m["close"], cfg["ema_fast_len"])
    df_15m["ema_slow"] = ema(df_15m["close"], cfg["ema_slow_len"])

    # simple VWAP on 5m
    if cfg["use_vwap"]:
        pv = (df_5m["close"] * df_5m["volume"]).cumsum()
        vv = df_5m["volume"].cumsum().replace(0, np.nan)
        df_5m["vwap"] = pv / vv
    else:
        df_5m["vwap"] = np.nan


def _is_bullish_trigger(df_5m: pd.DataFrame, idx: int) -> bool:
    """Very simple bullish trigger: bullish engulfing pattern."""
    if idx < 1:
        return False
    prev = df_5m.iloc[idx - 1]
    curr = df_5m.iloc[idx]

    prev_body_low = min(prev["open"], prev["close"])
    prev_body_high = max(prev["open"], prev["close"])
    curr_body_low = min(curr["open"], curr["close"])
    curr_body_high = max(curr["open"], curr["close"])

    return (
        curr["close"] > curr["open"]  # green candle
        and curr_body_low <= prev_body_low
        and curr_body_high >= prev_body_high
    )


def _is_bearish_trigger(df_5m: pd.DataFrame, idx: int) -> bool:
    """Very simple bearish trigger: bearish engulfing pattern."""
    if idx < 1:
        return False
    prev = df_5m.iloc[idx - 1]
    curr = df_5m.iloc[idx]

    prev_body_low = min(prev["open"], prev["close"])
    prev_body_high = max(prev["open"], prev["close"])
    curr_body_low = min(curr["open"], curr["close"])
    curr_body_high = max(curr["open"], curr["close"])

    return (
        curr["close"] < curr["open"]  # red candle
        and curr_body_low <= prev_body_low
        and curr_body_high >= prev_body_high
    )


def find_trend_pullback_signals(
    symbol: str,
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
    cfg: dict = STRATEGY_CONFIG,
) -> List[TrendPullbackSignal]:
    """
    Scan 5m data for bullish/bearish trend pullback signals.
    Underlying-only for now – no options legs yet.
    """
    compute_indicators(df_5m, df_15m, cfg)
    signals: List[TrendPullbackSignal] = []

    session_start = cfg["session_start"]
    session_end = cfg["session_end"]

    recent_high = None
    recent_low = None

    for i in range(len(df_5m)):
        row = df_5m.iloc[i]
        ts = row.name  # DateTimeIndex
        time_str = ts.strftime("%H:%M")

        if time_str < session_start or time_str > session_end:
            continue

        trend = classify_trend(df_15m, ts, cfg)
        price = row["close"]
        vwap = row.get("vwap", np.nan)

        # book-keeping
        if recent_high is None or price > recent_high:
            recent_high = price
        if recent_low is None or price < recent_low:
            recent_low = price

        # Bullish: pullback from recent high toward VWAP with bullish trigger
        if trend == "bullish" and recent_high is not None:
            pullback_pct = (recent_high - price) / recent_high * 100
            if pullback_pct <= cfg["max_pullback_pct_from_high"]:
                if cfg["use_vwap"] and not np.isnan(vwap):
                    dist_vwap_pct = abs(price - vwap) / vwap * 100
                    in_zone = dist_vwap_pct <= cfg["max_distance_from_vwap_pct"]
                else:
                    in_zone = True

                if in_zone and _is_bullish_trigger(df_5m, i):
                    signals.append(
                        TrendPullbackSignal(
                            symbol=symbol,
                            direction="long",
                            timestamp=ts,
                            entry_price_underlying=price,
                            swing_low=recent_low if recent_low is not None else price,
                            swing_high=recent_high,
                            notes="bullish_trend_pullback",
                        )
                    )

        # Bearish: pullback from recent low toward VWAP with bearish trigger
        if trend == "bearish" and recent_low is not None:
            pullback_pct = (price - recent_low) / recent_low * 100
            if pullback_pct <= cfg["max_pullback_pct_from_high"]:
                if cfg["use_vwap"] and not np.isnan(vwap):
                    dist_vwap_pct = abs(price - vwap) / vwap * 100
                    in_zone = dist_vwap_pct <= cfg["max_distance_from_vwap_pct"]
                else:
                    in_zone = True

                if in_zone and _is_bearish_trigger(df_5m, i):
                    signals.append(
                        TrendPullbackSignal(
                            symbol=symbol,
                            direction="short",
                            timestamp=ts,
                            entry_price_underlying=price,
                            swing_low=recent_low,
                            swing_high=recent_high if recent_high is not None else price,
                            notes="bearish_trend_pullback",
                        )
                    )

    return signals
