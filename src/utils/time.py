# src/utils/time.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime,time, timedelta
from typing import Optional, Union

import pandas as pd
from zoneinfo import ZoneInfo


# --- Time zone --------------------------------------------------------------

NY_TZ = ZoneInfo("America/New_York")

# Acceptable datetime-like types for our helpers
DateLike = Union[datetime, pd.Timestamp, pd.DatetimeIndex]


def now_ny() -> datetime:
    """
    Current time in America/New_York as an aware datetime.
    """
    return datetime.now(tz=NY_TZ)


def to_ny(obj: DateLike) -> DateLike:
    """
    Convert a naive/aware datetime, Timestamp, or DatetimeIndex to America/New_York.

    - If obj is naive: interpret it as NY time and attach NY tz.
    - If obj is aware: convert it to NY tz.
    """
    # Case 1: DatetimeIndex (what the QQQ runner uses on df.index)
    if isinstance(obj, pd.DatetimeIndex):
        if obj.tz is None:
            return obj.tz_localize(NY_TZ)
        else:
            return obj.tz_convert(NY_TZ)

    # Case 2: pandas Timestamp
    if isinstance(obj, pd.Timestamp):
        if obj.tz is None:
            return obj.tz_localize(NY_TZ)
        else:
            return obj.tz_convert(NY_TZ)

    # Case 3: plain Python datetime
    if isinstance(obj, datetime):
        if obj.tzinfo is None:
            # Assume it's already NY clock time, just attach tzinfo
            return obj.replace(tzinfo=NY_TZ)
        else:
            return obj.astimezone(NY_TZ)

    raise TypeError(f"to_ny: unsupported type {type(obj)}")


# --- Market hours helpers ---------------------------------------------------

@dataclass
class MarketHours:
    open_time: time = time(9, 30)   # 09:30
    close_time: time = time(16, 0)  # 16:00


DEFAULT_HOURS = MarketHours()


def is_weekend(dt: datetime) -> bool:
    """
    True if Saturday (5) or Sunday (6) in NY.
    """
    dt_ny = to_ny(dt)
    return dt_ny.weekday() >= 5


def is_regular_session(
    dt: Optional[datetime] = None,
    hours: MarketHours = DEFAULT_HOURS,
) -> bool:
    """
    True if dt is within regular RTH session (no holidays/half-days logic yet).

    This is "good enough" for gating intraday strategies/backtests.
    """
    if dt is None:
        dt = now_ny()

    dt_ny = to_ny(dt)
    if is_weekend(dt_ny):
        return False

    t_ = dt_ny.time()
    return hours.open_time <= t_ <= hours.close_time


def next_regular_session_open(
    dt: Optional[datetime] = None,
    hours: MarketHours = DEFAULT_HOURS,
) -> datetime:
    """
    Return the next regular-session open (ignores holidays for now).
    """
    if dt is None:
        dt = now_ny()

    dt_ny = to_ny(dt)

    # If we're before today's open on a weekday, use today
    if (not is_weekend(dt_ny)) and (dt_ny.time() < hours.open_time):
        base_day = dt_ny.date()
    else:
        # Otherwise move to next day
        base_day = (dt_ny + timedelta(days=1)).date()

    # Skip weekends
    while True:
        candidate = datetime.combine(base_day, hours.open_time, tzinfo=NY_TZ)
        if not is_weekend(candidate):
            return candidate
        base_day = base_day + timedelta(days=1)


@dataclass
class MarketHours:
    open_time: time = time(9, 30)   # 09:30
    close_time: time = time(16, 0)  # 16:00


DEFAULT_HOURS = MarketHours()


def is_weekend(dt: datetime) -> bool:
    """True if Saturday (5) or Sunday (6) in NY."""
    dt_ny = to_ny(dt)
    return dt_ny.weekday() >= 5


def is_regular_session(dt: Optional[datetime] = None,
                       hours: MarketHours = DEFAULT_HOURS) -> bool:
    """
    True if dt is within regular RTH session (no holidays/half days logic yet).

    This is "good enough" for strategy/backtest gating. You can bolt on
    a proper holiday calendar later if needed.
    """
    if dt is None:
        dt = now_ny()
    dt_ny = to_ny(dt)
    if is_weekend(dt_ny):
        return False

    t_ = dt_ny.time()
    return hours.open_time <= t_ <= hours.close_time


def next_regular_session_open(
    dt: Optional[datetime] = None,
    hours: MarketHours = DEFAULT_HOURS
) -> datetime:
    """
    Return the next regular-session open (ignores holidays for now).
    """
    if dt is None:
        dt = now_ny()
    dt_ny = to_ny(dt)

    # If we're before today's open on a weekday, use today
    if (not is_weekend(dt_ny)
            and dt_ny.time() < hours.open_time):
        base_day = dt_ny.date()
    else:
        # Otherwise move to next day
        base_day = (dt_ny + timedelta(days=1)).date()

    # Skip weekends
    while True:
        candidate = datetime.combine(base_day, hours.open_time, tzinfo=NY_TZ)
        if not is_weekend(candidate):
            return candidate
        base_day = base_day + timedelta(days=1)
