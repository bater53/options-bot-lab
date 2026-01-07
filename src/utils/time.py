# src/utils/time.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Optional

try:
    # Python 3.9+
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]

NY_TZ = ZoneInfo("America/New_York")


def now_ny() -> datetime:
    """Current time in America/New_York."""
    return datetime.now(tz=NY_TZ)


def to_ny(dt: datetime) -> datetime:
    """
    Convert any datetime to New York time.

    - If dt is naive (no tzinfo), assume it's already NY time and attach NY tz.
    - If dt is aware, convert to NY.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=NY_TZ)
    return dt.astimezone(NY_TZ)


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
