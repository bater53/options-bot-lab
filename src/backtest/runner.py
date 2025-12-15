import pandas as pd
import yfinance as yf

from src.utils.time import to_ny, group_by_date_ny, rth_only
from src.strategies.orb_v1 import orb_v1_day
from src.config import ORBConfig


def fetch_data(symbol: str, start: str, end: str):
    """
    yfinance limitation: 5m data is limited to recent history.
    We'll use what it provides; later we can swap to CSV/Polygon.
    """
    df5 = yf.download(symbol, start=start, end=end, interval="5m", progress=False)
    df1h = yf.download(symbol, start=start, end=end, interval="60m", progress=False)

    if df5.empty or df1h.empty:
        return df5, df1h

    df5 = to_ny(df5)
    df1h = to_ny(df1h)
    return df5, df1h


def run_orb_backtest(symbol="SPY", start="2025-06-01", end="2025-11-01", cfg=None):
    cfg = cfg or ORBConfig(symbol=symbol)

    df5, df1h = fetch_data(symbol, start, end)
    if df5.empty:
        raise RuntimeError(
            "No 5m data returned from yfinance. "
            "Try a more recent date range."
        )

    df5 = rth_only(df5)
    df1h = rth_only(df1h) if not df1h.empty else df1h

    days5 = group_by_date_ny(df5)
    days1h = group_by_date_ny(df1h) if not df1h.empty else {}

    trades = []
    for d, day5 in days5.items():
        day1h = days1h.get(d, pd.DataFrame())
        t = orb_v1_day(day5, day1h, cfg)
        if t and t.exit_time is not None:
            trades.append(t)

    return trades
