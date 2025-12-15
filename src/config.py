from dataclasses import dataclass

@dataclass
class ORBConfig:
    symbol: str = "SPY"
    tz: str = "America/New_York"

    # Opening range window (5m candles)
    or_candle_starts = ("09:30", "09:35", "09:40")
    or_end_time: str = "09:45"  # first candle close after range

    # Filters
    or_min_pct: float = 0.0015   # 0.15%
    or_max_pct: float = 0.0080   # 0.80%
    flat_band_pct: float = 0.0010  # 0.10% around EMA

    # Trend filter
    ema_period_1h: int = 20

    # Volume confirm (5m)
    vol_lookback: int = 20
    use_vol_confirm: bool = True

    # Risk/management
    stop_frac: float = 0.25
    tp_R: float = 2.0
    be_R: float = 1.0

    # Trade limits
    no_entry_after: str = "11:30"
    force_exit: str = "15:55"
    max_trades_per_day: int = 1
