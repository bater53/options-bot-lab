from dataclasses import dataclass
import pandas as pd
from ..utils.indicators import ema
from ..utils.time import rth_only

@dataclass
class Trade:
    side: str  # "long" or "short"
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    tp_price: float
    be_trigger: float
    exit_time: pd.Timestamp = None
    exit_price: float = None
    reason: str = None
    R: float = 0.0

    @property
    def pnl_R(self):
        if self.exit_price is None:
            return 0.0
        if self.side == "long":
            return (self.exit_price - self.entry_price) / self.R
        else:
            return (self.entry_price - self.exit_price) / self.R


def compute_opening_range(day_5m: pd.DataFrame, cfg):
    # OR candles start at 09:30, 09:35, 09:40
    or_candles = day_5m[day_5m.index.strftime("%H:%M").isin(cfg.or_candle_starts)]
    if len(or_candles) < 3:
        return None

    OR_high = or_candles["High"].max()
    OR_low  = or_candles["Low"].min()
    OR_mid  = (OR_high + OR_low) / 2
    OR_size = OR_high - OR_low
    OR_size_pct = OR_size / OR_mid
    return OR_high, OR_low, OR_mid, OR_size, OR_size_pct


def compute_bias(day_1h: pd.DataFrame, cfg, cutoff_ts: pd.Timestamp):
    # EMA on 1h closes using data up to cutoff
    d1h = day_1h[day_1h.index <= cutoff_ts]
    if len(d1h) < cfg.ema_period_1h:
        return None  # not enough EMA history that day

    ema20 = ema(d1h["Close"], cfg.ema_period_1h)
    last_close = d1h["Close"].iloc[-1]
    last_ema = ema20.iloc[-1]

    flat = abs(last_close - last_ema) / last_ema <= cfg.flat_band_pct
    bull = last_close > last_ema and not flat
    bear = last_close < last_ema and not flat
    return bull, bear, flat, float(last_ema), float(last_close)


def orb_v1_day(day_5m: pd.DataFrame, day_1h: pd.DataFrame, cfg) -> Trade | None:
    day_5m = rth_only(day_5m)
    if day_5m.empty:
        return None

    or_tuple = compute_opening_range(day_5m, cfg)
    if or_tuple is None:
        return None
    OR_high, OR_low, OR_mid, OR_size, OR_size_pct = or_tuple

    if OR_size_pct < cfg.or_min_pct or OR_size_pct > cfg.or_max_pct:
        return None  # skip day

    # cutoff at 09:45 close
    cutoff_ts = day_5m.index.min().normalize() + pd.Timedelta(hours=9, minutes=45)
    bias = compute_bias(day_1h, cfg, cutoff_ts)
    if bias is None:
        return None
    bull, bear, flat, ema_val, last_close_1h = bias
    if flat:
        return None

    trade_taken = False
    trade = None

    # Precompute rolling avg volume
    vol_roll = day_5m["Volume"].rolling(cfg.vol_lookback).mean()

    for i in range(len(day_5m) - 1):
        c = day_5m.iloc[i]
        t = day_5m.index[i]
        t_str = t.strftime("%H:%M")
        if t_str < cfg.or_end_time:
            continue
        if t_str > cfg.force_exit and trade:
            trade.exit_time = t
            trade.exit_price = c["Close"]
            trade.reason = "force_exit"
            break

        # manage open trade
        if trade and trade.exit_time is None:
            high = c["High"]
            low  = c["Low"]

            # Conservative intrabar ordering: stop first, then target
            if trade.side == "long":
                if low <= trade.stop_price:
                    trade.exit_time = t
                    trade.exit_price = trade.stop_price
                    trade.reason = "stop"
                elif high >= trade.tp_price:
                    trade.exit_time = t
                    trade.exit_price = trade.tp_price
                    trade.reason = "tp"
                elif high >= trade.be_trigger and trade.stop_price < trade.entry_price:
                    trade.stop_price = trade.entry_price  # move to breakeven

            else:  # short
                if high >= trade.stop_price:
                    trade.exit_time = t
                    trade.exit_price = trade.stop_price
                    trade.reason = "stop"
                elif low <= trade.tp_price:
                    trade.exit_time = t
                    trade.exit_price = trade.tp_price
                    trade.reason = "tp"
                elif low <= trade.be_trigger and trade.stop_price > trade.entry_price:
                    trade.stop_price = trade.entry_price

            if trade.exit_time is not None:
                break
            continue  # next candle

        # no entries if already traded or after cutoff
        if trade_taken or t_str > cfg.no_entry_after:
            continue

        # volume confirm
        vol_ok = True
        if cfg.use_vol_confirm:
            avg_vol = vol_roll.iloc[i]
            vol_ok = pd.notna(avg_vol) and c["Volume"] > avg_vol

        # trigger long / short
        close = c["Close"]

        if bull and close > OR_high and vol_ok:
            nxt = day_5m.iloc[i + 1]
            entry_price = float(nxt["Open"])
            entry_time = day_5m.index[i + 1]
            stop_price = float(OR_high - cfg.stop_frac * OR_size)
            R = entry_price - stop_price
            if R <= 0:
                continue
            tp_price = entry_price + cfg.tp_R * R
            be_trigger = entry_price + cfg.be_R * R

            trade = Trade(
                side="long",
                entry_time=entry_time,
                entry_price=entry_price,
                stop_price=stop_price,
                tp_price=tp_price,
                be_trigger=be_trigger,
                R=R
            )
            trade_taken = True
            continue

        if bear and close < OR_low and vol_ok:
            nxt = day_5m.iloc[i + 1]
            entry_price = float(nxt["Open"])
            entry_time = day_5m.index[i + 1]
            stop_price = float(OR_low + cfg.stop_frac * OR_size)
            R = stop_price - entry_price
            if R <= 0:
                continue
            tp_price = entry_price - cfg.tp_R * R
            be_trigger = entry_price - cfg.be_R * R

            trade = Trade(
                side="short",
                entry_time=entry_time,
                entry_price=entry_price,
                stop_price=stop_price,
                tp_price=tp_price,
                be_trigger=be_trigger,
                R=R
            )
            trade_taken = True
            continue

    return trade
