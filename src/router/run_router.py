# src/router/run_router.py
from __future__ import annotations

import os
import sys
import subprocess
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as e:
    raise SystemExit("Missing dependency: yfinance\nInstall: pip install yfinance") from e


Regime = Literal["TREND", "RANGE", "HIGH_VOL"]


@dataclass
class RegimeStats:
    regime: Regime
    close: float
    atr_pct: float
    adx_1h: float
    ema_sep_pct: float
 
    
def run_module_unbuffered(module: str) -> int:
    cmd = [sys.executable, "-u", "-m", module]
    print(f"[router] launching: {' '.join(cmd)}", flush=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.run(cmd, env=env).returncode



def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]).lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    out = df[keep].copy().dropna()
    return out


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    tr_sm = tr.ewm(alpha=1 / n, adjust=False).mean()
    plus_dm_sm = pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean()
    minus_dm_sm = pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean()

    plus_di = 100 * (plus_dm_sm / tr_sm.replace(0, np.nan))
    minus_di = 100 * (minus_dm_sm / tr_sm.replace(0, np.nan))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean().fillna(0.0)


def fetch(symbol: str, interval: str, period: str) -> pd.DataFrame:
    df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=False)
    return _normalize_ohlc(df)


def detect_regime(df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_daily: pd.DataFrame) -> RegimeStats:
    if df_15m.empty or df_1h.empty or df_daily.empty:
        return RegimeStats("RANGE", float("nan"), float("nan"), 0.0, 0.0)

    close15 = float(df_15m["close"].iloc[-1])
    atr15 = float(_atr(df_15m, 14).iloc[-1])
    atr_pct = atr15 / close15 if close15 else float("nan")

    adx_1h = float(_adx(df_1h, 14).iloc[-1])

    cD = df_daily["close"]
    ema20 = float(_ema(cD, 20).iloc[-1])
    ema50 = float(_ema(cD, 50).iloc[-1])
    closeD = float(cD.iloc[-1])
    ema_sep_pct = abs(ema20 - ema50) / closeD if closeD else 0.0

    # thresholds (tune later)
    HIGH_VOL_ATR_PCT = 0.012   # 1.2% ATR on 15m
    TREND_ADX = 20.0
    TREND_EMA_SEP = 0.003      # 0.3%

    if np.isfinite(atr_pct) and atr_pct >= HIGH_VOL_ATR_PCT:
        regime: Regime = "HIGH_VOL"
    elif adx_1h >= TREND_ADX and ema_sep_pct >= TREND_EMA_SEP:
        regime = "TREND"
    else:
        regime = "RANGE"

    return RegimeStats(regime, close15, atr_pct, adx_1h, ema_sep_pct)

def main() -> int:
    symbol = "SPY"  # regime “read” symbol
    print(f"[router] fetching regime data for {symbol} ...")

    df_15m = fetch(symbol, interval="15m", period="60d")
    df_1h = fetch(symbol, interval="60m", period="60d")
    df_daily = fetch(symbol, interval="1d", period="1y")

    stats = detect_regime(df_15m, df_1h, df_daily)

    print(
        f"[router] regime={stats.regime} | close={stats.close:.2f} | "
        f"ATR%={stats.atr_pct*100:.2f}% | ADX(1h)={stats.adx_1h:.1f} | EMAsep={stats.ema_sep_pct*100:.2f}%"
    )

    if stats.regime == "TREND":
        return run_module_unbuffered("src.options.run_trend_pullback_spy_qqq_spreads_lab")

    if stats.regime == "RANGE":
        rc1 = run_module_unbuffered("src.options.run_double_bottom_spy_options_lab")
        rc2 = run_module_unbuffered("src.options.run_double_bottom_qqq_options_lab")
        return 0 if (rc1 == 0 and rc2 == 0) else 1


    print("[router] HIGH_VOL -> NO TRADE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
