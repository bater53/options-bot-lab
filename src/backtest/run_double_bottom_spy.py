# src/backtest/run_double_bottom_spy.py

from __future__ import annotations

import pandas as pd
import yfinance as yf

from src.backtest.double_bottom_backtest import backtest_spy_double_bottom


def load_spy_5m(days: int = 60) -> pd.DataFrame:
    """
    Load SPY 5m candles using yfinance.

    NOTE: Yahoo only allows ~last 60 days of intraday (5m) data.
    If you request more than that, we clip to 60 days to avoid errors.
    """
    max_days_5m = 60
    use_days = min(days, max_days_5m)

    period = f"{use_days}d"
    print(f"Loading SPY 5m candles for about the last {use_days} days…")

    df = yf.download(
        "SPY",
        interval="5m",
        period=period,
        auto_adjust=False,  # explicit so behavior is stable
        progress=False,
    )

    if df.empty:
        raise RuntimeError(f"No SPY 5m data returned from yfinance for period={period!r}")

    # --- Debug: show raw columns from yfinance ---
    print("Raw SPY df.columns:", list(df.columns))

    # --- Normalize columns (handle tuple/MultiIndex & strings) ---
    flat_cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            # Typical MultiIndex from yfinance: (field, ticker), e.g. ('Open', 'SPY')
            # We WANT the field name: 'Open', 'High', 'Low', 'Close', 'Volume'
            field = c[0] if len(c) > 0 else "col"
            name = str(field).lower()
        else:
            name = str(c).lower()
        flat_cols.append(name)

    df.columns = flat_cols
    print("Normalized df.columns:", list(df.columns))

    # --- Ensure we have standard OHLCV columns the strategy expects ---
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns from yfinance data: {missing}")

    # Drop timezone if present, keep index clean
    if getattr(df.index, "tz", None) is not None:
        df = df.tz_convert(None)

    # Keep only the columns we care about, in a known order
    df = df[["open", "high", "low", "close", "volume"]]

    return df


def main() -> None:
    days = 60  # keep at 60 for 5m, yfinance limitation
    candles_5m = load_spy_5m(days=days)
    print(f"Loaded {len(candles_5m)} bars.")

    res = backtest_spy_double_bottom(
        candles_5m,
        max_risk_per_trade=200.0,
        max_bars_ahead=78,
    )

    print("\n=== Double Bottom Backtest on SPY (5m) ===")
    print("n_trades:", res.n_trades)
    print("n_wins:", res.n_wins)
    print("n_losses:", res.n_losses)
    print("win_rate:", f"{res.win_rate * 100:.1f}%")
    print("avg_R:", f"{res.avg_r:.2f}")

    if res.trades:
        print("\nFirst few trades:")
        for t in res.trades[:5]:
            r_val = t.r_multiple if t.r_multiple is not None else float("nan")
            print(
                f"- t_entry={t.trade.t_entry}, "
                f"entry={t.trade.underlying_entry:.2f}, "
                f"stop={t.trade.underlying_stop:.2f}, "
                f"target={t.trade.underlying_target:.2f}, "
                f"R={r_val:.2f}"
            )
    else:
        print("\n(No completed trades in this window.)")


if __name__ == "__main__":
    main()
