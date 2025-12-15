# src/backtest/run_trend_pullback_spy_qqq_lab.py

import pandas as pd
import yfinance as yf
from src.backtest.trend_pullback_backtest import backtest_trend_pullback_signals

from src.strategies.trend_pullback_spreads import (
    STRATEGY_CONFIG,
    find_trend_pullback_signals,
)

# Override symbols just for this backtest script:
STRATEGY_CONFIG["symbols"] = ["SPY"]

def load_5m_data(symbol: str, days: int = 60) -> pd.DataFrame:
    """
    Load 5m candles for a symbol using yfinance and normalize columns to:
    open, high, low, close, volume (lowercase).
    """
    max_days_5m = 60
    use_days = min(days, max_days_5m)

    period = f"{use_days}d"
    print(f"Loading {symbol} 5m candles for about the last {use_days} days…")

    df = yf.download(
        symbol,
        period=period,
        interval="5m",
        auto_adjust=False,
        prepost=False,
        progress=False,
    )

    if df.empty:
        raise RuntimeError(f"Got empty DataFrame for {symbol} 5m data")

    # Handle MultiIndex columns like ('Open', 'SPY') or simple columns like 'Open'
    if isinstance(df.columns, pd.MultiIndex):
        # Take the first level (Open/High/Low/Close/Volume) and lowercase it
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    # Now we may have e.g. 'open', 'high', 'low', 'close', 'adj close', 'volume'
    rename_map = {}
    for col in df.columns:
        if col.startswith("open"):
            rename_map[col] = "open"
        elif col.startswith("high"):
            rename_map[col] = "high"
        elif col.startswith("low"):
            rename_map[col] = "low"
        elif col.startswith("close") and "adj" not in col:
            rename_map[col] = "close"
        elif "volume" in col:
            rename_map[col] = "volume"

    df = df.rename(columns=rename_map)

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        # Helpful debug to see what we actually have
        print("DEBUG: df.columns from yfinance:", list(df.columns))
        raise RuntimeError(f"Missing columns {missing} after normalization")

    df = df[required]

    # Normalize timezone
    if df.index.tz is None:
        df = df.tz_localize("UTC").tz_convert("America/New_York")
    else:
        df = df.tz_convert("America/New_York")

    return df


def resample_to_15m(df_5m: pd.DataFrame) -> pd.DataFrame:
    # Just in case, only aggregate columns that exist
    agg_map = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    agg_map = {k: v for k, v in agg_map.items() if k in df_5m.columns}

    df_15m = df_5m.resample("15min").agg(agg_map).dropna()
    return df_15m

def main():
    print("=== SPY/QQQ Trend Pullback Signals LAB ===")
    print(f"Config: {STRATEGY_CONFIG}")

    for symbol in STRATEGY_CONFIG["symbols"]:
        print(f"\n--- {symbol} ---")

        df_5m = load_5m_data(symbol, days=60)
        df_15m = resample_to_15m(df_5m)
        
        # Find raw signals (your existing logic)
        signals = find_trend_pullback_signals(symbol, df_5m, df_15m, STRATEGY_CONFIG)
        print(f"Found {len(signals)} signals for {symbol}")

        # Collect signals into a list of dicts for the backtester
        signals_for_backtest = []

        for sig in signals:
            # Keep your current printout
            print(
                f"{sig.timestamp} | {sig.symbol} | {sig.direction}"
                f" | price={sig.entry_price_underlying:.2f}"
                f" | note={sig.notes}"
            )

            # Add a dict version for the backtester
            signals_for_backtest.append(
                {
                    "timestamp": sig.timestamp,
                    "symbol": sig.symbol,
                    "direction": sig.direction,
                    "note": sig.notes,
                }
            )

        # Overall stats for this symbol (with your tuned params)
        trades, stats = backtest_trend_pullback_signals(
            df_5m,
            signals_for_backtest,
            stop_pct=0.4,
            target_R=1.5,
        )


        print()
        print(f"=== {symbol} Trend Pullback Backtest (Underlying) ===")
        print(f"n_trades: {stats.n_trades}")
        print(f"n_wins:   {stats.n_wins}")
        print(f"n_losses: {stats.n_losses}")
        print(f"win_rate: {stats.win_rate:.1f}%")
        print(f"avg_R:    {stats.avg_R:.2f}")
        print(f"total_R:  {stats.total_R:.1f}")
        
        # 🔹 NEW: long-only vs short-only diagnostics
        long_signals = [s for s in signals_for_backtest if s["direction"] == "long"]
        short_signals = [s for s in signals_for_backtest if s["direction"] == "short"]

        if long_signals:
            long_trades, long_stats = backtest_trend_pullback_signals(
                df_5m,
                long_signals,
                stop_pct=0.4,
                target_R=1.5,
            )
            print()
            print(f"=== {symbol} Trend Pullback LONG-ONLY ===")
            print(f"n_trades: {long_stats.n_trades}")
            print(f"n_wins:   {long_stats.n_wins}")
            print(f"n_losses: {long_stats.n_losses}")
            print(f"win_rate: {long_stats.win_rate:.1f}%")
            print(f"avg_R:    {long_stats.avg_R:.2f}")
            print(f"total_R:  {long_stats.total_R:.1f}")

        if short_signals:
            short_trades, short_stats = backtest_trend_pullback_signals(
                df_5m,
                short_signals,
                stop_pct=0.4,
                target_R=1.5,
            )
            print()
            print(f"=== {symbol} Trend Pullback SHORT-ONLY ===")
            print(f"n_trades: {short_stats.n_trades}")
            print(f"n_wins:   {short_stats.n_wins}")
            print(f"n_losses: {short_stats.n_losses}")
            print(f"win_rate: {short_stats.win_rate:.1f}%")
            print(f"avg_R:    {short_stats.avg_R:.2f}")
            print(f"total_R:  {short_stats.total_R:.1f}")


if __name__ == "__main__":
    main()
