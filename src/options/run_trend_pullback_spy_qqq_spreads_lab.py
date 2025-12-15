# src/options/run_trend_pullback_spy_qqq_spreads_lab.py

import pathlib
import pandas as pd

from src.strategies.trend_pullback_spreads import (
    STRATEGY_CONFIG,
    find_trend_pullback_signals,
    plan_spread_from_signal,
)

# If you already have a data loader utility in options-bot-lab, import and use that.
# Here is a stub that you will adapt.
def load_5m_data(symbol: str, days: int = 60) -> pd.DataFrame:
    """
    TODO: Replace this with your real data loader (yfinance or your existing utils).
    Must return a 5m DataFrame with columns: open, high, low, close, volume
    and a DateTimeIndex in NY time.
    """
    raise NotImplementedError("Wire this to your real data source.")


def resample_to_15m(df_5m: pd.DataFrame) -> pd.DataFrame:
    df_15m = df_5m.resample("15min").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna()
    return df_15m


def main():
    print("=== SPY/QQQ Trend Pullback Spreads LAB ===")
    print(f"Config: {STRATEGY_CONFIG}")

    for symbol in STRATEGY_CONFIG["symbols"]:
        print(f"\n--- Symbol: {symbol} ---")
        # 1) Load data
        df_5m = load_5m_data(symbol)
        df_15m = resample_to_15m(df_5m)

        # 2) Find signals
        signals = find_trend_pullback_signals(symbol, df_5m, df_15m, STRATEGY_CONFIG)
        print(f"Found {len(signals)} signals for {symbol}")

        for sig in signals:
            print(
                f"{sig.timestamp} | {sig.symbol} | {sig.direction} | "
                f"price={sig.entry_price_underlying:.2f} | note={sig.notes}"
            )

            # 3) Plan a spread (stub for now)
            trade = plan_spread_from_signal(sig, option_chain_env=None)
            # For now, trade will be None until you wire up option chain logic.

    print("\nDone.")


if __name__ == "__main__":
    main()
