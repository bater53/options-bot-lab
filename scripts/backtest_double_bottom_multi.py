# scripts/backtest_double_bottom_multi.py

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

# --- Make project root importable as a package root ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.double_bottom_backtest import (  # noqa: E402
    backtest_spy_double_bottom,
)


TICKERS = ["SPY", "QQQ", "IWM"]


def fetch_5m(ticker: str) -> pd.DataFrame:
    """
    Fetch 5m OHLCV data for a given ticker, up to the last 60 days
    (falling back to 30d if needed). Returns a DataFrame with:
      index: datetime
      columns: adj_close, close, high, low, open, volume
    """
    for period in ["60d", "30d"]:
        print(f"\n[{ticker}] Trying period={period} ...")
        df = yf.download(
            ticker,
            period=period,
            interval="5m",
            auto_adjust=False,
            progress=False,
        )
        if df.empty:
            continue

        # Flatten MultiIndex columns if present (Price/Ticker style)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower().replace(" ", "_") for c in df.columns]
        else:
            df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

        df.index.name = "datetime"
        print(f"[{ticker}] Got data: shape={df.shape}")
        return df

    print(f"[{ticker}] ERROR: No data returned for 5m candles.")
    return pd.DataFrame()


def main() -> None:
    results = {}
    all_r_values = []
    total_trades = 0
    total_wins = 0
    total_losses = 0

    for ticker in TICKERS:
        df = fetch_5m(ticker)
        if df.empty:
            print(f"[{ticker}] Skipping backtest (no data).")
            continue

        # We re-use backtest_spy_double_bottom; it only cares about price path
        result = backtest_spy_double_bottom(
            df,
            max_risk_per_trade=200.0,
            max_bars_ahead=24,
        )

        results[ticker] = result

        print(f"\n=== {ticker} Double-Bottom Quality Backtest (5m, last 60d) ===")
        print("n_trades:", result.n_trades)
        print("n_wins:", result.n_wins)
        print("n_losses:", result.n_losses)
        print("win_rate:", result.win_rate)
        print("avg_r:", result.avg_r)

        # Aggregate stats
        total_trades += result.n_trades
        total_wins += result.n_wins
        total_losses += result.n_losses

        # Collect R multiples for overall avg
        for t in result.trades:
            if t.r_multiple is not None:
                all_r_values.append(t.r_multiple)

        # Optional: show first couple of trades per symbol
        for t in result.trades[:3]:
            print("-----")
            print("entry:", t.trade.underlying_entry,
                  "stop:", t.trade.underlying_stop,
                  "target:", t.trade.underlying_target)
            print("hit_target:", t.hit_target,
                  "hit_stop:", t.hit_stop,
                  "R:", t.r_multiple)

    # Overall summary across all tickers
    print("\n================= OVERALL SUMMARY (all tickers) =================")
    print("Tickers:", TICKERS)
    print("total_trades:", total_trades)
    print("total_wins:", total_wins)
    print("total_losses:", total_losses)
    if total_trades > 0:
        overall_win_rate = total_wins / total_trades
        print("overall_win_rate:", overall_win_rate)
    else:
        print("overall_win_rate: N/A (no trades)")

    if all_r_values:
        overall_avg_r = sum(all_r_values) / len(all_r_values)
        print("overall_avg_r:", overall_avg_r)
    else:
        print("overall_avg_r: N/A (no trades)")


if __name__ == "__main__":
    main()
