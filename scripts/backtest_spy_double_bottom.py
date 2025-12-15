# scripts/backtest_spy_double_bottom.py

import sys
from pathlib import Path

# --- Make project root importable as a package root ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.backtest.double_bottom_backtest import backtest_spy_double_bottom  # noqa: E402


def main():
    df = pd.read_csv(
        "data/spy_5m.csv",
        parse_dates=["datetime"],
        index_col="datetime",
    )

    print("Data shape:", df.shape)
    print(df.head())

    result = backtest_spy_double_bottom(
        df,
        max_risk_per_trade=200.0,
        max_bars_ahead=24,  # ~2 hours of 5m bars
    )

    print("\n=== Backtest Summary (SPY 5m, last ~30 days) ===")
    print("n_trades:", result.n_trades)
    print("n_wins:", result.n_wins)
    print("n_losses:", result.n_losses)
    print("win_rate:", result.win_rate)
    print("avg_r:", result.avg_r)

    for t in result.trades[:5]:
        print("-----")
        print("entry:", t.trade.underlying_entry,
              "stop:", t.trade.underlying_stop,
              "target:", t.trade.underlying_target)
        print("hit_target:", t.hit_target,
              "hit_stop:", t.hit_stop,
              "R:", t.r_multiple)


if __name__ == "__main__":
    main()
