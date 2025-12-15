import argparse
from src.backtest.runner import run_orb_backtest
from src.backtest.metrics import summarize

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--start", default="2024-09-01")
    ap.add_argument("--end", default="2025-11-01")
    args = ap.parse_args()

    trades = run_orb_backtest(symbol=args.symbol, start=args.start, end=args.end)
    stats = summarize(trades)

    print("\n=== ORB v1 Backtest Summary ===")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k:16s}: {v:.4f}")
        else:
            print(f"{k:16s}: {v}")

    print("\nSample trades (first 5):")
    for t in trades[:5]:
        print(f"{t.entry_time} {t.side.upper()} entry={t.entry_price:.2f} exit={t.exit_price:.2f} "
              f"R={t.pnl_R:.2f} reason={t.reason}")

if __name__ == "__main__":
    main()