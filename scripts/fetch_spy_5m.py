# scripts/fetch_spy_5m.py

import yfinance as yf
import pandas as pd


def fetch_spy_5m() -> pd.DataFrame:
    """Fetch SPY 5m data for up to the last 60 days (fallback 30d)."""
    for period in ["60d", "30d"]:
        print(f"Trying period={period} ...")
        df = yf.download(
            "SPY",
            period=period,
            interval="5m",
            auto_adjust=False,
            progress=False,
        )
        if not df.empty:
            print(f"Got data for period={period}: shape={df.shape}")
            return df

    print("ERROR: No data returned for SPY 5m.")
    return pd.DataFrame()


def main() -> None:
    df = fetch_spy_5m()
    if df.empty:
        print("No data to save. Exiting.")
        return

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower().replace(" ", "_") for c in df.columns]
    else:
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]

    expected = {"open", "high", "low", "close", "adj_close", "volume"}
    missing = expected - set(df.columns)
    if missing:
        print("Warning: missing columns:", missing)

    print("Final columns:", df.columns.tolist())

    df.index.name = "datetime"

    print("Downloaded shape:", df.shape)
    print(df.head())

    df.to_csv("data/spy_5m.csv")
    print("Saved data/spy_5m.csv")


if __name__ == "__main__":
    main()
