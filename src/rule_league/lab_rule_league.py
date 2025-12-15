# src/rule_league/lab_rule_league.py

import pandas as pd
import statistics
from pathlib import Path


def load_plans(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found. Run the LAB runners first.")
    df = pd.read_csv(p, parse_dates=["signal_time", "expiry"])
    return df


def summarize_group(df: pd.DataFrame, name: str) -> None:
    print(f"\n=== {name} ===")
    if df.empty:
        print("No trades.")
        return

    risks = df["risk_pct"].dropna().tolist()
    rewards = df["reward_pct"].dropna().tolist()
    Rs = df["R"].dropna().tolist()
    dtes = df["dte"].dropna().tolist()

    print(f"Trades: {len(df)}")

    if dtes:
        print(
            f"DTE: median={statistics.median(dtes):.1f}d, "
            f"min={min(dtes)}d, max={max(dtes)}d"
        )

    if risks:
        print(
            f"Risk %: median={statistics.median(risks):.2f}%, "
            f"min={min(risks):.2f}%, max={max(risks):.2f}%"
        )

    if rewards:
        print(
            f"Reward %: median={statistics.median(rewards):.2f}%, "
            f"min={min(rewards):.2f}%, max={max(rewards):.2f}%"
        )

    if Rs:
        print(
            f"R multiple: median={statistics.median(Rs):.2f}R, "
            f"min={min(Rs):.2f}R, max={max(Rs):.2f}R"
        )


def apply_variant_filter(
    df: pd.DataFrame,
    risk_min: float | None = None,
    risk_max: float | None = None,
    dte_min: int | None = None,
    dte_max: int | None = None,
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)

    if risk_min is not None:
        mask &= df["risk_pct"] >= risk_min
    if risk_max is not None:
        mask &= df["risk_pct"] <= risk_max
    if dte_min is not None:
        mask &= df["dte"] >= dte_min
    if dte_max is not None:
        mask &= df["dte"] <= dte_max

    return df[mask].copy()


def main() -> None:
    spy_df = load_plans("data/lab_spy_double_bottom_options.csv")
    qqq_df = load_plans("data/lab_qqq_double_bottom_options.csv")

    # Just in case, enforce ticker filters
    spy_df = spy_df[spy_df["ticker"] == "SPY"].copy()
    qqq_df = qqq_df[qqq_df["ticker"] == "QQQ"].copy()

    print("=== Double-Bottom Options LAB: Rule League Summary ===")

    # SPY – all
    summarize_group(spy_df, "SPY (all plans)")

    # SPY – core variant
    spy_core = apply_variant_filter(spy_df, risk_min=0.8, risk_max=1.3, dte_min=5, dte_max=10)
    summarize_group(spy_core, "SPY (core: risk 0.8–1.3%, DTE 5–10d)")

    # QQQ – all
    summarize_group(qqq_df, "QQQ (all plans)")

    # QQQ – core variant
    qqq_core = apply_variant_filter(qqq_df, risk_min=0.8, risk_max=1.3, dte_min=5, dte_max=10)
    summarize_group(qqq_core, "QQQ (core: risk 0.8–1.3%, DTE 5–10d)")


if __name__ == "__main__":
    main()
