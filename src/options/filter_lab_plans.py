from __future__ import annotations

import argparse
import pandas as pd


REQUIRED = [
    "rule_id","ticker","signal_time","entry","stop","target",
    "expiry","strike","option_type","risk_pct","reward_pct","R","dte"
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/lab_spy_double_bottom_options.csv")
    ap.add_argument("--out", dest="out_path", default="data/lab_spy_double_bottom_options_filtered.csv")

    # Friction model (percent of underlying move, round-trip)
    ap.add_argument("--cost_pct", type=float, default=0.25, help="round-trip friction as % of underlying (0.25 = 0.25%)")

    # Quality gates
    ap.add_argument("--min_net_R", type=float, default=1.20)
    ap.add_argument("--min_dte", type=int, default=5)
    ap.add_argument("--max_dte", type=int, default=10)
    ap.add_argument("--min_risk_pct", type=float, default=0.80)
    ap.add_argument("--max_risk_pct", type=float, default=1.30)
    ap.add_argument("--max_per_day", type=int, default=2)

    # Optional: restrict to CALL/PUT etc.
    ap.add_argument("--option_type", type=str, default="", help="e.g. CALL (leave blank for all)")
    args = ap.parse_args()

    df = pd.read_csv(args.in_path)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing columns: {missing}\nFound: {list(df.columns)}")

    # Parse time + NY trading day
    df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True, errors="coerce")
    df["_ts_ny"] = df["signal_time"].dt.tz_convert("America/New_York")
    df["_day_ny"] = df["_ts_ny"].dt.date

    # Optional type filter
    if args.option_type.strip():
        df = df[df["option_type"].str.upper() == args.option_type.strip().upper()]

    # De-dupe exact repeats (this kills your #4/#5 and #16/#17 type duplicates)
    dedupe_keys = ["ticker","signal_time","entry","stop","target","expiry","strike","option_type"]
    df = df.drop_duplicates(subset=dedupe_keys, keep="first")

    # Filters
    df = df[(df["dte"] >= args.min_dte) & (df["dte"] <= args.max_dte)]
    df = df[(df["risk_pct"] >= args.min_risk_pct) & (df["risk_pct"] <= args.max_risk_pct)]

    # Net edge after costs
    df["cost_pct"] = float(args.cost_pct)
    df["net_reward_pct"] = (df["reward_pct"] - df["cost_pct"]).clip(lower=0.0)
    df["net_R"] = df["net_reward_pct"] / df["risk_pct"]
    df = df[df["net_R"] >= args.min_net_R]

    # Keep best N per day (highest net_R)
    df = df.sort_values(["_day_ny", "net_R"], ascending=[True, False])
    df = df.groupby("_day_ny", group_keys=False).head(int(args.max_per_day))

    # Save + print shortlist
    df.to_csv(args.out_path, index=False)

    cols = ["rule_id","ticker","signal_time","entry","stop","target","dte","expiry","strike","option_type",
            "risk_pct","reward_pct","cost_pct","net_reward_pct","net_R"]
    print(f"[ok] input:  {args.in_path}")
    print(f"[ok] output: {args.out_path}")
    print(f"[ok] kept {len(df)} plan(s) after filters.")
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
